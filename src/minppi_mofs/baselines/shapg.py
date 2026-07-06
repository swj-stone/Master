"""
ShapG baseline for tabular feature importance.

ShapG (Zhao, Liu, Parilina 2025) computes Shapley values on a feature
correlation graph via local-neighbourhood approximation.  Applied to tabular
data the pipeline is:

  1. Build a Kendall-tau correlation matrix over features (``matrix_generator``).
  2. Reduce to a minimal-edge graph that preserves connectivity
     (``create_minimal_edge_graph``, version v3).
  3. Define a characteristic function  v(S) = AUC of an XGBoost classifier
     trained on feature subset S.
  4. Approximate Shapley values with ``shapG`` (depth + sampling budget).
  5. Rank features by decreasing Shapley value.

Usage::

    from minppi_mofs.baselines.shapg import compute_shapg_importance
    scores = compute_shapg_importance(X_train, y_train, depth=1, m=5, seed=42)
    # scores: dict {feature_name: shapley_value}

Run (stand-alone)::

    python src/minppi_mofs/baselines/shapg.py
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Graph construction helpers (adapted from shapG.utils)
# ---------------------------------------------------------------------------


def _corr_matrix(X: pd.DataFrame, *, method: str = "pearson") -> pd.DataFrame:
    """Pairwise correlation matrix.

    Uses ``pd.DataFrame.corr`` (vectorised) for speed.  Pearson is O(p² n)
    via numpy; Kendall is O(p² n log n) and should only be used for small
    datasets.

    Args:
        X: Feature DataFrame (n_samples, n_features).
        method: ``"pearson"``, ``"spearman"``, or ``"kendall"``.

    Returns:
        DataFrame of shape (p, p) with symmetric correlation coefficients.
    """
    return X.corr(method=method).fillna(0.0)


def _create_minimal_edge_graph(
    W: pd.DataFrame, *, reverse: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce weight matrix to minimal connected adjacency (v3 / Kruskal-like).

    Edges are processed in descending |weight| order; edge insertion stops
    as soon as all nodes belong to a single connected component (Union-Find).

    Args:
        W: Square weight / correlation matrix.
        reverse: If True, sort by descending absolute weight.

    Returns:
        (adjacency_df, reduced_weight_df).
    """
    cols = W.columns.tolist()
    p = len(cols)

    # Build flat edge list (w, i, j) using integer indices for Union-Find
    edges: list[tuple[float, int, int]] = []
    for i in range(p):
        for j in range(i + 1, p):
            edges.append((abs(W.iloc[i, j]), i, j))
    edges.sort(key=lambda e: e[0], reverse=reverse)

    parent = list(range(p))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    adj = pd.DataFrame(0, index=cols, columns=cols, dtype=np.int8)
    red = pd.DataFrame(0.0, index=cols, columns=cols)
    components = p

    for w, i, j in edges:
        if components == 1:
            break
        n1, n2 = cols[i], cols[j]
        adj.loc[n1, n2] = adj.loc[n2, n1] = 1
        red.loc[n1, n2] = red.loc[n2, n1] = w
        if union(i, j):
            components -= 1

    return adj, red


def build_feature_graph(
    X: pd.DataFrame, *, corr_method: str = "pearson", seed: int = 42
) -> nx.Graph:
    """Build a correlation-based feature graph from a tabular dataset.

    Steps: correlation matrix → minimal-edge graph → nx.Graph.

    Args:
        X: Feature DataFrame.
        corr_method: ``"pearson"`` (fast, default), ``"spearman"``, or
            ``"kendall"``.
        seed: Forwarded for reproducibility (not used in deterministic v3).

    Returns:
        A ``networkx.Graph`` whose nodes are feature names.
    """
    W = _corr_matrix(X, method=corr_method)
    adj, _ = _create_minimal_edge_graph(W)
    G = nx.Graph(adj)
    return G


# ---------------------------------------------------------------------------
# Local-neighbourhood Shapley approximation (adapted from shapG.shapley)
# ---------------------------------------------------------------------------


def _get_reachable_nodes(G: nx.Graph, node: str, depth: int) -> set[str]:
    """Nodes at distance exactly *depth* from *node* in G."""
    lengths = nx.single_source_shortest_path_length(G, node, cutoff=depth)
    return {n for n, d in lengths.items() if d == depth}


def shapg(
    G: nx.Graph,
    f: Callable[[set[str]], float],
    *,
    depth: int = 1,
    m: int = 5,
    approximate_by_ratio: bool = True,
    verbose: bool = False,
) -> dict[str, float]:
    """Approximate Shapley values via local-neighbourhood sampling (ShapG).

    Args:
        G: Feature graph (nodes = features).
        f: Characteristic function  f(S) -> float.
        depth: Maximum neighbourhood depth for the local subgraph.
        m: Maximum number of reachable nodes to sample per node (budget).
        approximate_by_ratio: If True, rescale all values so their sum
            equals f(N) (grand-coalition value).
        verbose: Print per-node progress.

    Returns:
        Dict mapping node (feature name) → approximated Shapley value.
    """
    rng = np.random.default_rng(42)
    values: dict[str, float] = {n: 0.0 for n in G.nodes()}
    full_value: Optional[float] = f(set(G.nodes())) if approximate_by_ratio else None

    node_iter = G.nodes()
    if verbose:
        try:
            from tqdm import tqdm

            node_iter = tqdm(list(G.nodes()), desc="ShapG")
        except ImportError:
            # tqdm not available — simple diagnostic print per node
            node_iter = list(G.nodes())
            print(f"ShapG: {len(node_iter)} nodes to process...", flush=True)

    for node in node_iter:
        reachable: set[str] = set()
        for d in range(1, depth + 1):
            reachable |= _get_reachable_nodes(G, node, d)

        if len(reachable) < m:
            # Small neighbourhood — enumerate all subsets
            reachable.add(node)
            denom = 2 ** (len(reachable) - 1)
            for s_size in range(len(reachable)):
                for S in itertools.combinations(reachable - {node}, s_size):
                    S_set = set(S)
                    values[node] += f(S_set | {node}) - f(S_set)
            values[node] /= max(denom, 1)
        else:
            # Large neighbourhood — Monte-Carlo with sampling
            n_reach = len(reachable)
            n_samples = int(
                np.ceil(
                    n_reach / m * (np.log2(n_reach) + 0.5772156649)  # Euler's γ
                )
            )
            coeff = 1.0 / (2**m * n_samples) * (n_reach + 1) / (m + 1)
            reachable_list = list(reachable)

            for _ in range(n_samples):
                sampled = set(
                    rng.choice(
                        reachable_list,
                        size=min(m, len(reachable_list)),
                        replace=False,
                    )
                )
                sampled.add(node)
                for s_size in range(len(sampled)):
                    for S in itertools.combinations(sampled - {node}, s_size):
                        S_set = set(S)
                        values[node] += f(S_set | {node}) - f(S_set)
            values[node] *= coeff

    # Ratio scaling
    if approximate_by_ratio:
        fN = full_value if full_value is not None else f(set(G.nodes()))
        total = sum(values.values())
        if total != 0:
            factor = fN / total
            values = {k: v * factor for k, v in values.items()}

    return values


# ---------------------------------------------------------------------------
# Tabular characteristic function (XGBoost AUC)
# ---------------------------------------------------------------------------


class _CachedAUCCharacteristic:
    """Characteristic function v(S)=AUC with LRU caching.

    Since ``set`` is unhashable, coalition S is converted to a sorted tuple
    before lookup.  The cache avoids re-training the same XGBoost model
    across repeated evaluations in ``shapg``'s inner loops.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        seed: int = 42,
        test_size: float = 0.2,
        max_samples: int = 5000,
        task: str = "binary",
    ) -> None:
        self._X = X
        self._y = y
        self._seed = seed
        self._test_size = test_size
        self._max_samples = max_samples
        self._task = task
        self._cache: dict[tuple[str, ...], float] = {}
        # Reusable model
        if task == "regression":
            self._model = XGBRegressor(
                n_estimators=15,
                max_depth=1,
                learning_rate=0.3,
                verbosity=0,
                random_state=seed,
            )
        elif task == "multiclass":
            self._model = LogisticRegression(
                max_iter=200,
                solver="lbfgs",
                random_state=seed,
            )
        else:
            self._model = XGBClassifier(
                n_estimators=15,
                max_depth=1,
                learning_rate=0.3,
                verbosity=0,
                random_state=seed,
            )

    def __call__(self, S: set[str]) -> float:
        key = tuple(sorted(S))
        if key in self._cache:
            return self._cache[key]

        cols = list(S)
        if len(cols) == 0:
            val = 0.0 if self._task == "regression" else 0.5
        else:
            X_sub = self._X[cols].values
            # Sub-sample large datasets for performance
            n = X_sub.shape[0]
            if n > self._max_samples:
                rng = np.random.default_rng(self._seed + hash(key) % 2**31)
                idx = rng.choice(n, size=self._max_samples, replace=False)
                X_sub = X_sub[idx]
                y_sub = self._y[idx]
            else:
                y_sub = self._y
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub,
                    y_sub,
                    test_size=self._test_size,
                    random_state=self._seed,
                    stratify=y_sub if self._task != "regression" else None,
                )
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub,
                    y_sub,
                    test_size=self._test_size,
                    random_state=self._seed,
                )
            if self._task == "regression":
                self._model.fit(X_tr, y_tr)
                y_pred = self._model.predict(X_te)
                val = float(r2_score(y_te, y_pred))
                if val < 0.0:
                    val = 0.0
            elif self._task == "multiclass":
                self._model.fit(X_tr, y_tr)
                val = float(accuracy_score(y_te, self._model.predict(X_te)))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._model.fit(X_tr, y_tr)
                y_prob = self._model.predict_proba(X_te)
                y_test_classes = np.unique(y_te)
                if len(y_test_classes) < 2:
                    val = 0.5
                elif y_prob.shape[1] == 2:
                    val = float(roc_auc_score(y_te, y_prob[:, 1]))
                else:
                    val = float(roc_auc_score(y_te, y_prob, multi_class="ovr"))
        self._cache[key] = val
        return val


def make_xgb_auc_characteristic(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    task: str = "binary",
) -> _CachedAUCCharacteristic:
    """Return a cached characteristic function  v(S) = XGBoost AUC.

    Args:
        X: Feature DataFrame.
        y: Binary target array.
        seed: Random seed.
        test_size: Held-out fraction for AUC eval.
        task: ``"binary"`` (classification) or ``"regression"``.

    Returns:
        Callable ``set[str] -> float`` with internal LRU cache.
    """
    return _CachedAUCCharacteristic(X, y, seed=seed, test_size=test_size, task=task)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_shapg_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    depth: int = 1,
    m: int = 5,
    seed: int = 42,
    verbose: bool = False,
    task: str = "binary",
) -> dict[str, float]:
    """Compute ShapG feature importance for a tabular dataset.

    Pipeline: build correlation graph → define XGBoost-AUC characteristic
    function → approximate Shapley values → return sorted scores.

    Args:
        X: Feature DataFrame.
        y: Binary target array.
        depth: Neighbourhood depth for the ShapG local subgraph.
        m: Sampling budget per node (max reachable nodes to draw).
        seed: Random seed controlling XGBoost, split, and sampling.
        verbose: If True, show progress bar and diagnostic messages.
        task: ``"binary"`` for classification (AUC), ``"regression"`` for R^2.

    Returns:
        Dict ``{feature_name: shapley_value}`` sorted descending by value.

    Examples:
        >>> scores = compute_shapg_importance(X_train, y_train)
        >>> list(scores.items())[:5]  # top-5 features
    """
    G = build_feature_graph(X, seed=seed)
    v = make_xgb_auc_characteristic(X, y, seed=seed, task=task)
    shapley = shapg(
        G,
        v,
        depth=depth,
        m=m,
        approximate_by_ratio=True,
        verbose=verbose,
    )
    # Sort descending
    return dict(sorted(shapley.items(), key=lambda kv: kv[1], reverse=True))


def rank_features(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Convert a score dict to a ranked list (highest first).

    Args:
        scores: Dict {feature_name: score}.

    Returns:
        List of (feature_name, score) tuples in descending score order.
    """
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("ShapG baseline — synthetic demo")
    rng = np.random.default_rng(42)
    n = 200
    X_demo = pd.DataFrame(
        {
            "f0": rng.normal(0, 1, n),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
        }
    )
    # f0 is highly predictive, f1 weak, f2 irrelevant
    y_demo = np.where(X_demo["f0"] + 0.1 * X_demo["f1"] > 0, 1, 0)

    scores = compute_shapg_importance(X_demo, y_demo, depth=1, m=2, verbose=False)
    print("Feature importance ranking:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.6f}")
    print("\nObservation: f0 should score highest (it determines the label).")
