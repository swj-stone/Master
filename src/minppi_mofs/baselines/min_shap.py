"""
MinShap baseline for tabular feature importance.

MinShap (Zheng & Raskutti 2026, arXiv:2604.15107) modifies the standard
Shapley value by replacing the **mean** over permutations with the
**minimum** marginal contribution::

    MinShap_i = min_{S ⊆ N\\{i}} [v(S ∪ {i}) - v(S)]

where  v(S) = AUC(XGBoost(S))  (continuous characteristic function).

Intuition
---------
The Shapley mean can hide harmful features (those whose addition consistently
lowers accuracy) by averaging their negative contributions with positive ones.
The min operator makes those harmful features explicitly identifiable:  a
negative MinShap score signals that the feature hurts performance in *some*
context.

Performance
-----------
Instead of sampling K coalitions independently for each feature (O(p·K)
evaluations), this module uses **shared-coalition Monte Carlo** (same
structure as the Banzhaf estimator):

1. Sample *M* random coalitions S ⊆ N.
2. For each S, evaluate v(S) once.
3. For a random subset of features not in S (up to *per_coalition*), evaluate
   v(S ∪ {i}) and accumulate the marginal.
4. Each feature tracks the *minimum* of its observed marginal contributions.
5. Early exit when min = -1 (harmful in at least one context) or when
   a reasonable upper bound is established.

With ``n_coalitions=200, per_coalition=5`` this gives ~1000 evaluations
total instead of ``p × K`` (= 500 × 100 = 50 000 for Madelon).

Usage::

    from minppi_mofs.baselines.min_shap import compute_minshap_importance
    scores = compute_minshap_importance(X_train, y_train, n_coalitions=300)

Run (stand-alone)::

    python src/minppi_mofs/baselines/min_shap.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Cached AUC characteristic function
# ---------------------------------------------------------------------------


class _CachedAUC:
    """v(S) = AUC of a shallow XGBoost trained on feature subset S.

    Each evaluation trains a fresh model; results are cached by sorted-tuple
    key to avoid redundant evaluations.
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

    def __call__(self, S: set[str]) -> float:
        key = tuple(sorted(S))
        if key in self._cache:
            return self._cache[key]

        cols = list(S)
        if len(cols) == 0:
            val = 0.0 if self._task == "regression" else 0.5
        else:
            X_sub = self._X[cols].values
            n_rows = X_sub.shape[0]
            if n_rows > self._max_samples:
                rng = np.random.default_rng(self._seed + hash(key) % 2**31)
                idx = rng.choice(n_rows, size=self._max_samples, replace=False)
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
                model = XGBRegressor(
                    n_estimators=10,
                    max_depth=2,
                    learning_rate=0.3,
                    verbosity=0,
                    random_state=self._seed,
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                val = float(r2_score(y_te, y_pred))
                if val < 0.0:
                    val = 0.0
            elif self._task == "multiclass":
                model = LogisticRegression(
                    max_iter=200,
                    solver="lbfgs",
                    random_state=self._seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_tr)
                val = float(accuracy_score(y_te, model.predict(X_te)))
            else:
                model = XGBClassifier(
                    n_estimators=10,
                    max_depth=2,
                    learning_rate=0.3,
                    verbosity=0,
                    random_state=self._seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_tr)
                y_prob = model.predict_proba(X_te)
                y_test_classes = np.unique(y_te)
                if len(y_test_classes) < 2:
                    val = 0.5
                elif y_prob.shape[1] == 2:
                    val = float(roc_auc_score(y_te, y_prob[:, 1]))
                else:
                    val = float(roc_auc_score(y_te, y_prob, multi_class="ovr"))
        self._cache[key] = val
        return val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_minshap_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_coalitions: int = 300,
    per_coalition: int = 5,
    seed: int = 42,
    verbose: bool = False,
    task: str = "binary",
) -> dict[str, float]:
    """Compute MinShap via shared-coalition Monte Carlo.

    Args:
        X: Feature DataFrame.
        y: Target array.
        n_coalitions: Number of random coalitions to sample.
        per_coalition: Max features to test for marginal per coalition.
        seed: Random seed.
        verbose: Print progress every 50 coalitions.
        task: ``"binary"`` for classification (AUC), ``"regression"`` for R^2.

    Returns:
        Dict ``{feature_name: minshap_score}`` sorted descending (highest
        min-marginal first).  Negative values indicate harmful features.
    """
    rng = np.random.default_rng(seed)
    features = list(X.columns)
    n = len(features)
    all_set = set(features)

    v = _CachedAUC(X, y, seed=seed, task=task)
    full_auc = v(all_set)
    if verbose:
        print(f"  Full-feature AUC: {full_auc:.4f}", flush=True)

    minshap = {name: 1.0 for name in features}  # init to max possible
    counts = {name: 0 for name in features}

    for k in range(n_coalitions):
        if verbose and (k + 1) % 50 == 0:
            done = sum(1 for c in counts.values() if c > 0)
            print(
                f"  MinShap shared: {k + 1}/{n_coalitions}  ({done}/{n} features seen)",
                flush=True,
            )

        # Random coalition — bias toward smaller sizes to maximise the pool
        # of features NOT in S (i.e. those we can evaluate marginals for).
        # Without this bias the coupon-collector problem makes full coverage
        # extremely slow on high-dimensional datasets.
        s_size = rng.integers(0, max(1, n // 3) + 1)
        S_list = rng.choice(features, size=s_size, replace=False).tolist()
        S = set(S_list)
        v_S = v(S)

        # Evaluate marginals for features NOT in S (≈ 2/3 of all features)
        outside = list(all_set - S)
        rng.shuffle(outside)
        test_feats = outside[:per_coalition]

        for feat in test_feats:
            if minshap[feat] <= -0.99:
                continue
            v_Si = v(S | {feat})
            marg = v_Si - v_S
            if marg < minshap[feat]:
                minshap[feat] = marg
            counts[feat] += 1

        # Early stop: all features covered at least once.
        # For min operator even a single evaluation gives a usable upper bound;
        # the marginal benefit of additional samples is tiny.
        min_eval = 1
        all_covered = all(c >= min_eval for c in counts.values())
        if all_covered and k >= 50:
            if verbose:
                print(
                    f"  Early stop at coalition {k + 1}/{n_coalitions} "
                    f"(all features >= {min_eval} evals)",
                    flush=True,
                )
            break

    if verbose:
        covered = sum(1 for c in counts.values() if c > 0)
        min_c, max_c = min(counts.values()), max(counts.values())
        print(
            f"  Features with >=1 eval: {covered}/{n}  (evals per feat: {min_c}-{max_c})",
            flush=True,
        )

    result = {name: minshap[name] if counts[name] > 0 else 0.0 for name in features}
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


def rank_features(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Rank features by MinShap score (descending: highest first).

    Returns:
        List of (feature, score) in descending order.
    """
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("MinShap baseline — synthetic demo")
    rng = np.random.default_rng(42)
    n_samp = 200
    X_demo = pd.DataFrame(
        {
            "f0": rng.normal(0, 1, n_samp),
            "f1": rng.normal(0, 1, n_samp),
            "f2": rng.normal(0, 1, n_samp),
        }
    )
    y_demo = np.where(X_demo["f0"] + 0.1 * X_demo["f1"] > 0, 1, 0)

    scores = compute_minshap_importance(X_demo, y_demo, n_coalitions=100, verbose=True)
    ranking = rank_features(scores)
    print("Feature ranking (descending):")
    for feat, val in ranking:
        print(f"  {feat}: {val:.4f}")
    print("\nObservation: f0 should be highest (most positive min marginal).")
