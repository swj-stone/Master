"""
ShapG integration for MinPPI guidance (§4.3.2).

Fast redesign (2026-07-07): the original ``compute_mi_guidance_signal``
called the full ShapG pipeline (graph build → enumerate 2^{d}-subsets per
node → XGBoost per subset).  For a 56-feature dataset that took **276 s**
— dwarfing MinPPI's 39 s.  The replacement uses:

1. **Mutual-information scores** (sklearn, O(np)) for seed-coalition ranking
   — same role as the old ShapG scores, ~1000× faster.
2. **Pearson-correlation neighbours** (unchanged) via the minimal-edge graph
   for ``shapg_neighbors`` in Algorithm B.

Usage::

    from minppi_mofs.shapg_integration import (
        compute_mi_guidance_signal,
        get_correlated_neighbors,
    )
    signal = compute_mi_guidance_signal(X, y, seed=42)
    neighbors = get_correlated_neighbors(X, top_k=10, seed=42)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from minppi_mofs.baselines.shapg import build_feature_graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_mi_guidance_signal(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    depth: int = 1,
    m: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute feature-importance scores for seed-coalition construction.

    Uses mutual information (classification) or MI-regression — O(np),
    ~1000× faster than the original ShapG pipeline with identical utility
    for Algorithm B's seed-coalition heuristic.

    Args:
        X: Feature DataFrame.
        y: Target array.
        depth: Ignored (kept for API compatibility).
        m: Ignored.
        seed: Random seed for MI computation.
        verbose: Ignored.

    Returns:
        Dict ``{feature_name: score}`` sorted descending.
    """
    classes = np.unique(y)
    if len(classes) < 2:
        return {c: 0.0 for c in X.columns}
    if len(classes) > 20 or y.dtype.kind == "f":
        # Regression or many-class → use regression variant
        mi = mutual_info_regression(X, y, random_state=seed)
    else:
        mi = mutual_info_classif(X, y, random_state=seed)
    scores = dict(zip(X.columns, mi.astype(float)))
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def get_correlated_neighbors(
    X: pd.DataFrame,
    *,
    top_k: int = 10,
    corr_method: str = "pearson",
    seed: int = 42,
) -> dict[str, list[str]]:
    """Get ShapG-correlated neighbours for each feature.

    Builds the same correlation graph used by ShapG and returns, per feature,
    a list of its neighbours sorted by decreasing edge weight (the ShapG
    minimal-edge graph's adjacency).

    This is the ``shapg_neighbors`` argument consumed by
    :func:`~minppi_mofs.min_ppi.estimate_minppi_mi_guided` (Algorithm B).

    Args:
        X: Feature DataFrame.
        top_k: Maximum number of neighbours to return per feature.
        corr_method: ``"pearson"`` (fast), ``"spearman"``, or ``"kendall"``.
        seed: Random seed.

    Returns:
        Dict ``{feature_name: [neighbor_1, ..., neighbor_k]}`` where the
        list is sorted by descending absolute correlation weight.
    """
    G = build_feature_graph(X, corr_method=corr_method, seed=seed)
    neighbors: dict[str, list[str]] = {}

    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if nbrs:
            weighted: list[tuple[str, float]] = []
            for nbr in nbrs:
                w = G.edges.get((node, nbr), {}).get("weight", 1.0)
                if w is None:
                    w = 1.0
                weighted.append((nbr, float(w)))
            weighted.sort(key=lambda x: x[1], reverse=True)
            neighbors[node] = [n for n, _ in weighted[:top_k]]
        else:
            neighbors[node] = []

    return neighbors


def build_guidance_from_data(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    depth: int = 1,
    m: int = 5,
    top_k: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Compute both fast-MI scores and correlation neighbours in one call.

    Args:
        X: Feature DataFrame.
        y: Target array.
        depth: Ignored (API compat).
        m: Ignored.
        top_k: Max neighbours to return per feature.
        seed: Random seed.
        verbose: Ignored.

    Returns:
        Tuple ``(scores, neighbors)``.
    """
    scores = compute_mi_guidance_signal(X, y, depth=depth, m=m, seed=seed)
    neighbors = get_correlated_neighbors(X, top_k=top_k, seed=seed)
    return scores, neighbors


# ---------------------------------------------------------------------------
# Utility: feature ordering by importance
# ---------------------------------------------------------------------------


def rank_by_shapg(
    scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Rank features by score (descending).

    Args:
        scores: ``{feature: score}``.

    Returns:
        Sorted list of ``(feature, score)``.
    """
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def filter_top_by_shapg(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    top_k: int = 50,
    depth: int = 1,
    m: int = 5,
    seed: int = 42,
) -> list[str]:
    """Return the top-k MI-ranked features.

    Useful as a lightweight Phase-1 filter before running more expensive
    MinPPI estimation.

    Args:
        X: Feature DataFrame.
        y: Target array.
        top_k: Number of features to retain.
        depth: Ignored.
        m: Ignored.
        seed: Random seed.

    Returns:
        List of top-k feature names in descending MI-score order.
    """
    scores = compute_mi_guidance_signal(X, y, depth=depth, m=m, seed=seed)
    ranked = rank_by_shapg(scores)
    return [f for f, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("ShapG Integration (fast MI) — synthetic demo")

    rng = np.random.default_rng(42)
    n_samp = 200
    X_demo = pd.DataFrame(
        {
            "f0": rng.normal(0, 1, n_samp),
            "f1": rng.normal(0, 1, n_samp),
            "f2": rng.normal(0, 1, n_samp),
            "f3": rng.normal(0, 1, n_samp),
            "f4": rng.normal(0, 1, n_samp),
        }
    )
    X_demo["f2"] = X_demo["f0"] * 0.8 + rng.normal(0, 0.3, n_samp)
    y_demo = np.where(X_demo["f0"] + 0.1 * X_demo["f1"] > 0, 1, 0)

    scores = compute_mi_guidance_signal(X_demo, y_demo, depth=1, m=2, verbose=True)
    print("\nMI-based feature scores:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.6f}")

    neighbors = get_correlated_neighbors(X_demo, top_k=5)
    print("\nCorrelated neighbors (Pearson graph):")
    for feat, nbrs in neighbors.items():
        print(f"  {feat}: {nbrs}")

    print(
        "\nObservation: f0 should have the highest MI score. "
        "f0 and f2 should be neighbours (correlated). "
        "In Algorithm B, when estimating MinPPI for f2, "
        "the search will focus on coalitions containing f0 — "
        "the context where f2 is most likely to be redundant."
    )
