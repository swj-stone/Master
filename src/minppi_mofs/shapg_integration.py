"""
ShapG integration for MinPPI guidance (§4.3.2).

Bridges the ShapG baseline (Zhao, Liu, Parilina 2025, *EAAI* 148) with
the MinPPI framework.  Provides two key services:

1. **ShapG guidance signal**: Shapley values over the feature correlation
   graph, used as both a baseline and a soft-prior for Algorithm B's
   seed-coalition construction.

2. **Correlated-neighbor lookup**: For each feature *i*, returns the
   features most correlated with *i* according to the ShapG graph.
   This is the ``shapg_neighbors`` input to
   :func:`~minppi_mofs.min_ppi.estimate_minppi_shapg_guided`.

The separation of ShapG as a guidance source (rather than the primary
importance signal) is a key architectural insight of MOFS-MinPPI (§4.5):
ShapG's correlation information is used to **guide search**, but the
**final decision** is made by actual model training in NSGA-II's fitness
evaluation.

Usage::

    from minppi_mofs.shapg_integration import (
        compute_shapg_guidance_signal,
        get_correlated_neighbors,
    )
    signal = compute_shapg_guidance_signal(X, y, depth=1, m=5, seed=42)
    neighbors = get_correlated_neighbors(X, top_k=10, seed=42)
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from minppi_mofs.baselines.shapg import (
    build_feature_graph,
    compute_shapg_importance,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_shapg_guidance_signal(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    depth: int = 1,
    m: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute ShapG feature importance as a guidance signal.

    Used for:
    - Priority ranking of features during seed-coalition construction
    - Correlation-aware feature ordering for Algorithm B

    Args:
        X: Feature DataFrame.
        y: Target array.
        depth: Neighbourhood depth for ShapG local subgraph.
        m: Sampling budget per node.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: shapg_score}`` sorted descending.
    """
    return compute_shapg_importance(X, y, depth=depth, m=m, seed=seed, verbose=verbose)


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
    :func:`~minppi_mofs.min_ppi.estimate_minppi_shapg_guided` (Algorithm B).

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
        # Sort neighbors by edge weight (if available) then degree
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
    """Compute both ShapG signal and neighbours in one call.

    Convenience wrapper — avoids building the feature graph twice.

    Args:
        X: Feature DataFrame.
        y: Target array.
        depth: ShapG neighbourhood depth.
        m: Sampling budget per node.
        top_k: Max neighbours to return per feature.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Tuple ``(shapg_scores, shapg_neighbors)``.
    """
    scores = compute_shapg_guidance_signal(
        X, y, depth=depth, m=m, seed=seed, verbose=verbose
    )
    neighbors = get_correlated_neighbors(X, top_k=top_k, seed=seed)
    return scores, neighbors


# ---------------------------------------------------------------------------
# Utility: feature ordering by importance
# ---------------------------------------------------------------------------


def rank_by_shapg(
    scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Rank features by ShapG score (descending).

    Args:
        scores: ``{feature: shapg_value}``.

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
    """Return the top-k ShapG-ranked features.

    Useful as a lightweight Phase-1 filter before running more expensive
    MinPPI estimation: restrict MinPPI computation to these top-k features.

    Args:
        X: Feature DataFrame.
        y: Target array.
        top_k: Number of features to retain.
        depth: ShapG depth.
        m: ShapG budget.
        seed: Random seed.

    Returns:
        List of top-k feature names in descending ShapG score order.
    """
    scores = compute_shapg_guidance_signal(X, y, depth=depth, m=m, seed=seed)
    ranked = rank_by_shapg(scores)
    return [f for f, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("ShapG Integration — synthetic demo")

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
    # Make f2 correlated with f0
    X_demo["f2"] = X_demo["f0"] * 0.8 + rng.normal(0, 0.3, n_samp)
    y_demo = np.where(X_demo["f0"] + 0.1 * X_demo["f1"] > 0, 1, 0)

    # ShapG guidance signal
    scores = compute_shapg_guidance_signal(X_demo, y_demo, depth=1, m=2, verbose=True)
    print("\nShapG scores:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.6f}")

    # Correlated neighbors
    neighbors = get_correlated_neighbors(X_demo, top_k=5)
    print("\nCorrelated neighbors (ShapG graph):")
    for feat, nbrs in neighbors.items():
        print(f"  {feat}: {nbrs}")

    print(
        "\nObservation: f0 should have the highest ShapG score. "
        "f0 and f2 should be neighbours (correlated). "
        "In Algorithm B, when estimating MinPPI for f2, "
        "the search will focus on coalitions containing f0 — "
        "the context where f2 is most likely to be redundant."
    )
