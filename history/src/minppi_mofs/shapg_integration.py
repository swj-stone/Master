"""MI-correlation guidance for MinPPI Algorithm B (§4.3.2).

The guidance signal has two explicitly separate components:

1. **Mutual-information scores** for seed-coalition ranking.
2. **Weighted correlation-tree neighbours** for local coalition contexts.

This lightweight heuristic is distinct from the full ShapG baseline.  Legacy
ShapG-labelled helper names remain only as deprecated compatibility wrappers.

Usage::

    from minppi_mofs.shapg_integration import (
        build_mi_correlation_guidance,
        compute_mi_guidance_signal,
        get_correlation_neighbors,
    )
    signal = compute_mi_guidance_signal(X, y, task="binary", seed=42)
    neighbors = get_correlation_neighbors(X, top_k=10, seed=42)
    signal, neighbors = build_mi_correlation_guidance(
        X, y, task="binary", top_k=10, seed=42
    )
"""

from __future__ import annotations

import warnings

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
    task: str,
    depth: int | None = None,
    m: int | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute feature-importance scores for seed-coalition construction.

    Uses mutual information for classification or regression as the ranking
    signal in Algorithm B's seed-coalition heuristic.

    Args:
        X: Feature DataFrame.
        y: Target array.
        task: One of ``"binary"``, ``"multiclass"``, or ``"regression"``.
        depth: Deprecated compatibility keyword; ignored.
        m: Deprecated compatibility keyword; ignored.
        seed: Random seed for MI computation.
        verbose: Ignored.

    Returns:
        Dict ``{feature_name: score}`` sorted descending.
    """
    valid_tasks = {"binary", "multiclass", "regression"}
    if task not in valid_tasks:
        raise ValueError(f"task must be one of {sorted(valid_tasks)}, got {task!r}")
    if depth is not None or m is not None:
        warnings.warn(
            "depth and m are deprecated for MI guidance and are ignored",
            DeprecationWarning,
            stacklevel=2,
        )

    if len(np.unique(y)) < 2:
        return {c: 0.0 for c in X.columns}
    if task == "regression":
        mi = mutual_info_regression(X, y, random_state=seed)
    else:
        mi = mutual_info_classif(X, y, random_state=seed)
    scores = dict(zip(X.columns, mi.astype(float)))
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def get_correlation_neighbors(
    X: pd.DataFrame,
    *,
    top_k: int = 10,
    corr_method: str = "pearson",
    seed: int = 42,
) -> dict[str, list[str]]:
    """Get correlation-graph neighbours for each feature.

    Builds a maximum-weight correlation tree and returns, per feature, a list
    of its neighbours sorted by decreasing edge weight.

    This is the ``correlation_neighbors`` argument consumed by
    :func:`~minppi_mofs.min_ppi.estimate_minppi_mi_correlation_guided`.

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


def get_correlated_neighbors(
    X: pd.DataFrame,
    *,
    top_k: int = 10,
    corr_method: str = "pearson",
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deprecated alias for :func:`get_correlation_neighbors`."""
    warnings.warn(
        "get_correlated_neighbors is deprecated; use get_correlation_neighbors",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_correlation_neighbors(
        X,
        top_k=top_k,
        corr_method=corr_method,
        seed=seed,
    )


def build_mi_correlation_guidance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    top_k: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Compute MI scores and weighted correlation-tree neighbours.

    Args:
        X: Feature DataFrame.
        y: Target array.
        task: One of ``"binary"``, ``"multiclass"``, or ``"regression"``.
        top_k: Max neighbours to return per feature.
        seed: Random seed.
        verbose: Ignored.

    Returns:
        Tuple ``(scores, neighbors)``.
    """
    scores = compute_mi_guidance_signal(X, y, task=task, seed=seed)
    neighbors = get_correlation_neighbors(X, top_k=top_k, seed=seed)
    return scores, neighbors


def _infer_legacy_task(y: np.ndarray) -> str:
    """Best-effort task inference used only by deprecated wrappers."""
    if y.dtype.kind == "f":
        return "regression"
    return "binary" if len(np.unique(y)) <= 2 else "multiclass"


def build_guidance_from_data(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str | None = None,
    depth: int = 1,
    m: int = 5,
    top_k: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Deprecated alias for :func:`build_mi_correlation_guidance`."""
    del depth, m
    warnings.warn(
        "build_guidance_from_data is deprecated; "
        "use build_mi_correlation_guidance",
        DeprecationWarning,
        stacklevel=2,
    )
    resolved_task = task if task is not None else _infer_legacy_task(y)
    return build_mi_correlation_guidance(
        X,
        y,
        task=resolved_task,
        top_k=top_k,
        seed=seed,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Utility: feature ordering by importance
# ---------------------------------------------------------------------------


def rank_by_mi(
    scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Rank features by score (descending).

    Args:
        scores: ``{feature: score}``.

    Returns:
        Sorted list of ``(feature, score)``.
    """
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rank_by_shapg(
    scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Deprecated alias for :func:`rank_by_mi`."""
    warnings.warn(
        "rank_by_shapg is deprecated; use rank_by_mi",
        DeprecationWarning,
        stacklevel=2,
    )
    return rank_by_mi(scores)


def filter_top_by_mi(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    top_k: int = 50,
    seed: int = 42,
) -> list[str]:
    """Return the top-k MI-ranked features.

    Useful as a lightweight Phase-1 filter before running more expensive
    MinPPI estimation.

    Args:
        X: Feature DataFrame.
        y: Target array.
        top_k: Number of features to retain.
        task: One of ``"binary"``, ``"multiclass"``, or ``"regression"``.
        seed: Random seed.

    Returns:
        List of top-k feature names in descending MI-score order.
    """
    scores = compute_mi_guidance_signal(X, y, task=task, seed=seed)
    ranked = rank_by_mi(scores)
    return [f for f, _ in ranked[:top_k]]


def filter_top_by_shapg(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str | None = None,
    top_k: int = 50,
    depth: int = 1,
    m: int = 5,
    seed: int = 42,
) -> list[str]:
    """Deprecated alias for :func:`filter_top_by_mi`."""
    del depth, m
    warnings.warn(
        "filter_top_by_shapg is deprecated; use filter_top_by_mi",
        DeprecationWarning,
        stacklevel=2,
    )
    resolved_task = task if task is not None else _infer_legacy_task(y)
    return filter_top_by_mi(
        X,
        y,
        task=resolved_task,
        top_k=top_k,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("MI-correlation guidance — synthetic demo")

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

    scores = compute_mi_guidance_signal(
        X_demo,
        y_demo,
        task="binary",
        verbose=True,
    )
    print("\nMI-based feature scores:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.6f}")

    neighbors = get_correlation_neighbors(X_demo, top_k=5)
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
