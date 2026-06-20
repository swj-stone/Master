"""
MOFS-MinPPI baseline wrapper — exposes the proposed method as a standard-scored
ranker compatible with the main-table registry.

Internal pipeline
-----------------
1. Split *X*, *y* into train / validation (for MinPPI estimation).
2. Pre-filter high-dimensional datasets via ShapG.
3. Build ShapG guidance signal + neighbour graph.
4. Estimate MinPPI via hybrid dispatcher.
5. Run MOFS-MinPPI NSGA-II search.
6. Extract feature importance from the Pareto front.

The returned ``dict[str, float]`` is compatible with all other baselines.
Additionally, the Pareto-fitness result is stored via a side-channel mutable
container so callers can compute hypervolume and compression rate without a
second search.

Usage:

    from minppi_mofs.baselines.mofs_minppi_baseline import (
        compute_mofs_minppi_importance,
        rank_features,
    )
    result_box = []
    scores = compute_mofs_minppi_importance(X, y, alpha=0.95, seed=42,
                                             _result_box=result_box)
    ranking = rank_features(scores)
    # result_box[0] is an MOFSMinPPIResult with pareto_masks / pareto_fitness

Run (stand-alone):

    python src/minppi_mofs/baselines/mofs_minppi_baseline.py
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from minppi_mofs.min_ppi import estimate_minppi
from minppi_mofs.mofs_minppi import (
    mofs_minppi_search,
    pareto_feature_importance,
)
from minppi_mofs.shapg_integration import build_guidance_from_data


# ---------------------------------------------------------------------------
# Per-dataset budget  (mirrors experiments/run_ablation.py)
# ---------------------------------------------------------------------------

DATASET_BUDGET: dict[str, dict] = {
    "spambase": {"K_adv": 30, "pop": 30, "gen": 20},
    "madelon": {"K_adv": 20, "pop": 20, "gen": 15},
    "isolet": {"K_adv": 15, "pop": 20, "gen": 10},
    "gisette": {"K_adv": 10, "pop": 15, "gen": 8},
    "credit_card_fraud": {"K_adv": 30, "pop": 30, "gen": 20},
}

MAX_FEATURES_FULL_MINPPI: int = 200


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prefilter_features(
    X: pd.DataFrame,
    y: np.ndarray,
    max_features: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Phase-1 ShapG filter: keep top *max_features*."""
    n = X.shape[1]
    if n <= max_features:
        return X
    sg_scores, _ = build_guidance_from_data(X, y, depth=1, m=3, seed=seed)
    ranked = sorted(sg_scores.items(), key=lambda kv: kv[1], reverse=True)
    keep = [f for f, _ in ranked[:max_features]]
    return X[keep]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_mofs_minppi_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    alpha: float = 0.95,
    beta: float = 1.0,
    K_adv: int = 30,
    pop_size: int = 30,
    generations: int = 20,
    val_size: float = 0.25,
    seed: int = 42,
    verbose: bool = False,
    _result_box: Optional[list] = None,
) -> dict[str, float]:
    """Compute MOFS-MinPPI feature importance.

    Args:
        X: Feature DataFrame (training set; internally split for val).
        y: Target array.
        alpha: Voting-game quota α ∈ (0, 1].
        beta: MinPPI guidance strength (β ≥ 0).
        K_adv: Adversarial search budget per feature.
        pop_size: NSGA-II population size (even).
        generations: NSGA-II generation count.
        val_size: Fraction of *X* held out for MinPPI estimation.
        seed: Random seed.
        verbose: Print progress.
        _result_box: If a ``list``, the :class:`MOFSMinPPIResult` is
            appended so the caller can extract HV / compression without
            re-running the search.

    Returns:
        Dict ``{feature_name: frequency_in_pareto_front}`` sorted descending.
    """
    # ---- Pre-filter high-dimensional datasets ----
    X = _prefilter_features(X, y, max_features=MAX_FEATURES_FULL_MINPPI, seed=seed)

    # ---- Split for MinPPI estimation ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    if verbose:
        print(f"  MOFS-MinPPI: {X.shape[1]} features (after pre-filter)", flush=True)

    # ---- ShapG guidance ----
    sg_scores, sg_neighbors = build_guidance_from_data(
        X_tr, y_tr, depth=1, m=5, seed=seed
    )

    # ---- MinPPI estimation ----
    minppi = estimate_minppi(
        X_tr,
        X_val,
        y_tr,
        y_val,
        alpha=alpha,
        method="hybrid",
        K_adv=K_adv,
        shapg_neighbors=sg_neighbors,
        shapg_scores=sg_scores,
        seed=seed,
        verbose=verbose,
    )

    # ---- MOFS-MinPPI search ----
    result = mofs_minppi_search(
        X_tr,
        X_val,
        y_tr,
        y_val,
        minppi=minppi,
        alpha=alpha,
        beta=beta,
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        verbose=verbose,
    )

    # ---- Side-channel for HV / compression ----
    if _result_box is not None:
        _result_box.append(result)

    return pareto_feature_importance(result)


def rank_features(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Convert a score dict to a ranked list (highest first)."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("MOFS-MinPPI baseline — synthetic demo")
    rng = np.random.default_rng(42)
    n = 200
    X_demo = pd.DataFrame(
        {
            "f0": rng.normal(0, 1, n),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "f3": rng.normal(0, 1, n),
            "f4": rng.normal(0, 1, n),
        }
    )
    y_demo = np.where(X_demo["f0"] + 0.1 * X_demo["f1"] > 0, 1, 0)

    result_box: list = []
    scores = compute_mofs_minppi_importance(
        X_demo,
        y_demo,
        alpha=0.90,
        K_adv=10,
        pop_size=10,
        generations=5,
        seed=42,
        verbose=True,
        _result_box=result_box,
    )

    print("\nFeature importance ranking:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.4f}")

    if result_box:
        res = result_box[0]
        print(f"\nPareto front: {len(res.pareto_masks)} solutions")
        print(f"Hypervolume: {res.hypervolume:.4f}")
        print(f"Runtime: {res.runtime_s:.1f}s")

    print("\nObservation: f0 should score highest (decisive).")
