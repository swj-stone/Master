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

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from minppi_mofs.min_ppi import estimate_minppi, estimate_minppi_random_mc
from minppi_mofs.mofs_minppi import (
    mofs_minppi_search,
)
from minppi_mofs.shapg_integration import build_guidance_from_data


# ---------------------------------------------------------------------------
# Per-dataset budget  (mirrors experiments/run_ablation.py)
# ---------------------------------------------------------------------------

DATASET_BUDGET: dict[str, dict] = {
    "spambase": {"K_adv": 30, "pop": 30, "gen": 20},
    "madelon": {"K_adv": 20, "pop": 20, "gen": 15},
    "isolet": {"K_adv": 8, "pop": 20, "gen": 10},
    "gisette": {"K_adv": 10, "pop": 15, "gen": 8},
    "credit_card_fraud": {"K_adv": 30, "pop": 30, "gen": 20},
    "crimedata": {"K_adv": 8, "pop": 15, "gen": 8},
    "abalone": {"K_adv": 8, "pop": 30, "gen": 20},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _refit_pareto(
    masks: np.ndarray,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    features: list[str],
    *,
    n_classes: int = 2,
    seed: int = 42,
    task: str = "binary",
) -> np.ndarray:
    """Re-evaluate Pareto-front masks with the same strong XGBoost as
    ``evaluate_topk`` (n_estimators=100, max_depth=4).

    Returns fitness array of shape (n_masks, 3):
        (|F|, 1−AUC, training_time_seconds).
    """
    fitness = np.zeros((len(masks), 3), dtype=float)
    for i, mask in enumerate(masks):
        cols = [features[j] for j in range(len(features)) if mask[j]]
        if len(cols) == 0:
            fitness[i] = (0.0, 1.0, 0.0)
            continue
        X_tr_sub = X_train[cols].values
        X_val_sub = X_val[cols].values
        t0 = __import__("time").perf_counter()

        if task == "regression":
            model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_sub, y_train)
            y_pred = model.predict(X_val_sub)
            quality = float(r2_score(y_val, y_pred))
            if quality < 0.0:
                quality = 0.0
            auc = quality
        else:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                verbosity=0,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_sub, y_train)
            y_prob = model.predict_proba(X_val_sub)
            if n_classes == 2:
                auc = float(roc_auc_score(y_val, y_prob[:, 1]))
            else:
                auc = float(accuracy_score(y_val, model.predict(X_val_sub)))
        elapsed = __import__("time").perf_counter() - t0
        fitness[i] = (float(len(cols)), 1.0 - auc, float(elapsed))
    return fitness


def _pareto_importance_from_fitness(
    masks: np.ndarray,
    fitness: np.ndarray,
    features: list[str],
) -> dict[str, float]:
    """Feature importance from refit fitness: best AUC among solutions
    that include each feature."""
    auc_col = 1.0 - fitness[:, 1]
    base = float(auc_col.min())
    scores: dict[str, float] = {}
    for j, f in enumerate(features):
        in_sol = masks[:, j] == 1
        scores[f] = float(auc_col[in_sol].max()) if in_sol.any() else base - 0.01
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


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
    task: str = "binary",
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
        task: ``"binary"`` (default) uses XGBClassifier + AUC;
            ``"regression"`` uses XGBRegressor + (1−R²).  When
            ``"regression"``, ShapG guidance is skipped.
        _result_box: If a ``list``, the :class:`MOFSMinPPIResult` is
            appended so the caller can extract HV / compression without
            re-running the search.

    Returns:
        Dict ``{feature_name: frequency_in_pareto_front}`` sorted descending.
    """
    # ---- Split for MinPPI estimation ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )
    n_classes = len(np.unique(y_val))

    if verbose:
        print(f"  MOFS-MinPPI: {X.shape[1]} features", flush=True)

    # ---- ShapG guidance (binary only; multiclass/regression use Algorithm A) ----
    sg_scores: dict[str, float] = {}
    sg_neighbors = None
    if n_classes == 2 and task != "regression":
        sg_scores, sg_neighbors = build_guidance_from_data(
            X_tr, y_tr, depth=1, m=5, seed=seed,
        )

    # ---- MinPPI estimation ----
    # Multiclass (e.g. 26-class isolet): shared-coalition Algorithm A
    # for tractable wall time. Binary: hybrid dispatcher (Algorithm B).
    if n_classes > 2:
        minppi = estimate_minppi_random_mc(
            X_tr, X_val, y_tr, y_val,
            alpha=alpha, K=200, shared_coalitions=True,
            per_coalition=5, seed=seed, verbose=verbose,
        )
    else:
        minppi = estimate_minppi(
            X_tr, X_val, y_tr, y_val,
            alpha=alpha, method="hybrid", K_adv=K_adv,
            shapg_neighbors=sg_neighbors, shapg_scores=sg_scores,
            seed=seed, verbose=verbose,
        )

    # ---- MOFS-MinPPI search (fast weak model) ----
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

    # ---- Refit Pareto front with strong model (matches evaluate_topk) ----
    n_classes = len(np.unique(y))
    refit_fitness = _refit_pareto(
        result.pareto_masks,
        X_tr,
        X_val,
        y_tr,
        y_val,
        result.features,
        n_classes=n_classes,
        seed=seed,
        task=task,
    )

    # ---- Side-channel: store refit fitness for HV / compression ----
    if _result_box is not None:
        _result_box.append({
            "fitness": refit_fitness,
            "masks": result.pareto_masks,
        })

    # ---- Compute importance from refit fitness ----
    return _pareto_importance_from_fitness(
        result.pareto_masks, refit_fitness, result.features
    )


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
