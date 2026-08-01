"""
NSGA-II Feature Selection baseline.

Standard multi-objective feature selection using NSGA-II with two
objectives: minimise cardinality ratio (|S|/n) and minimise error
(1 - clipped utility).  The evolutionary search remains independent of the
MinPPI estimator and voting-game logic, while candidate scoring uses the
package's canonical modeling boundary.  Experiment stages may optionally
pre-filter features with an already-frozen MinPPI result.

Usage (standalone)::

    from minppi_mofs.baselines.nsga2_fs import nsga2_feature_selection
    result = nsga2_feature_selection(X_train, y_train, X_val, y_val,
                                     task="binary", seed=42)
    # result is an NSGA2Result dataclass with pareto_front, hv, etc.

Usage (ranking adapter)::

    from minppi_mofs.baselines.nsga2_fs import nsga2_ranking
    ranking = nsga2_ranking(X, y, task="binary", seed=42)
    # Returns list of (feature_name, importance_score) tuples.

Run (stand-alone)::

    ../venv/Scripts/python.exe src/minppi_mofs/baselines/nsga2_fs.py
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from minppi_mofs.modeling import evaluate_fixed_subset


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NSGA2Result:
    """Output of ``nsga2_feature_selection``.

    Attributes:
        pareto_front: List of
            ``(binary_mask, (card_ratio, 1 - clipped_utility))`` tuples.
        hv: Hypervolume with respect to the configured reference point.
        runtime_s: Wall-clock runtime in seconds.
        converged_gen: First generation whose cumulative-best HV reaches
            ≥95% of the final-best HV (post-hoc, 1-indexed).  If no
            generation obtains positive HV, ``max_gen`` is returned as a
            sentinel for "no meaningful convergence".
        hv_history: HV at each generation (len = max_gen).
    """

    pareto_front: list  # list[tuple[np.ndarray, tuple[float, float]]]
    hv: float
    runtime_s: float
    converged_gen: int
    hv_history: list[float]
    logical_evaluation_count: int
    unique_subset_count: int
    memoization_hit_count: int


# ---------------------------------------------------------------------------
# Individual evaluation
# ---------------------------------------------------------------------------


def _evaluate_individual(
    mask: np.ndarray,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    task: str = "binary",
    utility: str | None = None,
    model_params: Mapping[str, object] | None = None,
    backend: str = "cpu",
    seed: int = 42,
    cardinality_denominator: int | None = None,
    utility_evaluator: Callable[[tuple[str, ...]], float] | None = None,
) -> tuple[float, float]:
    """Evaluate one candidate solution.

    Args:
        mask: Binary array of shape (n_features,).  1 = selected.
        X_tr, y_tr: Training data.
        X_val, y_val: Validation data.
        task: ``"binary"``, ``"multiclass"``, or ``"regression"``.
        seed: XGBoost random seed.

    Returns:
        ``(cardinality_ratio, 1 − utility)`` — both minimised.
    """
    n = X_tr.shape[1]
    denominator = n if cardinality_denominator is None else cardinality_denominator
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < max(1, n)
    ):
        raise ValueError(
            "cardinality_denominator must be a positive integer at least as large "
            "as the search dimension"
        )
    selected = tuple(X_tr.columns[j] for j in range(n) if mask[j])
    chosen_utility = utility or (
        "r2" if task == "regression" else "accuracy" if task == "multiclass" else "roc_auc"
    )
    params = (
        dict(model_params)
        if model_params is not None
        else {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_jobs": 1,
        }
    )
    value = (
        utility_evaluator(selected)
        if utility_evaluator is not None
        else evaluate_fixed_subset(
            X_tr,
            y_tr,
            X_val,
            y_val,
            selected,
            task=task,
            utility=chosen_utility,
            model_params=params,
            backend=backend,
            seed=seed,
        )
    )
    clipped = float(np.clip(value, 0.0, 1.0))
    return len(selected) / denominator, 1.0 - clipped


# ---------------------------------------------------------------------------
# NSGA-II core operators
# ---------------------------------------------------------------------------


def _non_dominated_sort(
    fitness: np.ndarray,
) -> list[np.ndarray]:
    """Fast non-dominated sort (Deb et al. 2002).

    Args:
        fitness: Array of shape (N, 2).  Both objectives minimised.

    Returns:
        List of front arrays, each containing indices into *fitness*.
    """
    n = fitness.shape[0]
    dominated_by: list[set] = [set() for _ in range(n)]
    dominates_count = [0] * n
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            fi, fj = fitness[i], fitness[j]
            if (fi <= fj).all() and (fi < fj).any():
                dominated_by[i].add(j)
            elif (fj <= fi).all() and (fj < fi).any():
                dominates_count[i] += 1
        if dominates_count[i] == 0:
            fronts[0].append(i)

    cur = 0
    while fronts[cur]:
        nxt: list[int] = []
        for i in fronts[cur]:
            for j in dominated_by[i]:
                dominates_count[j] -= 1
                if dominates_count[j] == 0:
                    nxt.append(j)
        cur += 1
        fronts.append(nxt)
    fronts.pop()
    return [np.array(f, dtype=int) for f in fronts]


def _crowding_distance(
    fitness: np.ndarray,
    front: np.ndarray,
) -> np.ndarray:
    """Crowding distance for a Pareto front.

    Boundary points receive ``inf``.
    """
    n_obj = fitness.shape[1]
    n = len(front)
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist
    for obj in range(n_obj):
        order = np.argsort(fitness[front, obj])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        f_range = fitness[front[order[-1]], obj] - fitness[front[order[0]], obj]
        if f_range > 0:
            dist[order[1:-1]] += (
                fitness[front[order[2:]], obj] - fitness[front[order[:-2]], obj]
            ) / f_range
    return dist


def _compute_hv(front: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume in 2-D for minimisation (standard sweep-line algorithm).

    Filters dominated and duplicate points, sorts by the first objective,
    and sweeps left to right accumulating rectangular contributions.
    """
    points = np.asarray(front, dtype=float)
    if points.size == 0:
        return 0.0
    # Keep only points that are not worse than the reference in every objective
    points = points[np.all(points <= ref, axis=1)]
    if len(points) == 0:
        return 0.0
    # De-duplicate and keep only non-dominated points
    points = np.unique(points, axis=0)
    points = points[_non_dominated_sort(points)[0]]
    if len(points) == 0:
        return 0.0
    # Sort by first objective ascending (second objective will be descending
    # because the points are mutually non-dominated)
    points = points[np.argsort(points[:, 0])]
    hv = 0.0
    best_y = float(ref[1])
    for x, y in points:
        if y < best_y:
            hv += (float(ref[0]) - float(x)) * (best_y - float(y))
            best_y = float(y)
    return max(hv, 0.0)


def _converged_gen(hv_history: list[float]) -> int:
    """First generation whose cumulative-best HV reaches ≥95% of the final best.

    This is a post-hoc metric — it does **not** influence the NSGA-II loop.
    An all-zero history has no meaningful positive-HV convergence, so its
    final generation is returned as a sentinel.
    """
    if not hv_history:
        return 0
    best = max(hv_history)
    if best == 0.0:
        # Avoid reporting generation 1 as "converged" when no dominated
        # hypervolume was obtained at any point in the run.
        return len(hv_history)
    cum_best = -np.inf
    for i, hv in enumerate(hv_history):
        if hv > cum_best:
            cum_best = hv
        if cum_best >= 0.95 * best:
            return i + 1  # 1-indexed generation
    return len(hv_history)


# ---------------------------------------------------------------------------
# Main NSGA-II loop
# ---------------------------------------------------------------------------


def nsga2_feature_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    task: str = "binary",
    utility: str | None = None,
    model_params: Mapping[str, object] | None = None,
    backend: str = "cpu",
    seed: int = 42,
    pop_size: int = 20,
    max_gen: int = 20,
    p_crossover: float = 0.5,
    p_mutation: Optional[float] = None,
    ref_point: tuple[float, float] = (1.0, 1.0),
    verbose: bool = False,
    cardinality_denominator: int | None = None,
    utility_evaluator: Callable[[tuple[str, ...]], float] | None = None,
) -> NSGA2Result:
    """Run NSGA-II for multi-objective feature selection.

    An individual is a binary mask ``m ∈ {0,1}^n`` where ``m[j]=1`` means
    feature *j* is selected.  Two objectives are minimised:
      - ``f₁(m) = |m|₀ / n``  (cardinality ratio)
      - ``f₂(m) = 1 − clipped utility``    (error)

    Operators: tournament selection (rank+CD), uniform crossover,
    bit-flip mutation.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data (used for objective evaluation).
        task: ``"binary"``, ``"multiclass"``, or ``"regression"``.
        seed: Random seed.
        pop_size: Population size (default 20).
        max_gen: Maximum generations (default 20).
        p_crossover: Crossover probability per gene (default 0.5).
        p_mutation: Per-bit mutation probability.  Default: ``1/n``.
        ref_point: Reference point for hypervolume.
        verbose: Print progress every 5 generations.

    Returns:
        ``NSGA2Result`` dataclass.
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[1]
    denominator = n if cardinality_denominator is None else cardinality_denominator
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < max(1, n)
    ):
        raise ValueError(
            "cardinality_denominator must be a positive integer at least as large "
            "as the search dimension"
        )
    if utility_evaluator is not None and not callable(utility_evaluator):
        raise TypeError("utility_evaluator must be callable")
    p_mut = p_mutation if p_mutation is not None else (1.0 / n if n else 0.0)
    reference = np.asarray(ref_point, dtype=float)
    t0 = time.perf_counter()
    utility_by_features: dict[tuple[str, ...], float] = {}
    logical_evaluation_count = 0
    memoization_hit_count = 0

    def evaluate_mask(mask: np.ndarray) -> tuple[float, float]:
        nonlocal logical_evaluation_count, memoization_hit_count
        logical_evaluation_count += 1
        selected = tuple(
            feature
            for feature, bit in zip(X_train.columns, mask, strict=True)
            if bit
        )
        if selected in utility_by_features:
            memoization_hit_count += 1
            clipped_utility = utility_by_features[selected]
        else:
            fitness = _evaluate_individual(
                mask,
                X_train,
                y_train,
                X_val,
                y_val,
                task=task,
                utility=utility,
                model_params=model_params,
                backend=backend,
                seed=seed,
                cardinality_denominator=denominator,
                utility_evaluator=utility_evaluator,
            )
            clipped_utility = 1.0 - fitness[1]
            utility_by_features[selected] = clipped_utility
        return len(selected) / denominator, 1.0 - clipped_utility

    # ── Initialisation ──
    pop = rng.integers(0, 2, size=(pop_size, n)).astype(np.float64)
    fit = np.array([evaluate_mask(pop[i]) for i in range(pop_size)])
    converged_gen: int = 0  # computed post-hoc below
    hv_history: list[float] = []

    # ── Generations ──
    for gen in range(max_gen):
        # Offspring via tournament selection + crossover + mutation
        offspring = np.zeros((pop_size, n))
        for i in range(0, pop_size, 2):
            # Tournament (2 candidates each)
            t1 = rng.integers(0, pop_size, 2)
            t2 = rng.integers(0, pop_size, 2)

            def _winner(a, b):
                fa, fb = fit[a], fit[b]
                if (fa <= fb).all() and (fa < fb).any():
                    return a
                if (fb <= fa).all() and (fb < fa).any():
                    return b
                return a if rng.random() < 0.5 else b

            p1_idx = _winner(t1[0], t1[1])
            p2_idx = _winner(t2[0], t2[1])

            # Uniform crossover
            cx_mask = rng.random(n) < p_crossover
            c1, c2 = pop[p1_idx].copy(), pop[p2_idx].copy()
            c1[cx_mask], c2[cx_mask] = c2[cx_mask], c1[cx_mask].copy()

            # Bit-flip mutation
            for child in (c1, c2):
                flip = rng.random(n) < p_mut
                child[flip] = 1.0 - child[flip]

            offspring[i] = c1
            if i + 1 < pop_size:
                offspring[i + 1] = c2

        off_fit = np.array(
            [evaluate_mask(offspring[j]) for j in range(pop_size)]
        )

        # Merge + non-dominated sort + select
        combined = np.vstack([pop, offspring])
        combined_fit = np.vstack([fit, off_fit])

        fronts = _non_dominated_sort(combined_fit)
        new_pop = np.zeros((pop_size, n))
        new_fit = np.zeros((pop_size, 2))
        idx = 0
        for fr in fronts:
            if idx + len(fr) <= pop_size:
                new_pop[idx : idx + len(fr)] = combined[fr]
                new_fit[idx : idx + len(fr)] = combined_fit[fr]
                idx += len(fr)
            else:
                cd = _crowding_distance(combined_fit, fr)
                order = np.argsort(cd)[::-1]
                remain = pop_size - idx
                new_pop[idx:] = combined[fr[order[:remain]]]
                new_fit[idx:] = combined_fit[fr[order[:remain]]]
                break
        pop, fit = new_pop, new_fit

        # HV tracking
        pf_idx = _non_dominated_sort(fit)[0]
        hv = _compute_hv(fit[pf_idx], reference)
        hv_history.append(hv)

        if verbose and (gen % 5 == 0 or gen == max_gen - 1):
            print(
                f"  NSGA2 gen {gen:3d}: HV={hv:.4f}  {len(pf_idx)} non-dominated",
                flush=True,
            )

    runtime = time.perf_counter() - t0
    converged_gen = _converged_gen(hv_history)
    pf_idx = _non_dominated_sort(fit)[0]
    hv_final = _compute_hv(fit[pf_idx], reference)

    # Build returnable front
    pf_list: list = []
    for i in pf_idx:
        pf_list.append((pop[i].astype(int), (float(fit[i, 0]), float(fit[i, 1]))))

    unique_subset_count = len(utility_by_features)
    if (
        logical_evaluation_count
        != unique_subset_count + memoization_hit_count
    ):
        raise RuntimeError("NSGA-II memoization accounting is inconsistent")
    return NSGA2Result(
        pareto_front=pf_list,
        hv=hv_final,
        runtime_s=runtime,
        converged_gen=converged_gen,
        hv_history=hv_history,
        logical_evaluation_count=logical_evaluation_count,
        unique_subset_count=unique_subset_count,
        memoization_hit_count=memoization_hit_count,
    )


# ---------------------------------------------------------------------------
# Ranking adapter for callers that require one score per feature
# ---------------------------------------------------------------------------


def nsga2_ranking(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str = "binary",
    seed: int = 42,
    pop_size: int = 20,
    max_gen: int = 20,
    verbose: bool = True,
) -> list[tuple[str, float]]:
    """Produce a feature ranking from NSGA-II for the main-table comparison.

    Strategy:
      1. Run NSGA-II to find the Pareto front.
      2. Select the Pareto-optimal individual with the best AUC (lowest f₂).
      3. Train a strong XGBoost on the features that individual selects.
      4. Extract ``feature_importances_`` (gain) and sort descending.
      5. Append unselected features at the bottom (zero importance).

    This gives NSGA-II a fair ranking that reflects its multi-objective
    selection, comparable to the rank-by-score output of SHAP/Powershap/etc.

    Args:
        X: Feature DataFrame.
        y: Target array.
        task: ``"binary"`` / ``"multiclass"`` / ``"regression"``.
        seed: Random seed.
        pop_size: NSGA-II population size.
        max_gen: Maximum NSGA-II generations.
        verbose: Print NSGA-II progress.

    Returns:
        List of ``(feature_name, importance_score)`` in descending order.
    """
    # Use the same data as train and validation (single-split evaluation)
    result = nsga2_feature_selection(
        X,
        y,
        X,
        y,
        task=task,
        seed=seed,
        pop_size=pop_size,
        max_gen=max_gen,
        verbose=verbose,
    )

    features = list(X.columns)
    n = len(features)

    if not result.pareto_front:
        # Degenerate: all individuals empty → uniform ranking
        return [(f, 0.0) for f in features]

    # Find the front solution with the best utility (lowest f₂ = 1 − AUC)
    best_mask = min(result.pareto_front, key=lambda x: x[1][1])[0]
    selected = [features[j] for j in range(n) if best_mask[j]]

    if len(selected) < 2:
        # Too few features selected → fall back to SHAP ranking
        from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance

        scores = compute_shap_topk_importance(X, y, seed=seed, task=task)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # Train strong XGBoost on selected features → get gain-based importance
    X_sel = X[selected].values
    if task == "regression":
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=seed,
        )
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
        model.fit(X_sel, y)

    imp = dict(zip(selected, model.feature_importances_.astype(float)))
    ranking = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)

    # Append unselected features with zero importance
    unselected = [f for f in features if f not in selected]
    ranking += [(f, 0.0) for f in unselected]

    return ranking


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("NSGA-II feature selection — synthetic demo")
    rng = np.random.default_rng(42)
    n_samp = 300
    X_demo = pd.DataFrame(
        {
            "f_signal": rng.normal(0, 1, n_samp),
            "f_weak": rng.normal(0, 1, n_samp),
            "f_noise1": rng.normal(0, 1, n_samp),
            "f_noise2": rng.normal(0, 1, n_samp),
            "f_noise3": rng.normal(0, 1, n_samp),
        }
    )
    y_demo = np.where(X_demo["f_signal"] + 0.2 * X_demo["f_weak"] > 0, 1, 0)

    result = nsga2_feature_selection(
        X_demo,
        y_demo,
        X_demo,
        y_demo,
        task="binary",
        seed=42,
        pop_size=10,
        max_gen=10,
        verbose=True,
    )
    print(f"\nHV: {result.hv:.4f}")
    print(f"Runtime: {result.runtime_s:.1f}s")
    print(f"Converged gen: {result.converged_gen}")
    print(f"Pareto front size: {len(result.pareto_front)}")

    ranking = nsga2_ranking(
        X_demo, y_demo, task="binary", seed=42, pop_size=10, max_gen=10
    )
    print("\nNSGA-II feature ranking:")
    for feat, val in ranking[:5]:
        print(f"  {feat:20s}  {val:.6f}")
    print("Observation: f_signal and f_weak should rank above noise features.")
