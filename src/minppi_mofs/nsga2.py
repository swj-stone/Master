"""
MOFS-MinPPI: NSGA-II with MinPPI-guided operators (§4.4).

Multi-objective feature selection via evolutionary search, minimising:

    (|F|,  1 − AUC(ℳ_F),  T_train(ℳ_F))

The key innovation is the **MinPPI-guided mutation and crossover** operators
(§4.4.1–4.4.2) that use three-way MinPPI classification:

- MinPPI_i = +1 (decisive): protect in mutation, keep whole blocks in crossover
- MinPPI_i = -1 (harmful): accelerate removal
- MinPPI_i = 0 (neutral): uniform treatment

This replaces SHAP-mean guidance with min-marginal guidance, providing a
cleaner signal from the voting-game formulation.

Usage::

    from minppi_mofs.nsga2 import mofs_minppi_search
    result = mofs_minppi_search(X_train, X_val, y_train, y_val,
                                 minppi=minppi_dict, alpha=0.95, seed=42)
    # result.pareto_front: list of (mask, fitness) tuples
    # result.minppi: the MinPPI values used for guidance
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# NSGA-II core algorithms
# ---------------------------------------------------------------------------


def non_dominated_sort(fitness: np.ndarray) -> list[list[int]]:
    """Non-dominated sorting (Deb et al. 2002, *IEEE TEC* 6(2)).

    All objectives are minimised.  A solution *a* dominates *b* iff
    ``∀j: a_j ≤ b_j`` and ``∃j: a_j < b_j``.

    Args:
        fitness: Array of shape (N, M) — N individuals, M objectives.

    Returns:
        List of fronts, each front is a list of indices into *fitness*.
        Front 0 is the Pareto front.
    """
    n = fitness.shape[0]
    dominated_count = np.zeros(n, dtype=int)
    dominates: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                dominates[i].append(j)
            elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                dominated_count[i] += 1

    fronts: list[list[int]] = []
    current = [i for i in range(n) if dominated_count[i] == 0]
    while current:
        fronts.append(current)
        next_front: list[int] = []
        for i in current:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current = next_front
    return fronts


def crowding_distance(fitness: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for a single non-dominated front.

    Larger distance → more isolated → preferred when selecting within a front.

    Args:
        fitness: Array of shape (N, M).
        front: List of indices defining this front.

    Returns:
        Array of length ``len(front)`` with crowding distances.
    """
    n_front = len(front)
    dist = np.zeros(n_front, dtype=float)
    if n_front <= 2:
        dist[:] = np.inf
        return dist

    n_obj = fitness.shape[1]
    for obj in range(n_obj):
        order = sorted(front, key=lambda i: fitness[i, obj])
        f_min = fitness[order[0], obj]
        f_max = fitness[order[-1], obj]
        if f_max == f_min:
            continue
        dist[0] = np.inf
        dist[-1] = np.inf
        for k in range(1, n_front - 1):
            dist[k] += (fitness[order[k + 1], obj] - fitness[order[k - 1], obj]) / (
                f_max - f_min
            )
    return dist


def tournament_select(
    pop_size: int,
    fronts: list[list[int]],
    crowding: list[np.ndarray],
    rank_map: dict[int, int],
    pos_map: dict[int, int],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Binary tournament selection — returns two parent indices.

    Lower rank (front number) wins.  On tie, higher crowding distance wins.

    Args:
        pop_size: Current population size.
        fronts: Non-dominated fronts.
        crowding: Crowding distances per front.
        rank_map: ``{individual_index: front_number}``.
        pos_map: ``{individual_index: position_within_front}``.
        rng: Seeded random generator.

    Returns:
        A tuple ``(parent_1_index, parent_2_index)``.
    """
    cand = rng.choice(pop_size, size=4, replace=False)

    def _better(a: int, b: int) -> int:
        r_a, r_b = rank_map[a], rank_map[b]
        if r_a < r_b:
            return a
        if r_b < r_a:
            return b
        cd_a = crowding[r_a][pos_map[a]]
        cd_b = crowding[r_b][pos_map[b]]
        return a if cd_a > cd_b else b

    return _better(cand[0], cand[1]), _better(cand[2], cand[3])


# ---------------------------------------------------------------------------
# MinPPI-guided operators (§4.4.1, §4.4.2)
# ---------------------------------------------------------------------------


def minppi_mutation(
    mask: np.ndarray,
    minppi: dict[int, int],
    *,
    beta: float = 1.0,
    p_base: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """MinPPI-guided bit-flip mutation (§4.4.1).

    Flip probability is modulated by MinPPI classification:

    .. math::

        p_{flip}(i) = \\begin{cases}
        p_{base} \\cdot \\exp(-\\beta) & \\text{if MinPPI}_i = +1 \\\\
        p_{base} \\cdot \\exp(+\\beta) & \\text{if MinPPI}_i = -1 \\\\
        p_{base} & \\text{if MinPPI}_i = 0
        \\end{cases}

    Directional bias:
    - ``mask_i = 1`` and ``MinPPI_i = -1``: strongly bias toward flipping to 0
    - ``mask_i = 0`` and ``MinPPI_i = +1``: strongly bias toward flipping to 1

    Args:
        mask: Binary mask of shape (n_features,).
        minppi: ``{feature_index: value}`` with values in {-1, 0, +1}.
        beta: Guidance strength (β ≥ 0). β = 0 → uniform baseline.
        p_base: Base mutation probability.
        rng: Seeded random generator.

    Returns:
        Mutated binary mask.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(mask)
    mutated = mask.copy()

    for i in range(n):
        val = minppi.get(i, 0)
        if val == -1:
            p = min(p_base * np.exp(beta), 0.5)
        elif val == +1:
            p = p_base * np.exp(-beta)
        else:
            p = p_base

        if rng.random() < p:
            if val == -1 and mutated[i] == 1:
                # Harmful + currently selected → strong bias to remove
                mutated[i] = 0 if rng.random() < 0.8 else 1
            elif val == +1 and mutated[i] == 0:
                # Decisive + currently absent → bias to add
                mutated[i] = 1 if rng.random() < 0.8 else 0
            else:
                mutated[i] = 1 - mutated[i]  # uniform flip

    return mutated


def minppi_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    minppi: dict[int, int],
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """MinPPI-guided crossover (§4.4.2).

    Decisive features (MinPPI = +1) are inherited as a **block** from one
    parent to preserve winning coalitions.  Non-decisive positions use
    uniform crossover.

    Args:
        parent1: First parent mask (n_features,).
        parent2: Second parent mask (n_features,).
        minppi: ``{feature_index: value}``.
        rng: Seeded random generator.

    Returns:
        Child binary mask.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(parent1)
    child = np.zeros(n, dtype=np.int8)

    # Decisive block: inherit as a contiguous block from one parent
    decisive_idx = [i for i in range(n) if minppi.get(i, 0) == +1]
    if decisive_idx:
        src = rng.choice([parent1, parent2])
        for i in decisive_idx:
            child[i] = src[i]

    # Non-decisive positions: uniform crossover
    for i in range(n):
        if minppi.get(i, 0) != +1:
            child[i] = parent1[i] if rng.random() < 0.5 else parent2[i]

    return child


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


def _evaluate_mask(
    mask: np.ndarray,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    features: list[str],
    *,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Evaluate a single mask on the three objectives.

    Returns:
        ``(n_features, 1 − AUC, training_time_seconds)``.
    """
    cols = [features[j] for j in range(len(features)) if mask[j]]
    if len(cols) == 0:
        return (0.0, 1.0, 0.0)

    X_tr_sub = X_train[cols].values
    X_val_sub = X_val[cols].values
    n_classes = len(np.unique(y_val))

    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.3,
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
        y_pred = model.predict(X_val_sub)
        auc = float(accuracy_score(y_val, y_pred))

    elapsed = time.perf_counter() - t0
    return (float(len(cols)), 1.0 - auc, float(elapsed))


# ---------------------------------------------------------------------------
# Hypervolume computation
# ---------------------------------------------------------------------------


def compute_hypervolume(
    fitness: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """Compute hypervolume of a Pareto front relative to a reference point.

    Uses a simple 2D sweep for (|F|, 1−AUC).  For 3D, falls back to
    retangular-cell decomposition.

    Args:
        fitness: Pareto-optimal fitness values (K, M).
        ref_point: Reference point (M,) — worse than any feasible solution.

    Returns:
        Hypervolume value.
    """
    if len(fitness) == 0:
        return 0.0

    if fitness.shape[1] == 2:
        # 2D hypervolume: sort by x ascending, sweep left to right.
        # At slice [x_{i-1}, x_i], the best (lowest) y among points with
        # index < i determines the Pareto-envelope height.
        sorted_idx = np.argsort(fitness[:, 0])
        sorted_fit = fitness[sorted_idx]
        hv = 0.0
        prev_x = 0.0
        y_envelope = ref_point[1]  # best y so far (lower is better)
        for i in range(len(sorted_fit)):
            x, y = float(sorted_fit[i, 0]), float(sorted_fit[i, 1])
            if x > ref_point[0]:
                continue
            hv += (x - prev_x) * (ref_point[1] - y_envelope)
            prev_x = x
            y_envelope = min(y_envelope, y)
        hv += (ref_point[0] - prev_x) * (ref_point[1] - y_envelope)
        return max(0.0, float(hv))

    # 3D: Monte Carlo estimate
    rng = np.random.default_rng(42)
    n_samples = 5000
    f_min = fitness.min(axis=0)
    hits = 0
    for _ in range(n_samples):
        pt = rng.uniform(f_min, ref_point)
        if np.any(np.all(fitness <= pt, axis=1)):
            hits += 1
    volume = np.prod(ref_point - f_min)
    return float(hits / n_samples) * volume


# ---------------------------------------------------------------------------
# MOFS-MinPPI search result
# ---------------------------------------------------------------------------


@dataclass
class MOFSMinPPIResult:
    """Result of one MOFS-MinPPI run.

    Attributes:
        pareto_masks: Binary masks on the final Pareto front (P × n_features).
        pareto_fitness: Corresponding fitness values (P × 3).
        minppi: The MinPPI values used for guidance.
        hypervolume: Hypervolume of the final Pareto front.
        history: Per-generation best HV for convergence tracking.
        runtime_s: Total wall-clock time in seconds.
        features: Feature name list.
    """

    pareto_masks: np.ndarray
    pareto_fitness: np.ndarray
    minppi: dict[int, int]
    hypervolume: float
    history: list[dict]
    runtime_s: float
    features: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main MOFS-MinPPI search
# ---------------------------------------------------------------------------


def mofs_minppi_search(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    minppi: dict[str, int],
    alpha: float = 0.95,
    beta: float = 1.0,
    pop_size: int = 50,
    generations: int = 30,
    p_base: float = 0.1,
    ref_point: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = False,
) -> MOFSMinPPIResult:
    """Run the full MOFS-MinPPI evolutionary search (§4.4.3).

    Multi-objective formulation: minimise (|F|, 1−AUC, T_train).

    Pipeline:
    1. Initialise random population of binary masks.
    2. For each generation:
       a. Select parents via binary tournament on rank + crowding.
       b. MinPPI-guided crossover → child.
       c. MinPPI-guided mutation → child.
       d. Evaluate child on all three objectives.
       e. Merge parents + children, keep top *pop_size* via NSGA-II selection.
    3. Return final Pareto front.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        minppi: Pre-computed MinPPI dict ``{feature_name: value ∈ {-1,0,+1}}``.
        alpha: Quota parameter (for metadata).
        beta: Guidance strength (β ≥ 0). β = 0 → uniform baseline.
        pop_size: Population size (must be even).
        generations: Maximum number of generations.
        p_base: Base mutation probability.
        ref_point: Reference point for hypervolume (auto-computed if None).
        seed: Random seed.
        verbose: Print per-generation progress.

    Returns:
        :class:`MOFSMinPPIResult` with the final Pareto front and metadata.
    """
    rng = np.random.default_rng(seed)
    features = list(X_train.columns)
    n_features = len(features)
    pop_size = (pop_size // 2) * 2  # ensure even

    # Map string-keyed MinPPI to integer-indexed for operator use
    minppi_idx: dict[int, int] = {}
    for j, f in enumerate(features):
        minppi_idx[j] = minppi.get(f, 0)

    if verbose:
        n_pos = sum(1 for v in minppi.values() if v == +1)
        n_neg = sum(1 for v in minppi.values() if v == -1)
        print(
            f"  MOFS-MinPPI: pop={pop_size}, gen={generations}, β={beta:.1f}",
            flush=True,
        )
        print(
            f"  MinPPI distribution: +1={n_pos}, 0={n_features - n_pos - n_neg}, "
            f"-1={n_neg}",
            flush=True,
        )

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Initialise population
    # ------------------------------------------------------------------
    pop = rng.choice([0, 1], size=(pop_size, n_features)).astype(np.int8)
    fitness = np.full((pop_size, 3), np.inf, dtype=float)

    for i in range(pop_size):
        if np.sum(pop[i]) == 0:
            pop[i, rng.integers(0, n_features)] = 1
        fitness[i] = _evaluate_mask(
            pop[i], X_train, X_val, y_train, y_val, features, seed=seed
        )

    # Auto reference point if not given: worst observed × 1.1 in each dimension
    if ref_point is None:
        ref_point = np.max(fitness, axis=0) * 1.1
        ref_point[ref_point == 0] = 1.0

    history: list[dict] = []

    # ------------------------------------------------------------------
    # 2. Evolution loop
    # ------------------------------------------------------------------
    for gen in range(generations):
        fronts = non_dominated_sort(fitness)
        crowding = [crowding_distance(fitness, f) for f in fronts]
        rank_map: dict[int, int] = {}
        pos_map: dict[int, int] = {}
        for ri, front in enumerate(fronts):
            for pi, idx in enumerate(front):
                rank_map[idx] = ri
                pos_map[idx] = pi

        # Generate offspring
        offspring = np.empty_like(pop)
        for i in range(0, pop_size, 2):
            p1, p2 = tournament_select(
                pop_size, fronts, crowding, rank_map, pos_map, rng
            )
            c1 = minppi_crossover(pop[p1], pop[p2], minppi_idx, rng=rng)
            c2 = minppi_crossover(pop[p2], pop[p1], minppi_idx, rng=rng)
            c1 = minppi_mutation(c1, minppi_idx, beta=beta, p_base=p_base, rng=rng)
            c2 = minppi_mutation(c2, minppi_idx, beta=beta, p_base=p_base, rng=rng)
            offspring[i] = c1
            if i + 1 < pop_size:
                offspring[i + 1] = c2

        # Evaluate offspring
        off_fitness = np.full((pop_size, 3), np.inf, dtype=float)
        for i in range(pop_size):
            if np.sum(offspring[i]) == 0:
                offspring[i, rng.integers(0, n_features)] = 1
            off_fitness[i] = _evaluate_mask(
                offspring[i],
                X_train,
                X_val,
                y_train,
                y_val,
                features,
                seed=seed,
            )

        # Merge and select
        combined = np.vstack([pop, offspring])
        combined_fit = np.vstack([fitness, off_fitness])
        comb_fronts = non_dominated_sort(combined_fit)
        comb_crowd = [crowding_distance(combined_fit, f) for f in comb_fronts]

        selected: list[int] = []
        for front in comb_fronts:
            remaining = pop_size - len(selected)
            if len(front) <= remaining:
                selected.extend(front)
            else:
                cd = comb_crowd[comb_fronts.index(front)]
                order = sorted(range(len(front)), key=lambda k: cd[k], reverse=True)
                selected.extend([front[k] for k in order[:remaining]])
                break

        pop = combined[selected].astype(np.int8)
        fitness = combined_fit[selected]

        # Track hypervolume
        elite_idx = comb_fronts[0]
        hv = compute_hypervolume(combined_fit[elite_idx], ref_point)
        history.append(
            {
                "generation": gen + 1,
                "front0_size": len(comb_fronts[0]),
                "hypervolume": hv,
                "avg_features": float(pop.sum() / pop_size),
                "best_auc": float(1.0 - fitness[0, 1]),
            }
        )

        if verbose and (gen + 1) % 5 == 0:
            print(
                f"  Gen {gen + 1:>3d}/{generations}: "
                f"front0={len(comb_fronts[0]):>2d}, "
                f"HV={hv:.4f}, "
                f"avg_feats={pop.sum() / pop_size:.1f}, "
                f"best_AUC={1.0 - fitness[0, 1]:.4f}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 3. Final Pareto front
    # ------------------------------------------------------------------
    final_fronts = non_dominated_sort(fitness)
    elite = final_fronts[0]
    pareto_masks = pop[elite]
    pareto_fitness = fitness[elite]
    final_hv = compute_hypervolume(pareto_fitness, ref_point)

    runtime = time.perf_counter() - t_start

    if verbose:
        feat_counts = [int(m.sum()) for m in pareto_masks]
        print(
            f"  Final: {len(elite)} solutions on Pareto front, "
            f"HV={final_hv:.4f}, "
            f"feat counts: {sorted(feat_counts)[:10]}..., "
            f"runtime={runtime:.1f}s",
            flush=True,
        )

    return MOFSMinPPIResult(
        pareto_masks=pareto_masks,
        pareto_fitness=pareto_fitness,
        minppi=minppi_idx,
        hypervolume=final_hv,
        history=history,
        runtime_s=runtime,
        features=features,
    )


# ---------------------------------------------------------------------------
# Utility: Pareto front → feature importance
# ---------------------------------------------------------------------------


def pareto_feature_importance(
    result: MOFSMinPPIResult,
) -> dict[str, float]:
    """Extract feature importance from a MOFS-MinPPI Pareto front.

    Importance = frequency of each feature in the elite front,
    weighted by crowding distance (more isolated solutions count more).

    Args:
        result: A :class:`MOFSMinPPIResult`.

    Returns:
        Dict ``{feature_name: score}`` sorted descending.
    """
    masks = result.pareto_masks
    if len(masks) == 0:
        return {}
    freq = masks.mean(axis=0)
    scores = dict(zip(result.features, freq.astype(float)))
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("MOFS-MinPPI — synthetic demo")

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
    y_demo = np.where(X_demo["f_signal"] + 0.2 * X_demo["f_weak"] > 0, 1, 0).astype(int)

    n_tr = int(0.7 * n_samp)
    X_tr, X_val = X_demo.iloc[:n_tr], X_demo.iloc[n_tr:]
    y_tr, y_val = y_demo[:n_tr], y_demo[n_tr:]

    # Toy MinPPI: signal is decisive, noise is harmful
    toy_minppi = {
        "f_signal": +1,
        "f_weak": 0,
        "f_noise1": -1,
        "f_noise2": -1,
        "f_noise3": -1,
    }

    result = mofs_minppi_search(
        X_tr,
        X_val,
        y_tr,
        y_val,
        minppi=toy_minppi,
        alpha=0.90,
        beta=1.0,
        pop_size=20,
        generations=10,
        seed=42,
        verbose=True,
    )

    print(f"\nFinal Pareto front: {len(result.pareto_masks)} solutions")
    print(f"Hypervolume: {result.hypervolume:.4f}")
    print(f"Runtime: {result.runtime_s:.1f}s")

    imp = pareto_feature_importance(result)
    print("\nFeature importance (Pareto-front frequency):")
    for feat, score in imp.items():
        label = {+1: "DECISIVE", 0: "neutral", -1: "HARMFUL"}[toy_minppi.get(feat, 0)]
        print(f"  {feat:15s}  {score:.4f}  ({label})")

    print(
        "\nObservation: Decisive features should appear most often; "
        "harmful features should be excluded from the Pareto front. "
        "β=1 gives moderate guidance; β=0 would produce uniform behaviour."
    )
