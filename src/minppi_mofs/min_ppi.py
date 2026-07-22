"""
MinPPI estimation — MinSS / MinBz computation (§4.2–4.3).

Provides three estimation strategies for the Min-Marginal Power Index:

1. **Exact enumeration** (n ≤ 20): enumerate all 2^{n-1} coalitions per feature.
2. **Algorithm A — Random MC** (§4.3.1): sample K random coalitions, take min.
   Upward biased (sample min ≥ population min).  Proposition 1 gives the
   concentration bound for harmful detection.
3. **Algorithm B — MI-Guided Adversarial Search** (§4.3.2): use MI-guided
   correlation graph to find the coalitions where feature *i* is most likely
   to be harmful, then adversarially search them.

Key theoretical results implemented here:
- **Lemma 1** (Boundedness): MinPPI_i ∈ {-1, 0, +1}
- **Lemma 2** (Coincidence): MinSS_i = MinBz_i
- **Theorem 1** (Three-Way Classification): decisive / harmful / neutral

Usage::

    from minppi_mofs.min_ppi import estimate_minppi
    minppi = estimate_minppi(X_train, X_val, y_train, y_val, alpha=0.95,
                              method="mi_correlation_guided", seed=42)
    # minppi: dict {feature_name: minppi_value}  (values ∈ {-1, 0, +1})
"""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from minppi_mofs.voting_game import AccFullCache, make_acc_full_cache


SharedCoalitionStep = tuple[frozenset[str], tuple[str, ...]]
SharedCoalitionPlan = tuple[SharedCoalitionStep, ...]


@dataclass(frozen=True)
class AlgorithmBPerturbationTrial:
    """Guidance-independent random inputs for one feature/trial pair."""

    removal_draws: tuple[float, ...]
    add_count: int
    add_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.removal_draws) is not tuple:
            raise ValueError("Algorithm B removal draws must be an immutable tuple")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
            or value >= 1
            for value in self.removal_draws
        ):
            raise ValueError("Algorithm B removal draws must be finite numeric values")
        if (
            isinstance(self.add_count, bool)
            or not isinstance(self.add_count, int)
            or self.add_count < 0
        ):
            raise ValueError("Algorithm B add count must be a non-negative integer")
        if type(self.add_order) is not tuple or not all(
            isinstance(feature, str) for feature in self.add_order
        ):
            raise ValueError("Algorithm B add order must be an immutable string tuple")


@dataclass(frozen=True)
class AlgorithmBPerturbationSchedule:
    """Immutable Algorithm B stochastic schedule bound to ordered features."""

    features: tuple[str, ...]
    seed: int
    K_adv: int
    seed_orders: tuple[tuple[str, ...], ...]
    trials: tuple[tuple[AlgorithmBPerturbationTrial, ...], ...]

    def __post_init__(self) -> None:
        _validate_algorithm_b_perturbation_schedule(
            self, self.features, self.K_adv, self.seed
        )


def make_algorithm_b_perturbation_schedule(
    features: Sequence[str], *, K_adv: int, seed: int
) -> AlgorithmBPerturbationSchedule:
    """Pre-generate every Algorithm B random input without guidance state."""
    feature_order = tuple(features)
    if not feature_order or len(set(feature_order)) != len(feature_order):
        raise ValueError("Algorithm B schedule features must be nonempty and unique")
    if isinstance(K_adv, bool) or not isinstance(K_adv, int) or K_adv < 0:
        raise ValueError("Algorithm B schedule K_adv must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Algorithm B schedule seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    seed_orders: list[tuple[str, ...]] = []
    feature_trials: list[tuple[AlgorithmBPerturbationTrial, ...]] = []
    for feature in feature_order:
        others = tuple(candidate for candidate in feature_order if candidate != feature)
        seed_orders.append(tuple(rng.permutation(others).tolist()))
        trials: list[AlgorithmBPerturbationTrial] = []
        for _ in range(K_adv):
            trials.append(
                AlgorithmBPerturbationTrial(
                    removal_draws=tuple(float(value) for value in rng.random(len(feature_order))),
                    add_count=int(rng.integers(0, min(5, len(others)) + 1)),
                    add_order=tuple(rng.permutation(others).tolist()),
                )
            )
        feature_trials.append(tuple(trials))
    schedule = AlgorithmBPerturbationSchedule(
        features=feature_order,
        seed=seed,
        K_adv=K_adv,
        seed_orders=tuple(seed_orders),
        trials=tuple(feature_trials),
    )
    _validate_algorithm_b_perturbation_schedule(schedule, feature_order, K_adv, seed)
    return schedule


def algorithm_b_schedule_digest(schedule: AlgorithmBPerturbationSchedule) -> str:
    """Return the canonical SHA-256 identity of an immutable schedule."""
    if not isinstance(schedule, AlgorithmBPerturbationSchedule):
        raise TypeError("Algorithm B schedule must be AlgorithmBPerturbationSchedule")
    _validate_algorithm_b_perturbation_schedule(
        schedule, schedule.features, schedule.K_adv, schedule.seed
    )
    payload = json.dumps(
        asdict(schedule), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_algorithm_b_perturbation_schedule(
    schedule: AlgorithmBPerturbationSchedule,
    features: Sequence[str],
    K_adv: int,
    seed: int,
) -> None:
    if not isinstance(schedule, AlgorithmBPerturbationSchedule):
        raise ValueError("Algorithm B perturbation schedule has an invalid type")
    if (
        type(schedule.features) is not tuple
        or not schedule.features
        or not all(isinstance(feature, str) for feature in schedule.features)
        or len(set(schedule.features)) != len(schedule.features)
        or isinstance(schedule.K_adv, bool)
        or not isinstance(schedule.K_adv, int)
        or schedule.K_adv < 0
        or isinstance(schedule.seed, bool)
        or not isinstance(schedule.seed, int)
        or schedule.seed < 0
        or type(schedule.seed_orders) is not tuple
        or type(schedule.trials) is not tuple
    ):
        raise ValueError("Algorithm B perturbation schedule must be deeply immutable")
    feature_order = tuple(features)
    if (
        not feature_order
        or not all(isinstance(feature, str) for feature in feature_order)
        or len(set(feature_order)) != len(feature_order)
        or isinstance(K_adv, bool)
        or not isinstance(K_adv, int)
        or K_adv < 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("Algorithm B perturbation schedule authority is invalid")
    if (
        schedule.features != feature_order
        or schedule.K_adv != K_adv
        or schedule.seed != seed
        or len(schedule.seed_orders) != len(feature_order)
        or len(schedule.trials) != len(feature_order)
    ):
        raise ValueError("Algorithm B perturbation schedule identity is invalid")
    for index, feature in enumerate(feature_order):
        others = tuple(candidate for candidate in feature_order if candidate != feature)
        seed_order = schedule.seed_orders[index]
        if (
            type(seed_order) is not tuple
            or not all(isinstance(candidate, str) for candidate in seed_order)
            or set(seed_order) != set(others)
            or len(seed_order) != len(others)
        ):
            raise ValueError(
                "Algorithm B perturbation schedule seed order must be an immutable tuple"
            )
        trials = schedule.trials[index]
        if type(trials) is not tuple or len(trials) != K_adv:
            raise ValueError(
                "Algorithm B perturbation schedule trials must be an immutable tuple"
            )
        for trial in trials:
            if (
                not isinstance(trial, AlgorithmBPerturbationTrial)
                or type(trial.removal_draws) is not tuple
                or len(trial.removal_draws) != len(feature_order)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value < 0
                    or value >= 1
                    for value in trial.removal_draws
                )
                or isinstance(trial.add_count, bool)
                or not isinstance(trial.add_count, int)
                or trial.add_count < 0
                or trial.add_count > min(5, len(others))
                or type(trial.add_order) is not tuple
                or not all(
                    isinstance(candidate, str) for candidate in trial.add_order
                )
                or len(trial.add_order) != len(others)
                or set(trial.add_order) != set(others)
            ):
                raise ValueError("Algorithm B perturbation schedule trial is invalid")


def make_shared_coalition_plan(
    features: Sequence[str],
    *,
    K: int,
    per_coalition: int,
    seed: int,
) -> SharedCoalitionPlan:
    """Build a utility-independent shared-coalition evaluation plan."""
    feature_order = tuple(features)
    if len(set(feature_order)) != len(feature_order):
        raise ValueError("features must be unique")
    if K < 0:
        raise ValueError("K must be non-negative")
    if per_coalition < 0:
        raise ValueError("per_coalition must be non-negative")

    rng = np.random.default_rng(seed)
    hits = {feature: 0 for feature in feature_order}
    steps: list[SharedCoalitionStep] = []

    for _ in range(K):
        coalition_size = int(rng.integers(0, len(feature_order) + 1))
        coalition = frozenset(
            rng.choice(feature_order, size=coalition_size, replace=False).tolist()
        )
        absent = [feature for feature in feature_order if feature not in coalition]
        rng.shuffle(absent)
        evaluated = tuple(absent[:per_coalition])
        steps.append((coalition, evaluated))
        for feature in evaluated:
            hits[feature] += 1

    for feature in feature_order:
        if hits[feature] >= 2:
            continue
        others = [candidate for candidate in feature_order if candidate != feature]
        for _ in range(5 - hits[feature]):
            coalition_size = int(rng.integers(0, len(others) + 1))
            coalition = frozenset(
                rng.choice(others, size=coalition_size, replace=False).tolist()
            )
            steps.append((coalition, (feature,)))
            hits[feature] += 1

    return tuple(steps)


def _validate_shared_coalition_plan(
    plan: SharedCoalitionPlan,
    features: Sequence[str],
) -> None:
    """Validate every plan step before any model or cache is used."""
    valid_features = set(features)
    for step_index, (coalition, evaluated) in enumerate(plan):
        unknown_coalition = set(coalition) - valid_features
        if unknown_coalition:
            raise ValueError(
                f"shared_plan step {step_index} has unknown coalition features: "
                f"{sorted(unknown_coalition)}"
            )
        unknown_evaluated = set(evaluated) - valid_features
        if unknown_evaluated:
            raise ValueError(
                f"shared_plan step {step_index} has unknown evaluated features: "
                f"{sorted(unknown_evaluated)}"
            )
        not_absent = set(evaluated) & set(coalition)
        if not_absent:
            raise ValueError(
                f"shared_plan step {step_index} evaluated features must be absent "
                f"from the coalition: {sorted(not_absent)}"
            )


# ---------------------------------------------------------------------------
# Exact enumeration
# ---------------------------------------------------------------------------


def estimate_minppi_exact(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    utility: Optional[str] = None,
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Exact MinPPI via full enumeration of 2^{n-1} coalitions per feature.

    Feasible only for n ≤ 20 (max ~1M evaluations for n=20).

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter.
        task: One of ``"binary"``, ``"multiclass"``, ``"regression"``.
        acc_cache: Optional pre-built cache.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` where values are in {-1, 0, +1}.
    """
    if acc_cache is None:
        acc_cache = make_acc_full_cache(
            X_train, X_val, y_train, y_val,
            task=task, seed=seed, utility=utility,
        )
    acc_cache.set_threshold(alpha)

    features = list(X_train.columns)
    n = len(features)
    if n > 20:
        raise ValueError(
            f"Exact enumeration requires n ≤ 20, got {n}. "
            "Use estimate_minppi with method='random_mc' or "
            "'mi_correlation_guided'."
        )

    if verbose:
        print(f"  Exact MinPPI: {n} features, enumerating 2^{n - 1} per feature")

    minppi: dict[str, int] = {}
    for i, feat in enumerate(features):
        others = [candidate for candidate in features if candidate != feat]
        best_marg = 1  # init to max possible
        for r in range(len(others) + 1):
            for S_combo in itertools.combinations(others, r):
                S = set(S_combo)
                marg = acc_cache.eval_marginal(S, feat)
                if marg < best_marg:
                    best_marg = marg
                    if best_marg == -1:
                        break
            if best_marg == -1:
                break
        minppi[feat] = best_marg
        if verbose and (i + 1) % 5 == 0:
            n_neg = sum(1 for v in minppi.values() if v == -1)
            n_pos = sum(1 for v in minppi.values() if v == +1)
            print(
                f"  Exact: {i + 1}/{n}  (harmful={n_neg}, decisive={n_pos})",
                flush=True,
            )

    return minppi


# ---------------------------------------------------------------------------
# Algorithm A — Random Monte Carlo (§4.3.1)
# ---------------------------------------------------------------------------


def estimate_minppi_random_mc(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    K: int = 200,
    shared_coalitions: bool = False,
    per_coalition: int = 5,
    shared_plan: Optional[SharedCoalitionPlan] = None,
    utility: Optional[str] = None,
    acc_cache: Optional[AccFullCache] = None,
    model_params: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Estimate MinPPI via random coalition sampling (Algorithm A).

    Two modes:

    **Per-feature mode** (``shared_coalitions=False``, default):
    For each feature *i*, samples *K* random coalitions S ⊆ N \\ {i} and
    computes  min_k [v_alpha(S_k ∪ {i}) - v_alpha(S_k)].

    **Shared-coalition mode** (``shared_coalitions=True``):
    Draws *K* coalitions once.  For each coalition S, up to *per_coalition*
    absent features are evaluated (same strategy as Banzhaf/MinShap), followed
    by fixed supplemental steps for under-sampled features. The complete plan
    is utility-independent and can be reused across metric views.

    **Upward bias** (Lemma 3):  sample_min ≥ population_min almost surely.

    **Harmful detection** (Proposition 1): If MinPPI_i = -1 and the harmful
    event has probability p_-^i, then K ≥ log(δ)/log(1 - p_-^i) samples are
    needed to detect it with probability ≥ 1 − δ.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter.
        K: Number of random coalitions (per-feature or shared).
        shared_coalitions: If True, use shared-coalition sampling.
        per_coalition: Max absent features to evaluate per coalition
            (shared-coalition mode only).
        shared_plan: Optional immutable evaluation plan for shared-coalition
            mode. Supplying one guarantees identical coalition-feature
            comparisons across utility views.
        acc_cache: Optional pre-built accuracy cache.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` (values ∈ {-1, 0, +1}).
    """
    features = list(X_train.columns)
    n = len(features)
    if shared_plan is not None and not shared_coalitions:
        raise ValueError("shared_plan requires shared_coalitions=True")
    if shared_coalitions:
        if shared_plan is None:
            shared_plan = make_shared_coalition_plan(
                features,
                K=K,
                per_coalition=per_coalition,
                seed=seed,
            )
        _validate_shared_coalition_plan(shared_plan, features)

    rng = np.random.default_rng(seed)
    if acc_cache is None:
        acc_cache = make_acc_full_cache(
            X_train, X_val, y_train, y_val,
            task=task, seed=seed, utility=utility, model_params=model_params,
        )
    acc_cache.set_threshold(alpha)

    minppi: dict[str, int] = {}

    if shared_coalitions:
        assert shared_plan is not None
        # ── fixed shared-coalition mode (Banzhaf-style) ──
        # Execute every pre-generated step.  Supplemental coverage steps
        # are included in the plan and cannot depend on utility outcomes.
        if verbose:
            print(
                f"  Algorithm A (shared): K={K}, per_cc={per_coalition}, "
                f"plan_steps={len(shared_plan)}, {n} features",
                flush=True,
            )

        minppi = {f: 1 for f in features}
        hits = {f: 0 for f in features}
        n_eval = 0

        for coalition, evaluated in shared_plan:
            S = set(coalition)
            vS = 1 if acc_cache.is_winning(S) else 0
            n_eval += 1

            for feat in evaluated:
                vSi = 1 if acc_cache.is_winning(S | {feat}) else 0
                n_eval += 1
                marg = vSi - vS
                if marg < minppi[feat]:
                    minppi[feat] = marg
                hits[feat] += 1

        minppi = {f: minppi[f] if hits[f] > 0 else 0 for f in features}
        if verbose:
            n_neg = sum(1 for v in minppi.values() if v == -1)
            print(
                f"  Algorithm A (shared): {n_eval} evals, harmful={n_neg}/{n}",
                flush=True,
            )
        return minppi

    # ── per-feature mode ──
    if verbose:
        print(f"  Algorithm A (random MC): K={K}, {n} features", flush=True)

    for idx, feat in enumerate(features):
        others = [candidate for candidate in features if candidate != feat]
        best_marg = 1
        n_others = len(others)

        # Adaptive K: for tiny feature sets, exact enumeration is cheaper
        actual_K = min(K, 2**n_others)

        for _ in range(actual_K):
            s_size = rng.integers(0, n_others + 1)
            S = set(rng.choice(others, size=s_size, replace=False).tolist())
            marg = acc_cache.eval_marginal(S, feat)
            if marg < best_marg:
                best_marg = marg
                if best_marg == -1:
                    break

        minppi[feat] = best_marg
        if verbose and (idx + 1) % max(1, n // 5) == 0:
            n_neg = sum(1 for v in minppi.values() if v == -1)
            print(
                f"  Algorithm A: {idx + 1}/{n}  (harmful found={n_neg})",
                flush=True,
            )

    return minppi


# ---------------------------------------------------------------------------
# Algorithm B — MI-Correlation-Guided Adversarial Search (§4.3.2)
# ---------------------------------------------------------------------------


def estimate_minppi_mi_correlation_guided(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    K_adv: int = 30,
    correlation_neighbors: Optional[dict[str, list[str]]] = None,
    mi_scores: Optional[dict[str, float]] = None,
    perturbation_schedule: Optional[AlgorithmBPerturbationSchedule] = None,
    utility: Optional[str] = None,
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Estimate MinPPI via MI-correlation-guided search (Algorithm B).

    For each feature *i*:

    1. Identifies correlated neighbours C from a weighted correlation tree.
    2. Constructs a seed coalition S_seed that includes C (the context where
       *i* is most likely to be redundant).
    3. Perturbs S_seed *K_adv* times, evaluating v_alpha margins.
    4. Exits early if a harmful marginal (-1) is found.

    **Sample efficiency**: When harmful events are sparse (small p_-^i),
    Algorithm B targets the vulnerable contexts directly rather than
    sampling uniformly.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter.
        K_adv: Adversarial search budget per feature.
        correlation_neighbors: Pre-computed correlation-neighbor dict
            ``{feature: [neighbor_features, ...]}`` sorted by correlation.
            If None, falls back to random MC (Algorithm A).
        mi_scores: Mutual-information scores ``{feature: value}``.
            Used to construct the seed coalition. If None, uses uniform.
        acc_cache: Optional pre-built accuracy cache.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` (values ∈ {-1, 0, +1}).
    """
    if acc_cache is None:
        acc_cache = make_acc_full_cache(
            X_train, X_val, y_train, y_val,
            task=task, seed=seed, utility=utility,
        )
    acc_cache.set_threshold(alpha)

    features = list(X_train.columns)
    n = len(features)
    if perturbation_schedule is None:
        perturbation_schedule = make_algorithm_b_perturbation_schedule(
            features, K_adv=K_adv, seed=seed
        )
    _validate_algorithm_b_perturbation_schedule(
        perturbation_schedule, features, K_adv, seed
    )
    minppi: dict[str, int] = {}

    # Build seed coalition: top features by MI score (or uniform)
    if mi_scores is not None:
        feature_rank = {feature: index for index, feature in enumerate(features)}
        top_order = sorted(
            ((feature, score) for feature, score in mi_scores.items() if feature in feature_rank),
            key=lambda item: (-item[1], feature_rank[item[0]]),
        )
        top_half = [f for f, _ in top_order[: max(1, n // 2)]]
    else:
        top_half = list(features)

    if verbose:
        n_neighbors = (
            sum(len(v) for v in correlation_neighbors.values())
            if correlation_neighbors else 0
        )
        print(
            f"  Algorithm B (MI-correlation-guided): K_adv={K_adv}, "
            f"{n} features, "
            f"neighbors={n_neighbors}",
            flush=True,
        )

    for idx, feat in enumerate(features):
        best_marg = 1

        # Build seed context: include correlated neighbors when available
        if correlation_neighbors and feat in correlation_neighbors:
            correlated = [
                candidate
                for candidate in correlation_neighbors.get(feat, [])
                if candidate in features and candidate != feat
            ]
            # Build S_seed: features correlated with feat + top-scored features
            seed_core = set(correlated[: min(10, len(correlated))])
            # Pad to a reasonable coalition size with top features
            seed = seed_core.copy()
            for f in top_half:
                if len(seed) >= min(20, n // 3):
                    break
                if f not in seed and f != feat:
                    seed.add(f)
        else:
            # Fallback: random baseline coalition biased toward top features
            seed_size = min(20, n // 3)
            # Prefer top-ranked features
            top_pool = set(top_half[: max(seed_size * 2, len(top_half))])
            seed = set(
                candidate
                for candidate in perturbation_schedule.seed_orders[idx]
                if candidate in top_pool
            )
            seed = set(list(candidate for candidate in perturbation_schedule.seed_orders[idx] if candidate in seed)[:seed_size])

        # Adversarial search
        for trial in perturbation_schedule.trials[idx]:
            # Perturb seed: randomly flip features, biased toward keeping
            # correlated neighbors
            S = seed.copy()
            for f, removal_draw in zip(features, trial.removal_draws, strict=True):
                if f in S and removal_draw < 0.3:
                    S.discard(f)
            # Add some features from outside
            outside = [
                candidate
                for candidate in trial.add_order
                if candidate not in S and candidate != feat
            ]
            if outside:
                S.update(outside[: min(trial.add_count, len(outside))])

            marg = acc_cache.eval_marginal(S, feat)
            if marg < best_marg:
                best_marg = marg
                if best_marg == -1:
                    break

        minppi[feat] = best_marg
        if verbose and (idx + 1) % max(1, n // 5) == 0:
            n_neg = sum(1 for v in minppi.values() if v == -1)
            print(
                f"  Algorithm B: {idx + 1}/{n}  (harmful found={n_neg})",
                flush=True,
            )

    return minppi


def estimate_minppi_mi_guided(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    K_adv: int = 30,
    shapg_neighbors: Optional[dict[str, list[str]]] = None,
    shapg_scores: Optional[dict[str, float]] = None,
    utility: Optional[str] = None,
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Deprecated wrapper for MI-correlation-guided Algorithm B."""
    warnings.warn(
        "estimate_minppi_mi_guided is deprecated; use "
        "estimate_minppi_mi_correlation_guided",
        DeprecationWarning,
        stacklevel=2,
    )
    return estimate_minppi_mi_correlation_guided(
        X_train,
        X_val,
        y_train,
        y_val,
        alpha=alpha,
        task=task,
        K_adv=K_adv,
        correlation_neighbors=shapg_neighbors,
        mi_scores=shapg_scores,
        utility=utility,
        acc_cache=acc_cache,
        seed=seed,
        verbose=verbose,
    )


def _resolve_guidance_aliases(
    correlation_neighbors: Optional[dict[str, list[str]]],
    mi_scores: Optional[dict[str, float]],
    shapg_neighbors: Optional[dict[str, list[str]]],
    shapg_scores: Optional[dict[str, float]],
) -> tuple[Optional[dict[str, list[str]]], Optional[dict[str, float]]]:
    """Resolve deprecated ShapG-labelled parameters into canonical names."""
    if correlation_neighbors is not None and shapg_neighbors is not None:
        raise ValueError(
            "Do not provide both correlation_neighbors and shapg_neighbors"
        )
    if mi_scores is not None and shapg_scores is not None:
        raise ValueError("Do not provide both mi_scores and shapg_scores")
    if shapg_neighbors is not None or shapg_scores is not None:
        warnings.warn(
            "shapg_neighbors/shapg_scores are deprecated; use "
            "correlation_neighbors/mi_scores",
            DeprecationWarning,
            stacklevel=3,
        )
    return (
        correlation_neighbors
        if correlation_neighbors is not None else shapg_neighbors,
        mi_scores if mi_scores is not None else shapg_scores,
    )


# ---------------------------------------------------------------------------
# Hybrid dispatcher (§4.3.3)
# ---------------------------------------------------------------------------


def estimate_minppi_hybrid(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    K: int = 200,
    K_adv: int = 30,
    shared_coalitions: bool = False,
    correlation_neighbors: Optional[dict[str, list[str]]] = None,
    mi_scores: Optional[dict[str, float]] = None,
    perturbation_schedule: Optional[AlgorithmBPerturbationSchedule] = None,
    shapg_neighbors: Optional[dict[str, list[str]]] = None,
    shapg_scores: Optional[dict[str, float]] = None,
    utility: Optional[str] = None,
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Hybrid MinPPI estimation (§4.3.3).

    Dispatches based on feature count *n*:

    - n ≤ 20: exact enumeration
    - 20 < n ≤ 75: Algorithm B (MI-correlation-guided), fallback to Algorithm A
    - n > 75: Algorithm A + Algorithm B, take the min of both estimates

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter.
        K: Random MC budget (Algorithm A).
        K_adv: Adversarial search budget (Algorithm B).
        shared_coalitions: Forwarded to Algorithm A.
        correlation_neighbors: Correlation-neighbor dict for Algorithm B.
        mi_scores: MI scores for seed construction.
        shapg_neighbors: Deprecated alias for ``correlation_neighbors``.
        shapg_scores: Deprecated alias for ``mi_scores``.
        acc_cache: Optional pre-built accuracy cache.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` (values ∈ {-1, 0, +1}).
    """
    correlation_neighbors, mi_scores = _resolve_guidance_aliases(
        correlation_neighbors,
        mi_scores,
        shapg_neighbors,
        shapg_scores,
    )
    if acc_cache is None:
        acc_cache = make_acc_full_cache(
            X_train, X_val, y_train, y_val, task=task, seed=seed, utility=utility,
        )

    n = len(X_train.columns)

    if n <= 20:
        if verbose:
            print(f"  Hybrid: n={n} ≤ 20 → exact enumeration", flush=True)
        return estimate_minppi_exact(
            X_train, X_val, y_train, y_val,
            alpha=alpha, task=task, utility=utility, acc_cache=acc_cache,
            seed=seed, verbose=verbose,
        )

    if n <= 75:
        if verbose:
            print(f"  Hybrid: 20 < n={n} ≤ 75 → Algorithm B", flush=True)
        if correlation_neighbors is None:
            if verbose:
                print(
                    "  No correlation neighbors available → fallback to Algorithm A",
                    flush=True,
                )
            return estimate_minppi_random_mc(
                X_train, X_val, y_train, y_val,
                alpha=alpha, task=task, utility=utility, K=K,
                shared_coalitions=shared_coalitions, acc_cache=acc_cache,
                seed=seed, verbose=verbose,
            )
        return estimate_minppi_mi_correlation_guided(
            X_train, X_val, y_train, y_val,
            alpha=alpha, task=task, utility=utility, K_adv=K_adv,
            correlation_neighbors=correlation_neighbors, mi_scores=mi_scores,
            perturbation_schedule=perturbation_schedule,
            acc_cache=acc_cache, seed=seed, verbose=verbose,
        )

    # n > 75: run both and take element-wise min
    if verbose:
        print(f"  Hybrid: n={n} > 75 → Algorithm A + B combined", flush=True)

    minppi_a = estimate_minppi_random_mc(
        X_train, X_val, y_train, y_val,
        alpha=alpha, task=task, utility=utility, K=K,
        shared_coalitions=shared_coalitions, acc_cache=acc_cache,
        seed=seed, verbose=False,
    )

    minppi_b: dict[str, int] = {}
    if correlation_neighbors is not None:
        minppi_b = estimate_minppi_mi_correlation_guided(
            X_train, X_val, y_train, y_val,
            alpha=alpha, task=task, utility=utility, K_adv=K_adv,
            correlation_neighbors=correlation_neighbors, mi_scores=mi_scores,
            perturbation_schedule=perturbation_schedule,
            acc_cache=acc_cache, seed=seed, verbose=False,
        )

    combined: dict[str, int] = {}
    for feat in minppi_a:
        val_a = minppi_a[feat]
        val_b = minppi_b.get(feat, val_a)
        combined[feat] = min(val_a, val_b)

    if verbose:
        n_neg = sum(1 for v in combined.values() if v == -1)
        print(f"  Combined: {n_neg} harmful features found", flush=True)

    return combined


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def estimate_minppi(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    method: str = "hybrid",
    K: int = 200,
    K_adv: int = 30,
    shared_coalitions: bool = False,
    correlation_neighbors: Optional[dict[str, list[str]]] = None,
    mi_scores: Optional[dict[str, float]] = None,
    perturbation_schedule: Optional[AlgorithmBPerturbationSchedule] = None,
    shapg_neighbors: Optional[dict[str, list[str]]] = None,
    shapg_scores: Optional[dict[str, float]] = None,
    utility: Optional[str] = None,
    model_params: Optional[dict] = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, int]:
    """Estimate MinPPI for all features.

    This is the main entry point for MinPPI computation. Dispatches to one
    of four strategies based on *method*.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter α ∈ (0, 1].
        method: Estimation strategy:

            - ``"exact"`` — enumerate all coalitions (n ≤ 20 only)
            - ``"random_mc"`` — Algorithm A, random MC sampling
            - ``"mi_correlation_guided"`` — Algorithm B,
              MI-correlation-guided adversarial search
            - ``"hybrid"`` — auto-select based on *n* (§4.3.3, default)
        K: Number of random samples per feature (Algorithm A).
        K_adv: Adversarial search budget per feature (Algorithm B).
        shared_coalitions: Forwarded to Algorithm A (shared-coalition mode).
        correlation_neighbors: Correlation-neighbor dict for Algorithm B.
        mi_scores: MI scores for seed coalition construction.
        shapg_neighbors: Deprecated alias for ``correlation_neighbors``.
        shapg_scores: Deprecated alias for ``mi_scores``.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: minppi}`` where each value ∈ {-1, 0, +1},
        corresponding to the three-way classification of Theorem 1:
        ``+1`` = decisive, ``0`` = neutral, ``-1`` = harmful.

    Examples:
        >>> minppi = estimate_minppi(X_tr, X_val, y_tr, y_val, alpha=0.95)
        >>> harmful = {f for f, v in minppi.items() if v == -1}
        >>> decisive = {f for f, v in minppi.items() if v == +1}
    """
    if method == "mi_guided":
        warnings.warn(
            "method='mi_guided' is deprecated; use "
            "method='mi_correlation_guided'",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "mi_correlation_guided"

    methods = {
        "exact": estimate_minppi_exact,
        "random_mc": estimate_minppi_random_mc,
        "mi_correlation_guided": estimate_minppi_mi_correlation_guided,
        "hybrid": estimate_minppi_hybrid,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {sorted(methods.keys())}."
        )

    correlation_neighbors, mi_scores = _resolve_guidance_aliases(
        correlation_neighbors,
        mi_scores,
        shapg_neighbors,
        shapg_scores,
    )

    # Build acc_cache once and reuse
    acc_cache = make_acc_full_cache(
        X_train, X_val, y_train, y_val,
        task=task, seed=seed, utility=utility, model_params=model_params,
    )

    fn = methods[method]
    common: dict = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "alpha": alpha,
        "task": task,
        "utility": utility,
        "acc_cache": acc_cache,
        "seed": seed,
        "verbose": verbose,
    }

    if method == "exact":
        return fn(**common)  # type: ignore[call-arg]
    if method == "random_mc":
        return fn(K=K, shared_coalitions=shared_coalitions, **common)  # type: ignore[call-arg]
    if method == "mi_correlation_guided":
        return fn(
            K_adv=K_adv,
            correlation_neighbors=correlation_neighbors,
            mi_scores=mi_scores,
            perturbation_schedule=perturbation_schedule,
            **common,
        )  # type: ignore[call-arg]
    return fn(
        K=K,
        K_adv=K_adv,
        shared_coalitions=shared_coalitions,
        correlation_neighbors=correlation_neighbors,
        mi_scores=mi_scores,
        perturbation_schedule=perturbation_schedule,
        **common,
    )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Utility: MinPPI statistics
# ---------------------------------------------------------------------------


def minppi_stats(minppi: dict[str, int]) -> dict:
    """Compute summary statistics of a MinPPI result.

    Args:
        minppi: Dict ``{feature: value}`` with values in {-1, 0, +1}.

    Returns:
        Dict with ``n_total``, ``n_decisive``, ``n_neutral``, ``n_harmful``,
        ``pct_decisive``, ``pct_neutral``, ``pct_harmful``.
    """
    vals = list(minppi.values())
    n = len(vals)
    n_pos = sum(1 for v in vals if v == +1)
    n_zero = sum(1 for v in vals if v == 0)
    n_neg = sum(1 for v in vals if v == -1)
    return {
        "n_total": n,
        "n_decisive": n_pos,
        "n_neutral": n_zero,
        "n_harmful": n_neg,
        "pct_decisive": n_pos / n * 100 if n else 0,
        "pct_neutral": n_zero / n * 100 if n else 0,
        "pct_harmful": n_neg / n * 100 if n else 0,
    }


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("MinPPI estimation — synthetic demo")

    rng = np.random.default_rng(42)
    n_samp = 300
    X_demo = pd.DataFrame(
        {
            "f_signal": rng.normal(0, 1, n_samp),
            "f_harmful": rng.normal(0, 1, n_samp),
            "f_noise": rng.normal(0, 1, n_samp),
        }
    )
    # f_signal is predictive; f_harmful adds pure noise; f_noise is neutral
    y_demo = np.where(X_demo["f_signal"] > 0, 1, 0).astype(int)

    n_tr = int(0.7 * n_samp)
    X_tr, X_val = X_demo.iloc[:n_tr], X_demo.iloc[n_tr:]
    y_tr, y_val = y_demo[:n_tr], y_demo[n_tr:]

    # Exact enumeration (n=3 → only 2^{2}=4 coalitions per feature)
    minppi = estimate_minppi(
        X_tr, X_val, y_tr, y_val, alpha=0.90, method="exact", verbose=True
    )
    print("\nMinPPI results:")
    for feat, val in sorted(minppi.items(), key=lambda kv: kv[1]):
        label = {+1: "DECISIVE", 0: "neutral", -1: "HARMFUL"}[val]
        print(f"  {feat:20s} → {val:+d}  ({label})")

    stats = minppi_stats(minppi)
    print(
        f"\nSummary: {stats['n_total']} features — "
        f"{stats['n_decisive']} decisive, {stats['n_harmful']} harmful, "
        f"{stats['n_neutral']} neutral"
    )
    print(
        "\nObservation: Lemma 1 (boundedness) holds — all values ∈ {-1, 0, +1}. "
        "The harmful feature (pure noise added) should show MinPPI=-1 when "
        "α is high enough to detect the performance degradation."
    )

    # Compare with Algorithm A on the same data
    _demo_banner("Algorithm A comparison")
    minppi_mc = estimate_minppi(
        X_tr,
        X_val,
        y_tr,
        y_val,
        alpha=0.90,
        method="random_mc",
        K=50,
        verbose=True,
    )
    for feat in sorted(minppi_mc):
        ex = minppi[feat]
        mc = minppi_mc[feat]
        agree = "✓" if ex == mc else "✗ (upward bias)"
        print(f"  {feat:20s}  exact={ex:+d}  MC={mc:+d}  {agree}")
    print("\nObservation: Algorithm A shows upward bias (sample min >= true min).")
