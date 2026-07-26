"""Nested deterministic schedules for random and guided MinPPI checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from typing import Protocol

import numpy as np

from minppi_mofs.min_ppi import AlgorithmBPerturbationTrial


NESTED_SCHEDULE_SCHEMA_VERSION = 1
_RANDOM_ALGORITHM_ID = 1
GUIDED_ALGORITHM_ID = 2


class MarginalUtilityView(Protocol):
    """Minimal utility-view interface required by checkpoint evaluation."""

    @property
    def feature_order(self) -> Sequence[str]: ...

    def eval_marginal(self, coalition: set[str], feature: str) -> int: ...


@dataclass(frozen=True)
class NestedRandomSchedule:
    features: tuple[str, ...]
    seed: int
    checkpoints: tuple[int, ...]
    trials: tuple[tuple[frozenset[str], ...], ...]

    def __post_init__(self) -> None:
        _validate_random_schedule(self)


@dataclass(frozen=True)
class NestedGuidedSchedule:
    features: tuple[str, ...]
    seed: int
    checkpoints: tuple[int, ...]
    seed_orders: tuple[tuple[str, ...], ...]
    trials: tuple[tuple[AlgorithmBPerturbationTrial, ...], ...]

    def __post_init__(self) -> None:
        _validate_guided_schedule(self)

    @property
    def max_budget(self) -> int:
        return self.checkpoints[-1]


@dataclass(frozen=True)
class NestedProgress:
    algorithm: str
    feature_index: int
    feature_count: int
    budget: int
    max_budget: int


def _validate_seed(seed: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _validate_features(features: object) -> tuple[str, ...]:
    if not isinstance(features, tuple):
        raise ValueError("features must be an immutable tuple")
    if (
        not features
        or not all(isinstance(feature, str) and feature for feature in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("features must be non-empty unique strings")
    return features


def _validate_checkpoints(checkpoints: object) -> tuple[int, ...]:
    if not isinstance(checkpoints, tuple):
        raise ValueError("checkpoints must be an immutable tuple")
    if not checkpoints:
        raise ValueError("at least one checkpoint budget is required")
    if any(
        isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0
        for budget in checkpoints
    ):
        raise ValueError("checkpoint budgets must be positive integers")
    if any(
        current >= following
        for current, following in zip(checkpoints, checkpoints[1:])
    ):
        raise ValueError("checkpoint budgets must be strictly increasing and unique")
    return checkpoints


def _validated_budgets(checkpoints: Sequence[int]) -> tuple[int, ...]:
    if isinstance(checkpoints, (str, bytes)) or not isinstance(
        checkpoints, Sequence
    ):
        raise ValueError("checkpoints must be a non-string sequence")
    budgets = tuple(checkpoints)
    return _validate_checkpoints(budgets)


def _validated_scores(scores: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(scores, Mapping):
        raise ValueError("scores must be a feature-to-score mapping")
    if (
        not scores
        or not all(isinstance(feature, str) and feature for feature in scores)
    ):
        raise ValueError("score features must be non-empty strings")
    if any(
        isinstance(score, bool)
        or not isinstance(score, int)
        or score not in {-1, 0, 1}
        for score in scores.values()
    ):
        raise ValueError("scores must be integers in {-1, 0, 1}")
    return dict(scores)


def _validated_checkpoint_scores(
    *,
    component: str,
    budgets: tuple[int, ...],
    scores: Mapping[int, Mapping[str, int]] | None,
) -> dict[int, dict[str, int]]:
    if scores is None or not isinstance(scores, Mapping):
        raise ValueError(f"hybrid {component} checkpoints are incomplete")
    if any(
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget <= 0
        for budget in scores
    ):
        raise ValueError(f"hybrid {component} checkpoint keys are invalid")
    if set(scores) != set(budgets):
        raise ValueError(f"hybrid {component} checkpoints are incomplete")

    validated = {
        budget: _validated_scores(scores[budget]) for budget in budgets
    }
    first_feature_order = tuple(validated[budgets[0]])
    if any(
        tuple(validated[budget]) != first_feature_order
        for budget in budgets[1:]
    ):
        raise ValueError(
            f"hybrid {component} checkpoint feature orders conflict"
        )
    return validated


def _validate_random_schedule(schedule: NestedRandomSchedule) -> None:
    features = _validate_features(schedule.features)
    checkpoints = _validate_checkpoints(schedule.checkpoints)
    _validate_seed(schedule.seed)
    if not isinstance(schedule.trials, tuple):
        raise ValueError("schedule trials must be an immutable tuple")
    if len(schedule.trials) != len(features):
        raise ValueError("schedule trials must match the feature order")

    trial_count = min(checkpoints[-1], 2 ** (len(features) - 1))
    feature_set = frozenset(features)
    for feature_index, feature_trials in enumerate(schedule.trials):
        if not isinstance(feature_trials, tuple):
            raise ValueError("per-feature schedule trials must be immutable tuples")
        if len(feature_trials) != trial_count:
            raise ValueError("per-feature schedule trial count is invalid")
        allowed = feature_set - {features[feature_index]}
        if any(
            not isinstance(coalition, frozenset) or not coalition <= allowed
            for coalition in feature_trials
        ):
            raise ValueError("schedule coalition is invalid for its feature")


def _validate_guided_schedule(schedule: NestedGuidedSchedule) -> None:
    features = _validate_features(schedule.features)
    checkpoints = _validate_checkpoints(schedule.checkpoints)
    _validate_seed(schedule.seed)
    if (
        not isinstance(schedule.seed_orders, tuple)
        or not isinstance(schedule.trials, tuple)
    ):
        raise ValueError("guided schedule state must use immutable tuples")
    if (
        len(schedule.seed_orders) != len(features)
        or len(schedule.trials) != len(features)
    ):
        raise ValueError("guided schedule state must match the feature order")

    for feature_index, feature in enumerate(features):
        others = tuple(candidate for candidate in features if candidate != feature)
        seed_order = schedule.seed_orders[feature_index]
        if (
            not isinstance(seed_order, tuple)
            or not all(isinstance(candidate, str) for candidate in seed_order)
            or len(seed_order) != len(others)
            or set(seed_order) != set(others)
        ):
            raise ValueError(
                "guided schedule seed order must be an immutable feature permutation"
            )
        feature_trials = schedule.trials[feature_index]
        if (
            not isinstance(feature_trials, tuple)
            or len(feature_trials) != checkpoints[-1]
        ):
            raise ValueError(
                "guided schedule trials must be immutable max-budget tuples"
            )
        for trial in feature_trials:
            if (
                not isinstance(trial, AlgorithmBPerturbationTrial)
                or not isinstance(trial.removal_draws, tuple)
                or len(trial.removal_draws) != len(features)
                or any(
                    isinstance(draw, bool)
                    or not isinstance(draw, Real)
                    or not np.isfinite(draw)
                    or draw < 0
                    or draw >= 1
                    for draw in trial.removal_draws
                )
                or isinstance(trial.add_count, bool)
                or not isinstance(trial.add_count, int)
                or trial.add_count < 0
                or trial.add_count > min(5, len(others))
                or not isinstance(trial.add_order, tuple)
                or not all(
                    isinstance(candidate, str) for candidate in trial.add_order
                )
                or len(trial.add_order) != len(others)
                or set(trial.add_order) != set(others)
            ):
                raise ValueError("guided schedule perturbation trial is invalid")


def _feature_rng(seed: int, algorithm_id: int, feature_index: int):
    return np.random.default_rng(
        np.random.SeedSequence(
            [seed, algorithm_id, feature_index, NESTED_SCHEDULE_SCHEMA_VERSION]
        )
    )


def make_nested_random_schedule(
    features: Sequence[str],
    budgets: Sequence[int],
    seed: int,
) -> NestedRandomSchedule:
    """Generate one deterministic, prefix-nested trial stream per feature."""
    if isinstance(features, (str, bytes)):
        raise ValueError("features must be a sequence of feature names")
    if isinstance(budgets, (str, bytes)):
        raise ValueError("budgets must be a sequence of positive integers")
    feature_order = tuple(features)
    checkpoints = tuple(budgets)
    _validate_features(feature_order)
    _validate_checkpoints(checkpoints)
    _validate_seed(seed)

    trial_count = min(checkpoints[-1], 2 ** (len(feature_order) - 1))
    all_trials: list[tuple[frozenset[str], ...]] = []
    for feature_index, feature in enumerate(feature_order):
        others = tuple(
            candidate for candidate in feature_order if candidate != feature
        )
        rng = _feature_rng(seed, _RANDOM_ALGORITHM_ID, feature_index)
        feature_trials: list[frozenset[str]] = []
        for _ in range(trial_count):
            coalition_size = int(rng.integers(0, len(others) + 1))
            coalition = frozenset(
                rng.choice(others, size=coalition_size, replace=False).tolist()
            )
            feature_trials.append(coalition)
        all_trials.append(tuple(feature_trials))

    return NestedRandomSchedule(
        features=feature_order,
        seed=seed,
        checkpoints=checkpoints,
        trials=tuple(all_trials),
    )


def make_nested_guided_schedule(
    features: Sequence[str],
    budgets: Sequence[int],
    seed: int,
) -> NestedGuidedSchedule:
    """Generate one immutable guided master schedule per feature."""
    if isinstance(features, (str, bytes)):
        raise ValueError("features must be a sequence of feature names")
    if isinstance(budgets, (str, bytes)):
        raise ValueError("budgets must be a sequence of positive integers")
    feature_order = tuple(features)
    checkpoints = tuple(budgets)
    _validate_features(feature_order)
    _validate_checkpoints(checkpoints)
    _validate_seed(seed)

    seed_orders: list[tuple[str, ...]] = []
    all_trials: list[tuple[AlgorithmBPerturbationTrial, ...]] = []
    for feature_index, feature in enumerate(feature_order):
        others = tuple(
            candidate for candidate in feature_order if candidate != feature
        )
        rng = _feature_rng(seed, GUIDED_ALGORITHM_ID, feature_index)
        seed_orders.append(tuple(rng.permutation(others).tolist()))
        feature_trials = tuple(
            AlgorithmBPerturbationTrial(
                removal_draws=tuple(
                    float(value) for value in rng.random(len(feature_order))
                ),
                add_count=int(rng.integers(0, min(5, len(others)) + 1)),
                add_order=tuple(rng.permutation(others).tolist()),
            )
            for _ in range(checkpoints[-1])
        )
        all_trials.append(feature_trials)

    return NestedGuidedSchedule(
        features=feature_order,
        seed=seed,
        checkpoints=checkpoints,
        seed_orders=tuple(seed_orders),
        trials=tuple(all_trials),
    )


def nested_random_schedule_digest(schedule: NestedRandomSchedule) -> str:
    """Return a canonical digest binding all schedule-defining state."""
    _validate_random_schedule(schedule)
    feature_positions = {
        feature: position for position, feature in enumerate(schedule.features)
    }
    payload = {
        "schema_version": NESTED_SCHEDULE_SCHEMA_VERSION,
        "features": list(schedule.features),
        "seed": schedule.seed,
        "checkpoints": list(schedule.checkpoints),
        "trials": [
            [
                sorted(coalition, key=feature_positions.__getitem__)
                for coalition in feature_trials
            ]
            for feature_trials in schedule.trials
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def nested_guided_schedule_digest(schedule: NestedGuidedSchedule) -> str:
    """Return a canonical digest binding every guided schedule input."""
    if not isinstance(schedule, NestedGuidedSchedule):
        raise TypeError("guided schedule must be NestedGuidedSchedule")
    _validate_guided_schedule(schedule)
    payload = {
        "schema_version": NESTED_SCHEDULE_SCHEMA_VERSION,
        "algorithm_id": GUIDED_ALGORITHM_ID,
        "features": list(schedule.features),
        "seed": schedule.seed,
        "checkpoints": list(schedule.checkpoints),
        "seed_orders": [list(order) for order in schedule.seed_orders],
        "trials": [
            [
                {
                    "removal_draws": list(trial.removal_draws),
                    "add_count": trial.add_count,
                    "add_order": list(trial.add_order),
                }
                for trial in feature_trials
            ]
            for feature_trials in schedule.trials
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _view_feature_order(view: MarginalUtilityView) -> tuple[str, ...]:
    if hasattr(view, "feature_order"):
        value = view.feature_order
    else:
        raise ValueError("view must expose its feature order")
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError("view feature order is invalid") from error


def _validated_guidance(
    feature_order: tuple[str, ...],
    mi_scores: Mapping[str, float] | None,
    correlation_neighbors: Mapping[str, Sequence[str]] | None,
) -> tuple[dict[str, float] | None, dict[str, tuple[str, ...]] | None]:
    if mi_scores is None:
        validated_scores = None
    else:
        if not isinstance(mi_scores, Mapping):
            raise ValueError("MI scores must be a feature-to-score mapping")
        if any(feature not in mi_scores for feature in feature_order):
            raise ValueError("MI scores must include every schedule feature")
        scheduled_scores = {
            feature: mi_scores[feature] for feature in feature_order
        }
        if any(
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not np.isfinite(score)
            for score in scheduled_scores.values()
        ):
            raise ValueError("MI scores must be finite numeric values")
        validated_scores = scheduled_scores

    if correlation_neighbors is None:
        validated_neighbors = None
    else:
        if not isinstance(correlation_neighbors, Mapping):
            raise ValueError(
                "correlation neighbors must be a feature-to-sequence mapping"
            )
        validated_neighbors = {}
        for feature in feature_order:
            if feature not in correlation_neighbors:
                continue
            neighbors = correlation_neighbors[feature]
            if isinstance(neighbors, (str, bytes)) or not isinstance(
                neighbors, Sequence
            ):
                raise ValueError(
                    "correlation neighbors must use non-string sequences"
                )
            if not all(isinstance(item, str) for item in neighbors):
                raise ValueError(
                    "correlation neighbor sequences must contain strings"
                )
            validated_neighbors[feature] = tuple(neighbors)
    return validated_scores, validated_neighbors


def _guided_seed_coalition_unchecked(
    feature: str,
    feature_order: tuple[str, ...],
    seed_order: tuple[str, ...],
    mi_scores: Mapping[str, float] | None,
    correlation_neighbors: Mapping[str, Sequence[str]] | None,
) -> frozenset[str]:
    others = tuple(item for item in feature_order if item != feature)
    if mi_scores is None:
        top_half = feature_order
    else:
        rank = {item: index for index, item in enumerate(feature_order)}
        top_half = tuple(
            item
            for item, _ in sorted(
                ((item, mi_scores[item]) for item in feature_order),
                key=lambda pair: (-pair[1], rank[pair[0]]),
            )[: max(1, len(feature_order) // 2)]
        )
    correlated = (
        tuple(
            item
            for item in correlation_neighbors.get(feature, ())
            if item in others
        )
        if correlation_neighbors is not None
        else ()
    )
    if correlated:
        selected = set(correlated[:10])
        for item in top_half:
            if len(selected) >= min(20, len(feature_order) // 3):
                break
            if item != feature:
                selected.add(item)
        return frozenset(selected)
    seed_size = min(20, len(feature_order) // 3)
    top_pool = set(top_half)
    candidates = [
        item for item in seed_order if item in top_pool and item in others
    ]
    return frozenset(candidates[:seed_size])


def guided_seed_coalition(
    feature: str,
    feature_order: tuple[str, ...],
    seed_order: tuple[str, ...],
    *,
    mi_scores: Mapping[str, float] | None,
    correlation_neighbors: Mapping[str, Sequence[str]] | None,
) -> frozenset[str]:
    """Build a deterministic Algorithm B seed coalition without side effects."""
    features = _validate_features(feature_order)
    if feature not in features:
        raise ValueError("guided seed feature must belong to the feature order")
    others = tuple(item for item in features if item != feature)
    if (
        not isinstance(seed_order, tuple)
        or len(seed_order) != len(others)
        or set(seed_order) != set(others)
    ):
        raise ValueError(
            "guided seed order must be an immutable permutation of other features"
        )
    validated_scores, validated_neighbors = _validated_guidance(
        features, mi_scores, correlation_neighbors
    )
    return _guided_seed_coalition_unchecked(
        feature,
        features,
        seed_order,
        validated_scores,
        validated_neighbors,
    )


def estimate_guided_checkpoints(
    schedule: NestedGuidedSchedule,
    view: MarginalUtilityView,
    *,
    mi_scores: Mapping[str, float] | None,
    correlation_neighbors: Mapping[str, Sequence[str]] | None,
    progress: Callable[[NestedProgress], None] | None = None,
) -> dict[int, dict[str, int]]:
    """Evaluate guided checkpoint prefixes while advancing each cursor once."""
    if not isinstance(schedule, NestedGuidedSchedule):
        raise TypeError("schedule must be NestedGuidedSchedule")
    _validate_guided_schedule(schedule)
    if _view_feature_order(view) != schedule.features:
        raise ValueError("view feature order does not match the schedule")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    validated_scores, validated_neighbors = _validated_guidance(
        schedule.features, mi_scores, correlation_neighbors
    )

    scores_at: dict[int, dict[str, int]] = {
        budget: {} for budget in schedule.checkpoints
    }
    for feature_index, feature in enumerate(schedule.features):
        seed_coalition = _guided_seed_coalition_unchecked(
            feature,
            schedule.features,
            schedule.seed_orders[feature_index],
            validated_scores,
            validated_neighbors,
        )
        best = 1
        trials = schedule.trials[feature_index]
        cursor = 0
        for budget in schedule.checkpoints:
            while cursor < min(budget, len(trials)) and best != -1:
                trial = trials[cursor]
                coalition = set(seed_coalition)
                for candidate, draw in zip(
                    schedule.features, trial.removal_draws, strict=True
                ):
                    if candidate in coalition and draw < 0.3:
                        coalition.discard(candidate)
                outside = [
                    item
                    for item in trial.add_order
                    if item not in coalition and item != feature
                ]
                coalition.update(outside[: min(trial.add_count, len(outside))])
                best = min(best, view.eval_marginal(coalition, feature))
                cursor += 1
            scores_at[budget][feature] = best
            if progress is not None:
                progress(
                    NestedProgress(
                        algorithm="guided",
                        feature_index=feature_index + 1,
                        feature_count=len(schedule.features),
                        budget=budget,
                        max_budget=schedule.checkpoints[-1],
                    )
                )
    return scores_at


def estimate_random_checkpoints(
    schedule: NestedRandomSchedule,
    view: MarginalUtilityView,
    *,
    progress: Callable[[NestedProgress], None] | None = None,
) -> dict[int, dict[str, int]]:
    """Evaluate all nested checkpoints while advancing each trial cursor once."""
    _validate_random_schedule(schedule)
    if _view_feature_order(view) != schedule.features:
        raise ValueError("view feature order does not match the schedule")

    checkpoints = schedule.checkpoints
    scores_at: dict[int, dict[str, int]] = {
        budget: {} for budget in checkpoints
    }
    for feature_index, feature in enumerate(schedule.features):
        best = 1
        trials = schedule.trials[feature_index]
        cursor = 0
        for budget in checkpoints:
            target = min(budget, len(trials))
            while cursor < target and best != -1:
                best = min(best, view.eval_marginal(set(trials[cursor]), feature))
                cursor += 1
            scores_at[budget][feature] = best
            if progress is not None:
                progress(
                    NestedProgress(
                        algorithm="random",
                        feature_index=feature_index + 1,
                        feature_count=len(schedule.features),
                        budget=budget,
                        max_budget=checkpoints[-1],
                    )
                )
    return scores_at


def materialize_exact_checkpoints(
    checkpoints: Sequence[int],
    exact_scores: Mapping[str, int],
) -> dict[int, dict[str, int]]:
    """Copy one exact score map into independent checkpoint snapshots."""
    validated = _validated_scores(exact_scores)
    return {
        budget: dict(validated) for budget in _validated_budgets(checkpoints)
    }


def derive_hybrid_checkpoints(
    *,
    feature_count: int,
    checkpoints: Sequence[int],
    exact_scores: Mapping[str, int] | None,
    random_scores: Mapping[int, Mapping[str, int]] | None,
    guided_scores: Mapping[int, Mapping[str, int]] | None,
) -> dict[int, dict[str, int]]:
    """Derive dispatcher-equivalent hybrid scores without new evaluations."""
    if (
        isinstance(feature_count, bool)
        or not isinstance(feature_count, int)
        or feature_count <= 0
    ):
        raise ValueError("feature count must be a positive integer")
    budgets = _validated_budgets(checkpoints)

    if feature_count <= 20:
        if exact_scores is None:
            raise ValueError("hybrid exact component is required")
        return materialize_exact_checkpoints(budgets, exact_scores)

    guided = _validated_checkpoint_scores(
        component="guided",
        budgets=budgets,
        scores=guided_scores,
    )
    if feature_count <= 75:
        return {budget: dict(guided[budget]) for budget in budgets}

    random_values = _validated_checkpoint_scores(
        component="random",
        budgets=budgets,
        scores=random_scores,
    )
    if any(
        tuple(random_values[budget]) != tuple(guided[budget])
        for budget in budgets
    ):
        raise ValueError("hybrid component feature orders or sets conflict")
    return {
        budget: {
            feature: min(
                random_values[budget][feature],
                guided[budget][feature],
            )
            for feature in random_values[budget]
        }
        for budget in budgets
    }
