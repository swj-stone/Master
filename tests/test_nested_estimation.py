from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from minppi_mofs.coalition_cache import (
    CACHE_SCHEMA_VERSION,
    CoalitionCacheIdentity,
    CoalitionUtilityCache,
    CoalitionUtilityView,
    SQLiteCoalitionStore,
    hash_numeric_array,
    hash_numeric_frame,
)
from minppi_mofs.nested_estimation import (
    NestedGuidedSchedule,
    NestedProgress,
    NestedRandomSchedule,
    derive_hybrid_checkpoints,
    estimate_guided_checkpoints,
    estimate_random_checkpoints,
    guided_seed_coalition,
    make_nested_guided_schedule,
    make_nested_random_schedule,
    materialize_exact_checkpoints,
    nested_guided_schedule_digest,
    nested_random_schedule_digest,
)


FEATURES = ("a", "b", "c", "d", "e")
EXACT = {"a": -1, "b": 0}
RANDOM = {"a": 0, "b": 1}
GUIDED = {"a": 1, "b": -1}


class RecordingView:
    def __init__(
        self,
        feature_order: tuple[str, ...],
        values: dict[str, list[int]] | None = None,
    ) -> None:
        self.feature_order = feature_order
        self._values = values or {}
        self.calls: list[tuple[frozenset[str], str]] = []

    def eval_marginal(self, coalition: set[str], feature: str) -> int:
        self.calls.append((frozenset(coalition), feature))
        values = self._values.get(feature, [])
        feature_call_index = sum(
            called_feature == feature for _, called_feature in self.calls
        ) - 1
        return values[feature_call_index] if feature_call_index < len(values) else 1

    def calls_for(self, feature: str) -> int:
        return sum(called_feature == feature for _, called_feature in self.calls)


def make_real_shared_cache_views(tmp_path):
    X_train = pd.DataFrame(
        {
            "a": [0.0, 1.0, 0.1, 0.9, 0.2, 0.8],
            "b": [1.0, 0.0, 0.8, 0.2, 0.7, 0.3],
            "c": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        }
    )
    X_eval = pd.DataFrame(
        {
            "a": [0.05, 0.95, 0.15, 0.85],
            "b": [0.95, 0.05, 0.85, 0.15],
            "c": [0.15, 0.85, 0.25, 0.75],
        }
    )
    y_train = np.array([0, 1, 0, 1, 0, 1])
    y_eval = np.array([0, 1, 0, 1])
    identity = CoalitionCacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        protocol_hash="0" * 64,
        dataset_id="nested-integration",
        seed=17,
        namespace="selection_validation",
        train_features_sha256=hash_numeric_frame(X_train),
        train_target_sha256=hash_numeric_array(y_train),
        eval_features_sha256=hash_numeric_frame(X_eval),
        eval_target_sha256=hash_numeric_array(y_eval),
        ordered_features=("a", "b", "c"),
        task="binary",
        class_labels=(0, 1),
        model_params={"n_estimators": 2, "n_jobs": 1},
        backend="cpu",
        model_seed=17,
        library_versions={"numpy": np.__version__, "xgboost": "test-boundary"},
    )
    store = SQLiteCoalitionStore(
        tmp_path / "nested-integration.sqlite3",
        identity,
        commit_every_fits=1,
    )
    cache = CoalitionUtilityCache(
        identity,
        X_train,
        y_train,
        X_eval,
        y_eval,
        store,
    )
    random_view = CoalitionUtilityView(cache, utility="accuracy", alpha=0.8)
    guided_view = CoalitionUtilityView(cache, utility="accuracy", alpha=0.8)
    return cache, store, random_view, guided_view


def test_guided_schedule_uses_independent_feature_streams() -> None:
    schedule = make_nested_guided_schedule(
        FEATURES, budgets=(10, 25, 50, 100), seed=42
    )
    prefix = make_nested_guided_schedule(FEATURES, budgets=(10, 25), seed=42)

    for index in range(len(FEATURES)):
        assert prefix.seed_orders[index] == schedule.seed_orders[index]
        assert prefix.trials[index] == schedule.trials[index][:25]


def test_equal_guided_schedules_do_not_share_owned_tuple_containers() -> None:
    first = make_nested_guided_schedule(FEATURES, budgets=(2, 4), seed=42)
    second = make_nested_guided_schedule(FEATURES, budgets=(2, 4), seed=42)

    assert first == second
    assert first is not second
    assert first.seed_orders is not second.seed_orders
    assert first.trials is not second.trials
    for index in range(len(FEATURES)):
        assert first.seed_orders[index] is not second.seed_orders[index]
        assert first.trials[index] is not second.trials[index]


def test_guided_schedule_is_reproducible_and_digest_binds_its_state() -> None:
    baseline = make_nested_guided_schedule(FEATURES, budgets=(2, 4), seed=42)
    repeated = make_nested_guided_schedule(FEATURES, budgets=(2, 4), seed=42)

    assert baseline == repeated
    assert nested_guided_schedule_digest(baseline) == nested_guided_schedule_digest(
        repeated
    )
    assert nested_guided_schedule_digest(baseline) != nested_guided_schedule_digest(
        make_nested_guided_schedule(FEATURES, budgets=(2, 4), seed=43)
    )


def test_guided_digest_binds_checkpoints_seed_orders_and_every_trial() -> None:
    baseline = make_nested_guided_schedule(FEATURES, budgets=(2, 4), seed=42)
    changed_checkpoints = NestedGuidedSchedule(
        features=baseline.features,
        seed=baseline.seed,
        checkpoints=(1, 4),
        seed_orders=baseline.seed_orders,
        trials=baseline.trials,
    )
    changed_seed_orders = NestedGuidedSchedule(
        features=baseline.features,
        seed=baseline.seed,
        checkpoints=baseline.checkpoints,
        seed_orders=(
            tuple(reversed(baseline.seed_orders[0])),
            *baseline.seed_orders[1:],
        ),
        trials=baseline.trials,
    )
    changed_trials = NestedGuidedSchedule(
        features=baseline.features,
        seed=baseline.seed,
        checkpoints=baseline.checkpoints,
        seed_orders=baseline.seed_orders,
        trials=(
            (
                baseline.trials[0][1],
                baseline.trials[0][0],
                *baseline.trials[0][2:],
            ),
            *baseline.trials[1:],
        ),
    )

    digests = {
        nested_guided_schedule_digest(schedule)
        for schedule in (
            baseline,
            changed_checkpoints,
            changed_seed_orders,
            changed_trials,
        )
    }

    assert len(digests) == 4


@pytest.mark.parametrize(
    ("mi_scores", "correlation_neighbors", "expected"),
    [
        (None, None, frozenset({"e", "f"})),
        (
            {"a": 10.0, "b": 0.0, "c": 9.0, "d": 9.0, "e": 1.0, "f": 2.0},
            None,
            frozenset({"c", "d"}),
        ),
        (None, {"a": ("f",)}, frozenset({"b", "f"})),
        (
            {"a": 10.0, "b": 0.0, "c": 9.0, "d": 9.0, "e": 1.0, "f": 2.0},
            {"a": ("f",)},
            frozenset({"c", "f"}),
        ),
    ],
)
def test_guided_seed_coalition_covers_all_guidance_modes_with_stable_ties(
    mi_scores: dict[str, float] | None,
    correlation_neighbors: dict[str, tuple[str, ...]] | None,
    expected: frozenset[str],
) -> None:
    features = tuple("abcdef")
    seed_order = ("f", "e", "d", "c", "b")
    mi_before = None if mi_scores is None else dict(mi_scores)
    neighbors_before = (
        None
        if correlation_neighbors is None
        else {key: tuple(value) for key, value in correlation_neighbors.items()}
    )

    result = guided_seed_coalition(
        "a",
        features,
        seed_order,
        mi_scores=mi_scores,
        correlation_neighbors=correlation_neighbors,
    )

    assert result == expected
    assert mi_scores == mi_before
    assert correlation_neighbors == neighbors_before


@pytest.mark.parametrize(
    ("correlation_neighbors", "expected"),
    [
        (None, frozenset()),
        ({"a": ("b",)}, frozenset({"b"})),
    ],
)
def test_guided_seed_size_zero_preserves_brief_correlated_behavior(
    correlation_neighbors: dict[str, tuple[str, ...]] | None,
    expected: frozenset[str],
) -> None:
    result = guided_seed_coalition(
        "a",
        ("a", "b"),
        ("b",),
        mi_scores=None,
        correlation_neighbors=correlation_neighbors,
    )

    assert result == expected


def test_guided_seed_caps_correlated_neighbors_at_first_ten() -> None:
    features = (
        "target",
        *(f"p{index}" for index in range(1, 22)),
        *(f"n{index}" for index in range(1, 12)),
    )
    neighbors = tuple(f"n{index}" for index in range(1, 12))
    seed_order = tuple(reversed(features[1:]))

    result = guided_seed_coalition(
        "target",
        features,
        seed_order,
        mi_scores=None,
        correlation_neighbors={"target": neighbors},
    )

    assert result == frozenset(
        {"n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "p1"}
    )


def _reference_seed_coalition(
    feature: str,
    features: tuple[str, ...],
    seed_order: tuple[str, ...],
    mi_scores: dict[str, float] | None,
    neighbors: dict[str, tuple[str, ...]] | None,
) -> frozenset[str]:
    if mi_scores is None:
        ranked = features
    else:
        positions = {item: index for index, item in enumerate(features)}
        ranked = tuple(
            item
            for item, _ in sorted(
                mi_scores.items(),
                key=lambda pair: (-pair[1], positions[pair[0]]),
            )[: max(1, len(features) // 2)]
        )
    correlated = tuple(
        item
        for item in (() if neighbors is None else neighbors.get(feature, ()))
        if item in features and item != feature
    )
    target = min(20, len(features) // 3)
    if correlated:
        chosen = set(correlated[:10])
        for item in ranked:
            if len(chosen) >= target:
                break
            if item != feature:
                chosen.add(item)
        return frozenset(chosen)
    ranked_set = set(ranked)
    candidates = [
        item for item in seed_order if item in ranked_set and item != feature
    ]
    return frozenset(candidates[:target])


def _reference_guided_checkpoints(
    schedule: NestedGuidedSchedule,
    view: RecordingView,
    mi_scores: dict[str, float] | None,
    neighbors: dict[str, tuple[str, ...]] | None,
) -> dict[int, dict[str, int]]:
    result = {budget: {} for budget in schedule.checkpoints}
    for index, feature in enumerate(schedule.features):
        seed = _reference_seed_coalition(
            feature,
            schedule.features,
            schedule.seed_orders[index],
            mi_scores,
            neighbors,
        )
        best = 1
        cursor = 0
        for budget in schedule.checkpoints:
            while cursor < budget and best != -1:
                trial = schedule.trials[index][cursor]
                coalition = set(seed)
                for candidate, draw in zip(
                    schedule.features, trial.removal_draws, strict=True
                ):
                    if candidate in coalition and draw < 0.3:
                        coalition.remove(candidate)
                outside = [
                    item
                    for item in trial.add_order
                    if item not in coalition and item != feature
                ]
                coalition.update(outside[: trial.add_count])
                best = min(best, view.eval_marginal(coalition, feature))
                cursor += 1
            result[budget][feature] = best
    return result


@pytest.mark.parametrize(
    ("mi_scores", "correlation_neighbors"),
    [
        (None, None),
        (
            {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0},
            None,
        ),
        (None, {"a": ("e",), "c": ("b", "d")}),
        (
            {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0},
            {"a": ("e",), "c": ("b", "d")},
        ),
    ],
)
def test_guided_checkpoints_match_direct_reference_for_every_checkpoint(
    mi_scores: dict[str, float] | None,
    correlation_neighbors: dict[str, tuple[str, ...]] | None,
) -> None:
    schedule = make_nested_guided_schedule(
        FEATURES, budgets=(1, 3, 5), seed=17
    )
    values = {
        feature: [1, 0, 1, -1, 1]
        for feature in FEATURES
    }
    actual_view = RecordingView(FEATURES, values=values)
    reference_view = RecordingView(FEATURES, values=values)
    mi_before = None if mi_scores is None else dict(mi_scores)
    neighbors_before = (
        None
        if correlation_neighbors is None
        else {key: tuple(value) for key, value in correlation_neighbors.items()}
    )

    actual = estimate_guided_checkpoints(
        schedule,
        actual_view,
        mi_scores=mi_scores,
        correlation_neighbors=correlation_neighbors,
    )
    expected = _reference_guided_checkpoints(
        schedule,
        reference_view,
        mi_scores,
        correlation_neighbors,
    )

    assert actual == expected
    assert actual_view.calls == reference_view.calls
    assert mi_scores == mi_before
    assert correlation_neighbors == neighbors_before


def test_guided_checkpoints_ignore_unrelated_global_guidance_entries() -> None:
    features = ("a", "b", "c")
    schedule = make_nested_guided_schedule(features, budgets=(1, 3), seed=23)
    global_mi: dict[str, object] = {
        "a": 3.0,
        "b": 2.0,
        "c": 1.0,
        "outside": float("inf"),
    }
    subset_mi = {"a": 3.0, "b": 2.0, "c": 1.0}
    global_neighbors: dict[str, object] = {
        "a": ("b", "outside"),
        "outside": "not a neighbor sequence",
    }
    subset_neighbors = {"a": ("b", "outside")}
    global_mi_before = dict(global_mi)
    global_neighbors_before = dict(global_neighbors)
    global_view = RecordingView(features)
    subset_view = RecordingView(features)

    global_result = estimate_guided_checkpoints(
        schedule,
        global_view,
        mi_scores=global_mi,
        correlation_neighbors=global_neighbors,
    )
    subset_result = estimate_guided_checkpoints(
        schedule,
        subset_view,
        mi_scores=subset_mi,
        correlation_neighbors=subset_neighbors,
    )

    assert global_result == subset_result
    assert global_view.calls == subset_view.calls
    assert global_mi == global_mi_before
    assert global_neighbors == global_neighbors_before


def test_guided_checkpoint_evaluation_propagates_early_harm() -> None:
    schedule = make_nested_guided_schedule(
        FEATURES, budgets=(2, 3, 5), seed=9
    )
    view = RecordingView(FEATURES, values={"a": [1, 0, -1, 1, 1]})

    result = estimate_guided_checkpoints(
        schedule,
        view,
        mi_scores=None,
        correlation_neighbors=None,
    )

    assert result[2]["a"] == 0
    assert result[3]["a"] == -1
    assert result[5]["a"] == -1
    assert view.calls_for("a") == 3


def test_guided_progress_is_emitted_at_feature_checkpoint_boundaries() -> None:
    schedule = make_nested_guided_schedule(("a", "b"), budgets=(1, 2), seed=42)
    events: list[NestedProgress] = []

    estimate_guided_checkpoints(
        schedule,
        RecordingView(("a", "b")),
        mi_scores=None,
        correlation_neighbors=None,
        progress=events.append,
    )

    assert events == [
        NestedProgress("guided", 1, 2, 1, 2),
        NestedProgress("guided", 1, 2, 2, 2),
        NestedProgress("guided", 2, 2, 1, 2),
        NestedProgress("guided", 2, 2, 2, 2),
    ]


@pytest.mark.parametrize(
    ("mi_scores", "correlation_neighbors", "message"),
    [
        ({"a": 1.0}, None, "score|feature"),
        (
            {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": float("inf")},
            None,
            "finite|numeric",
        ),
        (None, {"a": "b"}, "sequence|neighbor"),
        (None, {"a": (1,)}, "string|neighbor"),
    ],
)
def test_guided_checkpoint_validation_precedes_view_side_effects(
    mi_scores: dict[str, float] | None,
    correlation_neighbors: object,
    message: str,
) -> None:
    schedule = make_nested_guided_schedule(FEATURES, budgets=(2,), seed=42)
    view = RecordingView(FEATURES)

    with pytest.raises((TypeError, ValueError), match=message):
        estimate_guided_checkpoints(
            schedule,
            view,
            mi_scores=mi_scores,
            correlation_neighbors=correlation_neighbors,
        )

    assert view.calls == []


def test_guided_checkpoint_rejects_mismatched_view_before_calls() -> None:
    schedule = make_nested_guided_schedule(FEATURES, budgets=(2,), seed=42)
    view = RecordingView(tuple(reversed(FEATURES)))

    with pytest.raises(ValueError, match="feature order"):
        estimate_guided_checkpoints(
            schedule,
            view,
            mi_scores=None,
            correlation_neighbors=None,
        )

    assert view.calls == []


def test_guided_checkpoint_rejects_corrupted_trial_before_calls() -> None:
    schedule = make_nested_guided_schedule(FEATURES, budgets=(2,), seed=42)
    trial = schedule.trials[0][0]
    object.__setattr__(
        trial,
        "removal_draws",
        (float("nan"), *trial.removal_draws[1:]),
    )
    view = RecordingView(FEATURES)

    with pytest.raises(ValueError, match="finite|trial|draw"):
        estimate_guided_checkpoints(
            schedule,
            view,
            mi_scores=None,
            correlation_neighbors=None,
        )

    assert view.calls == []


def test_random_schedule_is_budget_independent_and_prefix_nested() -> None:
    small = make_nested_random_schedule(FEATURES, budgets=(10, 25), seed=42)
    large = make_nested_random_schedule(
        FEATURES, budgets=(10, 25, 50, 100), seed=42
    )

    assert small.trials[0] == large.trials[0][:25]
    assert small.trials[1] == large.trials[1][:25]
    assert small.checkpoints == (10, 25)
    assert large.checkpoints == (10, 25, 50, 100)
    assert nested_random_schedule_digest(large) == nested_random_schedule_digest(
        make_nested_random_schedule(
            FEATURES, budgets=(10, 25, 50, 100), seed=42
        )
    )


def test_later_feature_stream_is_independent_of_earlier_feature_draw_count() -> None:
    first_feature_short = make_nested_random_schedule(
        FEATURES, budgets=(25,), seed=77
    )
    first_feature_long = make_nested_random_schedule(
        FEATURES, budgets=(25, 50), seed=77
    )

    assert first_feature_short.trials[1] == first_feature_long.trials[1][:25]


def test_digest_binds_seed_feature_order_checkpoints_and_trials() -> None:
    baseline = make_nested_random_schedule(FEATURES, budgets=(4, 8), seed=42)
    different_seed = make_nested_random_schedule(FEATURES, budgets=(4, 8), seed=43)
    different_order = make_nested_random_schedule(
        tuple(reversed(FEATURES)), budgets=(4, 8), seed=42
    )
    different_checkpoints = make_nested_random_schedule(
        FEATURES, budgets=(3, 8), seed=42
    )
    replacement = (
        frozenset()
        if baseline.trials[0][0] != frozenset()
        else frozenset({"b"})
    )
    modified_trials = tuple(
        tuple(
            replacement if feature_index == 0 and trial_index == 0 else trial
            for trial_index, trial in enumerate(feature_trials)
        )
        for feature_index, feature_trials in enumerate(baseline.trials)
    )
    different_trials = NestedRandomSchedule(
        features=baseline.features,
        seed=baseline.seed,
        checkpoints=baseline.checkpoints,
        trials=modified_trials,
    )

    digests = {
        nested_random_schedule_digest(schedule)
        for schedule in (
            baseline,
            different_seed,
            different_order,
            different_checkpoints,
            different_trials,
        )
    }
    assert len(digests) == 5


@pytest.mark.parametrize(
    ("features", "budgets", "seed", "message"),
    [
        (("a", "a"), (1,), 42, "feature"),
        (("a",), (), 42, "checkpoint|budget"),
        (("a",), (0,), 42, "positive"),
        (("a",), (2, 1), 42, "increasing"),
        (("a",), (1, 1), 42, "increasing|unique"),
        (("a",), (1,), -1, "seed"),
        (("a",), (1,), True, "seed"),
    ],
)
def test_schedule_rejects_invalid_features_budgets_and_seed(
    features: tuple[str, ...],
    budgets: tuple[int, ...],
    seed: int,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        make_nested_random_schedule(features, budgets=budgets, seed=seed)


def test_schedule_is_deeply_immutable_and_caps_trials_at_coalition_count() -> None:
    schedule = make_nested_random_schedule(("a", "b", "c"), budgets=(2, 100), seed=5)

    assert all(len(feature_trials) == 4 for feature_trials in schedule.trials)
    assert all(
        isinstance(coalition, frozenset)
        for feature_trials in schedule.trials
        for coalition in feature_trials
    )
    with pytest.raises(FrozenInstanceError):
        schedule.seed = 6
    with pytest.raises(ValueError, match="tuple|immutable"):
        NestedRandomSchedule(
            features=["a", "b"],
            seed=5,
            checkpoints=(1,),
            trials=((frozenset(),), (frozenset(),)),
        )


def test_large_feature_space_does_not_require_materializing_all_coalitions() -> None:
    features = tuple(f"f{index}" for index in range(128))

    schedule = make_nested_random_schedule(features, budgets=(1, 2), seed=42)

    assert len(schedule.trials) == 128
    assert all(len(feature_trials) == 2 for feature_trials in schedule.trials)


def test_checkpoint_evaluation_reuses_each_feature_cursor() -> None:
    features = tuple("abcdefghi")
    schedule = make_nested_random_schedule(
        features, budgets=(10, 25, 50, 100), seed=42
    )
    view = RecordingView(
        features,
        values={"a": [1] * 10 + [0] * 190},
    )

    result = estimate_random_checkpoints(schedule, view)

    assert tuple(result) == (10, 25, 50, 100)
    assert result[10]["a"] == 1
    assert result[25]["a"] == 0
    assert result[50]["a"] == 0
    assert result[100]["a"] == 0
    assert view.calls_for("a") == 100


def test_checkpoint_evaluation_propagates_early_harm_without_extra_calls() -> None:
    schedule = make_nested_random_schedule(FEATURES, budgets=(2, 3, 5), seed=9)
    view = RecordingView(FEATURES, values={"a": [1, 0, -1, 1, 1]})

    result = estimate_random_checkpoints(schedule, view)

    assert result[2]["a"] == 0
    assert result[3]["a"] == -1
    assert result[5]["a"] == -1
    assert view.calls_for("a") == 3


def test_checkpoint_evaluation_rejects_view_feature_order_before_calls() -> None:
    schedule = make_nested_random_schedule(FEATURES, budgets=(2,), seed=42)
    view = RecordingView(tuple(reversed(FEATURES)))

    with pytest.raises(ValueError, match="feature order"):
        estimate_random_checkpoints(schedule, view)

    assert view.calls == []


def test_checkpoint_evaluation_rejects_private_only_feature_order_before_side_effects() -> None:
    schedule = make_nested_random_schedule(FEATURES, budgets=(2,), seed=42)
    events: list[NestedProgress] = []

    class PrivateOnlyView:
        def __init__(self) -> None:
            self._cache = type("PrivateCache", (), {"feature_order": FEATURES})()
            self.calls = []

        def eval_marginal(self, coalition: set[str], feature: str) -> int:
            self.calls.append((coalition, feature))
            return 0

    view = PrivateOnlyView()
    with pytest.raises(ValueError, match="expose.*feature order"):
        estimate_random_checkpoints(schedule, view, progress=events.append)

    assert view.calls == []
    assert events == []


def test_real_cache_views_compose_with_random_and_guided_checkpoints(
    monkeypatch, tmp_path
) -> None:
    fit_calls = 0
    predictions = np.array(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
    )

    def deterministic_fit_predict(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return predictions.copy()

    monkeypatch.setattr(
        "minppi_mofs.coalition_cache.fit_predict",
        deterministic_fit_predict,
    )
    cache, store, random_view, guided_view = make_real_shared_cache_views(tmp_path)
    features = ("a", "b", "c")
    checkpoints = (1, 2)
    random_schedule = make_nested_random_schedule(
        features, checkpoints, seed=23
    )
    guided_schedule = make_nested_guided_schedule(
        features, checkpoints, seed=23
    )
    try:
        assert random_view.feature_order == features
        assert guided_view.feature_order == features

        random_scores = estimate_random_checkpoints(
            random_schedule, random_view
        )
        random_logical_counts = (
            random_view.model_fit_count,
            random_view.coalition_count,
        )
        guided_scores = estimate_guided_checkpoints(
            guided_schedule,
            guided_view,
            mi_scores=None,
            correlation_neighbors=None,
        )

        assert tuple(random_scores) == checkpoints
        assert tuple(guided_scores) == checkpoints
        assert all(tuple(scores) == features for scores in random_scores.values())
        assert all(tuple(scores) == features for scores in guided_scores.values())
        assert all(
            score in {-1, 0, 1}
            for result in (random_scores, guided_scores)
            for scores in result.values()
            for score in scores.values()
        )
        assert random_logical_counts == (
            random_view.model_fit_count,
            random_view.coalition_count,
        )
        assert random_view.coalition_count > 1
        assert guided_view.coalition_count > 1
        assert cache.stats.physical_model_fit_count == fit_calls
        assert 0 < cache.stats.physical_model_fit_count <= cache.stats.coalition_count
        assert cache.stats.cache_hit_count > 0
    finally:
        store.close()


def test_progress_is_emitted_only_after_each_feature_checkpoint() -> None:
    schedule = make_nested_random_schedule(("a", "b"), budgets=(1, 2), seed=42)
    view = RecordingView(("a", "b"))
    events: list[NestedProgress] = []

    estimate_random_checkpoints(schedule, view, progress=events.append)

    assert events == [
        NestedProgress("random", 1, 2, 1, 2),
        NestedProgress("random", 1, 2, 2, 2),
        NestedProgress("random", 2, 2, 1, 2),
        NestedProgress("random", 2, 2, 2, 2),
    ]


def test_exact_materialization_returns_independent_checkpoint_snapshots() -> None:
    exact_scores = dict(EXACT)

    result = materialize_exact_checkpoints([10, 25], exact_scores)
    result[10]["a"] = 1

    assert result[25] == EXACT
    assert exact_scores == EXACT
    assert result[10] is not result[25]


@pytest.mark.parametrize(
    ("feature_count", "expected"),
    [
        (20, EXACT),
        (21, GUIDED),
        (75, GUIDED),
        (76, {"a": 0, "b": -1}),
    ],
)
def test_hybrid_derivation_matches_dispatcher_boundary(
    feature_count: int, expected: dict[str, int]
) -> None:
    result = derive_hybrid_checkpoints(
        feature_count=feature_count,
        checkpoints=(10,),
        exact_scores=EXACT,
        random_scores={10: RANDOM},
        guided_scores={10: GUIDED},
    )

    assert result[10] == expected


@pytest.mark.parametrize("feature_count", [True, 0, -1, 1.0])
def test_hybrid_derivation_rejects_invalid_feature_count(
    feature_count: object,
) -> None:
    with pytest.raises(ValueError, match="feature count"):
        derive_hybrid_checkpoints(
            feature_count=feature_count,
            checkpoints=(10,),
            exact_scores=EXACT,
            random_scores={10: RANDOM},
            guided_scores={10: GUIDED},
        )


@pytest.mark.parametrize(
    ("feature_count", "random_scores", "guided_scores"),
    [
        (21, None, {10: {"a": 2}}),
        (76, {10: {"a": 2}}, {10: {"a": 0}}),
        (76, {10: {"a": 0}}, {10: {"a": -2}}),
    ],
)
def test_hybrid_derivation_rejects_out_of_domain_component_scores(
    feature_count, random_scores, guided_scores
) -> None:
    with pytest.raises(ValueError, match=r"\{-1, 0, 1\}"):
        derive_hybrid_checkpoints(
            feature_count=feature_count,
            checkpoints=(10,),
            exact_scores=None,
            random_scores=random_scores,
            guided_scores=guided_scores,
        )


def test_guided_only_hybrid_output_copies_shared_input_scores() -> None:
    shared_scores = dict(GUIDED)
    guided_scores = {10: shared_scores, 25: shared_scores}

    result = derive_hybrid_checkpoints(
        feature_count=21,
        checkpoints=(10, 25),
        exact_scores=None,
        random_scores=None,
        guided_scores=guided_scores,
    )
    result[10]["a"] = -1

    assert result[25] == GUIDED
    assert shared_scores == GUIDED
    assert result[10] is not result[25]
    assert result[10] is not shared_scores


@pytest.mark.parametrize(
    "checkpoints",
    [(), (True,), (0,), (10, 10), (25, 10), "10"],
)
def test_exact_materialization_rejects_invalid_checkpoints(
    checkpoints: object,
) -> None:
    with pytest.raises(ValueError, match="checkpoint|budget"):
        materialize_exact_checkpoints(checkpoints, EXACT)


@pytest.mark.parametrize(
    "exact_scores",
    [
        {"a": 2},
        {"a": True},
        {"a": 0.0},
        {"": 0},
        {},
        [("a", 0)],
    ],
)
def test_exact_materialization_rejects_invalid_scores(
    exact_scores: object,
) -> None:
    with pytest.raises(ValueError, match="score|feature|mapping"):
        materialize_exact_checkpoints((10,), exact_scores)


@pytest.mark.parametrize(
    ("feature_count", "exact_scores", "random_scores", "guided_scores", "message"),
    [
        (20, None, None, None, "exact"),
        (21, None, None, None, "guided"),
        (76, None, None, {10: GUIDED}, "random"),
        (21, None, None, {25: GUIDED}, "guided.*incomplete"),
        (21, None, None, {10: GUIDED, 25: GUIDED}, "guided.*incomplete"),
        (76, None, {25: RANDOM}, {10: GUIDED}, "random.*incomplete"),
        (76, None, {10: RANDOM, 25: RANDOM}, {10: GUIDED}, "random.*incomplete"),
        (21, None, None, {True: GUIDED}, "checkpoint"),
    ],
)
def test_hybrid_derivation_requires_complete_relevant_components(
    feature_count: int,
    exact_scores: dict[str, int] | None,
    random_scores: dict[int, dict[str, int]] | None,
    guided_scores: dict[int, dict[str, int]] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_hybrid_checkpoints(
            feature_count=feature_count,
            checkpoints=(10,),
            exact_scores=exact_scores,
            random_scores=random_scores,
            guided_scores=guided_scores,
        )


@pytest.mark.parametrize(
    ("feature_count", "exact_scores", "random_scores", "guided_scores", "expected"),
    [
        (20, EXACT, {True: {"": 9}}, {False: {"": 9}}, EXACT),
        (21, {"": 9}, {True: {"": 9}}, {10: GUIDED}, GUIDED),
        (76, {"": 9}, {10: RANDOM}, {10: GUIDED}, {"a": 0, "b": -1}),
    ],
)
def test_hybrid_derivation_ignores_irrelevant_components(
    feature_count: int,
    exact_scores: dict[str, int],
    random_scores: dict[int, dict[str, int]],
    guided_scores: dict[int, dict[str, int]],
    expected: dict[str, int],
) -> None:
    result = derive_hybrid_checkpoints(
        feature_count=feature_count,
        checkpoints=(10,),
        exact_scores=exact_scores,
        random_scores=random_scores,
        guided_scores=guided_scores,
    )

    assert result == {10: expected}


@pytest.mark.parametrize(
    ("random_scores", "guided_scores"),
    [
        (
            {10: {"a": 0, "b": 1}},
            {10: {"b": -1, "a": 1}},
        ),
        (
            {10: {"a": 0}},
            {10: {"a": 1, "b": -1}},
        ),
        (
            {10: {"a": 0, "b": 1}, 25: {"a": 0, "b": 0}},
            {10: {"a": 1, "b": -1}, 25: {"b": -1, "a": 1}},
        ),
    ],
)
def test_hybrid_derivation_rejects_component_feature_conflicts(
    random_scores: dict[int, dict[str, int]],
    guided_scores: dict[int, dict[str, int]],
) -> None:
    checkpoints = tuple(guided_scores)

    with pytest.raises(ValueError, match="feature order|feature set"):
        derive_hybrid_checkpoints(
            feature_count=76,
            checkpoints=checkpoints,
            exact_scores=None,
            random_scores=random_scores,
            guided_scores=guided_scores,
        )


def test_guided_derivation_rejects_cross_checkpoint_feature_order_conflict() -> None:
    with pytest.raises(ValueError, match="feature order"):
        derive_hybrid_checkpoints(
            feature_count=21,
            checkpoints=(10, 25),
            exact_scores=None,
            random_scores=None,
            guided_scores={
                10: {"a": 1, "b": 0},
                25: {"b": 0, "a": -1},
            },
        )


def _reference_random_checkpoints(
    schedule: NestedRandomSchedule,
    view: RecordingView,
) -> dict[int, dict[str, int]]:
    result = {budget: {} for budget in schedule.checkpoints}
    for index, feature in enumerate(schedule.features):
        best = 1
        cursor = 0
        for budget in schedule.checkpoints:
            while cursor < min(budget, len(schedule.trials[index])) and best != -1:
                best = min(
                    best,
                    view.eval_marginal(set(schedule.trials[index][cursor]), feature),
                )
                cursor += 1
            result[budget][feature] = best
    return result


def test_hybrid_checkpoints_equal_independent_direct_dispatcher_reference() -> None:
    features = ("a", "b", "c")
    checkpoints = (1, 3)
    random_schedule = make_nested_random_schedule(features, checkpoints, seed=13)
    guided_schedule = make_nested_guided_schedule(features, checkpoints, seed=13)
    random_values = {
        "a": [0, -1, 1],
        "b": [1, 1, 0],
        "c": [1, 0, 0],
    }
    guided_values = {
        "a": [1, 0, 0],
        "b": [0, -1, 1],
        "c": [1, 1, -1],
    }
    random_view = RecordingView(features, values=random_values)
    guided_view = RecordingView(features, values=guided_values)

    random_scores = estimate_random_checkpoints(random_schedule, random_view)
    guided_scores = estimate_guided_checkpoints(
        guided_schedule,
        guided_view,
        mi_scores=None,
        correlation_neighbors=None,
    )
    actual = derive_hybrid_checkpoints(
        feature_count=76,
        checkpoints=checkpoints,
        exact_scores=None,
        random_scores=random_scores,
        guided_scores=guided_scores,
    )

    reference_random_view = RecordingView(features, values=random_values)
    reference_guided_view = RecordingView(features, values=guided_values)
    reference_random = _reference_random_checkpoints(
        random_schedule, reference_random_view
    )
    reference_guided = _reference_guided_checkpoints(
        guided_schedule,
        reference_guided_view,
        None,
        None,
    )
    expected = {
        budget: {
            feature: min(
                reference_random[budget][feature],
                reference_guided[budget][feature],
            )
            for feature in features
        }
        for budget in checkpoints
    }

    assert actual == expected
    assert random_view.calls == reference_random_view.calls
    assert guided_view.calls == reference_guided_view.calls


def test_hybrid_derivation_does_not_mutate_or_alias_component_scores() -> None:
    random_scores = {10: dict(RANDOM), 25: {"a": -1, "b": 0}}
    guided_scores = {10: dict(GUIDED), 25: {"a": 0, "b": -1}}
    random_before = {budget: dict(scores) for budget, scores in random_scores.items()}
    guided_before = {budget: dict(scores) for budget, scores in guided_scores.items()}

    result = derive_hybrid_checkpoints(
        feature_count=76,
        checkpoints=(10, 25),
        exact_scores=None,
        random_scores=random_scores,
        guided_scores=guided_scores,
    )
    result[10]["a"] = 1

    assert result[25] == {"a": -1, "b": -1}
    assert random_scores == random_before
    assert guided_scores == guided_before


def test_singleton_schedules_support_empty_other_feature_sets() -> None:
    random_schedule = make_nested_random_schedule(("a",), (1, 2), seed=4)
    guided_schedule = make_nested_guided_schedule(("a",), (1, 2), seed=4)

    assert random_schedule.trials == ((frozenset(),),)
    assert guided_schedule.seed_orders == ((),)
    assert all(trial.add_order == () for trial in guided_schedule.trials[0])
    assert estimate_random_checkpoints(
        random_schedule, RecordingView(("a",))
    ) == {1: {"a": 1}, 2: {"a": 1}}
    assert estimate_guided_checkpoints(
        guided_schedule,
        RecordingView(("a",)),
        mi_scores=None,
        correlation_neighbors=None,
    ) == {1: {"a": 1}, 2: {"a": 1}}


def test_direct_nested_guided_schedule_rejects_non_trial_entries() -> None:
    with pytest.raises(ValueError, match="trial"):
        NestedGuidedSchedule(
            features=("a",),
            seed=4,
            checkpoints=(1,),
            seed_orders=((),),
            trials=((object(),),),
        )


def test_checkpoint_evaluation_does_not_construct_or_consume_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_schedule = make_nested_random_schedule(FEATURES, (1, 2), seed=8)
    guided_schedule = make_nested_guided_schedule(FEATURES, (1, 2), seed=8)

    def fail_rng(*args: object, **kwargs: object) -> None:
        pytest.fail("checkpoint evaluation must not access RNG")

    monkeypatch.setattr(
        "minppi_mofs.nested_estimation.np.random.default_rng", fail_rng
    )
    monkeypatch.setattr(
        "minppi_mofs.nested_estimation.np.random.SeedSequence", fail_rng
    )

    estimate_random_checkpoints(random_schedule, RecordingView(FEATURES))
    estimate_guided_checkpoints(
        guided_schedule,
        RecordingView(FEATURES),
        mi_scores=None,
        correlation_neighbors=None,
    )


def test_evaluator_checkpoint_result_dictionaries_are_independent() -> None:
    random_schedule = make_nested_random_schedule(FEATURES, (1, 2), seed=8)
    guided_schedule = make_nested_guided_schedule(FEATURES, (1, 2), seed=8)
    random_result = estimate_random_checkpoints(
        random_schedule, RecordingView(FEATURES)
    )
    guided_result = estimate_guided_checkpoints(
        guided_schedule,
        RecordingView(FEATURES),
        mi_scores=None,
        correlation_neighbors=None,
    )

    random_result[1]["a"] = -1
    guided_result[1]["a"] = -1

    assert random_result[2]["a"] == 1
    assert guided_result[2]["a"] == 1
