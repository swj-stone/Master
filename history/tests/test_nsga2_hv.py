"""
Unit tests for the corrected 2-D hypervolume computation and post-hoc
convergence metric in ``nsga2_fs``.

Run::

    ../venv/Scripts/python.exe -m pytest tests/test_nsga2_hv.py -v
"""

from __future__ import annotations

from inspect import signature
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import minppi_mofs.baselines.nsga2_fs as nsga2_module
import minppi_mofs.modeling as modeling
from minppi_mofs.baselines.nsga2_fs import (
    _compute_hv,
    _converged_gen,
    nsga2_feature_selection,
)


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


def test_hv_two_point_front() -> None:
    """Known two-point front → HV = 0.28 with ref (1.0, 1.0)."""
    front = np.array([[0.2, 0.8], [0.8, 0.2]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.28)


def test_hv_single_point() -> None:
    """Single point (0.25, 0.40) → HV = (1-0.25)*(1-0.40) = 0.45."""
    front = np.array([[0.25, 0.40]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.45)


def test_hv_respects_non_unit_reference_point() -> None:
    """The supplied reference point, rather than a hard-coded unit point, is used."""
    front = np.array([[0.2, 0.8], [0.8, 0.2]])
    ref = np.array([1.1, 1.2])
    # Rectangle union:
    #   0.9*0.4 + 0.3*1.0 - 0.3*0.4 = 0.54
    assert _compute_hv(front, ref) == pytest.approx(0.54)


def test_hv_ignores_duplicate_and_dominated_points() -> None:
    """Duplicates and a dominated point (0.9, 0.9) are filtered out."""
    front = np.array([[0.2, 0.8], [0.2, 0.8], [0.9, 0.9], [0.8, 0.2]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.28)


def test_hv_empty_front() -> None:
    """Empty front → 0."""
    assert _compute_hv(np.empty((0, 2)), np.array([1.0, 1.0])) == 0.0


def test_hv_all_outside_ref() -> None:
    """Points strictly worse than reference → 0 (all filtered out)."""
    front = np.array([[1.5, 0.5], [0.5, 1.5]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.0)


def test_hv_outside_points_do_not_remove_valid_contribution() -> None:
    """Reference-exceeding points are ignored without discarding valid points."""
    front = np.array([[0.25, 0.40], [1.20, 0.20], [0.20, 1.30]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.45)


def test_hv_with_dominated_single_point() -> None:
    """A single dominated point is removed; HV from non-dominated only."""
    front = np.array([[0.6, 0.7], [0.3, 0.4]])
    ref = np.array([1.0, 1.0])
    # (0.6, 0.7) is dominated by (0.3, 0.4) → only the latter counts
    # HV = (1-0.3)*(1-0.4) = 0.7 * 0.6 = 0.42
    assert _compute_hv(front, ref) == pytest.approx(0.42)


def test_hv_many_points_correctness() -> None:
    """Front of 5 well-spaced non-dominated points → manual verification."""
    # All are mutually non-dominated (x↓ y↑)
    front = np.array(
        [
            [0.1, 0.9],
            [0.3, 0.6],
            [0.5, 0.4],
            [0.7, 0.3],
            [0.9, 0.1],
        ]
    )
    ref = np.array([1.0, 1.0])
    hv = _compute_hv(front, ref)
    # Left→right sweep:
    #   best_y=1.0
    #   (0.1,0.9): y<1→(1-0.1)*(1-0.9)=0.9*0.1=0.09, best_y=0.9
    #   (0.3,0.6): y<0.9→(1-0.3)*(0.9-0.6)=0.7*0.3=0.21, best_y=0.6
    #   (0.5,0.4): y<0.6→(1-0.5)*(0.6-0.4)=0.5*0.2=0.10, best_y=0.4
    #   (0.7,0.3): y<0.4→(1-0.7)*(0.4-0.3)=0.3*0.1=0.03, best_y=0.3
    #   (0.9,0.1): y<0.3→(1-0.9)*(0.3-0.1)=0.1*0.2=0.02, best_y=0.1
    #   total = 0.09+0.21+0.10+0.03+0.02 = 0.45
    assert hv == pytest.approx(0.45)


def test_configured_reference_point_reaches_every_hv_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references: list[tuple[float, float]] = []
    monkeypatch.setattr(
        nsga2_module,
        "_compute_hv",
        lambda front, ref: references.append(tuple(ref)) or 0.2,
    )
    X = pd.DataFrame({"f0": [0.0, 1.0], "f1": [1.0, 0.0]})

    result = nsga2_feature_selection(
        X,
        np.array([0, 1]),
        X,
        np.array([0, 1]),
        task="binary",
        utility="roc_auc",
        model_params={"n_estimators": 1, "max_depth": 1, "learning_rate": 0.3, "n_jobs": 1},
        backend="cpu",
        seed=42,
        pop_size=4,
        max_gen=2,
        ref_point=(1.1, 1.1),
        utility_evaluator=lambda selected: .6,
    )

    assert result.hv == pytest.approx(0.2)
    # Last generation supplies the final HV; no second pass over its front.
    assert references == [(1.1, 1.1)] * 2


def test_initial_population_only_is_a_complete_exact_budget_schedule() -> None:
    evaluations = 0

    def evaluate(features):
        nonlocal evaluations
        evaluations += 1
        return .5

    X = pd.DataFrame({"f0": [0.0, 1.0], "f1": [1.0, 0.0]})
    result = nsga2_feature_selection(
        X,
        np.array([0, 1]),
        X,
        np.array([0, 1]),
        task="binary",
        utility="roc_auc",
        model_params={"n_estimators": 1, "max_depth": 1, "learning_rate": 0.3, "n_jobs": 1},
        backend="cpu",
        seed=42,
        pop_size=4,
        max_gen=0,
        ref_point=(1.1, 1.1),
        utility_evaluator=evaluate,
    )

    assert result.logical_evaluation_count == 4
    assert evaluations == result.unique_subset_count
    assert result.memoization_hit_count == 4 - evaluations
    assert result.hv_history == []
    assert result.converged_gen == 0


def test_duplicate_masks_are_evaluated_once_without_changing_logical_count() -> None:
    X = pd.DataFrame(
        {"f0": [0.0, 1.0], "f1": [1.0, 0.0]}
    )
    calls: list[tuple[str, ...]] = []

    def utility_evaluator(selected: tuple[str, ...]) -> float:
        calls.append(selected)
        return 0.5 + 0.1 * len(selected)

    result = nsga2_feature_selection(
        X,
        np.array([0, 1]),
        X,
        np.array([0, 1]),
        pop_size=4,
        max_gen=1,
        seed=42,
        utility_evaluator=utility_evaluator,
    )

    assert result.logical_evaluation_count == 8
    assert result.unique_subset_count == len(calls) == len(set(calls))
    assert result.unique_subset_count <= 4
    assert result.memoization_hit_count == 8 - result.unique_subset_count
    assert all(
        selected == tuple(feature for feature in X if feature in selected)
        for selected in calls
    )


def test_callback_memoization_preserves_complete_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pd.DataFrame(
        {
            "f0": [0.0, 1.0],
            "f1": [1.0, 0.0],
            "f2": [0.5, 0.5],
        }
    )

    def utility(selected: tuple[str, ...]) -> float:
        return 0.4 + 0.1 * len(selected) + (
            0.05 if "f0" in selected else 0.0
        )

    monkeypatch.setattr(nsga2_module, "prepare_subset_evaluator", lambda *args, **kw: utility)
    common = {
        "pop_size": 6,
        "max_gen": 3,
        "seed": 17,
        "ref_point": (1.1, 1.1),
    }
    legacy = nsga2_feature_selection(
        X, np.array([0, 1]), X, np.array([0, 1]), **common
    )
    callback = nsga2_feature_selection(
        X,
        np.array([0, 1]),
        X,
        np.array([0, 1]),
        utility_evaluator=utility,
        **common,
    )

    assert [
        (mask.tolist(), fitness)
        for mask, fitness in callback.pareto_front
    ] == [
        (mask.tolist(), fitness)
        for mask, fitness in legacy.pareto_front
    ]
    assert callback.hv == legacy.hv
    assert callback.hv_history == legacy.hv_history
    assert callback.converged_gen == legacy.converged_gen
    assert callback.logical_evaluation_count == (
        legacy.logical_evaluation_count
    )


def test_formal_population_and_generation_schedule_keeps_800_logical_evaluations() -> None:
    X = pd.DataFrame(
        {f"f{index}": [0.0, 1.0] for index in range(6)}
    )
    result = nsga2_feature_selection(
        X,
        np.array([0, 1]),
        X,
        np.array([0, 1]),
        pop_size=40,
        max_gen=19,
        seed=42,
        utility_evaluator=lambda selected: 0.5,
    )

    assert result.logical_evaluation_count == 800
    assert result.logical_evaluation_count == (
        result.unique_subset_count + result.memoization_hit_count
    )


def test_initial_all_unique_masks_have_no_memoization_hits() -> None:
    X = pd.DataFrame({"f0": [0.0, 1.0]})
    result = nsga2_feature_selection(
        X,
        np.array([0, 1]),
        X,
        np.array([0, 1]),
        pop_size=2,
        max_gen=0,
        seed=42,
        utility_evaluator=lambda selected: 0.5 + 0.1 * len(selected),
    )

    assert result.logical_evaluation_count == 2
    assert result.unique_subset_count == 2
    assert result.memoization_hit_count == 0


def test_default_search_budget_and_ranking_adapter_agree() -> None:
    columns = pd.DataFrame({"signal": [0., 1.], "noise": [1., 0.]})
    seen = []

    def utility(selected):
        seen.append(selected)
        return .5

    result = nsga2_feature_selection(
        columns, np.array([0, 1]), columns, np.array([0, 1]), utility_evaluator=utility
    )
    for selector in (nsga2_feature_selection, nsga2_module.nsga2_ranking):
        parameters = signature(selector).parameters
        assert parameters["pop_size"].default == 8
        assert parameters["max_gen"].default == 2
    assert result.logical_evaluation_count == 24
    assert len(result.hv_history) == 2
    assert result.unique_subset_count == len(set(seen)) == len(seen)
    assert result.memoization_hit_count == 24 - len(seen)


@pytest.mark.parametrize(("task", "metric"), [
    ("binary", "roc_auc"), ("multiclass", "ovr_auc"), ("regression", "r2"),
])
def test_prepared_search_matches_independent_canonical_models(task, metric, monkeypatch):
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(90, 3)), columns=["b[0]", "d", "a"])
    z = x["d"] + .1*x["a"]
    y = z.to_numpy() if task == "regression" else np.digitize(z, [-.4, .2] if task == "multiclass" else [0.])
    train, val = x.iloc[:60], x.iloc[60:]
    config = dict(task=task, utility=metric, seed=71, backend="cpu",
                  model_params=dict(n_estimators=3, max_depth=2, n_jobs=1))

    def reference(selected):
        if selected:
            pred = modeling.fit_predict(
                train.loc[:, list(selected)], y[:60], val.loc[:, list(selected)],
                **{k: v for k,v in config.items() if k != "utility"},
            )
        else:
            pred = modeling._empty_subset_predictions(y[:60], task=task, row_count=len(val))
        return modeling.compute_utility(y[60:], pred, metric=metric)

    baseline = nsga2_feature_selection(train, y[:60], val, y[60:], **config, utility_evaluator=reference)
    checked_frames = Mock(wraps=modeling._validated_frame)
    monkeypatch.setattr(modeling, "_validated_frame", checked_frames)
    events = []
    fast = nsga2_feature_selection(train, y[:60], val, y[60:], **config, progress=lambda **s: events.append(s))
    assert checked_frames.call_count == 2
    assert [s["logical_evaluation_count"] for s in events] == list(range(1, 25))
    assert sum(not s["cached"] for s in events) == fast.unique_subset_count == baseline.unique_subset_count
    assert events[-1]["memoization_hit_count"] == fast.memoization_hit_count == baseline.memoization_hit_count
    assert fast.hv_history == baseline.hv_history
    assert fast.hv == baseline.hv
    assert [(m.tolist(), v) for m,v in fast.pareto_front] == [(m.tolist(), v) for m,v in baseline.pareto_front]


def test_sort_reuse_preserves_tie_order_and_search_trajectory(monkeypatch):
    """Golden front from the unoptimized loop with repeated masks and objective ties."""
    x = pd.DataFrame([[0]*9, [1]*9], columns=list("abcdefghi"))
    calls = Mock(wraps=nsga2_module._non_dominated_sort)
    monkeypatch.setattr(nsga2_module, "_non_dominated_sort", calls)
    result = nsga2_feature_selection(
        x, [0, 1], x, [0, 1], seed=42, pop_size=11, max_gen=4,
        utility_evaluator=lambda names: .4 + .1*min(4, len(names)) + .05*("b" in names),
    )
    assert calls.call_count == 4
    assert result.unique_subset_count == 40 and result.memoization_hit_count == 15
    assert result.hv_history == [.6277777777777779] + [.688888888888889]*3
    assert ["".join(map(str, mask)) for mask, _ in result.pareto_front] == [
        "010000000", "110100001", "110010100", "010000000", "110000000", "110000100",
        "110000100", "010000001", "110010100", "110000110", "110010001",
    ]


# ---------------------------------------------------------------------------
# Convergence metric
# ---------------------------------------------------------------------------


def test_converged_gen_empty() -> None:
    """Empty history → 0."""
    assert _converged_gen([]) == 0


def test_converged_gen_strictly_increasing() -> None:
    """Monotonically increasing HV: gen where cum-best hits 95%."""
    h = [0.10, 0.20, 0.30, 0.40, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
    # cum-best: 0.10, 0.20, 0.30, 0.40, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55
    # 0.95*0.55 = 0.5225
    # gen 7 cum-best=0.52 → not yet
    # gen 8 cum-best=0.53 → yes
    assert _converged_gen(h) == 8


def test_converged_gen_plateau() -> None:
    """HV plateaus after gen 4; cumulative best reaches 95% early."""
    h = [0.20, 0.50, 0.60, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    # cum-best: 0.20, 0.50, 0.60, 0.75, 0.75, ...
    # 0.95*0.75 = 0.7125
    # gen 4 cum-best=0.75 → yes
    assert _converged_gen(h) == 4


def test_converged_gen_zero_hv_uses_max_generation_sentinel() -> None:
    """No positive HV means no meaningful convergence; return max generation."""
    assert _converged_gen([0.0, 0.0, 0.0]) == 3


def test_converged_gen_fluctuating() -> None:
    """HV fluctuates due to estimation noise; cumulative best is monotonic."""
    h = [0.30, 0.25, 0.35, 0.28, 0.33, 0.40, 0.39, 0.42, 0.41, 0.43]
    # cum-best: 0.30, 0.30, 0.35, 0.35, 0.35, 0.40, 0.40, 0.42, 0.42, 0.43
    # 0.95*0.43 = 0.4085
    # gen 8 cum-best=0.42 → yes
    assert _converged_gen(h) == 8
