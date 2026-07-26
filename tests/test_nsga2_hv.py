"""
Unit tests for the corrected 2-D hypervolume computation and post-hoc
convergence metric in ``nsga2_fs``.

Run::

    python -m pytest tests/test_nsga2_hv.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import minppi_mofs.baselines.nsga2_fs as nsga2_module
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
        "_evaluate_individual",
        lambda mask, *args, **kwargs: (float(mask.sum()) / 2.0, 0.4),
    )
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
    )

    assert result.hv == pytest.approx(0.2)
    assert references == [(1.1, 1.1)] * 3


def test_initial_population_only_is_a_complete_exact_budget_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluations = 0

    def evaluate(mask, *args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return (float(mask.sum()) / 2.0, 0.5)

    monkeypatch.setattr(nsga2_module, "_evaluate_individual", evaluate)
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

    monkeypatch.setattr(
        nsga2_module,
        "evaluate_fixed_subset",
        lambda *args, **_kwargs: utility(tuple(args[4])),
    )
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
