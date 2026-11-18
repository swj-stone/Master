import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from minppi_mofs.piv3_lite import MinPIScreen, _contexts, _decide, _paired_contrasts, screen


def test_shadow_contrasts_match_individual_coalitions():
    rng = np.random.default_rng(107)
    p, m = 10, 3
    signals = rng.normal(size=(p+m, 40))
    target = rng.normal(size=40)
    order = rng.permutation(p)
    tiers = (1, 3, 8)
    positions, residuals = _contexts(signals, target, order, tiers)
    dots = signals @ np.vstack([*residuals, signals[p:]]).T / len(target)
    norms = np.mean(signals**2, axis=1)
    control = p + rng.integers(m, size=p)
    for h, k in enumerate(tiers):
        actual = _paired_contrasts(dots, norms, positions, k, control, 2*h)
        for j in range(p):
            total = signals[order[order != j][:k]].sum(axis=0)
            a = (total + signals[j])/(k+1)
            b = (total + signals[control[j]])/(k+1)
            expected = np.mean((target-b)**2 - (target-a)**2)*(k+1)**2
            np.testing.assert_allclose(actual[j], expected, rtol=1e-12, atol=1e-12)


def test_min_can_remove_column_mean_keeps_and_protection_overrides():
    tiers = np.ones((2, 3, 4))
    tiers[:, 2, :3] = -1
    tiers[1, 2, 1] = .5
    gains = np.zeros((2, 4))
    gains[0, 2] = .9
    thresholds = np.full(2, .2)
    keep_min, _, _ = _decide(tiers, gains, thresholds, True, "min")
    keep_mean, _, _ = _decide(tiers, gains, thresholds, True, "mean")
    assert keep_min.tolist() == [False, True, True, True]
    assert keep_mean.all()
    assert _decide(tiers, gains, thresholds, False, "min")[0].all()


@pytest.mark.parametrize("task", ["binary", "regression", "multiclass"])
def test_screen_reproducible_and_retains_signal(task):
    rng = np.random.default_rng(43)
    x = pd.DataFrame(rng.normal(size=(1550, 100)), columns=[f"x{i}" for i in range(100)])
    x.iloc[::8, 6] = np.nan
    y = x.x0.to_numpy() + rng.normal(0, .1, len(x))
    if task != "regression":
        y = np.digitize(y, [-.5, .5] if task == "multiclass" else [0])
    with threadpool_limits(limits=1):
        names, diag = screen(x, y, task=task)
        assert (names, diag) == screen(x, y, task=task)
        mean_names, mean_diag = screen(x, y, task=task, aggregate="mean")
    assert names == [c for c in x if c in set(names)] and "x0" in names
    assert set(names) <= set(mean_names) and len(names) < x.shape[1]
    assert diag["tier_rates"] == mean_diag["tier_rates"]
    np.testing.assert_equal(np.min(diag["tier_rates"], axis=0), diag["aggregate_scores"])
    assert diag["front_fits"] == 0 and diag["sample_rows"] == 1024
    json.dumps(diag, allow_nan=False)


def test_interaction_without_marginal_signal_and_small_inputs_pass_through():
    pattern = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    x = np.tile(pattern, (400, 40))
    y = (x[:, 0]*x[:, 1] > 0).astype(int)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    with threadpool_limits(limits=1):
        names, diag = screen(frame, y, task="binary", seed=15)
    assert names == list(frame.columns) and diag["bypass"] == "weak global signal"
    model = clone(MinPIScreen()).fit(frame.iloc[:, :3], y)
    assert model.diagnostics_["bypass"] == "small input"
    pd.testing.assert_frame_equal(model.transform(frame.iloc[:, :3]), frame.iloc[:, :3])
    with pytest.raises(ValueError, match="order"):
        model.transform(frame.iloc[:, [1, 0, 2]])
    assert MinPIScreen().fit_transform(x[:, :3], y).shape == (len(y), 3)

