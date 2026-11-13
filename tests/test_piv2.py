import json

import numpy as np
import pandas as pd
import pytest

from minppi_mofs.piv2 import _changes, _folds, _selection, prefilter


def test_coalition_changes_match_direct_per_feature_predictions():
    rng = np.random.default_rng(17)
    signals = rng.normal(size=(8, 40))
    target = rng.normal(size=40)
    norms = np.mean(signals**2, axis=1)
    order = rng.permutation(8)
    for k in (1, 4, 7):
        actual = _changes(signals, target, norms, order, k)
        for j in range(8):
            subset = order[order != j][:k]
            old = signals[subset].mean(axis=0)
            new = (k*old + signals[j]) / (k+1)
            expected = np.mean((target-old)**2 - (target-new)**2)
            np.testing.assert_allclose(actual[j], expected, rtol=1e-12, atol=1e-14)


def test_one_consistently_weak_tier_can_outvote_two_strong_tiers():
    net = np.zeros((2, 3, 6), dtype=int)
    net[:, :, 3] = [4, 4, -2]
    net[0, :, 4] = [4, 4, -4]
    net[1, :, 4] = [4, 4, 1]
    rates, worst, protected, removed = _selection(net, np.arange(6)[::-1])
    np.testing.assert_array_equal(protected, [0, 1, 2])
    assert rates[:, 3].mean() > 0 and worst[3] < 0
    assert worst[4] < 0
    np.testing.assert_array_equal(removed, [3])


def test_fold_sampling_includes_rare_labels_and_never_overlaps():
    labels = np.repeat([0, 1, 2], [5000, 30, 3])
    a, b = _folds(labels, "multiclass", np.random.default_rng(3))
    assert not set(a).intersection(b)
    assert len(a) + len(b) == 1024
    assert set(labels[a]) == set(labels[b]) == {0, 1, 2}
    assert (labels[np.r_[a, b]] == 2).sum() == 3


@pytest.mark.parametrize("task", ["regression", "multiclass"])
def test_training_only_votes_are_deterministic_and_preserve_input_order(task):
    rng = np.random.default_rng(30)
    x = pd.DataFrame(rng.normal(size=(1400, 14)), columns=list("ZYXWVUTSRQPONM"))
    x.iloc[::11, 9] = np.nan
    y = x["Z"].to_numpy() + rng.normal(scale=.02, size=len(x))
    if task == "multiclass":
        y = np.digitize(y, [-.5, .5])
    rows = np.arange(1200)
    kept, state = prefilter(x.iloc[rows], y[rows], seed=3, task=task)
    x.iloc[1200:] = 42
    y[1200:] = 99
    assert (kept, state) == prefilter(x.iloc[rows], y[rows], seed=3, task=task)
    assert 7 <= len(kept) < 14 and "Z" in kept
    assert kept == [col for col in x.columns if col in set(kept)]
    assert state["sample_rows"] == 1024
    np.testing.assert_allclose(np.min(state["tier_rates"], axis=0), state["min_vote"])
    json.dumps(state, allow_nan=False)


def test_flat_and_undersupported_inputs_return_all_columns():
    x = pd.DataFrame(np.ones((20, 8)), columns=list("ABCDEFGH"))
    keep, state = prefilter(x, [0, 1]*10, task="binary")
    assert keep == list(x.columns) and state["min_vote"] == [0.]*8
    for small, labels in ((x.iloc[:, :2], [0, 1]*10), (x, [0]*19 + [1])):
        keep, state = prefilter(small, labels, task="binary")
        assert keep == list(small.columns) and state["bypass"]
