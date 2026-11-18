from itertools import combinations

import numpy as np
import pytest

from minppi_mofs.sketch_minpi import _budget_coefficients, reduced_width, sketch_top_from_training


def _enumerated_ridge(X, Y, h):
    """Stack every masked design and its own masked penalty; solve by SVD."""
    p = X.shape[1]
    blocks, targets = [], []
    for columns in combinations(range(p), h):
        mask = np.zeros(p)
        mask[list(columns)] = 1.
        blocks.extend([X * mask, np.diag(mask)])
        targets.extend([Y, np.zeros((p, Y.shape[1]))])
    return np.linalg.lstsq(np.vstack(blocks), np.vstack(targets), rcond=None)[0]


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
@pytest.mark.parametrize("h", [1, 2, 4])
def test_coefficients_equal_brute_force_ridge(shape, h):
    rng = np.random.default_rng(19)
    X = rng.normal(size=shape)
    X[:, -1] = X[:, 0]
    Y = rng.normal(size=(len(X), 3))
    np.testing.assert_allclose(_budget_coefficients(X, Y, h), _enumerated_ridge(X, Y, h),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("h, peers", [(1, (0, 1)), (3, (1, 2))])
def test_minimum_matches_exhaustive_held_coalition_votes(task, h, peers):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(37, 5))
    X[:, 3] += X[:, 1]
    y = X[:, 1] + .4 * X[:, 2] - .2 * X[:, 4]
    if task == "classification":
        y = np.digitize(y, [-.7, .1, .95])
        targets = np.eye(4)[y]
    else:
        targets = y[:, None]
    groups = np.repeat([0, 1, 2], [9, 12, 16])
    columns, info = sketch_top_from_training(X, y, h, task=task, groups=groups, ordered_groups=True)
    assert info["peer_window"] == peers
    by_size = np.zeros((2, X.shape[1]))
    for fold in info["folds"]:
        fit, held = fold["fit_rows"], fold["held_rows"]
        assert groups[fit].max() < groups[held].min()
        mean, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
        center = targets[fit].mean(axis=0)
        B = _enumerated_ridge((X[fit] - mean) / sd, targets[fit] - center, h)
        held_X = (X[held] - mean) / sd

        def loss(cols):
            prediction = center + held_X[:, cols] @ B[cols, :]
            return np.mean(np.square(targets[held] - prediction).sum(axis=1))

        for j in range(X.shape[1]):
            others = [i for i in range(X.shape[1]) if i != j]
            for tier, k in enumerate(peers):
                gains = [loss(list(S)) - loss([*S, j]) for S in combinations(others, k)]
                by_size[tier, j] += np.mean(gains) / len(info["folds"])
    expected = by_size.min(axis=0)
    np.testing.assert_allclose(info["scores"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(columns, np.sort(np.argsort(-expected, kind="stable")[:h]))


@pytest.mark.parametrize("strategy", ["row", "group", "forward"])
def test_support_sampling_and_missing_values_are_reproducible(strategy):
    rng = np.random.default_rng(99)
    y = rng.choice([-1, 1], size=900)
    X = np.column_stack((y, rng.normal(size=(len(y), 4))))
    X[:, 1] = np.nan
    X[::9, 2] = np.inf
    groups = None if strategy == "row" else np.repeat(np.arange(9), 100)
    options = dict(groups=groups, ordered_groups=strategy == "forward")
    cols, info = sketch_top_from_training(X, y, 2, **options)
    cols2, info2 = sketch_top_from_training(X, y, 2, **options)
    np.testing.assert_array_equal(cols, cols2)
    np.testing.assert_array_equal(info["scores"], info2["scores"])
    assert 0 in cols and info["scores"][1] == 0
    assert info["strategy"] == strategy and len(info["folds"]) == 2
    for fold, replay in zip(info["folds"], info2["folds"], strict=True):
        fit, held = fold["fit_rows"], fold["held_rows"]
        assert len(fit) <= 96 and len(held) <= 96
        assert not np.intersect1d(fit, held).size
        np.testing.assert_array_equal(fit, replay["fit_rows"])
        np.testing.assert_array_equal(held, replay["held_rows"])
        if groups is not None:
            assert not np.intersect1d(groups[fit], groups[held]).size
            if strategy == "forward":
                assert groups[fit].max() < groups[held].min()


def test_width_and_group_boundaries_and_ties():
    assert [reduced_width(p) for p in range(1, 9)] == [1, 1, 2, 3, 4, 5, 6, 6]
    assert reduced_width(8, 1.) == 7 and reduced_width(8, .01) == 1
    for value in (True, 0, -1., np.nan, 1.1):
        with pytest.raises(ValueError):
            reduced_width(3, value)
    X, y = np.ones((10, 4)), np.arange(10) % 2
    first, info = sketch_top_from_training(X, y, 3, groups=y, ordered_groups=True)
    assert len(info["folds"]) == 1
    np.testing.assert_array_equal(first, [0, 1, 2])
    full, full_info = sketch_top_from_training(X[:, :1], y, 1)
    np.testing.assert_array_equal(full, [0])
    assert not full_info["folds"]
    with pytest.raises(ValueError, match="distinct"):
        sketch_top_from_training(X, y, 2, groups=np.ones(len(y)))
    with pytest.raises(ValueError, match="identifiers"):
        sketch_top_from_training(X, y, 2, ordered_groups=True)
