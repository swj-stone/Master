from itertools import combinations

import numpy as np
import pytest

from minppi_mofs.pair_vote_minpi import pair_vote_top_from_training


def _block_game(columns, mate, train, check, target, observed):
    """Fit each active block independently by explicitly augmenting its ridge design."""
    value = 0.
    for j, partner in enumerate(mate):
        if j > partner:
            continue
        active = sorted({j, partner}.intersection(columns))
        if not active:
            continue
        width = len(active)
        design = np.vstack([train[:, active], np.eye(width)])
        targets = np.vstack([target, np.zeros((width, target.shape[1]))])
        beta = np.linalg.lstsq(design, targets, rcond=None)[0]
        error = observed - check[:, active] @ beta
        value += np.mean(np.sum(observed**2 - error**2, axis=1))
    return value


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("width,count", [(5, 1), (5, 4), (6, 3)])
def test_pair_score_matches_uniform_peer_coalition_enumeration(task, width, count):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(45, width))
    X[:, 1] = -.6 * X[:, 0] + .2 * X[:, 1]
    y = X[:, 1] + .4 * X[:, 4] - X[:, 3]
    if task == "classification":
        y = np.digitize(y, [-.7, .3, .7])
        targets = np.eye(4)[y]
    else:
        targets = y[:, None]
    groups = np.repeat([0, 1, 2], [12, 15, 18])
    cols, info = pair_vote_top_from_training(X, y, count, task=task, groups=groups,
                                             ordered_groups=True)
    assert len(info["folds"]) == 2
    assert info["peer_window"] == (0, min(width - 1, max(1, count - 1)))
    expected = np.zeros((2, width))
    all_cols = set(range(width))
    for fold in info["folds"]:
        fit, held, mate = fold["fit_rows"], fold["held_rows"], fold["mate"]
        assert groups[fit].max() < groups[held].min()
        np.testing.assert_array_equal(mate[mate], np.arange(width))
        assert sum(mate == np.arange(width)) == width % 2
        mean = X[fit].mean(axis=0)
        scale = X[fit].std(axis=0)
        train, check = (X[fit] - mean) / scale, (X[held] - mean) / scale
        center = targets[fit].mean(axis=0)
        target, observed = targets[fit] - center, targets[held] - center
        values = {}
        for k in range(width + 1):
            for subset in combinations(range(width), k):
                active = frozenset(subset)
                values[active] = _block_game(active, mate, train, check, target, observed)
        for j in range(width):
            for tier, k in enumerate(info["peer_window"]):
                gains = []
                for subset in combinations(all_cols - {j}, k):
                    active = frozenset(subset)
                    gains.append(values[active | {j}] - values[active])
                expected[tier, j] += np.mean(gains) / len(info["folds"])
    scores = expected.min(axis=0)
    np.testing.assert_allclose(info["scores"], scores, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(cols, np.sort(np.argsort(-scores, kind="stable")[:count]))


@pytest.mark.parametrize("ordered", [False, True])
def test_replay_degenerate_pairs_and_separate_training_groups(ordered):
    rng = np.random.default_rng(20)
    groups = np.repeat(np.arange(6), 33)
    first = rng.normal(size=len(groups))
    X = np.column_stack([first, -first, np.ones(len(first)), np.full(len(first), np.nan),
                         np.zeros(len(first))])
    y = np.digitize(first, [-.5, .2])
    kwargs = {"groups": groups, "ordered_groups": ordered}
    chosen, info = pair_vote_top_from_training(X, y, 3, **kwargs)
    again, replay = pair_vote_top_from_training(X, y, 3, **kwargs)
    np.testing.assert_array_equal(chosen, again)
    np.testing.assert_array_equal(info["scores"], replay["scores"])
    np.testing.assert_array_equal(info["scores"][2:], np.zeros(3))
    for fold, other in zip(info["folds"], replay["folds"], strict=True):
        fit, held = fold["fit_rows"], fold["held_rows"]
        assert not np.intersect1d(groups[fit], groups[held]).size
        if ordered:
            assert groups[fit].max() < groups[held].min()
        for key in ("fit_rows", "held_rows", "mate"):
            np.testing.assert_array_equal(fold[key], other[key])


def test_single_column_and_stable_ties():
    X = np.ones((30, 5))
    y = np.repeat([0, 1, 2], 10)
    only, full = pair_vote_top_from_training(X[:, :1], y, 1)
    np.testing.assert_array_equal(only, [0])
    assert full["folds"] == []
    tied, info = pair_vote_top_from_training(X, y, 3)
    np.testing.assert_array_equal(tied, [0, 1, 2])
    np.testing.assert_array_equal(info["scores"], np.zeros(5))
