from itertools import combinations

import numpy as np
import pytest

from minppi_mofs.tournament_minpi import tournament_top_from_training


def _unit(X):
    cols = []
    for col in X.T:
        valid = np.isfinite(col)
        z = np.zeros(len(col))
        if valid.any():
            z[valid] = col[valid] - col[valid].mean()
        norm = np.linalg.norm(z)
        cols.append(z / norm if norm else z)
    return np.column_stack(cols)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("h", [None, 1, 3, 5])
def test_tournament_matches_direct_subset_game(task, h):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(128, 7))
    y = X[:, 0] + X[:, 5] + rng.normal(size=len(X))
    X[:, 2] = -X[:, 1]
    X[::3, 1] = np.nan
    X[::2, 4] = np.inf
    X[:, 3] = np.nan
    if task == "classification":
        y = np.digitize(y, [0., 1.1, 3.1])
    groups = np.repeat(np.arange(4), 32)
    cols, info = tournament_top_from_training(X, y, h, task=task, groups=groups, ordered_groups=True)
    active = info["active_columns"]
    fit, held = (info["folds"][0][k] for k in ("fit_rows", "held_rows"))
    assert groups[fit].max() < groups[held].min()
    train, check = _unit(X[fit]), _unit(X[held])
    target = (y[:, None] == np.unique(y)) if task == "classification" else y[:, None]
    a, b = train.T @ _unit(target[fit]), check.T @ _unit(target[held])
    blocks = [set(pair) for pair in info["pairs"]]
    blocks.extend({j} for j in active if all(j not in block for block in blocks))

    def value(subset):
        total = 0.
        for block in blocks:
            idx = sorted(subset & block)
            if idx:
                gram = train[:, idx].T @ train[:, idx] * len(fit) / (len(fit) + 1.)
                np.fill_diagonal(gram, 1.)
                total += (np.linalg.solve(gram, a[idx]) * b[idx]).sum()
        return total

    values = {frozenset(S): value(set(S))
              for k in range(len(active) + 1) for S in combinations(active, k)}
    expected = []
    for j in active:
        peers = frozenset(active) - {j}
        expected.append(min(values[frozenset({j})],
                            values[peers | {j}] - values[peers]))
    np.testing.assert_allclose(info["scores"], expected, atol=1e-12)
    assert len(cols) == (info["budget"] if h is None else h)
    assert np.unique(cols).tolist() == cols.tolist()
    assert 0 < len(cols) < X.shape[1]
    if h == 5:
        assert len(active) == 2 * (7 - 5)
        assert set(range(7)) - set(active) <= set(cols)
        for pair in info["pairs"]:
            assert len(set(pair) & set(cols)) == 1
    if h is None:
        score = dict(zip(active, info["scores"], strict=True))
        ordered = [sorted(pair, key=lambda j: (score[j], -j))[0] for pair in info["pairs"]]
        ordered.sort(key=lambda j: (score[j], -j))
        mass = np.maximum(expected, 0.).sum()
        count = sum(np.maximum([score[j] for j in ordered], 0.).cumsum() <= .02 * mass + 1e-15)
        assert set(cols) == set(range(7)) - set(ordered[:max(1, count)])


def test_groups_replay_and_train_only_read_budget():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(900, 27))
    X[7, 14] = np.nan
    y = rng.normal(size=900)
    groups = np.repeat(np.arange(9), 100)
    contract = dict(task="regression", groups=groups, random_state=123)
    cols, info = tournament_top_from_training(X, y, **contract)
    fit, held = (info["folds"][0][k] for k in ("fit_rows", "held_rows"))
    assert len(fit) <= 64 and len(held) <= 64
    assert not (set(groups[fit]) & set(groups[held]))
    # Changing uninspected training rows preserves every vote.
    unused = np.setdiff1d(np.arange(len(X)), np.r_[fit, held])
    X[unused] = np.nan
    replay, other = tournament_top_from_training(X, y, **contract)
    np.testing.assert_array_equal(replay, cols)
    np.testing.assert_array_equal(other["scores"], info["scores"])


def test_singleton_and_invalid_input():
    X = np.ones((4, 2))
    y = np.arange(4)
    cols, info = tournament_top_from_training(X[:, :1], y)
    assert cols.tolist() == [0] and info["folds"] == []
    cols, info = tournament_top_from_training(X, y)
    assert cols.tolist() == [0] and not info["scores"].any()
    with pytest.raises(ValueError):
        tournament_top_from_training(X, y, count=0)
    with pytest.raises(ValueError, match="two distinct"):
        tournament_top_from_training(X, y, groups=np.zeros(4))
