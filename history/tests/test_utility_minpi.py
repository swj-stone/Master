from itertools import combinations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from minppi_mofs.masked_loss_minpi import joint_ridge_cis_scores
from minppi_mofs.piv2 import _folds
from minppi_mofs.utility_minpi import _endpoint_gains, _fitted_halves, utility_top_from_training


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_size_conditioned_scores_equal_all_masked_coalitions(task):
    rng = np.random.default_rng(34)
    n, p = 51, 5
    X = rng.normal(size=(n, p))
    X[::4, 1] = np.nan
    X[:, 4] = 0.1
    y = np.array(["a", "b", "c"] * 17) if task == "classification" else rng.normal(size=n)
    target = (y[:, None] == np.unique(y)).astype(float) if task == "classification" else y[:, None]
    folds = _folds(y, task, np.random.default_rng(91), row_limit=256)
    expected = np.zeros((p, p))
    for fit, held in (folds, folds[::-1]):
        mean = np.nanmean(X[fit, :4], axis=0)
        train = np.where(np.isfinite(X[fit, :4]), X[fit, :4], mean)
        query = np.where(np.isfinite(X[held, :4]), X[held, :4], mean)
        scale = train.std(axis=0)
        train = np.column_stack(((train - mean) / scale, np.zeros(len(fit))))
        query = np.column_stack(((query - mean) / scale, np.zeros(len(held))))
        ridge = Ridge(alpha=1, solver="svd").fit(train, target[fit])
        losses = {}
        for size in range(p + 1):
            for coalition in combinations(range(p), size):
                mask = np.zeros(p, dtype=bool)
                mask[list(coalition)] = True
                pred = ridge.predict(query * mask).reshape(len(held), -1)
                losses[frozenset(coalition)] = np.square(pred - target[held]).sum()
        for j in range(p):
            others = sorted(set(range(p)) - {j})
            for k in range(p):
                change = [losses[frozenset(peers)] - losses[frozenset((*peers, j))]
                          for peers in combinations(others, k)]
                expected[k, j] += np.mean(change) / n

    empty, full = np.zeros(p), np.zeros(p)
    for query, centered, beta in _fitted_halves(X, y, task, 91):
        a, b = _endpoint_gains(query, centered, beta)
        empty += a / n
        full += b / n
    for k in range(p):
        np.testing.assert_allclose(empty + k * (full - empty) / (p - 1),
                                   expected[k], atol=2e-13)
    np.testing.assert_allclose(empty, joint_ridge_cis_scores(X, y, task, 91), atol=2e-13)
    for tiers in (None, (0,), (1, 3, 4)):
        support, info = utility_top_from_training(X, y, 3, task=task, random_state=91, tiers=tiers)
        sizes = [0, 1, 2] if tiers is None else list(tiers)
        order = np.argsort(-expected[sizes].min(axis=0), kind="stable")[:3]
        np.testing.assert_array_equal(support, np.sort(order))
        assert info["peer_window"] == (min(sizes), max(sizes))
        assert info["ballot_rows"] == len(X)


def test_wide_dual_and_ties():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 80))
    y = X[:, 67]
    selected, info = utility_top_from_training(X, y, 1, task="regression", random_state=5)
    assert selected.tolist() == [67]
    assert info["peer_window"] == (0, 0)
    const = np.full((6, 3), np.nan)
    selected, _ = utility_top_from_training(const, np.zeros(6), 2, task="regression")
    assert selected.tolist() == [0, 1]
    all_columns, info = utility_top_from_training(X[:, :1], y, 1, task="regression")
    assert all_columns.tolist() == [0] and info["ballot_rows"] == 0
    with pytest.raises(ValueError):
        utility_top_from_training(X, y, True)
    with pytest.raises(ValueError):
        utility_top_from_training(X, y, 1, tiers=[-1])
