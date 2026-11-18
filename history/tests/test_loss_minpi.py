import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from minppi_mofs.loss_minpi import held_loss_votes, loss_top_from_training
from minppi_mofs.piv2 import _folds
from minppi_mofs.word_minpi import exact_top_from_votes


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cross_fitted_ballots_match_separate_univariate_ridge_fits(task):
    rng = np.random.default_rng(366)
    X = rng.normal(size=(270, 6))
    X[::7, 1] = np.nan
    X[:, 4] = 0.1
    X[:, 5] = np.nan
    y = rng.integers(0, 3, len(X)) if task == "classification" else rng.normal(size=len(X))
    if task == "classification":
        y = np.array(["blue", "green", "red"])[y]
        targets = (y[:, None] == np.unique(y)).astype(float)
    else:
        targets = y[:, None]

    folds = _folds(y, task, np.random.default_rng(11), row_limit=256)
    truth = np.zeros(X.shape, dtype=np.int8)
    for train, test in (folds, folds[::-1]):
        baseline = ((targets[test] - targets[train].mean(axis=0))**2).sum(axis=1)
        for j in range(4):
            ref = X[train, j, None]
            ref_mean = np.nanmean(ref)
            ref = np.nan_to_num(ref, nan=ref_mean)
            query = np.nan_to_num(X[test, j, None], nan=ref_mean)
            model = make_pipeline(StandardScaler(), Ridge(alpha=1, solver="svd"))
            predicted = model.fit(ref, targets[train]).predict(query).reshape(len(test), -1)
            fitted = ((targets[test] - predicted)**2).sum(axis=1)
            truth[test, j] = np.sign(baseline - fitted)

    ballots = held_loss_votes(X, y, task, 11)
    np.testing.assert_array_equal(ballots, truth[np.concatenate(folds)])
    columns, info = loss_top_from_training(X, y, 2, task=task, random_state=11, tiers=(0, 3))
    expected, _ = exact_top_from_votes(ballots, 2, tiers=(0, 3))
    np.testing.assert_array_equal(columns, expected)
    np.testing.assert_array_equal(held_loss_votes(X, y, task, 11), ballots)
    assert info["ballot_rows"] == 256


def test_query_target_does_not_change_other_query_ballots_in_its_fold():
    rng = np.random.default_rng(84)
    X = rng.normal(size=(90, 8))
    y = rng.normal(size=len(X))
    left, right = _folds(y, "regression", np.random.default_rng(28), row_limit=256)
    before = held_loss_votes(X, y, "regression", 28)
    center = y[right].mean()
    y[left[0]] = center - np.copysign(100.0, y[left[0]] - center)
    changed = held_loss_votes(X, y, "regression", 28)
    np.testing.assert_array_equal(before[1:len(left)], changed[1:len(left)])
    assert np.any(before[0] != changed[0])
    np.testing.assert_array_equal(held_loss_votes(X, np.full(len(X), 0.1), "regression", 28),
                                  np.zeros_like(before))
