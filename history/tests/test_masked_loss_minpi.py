import numpy as np
import pytest
from sklearn.linear_model import Ridge

from minppi_mofs.masked_loss_minpi import joint_ridge_cis_scores, masked_loss_top_from_training, masked_loss_votes
from minppi_mofs.piv2 import _folds
from minppi_mofs.word_minpi import exact_top_from_votes


@pytest.mark.parametrize("task,p", [("classification", 5), ("regression", 91)])
def test_votes_match_explicit_feature_masking_with_joint_ridge(task, p):
    rng = np.random.default_rng(192)
    X = rng.normal(size=(90, p))
    X[:, 3] = X[:, 2]
    X[:, 0] = 0.1
    X[:, -1] = np.nan
    X[::3, 1] = np.nan
    y = rng.integers(0, 3, len(X)) if task == "classification" else rng.normal(size=len(X))
    targets = (y[:, None] == np.unique(y)).astype(float) if task == "classification" else y[:, None]
    folds = _folds(y, task, np.random.default_rng(12), row_limit=256)
    expected = np.zeros_like(X, dtype=np.int8)
    singletons = np.zeros_like(X)
    for ref, query in (folds, folds[::-1]):
        # Drop known constant and empty columns; fit the joint model once.
        fit = X[ref, 1:-1]
        mean = np.nanmean(fit, axis=0)
        fit = np.where(np.isfinite(fit), fit, mean)
        check = np.where(np.isfinite(X[query, 1:-1]), X[query, 1:-1], mean)
        scale = fit.std(axis=0)
        fit, check = (fit - mean) / scale, (check - mean) / scale
        model = Ridge(alpha=1, solver="svd").fit(fit, targets[ref])
        error = np.square(model.predict(check).reshape(len(query), -1) - targets[query]).sum(axis=1)
        empty = np.square(model.predict(np.zeros_like(check)).reshape(len(query), -1) - targets[query]).sum(axis=1)
        for j in range(check.shape[1]):
            masked = check.copy()
            masked[:, j] = 0
            pred = model.predict(masked).reshape(len(query), -1)
            without = np.square(pred - targets[query]).sum(axis=1)
            expected[query, j + 1] = np.sign(without - error)
            only = np.zeros_like(check)
            only[:, j] = check[:, j]
            pred = model.predict(only).reshape(len(query), -1)
            singletons[query, j + 1] = empty - np.square(pred - targets[query]).sum(axis=1)
    ballots = masked_loss_votes(X, y, task, 12)
    np.testing.assert_array_equal(ballots, expected[np.concatenate(folds)])
    chosen, info = masked_loss_top_from_training(X, y, 3, task=task, random_state=12)
    np.testing.assert_array_equal(chosen, exact_top_from_votes(ballots, 3)[0])
    assert info["ballot_rows"] == len(X)
    np.testing.assert_allclose(joint_ridge_cis_scores(X, y, task, 12),
                               singletons.mean(axis=0), atol=1e-12)
