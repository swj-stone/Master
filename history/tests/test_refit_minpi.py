import numpy as np
import pytest
from sklearn.linear_model import Ridge

from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.piv2 import _folds
from minppi_mofs.refit_minpi import _refit_votes, refit_top_from_training


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_nested_path_matches_candidate_by_candidate_ridge_refits(task):
    rng = np.random.default_rng(439)
    X = rng.normal(size=(73, 9))
    X[::4, 1] = np.nan
    X[:, -1] = 3
    y = rng.integers(0, 3, len(X)) if task == "classification" else rng.normal(size=len(X))
    folds = _folds(y, task, np.random.default_rng(13), row_limit=256)
    target = (y[:, None] == np.unique(y)).astype(float) if task == "classification" else y[:, None]
    selected = []
    minimum = np.full(X.shape[1], np.inf)
    for size, next_size in [(0, 1), (1, 2), (2, 4), (4, 5)]:
        expected = np.zeros(X.shape[1])
        for ref, query in (folds, folds[::-1]):
            fit, check = _standardize(X[ref], X[query])
            center = target[ref].mean(axis=0)
            def error(cols):
                pred = center
                if cols:
                    model = Ridge(alpha=1, solver="svd", fit_intercept=False)
                    model.fit(fit[:, cols], target[ref] - center)
                    pred = center + model.predict(check[:, cols]).reshape(len(query), -1)
                return np.square(target[query] - pred).sum()
            base = error(selected)
            gains = _refit_votes(fit, check, target[ref] - center, target[query] - center, selected)
            for j in range(X.shape[1]):
                if j not in selected:
                    direct = base - error(selected + [j])
                    assert gains[j] == pytest.approx(direct, abs=1e-10)
                    expected[j] += direct
        minimum = np.minimum(minimum, expected)
        remaining = [j for j in range(X.shape[1]) if j not in selected]
        ranked = sorted(remaining, key=lambda j: -minimum[j])
        selected.extend(ranked[:next_size - size])
    actual, info = refit_top_from_training(X, y, 5, task=task, random_state=13)
    assert list(actual) == sorted(selected)
    assert info["sizes"] == [0, 1, 2, 4]


def test_refit_ties_full_budget_and_explicit_levels():
    X = np.zeros((16, 8))
    y = np.arange(16, dtype=float)
    assert refit_top_from_training(X, y, 3, task="regression", tiers=(0, 2))[0].tolist() == [0, 1, 2]
    assert refit_top_from_training(X, y, 8, task="regression")[1]["ballot_rows"] == 0
    for tiers in ((1,), (0, 3)):
        with pytest.raises(ValueError):
            refit_top_from_training(X, y, 3, task="regression", tiers=tiers)
