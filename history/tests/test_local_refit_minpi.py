import numpy as np
import pytest
from sklearn.linear_model import Ridge

from minppi_mofs import local_refit_minpi
from minppi_mofs.local_refit_minpi import local_refit_top_from_training
from minppi_mofs.piv2 import _folds


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("tiers,checkpoints", [(None, [0, 1, 2, 4]), ((5, 0, 3), [0, 3, 5])])
def test_local_adjacent_path_against_independent_candidate_refits(task, tiers, checkpoints, monkeypatch):
    rng = np.random.default_rng(73)
    X = rng.normal(size=(85, 10))
    X[::4, 1] = np.nan
    X[:, 4] = 0
    y = rng.integers(0, 4, len(X)) if task == "classification" else rng.normal(size=len(X))
    targets = (y[:, None] == np.unique(y)).astype(float) if task == "classification" else y[:, None]
    folds = _folds(y, task, np.random.default_rng(42), row_limit=256)
    data = []
    for train, check in (folds, folds[::-1]):
        center = np.nanmean(X[train], axis=0)
        a = np.where(np.isfinite(X[train]), X[train], center) - center
        b = np.where(np.isfinite(X[check]), X[check], center) - center
        scale = np.sqrt((a ** 2).mean(axis=0))
        scale[scale == 0] = 1
        intercept = targets[train].mean(axis=0)
        data.append((a / scale, b / scale, targets[train] - intercept, targets[check] - intercept))

    def total_error(cols):
        error = 0.
        for x_fit, x_eval, y_fit, y_eval in data:
            predicted = 0.
            if cols:
                model = Ridge(alpha=1, fit_intercept=False, solver="svd").fit(x_fit[:, cols], y_fit)
                predicted = model.predict(x_eval[:, cols]).reshape(y_eval.shape)
            error += np.square(y_eval - predicted).sum()
        return error

    count = 6
    selected = []
    expected_prefixes = set()
    for size, next_size in zip(checkpoints, checkpoints[1:] + [count]):
        unused = sorted(set(range(X.shape[1])) - set(selected))
        layers = [selected] if size == 0 else [selected, selected[:-1]]
        scores = []
        for prefix in layers:
            expected_prefixes.add(tuple(prefix))
            base = total_error(prefix)
            scores.append([base - total_error(prefix + [j]) for j in unused])
        order = np.argsort(-np.min(scores, axis=0), kind="stable")
        selected.extend(unused[i] for i in order[:next_size - size])

    actual_prefixes = set()
    original = local_refit_minpi._refit_votes

    def traced(fit, check, fit_target, check_target, prefix):
        actual_prefixes.add(tuple(prefix))
        return original(fit, check, fit_target, check_target, prefix)

    monkeypatch.setattr(local_refit_minpi, "_refit_votes", traced)
    support, info = local_refit_top_from_training(X, y, count, task=task, tiers=tiers)
    assert actual_prefixes == expected_prefixes
    assert support.tolist() == sorted(selected)
    assert info["sizes"] == checkpoints
    assert info["ballot_rows"] == len(y)


def test_local_refit_ties_empty_only_and_full():
    X = np.zeros((16, 7))
    y = np.arange(len(X), dtype=float)
    for tiers in (None, (0,), (0, 2)):
        support, _ = local_refit_top_from_training(X, y, 4, task="regression", tiers=tiers)
        assert support.tolist() == [0, 1, 2, 3]
    support, info = local_refit_top_from_training(X, y, 7, task="regression")
    assert support.tolist() == list(range(7)) and info["ballot_rows"] == 0
    for tiers in ((1,), (0, 4)):
        with pytest.raises(ValueError):
            local_refit_top_from_training(X, y, 4, task="regression", tiers=tiers)
