import numpy as np
import pytest
from sklearn.linear_model import Ridge

from minppi_mofs.environment_minpi import (
    InsufficientEnvironmentsError, environment_path_from_training, environment_top_from_training,
)
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.support_selector import PivotalSupportSelector


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_path_matches_explicit_held_ridge_refits(task):
    rng = np.random.default_rng(31)
    groups = np.repeat(np.arange(6), 15)
    X = rng.normal(size=(len(groups), 9))
    X[::5, 1] = np.nan
    y = X[:, 4] + rng.normal(size=len(X)) * .3
    if task == "classification":
        y = np.digitize(y, [-1, .5])
    opts = dict(task=task, groups=groups, ordered_groups=True, random_state=27)
    supports, info = environment_path_from_training(X, y, [2, 4, 5, 9], **opts)
    prefix = []
    for size, next_size in [(0, 1), (1, 2), (2, 4), (4, 5)]:
        scores = np.full(X.shape[1], np.inf)
        for fold in info["folds"]:
            fit, held = fold["fit_rows"], fold["held_rows"]
            assert groups[fit].max() < groups[held].min()
            a, b = _standardize(X[fit], X[held])
            if task == "classification":
                target = (y[:, None] == np.unique(y)).astype(float)
                _, codes, counts = np.unique(y[held], return_inverse=True, return_counts=True)
                weight = 1. / counts[codes]
            else:
                target = y[:, None]
                weight = np.ones(len(held))
            center = target[fit].mean(axis=0)
            def loss(cols):
                pred = center
                if cols:
                    ridge = Ridge(alpha=1., fit_intercept=False, solver="svd")
                    ridge.fit(a[:, cols], target[fit] - center)
                    pred = center + ridge.predict(b[:, cols]).reshape(len(held), -1)
                return np.dot(weight, np.square(target[held] - pred).sum(axis=1))
            null = loss([])
            for part in ([prefix, prefix[:-1]] if prefix else [[]]):
                before = loss(part)
                for j in range(X.shape[1]):
                    if j not in prefix:
                        scores[j] = min(scores[j], (before - loss(part + [j])) / null)
        pool = [j for j in range(X.shape[1]) if j not in prefix]
        prefix.extend(sorted(pool, key=lambda j: -scores[j])[:next_size-size])
        if next_size in supports:
            np.testing.assert_array_equal(supports[next_size], np.sort(prefix))
    for width in (2, 5):
        single, _ = environment_top_from_training(X, y, width, **opts)
        np.testing.assert_array_equal(supports[width], single)


def test_transform_keep_disjoint_groups_and_seed():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(50, 11))
    y = rng.normal(size=50)
    groups = np.repeat(np.arange(10), 5)
    s = PivotalSupportSelector(rule="environment", fraction=.5, task="regression")
    out = s.fit_transform(X, y, groups=groups)
    for fold in s.diagnostics_["folds"]:
        assert not set(groups[fold["fit_rows"]]) & set(groups[fold["held_rows"]])
    np.testing.assert_array_equal(out, X[:, s.get_support(indices=True)])
    with pytest.raises(InsufficientEnvironmentsError):
        s.fit(X, y, groups=np.zeros_like(groups))
    with pytest.raises(ValueError):
        environment_top_from_training(X, y, 4, task="regression", random_state=None)
