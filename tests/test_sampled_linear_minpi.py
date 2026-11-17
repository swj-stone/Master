"""Literal refits on the frozen schedule provide an independent MinPI oracle."""
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from minppi_mofs.sampled_linear_minpi import coalition_batches, rank_features


def literal(x, y, z, truth, task, seed, alpha):
    if task == "classification":
        labels = np.unique(y)
        y, truth = (y[:, None] == labels).astype(float), (truth[:, None] == labels).astype(float)
    else:
        y, truth = y[:, None], truth[:, None]
    base_loss = np.square(truth - y.mean(axis=0)).sum(axis=1).mean()

    def utility(cols):
        prediction = y.mean(axis=0)
        if len(cols):
            model = Ridge(alpha=1., solver="svd").fit(x[:, cols], y)
            prediction = model.predict(z[:, cols]).reshape(len(z), -1)
        loss = np.square(truth - prediction).sum(axis=1).mean()
        return 1. / (1. + loss / (base_loss or 1.))

    p = x.shape[1]
    quota = alpha * utility(np.arange(p))
    gains = [[] for _ in range(p)]
    margins = [[] for _ in range(p)]
    witnesses = [None] * p
    for base, group in coalition_batches(p, 4, seed):
        u = utility(base)
        for j in group:
            v = utility(np.sort(np.append(base, j)))
            margin = int(v >= quota) - int(u >= quota)
            margins[j].append(margin)
            gains[j].append(v - u)
            if margin == -1 and witnesses[j] is None:
                witnesses[j] = tuple(base)
    return np.min(margins, axis=1), np.mean(gains, axis=1), witnesses


@pytest.mark.parametrize("task,classes", [("classification", 2), ("classification", 3), ("regression", 0)])
def test_batch_matches_per_feature_independent_ridge_refits(task, classes):
    rng = np.random.default_rng(488)
    data = rng.normal(size=(62, 9))
    data[:, 0] = 2.  # Intercept and constant/duplicate columns remain well defined.
    data[:, 3] = data[:, 2]
    values = data[:, 1] + .6 * data[:, 4] + rng.normal(size=62)
    if classes:
        values = np.digitize(values, np.quantile(values, np.arange(1, classes)/classes))
    x, z, y, truth = data[:36], data[36:], values[:36], values[36:]
    expected, gain, witnesses = literal(x, y, z, truth, task, 19, .98)
    args = dict(task=task, seed=19, alpha=.98)
    actual = rank_features(x, y, z, truth, **args)
    mean = rank_features(x, y, z, truth, **args, method="mean")
    coef = rank_features(x, y, z, truth, **args, method="coefficient")
    np.testing.assert_array_equal(actual.minimum, expected)
    np.testing.assert_allclose(mean.scores, gain, atol=1e-13)
    assert actual.witness == witnesses
    np.testing.assert_allclose(actual.tie_scores, coef.scores)
    np.testing.assert_array_equal(actual.order, np.lexsort((np.arange(9), -coef.scores, -expected)))
    work = actual.work
    assert work["evaluated_marginals"] + work["absorbed_skips"] + work["losing_base_skips"] == 4 * 3 * 9
    assert work["scheduled_marginals"] == mean.work["evaluated_marginals"] == 4 * 3 * 9
    assert work["losing_base_skips"] + work["absorbed_skips"] > 0
    json.dumps(work)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 11])
def test_schedule_is_stratified_disjoint_and_complete_per_round(p):
    visits = [[] for _ in range(p)]
    for base, group in coalition_batches(p, 1, 42):
        assert not (set(base) & set(group))
        for j in group:
            visits[j].append(len(base))
    assert all(sizes == sorted(set(sizes)) for sizes in visits)
    assert all(len(sizes) == min(p, 3) for sizes in visits)


def test_full_column_at_unit_quota_and_constant_target():
    x = np.arange(12., dtype=float)[:, None]
    y = 2. * x[:, 0]
    result = rank_features(x[:8], y[:8], x[8:], y[8:], task="regression", alpha=1.)
    assert result.minimum.tolist() == [1]
    result = rank_features(x[:8], y[:8]*0, x[8:], y[8:]*0, task="regression", alpha=1.)
    assert result.minimum.tolist() == [0]
