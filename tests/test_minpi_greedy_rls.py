import numpy as np
import pytest
from sklearn.linear_model import Ridge

from experiments.minpi_compression.greedy_rls import greedy_ranking


@pytest.mark.parametrize("outputs", [1, 3])
def test_greedy_ranking_matches_explicit_leave_one_out_fits(outputs):
    rng = np.random.default_rng(6)
    x = rng.normal(size=(16, 6))
    x[:, -1] = 4.
    y = rng.normal(size=(len(x), outputs))
    order = []
    remaining = list(range(x.shape[1]))
    while remaining:
        errors = []
        for candidate in remaining:
            cols = order + [candidate]
            errors_i = []
            for test in range(len(x)):
                train = np.arange(len(x)) != test
                fit = Ridge(alpha=1.).fit(x[train][:, cols], y[train])
                errors_i.append(np.sum((fit.predict(x[test:test+1,cols]) - y[test])**2))
            errors.append(np.mean(errors_i))
        order.append(remaining.pop(np.argmin(errors)))
    np.testing.assert_array_equal(greedy_ranking(x, y), order)
