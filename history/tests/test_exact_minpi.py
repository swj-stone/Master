from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from minppi_mofs.exact_minpi import ExactMinPI, pivotal_rates


def enumeration(votes, k):
    n, p = votes.shape
    rates = []
    for j in range(p):
        seen = []
        for coalition in combinations([i for i in range(p) if i != j], k):
            total = votes[:, list(coalition)].sum(axis=1)
            seen.extend(np.sign(total + votes[:, j]) - np.sign(total))
        rates.append(np.mean(seen))
    return rates


def test_pivotal_rates_equal_exhaustive_with_zeros_and_boundaries():
    rng = np.random.default_rng(7)
    for p in range(1, 8):
        votes = rng.integers(-1, 2, size=(11, p), dtype=np.int8)
        votes[0, :] = 0
        votes[1, :] = -1
        votes[2, :] = 1
        actual = pivotal_rates(votes, range(p))
        for k in range(p):
            np.testing.assert_allclose(actual[k], enumeration(votes, k), atol=1e-12)


def test_selector_deterministic_and_preserves_schema():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(142, 30)), columns=[f'f{i}' for i in range(30)])
    y = (frame.f3 > 0).astype(int)
    frame.loc[:4, 'f12'] = np.nan
    selector = ExactMinPI().fit(frame, y)
    assert selector.n_features_in_ == 30
    assert selector.transform(frame).shape == (142, 3)
    assert selector.ranking_[0] == 3
    np.testing.assert_equal(selector.scores_, selector.tier_rates_.min(axis=0))
    np.testing.assert_equal(selector.ranking_, ExactMinPI().fit(frame, y).ranking_)
    with pytest.raises(ValueError):
        selector.transform(frame.iloc[:, ::-1])


def test_regression_and_small_class_errors():
    x = np.arange(100, dtype=float).reshape(-1, 1)
    selector = ExactMinPI(task='regression').fit(x, x.ravel()**2)
    assert list(selector.tier_sizes_) == [0]
    assert selector.transform(x).shape == x.shape
    with pytest.raises(ValueError):
        ExactMinPI().fit(x, np.ones(len(x)))

