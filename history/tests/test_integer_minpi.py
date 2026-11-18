from fractions import Fraction
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.exact_minpi import ExactMinPI, pivotal_rates, witness_margins
from minppi_mofs.integer_minpi import rational_top_from_votes
from minppi_mofs.support_selector import PivotalSupportSelector


def direct_rates(votes, tiers):
    w, p = votes.shape
    scores = []
    for j in range(p):
        sizes = []
        for k in tiers:
            total = count = 0
            for members in combinations([i for i in range(p) if i != j], k):
                base = votes[:, members].sum(1)
                total += int((np.sign(base+votes[:, j])-np.sign(base)).sum())
                count += 1
            sizes.append(Fraction(total, count*w))
        scores.append(sizes)
    return np.asarray(scores, dtype=object).T


def direct_order(votes):
    p = votes.shape[1]
    tiers = sorted({min(t, p-1) for t in (0, 4, 16)})
    scores = direct_rates(votes, tiers).min(axis=0)
    return sorted(range(p), key=lambda j: -scores[j])


def test_integer_certificate_against_direct_enumeration():
    rng = np.random.default_rng(4402)
    for p in (1, 2, 5, 7, 10):
        for _ in range(6):
            votes = rng.integers(-1, 2, (8, p), dtype=np.int8)
            votes[:, -1] = votes[:, 0]
            order = direct_order(votes)
            for h in {1, max(1, p//3), p}:
                for bits in (0, 3, 40):
                    actual, _ = rational_top_from_votes(votes, h, bits=bits)
                    np.testing.assert_array_equal(actual, sorted(order[:h]))
                np.testing.assert_array_equal(rational_top_from_votes(votes, h, prune=False)[0], sorted(order[:h]))


@pytest.mark.parametrize("unique", [True, False])
def test_sub_ulp_boundary(unique):
    p = 4096
    balanced = np.where(np.arange(p) < p//2, 1, -1)
    skew = np.ones(p)
    skew[0] = -1
    skew[-7:] = -1
    second = balanced.copy()
    if unique:
        second[:] = -1
        second[:2] = 1
        second[p//2:-2] = 1
    votes = np.stack([balanced, second, skew]).astype(np.int8)
    expected, _ = rational_top_from_votes(votes, 1, prune=False)
    selected, info = rational_top_from_votes(votes, 1)
    np.testing.assert_array_equal(expected, [1])
    np.testing.assert_array_equal(selected, expected)
    assert info["exact_columns"] == (2 if unique else p//2)


@pytest.mark.parametrize("arithmetic", ["float", "rational"])
def test_training_transformer(arithmetic):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.iloc[:180].copy()
    X.iloc[0, 0] = np.nan
    y = y.iloc[:180]
    estimator = PivotalSupportSelector(fraction=.1, arithmetic=arithmetic).fit(X, y)
    votes = np.sign(witness_margins(X, y, "classification", 42)).astype(np.int8)
    oracle = rational_top_from_votes if arithmetic == "rational" else top_from_votes
    expected = oracle(votes, 3, prune=False)[0]
    np.testing.assert_array_equal(estimator.get_support(indices=True), expected)
    np.testing.assert_array_equal(estimator.transform(X), X.iloc[:, expected].to_numpy())
    with pytest.raises(ValueError):
        estimator.transform(X.iloc[:, ::-1])
    with pytest.raises(ValueError):
        estimator.fit(pd.concat([X, X.iloc[:, :1]], axis=1), y)


@pytest.mark.parametrize("tiers", [(0,), (1,), (3,), (6,), (1, 3, 5), (4, 0, 1, 4)])
def test_custom_layers_against_uniform_peer_subsets(tiers):
    rng = np.random.default_rng(270313)
    for _ in range(3):
        votes = rng.integers(-1, 2, size=(8, 7), dtype=np.int8)
        votes[:, -1] = votes[:, 0]
        exact = direct_rates(votes, tiers)
        np.testing.assert_allclose(pivotal_rates(votes, tiers), exact.astype(float), atol=1e-14)
        order = sorted(range(7), key=exact.min(axis=0).__getitem__, reverse=True)
        for h in (1, 3, 7):
            for bits in (0, 8, 40):
                got, _ = rational_top_from_votes(votes, h, tiers=tiers, bits=bits)
                assert got.tolist() == sorted(order[:h])
            approx, _ = top_from_votes(votes, h, tiers=tiers)
            full_float, _ = top_from_votes(votes, h, tiers=tiers, prune=False)
            np.testing.assert_array_equal(approx, full_float)


@pytest.mark.parametrize("tiers", [[], True, [np.bool_(True)], [1.0], 3.5, [-1], [4], [[0]], "3"])
def test_invalid_explicit_layers_even_when_all_columns_requested(tiers):
    votes = np.ones((2, 4), dtype=np.int8)
    for select in [top_from_votes, rational_top_from_votes]:
        with pytest.raises(ValueError, match="Peer counts"):
            select(votes, 4, tiers=tiers)


def test_custom_layers_in_training_estimators():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:180], y[:180]
    tiers = [np.int64(3), 1, 3]
    votes = np.sign(witness_margins(X, y, "classification", 42)).astype(np.int8)
    for arithmetic, routine in [("float", top_from_votes), ("rational", rational_top_from_votes)]:
        est = PivotalSupportSelector(tiers=tiers, arithmetic=arithmetic).fit(X, y)
        np.testing.assert_array_equal(est.get_support(indices=True), routine(votes, 3, tiers=(1, 3))[0])
    full = ExactMinPI(tiers=tiers).fit(X, y)
    np.testing.assert_array_equal(full.tier_sizes_, [1, 3])
    np.testing.assert_allclose(full.tier_rates_, pivotal_rates(votes, (1, 3)))
    for select in [top_from_votes, rational_top_from_votes]:
        np.testing.assert_array_equal(select(votes, 3, tiers=3)[0], select(votes, 3, tiers=(3,))[0])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint64])
def test_unsigned_budget_at_a_partition_and_fallback(dtype):
    votes = np.array([[1, 1, -1, -1]], dtype=np.int8)
    expected, _ = rational_top_from_votes(votes, 1, tiers=(1,), prune=False)
    got, info = rational_top_from_votes(votes, dtype(1), tiers=(1,), bits=0)
    np.testing.assert_array_equal(got, expected)
    assert info["exact_columns"] >= 2
    np.testing.assert_array_equal(top_from_votes(votes, dtype(1), tiers=(1,))[0], expected)


def test_default_layers_for_older_estimator_state():
    est = ExactMinPI()
    del est.tiers
    assert est.get_params()["tiers"] is None
    X, y = load_breast_cancer(return_X_y=True)
    np.testing.assert_array_equal(est.fit(X[:180], y[:180]).tier_sizes_, [0, 4, 16])


def test_silent_peers_remain_in_uniform_subsets():
    rng = np.random.default_rng(1804)
    for _ in range(8):
        votes = rng.integers(-1, 2, size=(4, 7), dtype=np.int8)
        votes[:, [0, 3]] = 0
        tiers = (2, 5)
        order = np.argsort(-direct_rates(votes, tiers).min(axis=0), kind="stable")
        for count in (1, 3, 6):
            selected, _ = rational_top_from_votes(votes, count, tiers=tiers, bits=0)
            np.testing.assert_array_equal(selected, np.sort(order[:count]))
