"""Enumerated coverage games and width probes, independent of the selector's formulas."""
from fractions import Fraction
from itertools import combinations
from math import ceil, comb, log2

import numpy as np

from experiments.split_vote.path import coverage_width, select_width
from minppi_mofs.split_vote import pivotal_order


def test_single_voter_conditioned_sizes():
    for n in range(2, 10):
        others = set(range(1, n))
        for degree in range(1, n + 1):
            substitutes = set(range(1, degree))
            for k in range(n):
                panels = tuple(combinations(others, k))
                misses = sum(substitutes.isdisjoint(panel) for panel in panels)
                assert Fraction(misses, len(panels)) == Fraction(comb(n-degree, k), comb(n-1, k))


def test_normalization_changes_first_choice():
    covers = np.array([[1,0,0,0,0], [0,1,1,0,0], [0,0,0,1,0], [0,0,0,0,1]], bool)
    weights = np.array([5,11,1,1])
    gain = np.ones(5)
    for rule, winner in (("min",2), ("pooled",0)):
        order = pivotal_order(covers, weights, gain, rule)
        assert order[0] == winner
        assert sorted(order.tolist()) == list(range(5))


def test_bounded_width_schedules():
    rng = np.random.default_rng(67001)
    for p in range(1, 141):
        order = rng.permutation(p)
        utility = rng.normal(size=p)
        for mass in (None, np.zeros(p), rng.uniform(size=p)):
            seen = []
            def evaluate(cols):
                k = len(cols)
                assert cols == tuple(sorted(order[:k]))
                assert k not in seen
                seen.append(k)
                return utility[k-1]
            if mass is None:
                width, scores = select_width(order, evaluate)
                bound = ceil(log2(p)) + 3
            else:
                width, scores = coverage_width(order, mass, evaluate)
                bound = 6
            assert set(scores) == set(seen)
            assert width == min(k for k in seen if utility[k-1] >= max(scores.values())-.005-1e-14)
            assert p in seen and len(seen) <= min(p, bound)
