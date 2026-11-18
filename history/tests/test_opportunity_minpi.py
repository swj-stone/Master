from fractions import Fraction
from itertools import combinations

import numpy as np

from experiments.minpi_opportunity_pilot import opportunity_top


def test_opportunity_matches_direct_coalitions_and_excludes_zero_layers():
    rng = np.random.default_rng(6068)
    for p in (3, 6, 8):
        for _ in range(4):
            votes = rng.integers(-1, 2, (5, p), dtype=np.int8)
            tiers = (0, 1, p // 2, p - 1)
            means = []
            for k in tiers:
                total = [0] * p
                abs_mass = 0
                for j in range(p):
                    for s in combinations([i for i in range(p) if i != j], k):
                        base = votes[:, s].sum(axis=1)
                        gain = np.sign(base + votes[:, j]) - np.sign(base)
                        total[j] += int(gain.sum())
                        abs_mass += int(np.abs(gain).sum())
                if abs_mass:
                    means.append([Fraction(p * x, abs_mass) for x in total])
            scores = [min(layer[j] for layer in means) for j in range(p)]
            order = sorted(range(p), key=lambda j: (-scores[j], j))
            for h in (1, 2, p):
                np.testing.assert_array_equal(opportunity_top(votes, h, tiers), sorted(order[:h]))

    sidelined = np.array([[-1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(opportunity_top(sidelined, 1, (0, 4)), [1])
    np.testing.assert_array_equal(opportunity_top(sidelined, 2, (4,)), [0, 1])
    np.testing.assert_array_equal(opportunity_top(np.zeros((4, 5)), 3), [0, 1, 2])
