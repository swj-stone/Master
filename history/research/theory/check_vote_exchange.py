"""Reproduce the finite pivotal game and verify a fixed-support exchange by enumeration."""

from fractions import Fraction
from itertools import combinations, product

import numpy as np

from minppi_mofs.word_minpi import exact_top_from_votes


VOTES = ((0, 1, 1, 0), (1, 0, -1, 1), (1, 1, 1, 0))
TIERS = (0, 1, 2)
SIGNS = (-1, 0, 1)


def sign(value):
    return (value > 0) - (value < 0)


def mean(values):
    values = tuple(values)
    return Fraction(sum(values), len(values))


def payoff(votes, selected):
    return mean(sign(sum(row[j] for j in selected)) for row in votes)


def layer_gain(votes, feature, size):
    peers = [j for j in range(len(votes[0])) if j != feature]
    return mean(
        payoff(votes, subset + (feature,)) - payoff(votes, subset)
        for subset in combinations(peers, size)
    )


def row_gains(peers, removed, added):
    margin = sum(peers)

    def difference(m):
        return sign(m + added) - sign(m + removed)

    full = difference(margin)
    if not peers:
        return (full,)
    # The production analogue caches these three counts for the whole support.
    # Counting inline here keeps the small rational oracle independent of it.
    leave_one = sum(peers.count(t) * difference(margin - t) for t in SIGNS)
    return (full, Fraction(leave_one, len(peers)))


def verify_conditional_formula():
    cases = 0
    for m in range(5):
        sizes = (m, m - 1) if m else (0,)
        for peers in product(SIGNS, repeat=m):
            for removed, added in product(SIGNS, repeat=2):
                direct = tuple(
                    mean(sign(sum(subset) + added) - sign(sum(subset) + removed)
                         for subset in combinations(peers, size))
                    for size in sizes
                )
                assert direct == row_gains(peers, removed, added)
                cases += 1
    return cases


def main():
    layers = [[layer_gain(VOTES, j, k) for j in range(4)] for k in TIERS]
    scores = tuple(map(min, zip(*layers)))
    assert scores == (Fraction(2, 9), Fraction(1, 9), Fraction(-1, 9), Fraction(2, 9))
    selected, _ = exact_top_from_votes(np.array(VOTES), 2, tiers=TIERS)
    assert tuple(selected) == (0, 3)
    before, after = payoff(VOTES, (0, 3)), payoff(VOTES, (0, 1))
    assert (before, after) == (Fraction(2, 3), Fraction(1))
    gains = tuple(map(mean, zip(*(row_gains((r[0],), r[3], r[1]) for r in VOTES))))
    assert gains == (Fraction(1, 3), Fraction(1, 3))
    checked = verify_conditional_formula()
    print('T(A,B,C,D) =', ', '.join(map(str, scores)))
    print(f'Payoff AD -> AB: {before} -> {after}')
    print('Full and leave-one expected exchange gains:', ', '.join(map(str, gains)))
    print(f'{checked} row configurations matched direct enumeration.')


if __name__ == '__main__':
    main()

