"""Independent finite-game audit by direct subset enumeration and exact rational comparisons."""
from fractions import Fraction
from itertools import combinations, product
from math import comb, lcm
from pathlib import Path

import numpy as np

from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.integer_minpi import rational_top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale

ROOT = Path(__file__).resolve().parents[1]


def enumerated_layers(votes, tiers):
    """Enumerate actual indexed subsets, including distinct indices with identical ballots."""
    rows, width = votes.shape
    result = []
    for k in tiers:
        layer = []
        for j in range(width):
            total = 0
            for subset in combinations([i for i in range(width) if i != j], k):
                margin = votes[:, subset].sum(axis=1)
                total += int((np.sign(margin + votes[:, j]) - np.sign(margin)).sum())
            layer.append(Fraction(total, rows * comb(width - 1, k)))
        result.append(layer)
    return np.array(result, dtype=object)


def check_budgets(votes, tiers, layers):
    scores = layers.min(axis=0)
    order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
    for h in range(1, len(scores) + 1):
        expected = sorted(order[:h])
        for backend in (rational_top_from_votes, exact_top_from_votes):
            selected, _ = backend(votes, np.uint64(h), tiers=tiers)
            np.testing.assert_array_equal(selected, expected)
    return 2 * len(scores)


def check_small_games():
    comparisons = 0
    all_tiers = (0, 1, 2)
    for entries in product((-1, 0, 1), repeat=6):
        votes = np.array(entries, dtype=np.int8).reshape(2, 3)
        layers = enumerated_layers(votes, all_tiers)
        for size in (1, 2, 3):
            for tiers in combinations(all_tiers, size):
                comparisons += check_budgets(votes, tiers, layers[list(tiers)])
    rng = np.random.default_rng(22535)
    for width in (4, 7, 9):
        for k in range(width):
            votes = rng.integers(-1, 2, size=(6, width), dtype=np.int8)
            tiers = tuple(sorted({0, 1, k, width - 1}))
            comparisons += check_budgets(votes, tiers, enumerated_layers(votes, tiers))
    return comparisons


def check_word_thresholds():
    limit = 2**63 - 1
    count = 0
    for width in range(6, 100):
        for tiers in ((0, width//2), (2, width//3, width//2)):
            common = lcm(*(comb(width - 1, k) for k in tiers))
            if common <= limit:
                last_safe = limit // common
                assert native_scale(width, tiers, last_safe) == common
                assert native_scale(width, tiers, last_safe + 1) == 0
                count += 1
    return count


def check_gas_tie():
    """Sum over the number of positive peers directly on the published binary gas ballot."""
    with np.load(ROOT / "results/minpi-study/new-cohorts/gas-1129.npz") as record:
        votes = record["train_votes"]
    if not np.isin(votes, (-1, 1)).all():
        raise ValueError("The archived calculation requires binary gas votes")
    rows, width = votes.shape
    tiers = (0, 1, 4, 16)
    layers = []
    for k in tiers:
        totals = np.zeros(width, dtype=object)
        for row in votes:
            positives = int(np.count_nonzero(row > 0))
            for sign in (-1, 1):
                active = row == sign
                if not active.any():
                    continue
                plus = positives - int(sign == 1)
                minus = width - 1 - plus
                pivot = 0
                for i in range(k + 1):
                    margin = 2*i - k
                    change = int(np.sign(margin + sign) - np.sign(margin))
                    pivot += comb(plus, i) * comb(minus, k - i) * change
                totals[active] += pivot
        layers.append([Fraction(n, rows * comb(width - 1, k)) for n in totals])
    scores = np.min(layers, axis=0)
    order = sorted(range(width), key=lambda j: (-scores[j], j))
    h = 64
    cutoff = scores[order[h - 1]]
    assert scores[45] == scores[115] == cutoff == Fraction(-525, 32512)
    expected = sorted(order[:h])
    np.testing.assert_array_equal(exact_top_from_votes(votes, h, tiers=tiers)[0], expected)
    for prune in (False, True):
        floating, _ = top_from_votes(votes, h, tiers=tiers, prune=prune)
        assert set(expected) - set(floating) == {45}
        assert set(floating) - set(expected) == {115}
    return cutoff


if __name__ == "__main__":
    print(f"{check_small_games()} supports agreed with direct indexed-subset enumeration.")
    print(f"{check_word_thresholds()} native-word thresholds passed checks on both sides.")
    print(f"Gas columns 45 and 115 tied at {check_gas_tie()}. Both floating runs selected 115.")

