from fractions import Fraction
from itertools import combinations
from math import comb

import numpy as np
import pytest

from minppi_mofs.conditional_minpi import conditional_top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes


def enumerate_order(votes, count, tiers):
    """Enumerate indexed peer combinations, with selected votes fixed in every game."""
    rows, width = votes.shape
    selected = []
    for _ in range(count):
        candidates = [j for j in range(width) if j not in selected]
        fixed = votes[:, selected].sum(axis=1)
        scores = {}
        for j in candidates:
            peers = [i for i in candidates if i != j]
            layer_scores = []
            for k in sorted({min(k, len(peers)) for k in tiers}):
                total = 0
                for coalition in combinations(peers, k):
                    before = fixed + votes[:, coalition].sum(axis=1)
                    total += int((np.sign(before + votes[:, j]) - np.sign(before)).sum())
                layer_scores.append(Fraction(total, rows * comb(len(peers), k)))
            scores[j] = min(layer_scores)
        selected.append(min(candidates, key=lambda j: (-scores[j], j)))
    return selected


def test_redundant_column_loses_its_contribution_once_its_twin_is_selected():
    votes = np.array([[1, 1, 0]] * 3 + [[0, 0, 1]], dtype=np.int8)
    tiers = (0, 1)
    static, _ = exact_top_from_votes(votes, 2, tiers=tiers)
    np.testing.assert_array_equal(static, [0, 1])
    chosen, info = conditional_top_from_votes(votes, 2, tiers=tiers)
    assert list(info["selection_order"]) == [0, 2]
    np.testing.assert_array_equal(chosen, [0, 2])


@pytest.mark.parametrize("width,tiers", [(1, None), (4, None), (6, (2, 5)), (7, (0, 1, 4))])
def test_conditional_order_and_all_prefixes_match_exhaustive_games(width, tiers):
    rng = np.random.default_rng(74031)
    for _ in range(4):
        votes = rng.integers(-1, 2, (8, width), dtype=np.int8)
        expected = enumerate_order(votes, width, (0, 4, 16) if tiers is None else tiers)
        for count in sorted({1, max(1, width // 2), width}):
            chosen, info = conditional_top_from_votes(votes, count, tiers=tiers)
            assert list(info["selection_order"]) == expected[:count]
            np.testing.assert_array_equal(chosen, sorted(expected[:count]))


def test_all_abstentions_follow_input_order_through_last_selection():
    chosen, info = conditional_top_from_votes(np.zeros((3, 6), dtype=np.int8), 6)
    assert list(info["selection_order"]) == list(range(6))
    np.testing.assert_array_equal(chosen, np.arange(6))
