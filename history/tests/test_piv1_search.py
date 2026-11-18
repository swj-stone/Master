import numpy as np
import pytest

from minppi_mofs.piv1_search import search_subsets


def test_prefix_search_keeps_full_reference_and_never_repeats_masks():
    n = 56
    rng = np.random.default_rng(1)
    calls = []

    def evaluate(indices):
        assert indices == tuple(sorted(set(indices)))
        assert indices not in calls
        calls.append(indices)
        return (len(indices) / n, 0., 0.)

    result = search_subsets(None, rng.normal(size=n), rng.normal(size=n), evaluate, recipe="anchor")
    assert result["selected"] == tuple(range(n))
    assert result["utility"] == 1
    assert len(calls) == result["budget"] == 18
    assert len(result["trials"]) == len(calls)


def test_min_votes_change_candidates_and_determine_strong_search_budget():
    n = 100
    strength = np.linspace(1., 0., n)
    votes = np.r_[-np.ones(n // 2), np.ones(n // 2)]
    def evaluate(indices):
        return len(indices) / n

    weak = search_subsets(votes, strength, strength, evaluate, recipe="vote")
    strong = search_subsets(votes, strength, strength, evaluate, recipe="vote_strong")
    control = search_subsets(None, strength, strength, evaluate, recipe="anchor")
    assert strong["negative_count"] == 50
    assert strong["base_budget"] == weak["budget"] == control["budget"] == 24
    assert strong["budget"] == 18
    assert len(strong["trials"]) == 18
    assert {tuple(t["indices"]) for t in weak["trials"]} != {
        tuple(t["indices"]) for t in control["trials"]
    }
    assert strong["selected"] == tuple(range(n))  # hard-excluding the negative half would fail
    consistent = np.arange(n) % 5 == 0
    repeatable = search_subsets(votes, strength, strength, evaluate, recipe="vote_strong",
                                consistent_negative=consistent)
    assert repeatable["negative_count"] == 10
    assert repeatable["budget"] == 23


def test_ties_choose_a_small_mask_and_mismatched_evidence_is_rejected():
    strengths = np.arange(8)
    result = search_subsets(np.zeros(8), strengths, strengths, lambda _: .5)
    assert len(result["selected"]) == min(len(t["indices"]) for t in result["trials"])
    with pytest.raises(ValueError):
        search_subsets(np.zeros(7), strengths, strengths, lambda _: .5)
