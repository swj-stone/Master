from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from experiments import minpi_study_proposal as proposal
from experiments.audit_minpi_study_exact_targets import enumerated_layers
from experiments.minpi_study_frontier import stress_case
from experiments.minpi_study_precision_design import direct_layer, direct_minimum


def test_small_games_against_indexed_subsets():
    rng = np.random.default_rng(1917)
    for p in (1, 4, 7):
        votes = rng.integers(-1, 2, size=(6, p), dtype=np.int8)
        votes[:, -1] = votes[:, 0]
        for tiers in ((p - 1,), tuple(range(p))):
            scores = enumerated_layers(votes, tiers).min(axis=0)
            order = sorted(range(p), key=lambda j: (-scores[j], j))
            for h in range(1, p + 1):
                got, _ = proposal.verified_float_top(votes, np.uint64(h), tiers=tiers)
                np.testing.assert_array_equal(got, sorted(order[:h]))


def test_outsider_checks_even_with_a_bad_proposal(monkeypatch):
    votes = np.array([[1, -1, 1, 0, -1]], dtype=np.int8)
    monkeypatch.setattr(proposal, "top_from_votes", lambda *args, **kwargs: (np.array([1, 4]), {}))
    got, info = proposal.verified_float_top(votes, 2, tiers=(0,))
    np.testing.assert_array_equal(got, [0, 2])
    assert info["arithmetic"] == "int64"
    assert not info["proposal_accepted"] and info["proposal_swaps"] == 2
    assert info["exact_columns"] == 0


def test_strict_bound_and_full_input_validation():
    votes = np.array([[1, 0, -1]], dtype=np.int8)
    got, info = proposal.verified_float_top(votes, 1, tiers=(0,))
    np.testing.assert_array_equal(got, [0])
    assert info["proposal_accepted"] and info["proposal_swaps"] == 0
    with pytest.raises(ValueError):
        proposal.verified_float_top(votes, 3, tiers=(3,))


def test_archived_gas_tie():
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "results/minpi-study/new-cohorts/gas-1129.npz") as saved:
        votes = saved["train_votes"]
    tiers = (0, 1, 4, 16)
    order, denom, values = direct_minimum({k: direct_layer(votes, k) for k in tiers}, tiers)
    assert Fraction(values[order[63]], len(votes) * denom) == Fraction(-525, 32512)
    assert values[45] == values[115] == values[order[63]]
    got, info = proposal.verified_float_top(votes, 64, tiers=tiers)
    np.testing.assert_array_equal(got, np.sort(order[:64]))
    assert 45 in got and 115 not in got
    assert not info["proposal_accepted"]
    floating, _ = proposal.top_from_votes(votes, 64, tiers=tiers)
    assert info["proposal_swaps"] == len(set(got) - set(floating))


@pytest.mark.parametrize("unique, expected_boundary", [(False, 2048), (True, 2)])
def test_sub_float_resolution_and_wide_tie(unique, expected_boundary):
    votes = stress_case(4096, "near")
    if unique:
        votes[1] = -1
        votes[1, :2] = votes[1, 2048:-2] = 1
    tiers = (0, 4, 16)
    order, _, scores = direct_minimum({k: direct_layer(votes, k) for k in tiers}, tiers)
    assert (scores[order[0]] > scores[order[1]]) == unique
    got, info = proposal.verified_float_top(votes, 1, tiers=tiers)
    np.testing.assert_array_equal(got, order[:1])
    assert info["arithmetic"] == "dyadic"
    assert not info["proposal_accepted"]
    assert info["exact_columns"] == expected_boundary


def test_silent_and_active_columns_on_the_same_exact_boundary(monkeypatch):
    from minppi_mofs import integer_minpi

    row = np.r_[np.ones(62), -np.ones(62), np.zeros(4)].astype(np.int8)
    row[[0, -1]] = row[[-1, 0]]
    votes = np.stack([row, -row])
    score_columns = integer_minpi._rational_scores
    widths = []

    def trace(matrix, layers):
        assert np.any(matrix, axis=0).all()
        widths.append(matrix.shape[1])
        return score_columns(matrix, layers)

    monkeypatch.setattr(integer_minpi, "_rational_scores", trace)
    for select in (integer_minpi.rational_top_from_votes, proposal.verified_float_top):
        widths.clear()
        selected, info = select(votes, 3, tiers=(4, 16))
        # Opposite rows cancel exactly at every size, including for the silent
        # column at index 0. Only non-silent columns need exact accumulation.
        np.testing.assert_array_equal(selected, [0, 1, 2])
        assert widths == [124] and info["exact_columns"] == 124
