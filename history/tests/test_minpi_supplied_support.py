from itertools import combinations, product

import numpy as np
import pytest

from experiments import minpi_study_tie_keys as study
from experiments.audit_minpi_study_exact_targets import enumerated_layers
from experiments.minpi_study_boundary_cost import correction_inputs, oracle


def test_audit_accepts_unsorted_or_corrects_arbitrary_supplied_sets_without_generating(monkeypatch):
    def disallow_scoring(*args, **kwargs):
        raise AssertionError("Auditing a supplied list must not generate a new floating proposal")

    monkeypatch.setattr(study, "top_from_votes", disallow_scoring)
    votes = np.array([[1, 1, 0, -1], [1, -1, 1, -1], [0, 1, -1, -1]], dtype=np.int8)
    for tiers in ((0,), (1, 3), (0, 1, 2, 3)):
        scores = enumerated_layers(votes, tiers).min(axis=0)
        order = sorted(range(4), key=lambda j: (-scores[j], j))
        for count in range(1, 5):
            for proposal in combinations(range(4), count):
                expected = sorted(order[:count])
                selected, info = study.verify_supplied_top_keys(votes, count, proposal[::-1], tiers=tiers)
                np.testing.assert_array_equal(selected, expected)
                assert info["proposal_swaps"] == len(set(proposal).difference(expected))
                assert not info["proposal_accepted"] or sorted(proposal) == expected


def test_every_proposal_on_all_single_row_ternary_matrices_of_width_three():
    for row in product((-1, 0, 1), repeat=3):
        votes = np.asarray([row], dtype=np.int8)
        tiers = (0, 1, 2)
        scores = enumerated_layers(votes, tiers).min(axis=0)
        order = sorted(range(3), key=lambda j: (-scores[j], j))
        for size in range(1, 4):
            for supplied in combinations(range(3), size):
                got, _ = study.verify_supplied_top_keys(votes, size, supplied[::-1], tiers=tiers)
                np.testing.assert_array_equal(got, sorted(order[:size]))


def test_correct_supplied_set_can_require_arbitrary_precision():
    row = np.repeat([1, -1], 64).astype(np.int8)
    votes = np.stack([row, -row])
    got, info = study.verify_supplied_top_keys(votes, 1, [0], tiers=(0, 4, 16))
    np.testing.assert_array_equal(got, [0])
    assert not info["proposal_accepted"] and info["proposal_swaps"] == 0
    assert info["exact_columns"] == 128


@pytest.mark.parametrize("proposal", [[0], [0, 1, 2], [[0, 1]], [True, False], [0., 1.],
                                       [0, 0], [0, -1], [0, 3], np.array([0, 2**64-1], dtype=np.uint64)])
def test_audit_rejects_malformed_supplied_indices(proposal):
    with pytest.raises(ValueError):
        study.verify_supplied_top_keys([[1, -1, 0]], 2, proposal)


def test_audit_checks_full_budget_before_accepting_and_resolves_dyadic_cases():
    with pytest.raises(ValueError):
        study.verify_supplied_top_keys([[1, -1]], 2, [0, 0])
    for name, votes, tiers, count in correction_inputs():
        truth, _ = oracle(votes, tiers, count)
        erroneous = truth.copy()
        if name == "gas-1129":
            erroneous[erroneous == 45] = 115
        else:
            erroneous[:] = 0
        selected, info = study.verify_supplied_top_keys(votes, count, erroneous[::-1], tiers=tiers)
        np.testing.assert_array_equal(selected, truth)
        assert not info["proposal_accepted"] and info["proposal_swaps"] == 1
        assert info["exact_columns"] == (3 if name == "gas-1129" else 2)
