from itertools import combinations, combinations_with_replacement, product

import numpy as np

from experiments import minpi_study_tie_keys as study
from experiments.audit_minpi_study_exact_targets import enumerated_layers


def test_endpoint_order_against_every_completion_of_small_intervals():
    # Include equal endpoints, positive-width ties, and endpoint signs on both sides of zero.
    intervals = list(combinations_with_replacement((-1, 0, 1), 2))
    for bounds in product(intervals, repeat=3):
        lower, upper = np.array(bounds).T
        completions = list(product(*(range(a, b+1) for a, b in bounds)))
        for size in (1, 2):
            all_supports = {tuple(sorted(sorted(range(3), key=lambda j: (-x[j], j))[:size]))
                            for x in completions}
            for proposed in combinations(range(3), size):
                cols = np.array(proposed)
                others = np.setdiff1d(np.arange(3), cols)
                result = study.endpoint_keys_pass(cols, others, lower, upper, lower[cols].min())
                assert result == (all_supports == {proposed})


def test_binary64_proposal_on_indexed_subsets():
    rng = np.random.default_rng(73139)
    for p in (1, 2, 4, 6):
        for _ in range(3):
            votes = rng.integers(-1, 2, size=(4, p), dtype=np.int8)
            if p > 1:
                votes[:, -1] = votes[:, 0]
            for tiers in ((p-1,), tuple(range(p))):
                scores = enumerated_layers(votes, tiers).min(axis=0)
                order = sorted(range(p), key=lambda j: (-scores[j], j))
                for h in range(1, p+1):
                    selected, _ = study.verified_float_top_keys(votes, h, tiers=tiers)
                    np.testing.assert_array_equal(selected, sorted(order[:h]))


def test_equal_bounds_accept_only_proposal_favoured_by_original_index(monkeypatch):
    votes = np.array([[1, 1, -1]], dtype=np.int8)
    cols, accepted = study.verified_float_top_keys(votes, 1, tiers=(0,))
    np.testing.assert_array_equal(cols, [0])
    assert accepted["proposal_accepted"]
    monkeypatch.setattr(study, "top_from_votes", lambda *args, **kwargs: (np.array([1]), {}))
    cols, corrected = study.verified_float_top_keys(votes, 1, tiers=(0,))
    np.testing.assert_array_equal(cols, [0])
    assert not corrected["proposal_accepted"] and corrected["proposal_swaps"] == 1


def test_index_certificate_rejects_both_replayed_errors():
    for name, votes, tiers, h in study.correction_inputs():
        expected, _ = study.oracle(votes, tiers, h)
        selected, info = study.verified_float_top_keys(votes, h, tiers=tiers)
        np.testing.assert_array_equal(expected, selected)
        assert not info["proposal_accepted"] and info["proposal_swaps"] == 1
        assert info["exact_columns"] == (3 if name == "gas-1129" else 2)
