import itertools

import numpy as np

from experiments.minpi_study_dynamic_anchor import METHODS, select
from experiments.minpi_study_precision_design import direct_layer, direct_minimum


def check_all(votes, tiers, budgets):
    exact, _, _ = direct_minimum({k: direct_layer(votes, k) for k in tiers}, tiers)
    for h in budgets:
        counts = {}
        for method in METHODS:
            support, counts[method], _ = select(votes, h, tiers=tiers, method=method)
            np.testing.assert_array_equal(support, np.sort(exact[:h]))
        assert counts["dynamic_stream"] <= counts["static_stream"] == counts["static_batch"]


def test_anchor_and_index_ties_on_every_three_column_row():
    for row in itertools.product([-1.0, 0.0, 1.0], repeat=3):
        votes = np.array([row])
        check_all(votes, (0, 2), (1, 2, 3))


def test_streaming_arbitrary_precision_and_ternary_weights():
    votes = np.random.default_rng(17).integers(-1, 2, size=(4, 128), dtype=np.int8)
    votes[:, 25] = votes[:, 24]
    votes[:, 6:12] = 0
    check_all(votes, (0, 4, 16), (1, 13, 64, 127))
    assert select(votes, 13, tiers=(0, 4, 16), method="dynamic_stream")[2] == "python_int"


def test_equal_anchor_prefers_original_index_and_needs_no_extra_columns():
    votes = np.zeros((2, 9), dtype=bool)
    for method in METHODS:
        support, completed, _ = select(votes, 4, tiers=(0, 2), method=method)
        np.testing.assert_array_equal(support, np.arange(4))
        assert completed == 4


def test_updating_heap_cutoff_skips_a_column_eligible_at_the_initial_cutoff():
    votes = np.array([[-1, 0, -1, -1, -1, -1], [0, 0, -1, 1, 0, 1],
                      [1, 0, 1, 0, 0, -1], [1, 1, -1, 0, -1, -1],
                      [0, 0, -1, 0, 1, 1]], dtype=np.int8)
    for method in METHODS:
        support, completed, _ = select(votes, 2, tiers=(0, 2), method=method)
        np.testing.assert_array_equal(support, [0, 1])
        assert completed == (3 if method == "dynamic_stream" else 4)
