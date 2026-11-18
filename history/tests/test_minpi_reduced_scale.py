from fractions import Fraction
from math import lcm
from unittest.mock import patch

import numpy as np

from experiments import minpi_study_reduced_word as reduced
from experiments.audit_minpi_study_exact_targets import enumerated_layers
from experiments.minpi_study_frontier import stress_case
from minppi_mofs.word_minpi import exact_top_from_votes


def test_forced_reduction_matches_enumerated_subsets_at_every_budget():
    rows = np.array([[1, 0, -1, 1, 0, 1], [-1, 0, 1, -1, 0, -1],
                     [0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0]], dtype=np.int8)
    for ks in ((0,), (4, 1, 0), (5, 3, 2, 1, 0)):
        by_row = [enumerated_layers(row[None, :], ks) for row in rows]
        expected = sum(by_row) / len(rows)
        order = sorted(range(rows.shape[1]), key=lambda j: (-min(expected[:, j]), j))
        required = lcm(*(value.denominator for layer in by_row for value in layer.flat))
        weights = reduced.reduced_weights(rows, ks, required)
        assert weights is not None and weights[0] == required
        for weight in weights[1]:
            assert weight.dtype == np.int64
        if required > 1:
            assert reduced.reduced_weights(rows, ks, required - 1) is None
        for i, weight in enumerate(weights[1]):
            sums = reduced.word_totals(rows, weight)
            assert sums.dtype == np.int64
            assert [Fraction(n, len(rows) * required) for n in sums.tolist()] == expected[i].tolist()
        for h in range(1, 7):
            selected, _ = reduced.reduced_support(rows, h, weights[1])
            assert selected.tolist() == sorted(order[:h])


def test_complement_reuses_native_counts_for_both_signs_and_strict_falls_back():
    half = np.r_[np.ones(64, dtype=np.int8), -np.ones(64, dtype=np.int8)]
    matrix = np.stack((half, -half))
    assert reduced.reduced_weights(matrix, (16, 4, 0), reduced.WORD_MAX // 2)[0] == 316333711375
    with patch.object(reduced, "rational_top_from_votes", side_effect=AssertionError), \
         patch.object(reduced, "word_totals", wraps=reduced.word_totals) as native:
        chosen, info = reduced.reduced_top_from_votes(matrix, 1)
        assert native.call_count > 0
        assert info["arithmetic"] == "reduced_int64" and chosen.tolist() == [0]
    tied = stress_case(4096, "near")
    unique = tied.copy()
    unique[1] = -1
    unique[1, :2] = 1
    unique[1, 2048:-2] = 1
    for matrix, fallback_width in ((tied, 2048), (unique, 2)):
        result, info = reduced.reduced_top_from_votes(matrix, 1)
        assert info["arithmetic"] == "dyadic" and info["exact_columns"] == fallback_width
        assert result.tolist() == [1]


def test_zero_and_large_unanimous_or_absent_sign_rows():
    for value in (-1, 0, 1):
        matrix = np.full((2, 128), value, dtype=np.int8)
        scale, weights = reduced.reduced_weights(matrix, (16, 4, 0), 1)
        assert scale == 1
        assert all(layer.dtype == np.int64 for layer in weights)
        selected, info = reduced.reduced_top_from_votes(matrix, 63)
        assert info["arithmetic"] == "reduced_int64"
        np.testing.assert_array_equal(selected, np.arange(63))
        np.testing.assert_array_equal(selected, exact_top_from_votes(matrix, 63)[0])
