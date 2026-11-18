from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from experiments.audit_minpi_study_exact_targets import enumerated_layers
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from minppi_mofs.layer_word_minpi import _compare_ratios, layers_fit_word, layer_word_top_from_votes


def test_indexed_subsets_with_abstentions_repeated_columns_and_all_budgets():
    votes = np.array([[1, 0, 1, -1, -1, 1], [-1, 0, -1, 1, 1, 1],
                      [0, 0, 0, -1, -1, 1]], dtype=np.int8)
    for n in (1, 2, 3):
        for k in combinations((0, 1, 4), n):
            scores = enumerated_layers(votes, k).min(axis=0)
            order = sorted(range(6), key=lambda j: (-scores[j], j))
            for h in range(1, 7):
                selection, _ = layer_word_top_from_votes(votes, np.uint64(h), tiers=k)
                np.testing.assert_array_equal(selection, sorted(order[:h]))


def test_separate_word_range_and_cross_products_wider_than_int64():
    d = comb(80, 16)
    last = (2**63 - 1) // d
    assert layers_fit_word(81, (16, 4, 0), last)
    assert not layers_fit_word(81, (16, 4, 0), last + 1)
    assert d * (d - 2) > 2**63
    assert _compare_ratios((d - 1, d), (d - 2, d - 1)) == 1
    assert _compare_ratios((1 - d, d), (2 - d, d - 1)) == -1
    assert _compare_ratios((-d, d), (1 - d, d - 1)) == 0


def test_materials_and_fallback_match_independent_exact_layer_counts():
    root = Path(__file__).resolve().parents[1] / "results/minpi-study/new-cohorts"
    k = (0, 4, 16)
    for name, expected_path in (("supercon-1129", "layer_int64"), ("supercon-2027", "layer_int64"),
                                ("supercon-3469", "layer_int64"), ("gas-1129", "dyadic")):
        with np.load(root / (name + ".npz")) as record:
            votes = record["train_votes"]
        exact, _, _ = direct_minimum({t: direct_layer(votes, t) for t in k}, k)
        for pct in (2, 10, 50):
            h = (votes.shape[1] * pct + 99) // 100
            selected, info = layer_word_top_from_votes(votes, h, tiers=k)
            assert info["arithmetic"] == expected_path
            np.testing.assert_array_equal(selected, np.sort(exact[:h]))
