"""Trace the integer cross products actually compared on the three materials ballots."""
import argparse
import csv
import json
from functools import cmp_to_key
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.minpi_study import OUTPUT
from minppi_mofs import layer_word_minpi


def audit():
    with (OUTPUT/"layer-word/targets.csv").open(encoding="utf-8") as stream:
        cases = [row for row in csv.DictReader(stream) if row["range"] == "layer_only"]
    assert len(cases) == 9 and all(row["source"] == "supercon" for row in cases)
    compare = layer_word_minpi._compare_ratios
    records = []
    for case in cases:
        with np.load(OUTPUT/"new-cohorts"/f"supercon-{case['seed']}.npz") as archived:
            votes = archived["train_votes"]
        maximum, oversized, unlike = 0, 0, 0
        example = None

        def observe(left, right):
            nonlocal maximum, oversized, unlike, example
            if left[1] != right[1]:
                unlike += 1
                first = left[0] * right[1]
                second = right[0] * left[1]
                largest = max(abs(first), abs(second))
                oversized += largest > 2**63 - 1
                if largest > maximum:
                    maximum = largest
                    example = dict(left=left, right=right, left_product=str(first), right_product=str(second))
            return compare(left, right)

        # The selector calls the comparator directly for layer minima and via
        # cmp_to_key for the candidate cutoff and final ordered boundary.
        with patch.object(layer_word_minpi, "_compare_ratios", observe), patch.object(
                layer_word_minpi, "_ratio_key", cmp_to_key(observe)):
            selected, info = layer_word_minpi.layer_word_top_from_votes(votes, int(case["h"]))
        assert info["arithmetic"] == "layer_int64" and example is not None
        np.testing.assert_array_equal(selected, np.fromstring(case["support"], dtype=int, sep=" "))
        records.append(dict(seed=int(case["seed"]), budget_pct=int(case["budget_pct"]), h=int(case["h"]),
                            votes_sha256=case["votes_sha256"], unequal_denominator_comparisons=unlike,
                            comparisons_with_product_above_symmetric_int64_bound=oversized,
                            largest_absolute_product=str(maximum), magnitude_bits=maximum.bit_length(),
                            example=example))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    destination = parser.parse_args().output
    records = audit()
    with destination.open("x", encoding="utf-8") as out:
        json.dump(records, out, indent=2)
