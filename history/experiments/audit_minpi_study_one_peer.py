"""Check the binary one-peer identity, the Gas cutoff and batch weighting."""
import argparse
import csv
import json
from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np

from experiments.minpi_study import ROOT, file_hash
from experiments.minpi_study_precision_design import direct_layer, direct_minimum


def check_rows():
    checked = 0
    for p in range(2, 9):
        for votes in product((-1, 1), repeat=p):
            margin = sum(votes)
            for j, focal in enumerate(votes):
                total = 0
                for i, peer in enumerate(votes):
                    if i != j:
                        before = peer
                        after = int(peer + focal > 0) - int(peer + focal < 0)
                        total += after - before
                assert 2 * total == p * focal - margin
                checked += 1
    return checked


def gas_panel(seed):
    path = ROOT / f"results/minpi-study/new-cohorts/gas-{seed}.npz"
    with np.load(path) as archive:
        votes = archive["train_votes"]
    assert np.isin(votes, (-1, 1)).all()
    w, p = votes.shape
    layers = {k: direct_layer(votes, k) for k in (0, 1, 4, 16)}
    totals = votes.sum(axis=0).astype(object)
    affine_numerators = p * totals - sum(totals)
    assert layers[1][0] == p - 1
    np.testing.assert_array_equal(affine_numerators, 2 * layers[1][1])
    old = direct_minimum(layers, (0, 4, 16))[0][:64]
    order, denominator, scores = direct_minimum(layers, (0, 1, 4, 16))
    cutoff = scores[order[63]]
    tied = np.flatnonzero(scores == cutoff)
    result = dict(seed=seed, input_sha256=file_hash(path), w=w, p=p,
                  checked_columns=p, full_layer_mean=str(Fraction(sum(totals), p*w)),
                  added_at_half_width=sorted(set(map(int, order[:64])) - set(map(int, old))))
    if seed == 1129:
        np.testing.assert_array_equal(tied, [45, 57, 115])
        assert len(result["added_at_half_width"]) == 10
        result["cutoff_columns"] = list(map(int, tied))
        result["cutoff"] = str(Fraction(cutoff, w*denominator))
        result["layers_at_cutoff"] = {
            str(j): {str(k): str(Fraction(numerators[j], w*d))
                     for k, (d, numerators) in layers.items()} for j in tied}
    return result


def batch_weighting():
    path = ROOT / "results/minpi-study/arithmetic-consequence/gas-predictions.csv"
    totals, exact_hits, float_hits = Counter(), Counter(), Counter()
    with path.open(encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            key = (int(record["batch"]), int(record["target"]))
            totals[key] += 1
            exact_hits[key] += int(record["exact"] == record["target"])
            float_hits[key] += int(record["float"] == record["target"])
    differences = {}
    for batch in (8, 9, 10):
        keys = [key for key in totals if key[0] == batch]
        differences[batch] = sum(Fraction(exact_hits[key]-float_hits[key], totals[key])
                                 for key in keys) / len(keys)
    return dict(input_sha256=file_hash(path),
                exact_minus_float_BA={b: str(v) for b, v in differences.items()},
                equal_batch_difference=str(sum(differences.values()) / len(differences)),
                break_even_weight_10_over_9=str(-differences[9] / differences[10]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = dict(single_row_focal_checks=check_rows(),
                  gas_panels=[gas_panel(seed) for seed in (1129, 2027, 3469)],
                  held_out_batch_effects=batch_weighting())
    record = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(record, encoding="utf-8")
    print(record, end="")
