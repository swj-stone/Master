"""Verify exact ranks and binary64 rounding for the Section 4.1 witnesses."""
import argparse
import json
import struct
from fractions import Fraction
from math import comb, floor, nextafter
from pathlib import Path

import numpy as np

from experiments.audit_minpi_study_one_peer import gas_panel
from experiments.minpi_study_boundary_cost import complement, correction_inputs, vote_hash
from experiments.minpi_study_gas_replay import COLUMNS
from experiments.minpi_study_precision_design import direct_layer


def binary64_word(value):
    return "0x" + struct.pack(">d", value).hex()


def strict_winner(votes, tiers):
    w, p = votes.shape
    assert (w, p) == (3, 4096) and tiers == (0, 4, 16)
    assert np.isin(votes, (-1, 1)).all()
    positives = (votes > 0).sum(axis=1).tolist()
    assert positives == [2048, 2048, 4088]
    patterns = np.unique(votes.T, axis=0)
    layers = {k: direct_layer(votes, k) for k in tiers}
    gap = Fraction(comb(4087, 8), 3 * comb(4095, 16))
    scores = []
    classes = []
    for pattern in patterns:
        signs = pattern.tolist()
        members = np.flatnonzero((votes.T == pattern).all(axis=1))
        # At even k with binary ballots only a zero peer margin contributes.
        means = {}
        for k, (denominator, tally) in layers.items():
            row_counts = []
            for s, n in zip(signs, positives):
                positive_peers = n - int(s > 0)
                negative_peers = p - n - int(s < 0)
                row_counts.append(s * comb(positive_peers, k // 2)
                                    * comb(negative_peers, k // 2))
            count = sum(row_counts)
            assert (tally[members] == count).all()
            means[k] = Fraction(count, w * denominator)
            if k == 16 and signs[:2] != [1, 1]:
                assert sum(row_counts[:2]) <= 0
                assert Fraction(row_counts[2], w * denominator) <= gap
        score = min(means.values())
        scores.extend((score, int(j)) for j in members)
        # All members of each observed pattern occupy one contiguous index span.
        lo, hi = int(members[0]), int(members[-1])
        assert hi - lo + 1 == len(members)
        classes.append(dict(signs=signs, column_range_inclusive=[lo, hi], count=len(members),
                            layer_scores={k: str(mean) for k, mean in means.items()},
                            minimum=str(score), minimizing_layers=[k for k in tiers if means[k] == score]))

    scores.sort(key=lambda pair: (-pair[0], pair[1]))
    (winner, first), (runner_up, second), (best_other, _) = scores[:3]
    assert (first, second) == (1, 0) and winner - runner_up == gap
    assert best_other == gap < runner_up
    # Both scores are normal binary64 numbers' near neighbours in the same binade.
    # In [2^-3, 2^-2) adjacent binary64 values are separated by 2^(-3-52).
    assert Fraction(1, 8) < runner_up < winner < Fraction(1, 4)
    ulp = Fraction(1, 2**55)
    lower = (runner_up // ulp) * ulp
    assert (winner // ulp) * ulp == lower
    upper = lower + ulp
    assert Fraction.from_float(float(lower)) == lower
    assert Fraction.from_float(nextafter(float(lower), float("inf"))) == upper
    offsets = {j: (s - lower) / ulp for s, j in scores[:2]}
    assert all(0 < offset < Fraction(1, 2) for offset in offsets.values())
    pattern = binary64_word(float(lower))
    assert pattern == "0x3fc0ca670a942f96"
    assert all(binary64_word(float(s)) == pattern for s, _ in scores[:2])
    rounded_max = max(float(s) for s, _ in scores)
    rounded_leaders = sorted(j for s, j in scores if float(s) == rounded_max)
    assert rounded_leaders == [0, 1]

    return dict(vote_sha256=vote_hash(votes), shape=[w, p], tiers=tiers, h=1,
                column_classes=classes, winner=first, runner_up=second,
                winner_gap=str(gap), largest_minimum_among_other_columns=str(best_other),
                exact_lower_binary64=str(lower), exact_upper_binary64=str(upper), ulp=str(ulp),
                exact_midpoint=str((lower + upper) / 2),
                offsets_in_ulp={j: str(offset) for j, offset in offsets.items()},
                nearest_binary64_word=pattern, rounded_leaders=rounded_leaders,
                rounded_winner_with_input_order=rounded_leaders[0])


def gas_tie(votes):
    baseline = gas_panel(1129)
    w, p = votes.shape
    total = int(votes.sum())
    assert total == 12058
    denominator = 2 * w * (p - 1)
    per_column = {}
    for j in COLUMNS:
        column_sum = int(votes[:, j].sum())
        numerator = p * column_sum - total
        one_peer = Fraction(numerator, denominator)
        assert column_sum == 86 and one_peer == Fraction(-525, 32512)
        layers = {int(k): Fraction(s) for k, s in baseline["layers_at_cutoff"][str(j)].items()}
        assert layers[1] == one_peer < min(layers[k] for k in (0, 4, 16))
        per_column[j] = dict(vote_sum=column_sum, one_peer_numerator=numerator,
                             means={k: str(s) for k, s in layers.items()},
                             one_division_word=binary64_word(numerator / denominator))
    assert len({entry["one_division_word"] for entry in per_column.values()}) == 1
    return dict(vote_sha256=vote_hash(votes), shape=[w, p], total_vote_sum=total,
                common_denominator=denominator, columns=per_column)


def complete_fallback():
    case, grids = complement()
    measured_bits = [grid["bits"] for grid in grids]
    w, p = case["shape"]
    assert measured_bits == list(range(63 - w.bit_length()))
    weights = {int(k): Fraction(v) for k, v in case["row_weight_magnitudes"].items()}
    for grid in grids:
        q = 2**grid["bits"]
        bounds = []
        for magnitude in weights.values():
            scaled = q * magnitude
            lower = floor(scaled) + floor(-scaled)
            upper = lower + 2 * int(scaled.denominator != 1)
            bounds.append((lower, upper))
        pair = min(lo for lo, _ in bounds), min(hi for _, hi in bounds)
        assert pair == (grid["lower_grid_units"], grid["upper_grid_units"]) == (-1, 0)
        assert grid["refined"] == grid["fallback_actual"] == p == 128
    return dict(vote_sha256=case["vote_sha256"], shape=case["shape"], h=case["h"],
                double_D16=case["twice_D16"], exact_score_every_column=case["all_layer_scores"],
                first_and_last_precision=[measured_bits[0], measured_bits[-1]],
                precision_count=len(grids), exact_support=case["exact_support"],
                minimum_interval_grid_units=[-1, 0], refined=p, fully_rescored=p)


def run():
    cases = {name: (votes, tiers) for name, votes, tiers, _ in correction_inputs()}
    return dict(strict_winner=strict_winner(*cases["unique-4096"]),
                gas_tie=gas_tie(cases["gas-1129"][0]), complementary_rows=complete_fallback())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the exact audit record to this file")
    output = parser.parse_args().output
    record = json.dumps(run(), indent=2) + "\n"
    if output is None:
        print(record, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(record, encoding="utf-8")
