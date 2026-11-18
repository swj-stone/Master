"""Reconstruct the fixed ballot matrices and audit Figure 1c,d without timing."""
import argparse
import csv
import json
from fractions import Fraction
from math import comb
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.minpi_study_boundary_cost import DEFAULT_LAYERS, vote_hash
from experiments.minpi_study_precision_design import BITS, direct_layer, direct_minimum
from experiments.audit_minpi_study_rounding import complete_fallback
from minppi_mofs import integer_minpi


STUDY = Path(__file__).resolve().parents[1] / "results/minpi-study"
WIDTH = 4096
LAST_SEVEN = set(range(WIDTH - 7, WIDTH))


def rows(file):
    with file.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def observed_fallback(counts):
    keys = ("panel", "seed", "game", "h")
    gaps = {tuple(row[key] for key in keys): Fraction(row["gap_exact"])
            for row in rows(STUDY / "precision-design/targets.csv")
            if not row["panel"].startswith("constructed")}
    assert len(gaps) == 102 and all(gap >= 0 for gap in gaps.values())
    table = []
    for bits in BITS:
        cases = {tuple(row[key] for key in keys): int(row["exact_columns"])
                 for row in counts if not row["panel"].startswith("constructed")
                 and int(row["bits"]) == bits}
        assert cases.keys() == gaps.keys()
        late = []
        for tied in (False, True):
            group = [key for key, gap in gaps.items() if (gap == 0) == tied]
            active = [key for key in group if cases[key]]
            late.extend(active)
            table.append(dict(bits=bits, group="tied" if tied else "positive",
                              targets=len(group), fallback_targets=len(active),
                              fallback_columns=sum(cases[key] for key in active)))
        if bits == 40:
            assert late == [("gas", "1129", "mixed", "64")] and cases[late[0]] == 3
    return dict(matrix_count=len({key[:2] for key in gaps}), target_count=len(gaps),
                groups=table, fallback_at_40_bits=dict(panel="gas", seed=1129, h=64,
                                                      tiers=[0, 1, 4, 16], columns=3))


def ballots(both_negative):
    """None denotes a repeat of row 0; otherwise specify its two joint negatives."""
    votes = -np.ones((3, WIDTH), dtype=np.int8)
    if both_negative is None:
        votes[:2, :WIDTH // 2] = 1
    else:
        mixed = sorted(set(range(2, WIDTH)) - set(both_negative))
        assert len(mixed) == WIDTH - 4
        votes[:2, :2] = 1
        half = len(mixed) // 2
        votes[0, mixed[:half]] = 1
        votes[1, mixed[half:]] = 1
    votes[2] = 1
    votes[2, [0, *sorted(LAST_SEVEN)]] = -1
    assert np.count_nonzero(votes > 0, axis=1).tolist() == [2048, 2048, 4088]
    return votes


def run_construction(both_negative):
    votes = ballots(both_negative)
    layers = {k: direct_layer(votes, k) for k in DEFAULT_LAYERS}
    order, denom, scores = direct_minimum(layers, DEFAULT_LAYERS)
    winner = int(order[0])
    leaders = np.flatnonzero(scores == scores[winner])
    gap = Fraction(scores[winner] - scores[order[1]], 3 * denom)
    if both_negative is None:
        np.testing.assert_array_equal(leaders, np.arange(1, 2048))
        assert gap == 0 and winner == 1
    else:
        np.testing.assert_array_equal(leaders, [1])
        assert order[1] == 0 and gap == Fraction(comb(4087, 8), 3 * comb(4095, 16))

    observations = []
    for bits in BITS:
        with patch.object(integer_minpi, "_rational_scores",
                          wraps=integer_minpi._rational_scores) as full_score:
            got, diagnostics = integer_minpi.rational_top_from_votes(votes, 1, tiers=DEFAULT_LAYERS, bits=bits)
        np.testing.assert_array_equal(got, [winner])
        full_score.assert_called_once()
        actual = full_score.call_args.args[0]
        if both_negative is None:
            expected = np.arange(WIDTH // 2)
        elif bits == 0:
            expected = sorted(set(range(WIDTH)) - LAST_SEVEN - set(both_negative))
        else:
            expected = [0, 1]
        np.testing.assert_array_equal(actual, votes[:, expected])
        assert actual.shape[1] == diagnostics["exact_columns"]
        observations.append(dict(bits=bits, refined=diagnostics["refined"],
                                 fallback_columns=actual.shape[1],
                                 distinct_fallback_patterns=len(np.unique(actual.T, axis=0))))
    return dict(joint_negatives=both_negative, vote_sha256=vote_hash(votes),
                all_column_patterns=len(np.unique(votes.T, axis=0)), selected=winner,
                tied_at_maximum=len(leaders), exact_gap=str(gap), observations=observations)


def run():
    counts = [row for row in rows(STUDY / "zero-bounds/counts.csv") if row["policy"] == "zero_point"]
    assert all(row["oracle_match"] == "True" for row in counts)
    observed = observed_fallback(counts)
    strict = [run_construction(a) for a in ([2, 3], [2, 4095], [4094, 4095])]
    tied = run_construction(None)
    archived_hashes = {row["panel"]: row["votes_sha256"] for row in rows(STUDY / "zero-bounds/panels.csv")}
    for name, case in (("constructed_unique_max", strict[-1]), ("constructed_tied_max", tied)):
        assert case["vote_sha256"] == archived_hashes[name]
        expected = {int(row["bits"]): (int(row["refined"]), int(row["exact_columns"]))
                    for row in counts if row["panel"] == name and row["h"] == "1" and row["game"] == "default"}
        actual = {row["bits"]: (row["refined"], row["fallback_columns"]) for row in case["observations"]}
        assert actual == expected
    denominator = comb(127, 16)
    wide = comb(16383, 16)
    complement = complete_fallback()
    previous = json.loads((STUDY / "rounding-witnesses/audit.json").read_text(encoding="utf-8"))
    assert complement == previous["complementary_rows"]
    assert int(complement["double_D16"]) == 2 * denominator and complement["fully_rescored"] == 128
    assert denominator.bit_length() == 67 and (2 * denominator).bit_length() == 68
    assert wide.bit_length() == 180
    return dict(observed=observed, strict_overlap_variants=strict, repeated_balanced_row=tied,
                denominator_check=dict(D16_at_128=str(denominator), bits_at_128=denominator.bit_length(),
                                       bits_with_two_rows=(2 * denominator).bit_length(),
                                       bits_at_16384=wide.bit_length()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the resulting JSON to a new file")
    destination = parser.parse_args().output
    result = json.dumps(run(), indent=2) + "\n"
    if destination is None:
        print(result, end="")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(result)
