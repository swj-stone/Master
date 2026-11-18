"""Replay the exact cutoff and the archived binary64 discrepancy on Gas seed 1129."""
import argparse
import hashlib
import json
import platform
from fractions import Fraction
from math import ulp
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from minppi_mofs import certified_minpi
from minppi_mofs.word_minpi import exact_top_from_votes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("results/minpi-study/new-cohorts/gas-1129.npz")
VOTE_HASH = "e399f2f046d6b77d326b2143f63bba4deeb87353f4ff91dfee92a3ab813b973c"
TIERS = (0, 1, 4, 16)
H = 64
COLUMNS = (45, 57, 115)


def floating_path(votes, prune):
    """Read each tied column's layer scores on the batches the selector actually uses."""
    scorer = certified_minpi._scores
    layers = {j: [float(votes[:, j].mean())] for j in COLUMNS}

    def record(batch, weights):
        scores = scorer(batch, weights)
        for j in COLUMNS:
            # Each of these three ballot columns is unique in the full matrix.
            matches = np.flatnonzero(np.all(batch == votes[:, j, None], axis=0))
            if len(matches):
                layers[j].append(float(scores[matches.item()]))
        return scores

    with patch.object(certified_minpi, "_scores", record):
        support, stats = certified_minpi.top_from_votes(votes, H, tiers=TIERS, prune=prune)
    assert all(len(values) == len(TIERS) for values in layers.values())
    minima = [min(layers[j]) for j in COLUMNS]
    return support, {
        "minima_in_column_order": minima,
        "binary64_hex": [value.hex() for value in minima],
        "ulp_gaps_above_column_45": [int((value - minima[0]) / ulp(minima[0])) for value in minima],
        "selected_at_cutoff": [j for j in COLUMNS if j in support],
        "layer_column_scores_evaluated": stats["evaluated"],
    }


def replay():
    archive = ROOT / SOURCE
    with np.load(archive, allow_pickle=False) as data:
        votes = data["train_votes"]
    ballot_hash = hashlib.sha256(votes.tobytes(order="C")).hexdigest()
    assert ballot_hash == VOTE_HASH and votes.shape == (256, 128) and votes.dtype == np.int8
    for j in COLUMNS:
        np.testing.assert_array_equal(np.flatnonzero(np.all(votes == votes[:, j, None], axis=0)), [j])

    layers = {k: direct_layer(votes, k) for k in TIERS}
    order, denominator, numerators = direct_minimum(layers, TIERS)
    cutoff = numerators[order[H - 1]]
    rational = Fraction(cutoff, len(votes) * denominator)
    np.testing.assert_array_equal(np.flatnonzero(numerators == cutoff), COLUMNS)
    np.testing.assert_array_equal(order[62:65], COLUMNS)
    assert rational == Fraction(-525, 32512)
    binding = {
        str(j): [k for k, (d, n) in layers.items() if Fraction(n[j], len(votes) * d) == rational]
        for j in COLUMNS
    }
    assert all(ks == [1] for ks in binding.values())
    support = np.sort(order[:H])
    exact_support, _ = exact_top_from_votes(votes, H, tiers=TIERS)
    np.testing.assert_array_equal(exact_support, support)
    floating = {}
    for name, prune in (("exhaustive", False), ("pruned", True)):
        observed, record = floating_path(votes, prune)
        floating[name] = record
        # These assertions replay the *observed* rounding on the archived backend.
        # Different numerical libraries may produce different binary64 rounding.
        np.testing.assert_array_equal(np.setdiff1d(support, observed), [45])
        np.testing.assert_array_equal(np.setdiff1d(observed, support), [115])
        s45, s57, s115 = record["minima_in_column_order"]
        assert s45 < s57 < s115, "Binary64 order differs from the archived run"
    with archive.open("rb") as source:
        archive_hash = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "source": SOURCE.as_posix(), "source_sha256": archive_hash,
        "key": "train_votes", "shape": list(votes.shape), "dtype": str(votes.dtype),
        "vote_bytes_C_order_sha256": ballot_hash, "witness_seed": 1129,
        "peer_counts": list(TIERS), "budget": H, "columns_zero_based": list(COLUMNS),
        "exact_positions_one_based": [63, 64, 65], "exact_score": str(rational),
        "minimizing_layers": binding, "exact_selected_at_cutoff": [45, 57],
        "binary64": floating,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "system": platform.system(), "machine": platform.machine()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the checked record as JSON")
    output = parser.parse_args().output
    record = json.dumps(replay(), indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(record, encoding="utf-8")
    print(record, end="")
