"""Replay two numerical corrections on fixed ballots and one full-width fallback."""
import argparse
import hashlib
import json
import platform
from fractions import Fraction
from math import comb, lcm
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.minpi_study_frontier import stress_case, write_csv
from experiments.minpi_study_gas_replay import SOURCE, VOTE_HASH
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from experiments.minpi_study_proposal import verified_float_top
from minppi_mofs import integer_minpi
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYERS = (0, 4, 16)
PAIRS = 32
SEED = 12816


def vote_hash(votes):
    return hashlib.sha256(votes.tobytes(order="C")).hexdigest()


def correction_inputs():
    with np.load(ROOT / SOURCE, allow_pickle=False) as archived:
        gas = archived["train_votes"]
    assert vote_hash(gas) == VOTE_HASH
    yield "gas-1129", gas, (0, 1, 4, 16), 64

    unique = stress_case(4096, "near")
    unique[1] = -1
    unique[1, :2] = 1
    unique[1, 2048:-2] = 1
    np.testing.assert_array_equal((unique[:2] == 1).sum(axis=1), [2048, 2048])
    np.testing.assert_array_equal((unique[:2] == -1).sum(axis=1), [2048, 2048])
    np.testing.assert_array_equal(np.flatnonzero((unique[:2] == 1).all(axis=0)), [0, 1])
    yield "unique-4096", unique, DEFAULT_LAYERS, 1


def oracle(votes, tiers, h):
    layers = {k: direct_layer(votes, k) for k in tiers}
    order, denominator, scores = direct_minimum(layers, tiers)
    gap = Fraction(scores[order[h - 1]] - scores[order[h]], len(votes) * denominator)
    return np.sort(order[:h]), gap


def measurement(name, votes, tiers, h, rng):
    reference, gap = oracle(votes, tiers, h)
    approximate, _ = top_from_votes(votes, h, tiers=tiers)
    restored = np.setdiff1d(reference, approximate).tolist()
    removed = np.setdiff1d(approximate, reference).tolist()
    if name == "gas-1129":
        assert (gap, restored, removed) == (0, [45], [115])
    else:
        assert gap == Fraction(comb(4087, 8), 3 * comb(4095, 16))
        assert (restored, removed) == ([1], [0])
    methods = dict(direct=exact_top_from_votes, verified_proposal=verified_float_top)
    for method in methods.values():
        support, info = method(votes, h, tiers=tiers)
        np.testing.assert_array_equal(support, reference)
        assert info["arithmetic"] == "dyadic"

    trial_rows = []
    paired_ratios = []
    first_positions = rng.permutation(np.tile(list(methods), PAIRS // 2))
    for repeat, first in enumerate(first_positions):
        elapsed = {}
        order = (first, next(method for method in methods if method != first))
        for position, label in enumerate(order):
            method = methods[label]
            start = perf_counter_ns()
            selected, info = method(votes, h, tiers=tiers)
            stop = perf_counter_ns()
            np.testing.assert_array_equal(selected, reference)
            assert info["arithmetic"] == "dyadic"
            if label == "verified_proposal":
                assert not info["proposal_accepted"] and info["proposal_swaps"] == len(restored)
            elapsed[label] = (stop - start) / 1e6
            trial_rows.append(dict(case=name, repeat=repeat, position=position, method=label,
                                   elapsed_ms=elapsed[label], refined=info["refined"],
                                   exact_columns=info["exact_columns"]))
        paired_ratios.append(elapsed["direct"] / elapsed["verified_proposal"])
    summary = dict(case=name, w=len(votes), p=votes.shape[1], tiers=list(tiers), h=h,
                   vote_sha256=vote_hash(votes), exact_gap=str(gap),
                   floating_support=approximate.tolist(), exact_support=reference.tolist(),
                   restored_columns=restored, removed_columns=removed)
    for label in methods:
        records = [row for row in trial_rows if row["method"] == label]
        assert len({(row["refined"], row["exact_columns"]) for row in records}) == 1
        summary[label] = dict(refined=records[0]["refined"], exact_columns=records[0]["exact_columns"],
                              time_ms=quartiles([row["elapsed_ms"] for row in records]))
    summary["paired_direct_over_verified"] = quartiles(paired_ratios)
    return summary, trial_rows


def quartiles(values):
    return dict(zip(("q25", "median", "q75"), np.quantile(values, [0.25, 0.5, 0.75]).tolist()))


def complement():
    first = np.r_[np.ones(64, dtype=np.int8), -np.ones(64, dtype=np.int8)]
    votes = np.stack([first, -first])
    w, p = votes.shape
    common = lcm(*(comb(p - 1, k) for k in DEFAULT_LAYERS))
    assert w * comb(p - 1, 16) > 2**63 - 1 and native_scale(p, DEFAULT_LAYERS, w) == 0
    weight = {}
    for k in DEFAULT_LAYERS:
        denominator, numerators = direct_layer(votes, k)
        assert (numerators == 0).all()
        # An even-size coalition without abstentions is pivotal iff the peer margin is zero.
        probability = Fraction(comb(63, k // 2) * comb(64, k // 2), denominator)
        weight[k] = probability
        _, one_row = direct_layer(votes[:1], k)
        assert (one_row * probability.denominator == first.astype(object) * probability.numerator * denominator).all()
    assert weight[4] == Fraction(6048, 15875) and weight[16] == Fraction(13278496104, 63266742275)

    exact_selected, info = exact_top_from_votes(votes, 1, tiers=DEFAULT_LAYERS)
    np.testing.assert_array_equal(exact_selected, [0])
    assert info["arithmetic"] == "dyadic" and info["refined"] == info["exact_columns"] == p
    floating, _ = top_from_votes(votes, 1, tiers=DEFAULT_LAYERS)

    all_bits = []
    get_bounds = integer_minpi._bounds
    score_exact = integer_minpi._rational_scores
    for b in range(63 - w.bit_length()):
        observed = {}
        widths = []

        def capture_bounds(matrix, layer, *args, **kwargs):
            lower, upper = get_bounds(matrix, layer, *args, **kwargs)
            k = next(k for k in DEFAULT_LAYERS if comb(p - 1, k) == layer[0])
            endpoints = (0, 0) if k == 0 else (-1, 1)
            assert (lower == endpoints[0]).all() and (upper == endpoints[1]).all()
            observed[k] = endpoints
            return lower, upper

        def capture_score(matrix, layers):
            widths.append(matrix.shape[1])
            return score_exact(matrix, layers)

        with patch.object(integer_minpi, "_bounds", capture_bounds), \
             patch.object(integer_minpi, "_rational_scores", capture_score):
            selected, diag = integer_minpi.rational_top_from_votes(votes, 1, tiers=DEFAULT_LAYERS, bits=b)
        assert observed.keys() == weight.keys()
        assert widths == [p] and diag["refined"] == diag["exact_columns"] == p
        np.testing.assert_array_equal(selected, [0])
        lower, upper = (min(pair[i] for pair in observed.values()) for i in range(2))
        assert (lower, upper) == (-1, 0)
        all_bits.append(dict(bits=b, lower_grid_units=lower, upper_grid_units=upper,
                             refined=diag["refined"], fallback_actual=widths[0]))
    result = dict(shape=[w,p], K=list(DEFAULT_LAYERS), h=1, vote_sha256=vote_hash(votes),
                  row_weight_magnitudes={str(k): str(value) for k,value in weight.items()},
                  twice_D16=str(w * comb(p - 1, 16)), twice_common_D=str(w * common),
                  all_layer_scores="0", exact_support=exact_selected.tolist(),
                  floating_support=floating.tolist(), dyadic_default=info)
    return result, all_bits


def run(output):
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(SEED)
    results, trials = [], []
    for name, votes, tiers, h in correction_inputs():
        result, timings = measurement(name, votes, tiers, h, rng)
        results.append(result)
        trials.extend(timings)
    counterexample, grids = complement()
    files = [SOURCE.as_posix(), "docs/minpi-study/boundary-cost-protocol.md",
             "experiments/minpi_study_boundary_cost.py", "experiments/minpi_study_proposal.py",
             "experiments/minpi_study_gas_replay.py", "experiments/minpi_study_precision_design.py",
             "experiments/minpi_study_frontier.py", "src/minppi_mofs/word_minpi.py",
             "src/minppi_mofs/integer_minpi.py", "src/minppi_mofs/certified_minpi.py",
             "src/minppi_mofs/exact_minpi.py"]
    manifest = dict(pairs_per_case=PAIRS, schedule_seed=SEED, threads=1,
                    python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                    files={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files})
    output.mkdir(parents=True)
    write_csv(output / "trials.csv", trials)
    write_csv(output / "complement-bits.csv", grids)
    for name, data in (("corrections", results), ("complement", counterexample), ("manifest", manifest)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(1):
        run(parser.parse_args().output)
