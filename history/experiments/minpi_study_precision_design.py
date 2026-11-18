"""Check custom peer-count games and the cost of a fixed exact answer at four precisions."""
import argparse
import json
from fractions import Fraction
from math import ceil, comb, lcm
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import ROOT, file_hash
from experiments.minpi_study_frontier import stress_case
from experiments.minpi_study_raw_cost import panels
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.integer_minpi import rational_top_from_votes


REAL_CASES = {("gas", 1129), ("supercon", 1129), ("mhealth", 20261123),
              ("diabetes", 2027), ("crimedata", 3301), ("gisette", 3301)}
GAMES = {"raw": (0,), "small": (4,), "large": (16,), "default": (0, 4, 16),
         "odd": (1, 5, 15), "mixed": (0, 1, 4, 16)}
BITS = (0, 8, 20, 40)


def inputs():
    for name, seed, rebuild, _, _, _ in panels():
        if (name, seed) in REAL_CASES:
            yield name, seed, rebuild()
    tied = stress_case(4096, "near")
    yield "constructed_tied_max", 0, tied
    unique = tied.copy()
    unique[1] = -1
    unique[1, :2] = 1
    unique[1, 2048:-2] = 1
    yield "constructed_unique_max", 0, unique


def direct_layer(votes, k):
    """Independent tally: enumerate all feasible positive and negative peer counts."""
    denom = comb(votes.shape[1]-1, k)
    numerators = np.zeros(votes.shape[1], dtype=object)
    cache = {}
    for row in votes:
        pos, neg = int((row > 0).sum()), int((row < 0).sum())
        zero = len(row)-pos-neg
        for sign, size in ((1, pos), (-1, neg)):
            if size == 0:
                continue
            key = pos, neg, sign
            if key not in cache:
                plus, minus = pos-(sign == 1), neg-(sign == -1)
                total = 0
                for i in range(min(k, plus) + 1):
                    for j in range(min(k-i, minus) + 1):
                        a, b = i-j, i-j+sign
                        change = ((b > 0)-(b < 0))-((a > 0)-(a < 0))
                        if change and k-i-j <= zero:
                            total += change * comb(plus, i) * comb(minus, j) * comb(zero, k-i-j)
                cache[key] = total
            numerators[row == sign] += cache[key]
    return denom, numerators


def direct_minimum(layers, ks):
    denominator = lcm(*(layers[k][0] for k in ks))
    numerators = np.min([num * (denominator // denom) for denom, num in
                         (layers[k] for k in ks)], axis=0)
    order = np.argsort(-numerators, kind="stable")
    return order, denominator, numerators


def run(output):
    if output.exists():
        raise FileExistsError(output)
    targets, counts, trials = [], [], []
    rng = np.random.default_rng(44317)
    for name, seed, votes in inputs():
        w, p = votes.shape
        games = {key: tuple(sorted({min(p-1, k) for k in ks})) for key, ks in GAMES.items()}
        layers = {k: direct_layer(votes, k) for k in sorted(set().union(*games.values()))}
        default_order = direct_minimum(layers, games["default"])[0]
        for label, tiers in games.items():
            order, denom, scores = direct_minimum(layers, tiers)
            for h in sorted({1, ceil(.1*p), ceil(.5*p)}):
                truth = np.sort(order[:h])
                float_result = top_from_votes(votes, h, tiers=tiers)[0]
                full, _ = rational_top_from_votes(votes, h, tiers=tiers, prune=False)
                np.testing.assert_array_equal(truth, full)
                key = dict(panel=name, seed=seed, w=w, p=p, h=h, game=label,
                           tiers=" ".join(map(str, tiers)))
                gap = Fraction(scores[order[h-1]]-scores[order[h]], w * denom)
                omitted = np.setdiff1d(truth, float_result)
                added = np.setdiff1d(float_result, truth)
                if len(omitted):
                    float_full = top_from_votes(votes, h, tiers=tiers, prune=False)[0]
                    exhaustive_same = np.array_equal(float_result, float_full)
                else:
                    exhaustive_same = None
                targets.append(dict(**key, gap_exact=str(gap), gap=float(gap),
                    swaps_vs_default=len(set(truth)-set(default_order[:h])),
                    float_swaps=len(omitted), float_omitted=" ".join(map(str, omitted)),
                    float_added=" ".join(map(str, added)), float_full_same_on_mismatch=exhaustive_same,
                    cutoff_exact=str(Fraction(scores[order[h-1]], w*denom)),
                    support=" ".join(map(str, truth))))
                for r in range(-1, 3):
                    for position, b in enumerate(rng.permutation(BITS)):
                        start = perf_counter_ns()
                        chosen, info = rational_top_from_votes(votes, h, bits=b, tiers=tiers)
                        stop = perf_counter_ns()
                        np.testing.assert_array_equal(truth, chosen)
                        if r == -1:
                            counts.append(dict(**key, **info, oracle_match=True))
                        else:
                            trials.append(dict(**key, bits=b, repeat=r, position=position,
                                               elapsed_ms=(stop-start)/1e6))
    output.mkdir(parents=True)
    for name, records in (("targets", targets), ("counts", counts), ("trials", trials)):
        pd.DataFrame(records).to_csv(output/f"{name}.csv", index=False)
    files = ["docs/minpi-study/precision-design-protocol.md", "src/minppi_mofs/exact_minpi.py",
             "src/minppi_mofs/certified_minpi.py", "src/minppi_mofs/integer_minpi.py",
             "src/minppi_mofs/support_selector.py", "src/minppi_mofs/word_minpi.py",
             "experiments/minpi_study_raw_cost.py", "experiments/minpi_study_frontier.py",
             "experiments/minpi_study_precision_design.py", "tests/test_integer_minpi.py"]
    (output/"manifest.json").write_text(json.dumps({f: file_hash(ROOT/f) for f in files}, indent=2),
                                        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(1):
        run(parser.parse_args().output)
