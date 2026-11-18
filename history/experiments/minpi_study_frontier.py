"""Prospective measurements on fresh fixed vote matrices and rational boundary checks."""
import argparse
import csv
import hashlib
import itertools
import json
import platform
from fractions import Fraction
from functools import lru_cache
from math import ceil, comb
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

from minppi_mofs.certified_minpi import _scores, _validate, _weights, top_from_votes


TIERS = (0, 4, 16)
SEEDS = (34123, 34127, 34129)


def matrix(width, rows, abstain, seed):
    rng = np.random.default_rng([seed, width, rows, int(100 * abstain)])
    threshold = rng.normal(size=width) + rng.normal(size=(rows, 1))
    votes = np.where(rng.logistic(size=(rows, width)) < threshold, 1, -1).astype(np.int8)
    votes[rng.random(size=votes.shape) < abstain] = 0
    return votes


def row_counts(votes):
    pos = np.count_nonzero(votes > 0, axis=1)
    neg = np.count_nonzero(votes < 0, axis=1)
    return pos, neg, votes.shape[1] - pos - neg


def largest(votes, h):
    v = _validate(votes, h)
    weights = _weights(*row_counts(v), v.shape[1], (16,))
    return np.sort(np.argsort(-_scores(v, weights[16]), kind="stable")[:h])


def profile(votes):
    start = perf_counter_ns()
    counts = row_counts(votes)
    tallied = perf_counter_ns()
    weights = _weights(*counts, votes.shape[1], TIERS)
    weighted = perf_counter_ns()
    scores = np.stack([votes.mean(0), _scores(votes, weights[4]), _scores(votes, weights[16])])
    finished = perf_counter_ns()
    return scores, [(b-a)/1e6 for a, b in zip((start, tallied, weighted), (tallied, weighted, finished))]


def performance():
    trials, cases, stages = [], [], []
    for p, w, abstain, seed in itertools.product((128, 4096, 16384), (32, 192), (0., .75), SEEDS):
        votes = matrix(p, w, abstain, seed)
        identity = dict(p=p, w=w, abstain=abstain, seed=seed)
        rng = np.random.default_rng([seed, w, p, 2005])
        profile(votes)
        for repeat in range(5):
            scores, costs = profile(votes)
            stages.append(dict(**identity, repeat=repeat, tally_ms=costs[0], weight_ms=costs[1], score_ms=costs[2]))
        minimum, upper = scores.min(0), scores[-1]
        for fraction in (.02, .10, .50):
            h = ceil(fraction*p)
            key = dict(**identity, fraction=fraction, h=h)
            variants = {"full": lambda: top_from_votes(votes, h, prune=False)[0],
                        "cert": lambda: top_from_votes(votes, h)[0], "largest": lambda: largest(votes, h)}
            support, info = top_from_votes(votes, h)
            head = largest(votes, h)
            mask = np.ones(p, dtype=bool)
            mask[head] = False
            tau = minimum[head].min()
            extra = np.count_nonzero(mask & (upper >= tau))
            assert extra == info["refined"] - h
            np.testing.assert_array_equal(support, variants["full"]())
            cases.append(dict(**key, refined=info["refined"], extra=extra, extra_fraction=extra/p,
                              outside_tied=np.count_nonzero(mask & (upper == tau)),
                              largest_same=np.array_equal(head, support)))
            for repeat in range(5):
                for order, method in enumerate(rng.permutation(list(variants))):
                    start = perf_counter_ns()
                    current = variants[method]()
                    elapsed = (perf_counter_ns() - start) / 1e6
                    np.testing.assert_array_equal(current, head if method == "largest" else support)
                    trials.append(dict(**key, repeat=repeat, order=order, method=method, elapsed_ms=elapsed))
    return trials, cases, stages


@lru_cache(maxsize=4096)
def row_increment(pos, neg, zero, k, vote):
    if vote == 0:
        return Fraction(0)
    pos -= vote == 1
    neg -= vote == -1
    total = 0
    for a in range(min(pos, k) + 1):
        for b in range(min(neg, k-a) + 1):
            if a-b in (0, -vote):
                total += comb(pos, a) * comb(neg, b) * comb(zero, k-a-b)
    return Fraction(vote*total, comb(pos+neg+zero, k))


def rational_rates(votes):
    counts = [tuple(map(int, row)) for row in zip(*row_counts(votes))]
    profiles, mapping = np.unique(votes.T, axis=0, return_inverse=True)
    table = np.empty((len(TIERS), len(profiles)), dtype=object)
    for t, k in enumerate(TIERS):
        for j, column in enumerate(profiles):
            total = sum(row_increment(*count, k, int(v)) for count, v in zip(counts, column))
            table[t, j] = total / len(votes)
    return table[:, mapping]


def stress_case(width, construction):
    balanced = np.where(np.arange(width) < width//2, 1, -1).astype(np.int8)
    if construction == "homogeneous":
        return np.stack([balanced, np.roll(balanced, 1), np.roll(balanced, 2)])
    if construction == "zero":
        return np.zeros((3, width), dtype=np.int8)
    skew = np.ones(width, dtype=np.int8)
    skew[0] = -1
    if construction == "reversal":
        return np.stack([balanced, skew, skew])
    skew[-7:] = -1
    return np.stack([balanced, balanced, skew])


def direct(votes, column, k):
    indices = [j for j in range(votes.shape[1]) if j != column]
    total = 0
    for peers in itertools.combinations(indices, k):
        margin = votes[:, peers].sum(1)
        total += int((np.sign(margin + votes[:, column])-np.sign(margin)).sum())
    return Fraction(total, comb(len(indices), k) * len(votes))


def arithmetic():
    records = []
    for name, p in [("homogeneous", 20), ("reversal", 20), ("zero", 20),
                    ("near", 128), ("near", 512), ("near", 4096)]:
        votes = stress_case(p, name)
        rates = rational_rates(votes)
        if name == "reversal":
            for t, k in enumerate(TIERS):
                for j in (0, 1, p-1):
                    assert rates[t,j] == direct(votes, j, k)
        h = 1 if name == "near" else p//2
        minima = rates.min(0)
        ranking = np.argsort(-minima, kind="stable")
        exact = sorted(map(int, ranking[:h]))
        last = sorted(map(int, np.argsort(-rates[-1], kind="stable")[:h]))
        if name == "homogeneous":
            assert all(np.array_equal(ranking, np.argsort(-row, kind="stable")) for row in rates)
        if name == "reversal":
            assert 0 in last and 0 not in exact
        approximate, diag = top_from_votes(votes, h)
        np.testing.assert_array_equal(approximate, top_from_votes(votes, h, prune=False)[0])
        records.append(dict(name=name, p=p, h=h, rational=exact, largest_rational=last,
                            float_support=approximate.tolist(), float_matches=(approximate.tolist() == exact),
                            gap_column1_minus_0=str(minima[1]-minima[0]),
                            refined=int(diag["refined"])))
    return records


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(1):
        checks = arithmetic()
        timings, cases, stages = performance()
    for key, rows in (("trials", timings), ("cases", cases), ("stages", stages)):
        write_csv(output / f"{key}.csv", rows)
    (output / "arithmetic.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"src/minppi_mofs/certified_minpi.py", root/"src/minppi_mofs/exact_minpi.py",
             root/"docs/minpi-study/frontier-protocol.md"]
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    info = dict(files=manifest, python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                threads=1, matrices=len(stages)//5, budgets=len(cases), trials=len(timings))
    (output / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
