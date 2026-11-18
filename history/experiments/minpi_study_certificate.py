"""Replay the support-only algorithm on stored training splits; time its integration kernel separately."""
import argparse
import csv
import itertools
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, P, load_dataset
from experiments.minpi_study_kernel import make_matrix
from minppi_mofs.certified_minpi import _scores, _validate, _weights, select_minpi_support, top_from_votes
from minppi_mofs.exact_minpi import ExactMinPI, pivotal_rates, witness_margins


def replay():
    for dataset in P["datasets"]:
        X, y, task, *_ = load_dataset(dataset)
        for seed in P["formal_seeds"]:
            case = OUTPUT / "formal" / dataset / f"s{seed}"
            with np.load(case / "predictions.npz") as saved:
                rows = saved["train_indices"]
            record = json.loads((case / "record.json").read_text(encoding="utf-8"))
            reference = record["arms"]["exact"]["selected"]
            votes = np.sign(witness_margins(X[rows], y[rows], task, seed)).astype(np.int8)
            expected, _ = top_from_votes(votes, len(reference), prune=False)
            actual, info = top_from_votes(votes, len(reference))
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual, reference)
            yield dict(dataset=dataset, seed=seed, **info, possible_fraction=info["refined"]/info["p"],
                       evaluated_fraction=info["evaluated"]/(info["tiers"]*info["p"]), matches=True)


def timings():
    for width, abstain, seed in itertools.product((64, 512, 4096, 16384), (.1, .8), (127, 503, 887)):
        votes = make_matrix(seed, width, abstain)
        h = ceil(.1*width)
        def original():
            score = pivotal_rates(votes, (0, 4, 16)).min(axis=0)
            return np.sort(np.argsort(-score, kind="stable")[:h])
        variants = {"original": original, "full": lambda: top_from_votes(votes, h, prune=False)[0],
                    "certificate": lambda: top_from_votes(votes, h)[0]}
        ref, info = top_from_votes(votes, h)
        for run in variants.values():
            np.testing.assert_array_equal(run(), ref)
        rng = np.random.default_rng([seed, width, 315])
        times = {m: [] for m in variants}
        for _ in range(5):
            for method in rng.permutation(list(variants)):
                start = perf_counter()
                selected = variants[method]()
                times[method].append(perf_counter()-start)
                np.testing.assert_array_equal(ref, selected)
        for method, elapsed in times.items():
            yield dict(width=width, abstain_prob=abstain, seed=seed, k=h, method=method,
                       median_s=np.median(elapsed), min_s=min(elapsed), max_s=max(elapsed),
                       refined=info["refined"] if method == "certificate" else width)


def from_training_data():
    for dataset in P["primary_sources"]:
        X, y, task, *_ = load_dataset(dataset)
        for seed in P["formal_seeds"]:
            case = OUTPUT / "formal" / dataset / f"s{seed}"
            with np.load(case / "predictions.npz") as saved:
                rows = saved["train_indices"]
            xt, yt = X[rows], y[rows]
            reference = json.loads((case/"record.json").read_text(encoding="utf-8"))["arms"]["exact"]["selected"]
            variants = ("original", "full", "certificate")
            rng = np.random.default_rng([seed, 315])
            times = {m: [] for m in variants}
            for repetition in range(4):
                for method in rng.permutation(variants):
                    start = perf_counter()
                    if method == "certificate":
                        support = select_minpi_support(xt, yt, task=task, random_state=seed)
                    elif method == "original":
                        support = ExactMinPI(task=task, random_state=seed).fit(xt, yt).get_support(indices=True)
                    else:
                        votes = np.sign(witness_margins(xt, yt, task, seed)).astype(np.int8)
                        support, _ = top_from_votes(votes, len(reference), prune=False)
                    elapsed = perf_counter() - start
                    np.testing.assert_array_equal(support, reference)
                    if repetition:
                        times[method].append(elapsed)
            for method, elapsed in times.items():
                yield dict(dataset=dataset, seed=seed, n_train=len(xt), width=xt.shape[1], k=len(reference),
                           method=method, median_s=np.median(elapsed), min_s=min(elapsed), max_s=max(elapsed))


def single_layer_comparison():
    """Cost of certifying the minimum over merely returning the largest-stratum top set."""
    for dataset in P["primary_sources"]:
        X, y, task, *_ = load_dataset(dataset)
        p = X.shape[1]
        tier = min(16, p-1)
        for seed in P["formal_seeds"]:
            case = OUTPUT/"formal"/dataset/f"s{seed}"
            with np.load(case/"predictions.npz") as saved:
                rows = saved["train_indices"]
            arms = json.loads((case/"record.json").read_text(encoding="utf-8"))["arms"]
            h = arms["exact"]["k"]
            votes = np.sign(witness_margins(X[rows], y[rows], task, seed)).astype(np.int8)
            def largest_only():
                matrix = _validate(votes, h)
                pos = np.count_nonzero(matrix > 0, axis=1)
                neg = np.count_nonzero(matrix < 0, axis=1)
                weights = _weights(pos, neg, p-pos-neg, p, (tier,))[tier]
                score = _scores(matrix, weights)
                return np.sort(np.argsort(-score, kind="stable")[:h])
            variants = {"k16": largest_only, "exact": lambda: top_from_votes(votes, h)[0]}
            times = {m: [] for m in variants}
            rng = np.random.default_rng([seed, 714])
            for repetition in range(6):
                for method in rng.permutation(list(variants)):
                    start = perf_counter()
                    got = variants[method]()
                    elapsed = perf_counter()-start
                    np.testing.assert_array_equal(got, arms[method]["selected"])
                    if repetition:
                        times[method].append(elapsed)
            exact, single = (np.median(times[m]) for m in ("exact", "k16"))
            yield dict(dataset=dataset, seed=seed, width=p, k=h, exact_s=exact, single_layer_s=single,
                       exact_over_single=exact/single, same_support=arms["exact"]["selected"]==arms["k16"]["selected"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {"replay": replay, "timings": timings, "training": from_training_data, "single": single_layer_comparison}
    parser.add_argument("stage", choices=stages)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = stages[args.stage]()
    with threadpool_limits(1), args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = None
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
