"""Compare rational-oracle, interval-certified and double-precision supports."""
import argparse
import json
from fractions import Fraction
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, P, file_hash, load_dataset
from experiments.minpi_study_frontier import matrix, rational_rates, stress_case, write_csv
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.integer_minpi import rational_top_from_votes


def inputs():
    for p in (128, 4096, 16384):
        for w in (32, 192):
            for seed in (38119, 38149):
                if p == 16384 and (w != 192 or seed != 38119):
                    continue
                yield "fresh", seed, matrix(p, w, .75, seed), ceil(.1*p)
    for dataset in ("crimedata", "gisette"):
        X, y, task, *_ = load_dataset(dataset)
        seeds = P["formal_seeds"] if dataset == "crimedata" else P["formal_seeds"][:1]
        for seed in seeds:
            with np.load(OUTPUT / "formal" / dataset / f"s{seed}" / "predictions.npz") as archive:
                tr = archive["train_indices"]
            votes = np.sign(witness_margins(X[tr], y[tr], task, seed)).astype(np.int8)
            yield dataset, seed, votes, ceil(.1*votes.shape[1])


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    stress = []
    for name, p in [("homogeneous", 20), ("reversal", 20), ("zero", 20),
                    ("near", 128), ("near", 512), ("near", 4096)]:
        votes = stress_case(p, name)
        h = 1 if name == "near" else p//2
        rates = rational_rates(votes)
        minima = rates.min(axis=0)
        truth = np.sort(np.argsort(-minima, kind="stable")[:h])
        for bits in (0, 12, 40):
            selected, info = rational_top_from_votes(votes, h, bits=bits)
            np.testing.assert_array_equal(truth, selected)
            np.testing.assert_array_equal(truth, rational_top_from_votes(votes, h, bits=bits, prune=False)[0])
            float_selection = top_from_votes(votes, h)[0]
            stress.append(dict(case=name, p=p, h=h, selected=selected.tolist(), **info,
                               float_support=float_selection.tolist(), delta=str(Fraction(minima[1]-minima[0]))))
    (output / "stress.json").write_text(json.dumps(stress, indent=2), encoding="utf-8")

    trials, cases = [], []
    for dataset, seed, votes, h in inputs():
        w, p = votes.shape
        key = dict(dataset=dataset, seed=seed, w=w, p=p, h=h)
        methods = {"integer_full": lambda: rational_top_from_votes(votes, h, prune=False),
                   "integer_cert": lambda: rational_top_from_votes(votes, h),
                   "float_cert": lambda: top_from_votes(votes, h)}
        truth = methods["integer_full"]()[0]
        selected, diagnostic = methods["integer_cert"]()
        np.testing.assert_array_equal(truth, selected)
        approx = methods["float_cert"]()[0]
        cases.append(dict(**key, **diagnostic, float_matches=np.array_equal(approx, truth),
                          float_only=json.dumps(np.setdiff1d(approx, truth).tolist()),
                          rational_only=json.dumps(np.setdiff1d(truth, approx).tolist())))
        rng = np.random.default_rng([seed, p, w, 17])
        for repeat in range(5):
            for order, method in enumerate(rng.permutation(list(methods))):
                start = perf_counter_ns()
                result, _ = methods[method]()
                elapsed_ms = (perf_counter_ns()-start)/1e6
                np.testing.assert_array_equal(result, approx if method == "float_cert" else truth)
                trials.append(dict(**key, repeat=repeat, order=order, method=method, elapsed_ms=elapsed_ms))
    write_csv(output/"cases.csv", cases)
    write_csv(output/"trials.csv", trials)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"experiments/minpi_study_frontier.py", root/"src/minppi_mofs/integer_minpi.py",
             root/"src/minppi_mofs/certified_minpi.py", root/"docs/minpi-study/integer-protocol.md"]
    manifest = {path.relative_to(root).as_posix(): file_hash(path) for path in paths}
    (output/"sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    with threadpool_limits(1):
        run(parser.parse_args().output)
