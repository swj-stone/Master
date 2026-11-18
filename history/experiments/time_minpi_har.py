"""Time matched HAR pipelines at the previously evaluated 281-column budget."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_har_confirmation import (
    ROOT, SEEDS, choose, full_fit_pfi, load_har, local_refit_top_from_training,
    refit_top_from_training, restoration_top_from_training, sha256, train,
)


METHODS = ("local_refit", "refit", "grouped", "original", "matched_PFI", "full_fit_PFI", "full")
EARLIER = ROOT / "results/minpi-study/har-subject-confirmation-20261117a"


def screen(method, X, y, subjects, seed):
    count = 281
    if method == "full":
        return np.arange(X.shape[1])
    if method in ("refit", "local_refit"):
        selector = refit_top_from_training if method == "refit" else local_refit_top_from_training
        return selector(X, y, count, random_state=seed)[0]
    if method == "matched_PFI":
        return restoration_top_from_training(X, y, count, groups=subjects, random_state=seed, tiers=(0,))[0]
    if method == "full_fit_PFI":
        rank, _ = full_fit_pfi(X, y, subjects, seed)
        return np.sort(rank[:count])
    return choose("float" if method == "original" else method, X, y, "classification", seed, count)


def run(output):
    earlier_protocol = EARLIER / "protocol.json"
    previous = json.loads(earlier_protocol.read_text(encoding="utf-8"))
    for path, digest in previous["source_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise RuntimeError(f"Source differs from quality run: {path}")
    output.mkdir(parents=True, exist_ok=False)
    oracle = EARLIER / "supports-and-predictions.npz"
    protocol = dict(script_sha256=sha256(Path(__file__)), previous_protocol_sha256=sha256(earlier_protocol),
                    oracle_sha256=sha256(oracle), methods=METHODS, seeds=SEEDS, repeats=3, threads=1,
                    timing="Raw input: training-only selection, column extraction, fit and predict. "
                           "Full uses raw complete matrices directly. Excludes reading files and oracle checks.",
                    aggregation="Per-method median of three rotated runs within seed, then median over seeds.")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    parts, _ = load_har(False)
    X, y, subjects = parts["train"]
    Xt, truth, test_subjects = parts["test"]
    rows = []
    with np.load(oracle) as saved:
        np.testing.assert_array_equal(test_subjects, saved["test_subjects"])
        np.testing.assert_array_equal(truth, saved["truth"])
        np.testing.assert_array_equal(subjects, saved["train_subjects"])
        for seed in SEEDS:
            order = np.random.default_rng([seed, 7071]).permutation(METHODS)
            for rep in range(3):
                for position, method in enumerate(np.roll(order, 2 * rep)):
                    start = perf_counter_ns()
                    cols = screen(method, X, y, subjects, seed)
                    selected = perf_counter_ns()
                    fit, test = (X, Xt) if method == "full" else (X[:, cols], Xt[:, cols])
                    pred, _ = train("ridge", fit, y, test, "classification", seed)
                    end = perf_counter_ns()
                    budget = 100 if method == "full" else 50
                    key = f"s{seed}_{budget}_{method}"
                    np.testing.assert_array_equal(cols, saved[key + "_support"])
                    np.testing.assert_array_equal(pred, saved[key + "_pred"])
                    rows.append(dict(method=method, seed=seed, repeat=rep, position=position, width=len(cols),
                                     selection_ms=(selected-start)/1e6, total_ms=(end-start)/1e6))
            print(f"Validated and timed HAR seed {seed}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "timings.csv", index=False)
    inner = frame.groupby(["method", "seed"])[["selection_ms", "total_ms"]].median()
    summary = inner.groupby("method").median()
    summary.to_csv(output / "summary.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
