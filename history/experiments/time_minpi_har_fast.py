"""Audit accelerated local refitting against archived HAR predictions while timing it."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.time_minpi_har import EARLIER, ROOT, SEEDS, load_har, screen, sha256, train
from minppi_mofs.local_refit_fast import local_refit_fast_top_from_training


METHODS = ("fast", "local_refit", "grouped", "matched_PFI", "full")


def run(output):
    previous = json.loads((EARLIER / "protocol.json").read_text(encoding="utf-8"))
    sources = dict(previous["source_sha256"])
    for path, expected in sources.items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"HAR confirmation source changed: {path}")
    for path in [Path(__file__), ROOT / "experiments/time_minpi_har.py",
                 ROOT / "src/minppi_mofs/local_refit_fast.py"]:
        sources[path.relative_to(ROOT).as_posix()] = sha256(path)
    saved = EARLIER / "supports-and-predictions.npz"
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(dict(
        sources=sources, oracle_sha256=sha256(saved), seeds=SEEDS, methods=METHODS, times_per_seed=3,
        aggregation="Median over repeats and then median over selector seeds, within method. Single thread.",
        timing="Raw inputs through selection, column slicing, ridge training and prediction. "
               "Loading, support/prediction audit and summary are outside the clock.",
        equivalence="All 45 outputs checked against prior HAR predictions and selected columns. "
                    "The accelerated implementation uses the archived local_refit oracle."),
        indent=2), encoding="utf-8")
    parts, _ = load_har(False)
    X, y, train_subjects = parts["train"]
    Xt, truth, test_subjects = parts["test"]
    rows = []
    with np.load(saved) as oracle:
        np.testing.assert_array_equal(oracle["truth"], truth)
        np.testing.assert_array_equal(oracle["train_subjects"], train_subjects)
        np.testing.assert_array_equal(oracle["test_subjects"], test_subjects)
        for seed in SEEDS:
            order = np.random.default_rng([seed, 1297]).permutation(METHODS)
            for r in range(3):
                for position, method in enumerate(np.roll(order, 2*r)):
                    start = perf_counter_ns()
                    if method == "fast":
                        cols = local_refit_fast_top_from_training(X, y, 281, random_state=seed)[0]
                    else:
                        cols = screen(method, X, y, train_subjects, seed)
                    selected = perf_counter_ns()
                    fit, test = (X, Xt) if method == "full" else (X[:, cols], Xt[:, cols])
                    pred, _ = train("ridge", fit, y, test, "classification", seed)
                    end = perf_counter_ns()
                    label = "local_refit" if method == "fast" else method
                    key = f"s{seed}_{100 if method == 'full' else 50}_{label}"
                    np.testing.assert_array_equal(oracle[key + "_support"], cols)
                    np.testing.assert_array_equal(oracle[key + "_pred"], pred)
                    rows.append(dict(seed=seed, method=method, repeat=r, position=position,
                                     selection_ms=(selected-start)/1e6, total_ms=(end-start)/1e6))
            print(f"Validated fast HAR seed {seed}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "timings.csv", index=False)
    group = frame.groupby(["method", "seed"])[["selection_ms", "total_ms"]].median()
    summary = group.groupby("method").median()
    summary.to_csv(output / "summary.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
