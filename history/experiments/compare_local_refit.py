"""Compare adjacent-prefix minPI against its full-prefix parent on fixed, previously seen panels."""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_iteration_benchmark import (
    BUDGETS, PHASES, ROOT, SEEDS, SPLIT_SEED, file_hash, phase_panels, quality, select, summarize, train,
)
from minppi_mofs.local_refit_minpi import local_refit_top_from_training


PARENT = ROOT / "results/minpi-study/predictive-iteration-20261117a"
TIMED = ("local_refit", "refit", "F", "PFI", "full")


def measure(phase, panel, output):
    prefix = f"{phase}-{panel.source}-{panel.fold}"
    X, y, Xt = panel.X[panel.tr], panel.y[panel.tr], panel.X[panel.te]
    groups = panel.groups[panel.tr]
    with np.load(PARENT / f"{prefix}-split.npz") as old:
        for name, value in (("train", panel.tr), ("test", panel.te), ("y", panel.y[panel.te]),
                            ("groups", panel.groups[panel.te])):
            np.testing.assert_array_equal(old[name], value)
    observations = []
    for seed in SEEDS:
        saved = {}
        rng = np.random.default_rng([seed, SPLIT_SEED, panel.fold, 7039])
        with np.load(PARENT / f"{prefix}-s{seed}.npz") as oracle:
            for budget in BUDGETS:
                count = ceil(X.shape[1] * budget / 100)
                order = rng.permutation(TIMED)
                for rep in range(3):
                    for pos, method in enumerate(np.roll(order, rep)):
                        start = perf_counter_ns()
                        if method == "local_refit":
                            columns, _ = local_refit_top_from_training(
                                X, y, count, task=panel.task, random_state=seed)
                        else:
                            columns = select(method, X, y, groups, panel, seed, count)
                        ranked = perf_counter_ns()
                        a, b = (X, Xt) if method == "full" else (X[:, columns], Xt[:, columns])
                        pred, _ = train("ridge", a, y, b, panel.task, seed)
                        end = perf_counter_ns()
                        expected = X.shape[1] if method == "full" else count
                        if (len(columns) != expected or not np.array_equal(columns, np.unique(columns))
                                or columns[0] < 0 or columns[-1] >= X.shape[1]):
                            raise AssertionError(f"Invalid {method} support")
                        for kind, value in (("columns", columns), ("pred", pred)):
                            key = f"b{budget}_{method}_{kind}"
                            if method != "local_refit":
                                np.testing.assert_array_equal(oracle[key], value)
                            elif rep:
                                np.testing.assert_array_equal(saved[key], value)
                            else:
                                saved[key] = value
                        observations.append(dict(phase=phase, source=panel.source, fold=panel.fold,
                                                 seed=seed, budget=budget, method=method, repeat=rep,
                                                 width=len(columns), position=pos, score=quality(panel, pred),
                                                 select_ms=(ranked-start)/1e6, total_ms=(end-start)/1e6))
        np.savez_compressed(output / f"{prefix}-s{seed}.npz", **saved)
    path = output / "trials.csv"
    pd.DataFrame(observations).to_csv(path, mode="a", index=False, header=not path.exists())
    print(prefix, len(observations), flush=True)


def run(output):
    parent = PARENT / "protocol.json"
    earlier = json.loads(parent.read_text(encoding="utf-8"))
    hashes = dict(earlier["source_hashes"])
    for path in (Path(__file__), ROOT / "src/minppi_mofs/local_refit_minpi.py",
                 ROOT / "src/minppi_mofs/masked_loss_minpi.py"):
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
    def checked_sources():
        for name, checksum in hashes.items():
            if file_hash(ROOT / name) != checksum:
                raise ValueError(f"Source changed: {name}")
    checked_sources()
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(parent=str(PARENT), earlier_protocol_sha256=file_hash(parent), phases=PHASES,
                    primary="Gas 50% and material 10% following previously seen results",
                    methods=TIMED, budgets=BUDGETS, seeds=SEEDS, repeats=3, threads=1,
                    rules="Default adjacent-prefix min vs original running min, with fixed controls",
                    splits="Identical train/test sets, group labels and outcomes verified against parent split archives",
                    saved="New support and predictions stored here; every rerun of each control verified against parent archive",
                    source_sha256=hashes)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for phase in PHASES:
        for panel in phase_panels(phase):
            measure(phase, panel, output)
    summarize(output)
    checked_sources()
    records = pd.read_csv(output / "cases.csv")
    earlier_cases = pd.read_csv(PARENT / "cases.csv")
    keys = ["phase", "source", "budget", "fold", "seed", "method"]
    keep = keys + ["score"]
    old = earlier_cases.loc[~earlier_cases.method.isin(TIMED), keep].assign(run="parent")
    table = pd.concat([records[keep].assign(run="current"), old], ignore_index=True)
    table.to_csv(output / "paired_quality.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
