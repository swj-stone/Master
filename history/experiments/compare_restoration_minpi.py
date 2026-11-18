"""Paired replay of restoration votes, their two peer layers, and the earlier controls."""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_iteration_benchmark import (
    BUDGETS, DIGESTS, PHASES, ROOT, SEEDS, SPLIT_SEED, file_hash, phase_panels,
    quality, select, summarize, train,
)
from minppi_mofs.restoration_minpi import restoration_top_from_training


EARLIER = ROOT / "results/minpi-study/predictive-iteration-20261117a"
RESTORE = ("restore", "restore_zero", "restore_target")
CONTROLS = ("grouped", "refit", "PFI", "PFI_large", "full")


def bench(output, earlier, phase, panel, repeats):
    prefix = f"{phase}-{panel.source}-{panel.fold}"
    X, y, Xt = panel.X[panel.tr], panel.y[panel.tr], panel.X[panel.te]
    groups = panel.groups[panel.tr]
    with np.load(earlier / f"{prefix}-split.npz") as split:
        for key, values in (("train", panel.tr), ("test", panel.te), ("groups", panel.groups[panel.te]),
                            ("y", panel.y[panel.te])):
            np.testing.assert_array_equal(split[key], values)
    np.savez_compressed(output / f"{prefix}-split.npz", train=panel.tr, test=panel.te,
                        train_groups=groups, test_groups=panel.groups[panel.te], y=panel.y[panel.te])
    grouped = panel.source in DIGESTS
    methods = [*RESTORE]
    if grouped:
        methods.append("restore_iid")
    methods.extend(m for m in CONTROLS if grouped or m != "PFI_large")
    rows = []
    for seed in SEEDS:
        saved = {}
        order = np.random.default_rng([seed, SPLIT_SEED, panel.fold, 8713]).permutation(methods)
        with np.load(earlier / f"{prefix}-s{seed}.npz") as baseline:
            for budget in BUDGETS:
                count = ceil(X.shape[1] * budget / 100)
                target = X.shape[1] - count
                tiers = {"restore_zero": (0,), "restore_target": (target,)}
                for rep in range(repeats):
                    for position, method in enumerate(np.roll(order, rep)):
                        start = perf_counter_ns()
                        if method.startswith("restore"):
                            columns, info = restoration_top_from_training(
                                X, y, count, task=panel.task, random_state=seed, tiers=tiers.get(method),
                                groups=groups if grouped and method != "restore_iid" else None,
                                ordered_groups=panel.source == "gas" and method != "restore_iid")
                        else:
                            columns = select(method, X, y, groups, panel, seed, count)
                        selected = perf_counter_ns()
                        a, b = (X, Xt) if method == "full" else (X[:, columns], Xt[:, columns])
                        pred, _ = train("ridge", a, y, b, panel.task, seed)
                        end = perf_counter_ns()
                        expected = len(columns) == (X.shape[1] if method == "full" else count)
                        if (not expected or not np.array_equal(np.unique(columns), columns)
                                or columns[0] < 0 or columns[-1] >= X.shape[1]):
                            raise AssertionError(f"Invalid support for {method}")
                        key = f"b{budget}_{method}"
                        for field, value in (("columns", columns), ("pred", pred)):
                            name = f"{key}_{field}"
                            if method in CONTROLS and rep == 0:
                                np.testing.assert_array_equal(baseline[name], value)
                            if rep:
                                np.testing.assert_array_equal(saved[name], value)
                            else:
                                saved[name] = value
                        if method.startswith("restore") and rep == 0:
                            fi, va = info["fit_rows"], info["assessment_rows"]
                            if np.intersect1d(fi, va).size:
                                raise AssertionError("Inner audit rows used to fit screen")
                            if grouped and method != "restore_iid" and np.intersect1d(groups[fi], groups[va]).size:
                                raise AssertionError("Inner audit group used to fit screen")
                            saved[f"{key}_fit"] = panel.tr[fi]
                            saved[f"{key}_assess"] = panel.tr[va]
                            for field in ("tiers", "scores", "layer_scores"):
                                saved[f"{key}_{field}"] = np.asarray(info[field])
                        rows.append(dict(phase=phase, source=panel.source, fold=panel.fold, seed=seed,
                                         method=method, budget=budget, width=len(columns), repeat=rep,
                                         position=position, score=quality(panel, pred),
                                         select_ms=(selected-start)/1e6, total_ms=(end-start)/1e6))
        np.savez_compressed(output / f"{prefix}-s{seed}.npz", **saved)
    path = output / "trials.csv"
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"Finished {prefix}: {len(rows)} calls", flush=True)


def run(output, earlier, repeats):
    previous = json.loads((earlier / "protocol.json").read_text(encoding="utf-8"))
    paths = [ROOT / name for name in previous["source_hashes"]]
    for path in paths:
        if file_hash(path) != previous["source_hashes"][str(path.relative_to(ROOT))]:
            raise ValueError(f"Earlier experimental sources changed: {path}")
    paths += [Path(__file__), ROOT / "src/minppi_mofs/restoration_minpi.py",
              ROOT / "src/minppi_mofs/masked_loss_minpi.py"]
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in paths}
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(primary="development/Gas/50%, historical development target",
                    phases=PHASES, seeds=SEEDS, budgets=BUDGETS, repeats=repeats, threads=1,
                    earlier=str(earlier.resolve()), earlier_protocol_sha256=file_hash(earlier / "protocol.json"),
                    source_sha256=hashes, matched_pfi="restore_zero: no other disturbed feature",
                    target="restore_target: p-h other features perturbed",
                    restore="Min over zero and p-h other disturbed features",
                    groups="Gas latest outer-training batch, materials unordered outer-training groups; restore_iid omits groups; transfer data use row-wise splits",
                    timing="Independent screens from raw outer training data through prediction; three rotated runs, equal seed and fold score, median times")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for phase in PHASES:
        for panel in phase_panels(phase):
            bench(output, earlier, phase, panel, repeats)
    summarize(output)
    old = pd.read_csv(earlier / "cases.csv")
    measured = pd.read_csv(output / "cases.csv")
    fields = ["phase","source","fold","seed","budget","method","score"]
    old_only = old[~old.method.isin(CONTROLS)][fields].assign(run="earlier")
    paired = pd.concat([measured[fields].assign(run="current"), old_only], ignore_index=True)
    paired.to_csv(output / "paired_quality.csv", index=False)
    if hashes != {str(path.relative_to(ROOT)): file_hash(path) for path in paths}:
        raise RuntimeError("Source files changed during the benchmark")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--earlier", type=Path, default=EARLIER)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Use positive repeats")
    with threadpool_limits(limits=1):
        run(args.output, args.earlier, args.repeats)
