"""Choose a ridge screen and its budget on development folds, then replay historical holdouts."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import (
    ROOT, SEEDS, SPLIT_SEED, ceil, choose, file_hash, panels, quality, train,
)


FLOOR_DELTA = .02
HIGH_BUDGET = ("rational", "float", "grouped", "loss", "f", "full")
MINPI = ("rational", "float", "conditional", "grouped", "loss")
SOURCES = ("gas", "supercon")


def timed(panel, configs, output, stage, repeats):
    X, y, Xt = panel.X[panel.tr], panel.y[panel.tr], panel.X[panel.te]
    if np.intersect1d(panel.groups[panel.tr], panel.groups[panel.te]).size:
        raise AssertionError("Split groups overlap")
    records = []
    for seed in SEEDS:
        saved = {}
        order = np.random.default_rng(np.random.SeedSequence([SPLIT_SEED, seed, panel.fold]))
        permutation = order.permutation(len(configs))
        for repeat in range(repeats):
            for position, config_index in enumerate(np.roll(permutation, repeat)):
                method, budget = configs[config_index]
                count = ceil(X.shape[1] * budget / 100)
                start = perf_counter_ns()
                support = choose(method, X, y, panel.task, seed, count)
                selected = perf_counter_ns()
                x_fit, x_eval = (X, Xt) if method == "full" else (X[:, support], Xt[:, support])
                pred, _ = train("ridge", x_fit, y, x_eval, panel.task, seed)
                end = perf_counter_ns()
                prefix = f"{method}-{budget}"
                if repeat:
                    np.testing.assert_array_equal(pred, saved[prefix+"-prediction"])
                    np.testing.assert_array_equal(support, saved[prefix+"-support"])
                else:
                    saved[prefix+"-prediction"] = pred
                    saved[prefix+"-support"] = support
                if len(support) != count or not np.array_equal(support, np.unique(support)):
                    raise AssertionError("Invalid selected columns")
                records.append(dict(source=panel.source, fold=panel.fold, seed=seed,
                                    method=method, budget=budget, repeat=repeat, position=position,
                                    width=count, score=quality(panel, pred),
                                    select_ms=(selected-start)/1e6, total_ms=(end-start)/1e6))
        np.savez_compressed(output / f"{stage}-{panel.source}-{panel.fold}-{seed}.npz", **saved)
    np.savez_compressed(output / f"{stage}-{panel.source}-{panel.fold}-split.npz", train=panel.tr,
                        test=panel.te, groups=panel.groups[panel.te], y=panel.y[panel.te])
    print(f"Completed {stage} {panel.source} fold {panel.fold}", flush=True)
    return pd.DataFrame(records)


def aggregate(trials):
    # Equal weight to folds and selector seeds; avoid counting repeated timings as new outcomes.
    case = trials.groupby(["source", "fold", "seed", "method", "budget"], as_index=False).agg(
        score=("score", "first"), fit_ms=("total_ms", "median"),
        measured_ms=("total_ms", "sum"), width=("width", "first"))
    return case.groupby(["source", "method", "budget"], as_index=False).agg(
        inner_score=("score", "mean"), mean_fit_ms=("fit_ms", "mean"),
        single_pass_ms=("fit_ms", "sum"), measured_ms=("measured_ms", "sum"),
        width=("width", "first"), n_cases=("score", "size"))


def pick(choices):
    selected = []
    for source in SOURCES:
        table = choices[choices.source == source]
        full_score = float(table.loc[table.method == "full", "inner_score"].item())
        for policy, methods in (("minpi", MINPI), ("F", ("f",))):
            grid = table[table.method.isin((*methods, "full"))]
            qualified = grid[grid.inner_score >= full_score - FLOOR_DELTA]
            best = qualified.sort_values(["mean_fit_ms", "inner_score", "method", "budget"],
                                         ascending=[True, False, True, True]).iloc[0]
            selected.append(dict(source=source, policy=policy, method=best.method,
                                 budget=int(best.budget), inner_full=full_score,
                                 inner_score=float(best.inner_score), n_configs=len(grid),
                                 n_eligible=len(qualified), mean_fit_ms=float(best.mean_fit_ms),
                                 search_one_pass_s=grid.single_pass_ms.sum()/1000,
                                 search_all_repeats_s=grid.measured_ms.sum()/1000))
    return pd.DataFrame(selected)


def publish_outer(trials, selection, output):
    records = []
    for source in SOURCES:
        first = selection[selection.source == source]
        options = {row.policy: (row.method, row.budget) for row in first.itertuples()}
        options.update(full=("full", 100), fixed_original=("rational", 10))
        data = trials[trials.source == source]
        keys = ["seed", "repeat"]
        full = data[data.method == "full"][keys+["score", "total_ms"]]
        for label, (method, budget) in options.items():
            row = data[(data.method == method) & (data.budget == budget)]
            row = row.merge(full, on=keys, suffixes=("", "_full"), validate="one_to_one")
            by_seed = row.assign(ratio=row.total_ms/row.total_ms_full).groupby("seed")
            records.append(dict(source=source, policy=label, method=method, budget=budget,
                                width=int(row.width.iloc[0]), score=by_seed.score.first().mean(),
                                delta_full=(row.score-row.score_full).mean(),
                                median_fit_ms=by_seed.total_ms.median().median(),
                                median_time_ratio_full=by_seed.ratio.median().median()))
    pd.DataFrame(records).to_csv(output / "outer-summary.csv", index=False)


def run(output, base, repeats):
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in previous["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError(f"Source has changed since the development benchmark: {name}")
    config = dict(tolerance=FLOOR_DELTA, budgets=[10,25,50,75,100],
                  policy_comparison="minPI variants plus full; F budgets plus full",
                  objective="minimize average case-median end-to-end milliseconds subject to mean validation quality within tolerance of full",
                  inner_base_sha256=file_hash(base/"trials.csv"), inner_code_sha256=previous["source_sha256"],
                  this_code_sha256=file_hash(Path(__file__)), seeds=SEEDS, timing_repeats=repeats,
                  historical_outer=True, cpu=platform.processor(), threads=1)
    (output / "protocol.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    prior = pd.read_csv(base/"trials.csv")
    prior = prior[prior.source.isin(SOURCES) & prior.method.isin((*MINPI, "f", "full"))].copy()
    # One full-width comparator per fold/seed. It was already timed without a column copy.
    prior = prior[(prior.method != "full") | (prior.budget == 10)].copy()
    prior.loc[prior.method == "full", "budget"] = 100
    additional = []
    for panel in panels("development", SOURCES):
        with np.load(base/f"{panel.source}-{panel.fold}-split.npz") as split:
            np.testing.assert_array_equal(panel.tr, split["train"])
            np.testing.assert_array_equal(panel.te, split["test"])
        configs = [(m, 100 if m == "full" else 75) for m in HIGH_BUDGET]
        additional.append(timed(panel, configs, output, "inner75", repeats))
    additional = pd.concat(additional, ignore_index=True)
    additional.to_csv(output / "inner75-trials.csv", index=False)
    # The new full timings are a session drift check, never another config in the policy search.
    grid = aggregate(pd.concat([prior, additional[additional.method != "full"]], ignore_index=True))
    grid.to_csv(output / "inner-grid.csv", index=False)
    selection = pick(grid)
    selection.to_csv(output / "frozen-selection.csv", index=False)
    print(selection.to_string(index=False), flush=True)
    outer = []
    for panel in panels("replay", SOURCES):
        subset = selection[selection.source == panel.source]
        configs = sorted({(r.method, r.budget) for r in subset.itertuples()} |
                         {("full", 100), ("rational", 10)})
        outer.append(timed(panel, configs, output, "outer", repeats))
    outer = pd.concat(outer, ignore_index=True)
    outer.to_csv(output / "outer-trials.csv", index=False)
    publish_outer(outer, selection, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base", type=Path, default=ROOT/"results/minpi-study/predictive-redesign-dev-20261117c")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive timing repeats required")
    with threadpool_limits(limits=1):
        run(args.output, args.base, args.repeats)
