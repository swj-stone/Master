"""Paired, fixed-budget evaluation of the next pivotal minPI rules."""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import (
    Panel, SEEDS, SPLIT_SEED, choose, panels, quality,
)
from experiments.minpi_held_permutation_control import PERMUTATIONS, linear_permutation_losses, pfi, ridge
from experiments.minpi_study import CONFIG, ROOT, file_hash, train
from experiments.minpi_study_new_cohorts import DIGESTS
from minppi_mofs.coalition_minpi import coalition_top_from_training
from minppi_mofs.refit_minpi import refit_top_from_training
from minppi_mofs.utility_minpi import utility_top_from_training


NEW_RULES = {"utility": utility_top_from_training, "coalition": coalition_top_from_training,
             "refit": refit_top_from_training}
METHODS = (*NEW_RULES, "grouped", "original", "F", "random", "PFI", "PFI_large", "full")
BUDGETS = (10, 50)
PHASES = ("development", "replay", "transfer")


def phase_panels(phase):
    if phase in ("development", "replay"):
        yield from panels(phase, ("gas", "supercon"))
        return
    for name, loader, task in (("breast_cancer", load_breast_cancer, "classification"),
                               ("diabetes", load_diabetes, "regression")):
        X, y = loader(return_X_y=True, **({"scaled": False} if name == "diabetes" else {}))
        ids = np.arange(len(y))
        split = StratifiedKFold if task == "classification" else KFold
        for fold, (tr, te) in enumerate(split(3, shuffle=True, random_state=SPLIT_SEED).split(X, y), 1):
            yield Panel(name, fold, task, X, y, ids, tr, te)


def full_fit_pfi(X, y, task, seed, groups=None):
    rng = np.random.default_rng(seed)
    if groups is None:
        fit, assess = train_test_split(np.arange(len(y)), test_size=.25, random_state=seed,
                                       stratify=y if task == "classification" else None)
    else:
        unique = np.unique(groups)
        held = (groups == unique.max() if task == "classification"
                else np.isin(groups, rng.permutation(unique)[:max(1, len(unique)//4)]))
        fit = np.flatnonzero(~held)
        assess = np.flatnonzero(held)
        assess = rng.choice(assess, min(512, len(assess)), replace=False)
    model = ridge(task, 1.).fit(X[fit], y[fit])
    orders = [rng.permutation(len(assess)) for _ in range(PERMUTATIONS)]
    target = y[assess]
    weights = None
    if task == "classification":
        _, codes, counts = np.unique(target, return_counts=True, return_inverse=True)
        weights = 1 / (len(counts) * counts[codes])
    losses = linear_permutation_losses(model, X[assess], target, orders, task, weights)
    return np.argsort(-losses, kind="stable")


def select(method, X, y, groups, panel, seed, count):
    if method in NEW_RULES:
        return NEW_RULES[method](X, y, count, task=panel.task, random_state=seed)[0]
    if method in ("PFI", "PFI_large"):
        if panel.source in DIGESTS and method == "PFI":
            order, _ = pfi(X, y, groups, panel.task, seed, backend="linear")
        else:
            order = full_fit_pfi(X, y, panel.task, seed, groups if panel.source in DIGESTS else None)
        return np.sort(order[:count])
    return choose({"original": "float", "F": "f"}.get(method, method), X, y, panel.task, seed, count)


def bench(output, phase, panel, repeats):
    X, y, Xt = panel.X[panel.tr], panel.y[panel.tr], panel.X[panel.te]
    groups = panel.groups[panel.tr]
    if np.intersect1d(groups, panel.groups[panel.te]).size:
        raise AssertionError("Evaluation group occurs in training")
    prefix = f"{phase}-{panel.source}-{panel.fold}"
    np.savez_compressed(output / f"{prefix}-split.npz", train=panel.tr, test=panel.te,
                        groups=panel.groups[panel.te], y=panel.y[panel.te])
    rows = []
    methods = [m for m in METHODS if m != "PFI_large" or panel.source in DIGESTS]
    for seed in SEEDS:
        saved = {}
        order = np.random.default_rng([SPLIT_SEED, panel.fold, seed]).permutation(len(methods))
        for budget in BUDGETS:
            count = ceil(X.shape[1] * budget / 100)
            for rep in range(repeats):
                for position, index in enumerate(np.roll(order, rep)):
                    method = methods[index]
                    start = perf_counter_ns()
                    columns = np.asarray(select(method, X, y, groups, panel, seed, count))
                    selected = perf_counter_ns()
                    fit_x, eval_x = (X, Xt) if method == "full" else (X[:, columns], Xt[:, columns])
                    pred, _ = train("ridge", fit_x, y, eval_x, panel.task, seed)
                    end = perf_counter_ns()
                    expected = X.shape[1] if method == "full" else count
                    if (columns.shape != (expected,) or not np.array_equal(np.unique(columns), columns)
                            or columns[0] < 0 or columns[-1] >= X.shape[1]):
                        raise AssertionError(f"Invalid {method} support")
                    for what, value in (("columns", columns), ("pred", pred)):
                        key = f"b{budget}_{method}_{what}"
                        if rep:
                            np.testing.assert_array_equal(value, saved[key])
                        else:
                            saved[key] = value
                    rows.append(dict(phase=phase, source=panel.source, fold=panel.fold,
                                     budget=budget, width=expected, seed=seed, repeat=rep,
                                     method=method, position=position, score=quality(panel, pred),
                                     select_ms=(selected-start)/1e6, total_ms=(end-start)/1e6))
        np.savez_compressed(output / f"{prefix}-s{seed}.npz", **saved)
    path = output / "trials.csv"
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"Finished {prefix}: {len(rows)} calls", flush=True)


def summarize(output):
    trials = pd.read_csv(output / "trials.csv")
    keys = ["phase", "source", "budget", "method"]
    cols = ["score", "select_ms", "total_ms"]
    cases = trials.groupby(keys+["fold", "seed"], as_index=False)[cols].median()
    cases.to_csv(output / "cases.csv", index=False)
    how = dict(score="mean", select_ms="median", total_ms="median")
    folds = cases.groupby(keys+["fold"], as_index=False).agg(how)
    folds.to_csv(output / "folds.csv", index=False)
    folds.groupby(keys, as_index=False).agg(how).to_csv(output / "summary.csv", index=False)


def run(output, repeats, phases):
    sources = [Path(__file__), CONFIG]
    sources.extend(ROOT / "experiments" / name for name in (
        "compare_predictive_minpi.py", "minpi_held_permutation_control.py",
        "minpi_study.py", "minpi_study_new_cohorts.py"))
    sources.extend(ROOT / "src/minppi_mofs" / name for name in (
        "utility_minpi.py", "coalition_minpi.py", "refit_minpi.py", "grouped_minpi.py", "certified_minpi.py",
        "exact_minpi.py", "integer_minpi.py", "word_minpi.py", "piv2.py"))
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(methods=METHODS, phases=phases, budgets=BUDGETS, seeds=SEEDS,
                    split_seed=SPLIT_SEED, repeats=repeats, threads=1, sklearn=sklearn.__version__,
                    numpy=np.__version__, cohort_digests=DIGESTS, source_hashes=hashes,
                    development="Gas forward training batches 4–7; three composition-grouped materials folds",
                    replay="Previously studied outer holdouts: later gas batches and held-out materials groups",
                    transfer="Three fixed shuffled folds in breast_cancer and diabetes, stratified for classification",
                    pfi="Gas/materials: prior small screen and full-fit PFI_large with up to 512 assessment rows of separate batches/groups. For transfer, PFI fits 75% of outer training, assesses remaining 25%. Three permutations for every screen.",
                    original="Original static minPI using the project's float scorer",
                    metrics="Balanced accuracy for classification, equal batch BA for Gas, R2 for regression",
                    timing="From raw training inputs to final prediction including selection, column extraction, training, and prediction; median over repeats, seeds, then folds",
                    aggregation="Scores averaged equally over seeds then folds; repeats verify determinism")
    (output / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for phase in phases:
        for panel in phase_panels(phase):
            bench(output, phase, panel, repeats)
    if hashes != {str(path.relative_to(ROOT)): file_hash(path) for path in sources}:
        raise RuntimeError("Source changed during benchmark")
    summarize(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=PHASES)
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.phases)) != len(args.phases):
        parser.error("Select positive repeats and distinct phases")
    with threadpool_limits(limits=1):
        run(args.output, args.repeats, args.phases)
