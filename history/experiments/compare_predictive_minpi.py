"""Compare fixed-budget minPI variants from raw training rows through ridge prediction."""
import argparse
import json
import platform
import warnings
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_wine
from sklearn.feature_selection import f_classif, f_regression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.minpi_study import CONFIG, ROOT, file_hash, train
from experiments.minpi_study_new_cohorts import DIGESTS, batch_scores, load_cohort
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.conditional_minpi import conditional_top_from_votes
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.grouped_minpi import grouped_control_from_training, grouped_top_from_training
from minppi_mofs.loss_minpi import held_loss_votes, loss_top_from_training
from minppi_mofs.word_minpi import exact_top_from_votes


METHODS = ("rational", "float", "conditional", "grouped", "grouped_control", "loss",
           "loss_vote_mean", "f", "random", "full")
SEEDS = (1129, 2027, 3469)
SPLIT_SEED = 41437


@dataclass
class Panel:
    source: str
    fold: int
    task: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    tr: np.ndarray
    te: np.ndarray


def panels(phase, sources):
    for name in sources:
        if name == "wine":
            X, y = load_wine(return_X_y=True)
            groups = np.arange(len(y))
            folds = StratifiedKFold(3, shuffle=True, random_state=SPLIT_SEED)
            for fold, (tr, te) in enumerate(folds.split(X, y), start=1):
                yield Panel(name, fold, "classification", X, y, groups, tr, te)
            continue
        X, y, groups, development, historical_test, task = load_cohort(name)
        if phase == "replay":
            yield Panel(name, 1, task, X, y, groups, development, historical_test)
        elif name == "gas":
            for batch in (4, 5, 6, 7):
                tr = development[groups[development] < batch]
                te = development[groups[development] == batch]
                yield Panel(name, batch, task, X, y, groups, tr, te)
        else:
            # Keep duplicate descriptor vectors and normalized compositions together.
            dev_groups = groups[development]
            permuted = np.random.default_rng(SPLIT_SEED).permutation(np.unique(dev_groups))
            for fold, held_groups in enumerate(np.array_split(permuted, 3), start=1):
                held = np.isin(dev_groups, held_groups)
                yield Panel(name, fold, task, X, y, groups, development[~held], development[held])


def choose(method, X, y, task, seed, count):
    if method == "full":
        return np.arange(X.shape[1])
    if method == "random":
        return np.sort(np.random.default_rng(seed).permutation(X.shape[1])[:count])
    if method == "f":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            scores = (f_classif if task == "classification" else f_regression)(X, y)[0]
        rank = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
        return np.sort(rank[:count])
    if method in ("grouped", "grouped_control"):
        scorer = grouped_top_from_training if method == "grouped" else grouped_control_from_training
        return scorer(X, y, count, task=task, random_state=seed)[0]
    if method == "loss":
        return loss_top_from_training(X, y, count, task=task, random_state=seed)[0]
    if method == "loss_vote_mean":
        votes = held_loss_votes(X, y, task, seed)
        order = np.argsort(-votes.sum(axis=0), kind="stable")
        return np.sort(order[:count])
    votes = np.sign(witness_margins(X, y, task, seed)).astype(np.int8)
    scorer = {"rational": exact_top_from_votes, "float": top_from_votes,
              "conditional": conditional_top_from_votes}[method]
    return scorer(votes, count)[0]


def quality(panel, prediction):
    gold = panel.y[panel.te]
    if panel.source == "gas":
        return np.mean(list(batch_scores(gold, prediction, panel.groups[panel.te]).values()))
    if panel.task == "regression":
        return r2_score(gold, prediction)
    return balanced_accuracy_score(gold, prediction)


def bench(panel, output, budget, seeds, repeats):
    X, y, Xt = panel.X[panel.tr], panel.y[panel.tr], panel.X[panel.te]
    count = ceil(X.shape[1] * budget / 100)
    # Sequential conditioning scales with the number selected; only evaluate it
    # at the primary budget, where all methods receive the same search budget.
    methods = [m for m in METHODS if m != "conditional" or budget == 10]
    if np.intersect1d(panel.groups[panel.tr], panel.groups[panel.te]).size:
        raise ValueError("Overlapping train and evaluation groups")
    panel_id = f"{panel.source}-{panel.fold}-b{budget}"
    for seed in seeds:
        records, saved = [], {}
        rotation = np.random.default_rng(np.random.SeedSequence([SPLIT_SEED, seed, panel.fold]))
        initial = rotation.permutation(methods)
        for repeat in range(repeats):
            for position, method in enumerate(np.roll(initial, repeat)):
                start = perf_counter_ns()
                support = choose(method, X, y, panel.task, seed, count)
                selected = perf_counter_ns()
                x_fit, x_eval = (X, Xt) if method == "full" else (X[:, support], Xt[:, support])
                pred, _ = train("ridge", x_fit, y, x_eval, panel.task, seed)
                end = perf_counter_ns()
                support_key, pred_key = f"{method}_support", f"{method}_prediction"
                expected = X.shape[1] if method == "full" else count
                if len(support) != expected or not np.array_equal(np.unique(support), support):
                    raise AssertionError("Invalid support")
                if repeat:
                    np.testing.assert_array_equal(saved[support_key], support)
                    np.testing.assert_array_equal(saved[pred_key], pred)
                else:
                    saved[support_key], saved[pred_key] = support, pred
                records.append(dict(source=panel.source, fold=panel.fold, budget=budget,
                                    seed=seed, repeat=repeat,
                                    position=position, method=method, width=len(support),
                                    metric="R2" if panel.task == "regression" else "BA",
                                    score=quality(panel, pred), select_ms=(selected-start)/1e6,
                                    downstream_ms=(end-selected)/1e6, total_ms=(end-start)/1e6))
        np.savez_compressed(output / f"{panel_id}-{seed}.npz", **saved)
        trials = output / "trials.csv"
        pd.DataFrame(records).to_csv(trials, index=False, header=not trials.exists(), mode="a")
        print(f"Measured {panel_id} seed {seed}", flush=True)


def summarize(output):
    rows = pd.read_csv(output / "trials.csv")
    keys = ["source", "fold", "budget", "seed"]
    ref = rows[rows.method == "rational"][keys + ["repeat", "score", "select_ms", "total_ms"]]
    pairs = rows.merge(ref, on=keys+["repeat"], suffixes=("", "_rational"), validate="many_to_one")
    pairs["delta_rational"] = pairs.score - pairs.score_rational
    for kind in ("total", "select"):
        pairs[f"{kind}_ratio"] = pairs[f"{kind}_ms"] / pairs[f"{kind}_ms_rational"]
    full = rows[rows.method == "full"][keys + ["repeat", "score", "total_ms"]]
    pairs = pairs.merge(full, on=keys+["repeat"], suffixes=("", "_full"), validate="many_to_one")
    pairs["delta_full"] = pairs.score - pairs.score_full
    pairs["total_ratio_full"] = pairs.total_ms / pairs.total_ms_full
    values = ["score", "delta_rational", "delta_full", "select_ms", "total_ms", "select_ratio",
              "total_ratio", "total_ratio_full"]
    cases = pairs.groupby(keys+["method"], as_index=False)[values].median()
    cases.to_csv(output / "cases.csv", index=False)
    # Average seed scores within each held-out fold, weight folds equally.
    # Timing ratios use the paired median across repeats, seeds, then folds.
    score_cols = values[:3]
    timing_cols = values[3:]
    aggregations = dict.fromkeys(score_cols, "mean") | dict.fromkeys(timing_cols, "median")
    folds = cases.groupby(["source", "fold", "budget", "method"], as_index=False).agg(aggregations)
    folds.to_csv(output / "folds.csv", index=False)
    summary = folds.groupby(["source", "budget", "method"], as_index=False).agg(aggregations)
    summary.to_csv(output / "summary.csv", index=False)


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), CONFIG, ROOT / "experiments/minpi_study.py",
                    ROOT / "experiments/minpi_study_new_cohorts.py"]
    for name in ("conditional_minpi", "grouped_minpi", "loss_minpi", "word_minpi", "certified_minpi",
                 "integer_minpi", "exact_minpi", "piv3_compact_filter", "piv2"):
        source_files.append(ROOT / f"src/minppi_mofs/{name}.py")
    checksums = {str(path.relative_to(ROOT)): file_hash(path) for path in source_files}
    metadata = dict(phase=args.phase, methods=list(METHODS), sources=args.sources, seeds=SEEDS,
                    budgets=args.budgets, conditional_budgets=[10],
                    repeats=args.repeats, split_seed=SPLIT_SEED, threads=1,
                    source_sha256=checksums, archives=DIGESTS, python=platform.python_version(),
                    numpy=np.__version__, sklearn=sklearn.__version__, platform=platform.platform(),
                    cpu=platform.processor())
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for panel in panels(args.phase, args.sources):
        np.savez_compressed(output / f"{panel.source}-{panel.fold}-split.npz", train=panel.tr,
                            test=panel.te, group=panel.groups[panel.te], y=panel.y[panel.te])
        for budget in args.budgets:
            bench(panel, output, budget, SEEDS, args.repeats)
    summarize(output)
    for path in source_files:
        if file_hash(path) != checksums[str(path.relative_to(ROOT))]:
            raise RuntimeError(f"Sources changed during the benchmark: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("development", "replay"), default="development")
    parser.add_argument("--sources", nargs="+", choices=("gas", "supercon", "wine"),
                        default=["gas", "supercon", "wine"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", choices=(10, 25, 50), default=[10, 25, 50])
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.sources)) != len(args.sources):
        parser.error("Use positive repeats and unique sources")
    if args.phase == "replay" and "wine" in args.sources:
        parser.error("The historical replay phase takes gas and/or supercon")
    with threadpool_limits(limits=1):
        run(args)
