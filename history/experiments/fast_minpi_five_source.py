"""Replay the five archived minPI sources, with end-to-end paired timings for every seed."""
import argparse
import json
import platform
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from experiments.backward_minpi_stress import ARCHIVE, SEEDS
from experiments.consensus_minpi_stress import select as consensus_select
from experiments.stable_minpi_stress import (
    DATASETS, ROOT, SPLIT_SEED, TOLERANCE, choose, digest, file_hash, folds, inputs, metric,
    prediction, write_json,
)
from minppi_mofs.sketch_minpi import reduced_width, sketch_top_from_training
from minppi_mofs.pair_vote_minpi import pair_vote_top_from_training


CONSENSUS_ARCHIVE = ROOT / "results/minpi-study/consensus-stress-20261117a"
CANDIDATES = ("sketch", "pair_vote")
CONTROLS = ("F_matched", "random_matched")
METHODS = (*CANDIDATES, *CONTROLS, "full", "consensus_policy")
CASE = ["dataset", "fold", "seed"]


def select(method, X, y, task, seed, groups, semantic_groups, widths):
    p = X.shape[1]
    if method == "full":
        return np.arange(p)
    if method == "consensus_policy":
        return consensus_select(method, X, y, task, seed, groups, semantic_groups, None)[0]
    if method == "sketch":
        return sketch_top_from_training(X, y, reduced_width(p), task=task, random_state=seed,
                                        groups=groups, ordered_groups=False)[0]
    if method == "pair_vote":
        return pair_vote_top_from_training(X, y, reduced_width(p), task=task, random_state=seed,
                                           groups=groups, ordered_groups=False)[0]
    baseline = method.split("_", maxsplit=1)[0]
    if widths["sketch"] != widths["pair_vote"]:
        raise ValueError("A shared matched-width control requires identical candidate widths")
    return choose(baseline.lower(), X, y, task, seed, widths["sketch"])


def check_columns(columns, p):
    if (columns.ndim != 1 or not len(columns) or columns.dtype.kind not in "iu"
            or columns[0] < 0 or columns[-1] >= p or not np.array_equal(np.unique(columns), columns)):
        raise ValueError("Invalid selected columns")


def data_folds():
    protocol = json.loads((ARCHIVE / "protocol.json").read_text(encoding="utf-8"))
    for name in DATASETS:
        X, y, task, groups, source, duplicates = inputs(name)
        expected = dict(source=source, X=digest(X), y=digest(y),
                        groups=None if groups is None else digest(groups))
        if expected != protocol["input_hashes"][name]:
            raise ValueError(f"Inputs changed for {name}")
        for fold, (tr, te) in enumerate(folds(X, y, task, groups), 1):
            with np.load(ARCHIVE / f"{name}-{fold}-split.npz") as split:
                np.testing.assert_array_equal(tr, split["train"])
                np.testing.assert_array_equal(te, split["test"])
                np.testing.assert_array_equal(y[te], split["gold"])
            if np.intersect1d(duplicates[tr], duplicates[te]).size:
                raise ValueError("Repeated feature vectors cross an archived split")
            if groups is not None and np.intersect1d(groups[tr], groups[te]).size:
                raise ValueError("Groups cross an archived split")
            yield name, fold, X[tr], y[tr], X[te], y[te], task, None if groups is None else groups[tr]


def protocol(repeats):
    previous = json.loads((CONSENSUS_ARCHIVE / "protocol.json").read_text(encoding="utf-8"))
    files = [ROOT / key for key in previous["sources"]]
    files += [Path(__file__), ROOT/"src/minppi_mofs/sketch_minpi.py", ROOT/"src/minppi_mofs/pair_vote_minpi.py"]
    archives = [ARCHIVE/"protocol.json", CONSENSUS_ARCHIVE/"protocol.json"]
    for name in DATASETS:
        for fold in (1, 2, 3):
            archives += [ARCHIVE/f"{name}-{fold}-split.npz", CONSENSUS_ARCHIVE/f"{name}-{fold}.npz"]
    return dict(
        sources={str(p.relative_to(ROOT)): file_hash(p) for p in files},
        archives={str(p.relative_to(ROOT)): file_hash(p) for p in archives},
        datasets=DATASETS, seeds=SEEDS, methods=METHODS, repeats=repeats,
        python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
        cpu=platform.processor(), platform=platform.platform(), threads=1, tolerance=TOLERANCE,
        interpretation="Development replay of 15 known test folds; seeds are repeated measures.",
        selection="Both candidates use only the training split. The ANOVA or regression F-test and "
                  "uniform random controls both use the common candidate width, checked in warm-up.",
        timing="One unmeasured cycle followed by fresh selection, column extraction, final standardized "
               "ridge fit and prediction in every arm and repeat. Methods rotate from a fixed shuffled "
               "order. Full uses all columns directly. Data loading, scoring and serialization excluded.",
        success="width<p AND score-full_score>=-0.02 AND median(total_ms)<median(full_total_ms); "
                "paired_ratio is the median within-repeat total/full ratio. Additional margins are "
                "ratio<=0.95 and ratio<=0.90.",
    )


def run_case(name, fold, seed, X, y, Xt, gold, task, groups, oracle, repeats):
    records, outputs, widths = [], {}, {}
    p = X.shape[1]
    methods = np.random.default_rng([SPLIT_SEED, fold, seed]).permutation(METHODS)
    # Warm all arms and establish the control widths without consulting test outcomes.
    for method in METHODS:
        columns = select(method, X, y, task, seed, groups, name == "crimedata", widths)
        check_columns(columns, p)
        widths[method] = len(columns)
        pred = prediction(X, y, Xt, columns, task, seed)
        outputs[f"{method}_support"] = columns
        outputs[f"{method}_pred"] = pred
        if method in ("full", "consensus_policy"):
            for suffix in ("support", "pred"):
                np.testing.assert_array_equal(outputs[f"{method}_{suffix}"],
                                              oracle[f"{method}_s{seed}_{suffix}"])
    scores = {method: metric(gold, outputs[f"{method}_pred"], task) for method in METHODS}
    for repeat in range(repeats):
        for position, method in enumerate(np.roll(methods, repeat)):
            start = perf_counter_ns()
            columns = select(method, X, y, task, seed, groups, name == "crimedata", widths)
            selected = perf_counter_ns()
            pred = prediction(X, y, Xt, columns, task, seed)
            end = perf_counter_ns()
            np.testing.assert_array_equal(columns, outputs[f"{method}_support"])
            np.testing.assert_array_equal(pred, outputs[f"{method}_pred"])
            records.append(dict(dataset=name, fold=fold, seed=seed, method=method, repeat=repeat,
                                position=position, width=len(columns), p=p, score=scores[method],
                                select_ms=(selected-start)/1e6, refit_ms=(end-selected)/1e6,
                                total_ms=(end-start)/1e6))
    return records, outputs


def summarize_timings(output, trials):
    """Write common paired timing metrics for the replay and deployment experiments."""
    trials = pd.DataFrame(trials)
    trials.to_csv(output/"trials.csv", index=False)
    full = trials.loc[trials.method == "full", CASE + ["repeat", "score", "total_ms"]]
    full = full.rename(columns={"score": "full_score", "total_ms": "full_total_ms"})
    paired = trials.merge(full, on=CASE+["repeat"], validate="many_to_one")
    paired["paired_ratio"] = paired.total_ms / paired.full_total_ms
    paired["faster_repeats"] = (paired.paired_ratio < 1).astype(int)
    keys = CASE + ["method", "width", "p"]
    values = ["score", "full_score", "select_ms", "refit_ms", "total_ms", "full_total_ms", "paired_ratio"]
    cases = paired.groupby(keys, as_index=False).agg(dict.fromkeys(values, "median") | {"faster_repeats":"sum"})
    cases["gap"] = cases.score - cases.full_score
    cases["ratio"] = cases.total_ms / cases.full_total_ms
    cases["within"] = cases.gap >= -TOLERANCE
    cases["reduced"] = cases.width < cases.p
    cases["reduced_pass"] = cases.reduced & cases.within
    cases["joint"] = cases.reduced_pass & (cases.ratio < 1)
    cases["joint_5pct"] = cases.reduced_pass & (cases.ratio <= .95)
    cases["joint_10pct"] = cases.reduced_pass & (cases.ratio <= .90)
    cases["paired_joint"] = cases.reduced_pass & (cases.paired_ratio < 1)
    cases.to_csv(output/"cases.csv", index=False)
    stats = dict(cases=("fold","size"), reduced=("reduced","sum"), within=("within","sum"),
                 reduced_pass=("reduced_pass","sum"), joint=("joint","sum"),
                 joint_5pct=("joint_5pct","sum"), joint_10pct=("joint_10pct","sum"),
                 paired_joint=("paired_joint","sum"), median_ratio=("ratio","median"),
                 worst_gap=("gap","min"), mean_gap=("gap","mean"))
    for keys, filename in [(["method"],"summary"), (["dataset","method"],"dataset-summary"),
                           (["seed","method"],"seed-summary")]:
        cases.groupby(keys).agg(**stats).to_csv(output/f"{filename}.csv")
    return cases


def summarize(output, trials):
    cases = summarize_timings(output, trials)
    old = pd.read_csv(CONSENSUS_ARCHIVE/"cases.csv")
    old = old[old.method == "consensus_policy"][CASE + ["width", "score", "joint"]]
    rematch = cases[cases.method == "consensus_policy"].merge(old, on=CASE, suffixes=("", "_historical"),
                                                              validate="one_to_one")
    np.testing.assert_allclose(rematch.score, rematch.score_historical, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(rematch.width, rematch.width_historical)
    rematch.to_csv(output/"consensus-rematch.csv", index=False)
    stability = []
    for name in DATASETS:
        for fold in (1, 2, 3):
            with np.load(output/f"{name}-{fold}.npz") as saved:
                for method in CANDIDATES:
                    for a, b in combinations(SEEDS, 2):
                        ca, cb = (saved[f"{method}_support_s{s}"] for s in (a,b))
                        sa, sb = (cases.loc[(cases.dataset==name) & (cases.fold==fold) &
                                            (cases.method==method) & (cases.seed==s),"score"].item()
                                  for s in (a,b))
                        stability.append(dict(dataset=name, fold=fold, method=method, a=a, b=b,
                                              jaccard=len(np.intersect1d(ca,cb))/len(np.union1d(ca,cb)),
                                              score_change=abs(sa-sb)))
    pd.DataFrame(stability).to_csv(output/"seed-overlap.csv", index=False)


def run(output, verify, repeats):
    current = protocol(repeats)
    if verify:
        saved_protocol = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
        if current["sources"] != saved_protocol["sources"] or current["archives"] != saved_protocol["archives"]:
            raise ValueError("Source or archive changed during fresh-process verification")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"protocol.json", current)
    records, audit = [], []
    for name, fold, X, y, Xt, gold, task, groups in data_folds():
        path = output/f"{name}-{fold}.npz"
        if verify:
            with np.load(path) as saved:
                for seed in SEEDS:
                    for method in CANDIDATES:
                        c = select(method, X, y, task, seed, groups, name=="crimedata", {})
                        pred = prediction(X, y, Xt, c, task, seed)
                        np.testing.assert_array_equal(c, saved[f"{method}_support_s{seed}"])
                        np.testing.assert_array_equal(pred, saved[f"{method}_pred_s{seed}"])
                        audit.append(dict(dataset=name, fold=fold, seed=seed, method=method, width=len(c),
                                          score=metric(gold,pred,task), support_sha256=digest(c),
                                          prediction_sha256=digest(pred)))
        else:
            arrays = {}
            with np.load(CONSENSUS_ARCHIVE/f"{name}-{fold}.npz") as oracle:
                for seed in SEEDS:
                    rows, sample = run_case(name, fold, seed, X, y, Xt, gold, task, groups, oracle, repeats)
                    records.extend(rows)
                    arrays.update({f"{key}_s{seed}": val for key,val in sample.items()})
            np.savez_compressed(path, **arrays)
        print(f"{'Verified' if verify else 'Measured'} {name}, fold {fold}", flush=True)
    if verify:
        pd.DataFrame(audit).to_csv(output/"fresh-process-audit.csv", index=False)
    else:
        if current != protocol(repeats):
            raise ValueError("Sources or archives changed while benchmarking")
        summarize(output, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three timing repeats are required")
    with threadpool_limits(limits=1):
        run(args.output, args.verify, args.repeats)
