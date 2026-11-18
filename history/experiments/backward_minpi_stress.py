"""Replay the previously inspected five-source panel for the deletion-score minPI game."""
import argparse
import json
from math import ceil, floor
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.stable_minpi_stress import (
    DATASETS, FRACTIONS, PRIMARY_SEED, OTHER_SEEDS, ROOT, SPLIT_SEED, TOLERANCE,
    audit_inner, choose, choose_columns, digest, file_hash, folds, inputs, manifest,
    metric, prediction, write_json,
)
from minppi_mofs.backward_minpi import backward_path_from_training, backward_top_from_training
from minppi_mofs.environment_minpi import InsufficientEnvironmentsError
from minppi_mofs.stable_minpi import _partition, _calibrate_supports


ARCHIVE = ROOT / "results/minpi-study/stable-stress-20261117b"
SEEDS = (PRIMARY_SEED, *OTHER_SEEDS)
METHODS = ("backward_half", "backward_policy", "environment_half", "stable", "local_half", "F_half", "full")


def backward_policy(X, y, task, seed, groups):
    p = X.shape[1]
    info = dict(discovery_rows=[], validation_rows=[], rule="full")
    try:
        discovery, valid = _partition(y, groups, False, task, seed)
        info.update(discovery_rows=discovery.tolist(), validation_rows=[v.tolist() for v in valid])
        widths = sorted({ceil(p * f) for f in FRACTIONS} - {p})
        supports, inner = backward_path_from_training(
            X[discovery], y[discovery], widths, task=task, random_state=seed,
            groups=None if groups is None else groups[discovery], ordered_groups=False)
        columns, calibrated = _calibrate_supports(
            X, y, discovery, valid, supports, task, seed, TOLERANCE, rule="backward")
        return columns, {**info, **calibrated, "environment_path": inner}
    except InsufficientEnvironmentsError as exc:
        return np.arange(p), {**info, "reason": "insufficient_environments", "detail": str(exc)}


def select(method, X, y, task, seed, groups):
    width = ceil(X.shape[1] / 2)
    if method == "backward_half":
        return backward_top_from_training(X, y, width, task=task, random_state=seed, groups=groups)
    if method == "backward_policy":
        return backward_policy(X, y, task, seed, groups)
    if method == "F_half":
        return choose("f", X, y, task, seed, width), {}
    return choose_columns(method, X, y, task, seed, groups, width)


def references(archive, method, seed):
    if method in ("stable", "environment_half"):
        key = method if seed == PRIMARY_SEED else f"{method}_s{seed}"
    elif method == "full" or (method == "local_half" and seed == PRIMARY_SEED):
        key = method
    else:
        return None
    return archive[key+"_support"], archive[key+"_pred"]


def bench(output, name, fold, X, y, Xt, gold, task, groups, oracle):
    records, arrays, decisions = [], {}, {}
    for seed in SEEDS:
        rotation = np.random.default_rng([SPLIT_SEED,seed,fold]).permutation(METHODS)
        for rep in range(3 if seed == PRIMARY_SEED else 1):
            for position, method in enumerate(np.roll(rotation, 3 * rep)):
                start = perf_counter_ns()
                cols, info = select(method, X, y, task, seed, groups)
                selected = perf_counter_ns()
                pred = prediction(X, y, Xt, cols, task, seed)
                end = perf_counter_ns()
                assert np.array_equal(np.unique(cols), cols) and len(cols) > 0
                if method.endswith("_half"):
                    assert len(cols) == ceil(X.shape[1] / 2)
                key = f"{method}_s{seed}"
                for suffix, value in (("support",cols),("pred",pred)):
                    label = key + "_" + suffix
                    if rep == 0:
                        arrays[label] = value
                    else:
                        np.testing.assert_array_equal(arrays[label], value)
                previous = references(oracle, method, seed)
                if previous is not None:
                    np.testing.assert_array_equal(previous[0], cols)
                    np.testing.assert_array_equal(previous[1], pred)
                if method == "backward_policy":
                    audit_inner(info, groups)
                    if rep == 0:
                        decisions[key] = info
                records.append(dict(dataset=name, fold=fold, seed=seed, method=method, repeat=rep, position=position,
                                    score=metric(gold,pred,task), p=X.shape[1], width=len(cols),
                                    rule=info.get("rule",method), select_ms=(selected-start)/1e6,
                                    refit_ms=(end-selected)/1e6, total_ms=(end-start)/1e6))
    np.savez_compressed(output/f"{name}-{fold}.npz", **arrays)
    write_json(output/f"{name}-{fold}-decisions.json", decisions)
    return records


def summarize(output, records):
    trials = pd.DataFrame(records)
    trials.to_csv(output/"trials.csv",index=False)
    dims = ["dataset","fold","seed","method","width","p","rule"]
    cases = trials.groupby(dims, as_index=False)[["score","select_ms","refit_ms","total_ms"]].median()
    baseline = cases[cases.method=="full"][["dataset","fold","seed","score","refit_ms","total_ms"]]
    baseline = baseline.rename(columns={"score":"full_score","refit_ms":"full_refit_ms","total_ms":"full_total_ms"})
    cases = cases.merge(baseline,on=["dataset","fold","seed"],validate="many_to_one")
    cases["gap"] = cases.score - cases.full_score
    cases["within"] = cases.gap >= -TOLERANCE
    cases["reduced"] = cases.width < cases.p
    cases["reduced_pass"] = cases.reduced & cases.within
    cases["joint"] = cases.reduced_pass & (cases.total_ms < cases.full_total_ms)
    cases["break_even"] = [floor(r.select_ms/(r.full_refit_ms-r.refit_ms))+1
                           if r.reduced and r.refit_ms < r.full_refit_ms else np.nan for r in cases.itertuples()]
    cases.to_csv(output/"cases.csv",index=False)
    cases.groupby(["dataset","method"]).agg(
        cases=("fold","size"), mean_score=("score","mean"), mean_gap=("gap","mean"), worst_gap=("gap","min"),
        passing=("within","sum"), compressed=("reduced","sum"), compressed_passing=("reduced_pass","sum")
    ).to_csv(output/"quality.csv")
    primary = cases[cases.seed==PRIMARY_SEED]
    primary.groupby(["dataset","method"]).agg(mean_score=("score","mean"), total_ms=("total_ms","median"),
        select_ms=("select_ms","median"), reused_ridge_ms=("refit_ms","median"), joint=("joint","sum"),
        break_even_cases=("break_even","count"), median_finite_break_even=("break_even","median")
    ).to_csv(output/"primary-timing.csv")


def run(output, verify):
    archive_manifest = json.loads((ARCHIVE/"protocol.json").read_text(encoding="utf-8"))
    hashes = manifest()["sources"]
    for path in (Path(__file__), ROOT/"src/minppi_mofs/backward_minpi.py"):
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
    if verify:
        assert hashes == json.loads((output/"protocol.json").read_text(encoding="utf-8"))["sources"]
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"protocol.json",dict(sources=hashes, oracle_protocol=file_hash(ARCHIVE/"protocol.json"),
                   datasets=DATASETS, seeds=SEEDS, methods=METHODS, repeats_primary=3, repeats_secondary=1,
                   threads=1, tolerance=TOLERANCE, widths=FRACTIONS,
                   role="Post-design replay; performance of forward policies on all five sources was available before the deletion game.",
                   split="Original fold and inputs matched byte-for-byte. Discovery, validation and calibration score/split are shared by the two adaptive paths.",
                   timing="Three within-seed medians for seed 1129. Single-run sensitivity seeds used for quality only. End-to-end includes width calibration, support selection, extraction, ridge fitting and prediction."))
    records, audit = [], []
    for name in DATASETS:
        X, y, task, groups, source, duplicates = inputs(name)
        assert archive_manifest["input_hashes"][name] == dict(source=source,X=digest(X),y=digest(y),
                            groups=None if groups is None else digest(groups))
        for fold,(tr,te) in enumerate(folds(X,y,task,groups),1):
            with np.load(ARCHIVE/f"{name}-{fold}-split.npz") as split:
                np.testing.assert_array_equal(split["train"],tr)
                np.testing.assert_array_equal(split["test"],te)
                np.testing.assert_array_equal(split["gold"],y[te])
            assert not np.intersect1d(duplicates[tr],duplicates[te]).size
            g = None if groups is None else groups[tr]
            if verify:
                with np.load(output/f"{name}-{fold}.npz") as saved:
                    for seed in SEEDS:
                        for method in ("backward_half","backward_policy"):
                            c, _ = select(method,X[tr],y[tr],task,seed,g)
                            pred = prediction(X[tr],y[tr],X[te],c,task,seed)
                            np.testing.assert_array_equal(saved[f"{method}_s{seed}_support"],c)
                            np.testing.assert_array_equal(saved[f"{method}_s{seed}_pred"],pred)
                            audit.append(dict(dataset=name,fold=fold,seed=seed,method=method,width=len(c),
                                              score=metric(y[te],pred,task),support_sha256=digest(c),
                                              prediction_sha256=digest(pred)))
            else:
                with np.load(ARCHIVE/f"{name}-{fold}-outputs.npz") as oracle:
                    records.extend(bench(output,name,fold,X[tr],y[tr],X[te],y[te],task,g,oracle))
            print(f"{'Checked' if verify else 'Finished'} {name} {fold}",flush=True)
    if verify:
        pd.DataFrame(audit).to_csv(output/"fresh-process-audit.csv",index=False)
    else:
        summarize(output,records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.verify)
