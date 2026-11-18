"""Paired, repeated deployment from row sampling to prediction on frozen positive cases."""
import argparse
import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import sklearn
from sklearn.feature_selection import f_classif
from threadpoolctl import threadpool_limits

from experiments.grouped_solver_cost_audit import fit
from experiments.minpi_backward_confirmation import evaluate
from research.conditional_dense.confirmation import ROOT, zero_vote
from research.conditional_dense.shared_method import conditional_support
from research.conditional_dense.transfer.study import panels
from research.feature_game_eval.audit_controls import effect_size


METHODS = ("conditional", "sketch_F", "zero", "full")
REPEATS = len(METHODS)
HERE = Path(__file__).resolve().parent


def sample_rows(y, seed):
    limit = min(768, len(y))
    if len(y) <= limit:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, labels = np.unique(y, return_inverse=True)
    first = [rng.choice(np.flatnonzero(labels == j)) for j in range(len(classes))]
    rest = np.setdiff1d(np.arange(len(y)), first, assume_unique=True)
    return np.sort(np.r_[first, rng.choice(rest, limit-len(first), replace=False)])


def sketch_f(X, y, width, seed):
    rows = sample_rows(y, seed)
    centered = np.asarray(X[rows], dtype=np.float64)
    centered -= centered.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        scores, _ = f_classif(centered, y[rows])
    scores = np.nan_to_num(scores, nan=-np.inf)
    return np.sort(np.argsort(-scores, kind="stable")[:width])


def check_sampler(panel, width):
    """Observe actual sampler indices at Python function return, outside timing."""
    codes = {f.__code__: f.__name__ for f in (sketch_f, conditional_support, zero_vote)}
    captured = {}
    def observer(frame, event, arg):
        if event == "return" and frame.f_code in codes:
            captured[codes[frame.f_code]] = frame.f_locals["rows"].copy()
    old = sys.getprofile()
    sys.setprofile(observer)
    try:
        conditional_support(panel.X, panel.y, seed=panel.seed)
        zero_vote(panel.X, panel.y, width, panel.seed)
        chosen = sketch_f(panel.X, panel.y, width, panel.seed)
    finally:
        sys.setprofile(old)
    assert len(captured) == 3
    rows = captured["sketch_f"]
    for sample in captured.values():
        np.testing.assert_array_equal(sample, rows)
    assert len(rows) == len(np.unique(rows)) == min(768, len(panel.y))
    effects = effect_size(panel.X[rows], panel.y[rows], "classification")
    order = np.argsort(-effects, kind="stable")
    outside = np.setdiff1d(order, chosen)
    gap = float(max(0., effects[outside].max() - effects[chosen].min())) if len(outside) else 0.
    assert gap <= 1e-10, (panel.name, gap)
    return rows, gap, int(len(np.setdiff1d(order[:width], chosen)))


def deploy(panel, arm, width):
    start = perf_counter_ns()
    if arm == "sketch_F":
        cols = sketch_f(panel.X, panel.y, width, panel.seed)
    elif arm == "conditional":
        cols = conditional_support(panel.X, panel.y, seed=panel.seed)[0]
    elif arm == "zero":
        cols = zero_vote(panel.X, panel.y, width, panel.seed)
    elif arm == "full":
        cols = np.arange(panel.X.shape[1])
    else:
        raise ValueError(arm)
    selected = perf_counter_ns()
    pred, decision, timing = fit(panel, cols, "cholesky")
    end = perf_counter_ns()
    return cols, pred, decision, dict(select_ms=(selected-start)/1e6, total_ms=(end-start)/1e6, **timing)


def run(out, pass_id):
    out.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), HERE/"protocol.md", HERE.parent/"shared_method.py",
        HERE.parent/"confirmation.py", HERE.parent/"transfer/study.py", HERE.parent/"speech.py",
        ROOT/"experiments/grouped_solver_cost_audit.py", ROOT/"research/feature_game_eval/audit_controls.py"]
    protocol = dict(process=pass_id, threads=1, repeats=REPEATS, numpy=np.__version__,
        sklearn=sklearn.__version__, methods=METHODS, panels=[],
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    with (out/"trials.csv").open("x", newline="") as fp:
        writer = csv.DictWriter(fp,fieldnames=["panel","source","seed","fold","p","k","repeat","position",
            "method","score","select_ms","slice_ms","fit_ms","predict_ms","total_ms"])
        writer.writeheader()
        for panel in panels():
            key = f"{panel.name}-{panel.fold}-{panel.seed}"
            old = "conditional-speech-quality-20261118a" if panel.name=="spoken_arabic_digit" else "conditional-dense-quality-20261118a"
            with np.load(ROOT/"results/minpi-study"/old/(key+".npz")) as ref:
                width = len(ref["conditional_cols"])
                np.testing.assert_array_equal(ref["truth"], panel.gold)
                cache = {"truth":panel.gold}
                if panel.units is not None:
                    cache["units"] = panel.units
                rows, gap, mismatches = check_sampler(panel, width)
                cache["rows"] = rows
                for arm in METHODS:
                    cols,pred,decision,_ = deploy(panel,arm,width)
                    for suffix, arr in (("cols",cols),("pred",pred),("decision",decision)):
                        cache[arm+"_"+suffix] = arr
                        if arm != "sketch_F":
                            np.testing.assert_array_equal(arr,ref[arm+"_"+suffix])
            protocol["panels"].append(dict(panel=key,width=width,p=panel.X.shape[1],rows=len(rows),
                max_F_boundary_violation=gap,n_equal_effect_substitutions=mismatches))
            np.savez_compressed(out/(key+".npz"), **cache)
            order = np.random.default_rng([panel.seed,panel.fold,pass_id,61339]).permutation(METHODS)
            for repeat in range(REPEATS):
                for position,arm in enumerate(np.roll(order,repeat)):
                    cols,pred,decision,timing = deploy(panel,arm,width)
                    for suffix, arr in (("cols",cols),("pred",pred),("decision",decision)):
                        np.testing.assert_array_equal(arr,cache[arm+"_"+suffix])
                    score, _ = evaluate(panel.name,panel.gold,pred,panel.units,"classification")
                    writer.writerow(dict(panel=key,source=panel.name,seed=panel.seed,fold=panel.fold,
                        p=panel.X.shape[1],k=len(cols),repeat=repeat,position=position,method=arm,score=score,**timing))
            fp.flush()
            print(key,flush=True)
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--pass-id",required=True,type=int,choices=(1,2))
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.out,args.pass_id)
