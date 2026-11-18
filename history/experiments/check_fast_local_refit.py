"""Verify sufficient-statistic local refits on every archived split and time two fixed development budgets."""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.compare_local_refit import PARENT, local_refit_top_from_training
from experiments.minpi_iteration_benchmark import (
    BUDGETS, PHASES, ROOT, SEEDS, SPLIT_SEED, file_hash, phase_panels, quality, select, train,
)
from minppi_mofs.local_refit_fast import local_refit_fast_top_from_training


REFERENCE = ROOT / "results/minpi-study/local-refit-iteration-20261117a"
TIMED = ("fast", "local_refit", "full", "F", "PFI")


def measure(output):
    output.mkdir(parents=True, exist_ok=False)
    manifests = [REFERENCE / "protocol.json", PARENT / "protocol.json"]
    paths = [Path(__file__), ROOT / "src/minppi_mofs/local_refit_fast.py",
             ROOT / "src/minppi_mofs/local_refit_minpi.py", ROOT / "src/minppi_mofs/refit_minpi.py",
             ROOT / "src/minppi_mofs/masked_loss_minpi.py", ROOT / "src/minppi_mofs/piv2.py",
             ROOT / "experiments/compare_local_refit.py", ROOT / "experiments/minpi_iteration_benchmark.py"]
    checks = {str(path.relative_to(ROOT)):file_hash(path) for path in paths + manifests}
    protocol = dict(archives=[str(PARENT),str(REFERENCE)], checksums=checks,
                    verification="Exact support and same downstream predictions on all 90 historical cases",
                    timing="Development Gas 50%, materials 10%; three rotated single-thread, fresh raw-input calls per seed",
                    aggregation="Median times over repeats, then seeds, then folds")
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows, timings = [], []
    for phase in PHASES:
        for panel in phase_panels(phase):
            prefix = f"{phase}-{panel.source}-{panel.fold}"
            X,y,Xt = panel.X[panel.tr],panel.y[panel.tr],panel.X[panel.te]
            groups = panel.groups[panel.tr]
            with np.load(PARENT/f"{prefix}-split.npz") as old_split:
                np.testing.assert_array_equal(panel.tr,old_split["train"])
                np.testing.assert_array_equal(panel.te,old_split["test"])
            for seed in SEEDS:
                rng = np.random.default_rng([seed, SPLIT_SEED, panel.fold, 8851])
                with np.load(REFERENCE/f"{prefix}-s{seed}.npz") as oracle, np.load(PARENT/f"{prefix}-s{seed}.npz") as controls:
                    for budget in BUDGETS:
                        count = ceil(X.shape[1]*budget/100)
                        timed = phase=="development" and (panel.source,budget) in (("gas",50),("supercon",10))
                        order = rng.permutation(TIMED) if timed else ["fast"]
                        key = f"b{budget}_local_refit"
                        refcols, refpred = oracle[key+"_columns"], oracle[key+"_pred"]
                        for rep in range(3 if timed else 1):
                            for method in np.roll(order, rep):
                                start = perf_counter_ns()
                                if method in ("fast","local_refit"):
                                    selector = local_refit_fast_top_from_training if method=="fast" else local_refit_top_from_training
                                    columns, _ = selector(X,y,count,task=panel.task,random_state=seed)
                                else:
                                    columns = select(method,X,y,groups,panel,seed,count)
                                ranked = perf_counter_ns()
                                a,b = (X,Xt) if method=="full" else (X[:,columns],Xt[:,columns])
                                pred,_ = train("ridge",a,y,b,panel.task,seed)
                                end = perf_counter_ns()
                                if method == "fast" and rep == 0:
                                    rows.append(dict(phase=phase,source=panel.source,fold=panel.fold,seed=seed,budget=budget,
                                        same_support=np.array_equal(columns,refcols),same_prediction=np.array_equal(pred,refpred),
                                        score=quality(panel,pred),reference_score=quality(panel,refpred),
                                        columns=" ".join(map(str,columns))))
                                elif method != "fast":
                                    oldcols = refcols if method=="local_refit" else controls[f"b{budget}_{method}_columns"]
                                    oldpred = refpred if method=="local_refit" else controls[f"b{budget}_{method}_pred"]
                                    np.testing.assert_array_equal(columns,oldcols)
                                    np.testing.assert_array_equal(pred,oldpred)
                                if timed:
                                    timings.append(dict(source=panel.source,fold=panel.fold,seed=seed,budget=budget,repeat=rep,
                                        method=method,select_ms=(ranked-start)/1e6,total_ms=(end-start)/1e6))
            print(prefix,flush=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(output/"audit.csv",index=False)
    calls = pd.DataFrame(timings)
    calls.to_csv(output/"trials.csv",index=False)
    keys, ms = ["source","budget","method"], ["select_ms","total_ms"]
    cases = calls.groupby(keys+["fold","seed"])[ms].median().reset_index()
    folds = cases.groupby(keys+["fold"])[ms].median().reset_index()
    folds.groupby(keys)[ms].median().reset_index().to_csv(output/"summary.csv",index=False)
    if checks != {str(path.relative_to(ROOT)):file_hash(path) for path in paths + manifests}:
        raise RuntimeError("Source changed during the run")
    if not (audit.same_support.all() and audit.same_prediction.all()):
        raise ValueError("Coordinate or prediction difference; inspect the verified case list")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    with threadpool_limits(limits=1):
        measure(parser.parse_args().output)
