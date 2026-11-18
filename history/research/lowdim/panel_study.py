"""Run the original 69 and additional 57 registered feature-selection configurations."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.conditional_dense.original_tasks.paper_splits import f_support
from research.conditional_dense.original_tasks.regression import regression_support
from research.conditional_dense.vote_width.method import vote_support
from research.feature_game_eval import panels as bench
from research.feature_game_eval.audit_controls import effect_size
from research.lowdim.build_native import library_path
from research.lowdim.method import fast_support

HERE = Path(__file__).resolve().parent
METHODS = ("new", "previous", "full", "F", "random")
KEY = ["cohort", "dataset", "fold", "seed"]


def signature():
    files = [Path(__file__), HERE/"core-protocol.md", HERE/"method.py", HERE/"build_native.py", library_path(),
             Path(bench.__file__), Path(bench.original.__file__), Path(bench.extra.__file__),
             bench.ROOT/"research/conditional_dense/original_tasks/regression.py",
             bench.ROOT/"research/conditional_dense/original_tasks/paper_splits.py",
             bench.ROOT/"research/conditional_dense/shared_method.py",
             bench.ROOT/"research/conditional_dense/vote_width/method.py"]
    for suffix in ("*.c","*.cpp","*.dll","*.so","*.pyd"):
        files.extend(HERE.rglob(suffix))
    return {p.relative_to(bench.ROOT).as_posix():bench.checksum(p) for p in files}


def select(panel, method, width):
    if method in ("full", "random"):
        return bench.select(method, panel, width)
    X = panel.X
    if panel.name in bench.deployment.ADDITIONAL:
        X = bench._impute_validation(X, X[:0])[0]
    if method == "F":
        return f_support(X, panel.y, panel.task, width)
    if method == "new":
        return fast_support(X, panel.y, task=panel.task, seed=panel.seed)[0]
    if method == "previous":
        selector = regression_support if panel.task == "regression" else vote_support
        return selector(X, panel.y, seed=panel.seed)[0]
    raise ValueError(method)


def deploy(panel, method, width):
    start = perf_counter_ns()
    cols = select(panel, method, width)
    selected = perf_counter_ns()
    pred, decision = bench.fit(panel, cols)
    end = perf_counter_ns()
    return cols, pred, decision, (selected-start)/1e6, (end-start)/1e6


def f_boundary(panel, selected):
    X = panel.X
    if panel.name in bench.deployment.ADDITIONAL:
        X = bench._impute_validation(X, X[:0])[0]
    effects = np.concatenate([effect_size(X[:,i:i+256],panel.y,panel.task)
                              for i in range(0,X.shape[1],256)])
    outside = np.setdiff1d(np.arange(X.shape[1]),selected)
    gap = 0. if not len(outside) else max(0.,effects[outside].max()-effects[selected].min())
    exact = np.array_equal(selected,np.sort(np.argsort(-effects,kind="stable")[:len(selected)]))
    if gap > 1e-9:
        raise ValueError(f"{panel.name}: F boundary inconsistent ({gap})")
    return dict(independent_F_gap=gap, exact_F_set=exact)


def source_kind(name, family):
    if "simulation" in family:
        return "generated"
    return "artificial_challenge" if name == "madelon" else "observational"


def run(out, previous):
    archive = pd.read_csv(bench.ARCHIVE/"residual-126-corrected/quality/arrays.csv").set_index("panel")
    pass_id = 2 if previous else 1
    state = dict(process=pass_id, threads=1, arms=METHODS, sklearn=bench.sklearn.__version__,
                 numpy=np.__version__,sha256=signature(),base=bench.protocol())
    if previous:
        earlier = json.loads((previous/"complete.json").read_text(encoding="utf-8"))
        if earlier["sha256"] != state["sha256"] or earlier["base"] != state["base"]:
            raise ValueError("Source or dependency fingerprint changed")
    out.mkdir(parents=True,exist_ok=False)
    checks, inputs, unique = [], [], set()
    cols_csv = KEY+["task","metric","family","source_kind","p","n_train","n_test","classes","method",
                   "k","repeat","position","score","ap","select_ms","total_ms"]
    with (out/"times.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols_csv)
        writer.writeheader()
        for cohort,panel,truth,units,family in bench.panels():
            key = (cohort,panel.name,panel.fold,panel.seed)
            case = "-".join(map(str,key))
            if case in unique or case not in archive.index:
                raise ValueError(case)
            unique.add(case)
            n,p = panel.X.shape
            cfg = dict(zip(KEY,key))
            groups = getattr(panel,"groups",None)
            hashes = [bench.digest(x) for x in (panel.X,panel.y,panel.Xt,truth)]
            old_path = bench.ROOT/archive.loc[case,"source"]
            with np.load(old_path) as old:
                np.testing.assert_array_equal(old["input_fingerprints"],hashes)
                expected_full = old["full_pred"].copy()
            arrays = dict(truth=truth, input_fingerprints=np.array(hashes))
            if units is not None:
                arrays["units"] = np.asarray(units)
            inputs.append(dict(**cfg,train_sha=hashes[0],target_sha=hashes[1],
                               test_sha=hashes[2],truth_sha=hashes[3],
                               groups_sha=bench.digest(groups) if groups is not None else "none",
                               units_sha=bench.digest(units) if units is not None else "none"))
            width, scores = None, {}
            for arm in METHODS:
                cols, pred, decision, _, _ = deploy(panel,arm,width)
                if arm=="new":
                    width = len(cols)
                if arm in ("F","random"):
                    assert len(cols)==width
                if arm=="F":
                    checks.append(dict(**cfg,**f_boundary(panel,cols)))
                if arm=="full":
                    np.testing.assert_array_equal(pred,expected_full)
                assert len(cols)>0 and np.array_equal(cols,np.unique(cols)) and 0<=cols.min()<=cols.max()<p
                scores[arm] = bench.deployment.score_predictions(panel.name,truth,pred,units,panel.task,decision)[::2]
                arrays[arm+"_cols"], arrays[arm+"_pred"] = cols,pred
                if panel.name=="secom":
                    arrays[arm+"_decision"] = decision
            filename = case+".npz"
            if previous:
                with np.load(previous/filename) as reference:
                    assert set(arrays)==set(reference.files)
                    for name,v in arrays.items():
                        np.testing.assert_array_equal(v,reference[name])
            np.savez_compressed(out/filename,**arrays)
            order = np.random.default_rng([88351,panel.fold,panel.seed,pass_id]).permutation(METHODS)
            classes = len(np.unique(panel.y)) if panel.task=="classification" else 0
            metric = {"mhealth":"mean_person_worst_device_BA","gas":"mean_batch_BA",
                      "har":"mean_person_BA"}.get(panel.name,"R2" if panel.task=="regression" else "BA")
            for repeat in range(len(METHODS)):
                for position, arm in enumerate(np.roll(order,repeat)):
                    cols,pred,decision,sel,total = deploy(panel,arm,width)
                    np.testing.assert_array_equal(cols,arrays[arm+"_cols"])
                    np.testing.assert_array_equal(pred,arrays[arm+"_pred"])
                    if panel.name=="secom":
                        np.testing.assert_array_equal(decision,arrays[arm+"_decision"])
                    score,ap = scores[arm]
                    writer.writerow(dict(**cfg,task=panel.task,metric=metric,family=family,
                                         source_kind=source_kind(panel.name,family),
                                         p=p,n_train=n,n_test=len(truth),classes=classes,
                                         method=arm,k=len(cols),repeat=repeat,position=position,
                                         score=score,ap=ap,select_ms=sel,total_ms=total))
            stream.flush()
            if len(unique)%10==0 or len(unique)==126:
                print(f"{len(unique)}/126 {case}",flush=True)
    assert unique==set(archive.index) and len(unique)==126 and signature()==state["sha256"]
    pd.DataFrame(inputs).to_csv(out/"inputs.csv",index=False)
    pd.DataFrame(checks).to_csv(out/"f-audit.csv",index=False)
    (out/"complete.json").write_text(json.dumps(state,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--reference",type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.output,args.reference)
