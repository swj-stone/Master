"""Measure the low-dimensional minPI on the original eight formal datasets."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, P, ROOT, load_dataset, metrics, train
from research.conditional_dense.original_tasks.paper_splits import check_f, run_pipeline
from research.lowdim.build_native import library_path
from research.lowdim.method import fast_support

HERE = Path(__file__).resolve().parent
ARMS = ("new", "previous", "full", "F", "random")
EXISTING_ARMS = {"previous": "minpi", "full": "full", "F": "f", "random": "random"}
SOURCES = tuple(P["primary_sources"])
KEYS = ["dataset", "fold", "seed"]


def run_arm(arm, X, y, Xt, task, seed, k):
    if arm in EXISTING_ARMS:
        cols, pred, decision, _, select_ms, total_ms = run_pipeline(
            EXISTING_ARMS[arm], X, y, Xt, task, seed, k)
        return cols, pred, decision, select_ms, total_ms
    if arm != "new":
        raise ValueError(arm)
    begin = perf_counter_ns()
    cols, _ = fast_support(X, y, task=task, seed=seed)
    selected = perf_counter_ns()
    A, B = (X, Xt) if len(cols) == X.shape[1] else (X[:, cols], Xt[:, cols])
    pred, decision = train("ridge", A, y, B, task, seed)
    done = perf_counter_ns()
    return cols, pred, decision, (selected-begin)/1e6, (done-begin)/1e6


def inputs(dataset, seed, X, y, order_map, task):
    archive = OUTPUT/"formal"/dataset/f"s{seed}"
    record = json.loads((archive/"record.json").read_text())
    np.testing.assert_array_equal(order_map, record["original_column_indices"])
    tr, te = train_test_split(np.arange(len(y)), test_size=P["test_fraction"],
                              random_state=seed, stratify=y if task=="classification" else None)
    with np.load(archive/"predictions.npz") as old:
        np.testing.assert_array_equal(tr, old["train_indices"])
        np.testing.assert_array_equal(te, old["test_indices"])
        np.testing.assert_array_equal(y[te], old["y_test"])
        old_pred = old["full_ridge_pred"]
        old_decision = old.get("full_ridge_score")
    return (X[tr], y[tr], X[te], y[te], tr, te, old_pred, old_decision)


def hash_source_files():
    files = [Path(__file__), HERE/"formal-protocol.md", HERE/"method.py",
             HERE/"build_native.py", library_path(),
             ROOT/"research/conditional_dense/original_tasks/paper_splits.py",
             ROOT/"research/conditional_dense/original_tasks/regression.py",
             ROOT/"research/conditional_dense/shared_method.py",
             ROOT/"research/conditional_dense/vote_width/method.py",
             ROOT/"research/feature_game_eval/audit_controls.py",
             ROOT/"experiments/minpi_study.py", ROOT/"configs/minpi-study.json"]
    for suffix in ("*.cpp", "*.c", "*.dll", "*.pyd", "*.so"):
        files.extend(HERE.rglob(suffix))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def run(out, previous=None):
    state = dict(sha256=hash_source_files(), arms=ARMS, numpy=np.__version__,
                 sklearn=sklearn.__version__, python=platform.python_version())
    process = 2 if previous else 1
    if previous:
        assert json.loads((previous/"protocol.json").read_text())["sha256"] == state["sha256"]
    out.mkdir(parents=True, exist_ok=False)
    keys = KEYS+["task","metric","p","k","method","repeat","position","select_ms","total_ms"]
    score_rows = []
    source_hashes = {}
    n = 0
    with (out/"times.csv").open("x",newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for dataset in SOURCES:
            raw, target, task, _, order, _, source_hashes[dataset] = load_dataset(dataset)
            metric = "auc" if task == "classification" else "r2"
            task_name = ("binary" if len(np.unique(target)) == 2 else "multiclass"
                         ) if task == "classification" else "regression"
            for fold, seed in enumerate(P["formal_seeds"],1):
                X,y,Xt,truth,tr,te,old_pred,old_decision = inputs(dataset,seed,raw,target,order,task)
                cfg = dict(dataset=dataset,fold=fold,seed=seed,task=task_name,metric=metric,p=X.shape[1])
                width = None
                arrays = dict(train=tr,test=te,truth=truth)
                for arm in ARMS:
                    cols,pred,decision,sel,total = run_arm(arm,X,y,Xt,task,seed,width)
                    if arm == "new":
                        width = len(cols)
                        assert 1 <= width <= X.shape[1]
                    if arm in ("F","random"):
                        assert len(cols) == width
                    if arm == "F":
                        check_f(X, y, task, cols)
                    if arm == "random":
                        np.testing.assert_array_equal(cols,np.sort(np.random.default_rng(seed).permutation(X.shape[1])[:width]))
                    np.testing.assert_array_equal(np.unique(cols),cols)
                    assert cols.min()>=0 and cols.max()<X.shape[1]
                    measurements = metrics(truth,pred,decision)
                    if arm == "full":
                        np.testing.assert_array_equal(cols, np.arange(X.shape[1]))
                        np.testing.assert_allclose(pred,old_pred,atol=1e-6,rtol=1e-6)
                        if old_decision is not None:
                            np.testing.assert_allclose(decision,old_decision,atol=2e-5,rtol=2e-5)
                        archived = metrics(truth,old_pred,old_decision)
                        np.testing.assert_allclose(measurements[metric], archived[metric], atol=1e-12,rtol=0.)
                    if arm == "previous" and X.shape[1]>32:
                        np.testing.assert_array_equal(cols,arrays["new_cols"])
                    arrays[arm+"_cols"],arrays[arm+"_pred"] = cols,pred
                    if decision is not None:
                        arrays[arm+"_decision"] = decision
                    score_rows.append(dict(**cfg,method=arm,k=len(cols),**measurements,
                                           warmup_total_ms=total,warmup_select_ms=sel))
                filename = f"{dataset}-{seed}.npz"
                if previous:
                    with np.load(previous/filename) as first:
                        assert set(arrays)==set(first.files)
                        for field,value in arrays.items():
                            np.testing.assert_array_equal(first[field],value)
                np.savez_compressed(out/filename,**arrays)
                schedule = np.random.default_rng([718441,seed,process]).permutation(ARMS)
                for repeat in range(len(ARMS)):
                    for position,arm in enumerate(np.roll(schedule,repeat)):
                        cols,pred,decision,sel,total = run_arm(arm,X,y,Xt,task,seed,width)
                        writer.writerow(dict(**cfg,k=len(cols),method=arm,repeat=repeat,
                                             position=position,select_ms=sel,total_ms=total))
                        np.testing.assert_array_equal(cols,arrays[arm+"_cols"])
                        np.testing.assert_array_equal(pred,arrays[arm+"_pred"])
                        if decision is not None:
                            np.testing.assert_array_equal(decision,arrays[arm+"_decision"])
                fp.flush()
                n += 1
            print(f"Completed {dataset}, {n}/40",flush=True)
    assert n == 40
    pd.DataFrame(score_rows).to_csv(out/"scores.csv",index=False)
    (out/"protocol.json").write_text(json.dumps(dict(**state,process=process,sources=source_hashes,
                                      repeats=len(ARMS),threads=1,configurations=n),indent=2))


def summarize(first, second, out):
    keys = KEYS+["method"]
    medians = []
    for process, folder in enumerate((first,second),1):
        protocol = json.loads((folder/"protocol.json").read_text())
        assert protocol["sha256"]==hash_source_files() and protocol["process"]==process
        t = pd.read_csv(folder/"times.csv")
        assert len(t)==40*len(ARMS)**2 and not t.duplicated(keys+["repeat"]).any()
        assert (t.total_ms>=t.select_ms).all() and (t.select_ms>=0).all()
        assert t.groupby(keys).position.nunique().eq(len(ARMS)).all()
        assert not t.duplicated(KEYS+["repeat","position"]).any()
        medians.append(t.groupby(keys)[["select_ms","total_ms"]].median().assign(process=process).reset_index())
    frames = [pd.read_csv(folder/"scores.csv").set_index(keys).sort_index() for folder in (first,second)]
    scores = frames[0]
    pd.testing.assert_frame_equal(scores.drop(columns=["warmup_total_ms","warmup_select_ms"]),
                                 frames[1].drop(columns=["warmup_total_ms","warmup_select_ms"]))
    assert len(scores)==200 and not scores.index.duplicated().any()
    medians = pd.concat(medians, ignore_index=True)
    times = medians.set_index(keys+["process"])
    cases = []
    verified_arrays = 0
    for (dataset,fold,seed), group in scores.groupby(level=KEYS):
        rows = group.droplevel(KEYS)
        new = rows.loc["new"]
        metric = new.metric
        case = dict(dataset=dataset,fold=fold,seed=seed,task=new.task,metric=metric,p=int(new.p),k=int(new.k),
                    phenotype="original 8-source formal splits",reduced=new.k<new.p)
        name = f"{dataset}-{seed}.npz"
        with np.load(first/name) as a, np.load(second/name) as b:
            assert set(a.files)==set(b.files)
            for field in a.files:
                np.testing.assert_array_equal(a[field],b[field]); verified_arrays += 1
            for arm in ARMS:
                measures = metrics(a["truth"],a[arm+"_pred"],a.get(arm+"_decision"))
                for label,value in measures.items():
                    np.testing.assert_allclose(rows.at[arm,label],value,atol=1e-12,rtol=0)
                    case[label+"_"+arm] = value
                case["score_"+arm] = measures[metric]
                case["k_"+arm] = len(a[arm+"_cols"])
                for process in (1,2):
                    timing = times.loc[(dataset,fold,seed,arm,process)]
                    case[f"total_ms{process}_{arm}"] = timing.total_ms
                    case[f"select_ms{process}_{arm}"] = timing.select_ms
        for arm in ARMS[1:]:
            gap = case["score_new"] - case["score_"+arm]
            paired = [case[f"total_ms{i}_new"]/case[f"total_ms{i}_{arm}"] for i in (1,2)]
            case["gap_"+arm] = gap
            case["ratio_"+arm] = max(paired)
            case["faster_"+arm] = max(paired)<1
            case["better_"+arm] = gap>1e-12
            case["better_faster_"+arm] = max(paired)<1 and gap>1e-12
        case["cut_faster"] = case["reduced"] and case["faster_full"]
        case["within_002_cut_faster"] = case["cut_faster"] and case["gap_full"]>=-.02-1e-12
        cases.append(case)
    out.mkdir(parents=True,exist_ok=False)
    medians.to_csv(out/"process-medians.csv",index=False)
    frame = pd.DataFrame(cases)
    frame.to_csv(out/"cases.csv",index=False)
    aggregates = dict(n=("k","size"),p=("p","first"),k_min=("k","min"),k_max=("k","max"),
                      reduced=("reduced","sum"),cut_faster=("cut_faster","sum"),
                      within_002_cut_faster=("within_002_cut_faster","sum"))
    for arm in ARMS:
        aggregates["score_"+arm] = ("score_"+arm,"median")
        for process in (1,2):
            for prefix in ("select_ms","total_ms"):
                key = f"{prefix}{process}_{arm}"
                aggregates[key] = (key,"median")
        if arm != "new":
            for key in ("gap","ratio"):
                col = key+"_"+arm
                aggregates[col] = (col,"median")
            for key in ("faster","better","better_faster"):
                col = key+"_"+arm
                aggregates[col] = (col,"sum")
    summary = frame.groupby(["dataset","task","metric"],as_index=False).agg(**aggregates)
    summary.to_csv(out/"sources.csv",index=False)
    facts = dict(datasets=frame.dataset.nunique(),cases=len(frame),score_vectors=len(scores),
                 timed_pipelines=2*len(frame)*len(ARMS)**2,cross_process_arrays=verified_arrays)
    (out/"verification.json").write_text(json.dumps(facts,indent=2))
    print(facts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action",required=True)
    measure = commands.add_parser("run")
    measure.add_argument("out",type=Path)
    measure.add_argument("--previous",type=Path)
    report = commands.add_parser("summarize")
    for arg in ("first","second","out"):
        report.add_argument(arg,type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.action == "run":
            run(args.out,args.previous)
        else:
            summarize(args.first,args.second,args.out)
