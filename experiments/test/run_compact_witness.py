"""Exploratory seed-42 witnessed pruning from the compact Gain anchor."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from datetime import datetime

from experiments.test import run_compact_minpi as parent
from minppi_mofs.experiment_protocol import atomic_write_json,canonical_hash

CONTROLS=("tree_mean_pfi","compact_gain_pfi")
METHODS=("witness_min","witness_mean","witness_small","witness_min_pfi",*CONTROLS)
DEFAULT=parent.ROOT/"results/test/thesis_sampling/PIv3-compact-witness-raw-six-seed42"
EXTRA_SOURCES=("src/minppi_mofs/piv3_compact_witness.py","experiments/test/run_compact_witness.py")
read=parent.read


def guard_proposals(cols, candidate, support, anchor, order, downstream, control=False):
    options=[("pfi",candidate)] if downstream else []
    enlarged=set(order[:2*len(anchor)])
    if control:
        rescue=support if candidate!=support else [c for c in cols if c in enlarged]
        if rescue!=candidate and rescue!=cols:
            options.append(("prefilter" if candidate!=support else "double_prefilter",rescue))
    else:
        options.extend([("witness",support),("gain_anchor",anchor),
                        ("double_gain_anchor",[c for c in cols if c in enlarged])])
    seen=set()
    for name, coords in options:
        key=tuple(coords)
        if key not in seen:
            seen.add(key)
            yield name,coords


def trial(dataset,method,out):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from experiments.test.piv3_raw_splits import load_split
    from experiments.test.piv3_lite_arm import _matrix,_metrics,_predict,_pfi,_positive,_failures
    from minppi_mofs.modeling import make_model
    from minppi_mofs import piv3_compact_witness as witness, piv3_compact_tree as tree

    clock=time.perf_counter
    started=clock()
    fits=0
    def progress(phase,**details):
        atomic_write_json(out/"progress.json",dict(phase=phase,fits=fits,elapsed_s=clock()-started,**details))

    progress("loading")
    meta=read(parent.DEFAULT_ROOT/dataset/"metadata.json")
    policy={k:v for k,v in meta["policy"].items() if k!="utility"}
    task,utility=policy["task"],meta["utility"]
    regression, fraud=task=="regression",dataset=="credit_card_fraud"
    train,y=load_split(parent.DEFAULT_ROOT,dataset,"train")
    val,yv=load_split(parent.DEFAULT_ROOT,dataset,"val")
    cols=list(train.columns)
    if list(val.columns)!=cols or cols!=meta["features"]:
        raise ValueError("Unexpected original column order")
    ranking_idx,guard_idx=train_test_split(np.arange(len(val)),test_size=.5,random_state=42,
                                           stratify=None if regression else yv)
    ranking_idx,guard_idx=np.sort(ranking_idx),np.sort(guard_idx)
    ranking,yr=val.iloc[ranking_idx],yv[ranking_idx]
    guard,yg=val.iloc[guard_idx],yv[guard_idx]
    def subset(frame,names):
        return frame if names==cols else frame.loc[:,names]

    models={}
    def train_model(names):
        nonlocal fits
        key=tuple(names)
        if key not in models:
            est=make_model(**policy)
            est.fit(_matrix(subset(train,names)),y)
            fits+=1
            models[key]=est
        return models[key]

    def pred(est,frame,names):
        return _predict(est,_matrix(subset(frame,names)),regression)

    def score(labels,pp):
        return _metrics(labels,pp,utility,fraud)

    progress("reference")
    begin=clock()
    full=train_model(cols)
    full_pred=pred(full,guard,cols)
    reference=score(yg,full_pred)
    ref_s=clock()-begin

    aggregate=method.split("_")[1]
    progress("witness_selection")
    begin=clock()
    selector=tree.select if method in CONTROLS else witness.select
    support,diag=selector(full,train,y,ranking,yr,task=task,utility=utility,seed=42,aggregate=aggregate)
    if method in CONTROLS:
        anchor,order=support,diag["order"]
    else:
        anchor,order=diag["anchor"],diag["anchor_order"]
    lookup=set(cols)
    if (not anchor or not support or len(set(order))!=len(order) or
            not set(support)<=set(anchor)<=set(order)<=lookup or set(anchor)!=set(order[:len(anchor)])):
        raise ValueError("Invalid witness support or original anchor ranking")
    support_set,anchor_set=set(support),set(anchor)
    if (support!=[c for c in cols if c in support_set] or anchor!=[c for c in cols if c in anchor_set]):
        raise ValueError("Expected original feature order")
    front_s=clock()-begin

    downstream=method.endswith("_pfi")
    rank_s=0.
    candidate=support
    if downstream:
        progress("downstream")
        begin=clock()
        estimator=train_model(support)
        importance,details=_pfi(estimator,subset(ranking,support),yr,utility,
                                train.columns.get_indexer(support).tolist(),progress)
        candidate=_positive(support,importance)
        atomic_write_json(out/"ranking.json",details)
        rank_s=clock()-begin
    rank_fits=fits-1
    progress("quality_check")
    begin=clock()
    before=fits
    checks=[]
    guard_predictions=[full_pred]
    chosen=cols
    accepted="full_fallback"
    for route,names in guard_proposals(cols,candidate,support,anchor,order,downstream,method in CONTROLS):
        pp=full_pred if names==cols else pred(train_model(names),guard,names)
        quality=score(yg,pp)
        failed=_failures(reference,quality)
        checks.append(dict(route=route,names=names,k=len(names),metrics=quality,failed=failed))
        guard_predictions.append(pp)
        if not failed:
            chosen,accepted=names,route
            break
    guard_s=clock()-begin
    guard_fits=fits-before
    accepted_scores=reference if accepted=="full_fallback" else checks[-1]["metrics"]
    frozen=dict(dataset=dataset,method=method,seed=42,policy=policy,front=support,initial=candidate,
                checks=checks,selected=chosen,accepted_route=accepted,anchor=anchor,
                ranking_rows_hash=canonical_hash(ranking_idx.tolist()),guard_rows_hash=canonical_hash(guard_idx.tolist()))
    fingerprint=canonical_hash(frozen)
    atomic_write_json(out/"frozen.json",frozen)
    atomic_write_json(out/"front.json",diag)

    progress("final_refit")
    begin=clock()
    final=make_model(**policy)
    final.fit(np.concatenate([_matrix(subset(f,chosen)) for f in (train,val)]),np.concatenate([y,yv]))
    fits+=1
    atomic_write_json(out/"test_access.json",dict(frozen_hash=fingerprint,test_y_sha256=meta["sha256"]["test_y.npy"],
                                               test_x_sha256=meta["sha256"]["test.npy"]))
    test,ye=load_split(parent.DEFAULT_ROOT,dataset,"test")
    test_pred=pred(final,test,chosen)
    test_scores=score(ye,test_pred)
    final_s=clock()-begin
    np.savez_compressed(out/"predictions.npz",y_guard=yg,guard_rows=guard_idx,
                        guard_full=full_pred,guard_checks=np.stack(guard_predictions[1:]),
                        y_test=ye,test_pred=test_pred,guard_selected=full_pred if accepted=="full_fallback" else guard_predictions[-1])
    result=dict(dataset=dataset,method=method,width=len(cols),seed=42,anchor_k=len(anchor),front_k=len(support),
                initial_k=len(candidate),k=len(chosen),selected=chosen,frozen_hash=fingerprint,
                fit_count=fits,fits=fits,front_fits=0,rank_fits=rank_fits,guard_fits=guard_fits,
                front_s=front_s,rank_s=rank_s,selector_s=front_s+rank_s,guard_s=guard_s,
                final_s=final_s,reference_s=ref_s,validation_full=reference,validation_selected=accepted_scores,
                checks=len(checks),accepted_route=accepted,fallback=accepted=="full_fallback",
                repaired=len(checks)>1 and accepted!="full_fallback",witness_predictions=diag.get("witness_predictions",0),
                uses_fitted_model=True,test_scores=test_scores,elapsed_s=clock()-started,status="completed")
    atomic_write_json(out/"result.json",result)
    progress("completed",k=len(chosen))


def worker(dataset,method,out):
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            trial(dataset,method,out)
    except Exception:
        (out/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


def build_protocol(primary):
    spec=read(primary/"protocol.json")
    for name,checksum in spec["source_sha256"].items():
        if parent._sha(parent.ROOT/name)!=checksum:
            raise ValueError(f"Primary benchmark source changed: {name}")
    if spec["seed"]!=42 or spec["benchmark_sha256"]!=parent._sha(parent.BASE/"protocol.json"):
        raise ValueError("Primary protocol fingerprint differs")
    for d in parent.DATASETS:
        for rep in (1,2,3):
            expected=parent.METHODS if rep==1 else ("tree_min","pair_min","tree_min_pfi","pair_min_pfi")
            for m in expected:
                folder=primary/(f"repeat-{rep}" if rep>1 else "")/d/m
                if not (folder/"watchdog.json").is_file() or read(folder/"watchdog.json")["status"] not in (
                        "completed","timed_out","budget_exhausted"):
                    raise ValueError(f"Primary timing is unfinished for {d}/{m}, round {rep}")
        parent.prepare(parent.DEFAULT_ROOT,d)
        if spec["data"][d]["hash"]!=parent._sha(parent.DEFAULT_ROOT/d/"metadata.json"):
            raise ValueError("Source data changed")
    return dict(datasets=list(parent.DATASETS),seed=42,methods=list(METHODS),limit_s=300.,data=spec["data"],
                primary_sha256=parent._sha(primary/"protocol.json"),benchmark_sha256=spec["benchmark_sha256"],
                source_sha256={**spec["source_sha256"],**{s:parent._sha(parent.ROOT/s) for s in EXTRA_SOURCES}},
                exploratory=True,guard="PFI candidate if applicable, witness support, original gain anchor, double gain anchor, then full; cache models and deduplicate. Thresholds identical to parent.")


def run(root,primary,report_only=False):
    if root.resolve() in (primary.resolve(),parent.BASE.resolve()):
        raise ValueError("Witness test requires its own run directory")
    root.mkdir(parents=True,exist_ok=True)
    with parent.exclusive_run(primary),parent.exclusive_run(root):
        spec=build_protocol(primary)
        if (root/"protocol.json").exists() and read(root/"protocol.json")!=spec:
            raise ValueError("Witness source/input identity changed")
        if report_only:
            if not (root/"protocol.json").is_file():
                raise ValueError("Missing witness run")
        else:
            atomic_write_json(root/"protocol.json",spec)
            with (root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
                def log(msg):
                    line=f"[{datetime.now(parent.MSK):%Y-%m-%d %H:%M:%S}] {msg}"
                    print(line,flush=True)
                    print(line,file=stream,flush=True)
                for i,d in enumerate(parent.DATASETS):
                    order=METHODS[i%len(METHODS):]+METHODS[:i%len(METHODS)]
                    for m in order:
                        folder=root/d/m
                        if (folder/"watchdog.json").exists():
                            old=read(folder/"watchdog.json")
                            if old["status"] not in ("completed","timed_out"):
                                raise RuntimeError(f"Check {folder}")
                            continue
                        if folder.exists():
                            raise RuntimeError(f"Interrupted run, unknown spent time: {folder}")
                        folder.mkdir(parents=True)
                        log(f"{d}/{m} started")
                        proc=mp.get_context("spawn").Process(target=worker,args=(d,m,folder))
                        try:
                            timer=parent.supervise(proc,folder,parent.LIMIT,log,heartbeat_s=15.,
                                heartbeat_formatter=lambda p,e,left:f"{d}/{m} {p.get('phase','loading')} elapsed={e:.1f}s left={left:.1f}s")
                        finally:
                            proc.close()
                        if timer["status"]=="error":
                            raise RuntimeError(f"Inspect {folder/'error.log'}")
                        log(f"{d}/{m} {timer['status']} cold={timer['elapsed_s']:.2f}s k={timer.get('result',{}).get('k')}")
        from experiments.test.report_compact_witness import report
        report(root,primary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--primary",type=Path,default=parent.DEFAULT)
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    run(args.run.resolve(),args.primary.resolve(),args.report_only)

