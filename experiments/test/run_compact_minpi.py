"""Seed-42 compact feature selectors, all using the same frozen XGBoost and validation guard."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from datetime import datetime

from experiments.test.run_piv3_lite_six import DATASETS, DEFAULT as BASE, DEFAULT_ROOT, ROOT, MSK, LIMIT, exclusive_run, prepare, supervise, _sha
from minppi_mofs.experiment_protocol import atomic_write_json,canonical_hash

METHODS=("tree_min","tree_mean","compact_gain","compact_pfi","pair_min","pair_mean","pair_relief","tree_min_pfi","pair_min_pfi")
DEFAULT=ROOT/"results/test/thesis_sampling/PIv3-compact-raw-six-seed42"
SOURCES=("src/minppi_mofs/piv3_compact_tree.py","src/minppi_mofs/piv3_compact_filter.py",
         "experiments/test/run_compact_minpi.py","experiments/test/piv3_lite_arm.py","src/minppi_mofs/modeling.py","src/minppi_mofs/utility.py")

def read(path):
    return json.loads(path.read_bytes())

def trial(dataset,method,out):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from experiments.test.piv3_raw_splits import load_split
    from experiments.test.piv3_lite_arm import _matrix,_metrics,_predict,_pfi,_positive,_failures
    from minppi_mofs.modeling import make_model
    from minppi_mofs import piv3_compact_tree as tree, piv3_compact_filter as pair

    started=time.perf_counter()
    fits=0
    def progress(phase,**details):
        atomic_write_json(out/"progress.json",dict(phase=phase,fits=fits,elapsed_s=time.perf_counter()-started,**details))
    progress("loading")
    meta=read(DEFAULT_ROOT/dataset/"metadata.json")
    policy={k:v for k,v in meta["policy"].items() if k!="utility"}
    task,utility=policy["task"],meta["utility"]
    regression, fraud=task=="regression",dataset=="credit_card_fraud"
    train,y=load_split(DEFAULT_ROOT,dataset,"train")
    val,yv=load_split(DEFAULT_ROOT,dataset,"val")
    cols=list(train.columns)
    rank_rows, guard_rows=train_test_split(np.arange(len(val)),test_size=.5,random_state=42,stratify=None if regression else yv)
    rank_rows, guard_rows=np.sort(rank_rows),np.sort(guard_rows)
    ranking=val.iloc[rank_rows]
    yr=yv[rank_rows]
    guard=val.iloc[guard_rows]
    yg=yv[guard_rows]
    def fit(names):
        nonlocal fits
        model=make_model(**policy)
        model.fit(_matrix(train if names==cols else train.loc[:,names]),y)
        fits+=1
        return model
    def pred(model,frame,names):
        return _predict(model,_matrix(frame if names==cols else frame.loc[:,names]),regression)
    def scores(labels,prediction):
        return _metrics(labels,prediction,utility,fraud)
    def canon(names):
        members=set(names)
        if not members or len(members)!=len(names) or not members<=set(cols):
            raise ValueError("Invalid candidate coordinates")
        return [n for n in cols if n in members]

    progress("reference")
    begin=time.perf_counter()
    full=fit(cols)
    full_pred=pred(full,guard,cols)
    reference=scores(yg,full_pred)
    ref_s=time.perf_counter()-begin
    progress("compact_selection")
    begin=time.perf_counter()
    options=dict(task=task,utility=utility,seed=42)
    if method=="compact_pfi":
        importance, extra=_pfi(full,ranking,yr,utility,list(range(len(cols))),progress)
        used=full.get_booster().get_score(importance_type="weight")
        active=np.array(sorted(int(key.removeprefix("f")) for key in used),dtype=int)
        active=active if len(active) else np.arange(len(cols))
        order=[cols[i] for i in active[np.argsort(-importance[active],kind="stable")]]
        names,diag=tree.choose_prefix(full,train,ranking,yr,order=order,**options)
        diag.update(order=order,ranking_details=extra)
    elif method.startswith("pair_"):
        aggregate="relief" if method=="pair_relief" else "mean" if method=="pair_mean" else "min"
        names,diag=pair.select(train,y,ranking,yr,aggregate=aggregate,**options)
    else:
        aggregate="gain" if method=="compact_gain" else "mean" if method=="tree_mean" else "min"
        names,diag=tree.select(full,train,y,ranking,yr,aggregate=aggregate,**options)
    front=canon(names)
    order=list(diag["order"])
    if set(order[:len(front)])!=set(front) or len(set(order))!=len(order):
        raise ValueError("The selected subset must be the leading ranked prefix")
    front_s=time.perf_counter()-begin
    rank_s=0.
    candidate=front
    front_model=None
    downstream=method.endswith("_pfi") and method!="compact_pfi"
    if downstream:
        progress("downstream")
        begin=time.perf_counter()
        front_model=full if front==cols else fit(front)
        labels=ranking if front==cols else ranking.loc[:,front]
        importance,rank_details=_pfi(front_model,labels,yr,utility,train.columns.get_indexer(front).tolist(),progress)
        candidate=canon(_positive(front,importance))
        atomic_write_json(out/"ranking.json",rank_details)
        rank_s=time.perf_counter()-begin
    # One proposed model and at most one predeclared rescue; every model fit is charged.
    # The rescue uses twice the original proposal along the same ordering; after an
    # additional downstream selection, it restores its prefilter input instead.
    candidates=[candidate]
    repair=front if downstream and candidate!=front else canon(order[:min(len(order),2*len(front))])
    if repair!=candidate and repair!=cols:
        candidates.append(repair)
    progress("quality_check")
    begin=time.perf_counter()
    before_guard=fits
    checks=[]
    selected=cols
    chosen_pred=full_pred
    for proposed in candidates:
        if proposed==cols:
            pp=full_pred
        else:
            model=front_model if front_model is not None and proposed==front else fit(proposed)
            pp=pred(model,guard,proposed)
        quality=scores(yg,pp)
        failed=_failures(reference,quality)
        checks.append(dict(names=proposed,k=len(proposed),metrics=quality,failed=failed))
        if not failed:
            selected=proposed
            chosen_pred=pp
            break
    guard_fits=fits-before_guard
    guard_s=time.perf_counter()-begin
    fallback=bool(checks[-1]["failed"])
    frozen=dict(dataset=dataset,method=method,seed=42,policy=policy,front=front,initial=candidate,checks=checks,
                selected=selected,guard_rows_hash=canonical_hash(guard_rows.tolist()),ranking_rows_hash=canonical_hash(rank_rows.tolist()))
    atomic_write_json(out/"frozen.json",frozen)
    atomic_write_json(out/"front.json",diag)
    progress("final_refit")
    begin=time.perf_counter()
    final=make_model(**policy)
    arrays=[_matrix(f if selected==cols else f.loc[:,selected]) for f in (train,val)]
    x=np.concatenate(arrays)
    del arrays
    final.fit(x,np.concatenate([y,yv]))
    fits+=1
    del x
    identity=canonical_hash(frozen)
    atomic_write_json(out/"test_access.json",dict(frozen_hash=identity,test_y_sha256=meta["sha256"]["test_y.npy"],
                                                  test_x_sha256=meta["sha256"]["test.npy"]))
    test,ye=load_split(DEFAULT_ROOT,dataset,"test")
    prediction=pred(final,test,selected)
    values=scores(ye,prediction)
    final_s=time.perf_counter()-begin
    np.savez_compressed(out/"predictions.npz",y_guard=yg,guard_rows=guard_rows,guard_full=full_pred,
                        guard_selected=chosen_pred,y_test=ye,test_pred=prediction)
    result=dict(dataset=dataset,method=method,width=len(cols),seed=42,front_k=len(front),initial_k=len(candidate),
                k=len(selected),selected=selected,frozen_hash=identity,fit_count=fits,fits=fits,front_fits=0,
                uses_fitted_model=not method.startswith("pair"),guard_fits=guard_fits,front_s=front_s,rank_s=rank_s,
                selector_s=front_s+rank_s,guard_s=guard_s,final_s=final_s,reference_s=ref_s,guard_ref=reference,
                validation_full=reference,guard_selected=scores(yg,chosen_pred),validation_selected=scores(yg,chosen_pred),checks=len(checks),
                fallback=fallback,repaired=(len(checks)>1 and not fallback),test_scores=values,
                elapsed_s=time.perf_counter()-started,status="completed")
    atomic_write_json(out/"result.json",result)
    progress("completed",k=len(selected))

def worker(dataset,method,out):
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            trial(dataset,method,out)
    except Exception:
        (out/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise

def run(root, reps):
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root),(root/"run.log").open("a",encoding="utf-8",buffering=1) as stream:
        def log(msg):
            line=f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {msg}"
            print(line,flush=True)
            print(line,file=stream,flush=True)
        old=read(BASE/"protocol.json")
        for d in DATASETS:
            prepare(DEFAULT_ROOT,d)
            if _sha(DEFAULT_ROOT/d/"metadata.json")!=old["data"][d]["hash"]:
                raise ValueError("Canonical input differs from original benchmark")
        protocol=dict(data=old["data"],datasets=list(DATASETS),seed=42,methods=list(METHODS),limit_s=LIMIT,
                      source_sha256={f:_sha(ROOT/f) for f in SOURCES},benchmark_sha256=_sha(BASE/"protocol.json"),
                      guard="Same archived 3% scoring half and thresholds (5% AP). First candidate; if needed single "
                            "doubled ranked prefix or prefilter support; otherwise full. Frozen before test.",
                      evaluation="5% max relative deterioration each of full and fast PFI; smaller delivered k "
                                 "and speed relative to standalone PFI, 1.10x timing parity. Report aggregate=mean, "
                                 "matched conventional ranks, all fallbacks; all experiments exploratory on seed 42.")
        if (root/"protocol.json").exists() and read(root/"protocol.json")!=protocol:
            raise ValueError("Source or data changed. Use a new output directory.")
        atomic_write_json(root/"protocol.json",protocol)
        for rep in range(1,reps+1):
            for i in range(len(DATASETS)):
                d=DATASETS[i if rep%2 else -i-1]
                methods=METHODS if rep==1 else ("tree_min","pair_min","tree_min_pfi","pair_min_pfi")
                shift=(i+rep-1)%len(methods)
                order=methods[shift:]+methods[:shift]
                for m in order:
                    path=root/d/m if rep==1 else root/f"repeat-{rep}"/d/m
                    if (path/"watchdog.json").exists():
                        if read(path/"watchdog.json")["status"]=="error":
                            raise RuntimeError(f"Inspect {path/'error.log'}")
                        continue
                    if path.exists():
                        raise RuntimeError(f"Interrupted route at {path}; elapsed budget unknown")
                    path.mkdir(parents=True)
                    prior=sum(read(root/(f"repeat-{r}" if r>1 else "")/d/m/"watchdog.json")["elapsed_s"] for r in range(1,rep))
                    limit=LIMIT-prior
                    if limit<=1:
                        atomic_write_json(path/"watchdog.json",dict(status="budget_exhausted",elapsed_s=0))
                        continue
                    log(f"{rep} {d}/{m} started, remaining cumulative budget {limit:.2f}s")
                    proc=mp.get_context("spawn").Process(target=worker,args=(d,m,path))
                    try:
                        t=supervise(proc,path,limit,log,heartbeat_s=15.,heartbeat_formatter=lambda p,e,left:
                            f"{rep} {d}/{m} {p.get('phase','loading')} elapsed={e:.1f}s left={left:.1f}s")
                    finally:
                        proc.close()
                    if t["status"]=="error":
                        raise RuntimeError(f"Inspect {path/'error.log'}")
                    result=t.get("result",{})
                    log(f"{rep} {d}/{m} {t['status']} k={result.get('k')} elapsed={t['elapsed_s']:.2f}s")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    parser.add_argument("--repeats",type=int,choices=(1,2,3),default=3)
    args=parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    run(args.run.resolve(),args.repeats)

