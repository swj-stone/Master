"""Timing-only replay of RF rankings on labeled voting rows of frozen splits."""
import argparse
import ctypes
import json
import math
import os
import zlib
from pathlib import Path
from threading import Timer
from time import perf_counter
from unittest.mock import patch
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import quota_coalition_rf as rf,validation_predictive
from experiments.minpi_rework import predictive_confirmation as confirm
from experiments.minpi_rework.availability_trial import SOURCES,DEST as DEVELOPMENT
from experiments.minpi_rework.predictive_mask_study import centered
from minppi_mofs import predictive_quota as pq


DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/predictive-veto-timing"
METHODS=("early","native_all","blas_all")
EARLY=pq._least_worst


@njit(cache=True)
def native_all(current,delta,keep,w,tol,last_size):
    # Matched copy of _least_worst with the early-break comparison removed.
    sizes=len(w)
    checks=0
    best=np.inf
    values=np.full(len(keep),np.inf)
    worst_sizes=np.zeros(len(keep),np.int64)
    for j in range(len(keep)):
        if not keep[j]:
            continue
        worst=-np.inf
        worst_size=0
        for z in range(sizes+3):
            s=last_size if z==0 else sizes-1 if z==1 else 0 if z==2 else z-3
            if (z==1 and s==last_size) or (z==2 and (s==last_size or s==sizes-1)):
                continue
            if z>=3 and (s==last_size or s==sizes-1 or s==0):
                continue
            val=0.
            for h in range(len(current)):
                val+=(current[h]+delta[j,h])*w[s,h]
            checks+=1
            if val>worst:
                worst,worst_size=val,s
        if worst<=best+tol:
            values[j],worst_sizes[j]=worst,worst_size
            best=min(best,worst)
    for j in range(len(keep)):
        if keep[j] and values[j]<=best+tol:
            return j,worst_sizes[j],checks
    raise ValueError("No finite candidate vote")


def blas_all(current,delta,keep,w,tol,last_size):
    ids=np.flatnonzero(keep)
    loss=((current+delta[ids])@w.T).max(axis=1)
    j=ids[np.flatnonzero(loss<=loss.min()+tol)[0]]
    return j,-1,len(ids)*len(w)


@njit(cache=True)
def batched(fn,cs,ds,ks,w,tol,ls,reps):
    total=0
    for _ in range(reps):
        for i in range(len(cs)):
            j,_,q=fn(cs[i],ds[i],ks[i],w,tol,ls[i])
            total+=q+j
    return total


def snapshots(pg):
    st=pg.structure
    inc,g,by=st.incidence,st.gram,st.by_feature
    p=inc.shape[1]
    w=st.weights[1:]*p/np.arange(1,p+1)[:,None]
    tol=128*np.finfo(float).eps*(np.abs(pg.linear).sum()+np.abs(g.data).sum()+pg.null)
    current=np.zeros(len(st.totals))
    keep=np.ones(p,bool)
    alive=np.ones(inc.shape[0],bool)
    delta=pg.initial.copy()
    last=p-1
    steps=[]
    order=[]
    work=[]
    while keep.any():
        steps.append((current.copy(),delta.copy(),keep.copy(),last))
        j,after,k=EARLY(current,delta,keep,w,tol,last)
        j2,l2,k2=native_all(current,delta,keep,w,tol,last)
        j3,_,k3=blas_all(current,delta,keep,w,tol,last)
        assert j==j2==j3 and after==l2 and k<=k2==k3==p*int(keep.sum())
        work.append(k)
        order.append(j)
        current+=delta[j]
        pq._remove(j,alive,delta,pg.linear,g.indptr,g.indices,g.data,inc.indptr,inc.indices,by.indptr,by.indices,st.unions)
        keep[j]=False
        last=after
    cs,ds,ks,ls=(np.stack(col) for col in zip(*steps))
    return steps,(cs,ds,ks,w,tol,ls),np.array(order),sum(work)


def fixture():
    for p in (1,2,3,10):
        c=np.zeros(3)
        d=np.zeros((p,3))
        mask=np.ones(p,bool)
        w=np.ones((p,3))
        for last in range(p):
            for tol in (0.,1e-12):
                assert native_all(c,d,mask,w,tol,last)==EARLY(c,d,mask,w,tol,last)==(0,last,p*p)
                d[:,0]=np.arange(p)*1e-13
                a=EARLY(c,d,mask,w,tol,last)
                other=native_all(c,d,mask,w,tol,last)
                assert a[:2]==other[:2] and 1<=a[2]<=other[2]==p*p
                d[:]=0.
        stack=(np.stack([c]),np.stack([d]),np.stack([mask]),w,0.,np.array([0]))
        for func in [EARLY,native_all]:
            batched(func,*stack,1)


def run_one(panel,source,name,seed,historical,ext_lock):
    x,y,task,idx=source.split(name,seed)
    hidden=y[idx["test"]]
    assert np.isnan(hidden).all() if task=="regression" else (hidden==-1).all()
    if panel=="external":
        receipt=confirm.RUN/f"{name}-{seed}"/"selection.json"
        assert b.digest(receipt)==ext_lock["receipts"][receipt.parent.name]
        meta=json.loads(receipt.read_bytes())
        for part in ("fit","vote"):
            np.testing.assert_array_equal(idx[part],meta["idx"][part])
        refs=[(z["support"],z["checks"]) for m,z in meta["records"].items() if m in ("min_min","min_pooled","min_intact","min_clean")]
    else:
        r=historical[(historical.dataset==name)&(historical.seed==seed)]
        refs=[(json.loads(v.support),int(v.checks)) for v in r.itertuples()]
    assert len(refs)==4
    t=perf_counter()
    model=rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
    fit_s=perf_counter()-t
    t=perf_counter()
    game=model.game(x[idx["vote"]])
    target=centered(model,y[idx["vote"]],task)
    trace_s=perf_counter()-t
    t=perf_counter()
    pg=pq.PredictiveGame.from_paths(game.a,game.incidence,target)
    setup_s=perf_counter()-t
    steps,stack,order,checks=snapshots(pg)
    p=len(order)
    for support,expected in refs:
        np.testing.assert_array_equal(np.sort(order[p-len(support):]),support)
        assert checks==expected
    reps=math.ceil(512/p)
    checksum={k:None for k in METHODS}
    early_n=p*p*(p+1)//2
    rng=np.random.default_rng(zlib.crc32(name.encode())+seed)
    rows=[]
    for repeat in range(3):
        for m in rng.permutation(METHODS):
            f={"early":EARLY,"native_all":native_all,"blas_all":blas_all}[m]
            # Patching and identity checks are outside the path timing.
            with patch.object(pq,"_least_worst",f):
                t=perf_counter()
                result,total=pg.ranked_path("min",screen=m!="blas_all")
                rank_s=perf_counter()-t
            np.testing.assert_array_equal(result,order)
            assert total==(checks if m=="early" else early_n)
            checksum[m]=int(order.sum()+total)
            w,tol=stack[3:5]
            accum=0
            t=perf_counter()
            for _ in range(reps):
                for cur,d,keep,last in steps:
                    j,_,n=f(cur,d,keep,w,tol,last)
                    accum+=j+n
            decision_s=(perf_counter()-t)/reps
            assert accum==reps*checksum[m]
            compiled_s=np.nan
            if m!="blas_all":
                t=perf_counter()
                val=batched(f,*stack,reps)
                compiled_s=(perf_counter()-t)/reps
                assert val==accum
            rows.append(dict(panel=panel,dataset=name,seed=seed,p=p,voting_rows=len(idx["vote"]),repeat=repeat,
                method=m,checks=total,available_queries=early_n,core_s=decision_s,compiled_core_s=compiled_s,
                rank_s=rank_s,fit_s=fit_s,path_setup_s=trace_s,moment_setup_s=setup_s,
                fit_to_rank_s=fit_s+trace_s+setup_s+rank_s,core_cycles=reps))
    return rows


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--names",nargs="+")
    arg.add_argument("--seeds",nargs="+",type=int,default=list(range(42,47)))
    args=arg.parse_args()
    fixture()
    histcols=["dataset","seed","method","support","checks"]
    hist=pd.read_csv(DEVELOPMENT/"two-sample-five.csv",usecols=histcols)
    hist=hist[hist.method.isin(["min_min","min_pooled","min_intact","min_clean"])]
    ext_lock=json.loads((confirm.OUT/"lock.json").read_bytes())
    jobs=[("development",src,n,s) for src in SOURCES for n in src.SOURCES for s in args.seeds]
    jobs.extend(("external",validation_predictive,n,s) for n in validation_predictive.SOURCES for s in args.seeds)
    jobs=[j for j in jobs if not args.names or j[2] in args.names]
    DEST.mkdir(parents=True,exist_ok=True)
    out=DEST/"repeats.csv"
    df=pd.read_csv(out) if out.exists() else pd.DataFrame()
    done=set(zip(df.dataset,df.seed)) if len(df) else set()
    begin=perf_counter()
    for i,(panel,src,n,s) in enumerate(jobs):
        if (n,s) in done:
            continue
        watchdog=Timer(3600,lambda:os._exit(124))
        watchdog.daemon=True
        watchdog.start()
        try:
            tick=perf_counter()
            new=run_one(panel,src,n,s,hist,ext_lock)
            df=pd.concat([df,pd.DataFrame(new)],ignore_index=True)
            staged=out.with_suffix(".tmp")
            df.to_csv(staged,index=False)
            staged.replace(out)
        finally:
            watchdog.cancel()
        elapsed=perf_counter()-begin
        print(f"[{i+1}/{len(jobs)}] {n}/{s} {perf_counter()-tick:.2f}s, elapsed {elapsed:.1f}s",flush=True)


if __name__=="__main__":
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        main()
