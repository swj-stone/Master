"""Fair exhaustive straight-size comparator and matched no-allocation union update."""
import ctypes
import json
import os
from threading import Timer
from time import perf_counter
from unittest.mock import patch
import zlib
import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from experiments.minpi_rework import predictive_veto_timing as timing
from experiments.minpi_rework import quota_coalition_rf as rf
from minppi_mofs import predictive_quota as pq


@njit(cache=True)
def straight(current,delta,keep,w,tol,last_size):
    """Exhaustive integer sizes in a regular loop, with the same scalar dot expression."""
    best=np.inf
    values=np.full(len(keep),np.inf)
    arg=np.zeros(len(keep),np.int64)
    count=0
    for j in range(len(keep)):
        if not keep[j]:
            continue
        worst=-np.inf
        for s in range(len(w)):
            val=0.
            for h in range(len(current)):
                val+=(current[h]+delta[j,h])*w[s,h]
            if val>worst:
                worst=val
                arg[j]=s
        values[j]=worst
        best=min(best,worst)
        count+=len(w)
    for j in range(len(keep)):
        if keep[j] and values[j]<=best+tol:
            return j,arg[j],count
    raise ValueError("No finite candidate vote")


@njit(cache=True)
def merge_update(j,alive,delta,linear,ptr,index,weights,ip,ii,jptr,jgroups,unions):
    for g in jgroups[jptr[j]:jptr[j+1]]:
        if not alive[g]:
            continue
        ga=ip[g]
        gb=ip[g+1]
        for k in range(ga,gb):
            delta[ii[k],gb-ga]-=linear[g]
        for edge in range(ptr[g],ptr[g+1]):
            h=index[edge]
            if not alive[h]:
                continue
            degree=unions[edge]
            v=weights[edge] if h==g else 2*weights[edge]
            for k in range(ga,gb):
                delta[ii[k],degree]+=v
            if h!=g:
                k=ga
                for m in range(ip[h],ip[h+1]):
                    f=ii[m]
                    while k<gb and ii[k]<f:
                        k+=1
                    if k==gb or ii[k]!=f:
                        delta[f,degree]+=v
        alive[g]=False


@njit(cache=True)
def fused(fn,update,initial,linear,ptr,index,mass,ip,ii,jptr,jgroups,unions,w,tol):
    p=len(jptr)-1
    keep=np.ones(p,np.bool_)
    alive=np.ones(len(ip)-1,np.bool_)
    delta=initial.copy()
    current=np.zeros(initial.shape[1])
    path=np.empty(p,np.int64)
    last=p-1
    count=0
    for step in range(p):
        j,last,q=fn(current,delta,keep,w,tol,last)
        count+=q
        path[step]=j
        current+=delta[j]
        update(j,alive,delta,linear,ptr,index,mass,ip,ii,jptr,jgroups,unions)
        keep[j]=False
    return path,count


def one(source,name,seed,panel,archive):
    x,y,task,idx=source.split(name,seed)
    t=perf_counter()
    model=rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
    raw=model.game(x[idx["vote"]])
    target=timing.centered(model,y[idx["vote"]],task)
    pg=pq.PredictiveGame.from_paths(raw.a,raw.incidence,target)
    setup=perf_counter()-t
    inc,by,g=pg.structure.incidence,pg.structure.by_feature,pg.structure.gram
    p=inc.shape[1]
    w=pg.structure.weights[1:]*p/np.arange(1,p+1)[:,None]
    tol=128*np.finfo(float).eps*(np.abs(pg.linear).sum()+np.abs(g.data).sum()+pg.null)
    args=(pg.initial,pg.linear,g.indptr,g.indices,g.data,inc.indptr,inc.indices,by.indptr,by.indices,pg.structure.unions,w,tol)
    oracle,oc=pg.ranked_path()
    expected=archive[(archive.dataset==name)&(archive.seed==seed)]
    assert len(expected)==4
    for row in expected.itertuples():
        keep=json.loads(row.support)
        np.testing.assert_array_equal(np.sort(oracle[p-len(keep):]),keep)
        assert int(row.checks)==oc
    rng=np.random.default_rng(zlib.crc32(name.encode())+seed+722)
    methods=("early","native_all","straight","alloc_early","blas")
    rows=[]
    for repeat in range(-1,3):
        for m in rng.permutation(methods):
            if m=="blas":
                with patch.object(pq,"_remove",merge_update):
                    t=perf_counter()
                    order,n=pg.ranked_path(screen=False)
                    elapsed=perf_counter()-t
            else:
                fn=timing.EARLY if m in ("early","alloc_early") else straight if m=="straight" else timing.native_all
                update=pq._remove if m=="alloc_early" else merge_update
                t=perf_counter()
                order,n=fused(fn,update,*args)
                elapsed=perf_counter()-t
            np.testing.assert_array_equal(oracle,order)
            assert n==(oc if m in ("early","alloc_early") else p*p*(p+1)//2)
            if repeat>=0:
                rows.append(dict(panel=panel,dataset=name,seed=seed,p=p,method=m,repeat=repeat,rank_s=elapsed,
                                 fit_to_rank_s=setup+elapsed,setup_s=setup,checks=n))
    return rows


def run():
    devcols=["dataset","seed","method","support","checks"]
    wanted=("min_min","min_pooled","min_intact","min_clean")
    dev=pd.read_csv(timing.DEVELOPMENT/"two-sample-five.csv",usecols=devcols)
    ext=[]
    lock=json.loads((timing.confirm.OUT/"lock.json").read_bytes())["receipts"]
    for name,seed in timing.confirm.jobs():
        folder=timing.confirm.RUN/f"{name}-{seed}"
        assert timing.b.digest(folder/"selection.json")==lock[folder.name]
        meta=json.loads((folder/"selection.json").read_bytes())
        for m in wanted:
            ext.append(dict(dataset=name,seed=seed,method=m,checks=meta["records"][m]["checks"],
                support=json.dumps(meta["records"][m]["support"])))
    archive=pd.concat([dev[dev.method.isin(wanted)],pd.DataFrame(ext)],ignore_index=True)
    sources=(*timing.SOURCES,timing.validation_predictive)
    path=timing.DEST/"native-repeats.csv"
    df=pd.read_csv(path) if path.exists() else pd.DataFrame()
    done=set(zip(df.dataset,df.seed)) if len(df) else set()
    for src in sources:
        panel="external" if src is timing.validation_predictive else "development"
        for name in src.SOURCES:
            for seed in range(42,47):
                if (name,seed) in done:
                    continue
                timer=Timer(3600,lambda:os._exit(124))
                timer.daemon=True
                timer.start()
                try:
                    rows=one(src,name,seed,panel,archive)
                    df=pd.concat([df,pd.DataFrame(rows)],ignore_index=True)
                    temp=path.with_suffix(".tmp")
                    df.to_csv(temp,index=False)
                    temp.replace(path)
                finally:
                    timer.cancel()
            print(panel,name,"complete",flush=True)


if __name__=="__main__":
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        run()
