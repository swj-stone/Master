"""Predeclared four-source wide-input comparison; collect and score are separate commands."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import wide_veto_data as data, predictive_veto_native as native
from experiments.minpi_rework import predictive_veto_timing as timing, quota_coalition_rf as rf
from experiments.minpi_rework import predictive_guard as guard
from minppi_mofs import predictive_quota as pq


ROOT=b.ROOT/"results/minpi_rework/wide-veto"
OUT=b.ROOT/"thesis-paper/evidence/minpi_rework/wide-veto"
ARMS=("early","straight","blas","coalesced_early","coalesced_straight","coalesced_blas")


@njit(cache=True)
def _representatives(keep,live):
    active=keep.copy()
    first=-1
    for j in range(len(keep)):
        if keep[j] and live[j]==0:
            if first<0:
                first=j
            else:
                active[j]=False
    return active


@njit(cache=True)
def _step(j,fold,live,keep,order,t,alive,delta,current,linear,ptr,idx,gv,ip,ii,jp,ji,unions):
    if fold and live[j]==0:
        # All dormant columns have identical loss; no group changes in this run.
        for col in range(j,len(keep)):
            if keep[col]:
                if live[col]:
                    break
                order[t]=col
                t+=1
                keep[col]=False
        return t
    if fold:
        for pos in range(jp[j],jp[j+1]):
            g=ji[pos]
            if alive[g]:
                for k in range(ip[g],ip[g+1]):
                    live[ii[k]]-=1
    current+=delta[j]
    native.merge_update(j,alive,delta,linear,ptr,idx,gv,ip,ii,jp,ji,unions)
    keep[j]=False
    order[t]=j
    return t+1


@njit(cache=True)
def _folded(fn,initial,linear,ptr,idx,gv,ip,ii,jp,ji,unions,w,tol):
    live=np.diff(jp)
    keep=np.ones(len(live),np.bool_)
    alive=np.ones(len(ip)-1,np.bool_)
    delta=initial.copy()
    current=np.zeros(len(initial[0]))
    order=np.empty(len(keep),np.int64)
    t=count=0
    last=len(w)-1
    while t<len(keep):
        mask=_representatives(keep,live)
        j,last,q=fn(current,delta,mask,w,tol,last)
        count+=q
        t=_step(j,True,live,keep,order,t,alive,delta,current,linear,ptr,idx,gv,ip,ii,jp,ji,unions)
    return order,count


def args_for(game):
    st=game.structure
    inc,by,g=st.incidence,st.by_feature,st.gram
    p=inc.shape[1]
    w=st.weights[1:]*p/np.arange(1,p+1)[:,None]
    tol=128*np.finfo(float).eps*(abs(game.linear).sum()+abs(g.data).sum()+game.null)
    return (game.initial,game.linear,g.indptr,g.indices,g.data,inc.indptr,inc.indices,by.indptr,
            by.indices,st.unions,np.ascontiguousarray(w),tol)


def rank(method,args):
    folded=method.startswith("coalesced_")
    if not method.endswith("blas"):
        fn=timing.EARLY if method.endswith("early") else native.straight
        return (_folded(fn,*args) if folded else native.fused(fn,native.merge_update,*args))
    initial,linear,ptr,idx,gv,ip,ii,jp,ji,unions,w,tol=args
    live=np.diff(jp)
    keep=np.ones(len(live),bool)
    alive=np.ones(len(ip)-1,bool)
    current=np.zeros(initial.shape[1])
    delta=initial.copy()
    order=np.empty(len(live),int)
    t=count=0
    wt=np.ascontiguousarray(w.T)
    while t<len(live):
        mask=_representatives(keep,live) if folded else keep
        ids=np.flatnonzero(mask)
        losses=((current+delta[ids])@wt).max(axis=1)
        j=ids[np.flatnonzero(losses<=losses.min()+tol)[0]]
        count+=len(ids)*len(w)
        t=_step(j,folded,live,keep,order,t,alive,delta,current,linear,ptr,idx,gv,ip,ii,jp,ji,unions)
    return order,count


def digest_paths():
    import minppi_mofs.availability_minpi as availability
    files=[Path(m.__file__) for m in (native,timing,rf,guard,pq,data,b,availability)]
    files.extend((Path(__file__),data.MANIFEST,Path(__file__).with_name("wide_veto_protocol.md")))
    return {file.name:b.digest(file) for file in files}


def choose(model,raw,pg,y,task,scale,order):
    p=raw.p
    preds=guard.margin_path(model,raw,order)
    clean=guard.passing_native(y,preds,task,scale)
    risk=pg.path_curves(order)[:,1:]*(p/np.arange(1,p+1))
    good=clean & (risk.max(axis=1)<=.02*pg.null+1e-12*max(1.,pg.null))
    return np.sort(order[np.flatnonzero(good)[-1]:])


def bench(x,y,task,idx,seed):
    t=perf_counter()
    model=rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
    fit_s=perf_counter()-t
    t=perf_counter()
    raw=model.game(x[idx["vote"]])
    game=pq.PredictiveGame.from_paths(raw.a,raw.incidence,timing.centered(model,y[idx["vote"]],task))
    args=args_for(game)
    prepare_s=perf_counter()-t
    path=None
    rows=[]
    rng=np.random.default_rng(seed)
    for repeat in range(3):
        for name in rng.permutation(ARMS):
            tick=perf_counter()
            order,n=rank(name,args)
            elapsed=perf_counter()-tick
            if path is None:
                path=order.copy()
            else:
                np.testing.assert_array_equal(path,order)
            rows.append(dict(method=str(name),repeat=repeat,rank_s=elapsed,checks=int(n)))
    tick=perf_counter()
    val=model.game(x[idx["width"]])
    pg=pq.PredictiveGame.from_paths(val.a,val.incidence,timing.centered(model,y[idx["width"]],task))
    scale=max(np.std(y[idx["fit"]]),1e-12)
    selected=choose(model,val,pg,y[idx["width"]],task,scale,path)
    mdi=choose(model,val,pg,y[idx["width"]],task,scale,np.argsort(model.model.feature_importances_,kind="stable"))
    choice=selected if len(selected)<len(mdi) else mdi
    acceptance_s=perf_counter()-tick
    sets=dict(full=np.arange(raw.p),min=selected,mdi=mdi,portfolio=choice)
    forecasts={}
    for key,cols in sets.items():
        tick=perf_counter()
        forecasts[key]=model.predict(x[np.ix_(idx["test"],cols)],cols)
        if key=="portfolio":
            pred_s=perf_counter()-tick
    return forecasts,dict(sets={key:cols.tolist() for key,cols in sets.items()},path=path.tolist(),
         rows=rows,fit_s=fit_s,prepare_s=prepare_s,acceptance_s=acceptance_s,pred_s=pred_s,
         voting_active=int((np.diff(game.structure.by_feature.indptr)>0).sum()),p=raw.p,scale=float(scale))


def warm():
    r=np.random.default_rng(974)
    from scipy.sparse import csc_matrix,csr_matrix
    inc=csr_matrix([[1,0,0,0,0],[1,0,1,0,0],[0,0,1,0,1]])
    g=pq.PredictiveGame.from_paths(csc_matrix(r.integers(-2,3,(8,3)).astype(float)),inc,r.normal(size=8))
    args=args_for(g)
    for name in ARMS:
        observed,_=rank(name,args)
        np.testing.assert_array_equal(observed,g.ranked_path()[0])


def collect():
    warm()
    stamp=digest_paths()
    OUT.mkdir(parents=True,exist_ok=True)
    lock={}
    for name in ("tehran","uji","oraltox","adverts"):
        for seed in range(42,47):
            folder=ROOT/f"{name}-{seed}"
            folder.mkdir(parents=True,exist_ok=True)
            receipt=folder/"selection.json"
            if not receipt.exists():
                timer=Timer(3600,lambda:os._exit(124))
                timer.daemon=True
                timer.start()
                try:
                    t=perf_counter()
                    x,y,task,idx=data.split(name,seed)
                    assert (y[idx["test"]]==-1).all() if task=="classification" else np.isnan(y[idx["test"]]).all()
                    pred,rec=bench(x,y,task,idx,seed)
                    np.savez_compressed(folder/"forecasts.npz",**pred)
                    rec.update(code=stamp,forecast=b.digest(folder/"forecasts.npz"),dataset=name,seed=seed,task=task,
                               splits={k:v.tolist() for k,v in idx.items()})
                    tmp=folder/"selection.tmp"
                    tmp.write_text(json.dumps(rec),encoding="utf8")
                    tmp.replace(receipt)
                    print(folder.name,f"{perf_counter()-t:.1f}s",flush=True)
                finally:
                    timer.cancel()
            rec=json.loads(receipt.read_bytes())
            assert rec["code"]==stamp and b.digest(folder/"forecasts.npz")==rec["forecast"]
            lock[folder.name]=b.digest(receipt)
    assert len(lock)==20
    (OUT/"lock.json").write_text(json.dumps(dict(code=stamp,receipts=lock),indent=2),encoding="utf8")
    bundle=b.ROOT/"packages/minpi-wide-veto-forecasts.zip"
    with ZipFile(bundle,"w",ZIP_DEFLATED) as z:
        z.write(OUT/"lock.json","lock.json")
        for key in lock:
            for f in ("selection.json","forecasts.npz"):
                z.write(ROOT/key/f,key+"/"+f)
    (OUT/"archive-sha256.txt").write_text(b.digest(bundle)+"  "+bundle.name+"\n",encoding="utf8")


if __name__=="__main__":
    if os.name=="nt" and not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
        raise OSError("CPU affinity not set")
    with threadpool_limits(1):
        collect()
