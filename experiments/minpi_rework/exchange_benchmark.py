"""Fixed-width exchange screening, a small-feature search and exact matched controls."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, legacy_gain as legacy
from experiments.minpi_rework.forward import development_split
from minppi_mofs.additive_minpi import fit_additive, quadratic
from minppi_mofs.exchange_minpi import solve, value, enumerate_best


def nested_search(gram,d,order,k,mode,window):
    """Hold leading features fixed and optimize a band of columns around rank k."""
    p=len(d)
    w=min(window,p)
    lo=min(max(k-w//2,0),p-w)
    fixed,free=order[:lo],order[lo:lo+w]
    g=gram[np.ix_(free,free)]
    coef=d[free]-2*gram[np.ix_(free,fixed)].sum(axis=1)
    chosen,info=solve(g,coef,k-lo,mode=mode,backend="enum")
    return np.sort(np.r_[fixed,free[chosen]]),dict(**info,universe=w,local_k=k-lo)


def greedy_exchange(gram,d,order,k):
    """Swap until no strictly improving one-for-one move remains."""
    support=np.zeros(len(d),bool)
    support[order[:k]]=True
    off=gram.copy()
    np.fill_diagonal(off,0.)
    while True:
        inside,outside=np.flatnonzero(support),np.flatnonzero(~support)
        if not len(outside):
            break
        marginal=d-2*off[:,inside].sum(axis=1)
        gain=marginal[outside,None]-marginal[inside]+2*off[np.ix_(outside,inside)]
        i,j=np.unravel_index(gain.argmax(),gain.shape)
        if gain[i,j]<=1e-12*max(np.max(np.abs(d)),1e-12):
            break
        support[outside[i]],support[inside[j]]=True,False
    return np.flatnonzero(support)


def select(x,y,task,idx,seed,method,kind="linear",window=24,target="labels"):
    start=perf_counter()
    fit,vote,width,test=(idx[k] for k in ("fit","vote","width","test"))
    pool=np.sort(np.r_[fit,vote])
    dev=np.sort(np.r_[pool,width])
    p=x.shape[1]
    trace=[]
    anchor_fit=target=="teacher"
    if anchor_fit:
        receiver=fit_additive(x[pool],y[pool],task,kind)
    if method=="full":
        cols=np.arange(p)
        clock=perf_counter()-start
    else:
        if method=="gain":
            order,_=legacy.gain_rank("gain",x,y,task,idx,seed)
        else:
            teacher=receiver if anchor_fit else fit_additive(x[fit],y[fit],task,kind)
            parts=teacher.parts(x[pool if anchor_fit else vote])-teacher.center
            if method=="energy":
                energy=np.einsum("npc,npc->p",parts,parts)/len(parts)
            else:
                if anchor_fit:
                    response=parts.sum(axis=1)
                else:
                    base=teacher.intercept+teacher.center.sum(axis=0)
                    response=(np.eye(len(base))[y[vote]] if task=="classification" else y[vote,None])-base
                gram,d=quadratic(parts,response)
                energy=np.diag(gram)
            order=np.lexsort((np.arange(p),-energy))
        widths=sorted({1,*(math.ceil(f*p) for f in [.125,.25,.5,.75,1.])})
        # Receiver is fixed-part additive compression; all controls get the same width guard.
        if not anchor_fit:
            receiver=fit_additive(x[pool],y[pool],task,kind)
        validation=receiver.parts(x[width])-receiver.center
        intercept=receiver.intercept+receiver.center.sum(axis=0)
        scale=max(np.std(y[pool]),1e-12)
        def scores(cols):
            pred=intercept+validation[:,cols].sum(axis=1)
            return b.metrics(y[width],pred if task=="classification" else pred.ravel(),task,scale)
        full=scores(np.arange(p))
        for k in widths:
            cols=np.sort(order[:k])
            if k==p:
                break
            if method in ("width","unrestricted","direct"):
                cols,info=nested_search(gram,d,order,k,method,window)
            elif method=="greedy":
                cols=greedy_exchange(gram,d,order,k)
                info={}
            else:
                info={}
            metric=scores(cols)
            objective=value(gram,d,cols) if method not in ("energy","gain") else None
            trace.append(dict(k=k,objective=objective,selected=cols.tolist(),**info))
            # Guard demands parity with full on the tuning fold.
            valid=(metric["auc"]>=full["auc"] and metric["accuracy"]>=full["accuracy"]
                   if task=="classification" else metric["r2"]>=full["r2"] and metric["mae"]<=full["mae"])
            if valid:
                break
        clock=perf_counter()-start
    deploy=(receiver if anchor_fit else fit_additive(x[dev],y[dev],task,kind)).restrict(cols)
    pred=deploy.predict(x[np.ix_(test,cols)])
    return dict(method=method,k=len(cols),p=p,support=cols.tolist(),trace=trace,selection_s=clock,
                deployment_s=perf_counter()-start),pred if task=="classification" else pred.ravel()


def load_dev(name,seed,manifest):
    if (name,seed) in manifest:
        root,entry=manifest[name,seed]
        x,y,task,idx=legacy.read_split(root,entry,seed)
        outer=pd.read_csv(root/entry["test_relpath"],usecols=[entry["target"]])[entry["target"]].to_numpy()
        with np.load(legacy.DEST/"novel-input-rows.npz") as a:
            mask=a[f"{name}-{seed}"].copy()
        return x,y,task,idx,outer,mask
    x,y,task,idx=development_split(name,seed)
    outer=y[idx["test"]].copy()
    y=y.copy()
    y[idx["test"]]=-1
    return x,y,task,idx,outer,np.ones(len(outer),bool)


def warm():
    enumerate_best(np.eye(3),np.arange(3,dtype=float),1)


def development(args):
    out=b.ROOT/"results/minpi_rework"/args.run
    out.mkdir(parents=True,exist_ok=False)
    warm()
    manifest=legacy.jobs() if any(n in legacy.CONFIG["datasets"] for n in args.names) else {}
    rows=[]
    for name in args.names:
        for seed in args.seeds:
            start=perf_counter()
            x,y,task,idx,outer,mask=load_dev(name,seed,manifest)
            scale=max(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]]),1e-12)
            records=[]
            for method in np.random.default_rng(seed).permutation(args.methods):
                rec,pred=select(x,y,task,idx,seed,method,args.kind,args.window,args.target)
                records.append(rec)
                metric=b.metrics(outer[mask],pred[mask],task,scale)
                rows.append(dict(dataset=name,task=task,seed=seed,**{k:rec[k] for k in ("method","p","k","selection_s","deployment_s")},**metric))
            (out/f"{name}-{seed}.json").write_text(json.dumps(records),encoding="utf8")
            pd.DataFrame(rows).to_csv(out/"scores.csv",index=False)
            print(f"{name}/{seed} {perf_counter()-start:.2f}s",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--kind",choices=["linear","spline"],default="linear")
    cli.add_argument("--names",nargs="+",default=["glass","raisin","wifi","biodeg","landsat","musk1","wall","aquatic","power","yacht","slump","estate","hardware","fires"])
    cli.add_argument("--methods",nargs="+",default=["width","unrestricted","direct","energy","full"])
    cli.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    cli.add_argument("--window",type=int,default=24)
    cli.add_argument("--target",choices=["labels","teacher"],default="labels")
    cli.add_argument("--run",required=True)
    with threadpool_limits(1):
        development(cli.parse_args())
