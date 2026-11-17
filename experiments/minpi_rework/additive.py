"""Evaluate additive vote paths and matched controls with one shared receiver."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, legacy_gain as legacy
from experiments.minpi_rework import forward, path_guard
from minppi_mofs.additive_minpi import fit_additive, rank_additive


METHODS = ("min","pooled","point","energy","coefficient","full")


def select(method,x,y,task,idx,seed,kind,receiver,zero=False):
    start=perf_counter()
    p=x.shape[1]
    fit,vote,width,test=(idx[k] for k in ("fit","vote","width","test"))
    pool=np.sort(np.r_[fit,vote])
    dev=np.sort(np.r_[pool,width])
    def ranking(method,x,y,task,idx,seed):
        if method in ("coefficient","rfe","univariate"):
            return b.ranked(method,x,y,task,idx,seed)
        if method=="gain":
            return legacy.gain_rank(method,x,y,task,idx,seed)
        return rank_additive(x[fit],y[fit],x[vote],y[vote],task,kind,method)
    if receiver=="ridge":
        return path_guard.select(method,x,y,task,idx,seed,ranking,zero_tolerance=zero)
    order=np.arange(p) if method=="full" else ranking(method,x,y,task,idx,seed)[0]
    rank_s=perf_counter()-start
    def shape(train,evals):
        model=fit_additive(x[train],y[train],task,kind)
        center=model.center
        return model.parts(x[evals])-center,model.intercept+center.sum(axis=0)
    width_k=p
    if method!="full":
        parts,offset=shape(pool,width)
        preds=offset+np.cumsum(parts[:,order],axis=1)
        scale=max(1e-12,np.std(y[pool]))
        pred=preds if task=="classification" else preds[:,:,0]
        full=b.metrics(y[width],pred[:,-1],task,scale)
        for k in range(1,p):
            candidate=pred[:,k-1]
            if task=="classification" and (candidate.argmax(axis=1)==y[width]).mean()<full["accuracy"]-(0. if zero else .01):
                continue
            metric=b.metrics(y[width],candidate,task,scale)
            if zero:
                good=(metric["auc"]>=full["auc"] and metric["accuracy"]>=full["accuracy"] if task=="classification"
                      else metric["r2"]>=full["r2"] and metric["mae"]<=full["mae"])
            else:
                good=b.eligible(metric,full,task)
            if good:
                width_k=k
                break
    clock=perf_counter()-start
    # Model is trained using all development columns, but deployment needs retained inputs only.
    # Score here by summing separate functions of only those inputs.
    model=fit_additive(x[dev],y[dev],task,kind)
    selected=np.sort(order[:width_k])
    prediction=model.restrict(selected).predict(x[np.ix_(test,selected)])
    if task=="regression":
        prediction=prediction.ravel()
    return dict(method=method,k=width_k,p=p,order=order.tolist(),rank_s=rank_s,selection_s=clock,
                deployment_s=perf_counter()-start),prediction


def develop(args):
    out=b.ROOT/"results/minpi_rework"/args.run
    out.mkdir(parents=True,exist_ok=False)
    manifest=legacy.jobs() if any(n in legacy.CONFIG["datasets"] for n in args.names) else {}
    rows=[]
    for name in args.names:
        for seed in args.seeds:
            start=perf_counter()
            if (name,seed) in manifest:
                root,entry=manifest[name,seed]
                x,y,task,idx=legacy.read_split(root,entry,seed)
                outer=pd.read_csv(root/entry["test_relpath"],usecols=[entry["target"]])[entry["target"]].to_numpy()
                with np.load(legacy.DEST/"novel-input-rows.npz") as archive:
                    mask=archive[f"{name}-{seed}"].copy()
            else:
                x,y,task,idx=forward.development_split(name,seed)
                outer=y[idx["test"]].copy()
                y=y.copy()
                y[idx["test"]]=-1
                mask=np.ones(len(outer),bool)
            scale=max(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]]),1e-12)
            for method in np.random.default_rng(seed).permutation(args.methods):
                rec,pred=select(method,x,y,task,idx,seed,args.kind,args.receiver,args.zero)
                metrics=b.metrics(outer[mask],pred[mask],task,scale)
                rows.append(dict(dataset=name,task=task,seed=seed,selected=" ".join(map(str,sorted(rec["order"][:rec["k"]]))),
                                 **{k:rec[k] for k in ("p","method","k","selection_s","deployment_s","rank_s")},**metrics))
            pd.DataFrame(rows).to_csv(out/"scores.csv",index=False)
            print(f"{name}/{seed} {perf_counter()-start:.2f}s",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--kind",choices=["linear","spline"],default="spline")
    cli.add_argument("--receiver",choices=["ridge","trim"],default="trim")
    cli.add_argument("--names",nargs="+",default=["glass","raisin","wifi","biodeg","musk1","aquatic","power","yacht","slump","estate","hardware","fires"])
    cli.add_argument("--methods",nargs="+",default=list(METHODS))
    cli.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    cli.add_argument("--zero",action="store_true")
    cli.add_argument("--run",required=True)
    with threadpool_limits(1):
        develop(cli.parse_args())
