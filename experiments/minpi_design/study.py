"""Evaluate compensated endpoint votes with independent selection rows."""
import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import legacy_gain
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import validation_predictive as seven
from experiments.minpi_design.game import Design, Game, endpoint_losses


OUT = b.ROOT/"thesis-paper/evidence/minpi_design"


def load(name,seed):
    if name in seven.SOURCES:
        x,y,task,idx = seven.split(name,seed)
        return x,y,task,idx,lambda:seven._partition(name,seed)[1][idx["test"]]
    directory, row = legacy_gain.jobs()[name,seed]
    x,y,task,idx = legacy_gain.read_split(directory,row,seed)
    return x,y,task,idx,lambda:pd.read_csv(directory/row["test_relpath"],usecols=[row["target"]])[row["target"]].to_numpy()


def target(y, task, c):
    return np.eye(c)[np.asarray(y,int)] if task=="classification" else np.asarray(y,float)[:,None]


def metrics(y, pred, task, scale):
    return b.metrics(y, pred if task=="classification" else pred[:,0], task, scale)


def select(job, basis, methods):
    x,y,task,idx,test_labels=load(*job)
    name,seed=job
    fit=np.r_[idx["fit"],idx["vote"]]
    val,test=idx["width"],idx["test"]
    tick=perf_counter()
    design,z=Design.fit(x[fit],basis)
    c=len(np.unique(y[fit])) if task=="classification" else 1
    yt=target(y[fit],task,c)
    bias=yt.mean(axis=0)
    d=design.center.shape[1]
    g=Game(z,yt-bias,d)
    zv=design.transform(x[val])
    yv=target(y[val],task,c)-bias
    null=np.square(yv).sum()/len(yv)
    scale=max(1e-15,np.std(y[fit]))
    full=np.arange(g.p)
    common_s=perf_counter()-tick
    full_predictions={}
    for mode in ("robust","intact"):
        tick=perf_counter()
        coef=g.head(full,mode=="robust")
        pv,loss=endpoint_losses(zv,yv,full,coef,d)
        full_predictions[mode]=(metrics(y[val],pv+bias,task,scale),loss,perf_counter()-tick,coef)
    locked=[]
    for method in methods:
        tick=perf_counter()
        details=dict(exact=0,possible=0)
        if method in ("min","min_bound","pooled","intact","min_exhaustive"):
            path,details=g.deletion_path(method)
        else:
            if method=="mdi":
                model=(RandomForestClassifier if task=="classification" else RandomForestRegressor)(
                    n_estimators=100,max_depth=4,random_state=seed,n_jobs=1)
                model.fit(x[fit],y[fit])
                score=model.feature_importances_
            elif method=="marginal":
                score=(g.cross**2).reshape(g.p,-1).sum(axis=1)
            elif method=="random":
                score=np.random.default_rng(seed).normal(size=g.p)
            elif method=="qr":
                # Use raw standardized columns, with pivot order intact.
                _,_,piv=qr(z[:,::d],pivoting=True,mode="economic",check_finite=False)
                score=np.empty(g.p)
                score[piv]=np.arange(g.p,0,-1)
            else:
                raise ValueError(method)
            order=np.argsort(-score,kind="stable")
            path=[np.sort(order[:k]) for k in range(g.p,0,-1)]
        rank_s=perf_counter()-tick
        for mode, (base,loss,base_s,_) in full_predictions.items():
            tick=perf_counter()
            for cols in reversed(path):
                coef=g.head(cols,mode=="robust")
                pred,ls=endpoint_losses(zv,yv,cols,coef,d)
                if mode=="robust" and np.max(ls-loss)>.02*null+1e-12:
                    continue
                if b.eligible(metrics(y[val],pred+bias,task,scale),base,task):
                    break
            else:
                raise AssertionError("Full support must reproduce the validation reference")
            sec=perf_counter()-tick
            meta=dict(dataset=name,seed=seed,basis=basis,rank=method,head=mode,
                      task=task,k=len(cols),p=g.p,rank_s=rank_s,select_s=sec,
                      total_s=common_s+rank_s+base_s+sec,**details)
            locked.append((meta,cols,coef.copy()))
    zt=design.transform(x[test])
    test_y=test_labels()
    centered=target(test_y,task,c)-bias
    full_loss={}
    for mode,(_,_,_,coef) in full_predictions.items():
        pt,ls=endpoint_losses(zt,centered,full,coef,d)
        full_loss[mode]=ls
        locked.append((dict(dataset=name,seed=seed,basis=basis,rank="full",head=mode,
                       task=task,p=g.p,k=g.p,rank_s=0.,select_s=0.,total_s=common_s+full_predictions[mode][2],
                       exact=0,possible=0),full,coef))
    rows=[]
    for meta,cols,coef in locked:
        pt,ls=endpoint_losses(zt,centered,cols,coef,d)
        scores=metrics(test_y,pt+bias,task,scale)
        excess=max(ls-full_loss[meta["head"]])/(np.square(centered).sum()/len(centered))
        rows.append(dict(**meta,**scores,robust_excess=excess,cols=",".join(map(str,cols))))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--names",nargs="+",default=["madelon","spambase","gisette","abalone","isolet","credit_card_fraud"])
    p.add_argument("--seeds",nargs="+",type=int,default=[42])
    p.add_argument("--bases",nargs="+",default=["linear","spline"])
    p.add_argument("--methods",nargs="+",default=["min","min_bound","pooled","intact","mdi","marginal","qr","random"])
    p.add_argument("--tag",required=True)
    a=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/(a.tag+".csv")
    rows=[]
    jobs=[(name,seed,basis) for basis in a.bases for seed in a.seeds for name in a.names]
    for i,(name,seed,basis) in enumerate(jobs,1):
        tick=perf_counter()
        rows+=select((name,seed),basis,a.methods)
        pd.DataFrame(rows).to_csv(path,index=False)
        print(f"[{i}/{len(jobs)}] {name} {seed} {basis} {perf_counter()-tick:.2f}s",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
