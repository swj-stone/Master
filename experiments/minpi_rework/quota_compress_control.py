"""Stronger inexpensive search, declared after four primary sources and before its run.

Compare the best of energy-prefix descent, backward descent and forward
addition with one-swap improvement at each cardinality. Ties prefer lower
calibration error, then original column order. Reuse the original 14 tasks,
rho=.20 and seed=42, keep this secondary experiment separate.
"""
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_rework import quota_compress as q
from experiments.minpi_rework import quota_compress_run as study


def improve(g,keep):
    keep=keep.copy()
    while keep.any() and not keep.all():
        inside,outside=np.flatnonzero(keep),np.flatnonzero(~keep)
        v=g[:,outside].sum(axis=1)
        delta=(2*v+g.diagonal())[inside,None]+(g.diagonal()-2*v)[None,outside]-2*g[np.ix_(inside,outside)]
        i,j=np.unravel_index(delta.argmin(),delta.shape)
        if delta[i,j]>=-1e-12:
            break
        keep[inside[i]],keep[outside[j]]=False,True
    return keep


def from_feasible(g,keep):
    best=improve(g,keep)
    while best.any():
        inside=np.flatnonzero(best)
        linear=2*g[:,~best].sum(axis=1)+g.diagonal()
        candidate=best.copy()
        candidate[inside[np.argmin(linear[inside])]]=False
        candidate=improve(g,candidate)
        if not q.feasible(g,candidate,q.RHO):
            break
        best=candidate
    return best


def run():
    jobs=study.legacy.jobs()
    rows=[]
    for name in study.NAMES:
        x,y,task,idx=study.load(name,jobs)
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        teacher=study.fit_additive(x[pool],y[pool],task,"linear")
        parts=teacher.parts(x[idx["width"]])-teacher.center
        a=parts.transpose(0,2,1).reshape(-1,x.shape[1])
        g=(a.T@a)/max(1e-100,(a.sum(axis=1)**2).sum())
        t=perf_counter()
        energy_order=np.lexsort((np.arange(len(g)),-g.diagonal()))
        a=from_feasible(g,q.prefix(g,energy_order))
        b=from_feasible(g,np.ones(len(g),bool))
        keep=np.zeros(len(g),bool)
        while not q.feasible(g,keep,q.RHO):
            c=np.flatnonzero(~keep)
            returns=2*g[:,c].sum(axis=1)-g.diagonal()
            keep[c[np.argmax(returns[c])]]=True
            keep=improve(g,keep)
        f=from_feasible(g,keep)
        best=min([a,b,f],key=lambda z:(int(z.sum()),q.loss(g,z),tuple(np.flatnonzero(z))))
        search=perf_counter()-t
        assert q.feasible(g,best,q.RHO)
        cols=np.flatnonzero(best)
        pred=teacher.restrict(cols).predict(x[np.ix_(idx["test"],cols)])
        if task=="regression":
            pred=pred.ravel()
        sub=study.OUT/name
        sub.mkdir(parents=True,exist_ok=True)
        np.save(sub/"control-prediction.npy",pred)
        rows.append(dict(dataset=name,task=task,k=len(cols),p=len(g),search_s=search,
            energy_descent=int(a.sum()),backward=int(b.sum()),forward=int(f.sum()),
            relative_error=float(np.sqrt(max(0,q.loss(g,best)))),selected=" ".join(map(str,cols)),
            forecast=study.b.digest(sub/"control-prediction.npy"),scale=max(np.std(y[pool]),1e-12)))
        print(name,len(cols),f"{search:.3f}s",flush=True)
    pd.DataFrame(rows).to_csv(study.OUT/"control-lock.csv",index=False)


def score():
    f=pd.read_csv(study.OUT/"control-lock.csv")
    assert set(f.dataset)==set(study.NAMES) and not f.dataset.duplicated().any()
    jobs=study.legacy.jobs()
    rows=[]
    for r in f.to_dict("records"):
        n=r["dataset"]
        path=study.OUT/n/"control-prediction.npy"
        assert study.b.digest(path)==r["forecast"]
        if n in study.OLD:
            d,entry=jobs[n,study.SEED]
            target=d/entry["test_relpath"]
            assert study.b.digest(target)==entry["test_file_sha256"]
            y=pd.read_csv(target,usecols=[entry["target"]])[entry["target"]].to_numpy()
            with np.load(study.legacy.DEST/"novel-input-rows.npz") as masks:
                subsets=dict(primary=slice(None),novel=masks[f"{n}-{study.SEED}"])
        else:
            _,v,_,idx=study.development_split(n,study.SEED)
            y=v[idx["test"]]
            subsets=dict(primary=slice(None))
        pred=np.load(path)
        for p,s in subsets.items():
            rows.append(dict(**r,part=p,**study.b.metrics(y[s],pred[s],r["task"],r["scale"])))
    study.EVIDENCE.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(study.EVIDENCE/"local-control.csv",index=False)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["run","score"])
    args=parser.parse_args()
    with threadpool_limits(1):
        (run if args.action=="run" else score)()
