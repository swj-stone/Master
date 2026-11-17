"""Independently check the negative edges and optional ridge guard on every eligible split."""
import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import witness_guard as experiment
from experiments.minpi_compression import benchmark as b
from minppi_mofs.dictator_minpi import rank_bounds


def svd_predict(x,y,task,train,assess,columns):
    scale = x[train].std(axis=0)
    z = (x-x[train].mean(axis=0))/np.where(scale==0,1.,scale)
    targets = b.targets(y[train],task,len(np.unique(y[train])))
    if len(columns):
        model = Ridge(alpha=1., solver="svd").fit(z[train][:,columns],targets)
        return model.predict(z[assess][:,columns])
    return np.broadcast_to(targets.mean(axis=0), (len(assess),*targets.shape[1:]))


def audit():
    receipts = json.loads((experiment.OUT/"lock.json").read_bytes())
    rows = []
    n_witness = 0
    for key,digest in receipts.items():
        file=experiment.OUT/key/"selection.json"
        assert b.digest(file)==digest
        state=json.loads(file.read_bytes())
        x,y,task,idx=experiment.split(state["dataset"],state["seed"])
        recs={r["method"]:r for r in state["rows"] if r["repeat"]==0}
        w=recs["witness"]
        if w["h"]:
            train,vote=idx["fit"],idx["vote"]
            prep=b.transform(x,train)
            rank=rank_bounds(prep.transform(x[train]),y[train],prep.transform(x[vote]),y[vote],
                             task=task,seed=state["seed"],closure=True,**b.DESIGN["engine"])
            observed=np.flatnonzero(rank.minimum==-1)
            assert set(w["negative"])==set(w["prefix"]) & set(observed)
            truth=b.targets(y[vote],task,len(np.unique(y[train])))
            def loss(cols):
                pred=svd_predict(x,y,task,train,vote,cols)
                return np.square(pred-truth).sum()/len(vote)
            scale=loss([]) or 1.
            quota=.95/(1.+loss(np.arange(x.shape[1]))/scale)
            for j in w["negative"]:
                left=list(rank.witness[j])
                assert j not in left and 1./(1.+loss(left)/scale)>=quota>1./(1.+loss(sorted(left+[j]))/scale)
                n_witness+=1
        if not w["optional_refits"]:
            continue
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        v=idx["width"]
        def inner(cols):
            return b.metrics(y[v],svd_predict(x,y,task,pool,v,cols),task,max(y[pool].std(),1e-12))
        full=inner(np.arange(x.shape[1]))
        for method in ("witness","tail","uniform"):
            rec=recs[method]
            proposed=sorted(set(w["prefix"])-set(rec["removed"]))
            tested=inner(proposed)
            accepted=b.eligible(tested,full,task)
            assert len(rec["removed"])==w["h"] and accepted==rec["accepted"]
            assert sorted(rec["columns"])==(proposed if accepted else sorted(w["prefix"]))
            for metric,val in tested.items():
                np.testing.assert_allclose(rec["optional"][metric],val,atol=1e-10)
            rows.append(dict(dataset=state["dataset"],seed=state["seed"],task=task,method=method,
                             h=w["h"],from_width=len(w["prefix"]),accepted=accepted,
                             same_deletion_as_witness=set(rec["removed"])==set(w["removed"]),**tested))
    pd.DataFrame(rows).to_csv(experiment.EVIDENCE/"guard-audit.csv",index=False)
    print(f"Verified {n_witness} negative feature/split instances and {len(rows)} optional guard predictions")


def paired_intervals():
    """Resample whole sources within domain, keeping each paired five-split mean together."""
    raw=pd.read_csv(experiment.EVIDENCE/"scores.csv")
    rows=[]
    for domain,g in raw.groupby("task"):
        selected=g[g.method=="witness"].set_index(["dataset","seed"])
        for method in ("coefficient","tail","uniform","full"):
            other=g[g.method==method].set_index(["dataset","seed"])
            fields=["auc","accuracy"] if domain=="classification" else ["r2","scaled_mae"]
            log_fields=["retained","deployment_s"] if method=="full" else ["retained","selection_s","deployment_s"]
            diffs=selected[fields]-other[fields]
            for key in log_fields:
                diffs[key]=np.log(selected[key]/other[key])
            means=diffs.groupby(level="dataset").mean()
            draws=np.asarray(list(product(range(len(means)), repeat=len(means))))
            for key in means:
                s=means[key].to_numpy()
                values=np.r_[s.mean(),np.quantile(s[draws].mean(axis=1),[.025,.975])]
                est,lo,hi=np.exp(values) if key in log_fields else values
                rows.append(dict(task=domain,baseline=method,quantity=key,estimate=est,lo=lo,hi=hi))
    pd.DataFrame(rows).to_csv(experiment.EVIDENCE/"paired-source-intervals.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        audit()
        paired_intervals()
