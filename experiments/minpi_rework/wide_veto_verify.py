"""Independently verify wide-panel predictions by pruning sklearn's fitted node arrays."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_rework import wide_veto_run as run,wide_veto_data as data,quota_coalition_rf as rf
from experiments.minpi_compression import benchmark as b


def native_after_pruning(forest,x,keep):
    backups=[]
    reached=set()
    try:
        for estimator in forest.model.estimators_:
            tree=estimator.tree_
            left,right=tree.children_left,tree.children_right
            stop=np.flatnonzero((left!=-1)&~np.isin(tree.feature,keep))
            backups.append((left,right,stop,left[stop].copy(),right[stop].copy()))
            left[stop]=right[stop]=-1
            todo=[0]
            while todo:
                node=todo.pop()
                if left[node]!=-1:
                    reached.add(int(tree.feature[node]))
                    todo.extend((left[node],right[node]))
        return forest.native(x),len(reached)
    finally:
        for left,right,stop,l,r in backups:
            left[stop],right[stop]=l,r


def main():
    lock=json.loads((run.OUT/"lock.json").read_bytes())
    rows=[]
    for key,hashed in lock["receipts"].items():
        folder=run.ROOT/key
        assert b.digest(folder/"selection.json")==hashed
        r=json.loads((folder/"selection.json").read_bytes())
        assert r["forecast"]==b.digest(folder/"forecasts.npz")
        x,y,task,idx=data.split(r["dataset"],r["seed"])
        rfmodel=rf.fit(x[idx["fit"]],y[idx["fit"]],task,r["seed"])
        with np.load(folder/"forecasts.npz",allow_pickle=False) as f:
            for arm,cols in r["sets"].items():
                pred,n=native_after_pruning(rfmodel,x[idx["test"]],cols)
                err=float(np.max(np.abs(pred-f[arm])))
                np.testing.assert_allclose(pred,f[arm],rtol=2e-10,atol=2e-9)
                rows.append(dict(dataset=r["dataset"],seed=r["seed"],arm=arm,k=len(cols),
                                 reachable_columns=n,maximum_absolute_discrepancy=err))
    df=pd.DataFrame(rows)
    df.to_csv(run.OUT/"native-audit.csv",index=False)
    print(len(df),"native sklearn recalculations; worst difference",df.maximum_absolute_discrepancy.max())
    print(df.groupby(["dataset","arm"])[["k","reachable_columns"]].mean().to_string())


if __name__=="__main__":
    with threadpool_limits(1):
        main()
