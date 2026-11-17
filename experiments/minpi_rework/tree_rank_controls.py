"""Post-release comparisons with fixed full-model ablation and mean absolute TreeSHAP."""
import argparse
import ctypes
import os
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import DMatrix
from experiments.minpi_compression import benchmark as b, legacy_gain as legacy
from experiments.minpi_rework import quota_compress_tree as tree, quota_compress_tree_cost as cost
from experiments.minpi_rework import quota_compress_tree_confirmation as confirm
from minppi_mofs.modeling import make_model


OUT=b.ROOT/"results/minpi_rework/tree-ranking-controls"
DEST=confirm.DEST
METHODS=("treeshap","full_ablation")


def collect():
    rows=[]
    for i,(panel,source,name,seed) in enumerate(confirm.jobs()):
        x,y,task,idx=source.split(name,seed)
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        val=x[idx["width"]]
        problem=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
        model=make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
        model.fit(x[pool],y[pool])
        f=tree.extract(model,val[:1])
        game=f.cache(val)
        pred={}
        for m in METHODS:
            costs=[]
            for _ in range(3):
                t=perf_counter()
                if m=="treeshap":
                    contrib=model.get_booster().predict(DMatrix(val),pred_contribs=True,approx_contribs=False,strict_shape=True)
                    weights=np.abs(contrib[:,:,:-1]).mean(axis=(0,1))
                else:
                    singleton=(game.a@game.incidence).toarray()
                    weights=np.einsum("ij,ij->j",singleton,singleton)
                order=np.lexsort((-np.arange(len(weights)),-weights))
                cols,path,_=cost.compiled(game,order)
                costs.append(perf_counter()-t)
            yhat=f.predict(x[np.ix_(idx["test"],cols)],cols)
            pred[m]=tree.as_scores(yhat,task)
            cal=f.predict(val[:,cols],cols)
            error=np.linalg.norm(f.predict(val,np.arange(game.p))-cal)/max(game.norm,1e-100)
            assert error<=.2+1e-10
            rows.append(dict(dataset=name,seed=seed,panel=panel,task=task,method=m,k=len(cols),p=game.p,
                        support=cols.tolist(),rank_and_search_s=float(np.median(costs)),relative_error=error))
        key=OUT/f"{name}-{seed}"
        key.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(key/"predictions.npz",**pred)
        for r in rows[-len(METHODS):]:
            r["digest"]=b.digest(key/"predictions.npz")
        print(f"[{i+1}/65] {name}/{seed}",flush=True)
    pd.DataFrame(rows).to_csv(DEST/"rank-control-receipts.csv",index=False)


def score():
    records=pd.read_csv(DEST/"rank-control-receipts.csv").set_index(["dataset","seed","method"])
    result=[]
    for panel,source,name,seed in confirm.jobs():
        sub=records.loc[(name,seed)]
        assert len(sub)==len(METHODS) and sub.digest.nunique()==1
        vals=source.score(name,seed,OUT/f"{name}-{seed}"/"predictions.npz",sub.digest.iloc[0])
        for m in METHODS:
            result.append(dict(dataset=name,seed=seed,method=m,**sub.loc[m].to_dict(),**vals[m]))
    pd.DataFrame(result).to_csv(DEST/"rank-control-scores.csv",index=False)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=["collect","score"])
    if os.name=="nt":
        ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<4))
    with threadpool_limits(1):
        (collect if cli.parse_args().action=="collect" else score)()
