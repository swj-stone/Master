"""Matched scaled and unscaled selectors on opened development sources."""
import argparse
import ctypes
import json
import os
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import DMatrix
from experiments.minpi_compression import legacy_gain as legacy,benchmark as b
from experiments.minpi_rework import quota_compress_tree as tree,quota_compress_scale as scale
from experiments.minpi_rework.quota_compress_rescue_run import jobs
from minppi_mofs.modeling import make_model


ROOT=b.ROOT/"results/minpi_rework/quota-scale"
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/quota-scale"


def select(panel,name,seed,api,job,d):
    x,y,task,idx=legacy.read_split(*job,seed) if job else api.split(name,seed)
    val=x[idx["width"]]
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    problem=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
    t=perf_counter()
    model=make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    f=tree.extract(model,val[:1])
    game=f.cache(val)
    common=perf_counter()-t
    full=f.predict(val,np.arange(game.p))
    stored={}
    pred={}
    rng=np.random.default_rng(seed)
    for method in rng.permutation(("gain","ablation","treeshap","exact")):
        t=perf_counter()
        if method in ("gain","exact"):
            gain=model.get_booster().get_score(importance_type="gain")
            weights=np.array([gain.get(f"f{j}",0.) for j in range(game.p)])
        elif method=="ablation":
            singleton=(game.a@game.incidence).toarray()
            weights=np.einsum("ij,ij->j",singleton,singleton)
        else:
            values=model.get_booster().predict(DMatrix(val),pred_contribs=True,approx_contribs=False,strict_shape=True)
            weights=np.abs(values[:,:,:-1]).mean(axis=(0,1))
        order=np.lexsort((-np.arange(len(weights)),-weights))
        rank_s=perf_counter()-t
        for scaled in rng.permutation((False,True)):
            oracle=None
            for votes in ([False,True] if scaled and method=="exact" else [False]):
                t=perf_counter()
                cols,path,risk,lam,counts=scale.search(game,order,scaled,method=="exact",True,votes)
                search_s=perf_counter()-t
                if oracle is not None:
                    np.testing.assert_array_equal(oracle,path)
                oracle=path
                key=method+("_scaled" if scaled else "_raw")+("_votes" if votes else "")
                cal=f.predict(val[:,cols],cols)
                test=tree.as_scores(f.empty+lam*(cal-f.empty),task)
                err=np.linalg.norm(full-(f.empty+lam*(cal-f.empty)))/max(1e-100,game.norm)
                assert err<=.2+1e-8
                np.testing.assert_allclose(err**2,risk/max(1e-100,game.norm**2),atol=2e-8)
                t=perf_counter()
                raw=f.predict(x[np.ix_(idx["test"],cols)],cols)
                pred[key]=tree.as_scores(f.empty+lam*(raw-f.empty),task)
                pred_s=perf_counter()-t
                stored[key]=dict(k=len(cols),p=game.p,cols=cols.tolist(),lam=lam,relative_error=err,
                    rank_s=rank_s,search_s=search_s,common_s=common,predict_s=pred_s,
                    pipeline_s=common+rank_s+search_s+pred_s,checks=int(counts[0]),vetoes=int(counts[1]))
    t=perf_counter()
    pred["full"]=tree.as_scores(f.predict(x[idx["test"]],np.arange(game.p)),task)
    stored["full"]=dict(k=game.p,p=game.p,cols=list(range(game.p)),lam=1.,relative_error=0.,
        pipeline_s=common+perf_counter()-t)
    d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/"forecasts.npz",**pred)
    info=dict(task=task,scale=float(max(np.std(y[pool]),1e-12)),records=stored,forecast=b.digest(d/"forecasts.npz"))
    (d/"receipt.json").write_text(json.dumps(info),encoding="utf8")
    print(name,seed,[(m,stored[m+"_raw"]["k"],stored[m+"_scaled"]["k"],round(stored[m+"_scaled"]["lam"],3))
        for m in ("exact","ablation","treeshap","gain")],flush=True)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=("select","score"))
    cli.add_argument("--all",action="store_true")
    args=cli.parse_args()
    tag="all" if args.all else "pilot"
    work=list(jobs(args.all))
    DEST.mkdir(exist_ok=True,parents=True)
    if args.action=="select":
        scale.check()
        locked={}
        for p,n,s,api,job in work:
            d=ROOT/f"{n}-{s}"
            if not (d/"receipt.json").exists():
                select(p,n,s,api,job,d)
            rec=json.loads((d/"receipt.json").read_bytes())
            assert rec["forecast"]==b.digest(d/"forecasts.npz")
            locked[d.name]=b.digest(d/"receipt.json")
        (DEST/f"lock-{tag}.json").write_text(json.dumps(locked,indent=2),encoding="utf8")
    else:
        locked=json.loads((DEST/f"lock-{tag}.json").read_bytes())
        assert set(locked)=={f"{n}-{s}" for _,n,s,_,_ in work}
        rows=[]
        for p,n,s,api,job in work:
            d=ROOT/f"{n}-{s}"
            assert b.digest(d/"receipt.json")==locked[d.name]
            rec=json.loads((d/"receipt.json").read_bytes())
            assert b.digest(d/"forecasts.npz")==rec["forecast"]
            if not job:
                metrics=api.score(n,s,d/"forecasts.npz",rec["forecast"])
            else:
                dest,row=job
                path=dest/row["test_relpath"]
                assert b.digest(path)==row["test_file_sha256"]
                y=pd.read_csv(path,usecols=[row["target"]])[row["target"]].to_numpy()
                with np.load(d/"forecasts.npz") as forecasts:
                    metrics={m:b.metrics(y,forecasts[m],rec["task"],rec["scale"]) for m in forecasts}
            for m,item in rec["records"].items():
                rows.append(dict(panel=p,dataset=n,seed=s,method=m,**metrics[m],**{k:v for k,v in item.items() if k!="cols"}))
        pd.DataFrame(rows).to_csv(DEST/f"scores-{tag}.csv",index=False)


if __name__=="__main__":
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<3)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        main()
