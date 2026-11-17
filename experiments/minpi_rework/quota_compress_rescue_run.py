"""Bounded repair from three starts on previously opened data; separate forecast and score steps."""
import argparse
import ctypes
import json
import os
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import legacy_gain as legacy,benchmark as b
from experiments.minpi_rework import validation_panel,validation_wide
from experiments.minpi_rework import quota_compress_tree as trees,quota_compress_tree_cost as cost
from experiments.minpi_rework import quota_compress_tree_native_exact as exact
from experiments.minpi_rework import quota_compress_rescue as rescue,quota_compress_rescue_check as check
from minppi_mofs.modeling import make_model


ROOT=b.ROOT/"results/minpi_rework/quota-rescue"
DEST=b.ROOT/"thesis-paper/evidence/minpi_rework/quota-rescue"


def jobs(all_seeds):
    old=legacy.jobs()
    for seed in legacy.CONFIG["seeds"] if all_seeds else (42,):
        for name in legacy.CONFIG["datasets"]:
            yield "historical",name,seed,None,old[name,seed]
    for api in (validation_panel,validation_wide):
        for seed in range(42,47) if all_seeds else (42,):
            for name in api.SOURCES:
                yield "core" if api is validation_panel else "wide",name,seed,api,None


def starting(game,ranking,method):
    if method=="gain":
        return cost.compiled(game,ranking)[0]
    if method=="ablation":
        d=(game.a@game.incidence).toarray()
        loss=np.einsum("ij,ij->j",d,d)
        order=np.lexsort((-np.arange(len(loss)),-loss))
        return cost.compiled(game,order)[0]
    inc=game.incidence
    by=inc.tocsc()
    a=game.a
    path,_=exact.direct(len(game.full),a.indptr,a.indices,a.data,inc.indptr,inc.indices,by.indptr,by.indices,
        ranking,.2*game.norm+1e-12*max(1.,game.norm),True)
    return np.setdiff1d(np.arange(game.p),path)


def select(panel,name,seed,api,job,directory):
    if job:
        x,y,task,idx=legacy.read_split(*job,seed)
    else:
        x,y,task,idx=api.split(name,seed)
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    val=x[idx["width"]]
    problem=("binary" if len(np.unique(y[pool]))==2 else "multiclass") if task=="classification" else task
    t=perf_counter()
    model=make_model(task=problem,seed=seed,backend=legacy.CONFIG["backend"],model_params=legacy.CONFIG["model_params"])
    model.fit(x[pool],y[pool])
    f=trees.extract(model,val[:1])
    g=f.cache(val)
    gain=model.get_booster().get_score(importance_type="gain")
    order=np.array(sorted(range(x.shape[1]),key=lambda j:(gain.get(f"f{j}",0.),j),reverse=True))
    shared_s=perf_counter()-t
    rng=np.random.default_rng(seed)
    records,forecasts={},{}
    for init in rng.permutation(("exact","gain","ablation")):
        t=perf_counter()
        start=starting(g,order,init)
        start_s=perf_counter()-t
        agreed=None
        for mode in rng.permutation(("start","one","pair","votes")):
            t=perf_counter()
            if mode=="start":
                cols=start
                moves=np.empty((0,3),int)
                counts=np.zeros(5,int)
                stop=-1
            else:
                cols,moves,counts,stop=rescue.search(g,start,pair=mode!="one",ballot=mode=="votes")
            search_s=perf_counter()-t
            if mode in ("pair","votes"):
                if agreed is None:
                    agreed=(cols,moves)
                else:
                    np.testing.assert_array_equal(cols,agreed[0])
                    np.testing.assert_array_equal(moves,agreed[1])
            risk=check.physical(g,cols)
            assert risk<=(.2*g.norm+1e-12*max(1.,g.norm))**2
            m=f"{init}_{mode}"
            t=perf_counter()
            forecasts[m]=trees.as_scores(f.predict(x[np.ix_(idx["test"],cols)],cols),task)
            pred_s=perf_counter()-t
            records[m]=dict(k=len(cols),p=g.p,cols=cols.tolist(),moves=moves.tolist(),start_s=start_s,
                shared_s=shared_s,search_s=search_s,pred_s=pred_s,pipeline_s=shared_s+start_s+search_s+pred_s,
                rho=np.sqrt(risk)/max(g.norm,1e-100),vetoes=int(counts[0]),checks=int(counts[1]),
                pair_checks=int(counts[2]),swaps=int(counts[3]),repairs=int(counts[4]),stop=stop)
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/"forecasts.npz",**forecasts)
    receipt=dict(panel=panel,task=task,scale=float(max(np.std(y[pool]),1e-12)),records=records,
        forecast=b.digest(directory/"forecasts.npz"),seed=seed)
    (directory/"receipt.json").write_text(json.dumps(receipt),encoding="utf8")
    print(name,seed,"widths (start -> repair)",[(records[i+"_start"]["k"],records[i+"_pair"]["k"]) for i in ("gain","ablation","exact")],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["select","score"])
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    work=list(jobs(args.all))
    tag="all" if args.all else "pilot"
    DEST.mkdir(parents=True,exist_ok=True)
    locked={}
    t=perf_counter()
    if args.action=="select":
        check.check()
        for i,(panel,name,seed,api,job) in enumerate(work):
            d=ROOT/f"{name}-{seed}"
            if not (d/"receipt.json").exists():
                select(panel,name,seed,api,job,d)
            receipt=json.loads((d/"receipt.json").read_bytes())
            assert receipt["forecast"]==b.digest(d/"forecasts.npz")
            locked[d.name]=b.digest(d/"receipt.json")
            print(f"[{i+1}/{len(work)}] {perf_counter()-t:.1f}s elapsed",flush=True)
        (DEST/f"lock-{tag}.json").write_text(json.dumps(locked,indent=2),encoding="utf8")
        return
    locked=json.loads((DEST/f"lock-{tag}.json").read_bytes())
    assert set(locked)=={f"{n}-{s}" for _,n,s,_,_ in work}
    rows=[]
    for panel,name,seed,api,job in work:
        d=ROOT/f"{name}-{seed}"
        assert b.digest(d/"receipt.json")==locked[d.name]
        receipt=json.loads((d/"receipt.json").read_bytes())
        assert receipt["forecast"]==b.digest(d/"forecasts.npz")
        if job:
            dataset,row=job
            path=dataset/row["test_relpath"]
            assert b.digest(path)==row["test_file_sha256"]
            y=pd.read_csv(path,usecols=[row["target"]])[row["target"]].to_numpy()
            with np.load(d/"forecasts.npz") as pred:
                metrics={m:b.metrics(y,pred[m],receipt["task"],receipt["scale"]) for m in pred}
        else:
            metrics=api.score(name,seed,d/"forecasts.npz",receipt["forecast"])
        for m,rec in receipt["records"].items():
            rows.append(dict(panel=panel,dataset=name,seed=seed,method=m,**metrics[m],
                **{k:v for k,v in rec.items() if k not in ("cols","moves")},accepted=len(rec["moves"])))
    pd.DataFrame(rows).to_csv(DEST/f"scores-{tag}.csv",index=False)


if __name__=="__main__":
    if os.name=="nt":
        dll=ctypes.WinDLL("kernel32",use_last_error=True)
        if not dll.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<7)):
            raise ctypes.WinError(ctypes.get_last_error())
    with threadpool_limits(1):
        main()
