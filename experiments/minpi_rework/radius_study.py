"""Time complete pipelines for exact sparse and dense fixed feature budgets."""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework.exchange_benchmark import load_dev
from minppi_mofs.additive_minpi import fit_additive,quadratic
from minppi_mofs.exchange_bnb import bnb_best
from minppi_mofs.exchange_minpi import integer_game,screen,value
from minppi_mofs.radius_minpi import incumbent,radius_best


METHODS=["radius","free","tangent","bnb","unrestricted","width","greedy","energy","gain","full"]


def solve_one(gram,d,k,method):
    if method.startswith("sector_"):
        from minppi_mofs.sector_minpi import sector_best
        return sector_best(gram,d,k,method.removeprefix("sector_"))
    if method in ("radius","tangent","free","bnb"):
        return radius_best(gram,d,k,method,quantized_psd=True)
    if method=="greedy":
        return np.flatnonzero(incumbent(gram,d,k)),{}
    state=screen(gram,d,k,method)
    keep,free=state.keep,state.free
    slots=k-len(keep)
    if slots==0 or slots==len(free):
        return (keep if slots==0 else np.r_[keep,free]),dict(free=len(free),nodes=0)
    local,stats=bnb_best(gram[np.ix_(free,free)],d[free]-2*gram[np.ix_(free,keep)].sum(axis=1),
                         slots,edges=state.order)
    return np.sort(np.r_[keep,free[local]]),dict(free=len(free),nodes=stats["nodes"])


def run_one(x,y,task,idx,seed,method,kind="spline"):
    start=perf_counter()
    pool=np.sort(np.r_[idx["fit"],idx["vote"]])
    model=fit_additive(x[pool],y[pool],task,kind)
    p=x.shape[1]
    if method=="gain":
        order,_=legacy.gain_rank("gain",x,y,task,{**idx,"fit":pool},seed)
    elif method!="full":
        parts=model.parts(x[pool])-model.center
        if method=="energy":
            order=np.lexsort((np.arange(p),-np.einsum("npc,npc->p",parts,parts)))
        else:
            g,d=quadratic(parts,parts.sum(axis=1))
            gram,d,scale=integer_game(g,d)
    setup=perf_counter()-start
    small=min(5,p//2)
    records,forecasts={},{}
    for budget,k in [("sparse",small),("dense",p-small)]:
        search_start=perf_counter()
        if method=="full":
            cols=np.arange(p)
            meta={}
        elif method in ("gain","energy"):
            cols=np.sort(order[:k])
            meta={}
        else:
            cols,meta=solve_one(gram,d,k,method)
            meta["utility_integer"]=int(value(gram,d,cols))
            meta["utility"]=meta["utility_integer"]/scale
        elapsed=perf_counter()-search_start
        t=perf_counter()
        pred=model.restrict(cols).predict(x[np.ix_(idx["test"],cols)])
        records[budget]=dict(method=method,p=p,k=len(cols),support=cols.tolist(),setup_s=setup,
                             search_s=elapsed,selection_s=setup+elapsed,
                             deployment_s=setup+elapsed+perf_counter()-t,**meta)
        forecasts[budget]=pred if task=="classification" else pred.ravel()
    return records,forecasts


def warm():
    from experiments.minpi_rework.budget import warm_budget
    from minppi_mofs.radius_minpi import _radius_bounds
    warm_budget()
    g=np.eye(7)
    d=np.arange(7,dtype=float)
    bnb_best(g,d,3)
    _radius_bounds(g,d,np.arange(3),np.arange(3,7),True)
    _radius_bounds(g,d,np.arange(3),np.arange(3,7),False)
    from minppi_mofs.sector_minpi import sector_best
    sector_best(g,d,3)


def pilot(args):
    out=b.ROOT/"results/minpi_rework"/args.run
    out.mkdir(parents=True,exist_ok=False)
    warm()
    jobs=legacy.jobs() if any(n in legacy.CONFIG["datasets"] for n in args.names) else {}
    rows=[]
    for name in args.names:
        for seed in args.seeds:
            start=perf_counter()
            x,y,task,idx,truth,mask=load_dev(name,seed,jobs)
            scale=max(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]]),1e-12)
            receipts={}
            for method in np.random.default_rng(seed).permutation(args.methods):
                info,preds=run_one(x,y,task,idx,seed,method,args.kind)
                receipts[method]=info
                for budget,rec in info.items():
                    m=b.metrics(truth[mask],preds[budget][mask],task,scale)
                    rows.append(dict(dataset=name,seed=seed,task=task,budget=budget,**rec,**m))
            for budget in ("sparse","dense"):
                values={r[budget]["utility_integer"] for meth,r in receipts.items()
                        if meth in METHODS[:6] or meth.startswith("sector_")}
                assert len(values)<=1,(name,seed,budget,values)
            (out/f"{name}-{seed}.json").write_text(json.dumps(receipts),encoding="utf8")
            pd.DataFrame(rows).to_csv(out/"scores.csv",index=False)
            print(f"{name}/{seed} {perf_counter()-start:.2f}s",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--names",nargs="+",default=["musk1","biodeg","landsat","wall","glass","yacht","aquatic","power","raisin","wifi","slump","estate","hardware","fires"])
    cli.add_argument("--seeds",type=int,nargs="+",default=[42])
    cli.add_argument("--methods",nargs="+",default=METHODS)
    cli.add_argument("--kind",default="spline")
    cli.add_argument("--run",required=True)
    with threadpool_limits(1):
        pilot(cli.parse_args())
