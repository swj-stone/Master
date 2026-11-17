"""Five-input additive deployment and guarded fallback on a common prediction model."""
import argparse
import json
from math import ceil
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, legacy_gain as legacy
from experiments.minpi_rework.exchange_benchmark import greedy_exchange,load_dev,warm
from minppi_mofs.additive_minpi import fit_additive,quadratic
from minppi_mofs.exchange_minpi import integer_game,solve,value,dominance


def budget_select(x,y,task,idx,seed,method,kind="spline",budget=5):
    start=perf_counter()
    fit,vote,width,test=(idx[key] for key in ("fit","vote","width","test"))
    pool=np.sort(np.r_[fit,vote])
    p=x.shape[1]
    receiver=fit_additive(x[pool],y[pool],task,kind)
    k=min(budget,p)
    full=np.arange(p)
    stats={}
    if method=="full":
        chosen=fallback=full
    else:
        if method=="gain":
            order,_=legacy.gain_rank("gain",x,y,task,idx,seed)
        else:
            parts=receiver.parts(x[pool])-receiver.center
            energy=np.einsum("nfc,nfc->f",parts,parts)
            order=np.lexsort((full,-energy))
        if method in ("energy","gain"):
            chosen=np.sort(order[:k])
        else:
            gram,d=quadratic(parts,parts.sum(axis=1))
            gram,d,scale=integer_game(gram,d)
            if method=="greedy":
                chosen=greedy_exchange(gram,d,order,k)
            else:
                chosen,stats=solve(gram,d,k,mode=method,backend="enum")
            stats["utility"]=value(gram,d,chosen)/scale
            stats["roundoff_regret"]=k*k/scale
        validation=receiver.parts(x[width])-receiver.center
        base=receiver.intercept+receiver.center.sum(axis=0)
        def predict(s):
            output=base+validation[:,s].sum(axis=1)
            return output if task=="classification" else output.ravel()
        anchor=predict(full)
        label_scale=max(np.std(y[pool]),1e-12)
        ref=b.metrics(y[width],anchor,task,label_scale)
        fallback=full
        # Independent labeled guard plus an empirical bound on departure from the teacher.
        for cols in [chosen,*[np.sort(order[:v]) for v in sorted({ceil(p*f) for f in (.125,.25,.5,.75)}) if k<v<p]]:
            output=predict(cols)
            if task=="classification":
                faithful=np.mean(anchor.argmax(axis=1)!=output.argmax(axis=1))<=.01
            else:
                faithful=np.mean(np.abs(anchor-output))<=.02*label_scale
            if faithful and b.eligible(b.metrics(y[width],output,task,label_scale),ref,task):
                fallback=cols
                break
    selected_s=perf_counter()-start
    def sparse(cols):
        pred=receiver.restrict(cols).predict(x[np.ix_(test,cols)])
        return pred if task=="classification" else pred.ravel()
    auto_pred=sparse(fallback)
    elapsed=perf_counter()-start
    fixed_pred=auto_pred if np.array_equal(chosen,fallback) else sparse(chosen)
    return dict(method=method,p=p,k=len(fallback),fixed_k=len(chosen),support=fallback.tolist(),fixed=chosen.tolist(),
                selection_s=selected_s,deployment_s=elapsed,**stats),dict(adaptive=auto_pred,fixed=fixed_pred)


def warm_budget():
    warm()
    g=np.diag(np.arange(4,dtype=float))
    d=np.arange(4,dtype=float)
    dominance(g,d,2)
    solve(g,d,2,mode="width",backend="enum",dispatch=False)
    from minppi_mofs.exchange_minpi import enumerate_best
    edges=np.zeros((4,4),bool)
    edges[0,3]=True
    enumerate_best(g,d,2,edges)


def pilot(args):
    out=b.ROOT/"results/minpi_rework"/args.run
    out.mkdir(parents=True,exist_ok=False)
    warm_budget()
    manifest=legacy.jobs() if any(n in legacy.CONFIG["datasets"] for n in args.names) else {}
    rows=[]
    for name in args.names:
        for seed in args.seeds:
            start=perf_counter()
            x,y,task,idx,truth,mask=load_dev(name,seed,manifest)
            scale=max(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]]),1e-12)
            records=[]
            for method in np.random.default_rng(seed).permutation(args.methods):
                rec,pred=budget_select(x,y,task,idx,seed,method,args.kind,args.budget)
                records.append(rec)
                for policy,scores in pred.items():
                    metric=b.metrics(truth[mask],scores[mask],task,scale)
                    rows.append(dict(dataset=name,seed=seed,task=task,policy=policy,method=method,p=rec["p"],
                                     k=rec["k" if policy=="adaptive" else "fixed_k"],deployment_s=rec["deployment_s"],
                                     selection_s=rec["selection_s"],**metric))
            (out/f"{name}-{seed}.json").write_text(json.dumps(records),encoding="utf8")
            frame=pd.DataFrame(rows)
            for policy,part in frame.groupby("policy"):
                part.to_csv(out/f"{policy}.csv",index=False)
            print(f"{name}/{seed} {perf_counter()-start:.2f}s",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--names",nargs="+",default=["glass","raisin","wifi","biodeg","landsat","musk1","wall","aquatic","power","yacht","slump","estate","hardware","fires"])
    cli.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    cli.add_argument("--methods",nargs="+",default=["width","unrestricted","direct","energy","greedy","full"])
    cli.add_argument("--kind",default="spline",choices=["linear","spline"])
    cli.add_argument("--budget",type=int,default=5)
    cli.add_argument("--run",required=True)
    with threadpool_limits(1):
        pilot(cli.parse_args())
