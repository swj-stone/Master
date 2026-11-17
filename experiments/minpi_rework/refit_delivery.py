"""Validate and deploy separately refitted spline models on the nominated raw columns."""
import argparse
import json
from math import ceil, sqrt
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, legacy_gain as legacy
from experiments.minpi_rework.budget import warm_budget
from experiments.minpi_rework.exchange_benchmark import greedy_exchange, load_dev, nested_search
from minppi_mofs.additive_minpi import fit_additive
from minppi_mofs.exchange_minpi import integer_game, solve, value


def delta_stats(change):
    """Paired mean observation-level loss difference and its empirical standard error."""
    n=len(change)
    return float(np.mean(change)), float(np.std(change,ddof=1)/sqrt(n)) if n>1 else float("inf")


def paired_guard(y,anchor,reference,candidate,task,scale):
    eps=8*np.finfo(float).eps
    if task=="classification":
        loss=(anchor.argmax(axis=1)==y).astype(float)-(candidate.argmax(axis=1)==y)
        avg,se=delta_stats(loss)
        auc=b.metrics(y,candidate,task,scale)["auc"]
        return bool(avg+se<=.01+eps and auc>=reference["auc"]-.01-eps),dict(
            acc_loss=avg,acc_se=se,auc=auc,auc_loss=reference["auc"]-auc)
    error=(candidate-y)/scale
    full_error=(anchor-y)/scale
    mae,mae_se=delta_stats(abs(error)-abs(full_error))
    mse,mse_se=delta_stats(error*error-full_error*full_error)
    return bool(mae+mae_se<=.02+eps and mse+mse_se<=.02+eps),dict(
        scaled_mae_loss=mae,scaled_mae_se=mae_se,scaled_mse_loss=mse,scaled_mse_se=mse_se)


def game_stats(teacher,x,pool,need_gram):
    """Use additive shape covariance to reproduce the intact teacher on fitting data."""
    p=x.shape[1]
    gram=np.zeros((p,p)) if need_gram else None
    energy=np.zeros(p)
    for lo in range(0,len(pool),2048):
        part=teacher.parts(x[pool[lo:lo+2048]])-teacher.center
        if need_gram:
            mat=part.transpose(0,2,1).reshape(-1,p)
            gram+=mat.T@mat
        else:
            energy+=np.einsum("npc,npc->p",part,part)
    if gram is None:
        return energy/len(pool),None,None,None
    gram/=len(pool)
    energy=gram.diagonal().copy()
    gram,d,scale=integer_game(gram,2*gram.sum(axis=1)-energy)
    return energy,gram,d,scale


def refit_select(x,y,task,idx,seed,method,kind="spline",budget=5):
    """Return an adaptive/fixed proposal receipt and raw outer forecasts.

    The label-blind outer inputs are sliced to the deployed support. Both validation and
    delivery use the same model, trained on the pooled fitting and voting rows.
    """
    if method not in ("width","greedy","energy","gain","full","unrestricted","direct"):
        raise ValueError(method)
    start=perf_counter()
    fit,vote,width,test=(idx[k] for k in ("fit","vote","width","test"))
    pool=np.sort(np.r_[fit,vote])
    p=x.shape[1]
    ids=np.arange(p)
    fixed_width=min(budget,p)
    if budget<1 or p<1:
        raise ValueError("Positive feature width and budget required")
    teacher=fit_additive(x[pool],y[pool],task,kind)
    def output(model,where,cols):
        out=model.restrict(np.arange(len(cols))).predict(x[np.ix_(where,cols)])
        return out if task=="classification" else out.ravel()
    def refit(cols):
        return fit_additive(x[np.ix_(pool,cols)],y[pool],task,kind)
    fits=1
    trace=[]
    cache={}
    if method=="full":
        chosen=fixed=ids
        selected_model=fixed_model=teacher
    else:
        scale=max(float(np.std(y[pool])),1e-12)
        anchor=output(teacher,width,ids)
        reference=b.metrics(y[width],anchor,task,scale)
        if method=="gain":
            # Gain sees the same labeled pool as the fitted shape functions.
            order,_=legacy.gain_rank(method,x,y,task,{**idx,"fit":pool},seed)
        else:
            energy,gram,d,units=game_stats(teacher,x,pool,method!="energy")
            order=np.lexsort((ids,-energy))
        def nominate(k):
            if k==p: return ids,dict(search="full")
            details={}
            if method in ("energy","gain"):
                cols=np.sort(order[:k])
                details["search"]=method
            elif method=="greedy":
                cols=greedy_exchange(gram,d,order,k)
                details["search"]="global_one_exchange"
            elif k in (fixed_width,p-fixed_width):
                cols,details=solve(gram,d,k,mode=method,backend="enum")
                details["search"]="global"
            else:
                cols,details=nested_search(gram,d,order,k,method,window=24)
                details["search"]="energy_window"
            if method not in ("energy","gain"):
                details["utility"]=float(value(gram,d,cols)/units)
            return cols,details
        widths=sorted({1,fixed_width,max(1,p-fixed_width),p,*(ceil(p*f) for f in (.125,.25,.5,.75))})
        chosen=ids
        selected_model=teacher
        for k in widths:
            if k==p:
                trace.append(dict(k=p,pass_guard=True,search="full"))
                break
            cols,details=nominate(k)
            model=refit(cols)
            fits+=1
            passed,observed=paired_guard(y[width],anchor,reference,output(model,width,cols),task,scale)
            trace.append(dict(k=k,support=cols.tolist(),pass_guard=passed,**details,**observed))
            if k==fixed_width: cache[k]=cols,model
            if passed:
                chosen,selected_model=cols,model
                break
    selection_s=perf_counter()-start
    adaptive=output(selected_model,test,chosen)
    deployment_s=perf_counter()-start
    if method=="full" or fixed_width==p:
        fixed,fixed_model=ids,teacher
    elif fixed_width in cache:
        fixed,fixed_model=cache[fixed_width]
    else:
        # A smaller width already passed; this extra fit belongs only to the separate fixed arm.
        fixed,_=nominate(fixed_width)
        fixed_model=refit(fixed)
    frozen=adaptive if np.array_equal(chosen,fixed) else output(fixed_model,test,fixed)
    return dict(method=method,p=p,k=len(chosen),fixed_k=len(fixed),support=chosen.tolist(),fixed=fixed.tolist(),
                selection_s=selection_s,deployment_s=deployment_s,fixed_extra_s=perf_counter()-start-deployment_s,
                adaptive_receiver_fits=fits,trace=trace), dict(adaptive=adaptive,fixed=frozen)


def pilot(args):
    out=b.ROOT/"results/minpi_rework"/args.run
    out.mkdir(parents=True,exist_ok=False)
    warm_budget()
    manifest=legacy.jobs() if any(n in legacy.CONFIG["datasets"] for n in args.names) else {}
    rows=[]
    for name in args.names:
        for seed in args.seeds:
            clock=perf_counter()
            x,y,task,idx,truth,mask=load_dev(name,seed,manifest)
            scale=max(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]]),1e-12)
            forecasts={}
            records=[]
            for method in np.random.default_rng(seed).permutation(args.methods):
                rec,pred=refit_select(x,y,task,idx,seed,method)
                records.append(rec)
                for policy,array in pred.items(): forecasts[method+"_"+policy]=array
            (out/f"{name}-{seed}.json").write_text(json.dumps(records),encoding="utf8")
            np.savez_compressed(out/f"{name}-{seed}.npz",**forecasts)
            for rec in records:
                for policy in ("adaptive","fixed"):
                    pred=forecasts[rec["method"]+"_"+policy]
                    rows.append(dict(dataset=name,seed=seed,task=task,policy=policy,method=rec["method"],p=rec["p"],
                        k=rec["k" if policy=="adaptive" else "fixed_k"],selection_s=rec["selection_s"],
                        deployment_s=rec["deployment_s"],**b.metrics(truth[mask],pred[mask],task,scale)))
            pd.DataFrame(rows).to_csv(out/"scores.csv",index=False)
            print(f"{name}/{seed} {perf_counter()-clock:.2f}s",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--names",nargs="+",required=True)
    cli.add_argument("--seeds",type=int,nargs="+",default=[42])
    cli.add_argument("--methods",nargs="+",default=["width","greedy","energy","gain","full"])
    cli.add_argument("--run",required=True)
    with threadpool_limits(1):
        pilot(cli.parse_args())
