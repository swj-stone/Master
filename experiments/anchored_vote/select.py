"""Refit-guarded compression inside a measured native Gain anchor."""
from time import perf_counter
import json

import numpy as np

from experiments.minpi_replication.direct_gain import fit_native, gain_order, predict_native, validation_utility
from experiments.split_vote.path import select_width
from minppi_mofs.split_vote import pivotal_order, split_voters


def select(x, y, task, idx, seed, method, device="cuda"):
    start=perf_counter()
    fit=np.sort(idx["fit"])
    valid=np.sort(np.r_[idx["vote"],idx["width"]])
    dev=np.sort(np.r_[fit,valid])
    train=x[fit]
    val=x[valid]
    full=tuple(range(x.shape[1]))
    p=len(full)
    model=fit_native(train,y[fit],task,seed,device)
    booster=model.get_booster()
    actual=json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    if not actual.startswith(device):
        raise RuntimeError(f"Expected {device}, got {actual}")
    ordinary=np.array(gain_order(model,p))
    scores={full:validation_utility(y[valid],predict_native(model,val,task),task)}
    def observe(cols):
        key=tuple(sorted(map(int,cols)))
        if key not in scores:
            other=fit_native(train[:,key],y[fit],task,seed,device)
            scores[key]=validation_utility(y[valid],predict_native(other,val[:,key],task),task)
        return scores[key]
    totals=booster.get_score(importance_type="total_gain")
    total=np.array([totals.get(f"f{j}",0.) for j in full])
    trace={}
    fixed={}
    receipt={}
    if method=="exhaustive":
        for k in range(1,p+1):
            trace[k]=observe(ordinary[:k])
        k=next(k for k in trace if trace[k]>=max(trace.values())-1e-14)
        near=next(k for k in trace if trace[k]>=max(trace.values())-.005-1e-14)
        cols=ordinary[:k]
        receipt["near_support"]=sorted(ordinary[:near].tolist())
    elif method=="gain_grid":
        k,trace=select_width(ordinary,observe)
        cols=ordinary[:k]
    elif method in {"min","mean","gain"}:
        mass=np.cumsum(total[ordinary])
        if mass[-1]==0:
            mass=np.arange(1,p+1)
        ks=sorted({p, *(int(np.searchsorted(mass,mass[-1]*q))+1 for q in (.5,.875))})
        for k in ks:
            fixed[k]=observe(ordinary[:k])
        def narrow():
            return min(k for k,u in fixed.items() if u>=max(fixed.values())-.005-1e-14)
        hi=narrow()
        lo=max((k for k in fixed if k<hi),default=0)
        if hi-lo>1:
            mid=(lo+hi)//2
            fixed[mid]=observe(ordinary[:mid])
        anchor_width=narrow()
        anchor=ordinary[:anchor_width]
        floor=max(fixed.values())-.005
        cols=anchor
        if method=="gain" or anchor_width==1:
            ranked=anchor
        else:
            # Restrict the voting game itself to the columns available in the anchor.
            voters,weights=split_voters(train[:,anchor],y[fit],total[anchor],task=="classification",seed)
            gains=booster.get_score(importance_type="gain")
            tie=np.array([gains.get(f"f{j}",0.) for j in anchor])
            ranked=anchor[pivotal_order(voters,weights,tie,rule="min" if method=="min" else "pooled")]
        lo,hi=0,anchor_width
        for _ in range(2):
            k=(lo+hi)//2
            if k<=lo:
                break
            u=observe(ranked[:k])
            trace[k]=u
            if u>=floor-1e-14:
                cols=ranked[:k]
                hi=k
            else:
                lo=k
        receipt.update(anchor=sorted(anchor.tolist()),floor=floor,anchor_curve=fixed)
    else:
        raise ValueError(method)
    cols=sorted(map(int,cols))
    selected_value=scores[tuple(cols)]
    selection=perf_counter()-start
    final=fit_native(x[np.ix_(dev,cols)],y[dev],task,seed,device)
    pred=predict_native(final,x[np.ix_(idx["test"],cols)],task)
    elapsed=perf_counter()-start
    receipt.update(k=len(cols),selected=cols,utility=selected_value,trace=trace,fits=len(scores)+1,
                   seconds=elapsed,selection_s=selection,unique_masks=len(scores))
    return receipt,pred
