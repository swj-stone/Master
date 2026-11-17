"""Refit vetoes for deleting short blocks from an already feasible Gain anchor."""
from time import perf_counter

import numpy as np
from xgboost import DMatrix

from experiments.minpi_replication import direct_gain as native
from experiments.minpi_rework.boosted_additive import probability_parts
from experiments.vote_increment.adaptive.game import normalized


ARMS=("min","mean","gain","intact","anchor_only")


def fit_policy(x,y,task,idx,seed,arm):
    start=perf_counter()
    tr,v,va=(np.sort(idx[k]) for k in ("fit","vote","width"))
    dev=np.sort(np.r_[tr,v,va])
    p=x.shape[1]
    full=tuple(range(p))
    models={}
    scores={}
    def evaluate(cols):
        if cols not in scores:
            model=native.fit_native(x[np.ix_(tr,cols)],y[tr],task,seed,"cpu")
            pred=native.predict_native(model,x[np.ix_(va,cols)],task)
            models[cols]=model
            scores[cols]=native.validation_utility(y[va],pred,task)
        return scores[cols]
    original=evaluate(full)
    base_model=models[full]
    gainmap=base_model.get_booster().get_score(importance_type="gain")
    gain=np.array([gainmap.get(f"f{j}",0.) for j in range(p)])
    first=np.where(normalized(gain)>=1.)[0]
    candidate=tuple(first.tolist()) if len(first) else (int(gain.argmax()),)
    evaluate(candidate)
    floor=max(scores.values())-.005
    anchor=candidate if scores[candidate]>=floor-1e-14 else full
    active=np.asarray(anchor)
    a=len(active)
    selected=anchor
    model=models[anchor]
    local=model.get_booster().get_score(importance_type="gain")
    native_gain=np.array([local.get(f"f{j}",0.) for j in range(a)])
    tic=perf_counter()
    delta=None
    if a>1 and arm not in {"gain","anchor_only"}:
        # Use paired votes over all voting observations, except a prespecified cap.
        sample=v
        if len(sample)>512:
            sample=np.sort(np.random.default_rng(seed).choice(sample,512,replace=False))
        dm=DMatrix(x[np.ix_(sample,active)],nthread=1)
        contrib=model.get_booster().predict(dm,pred_contribs=True,approx_contribs=True,strict_shape=True)
        raw=contrib[:,:,:-1].transpose(0,2,1).astype(float)
        bias=contrib[:,:,-1].astype(float)
        if task=="classification":
            parts,bias=probability_parts(raw,bias)
            target=np.eye(parts.shape[-1])[y[sample].astype(int)]-bias
        else:
            parts=raw
            target=y[sample,None]-bias
        n=len(sample)
        variance=max(1e-12,float(np.sum((target-target.mean(axis=0))**2)/n))
        net=parts.sum(axis=1)
        def penalty(indices):
            total=parts[:,indices,:].sum(axis=1)
            singleton=(2*np.sum(total*target)-np.sum(total**2))/n
            cross=2*np.sum(total*(net-total))/n
            return (singleton-np.array([1/6,.5,5/6])*cross)/variance
        risks=np.stack([penalty([j]) for j in range(a)],axis=1)
        if arm=="min":
            ranking=np.max(risks,axis=0)
            delta=lambda ix:float(np.max(penalty(ix)))
        elif arm=="mean":
            ranking=np.mean(risks,axis=0)
            delta=lambda ix:float(np.mean(penalty(ix)))
        elif arm=="intact":
            # Use the native Gain order and allow the same gate with only the dense layer.
            ranking=native_gain
            delta=lambda ix:float(penalty(ix)[-1])
        else:
            raise ValueError(arm)
    else:
        ranking=native_gain
    order=np.lexsort((-active,native_gain,ranking))
    vote_s=perf_counter()-tic
    probes=[]
    if a>1 and arm!="anchor_only":
        # Candidate blocks span a quarter down to one feature. All algorithms
        # get at most four distinct shortlisted supports and stop at first pass.
        sizes=sorted(set(max(1,min(a-1,int(np.ceil(a*f)))) for f in [.25,.125,.0625,1/a]),reverse=True)
        for g in sizes:
            ix=order[:g]
            cols=tuple(np.delete(active,ix).tolist())
            expected=delta(ix) if delta else None
            probes.append(dict(size=g,predicted_loss=expected))
            if expected is not None and expected>.005:
                continue
            if evaluate(cols)>=floor-1e-14:
                selected=cols
                break
    final=native.fit_native(x[np.ix_(dev,selected)],y[dev],task,seed,"cpu")
    pred=native.predict_native(final,x[np.ix_(idx["test"],selected)],task)
    elapsed=perf_counter()-start
    return dict(cols=list(selected),k=len(selected),fits=len(scores)+1,seconds=elapsed,
                vote_s=vote_s,floor=floor,validation=scores[selected],anchor=list(anchor),
                order=active[order].tolist(),screen=probes,trace=[(list(c),u) for c,u in scores.items()]),pred
