"""Size-specific held-row evidence for a shared feature quota."""
from fractions import Fraction
from math import comb
import json

import numpy as np
from xgboost import DMatrix

from experiments.minpi_rework.boosted_additive import probability_parts


def normalized(z):
    z=np.maximum(z,0.)
    mass=z.sum(axis=-1,keepdims=True)
    return np.divide(z*z.shape[-1],mass,out=np.zeros_like(z),where=mass>0)


def margin_votes(model,x,y,task,seed):
    # Stratify bootstrap-free sampling by class when subsampling prediction rows.
    if len(x)>512:
        rng=np.random.default_rng(seed)
        if task=="regression":
            ix=np.sort(rng.choice(len(x),512,replace=False))
        else:
            from minppi_mofs.piv1 import _sample_rows
            metric="roc_auc" if len(np.unique(y))==2 else "ovr_auc"
            ix=_sample_rows(y,metric,rng)
        x,y=x[ix],y[ix]
    dm=DMatrix(x,nthread=1)
    boost=model.get_booster()
    contrib=boost.predict(dm,pred_contribs=True,approx_contribs=True,strict_shape=True)
    margins=boost.predict(dm,output_margin=True,strict_shape=True)
    np.testing.assert_allclose(contrib.sum(axis=2),margins,rtol=1e-5,atol=4e-5)
    raw=contrib[:,:,:-1].astype(float).transpose(0,2,1)
    bias=contrib[:,:,-1].astype(float)
    if task=="classification":
        parts,bias=probability_parts(raw,bias)
        residual=np.eye(parts.shape[-1])[y.astype(int)]-bias
    else:
        parts=raw
        residual=y[:,None]-bias
    energy=np.einsum("npc,npc->p",parts,parts)
    singleton=2*np.einsum("npc,nc->p",parts,residual)-energy
    dependence=2*(np.einsum("npc,nc->p",parts,parts.sum(axis=1))-energy)
    return normalized(singleton[None,:]-np.array([0.,.5,1.])[:,None]*dependence)


def path_votes(model,p):
    """Split gain is reachable only when the whole preceding path is available.

    For the weighted unanimity game on distinct path variables, a member is
    pivotal when all d−1 other members appear in a uniform coalition of size k.
    """
    trees=json.loads(model.get_booster().save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
    mass=np.zeros((5,p))
    for tree in trees:
        groups=[]
        for i,left in enumerate(tree["left_children"]):
            prior=() if i==0 else groups[tree["parents"][i]]
            if left<0:
                groups.append(prior)
                continue
            j=tree["split_indices"][i]
            group=prior if j in prior else prior+(j,)
            groups.append(group)
            mass[len(group),list(group)]+=max(0.,tree["loss_changes"][i])
    tiers=(0,(p-1)//2,p-1)
    prob=np.zeros((3,len(mass)))
    for s,k in enumerate(tiers):
        for d in range(1,min(len(mass),p+1)):
            prob[s,d]=float(Fraction(comb(k,d-1),comb(p-1,d-1)))
    return normalized(prob@mass)


def pair_votes(model,x,y,task,seed):
    """Exact pivotal credits for the largest observed supporting feature in a labeled pair."""
    rng=np.random.default_rng(seed)
    max_pairs=1024
    if task=="classification":
        classes=np.unique(y)
        target=np.resize(rng.permutation(classes),max_pairs)
        plus=np.empty(max_pairs,int)
        minus=np.empty_like(plus)
        for c in classes:
            rows=target==c
            plus[rows]=rng.choice(np.flatnonzero(y==c),rows.sum())
            minus[rows]=rng.choice(np.flatnonzero(y!=c),rows.sum())
    else:
        lhs,rhs=rng.integers(0,len(y),size=(2,max_pairs))
        plus=np.where(y[lhs]>=y[rhs],lhs,rhs)
        minus=np.where(y[lhs]>=y[rhs],rhs,lhs)
    keep=y[plus]!=y[minus]
    plus,minus=plus[keep],minus[keep]
    rows,inverse=np.unique(np.r_[plus,minus],return_inverse=True)
    ix,iy=np.split(inverse,2)
    dm=DMatrix(x[rows],nthread=1)
    contrib=model.get_booster().predict(dm,pred_contribs=True,approx_contribs=True,strict_shape=True)
    parts=contrib[:,:,:-1].transpose(0,2,1).astype(float)
    if task=="classification":
        parts,_=probability_parts(parts,contrib[:,:,-1].astype(float))
        c=target[keep]
        evidence=parts[ix,:,c]-parts[iy,:,c]
    else:
        evidence=(parts[ix,:,0]-parts[iy,:,0])*np.abs(y[plus]-y[minus])[:,None]
    return facility_votes(np.maximum(evidence,0.))


def facility_votes(values):
    # A pair with nonnegative supports a_j contributes max_{j in S} a_j. Write this
    # max as a weighted sum of OR games for the sets of its r strongest supporters.
    n,p=values.shape
    ranks=np.argsort(-values,axis=1,kind="stable")
    sorted_value=np.take_along_axis(values,ranks,axis=1)
    levels=sorted_value-np.column_stack((sorted_value[:,1:],np.zeros(n)))
    tiers=(0,(p-1)//2,p-1)
    result=[]
    for k in tiers:
        discount=np.array([comb(p-r,k)/comb(p-1,k) for r in range(1,p+1)])
        credits=np.cumsum((levels*discount)[:,::-1],axis=1)[:,::-1]
        mass=np.bincount(ranks.ravel(),weights=credits.ravel(),minlength=p)
        result.append(mass)
    return normalized(np.stack(result))
