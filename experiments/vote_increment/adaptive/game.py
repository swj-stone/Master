"""Size-specific held-row evidence for a shared feature quota."""
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
