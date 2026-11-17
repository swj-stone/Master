"""Conditional retirement with one surviving substitute per feature."""
import numpy as np
from scipy.stats import rankdata


def redundancy(x,seed):
    rng=np.random.default_rng(seed)
    ix=rng.choice(len(x),min(1024,len(x)),replace=False)
    rows=x[ix].astype(float)
    rows=np.where(np.isfinite(rows),rows,np.nan_to_num(np.nanmedian(rows,axis=0)))
    basis=rankdata(rows,axis=0)
    basis-=basis.mean(axis=0)
    basis/=np.maximum(np.linalg.norm(basis,axis=0),1e-12)
    weights=np.square(basis.T@basis)
    np.fill_diagonal(weights,0.)
    return weights


def retire(bank,gain,weights,rule):
    """A negative net witness is recomputed after every retirement."""
    p=len(gain)
    active=np.arange(p)
    removed=[]
    while len(active)>1:
        values=bank[:,active]
        transfer=weights[np.ix_(active,active)] * np.maximum(values[:,None,:],0.)
        net=values-transfer.max(axis=-1)
        score=net.min(axis=0) if rule=="min" else net.mean(axis=0)
        lowest=np.lexsort((active,gain[active],score))[0]
        if score[lowest]>=-1e-14:
            break
        removed.append(int(active[lowest]))
        active=np.delete(active,lowest)
    order=active[np.lexsort((-active,-gain[active]))]
    return np.r_[order,np.asarray(removed[::-1],dtype=int)],len(active)


def candidate_widths(order,frontier,gain):
    p=len(order)
    live=np.cumsum(gain[order[:frontier]])
    if live[-1]==0:
        live=np.arange(1,frontier+1)
    # Unaided Gain has frontier p and therefore receives the ordinary gain mass controls.
    ks={1+int(np.searchsorted(live,q*live[-1])) for q in (.5,.85)}
    if frontier<p:
        ks.add(frontier)
    else:
        ks.add(1+int(np.searchsorted(live,.95*live[-1])))
    return sorted(ks|{p})
