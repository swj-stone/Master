"""Block-omission votes for a common, unaltered native Gain prefix."""
import numpy as np
from xgboost import DMatrix


def block_votes(model,x,y,task,order,seed):
    """Signed utility of restoring the omitted block to random prefix coalitions."""
    rng=np.random.default_rng(seed)
    if len(x)>1024:
        if task=='classification':
            label=np.unique(y)
            ix=np.sort(np.concatenate([rng.choice(np.flatnonzero(y==c), min((y==c).sum(),1024//len(label)),replace=False) for c in label]))
        else:
            ix=np.sort(rng.choice(len(x),1024,replace=False))
        x,y=x[ix],y[ix]
    p=len(order)
    raw=model.get_booster().predict(DMatrix(x,nthread=1),pred_contribs=True,approx_contribs=True,strict_shape=True)
    if task=='classification':
        # A balanced pair's outcome estimates one-vs-rest class-margin concordance.
        classes=np.unique(y)
        labels=np.resize(rng.permutation(classes),4096)
        a=np.empty(len(labels),int)
        b=np.empty(len(labels),int)
        for c in classes:
            i=labels==c
            a[i]=rng.choice(np.flatnonzero(y==c),i.sum())
            b[i]=rng.choice(np.flatnonzero(y!=c),i.sum())
        target=labels if raw.shape[1]>1 else np.zeros_like(labels)
        parts=(raw[a,target,:-1]-raw[b,target,:-1])[:,order]
        if raw.shape[1]==1:
            parts=parts*np.where(labels==1,1.,-1.)[:,None]
        def utility(z):
            return ((z>1e-7)+.5*(np.abs(z)<=1e-7)).mean(axis=0)
    else:
        parts=raw[:,0,:-1][:,order]
        residual=y[:,None]-raw[:,0,-1:]
        scale=max(np.var(y),1e-14)
        def utility(z):
            return 1-np.mean((residual-z)**2,axis=0)/scale
    prefix=parts.cumsum(axis=1)
    block=prefix[:,-1:]-prefix
    votes=[]
    for q in (.125,.5,.875):
        ballot=np.zeros(p)
        for _ in range(8):
            # Permutation priorities give an exactly cardinality-conditioned
            # coalition within the first k columns at each k.
            priorities=rng.permutation(p)
            # Each prefix uses its own k-sized quantile of priorities.
            available=[]
            selected=np.zeros(p,bool)
            active=np.zeros(len(parts),dtype=float)
            accumulated=np.empty_like(prefix)
            for j in range(p):
                available.append(j)
                target_n=int(np.floor((j+1)*q+.5))
                chosen=np.zeros(p,bool)
                if target_n:
                    prior=priorities[:j+1]
                    chosen[np.argpartition(prior,target_n-1)[:target_n]]=True
                for col in np.flatnonzero(selected!=chosen):
                    active += parts[:,col] if chosen[col] else -parts[:,col]
                selected=chosen
                accumulated[:,j]=active
            ballot+=utility(accumulated+block)-utility(accumulated)
        votes.append(ballot/8)
    return np.asarray(votes)


def nominate(loss,threshold,low,hi):
    # Reading toward the full set excludes isolated, nonmonotone dips.
    envelope=np.maximum.accumulate(loss[::-1])[::-1]
    widths=np.flatnonzero(envelope<=threshold+1e-14)+1
    return int(np.clip(widths[0] if len(widths) else len(loss),low,hi))
