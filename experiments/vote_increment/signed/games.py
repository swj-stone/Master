"""Signed out-of-sample improvements assigned to splits in random feature coalitions."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.special import ndtr


def sample_rows(y, task, n, rng):
    if len(y)<=n:
        return np.arange(len(y))
    if task=="classification":
        groups=np.unique(y)
        q=n//len(groups)
        # An AUC-oriented sketch gives rare classes a representative vote.
        picks=[rng.choice(np.flatnonzero(y==g),min(q,np.count_nonzero(y==g)),replace=False) for g in groups]
        return np.sort(np.concatenate(picks))
    return np.sort(rng.choice(len(y),n,replace=False))


def responses(y, task, classes, center, scale):
    return np.eye(classes)[y.astype(int)] if task=="classification" else ((y-center)/scale)[:,None]


def random_subspace(x,y,z,v,task,seed):
    rng=np.random.default_rng(seed)
    fit=sample_rows(y,task,3072,rng)
    vote=sample_rows(v,task,1024,rng)
    train=np.asarray(x[fit],np.float32)
    eval_x=np.asarray(z[vote],np.float32)
    median=np.nan_to_num(np.nanmedian(train,axis=0))
    train=np.where(np.isfinite(train),train,median)
    eval_x=np.where(np.isfinite(eval_x),eval_x,median)
    classes=len(np.unique(y))
    center,scale=float(np.mean(y[fit])),max(float(np.std(y[fit])),1e-12)
    targets=responses(y[fit],task,classes,center,scale)
    observed=responses(v[vote],task,classes,center,scale)
    p=train.shape[1]
    sizes=sorted({max(1,round(np.sqrt(p))), max(1,p//2),p})
    scores=[]
    aux=0
    for k in sizes:
        total=np.zeros(p)
        opportunities=np.zeros(p)
        for repeat in range(4):
            perm=rng.permutation(p)
            for offset in range(0,p,k):
                cols=np.take(perm, np.arange(offset,offset+k),mode="wrap")
                tree=DecisionTreeRegressor(max_depth=4,min_samples_leaf=8,random_state=seed+repeat)
                tree.fit(train[:,cols],targets)
                aux+=1
                opportunities[cols]+=1
                params=tree.tree_
                node=np.zeros(len(observed),dtype=int)
                for _ in range(4):
                    use=np.flatnonzero(params.children_left[node]>=0)
                    if not len(use):
                        break
                    parent=node[use]
                    j=params.feature[parent]
                    child=np.where(eval_x[use,cols[j]]<=params.threshold[parent],params.children_left[parent],params.children_right[parent])
                    old_err=np.square(observed[use]-params.value[parent,:,0]).sum(axis=1)
                    new_err=np.square(observed[use]-params.value[child,:,0]).sum(axis=1)
                    total+=np.bincount(cols[j],weights=old_err-new_err,minlength=p)
                    node[use]=child
        score=total/opportunities
        scores.append(score/max(np.abs(score).sum(),1e-12))
    return np.asarray(scores),aux


def pairwise_ballots(x,y,z,v,task,seed):
    """Vote on feature-space comparisons of two label-ordered neighbors.

    For a query and two landmarks each feature supplies d_j in [-1,1].
    A coalition wins if its sum is positive. Empty/tied contests get half credit.
    At intermediate sizes a finite-population Gaussian estimates contest margins.
    """
    rng=np.random.default_rng(seed)
    fit=sample_rows(y,task,2048,rng)
    query=sample_rows(v,task,384,rng)
    # Rank features using only fitting rows; absent observations use the middle rank.
    def quantile_scale(a,anchors):
        scaled=np.empty_like(a,dtype=float)
        for j in range(a.shape[1]):
            col=anchors[:,j]
            known=np.sort(col[np.isfinite(col)])
            if len(known):
                scaled[:,j]=(np.searchsorted(known,a[:,j],"left")+np.searchsorted(known,a[:,j],"right"))/(2*len(known))
                scaled[~np.isfinite(a[:,j]),j]=.5
            else:
                scaled[:,j]=.5
        return scaled
    train=quantile_scale(x[fit],x[fit])
    evaluate=quantile_scale(z[query],x[fit])
    frame=[]
    for row,label in zip(evaluate,v[query]):
        delta=np.abs(train-row)
        if task=="classification":
            near=np.flatnonzero(y[fit]==label)
            far=np.flatnonzero(y[fit]!=label)
        else:
            by_response=np.argsort(np.abs(y[fit]-label),kind="stable")
            near=by_response[:max(1,len(by_response)//8)]
            far=by_response[len(by_response)//2:]
        if not len(near) or not len(far):
            continue
        # Uniform and nearby pairs are pooled, so easy points and local reversals contribute.
        for pool in ("uniform","local"):
            left,right=near,far
            if pool=="local":
                d=delta.sum(axis=1)
                left=near[np.argsort(d[near],kind="stable")[:8]]
                right=far[np.argsort(d[far],kind="stable")[:8]]
            a=rng.choice(left,4)
            b=rng.choice(right,4)
            frame.extend(delta[b]-delta[a])
    d=np.asarray(frame)
    p=d.shape[1]
    n=p-1
    removed=d.sum(axis=1,keepdims=True)-d
    squares=np.square(d).sum(axis=1,keepdims=True)-np.square(d)
    avg=removed/max(1,n)
    var=np.maximum(0,squares/max(1,n)-avg**2)
    ties=lambda a: (a>1e-12).astype(float)+.5*(np.abs(a)<=1e-12)
    votes=[]
    for k in sorted({0,n//2,n}):
        mu=k*avg
        spread=np.sqrt(k*(n-k)*var/max(1,n-1))
        stochastic=spread>1e-10
        before=ties(mu)
        after=ties(mu+d)
        before[stochastic]=ndtr(mu[stochastic]/spread[stochastic])
        after[stochastic]=ndtr((mu+d)[stochastic]/spread[stochastic])
        result=(after-before).mean(axis=0)
        votes.append(result/max(float(np.abs(result).sum()),1e-12))
    return np.asarray(votes),0
