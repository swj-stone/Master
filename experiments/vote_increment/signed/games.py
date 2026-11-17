"""Signed out-of-sample improvements assigned to splits in random feature coalitions."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor


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
