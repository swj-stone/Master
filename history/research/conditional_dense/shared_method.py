"""Share a Cholesky factor for nested contexts and their one-column substitutions."""
from math import ceil, isqrt

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def _nested(X,Y,order,sizes,penalty):
    """Exact optimal ridge gain for each candidate in each realized context."""
    anchor = order[:sizes[-1]+1]
    B = np.ascontiguousarray(X[:, anchor])
    diagonal = np.einsum('ij,ij->j',X,X)
    cross = B.T @ X
    gram = np.ascontiguousarray(cross[:,anchor])
    gram.flat[::len(anchor)+1] += penalty
    L = cholesky(gram, lower=True, check_finite=False)
    rhs = X.T @ Y
    projected = solve_triangular(L,cross,lower=True,check_finite=False)
    transformed_y = solve_triangular(L,rhs[anchor],lower=True,check_finite=False)
    inverse = solve_triangular(L,np.eye(len(anchor),dtype=X.dtype),lower=True,check_finite=False)
    votes=[]
    for size in sizes:
        A=projected[:size]
        residual_var = np.maximum(0.,diagonal-np.einsum('ij,ij->j',A,A))
        residual_rhs = rhs - A.T @ transformed_y[:size]
        v=(residual_rhs**2).sum(axis=1)/(penalty+residual_var)
        if size:
            # Change the prefix only for candidates that belong to it.
            beta=inverse[:size+1,:size+1].T @ transformed_y[:size+1]
            invdiag=(inverse[:size+1,:size]**2).sum(axis=0)
            members=anchor[:size]
            v[members]=(beta[:size]**2).sum(axis=1)/invdiag
            residual_var[members]=np.maximum(0.,1/invdiag-penalty)
        votes.append(v)
    return np.stack(votes),residual_var


def conditional_support(X,y,count=None,*,seed=1129):
    """Single random peer order, adaptive width estimated from residual lengths.

    Eight equal column blocks define a context with one eighth, or one
    quarter, of all peers, capped at sqrt(number of sketch rows)*4 peers per
    larger context to bound computation. Sample 768 training rows including
    at least one per class. Missing, temporal and grouped calibration models
    are out of scope; this selector does not internally tune on a holdout.
    """
    X,y=np.asarray(X),np.asarray(y)
    if X.ndim!=2 or min(X.shape)<1 or y.shape!=(len(X),):
        raise ValueError("Expected aligned dense training rows and labels")
    n,p=X.shape
    if count is not None and (isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer))
                              or not 1<=count<=p):
        raise ValueError("Invalid support width")
    classes,ids=np.unique(y,return_inverse=True)
    limit=min(768,n)
    if not 2<=len(classes)<=limit: raise ValueError("Invalid number of classes")
    rng=np.random.default_rng(seed)
    if limit<n:
        first=[rng.choice(np.flatnonzero(ids==i)) for i in range(len(classes))]
        remaining=np.setdiff1d(np.arange(n),first,assume_unique=True)
        rows=np.sort(np.r_[first,rng.choice(remaining,limit-len(first),replace=False)])
    else: rows=np.arange(n)
    Z=np.array(X[rows],order='C',dtype=np.float32)
    if not np.isfinite(Z).all(): raise ValueError("Expected finite features")
    Z-=Z.mean(axis=0,dtype=np.float64)
    norm=np.linalg.norm(Z,axis=0)
    Z/=np.where(norm>0,norm/np.sqrt(np.float32(len(rows))),1.)
    Y=(ids[rows,None]==np.arange(len(classes))).astype(np.float32)
    Y-=Y.mean(axis=0)
    high=min(p-1,max(1,p//4),4*isqrt(limit))
    sizes=sorted({high//2,high})
    order=rng.permutation(p)
    votes,residual=_nested(Z,Y,order,sizes,np.sqrt(limit))
    # Allow one position per anchor plus one per residualized standard deviation.
    unexplained=np.sqrt(np.clip(residual[order[high:]]/limit,0,1))
    auto=min(p,max(1,high+ceil(float(unexplained.sum()))))
    scores=votes.min(axis=0)
    k=auto if count is None else count
    return np.sort(np.argsort(-scores,kind='stable')[:k]),dict(
        tiers=sizes,rank=scores,tier_scores=votes,width=k,proxy_dimension=auto,sketch_size=limit)
