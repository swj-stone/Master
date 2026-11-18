"""Permutation importance of a frozen pilot Ridge under held-out squared loss."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from research.algorithms.coalition_residual import _standardize


def pfi_top(X, y, task, seed, k):
    n = len(y)
    nfit = min(128, max(2,n//3))
    subset = np.random.default_rng(seed).choice(n,size=min(n,nfit+512),replace=False)
    train, pool = subset[:nfit], subset[nfit:]
    x, mean, scale = _standardize(X[train])
    z, _, _ = _standardize(X[pool], mean, scale)
    target = (np.eye(len(np.unique(y)))[np.unique(y,return_inverse=True)[1]]
              if task == "classification" else np.asarray(y,dtype=float)[:,None])
    model = Ridge(alpha=1.,solver="cholesky").fit(x,target[train])
    pred = model.predict(z).reshape(len(z),-1)
    if pred.shape[1]>1:
        top = np.partition(pred,-2,axis=1)[:,-2:]
        priority=top[:,1]-top[:,0]
    else:
        priority=pred.ravel()
    size=min(256,len(z))
    pos=((np.arange(size)+.5)*len(z)/size).astype(int)
    use=np.argsort(priority,kind="stable")[pos]
    result=permutation_importance(model,z[use],target[pool[use]],scoring="neg_mean_squared_error",
                                  n_repeats=3,random_state=seed,n_jobs=1)
    return np.sort(np.argsort(-result.importances_mean,kind="stable")[:k])
