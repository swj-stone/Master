import numpy as np
from scipy.linalg import solve

from research.algorithms.coalition_residual import _sample, _standardize, select_features


def test_path_matches_independently_refitted_ridge():
    rng=np.random.default_rng(43)
    X=rng.normal(size=(501,14))
    X[:,2]=X[:,0]
    X[::9,4]=np.nan
    y=X[:,0]+.8*X[:,7]+rng.normal(size=len(X))
    keep,info=select_features(X,y,task="regression",count=7)
    fit,rows,target=_sample(X,y,"regression",1129)
    assert not np.intersect1d(fit,rows).size
    x,_,_=_standardize(X[rows])
    target-=target.mean(axis=0)
    def objective(ids):
        if not ids:
            return float((target**2).sum())
        a=x[:,ids]
        coef=solve(a.T@a+np.eye(len(ids)),a.T@target,assume_a="pos")
        return float(np.square(a@coef-target).sum()+np.square(coef).sum())
    zero=objective([])
    initial=np.array([zero-objective([j]) for j in range(x.shape[1])])
    path=[]
    for j,score in zip(info["path"],info["gains"],strict=True):
        before=objective(path)
        all_scores=[min(initial[c],before-objective([*path,c])) if c not in path else -np.inf for c in range(x.shape[1])]
        np.testing.assert_allclose(score,max(all_scores),atol=1e-9)
        np.testing.assert_allclose(score,all_scores[j],atol=1e-9)
        path.append(int(j))
    np.testing.assert_array_equal(keep,np.sort(path))


def test_full_fallback_and_large_fixed_budget():
    X=np.random.default_rng(32).normal(size=(75,611))
    y=X[:,6]
    cols,_=select_features(X,y,count=510,task="regression")
    assert len(np.unique(cols))==510
    full,_=select_features(X,np.zeros(len(y)),task="regression")
    np.testing.assert_array_equal(full,np.arange(X.shape[1]))
