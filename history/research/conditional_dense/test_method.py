import itertools
import numpy as np
import pytest
from research.conditional_dense.shared_method import _nested, conditional_support


def objective(X,Y,ids,penalty):
    if not len(ids):return float((Y**2).sum())
    A=X[:,ids]
    coef=np.linalg.solve(A.T@A+penalty*np.eye(len(ids)),A.T@Y)
    return float(((Y-A@coef)**2).sum()+penalty*(coef**2).sum())


def test_schur_and_deletion_votes_match_refitting():
    rng=np.random.default_rng(92);X=rng.normal(size=(27,6));X[:,1]=X[:,0];X[:,-1]=0
    Y=rng.normal(size=(len(X),3));order=np.array([3,4,0,2,1,5]);sizes=[0,2,5]
    gains,_=_nested(X,Y,order,sizes,3.)
    for i,size in enumerate(sizes):
        for j in range(X.shape[1]):
            others=order[order!=j][:size]
            expected=objective(X,Y,others,3.)-objective(X,Y,np.r_[others,j],3.)
            assert gains[i,j]==pytest.approx(expected,abs=1e-11)


def test_exact_mean_over_uniform_fixed_size_peers():
    rng=np.random.default_rng(29); X=rng.normal(size=(14,4));Y=rng.normal(size=(14,2))
    gains=np.mean([_nested(X,Y,np.array(o),[1,2],2.)[0] for o in itertools.permutations(range(4))],axis=0)
    for i,s in enumerate([1,2]):
        for j in range(4):
            terms=[]
            for peers in itertools.combinations([v for v in range(4) if v!=j],s):
                terms.append(objective(X,Y,list(peers),2.)-objective(X,Y,[*peers,j],2.))
            assert gains[i,j]==pytest.approx(np.mean(terms),abs=1e-12)


def test_reproducible_width_minimum_and_constant_column():
    rng=np.random.default_rng(8);X=rng.normal(size=(1000,73));X[:,-1]=7
    y=np.argmax(X[:,:3],axis=1)
    a,info=conditional_support(X,y);b,_=conditional_support(X,y)
    assert np.array_equal(a,b) and len(set(a))==info['width']
    assert info['rank'][-1]==0
    np.testing.assert_allclose(info['rank'],info['tier_scores'].min(axis=0))
    c,_=conditional_support(X,y,17)
    np.testing.assert_array_equal(c,np.sort(np.argsort(-info['rank'],kind='stable')[:17]))
