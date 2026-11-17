"""Exhaustive physical replay of bounded exchange moves on small games."""
import numpy as np
from scipy import sparse
from experiments.minpi_rework import quota_compress_tree as tree,quota_compress_rescue as r


def physical(g,cols):
    omit=np.setdiff1d(np.arange(g.p),cols)
    lost=np.asarray(g.incidence[:,omit].sum(axis=1)).ravel()>0
    residual=np.asarray(g.a[:,lost].sum(axis=1)).ravel()
    return residual@residual


def check():
    g=tree.Game(sparse.csc_matrix(np.array([[.8,1.2,2.,-2.],[-.8,-1.2,-2.,2.]])),
                sparse.eye(4,format="csr"))
    for ballot in (False,True):
        out,moves,_,_=r.search(g,[0,1],ballot=ballot)
        np.testing.assert_array_equal(out,[2])
        np.testing.assert_array_equal(moves,[[2,1,0]])
        out,_,_,_=r.search(g,[0,1],ballot=ballot,pair=False)
        np.testing.assert_array_equal(out,[0,1])
    rng=np.random.default_rng(5011)
    for d in range(4,10):
        groups=[(j,) for j in range(d)]+[tuple(rng.choice(d,rng.integers(2,4),replace=False)) for _ in range(11)]
        inc=sparse.csr_matrix(([1. for s in groups for _ in s],([i for i,s in enumerate(groups) for _ in s],
            [j for s in groups for j in s])),shape=(len(groups),d))
        g=tree.Game(sparse.csc_matrix(rng.normal(size=(19,len(groups)))*rng.lognormal(0,1,len(groups))),inc)
        initial=g.search("gain_sweep",np.arange(d))[0]
        for pair in (False,True):
            cols,path,_,_=r.search(g,initial,pair=pair,rounds=1,slots=d,firsts=d)
            yes,ys,_,_=r.search(g,initial,pair=pair,ballot=True,rounds=1,slots=d,firsts=d)
            np.testing.assert_array_equal(cols,yes)
            np.testing.assert_array_equal(path,ys)
            best2=best1=np.inf
            for a in set(range(d))-set(initial):
                for j in initial:
                    s=(set(initial)|{a})-{j}
                    best1=min(best1,physical(g,sorted(s)))
                    for h in initial:
                        if h!=j:
                            best2=min(best2,physical(g,sorted(s-{h})))
            budget=(.2*g.norm+1e-12*max(1.,g.norm))**2
            if pair and best2<=budget:
                assert path.size and path[0,2]>=0
                a,j,h=path[0]
                assert np.isclose(physical(g,sorted((set(initial)|{a})-{j,h})),best2)
            elif best1<physical(g,initial)-1e-10*max(1e-100,(budget/.04)):
                a,j,h=path[0]
                assert h==-1 and np.isclose(physical(g,sorted((set(initial)|{a})-{j})),best1)
            else:
                assert not len(path)
            assert physical(g,cols)<=budget


if __name__=="__main__":
    check()
    print("One-for-two escape and brute-force exchange checks passed.")
