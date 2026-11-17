"""Independent finite checks of the exchange bounds, fixing and restricted deployment."""
from itertools import combinations
import numpy as np
from minppi_mofs.additive_minpi import fit_additive,quadratic
from minppi_mofs.exchange_minpi import dominance,screen,solve,value


def main():
    rng=np.random.default_rng(319)
    configs=0
    for p in range(2,10):
        for case in range(7):
            a=rng.integers(-2,3,size=(12,p))
            if case==0:
                a[:]=0
            if case==1:
                a[:,-1]=a[:,0]
            t=rng.integers(-4,5,size=12)
            g=a.T@a
            d=2*a.T@t-g.diagonal()
            for k in range(1,p):
                subsets=list(combinations(range(p),k))
                objectives=np.array([t@t-np.square(t-a[:,s].sum(axis=1)).sum() for s in subsets])
                best=objectives.max()
                opts=[set(s) for s,u in zip(subsets,objectives) if u==best]
                for rule in ("width","unrestricted"):
                    edges=dominance(g.astype(float),d,k,rule)
                    for i,j in zip(*np.nonzero(edges)):
                        for s in subsets:
                            if j in s and i not in s:
                                swap=list((set(s)-{j})|{i})
                                assert value(g,d,swap)>value(g,d,list(s))
                    state=screen(g.astype(float),d.astype(float),k,rule)
                    assert all(set(state.keep)<=s<=set(state.keep)|set(state.free) for s in opts)
                    chosen,_=solve(g.astype(float),d.astype(float),k,mode=rule,backend="enum",dispatch=False)
                    assert value(g,d,chosen)==best
                    assert tuple(chosen)==min(tuple(sorted(s)) for s in opts)
                configs+=1
    x=rng.normal(size=(80,8))
    x[::8,4]=np.nan
    x[:,7]=np.nan
    for kind in ("linear","spline"):
        m=fit_additive(x[:55],rng.normal(size=55),"regression",kind)
        for s in ([5],[3,0],[7,4,2],list(range(8))):
            drop=np.setdiff1d(np.arange(8),s)
            ref=m.intercept+m.center[drop].sum(axis=0)+m.parts(x[55:])[:,s].sum(axis=1)
            np.testing.assert_allclose(m.restrict(s).predict(x[55:,s]),ref,atol=1e-12)
    print(f"{configs} fixed-width integer problems, all optimum supports and both screening rules verified; restricted deployment matched")


if __name__=="__main__":
    main()
