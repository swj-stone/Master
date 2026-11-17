"""Exhaustive small-game verification of variable-width certified order and closures."""
from itertools import combinations
import numpy as np
from experiments.minpi_rework import quota_compress as q
from experiments.minpi_rework.quota_order import closure,ideal,order_edges,propagate,solve,width_votes


def run():
    rng=np.random.default_rng(2469)
    games=intervals=pairs=instances=removed_feasible=range_only=0
    for p in range(2,9):
        for it in range(4):
            x=rng.integers(-3,4,(4,p))
            if it==1: x[:,1]=x[:,0]
            if it==2: x[:]=0
            gram=(x.T@x).astype(float)
            d=2*gram.sum(axis=1)-gram.diagonal()
            supports={k:list(combinations(range(p),k)) for k in range(p+1)}
            def mask(s):
                a=np.zeros(p,bool); a[list(s)]=True; return a
            losses={s:q.loss(gram,mask(s)) for group in supports.values() for s in group}
            for lo in range(p+1):
                for hi in range(lo,p+1):
                    if lo>p-1 or hi<1: continue
                    edges=order_edges(gram,d,lo,hi,"range")
                    each=width_votes(gram,d,lo,hi)
                    for k in range(lo,hi+1):
                        np.testing.assert_array_equal(each[k-lo],order_edges(gram,d,k,k,"range"))
                    free=order_edges(gram,d,lo,hi,"free")
                    assert (edges>=free).all()
                    range_only+=int((edges&~free).any())
                    for i in range(p):
                        for j in range(p):
                            if i==j: continue
                            universe=set(range(p))-{i,j}
                            gaps=[losses[tuple(sorted(b+(j,)))]-losses[tuple(sorted(b+(i,)))]
                                  for k in range(max(1,lo),min(p-1,hi)+1)
                                  for b in combinations(universe,k-1)]
                            assert bool(edges[i,j])==(min(gaps)>0); pairs+=1
                    relation=closure(edges)
                    for k in range(lo,hi+1):
                        best_loss=min(losses[s] for s in supports[k])
                        bounds=[s for s in supports[k] if losses[s]==best_loss]
                        assert all(np.array_equal(ideal(gram,mask(s),edges),mask(s)) for s in bounds)
                    intervals+=1
            denom=max(1.,gram.sum())
            g=gram/denom
            for upper in (p,max(1,p-1)):
                admiss=[s for group in supports.values() for s in group if len(s)<=upper]
                rho=float(np.sqrt(max(0,sorted(losses[s]/denom for s in admiss)[len(admiss)//2]))+.001)
                viable=[s for s in admiss if q.feasible(g,mask(s),rho)]
                lo=min(map(len,viable))
                rel=closure(order_edges(gram,d,lo,upper,"range"))
                w,rhs=q.initial_ballots(g,[],rho)
                fixed=propagate(w,rhs,lo,upper,rel)
                legal=[s for s in viable if np.array_equal(ideal(gram,mask(s),rel),mask(s))]
                removed_feasible+=len(viable)-len(legal)
                assert min(map(len,legal))==lo
                assert all(np.all(mask(s)[fixed==1]) and not np.any(mask(s)[fixed==-1]) for s in legal)
                if it==0 and upper==p:
                    for backend in ("cone","ballot"):
                        for rule in ("none","free","range","hot_none","hot_range"):
                            z,info=solve(g,np.ones(p,bool),[],backend,rule,3.,rho)
                            assert z.sum()==lo and info["certified"] and info["lb"]==lo
                            instances+=1
            games+=1
    assert removed_feasible and range_only
    print(f"{games} games, {intervals} intervals, {pairs} exhaustive pair votes, {instances} solver proofs; "
          f"{removed_feasible} feasible nonideals excluded; {range_only} intervals sharper than free width")


if __name__=="__main__": run()
