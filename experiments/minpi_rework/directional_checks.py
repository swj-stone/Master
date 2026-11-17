"""Independent small-game checks against row-space fidelity, including exact ties."""
from itertools import product
from fractions import Fraction

import numpy as np
from threadpoolctl import threadpool_limits

from minppi_mofs.directional_quota import (
    FidelityGame, _bank_reject, _probe, classify, directional_search, propagate,
)


def physical_cases():
    """Exact rational counterexamples for loss cancellation and strict quota ties."""
    big=float(2**28)
    cases=[
        # A Gram formed in double loses the unit in the squared norm of the sum.
        ("large-opposites",[[big,1-big]],.2),
        ("exactly-cancelling",[[big,-big]],0.),
        ("exactly-cancelling-with-tail",[[big,-big,.25]],.125),
        ("slightly-above",[[1.+2**-44,1.-2**-44]],.5),
        ("exactly-equal",[[1.,1.]],.5),
        ("duplicate-across-rows",[[big,1-big,0.],[big,-big,1.]],.25),
        ("subnormal-square",[[1.2e-162,6e-163],[-2.4e-162,-2.4e-162]],.5),
        ("overflow-square",[[2e200,1e200],[-4e200,-4e200]],.5),
        ("mixed-exponents",[[2.**900,-2.**900,2.**-900],[0.,0.,2.**-900]],.25),
        ("zero-design",[[0.,0.,0.]],0.)
    ]
    for tag,values,radius in cases:
        a=np.array(values)
        y=[[Fraction(v) for v in row] for row in values]
        masks=np.array(list(product([False,True],repeat=a.shape[1])))
        losses=[sum(sum((coef for coef,on in zip(row,cols) if not on),start=Fraction(0))**2
                    for row in y) for cols in masks]
        passing=np.array([v<=Fraction(str(radius))**2*losses[0] for v in losses])
        yield tag,a,radius,masks,passing


def compatible(masks, state):
    return np.all(masks[:,state==1],axis=1) & ~np.any(masks[:,state==-1],axis=1)


def forbidden_labels(masks,state):
    assert np.all(compatible(masks,state)), "A signed feature label excludes a feasible support"


def checks():
    rng=np.random.default_rng(181109)
    games=nodes=label_tests=solver_tests=bank_tests=exact_ties=0
    for p in range(1,9):
        masks=np.array(list(product([False,True],repeat=p)))
        counts=masks.sum(axis=1)
        for case in ("dense","zero","cancel","opposites","rank1"):
            a=rng.integers(-3,4,size=(4,p))
            if case=="zero":
                a[:]=0
            elif case=="cancel" and p>=2:
                a[:,1]=-a[:,0]
            elif case=="opposites":
                a[:,1::2]=-a[:,:2*(p//2):2]
            elif case=="rank1":
                a[1:]=0
            # Dyadic scaling makes direct integer row-space errors and equality cases exact.
            game=FidelityGame.from_factor(a)
            g=game.gram
            losses=np.sum(((~masks).astype(int)@a.T)**2,axis=1)/64
            L=max(1e-16,np.abs(g).sum(axis=1).max())
            for rho in (0., .25, .5, 1.):
                quota=rho*rho*losses[0]
                legal=losses<=quota
                exact_ties+=int((losses==quota).sum())
                optimum=counts[legal].min()
                for sel in (len(masks)-1,int(rng.choice(np.flatnonzero(legal)))):
                    initial=masks[sel]
                    k=int(counts[sel])
                    viable=masks[legal & (counts<=k)]
                    state,*_=classify(g,initial,rho,L,steps=8,roundoff=game.roundoff)
                    forbidden_labels(viable,state)
                    label_tests+=1
                    for shared,cache,deadline in ((False,False,.5),(False,True,.5),
                                                  (True,False,.5),(True,True,.5),(True,True,0.)):
                        support,info=directional_search(game,initial,rho,steps=8,shared=shared,
                                                       cache=cache,max_seconds=deadline)
                        true_loss=np.sum(a[:,~support].sum(axis=1)**2)/64
                        assert true_loss<=quota
                        assert info["lb"]<=optimum<=support.sum()<=k
                        if info["certified"]:
                            assert info["lb"]==optimum==support.sum()
                        if deadline>0 and p<=5:
                            assert info["certified"]
                        if deadline==0:
                            assert info["nodes"]==0 and info["projection_steps"]==0
                        solver_tests+=1
                for _ in range(3):
                    k=int(rng.integers(p+1))
                    state=rng.integers(-1,2,size=p,dtype=np.int8)
                    legal_node=legal & compatible(masks,state) & (counts<=k)
                    # Warm starts deliberately range outside [0,1].
                    rejected,*_=_probe(g,state,k,rho,L,13,rng.normal(size=p)*3,game.roundoff)
                    assert not rejected or not legal_node.any()
                    nodes+=1
                    directions=rng.normal(size=(4,p))*5
                    w=directions@g
                    rhs=w.sum(axis=1)-rho*np.sqrt(np.maximum(0.,(directions*w).sum(axis=1)))
                    guarded=rhs-1e-10*(1.+abs(w).sum(axis=1))
                    order=np.argsort(-w,axis=1,kind="stable")
                    discarded=_bank_reject(w,guarded,order,len(w),state,k)
                    assert not discarded or not legal_node.any()
                    bank_tests+=1
                    if legal_node.any():
                        labels=state.copy()
                        for v in directions:
                            propagate(g,labels,v,k,rho,game.roundoff)
                        forbidden_labels(masks[legal_node],labels)
                        label_tests+=1
            games+=1
    boundaries=0
    for name,a,rho,masks,passing in physical_cases():
        game=FidelityGame.from_factor(a)
        for s,valid in zip(masks,passing):
            assert game.feasible(s,rho)==valid,(name,s,valid)
        k=masks[passing].sum(axis=1).min()
        for mode in (False,True):
            z,info=directional_search(game,np.ones(a.shape[1],bool),rho,max_seconds=2.,
                                      shared=mode,cache=mode)
            assert info["certified"] and info["lb"]==k==z.sum(),(name,info,z)
            assert passing[np.flatnonzero((masks==z).all(axis=1))[0]]
            boundaries+=1
    for invalid in (np.eye(2), np.ones((1,2))):
        try:
            directional_search(invalid,np.ones(2,bool))
        except ValueError:
            pass
        else:
            raise AssertionError("Unverified raw Gram accepted")
    print(f"{games} dyadic games, {exact_ties} exact quota ties, {label_tests} sign checks, "
          f"{nodes} projection decisions, {bank_tests} cached cuts, {solver_tests} bounded solves, "
          f"{boundaries} rational-boundary solves",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        checks()
