"""Solver-free optimal-width lower bounds from directional votes at trial capacities.

Secondary development experiment at the original 0.20 fidelity threshold.
Fix the strongest cheap incumbent before checking smaller capacities by
contradiction. Add directions from failed one-feature deletions of the incumbent.
All measurements concern existing calibration rows; no new outer outcomes.
"""
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_rework import quota_compress as q,quota_compress_run as study


def propagate(w,rhs,k):
    """Return false only when interval reasoning proves this budget infeasible."""
    state=np.zeros(w.shape[1],np.int8)
    guard=256*w.shape[1]*np.finfo(float).eps*(1+np.abs(w).sum(axis=1))
    target=rhs-guard
    while True:
        free=np.flatnonzero(state==0)
        slots=k-int((state==1).sum())
        if slots<0:
            return False,state
        known=w[:,state==1].sum(axis=1)
        if slots==0 or not len(free):
            state[free]=-1
            return not (known<target).any(),state
        row=w[:,free]
        positive=np.maximum(row,0.)
        order=np.sort(positive,axis=1)[:,::-1]
        null=np.zeros(len(w))
        sums=np.column_stack((null,order.cumsum(axis=1)))
        t=min(slots,len(free))
        if (known+sums[:,t]<target).any():
            return False,state
        def without(slots):
            t=min(slots,len(free))
            cutoff=order[:,t] if t<len(free) else null
            return sums[:,t,None]-np.maximum(positive-cutoff[:,None],0.)
        yes=(known[:,None]+without(slots)<target[:,None]).any(axis=0)
        no=(known[:,None]+row+without(slots-1)<target[:,None]).any(axis=0)
        if (yes&no).any():
            return False,state
        if not (yes|no).any():
            return True,state
        state[free[yes]],state[free[no]]=1,-1


def interval(w,rhs,inc):
    high=int(inc.sum())
    low=q.lower_bound(w,rhs)
    while low<high:
        pivot=(low+high)//2
        possible,_=propagate(w,rhs,pivot)
        if possible:
            high=pivot
        else:
            low=pivot+1
    feasible,state=propagate(w,rhs,int(inc.sum()))
    assert feasible and not ((state==1)&~inc).any() and not ((state==-1)&inc).any()
    return low,state


def check():
    from itertools import product
    rng=np.random.default_rng(9552)
    count=0
    for p in range(2,10):
        supports=np.array(list(product([False,True],repeat=p)))
        for _ in range(6):
            w=rng.normal(size=(2*p,p))
            anchor=supports[rng.integers(len(supports))]
            rhs=w@anchor-rng.uniform(0.,2.,len(w))
            feasible=supports[(supports@w.T>=rhs-1e-12).all(axis=1)]
            lb,_=interval(w,rhs,anchor)
            assert lb<=int(feasible.sum(axis=1).min())
            for k in range(p+1):
                possible,signs=propagate(w,rhs,k)
                exact=feasible[feasible.sum(axis=1)<=k]
                if not possible:
                    assert not len(exact)
                elif len(exact):
                    assert exact[:,signs==1].all() and not exact[:,signs==-1].any()
                count+=1
    print("Enumerated",count,"independent signed multi-ballot capacities.",flush=True)


def run():
    lock=pd.read_csv(study.OUT/"control-lock.csv").set_index("dataset")
    jobs=study.legacy.jobs()
    records=[]
    for n in study.NAMES:
        x,y,task,idx=study.load(n,jobs)
        pool=np.sort(np.r_[idx["fit"],idx["vote"]])
        teacher=study.fit_additive(x[pool],y[pool],task,"linear")
        parts=teacher.parts(x[idx["width"]])-teacher.center
        a=parts.transpose(0,2,1).reshape(-1,x.shape[1])
        g=(a.T@a)/max(1e-100,(a.sum(axis=1)**2).sum())
        inc=np.zeros(len(g),bool)
        inc[np.fromstring(str(lock.loc[n,"selected"]),sep=" ",dtype=int)]=True
        assert q.feasible(g,inc,q.RHO)
        _,path=q.greedy(g)
        initial=q.initial_ballots(g,path,q.RHO)
        omitted=(~inc).astype(float)
        trial=omitted[None,:]+np.eye(len(g))[inc]
        violation=np.einsum("ij,jk,ik->i",trial,g,trial)>q.RHO**2+1e-12
        extra=q.ballots(g,trial[violation],q.RHO)
        for mode,(w,rhs) in (("original",initial),("boundary",(np.vstack((initial[0],extra[0])),np.r_[initial[1],extra[1]]))):
            start=perf_counter()
            lb,signs=interval(w,rhs,inc)
            records.append(dict(dataset=n,p=len(g),k=int(inc.sum()),rule=mode,lower=lb,
                certified=lb==inc.sum(),forced_in=int((signs==1).sum()),forced_out=int((signs==-1).sum()),
                ballots=len(w),screen_s=perf_counter()-start))
    frame=pd.DataFrame(records)
    study.EVIDENCE.mkdir(parents=True,exist_ok=True)
    frame.to_csv(study.EVIDENCE/"early-certificates.csv",index=False)
    print(frame[frame.rule=="boundary"].to_string(index=False))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["run","check"])
    args=parser.parse_args()
    with threadpool_limits(1):
        (check if args.action=="check" else run)()
