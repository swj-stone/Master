"""Native nested-mask Monte Carlo selection from calibration predictions, without a Gram."""
import argparse
import ctypes
import os
from time import perf_counter

import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits

from experiments.minpi_rework import quota_coalition_rf as rf
from experiments.minpi_rework.availability_trial import DEST,SOURCES
from experiments.minpi_rework.predictive_mask_study import centered
from experiments.minpi_rework.predictive_union_control import Terms,MatchedSearch,antithetic_ranks


@njit(cache=True)
def _init(ig,features,ap,rows,values,target,ranks):
    m,p = ranks.shape
    highest = np.empty((m,len(ig)-1),np.int32)
    residual = np.zeros((m,len(target),p))
    for b in range(m):
        for g in range(len(ig)-1):
            s = 0
            for pos in range(ig[g],ig[g+1]):
                s = max(s,ranks[b,features[pos]])
            highest[b,g] = s-1
            for i in range(ap[g],ap[g+1]):
                residual[b,rows[i],s-1] += values[i]
        for t in range(len(target)):
            margin = 0.
            for s in range(p):
                margin += residual[b,t,s]
                residual[b,t,s] = target[t]-margin
    return residual,highest


@njit(cache=True)
def _diff(j,ap,rows,values,fptr,groups,highest,active,residual,work,out,apply):
    out[:] = 0.
    m,n,p = residual.shape
    for b in range(m):
        work[:]=0.
        for k in range(fptr[j],fptr[j+1]):
            g = groups[k]
            if not active[g]:
                continue
            threshold = highest[b,g]
            for i in range(ap[g],ap[g+1]):
                work[rows[i],threshold] += values[i]
        for t in range(n):
            change = 0.
            for s in range(p):
                change += work[t,s]
                if apply:
                    residual[b,t,s] += change
                else:
                    out[s] += change*(2.*residual[b,t,s]+change)


@njit(cache=True)
def _search(ap,rows,values,fp,groups,highest,residual,budget,tol):
    m,n,p = residual.shape
    alive = np.ones(highest.shape[1],np.bool_)
    kept = np.ones(p,np.bool_)
    current = np.zeros(p)
    costs = np.empty(p)
    trials = np.zeros((p,p))
    work = np.zeros((n,p))
    diff = np.zeros(p)
    path = np.empty(p,np.int32)
    length=0
    for _ in range(p):
        costs[:] = np.inf
        for j in range(p):
            if not kept[j]:
                continue
            # No active contribution implies exactly the same trial risk.
            applies=False
            for pos in range(fp[j],fp[j+1]):
                if alive[groups[pos]]:
                    applies=True
                    break
            if applies:
                _diff(j,ap,rows,values,fp,groups,highest,alive,residual,work,diff,False)
            else:
                diff[:]=0.
            peak=-np.inf
            for s in range(p):
                trials[j,s]=current[s]+diff[s]/m
                peak=max(peak,trials[j,s]*p/(s+1))
            costs[j]=peak
        best = costs.min()
        if best>budget+tol:
            break
        j = np.where(costs<=best+tol)[0][0]
        current[:] = trials[j]
        _diff(j,ap,rows,values,fp,groups,highest,alive,residual,work,diff,True)
        for i in range(fp[j],fp[j+1]):
            alive[groups[i]]=False
        kept[j]=False
        path[length]=j
        length+=1
    return path[:length]


def run(a,inc,target,b,draw):
    t=perf_counter()
    a=a.tocsc(copy=False)
    inc=inc.tocsr(copy=False)
    by=inc.tocsc()
    ranks=antithetic_ranks(inc.shape[1],b,np.random.default_rng(420901+draw))
    null=float(target@target)
    absolute=np.asarray(abs(a).sum(axis=1)).ravel()
    tol=128*np.finfo(float).eps*(null+2*np.abs(target)@absolute+absolute@absolute)
    residual,highest=_init(inc.indptr,inc.indices,a.indptr,a.indices,a.data,target,ranks)
    setup=perf_counter()-t
    t=perf_counter()
    path=_search(a.indptr,a.indices,a.data,by.indptr,by.indices,highest,residual,.02*null,tol)
    return path.tolist(),setup,perf_counter()-t,residual.nbytes+highest.nbytes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--names",nargs="+",default=["coimbra","grid","retina","ctloc"])
    parser.add_argument("--draw",type=int,default=0)
    args=parser.parse_args()
    if os.name=="nt":
        if not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<5)):
            raise OSError("CPU affinity not set")
    from scipy.sparse import csc_matrix,csr_matrix
    run(csc_matrix(np.eye(3)),csr_matrix(np.eye(3)),np.arange(3,dtype=float),8,0)
    rows=[]
    for source in SOURCES:
        for name in source.SOURCES:
            if name not in args.names:
                continue
            x,y,task,idx=source.split(name,42)
            pool=np.sort(np.r_[idx["fit"],idx["vote"]])
            model=rf.fit(x[pool],y[pool],task,42)
            game=model.game(x[idx["width"]])
            target=centered(model,y[idx["width"]],task)
            terms=Terms.build(game.a,game.incidence,target)
            for b in (8,32,128):
                path,setup,search,memory=run(game.a,game.incidence,target,b,args.draw)
                ranks=antithetic_ranks(game.p,b,np.random.default_rng(420901+args.draw))
                other=MatchedSearch(terms,ranks).run()[1]
                if path!=other:
                    raise AssertionError((name,b,path,other))
                rows.append(dict(dataset=name,seed=42,draw=args.draw,permutations=b,k=game.p-len(path),
                    setup_s=setup,search_s=search,selector_s=setup+search,working_bytes=memory))
                pd.DataFrame(rows).to_csv(DEST/"matched-mc-direct.csv",index=False)
                print(name,b,"same empirical path",f"{setup+search:.3f}s",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
