"""Worst exchange votes can certify an incumbent against entire swap distances."""
from math import comb
from itertools import combinations
import numpy as np
from numba import njit

from minppi_mofs.exchange_bnb import bnb_best
from minppi_mofs.exchange_minpi import enumerate_best,value


@njit(cache=True)
def _radius_bounds(g,d,inside,outside,conditioned):
    k=len(inside)
    q=len(outside)
    m=min(k,q)
    row_bounds=np.full((m,k),np.inf)
    col_bounds=np.full((m,q),np.inf)
    for a,i in enumerate(inside):
        for b,j in enumerate(outside):
            left=np.empty(k-1)
            right=np.empty(q-1)
            u=v=0
            for pos,l in enumerate(inside):
                if pos!=a:
                    left[u]=g[i,l]-g[j,l]
                    u+=1
            for pos,l in enumerate(outside):
                if pos!=b:
                    right[v]=g[i,l]-g[j,l]
                    v+=1
            gain=d[i]-d[j]
            observed=gain-2*left.sum()
            if conditioned:
                left.sort()
                right.sort()
                worst_sum=left.sum()
            else:
                # Width-free baseline uses the very same endpoint and distance aggregation.
                worst_sum=np.maximum(left,0.).sum()+np.maximum(right,0.).sum()
            for t in range(m):
                if t and conditioned:
                    worst_sum+=right[q-1-t]-left[t-1]
                twice=observed+gain-2*worst_sum
                row_bounds[t,a]=min(row_bounds[t,a],twice)
                col_bounds[t,b]=min(col_bounds[t,b],twice)
    bounds=np.empty(m)
    for t in range(m):
        bounds[t]=max(np.sort(row_bounds[t])[:t+1].sum(),np.sort(col_bounds[t])[:t+1].sum())/2
    return bounds


def tangent_bounds(g,d,inside,outside,eigen_floor):
    derivative=d+np.diag(g)-2*g[:,inside].sum(axis=1)
    maximum=min(len(inside),len(outside))
    rank=np.arange(1,maximum+1)
    return (np.cumsum(np.sort(derivative[inside])[:maximum])-
            np.cumsum(np.sort(derivative[outside])[::-1][:maximum])+2*rank*eigen_floor)


def incumbent(g,d,k):
    """Common mean-marginal warm start and exact one-swap descent."""
    p=len(d)
    off=g.copy()
    np.fill_diagonal(off,0.)
    score=d-2*(k-1)/max(1,p-1)*off.sum(axis=1)
    mask=np.zeros(p,bool)
    mask[np.argsort(-score,kind="stable")[:k]]=True
    while 0<k<p:
        ins,outs=np.flatnonzero(mask),np.flatnonzero(~mask)
        marginal=d-2*off[:,ins].sum(axis=1)
        delta=marginal[outs,None]-marginal[ins]+2*off[np.ix_(outs,ins)]
        i,j=np.unravel_index(delta.argmax(),delta.shape)
        if delta[i,j]<=0:
            break
        mask[outs[i]],mask[ins[j]]=True,False
    return mask


def radius_best(g,d,k,method="radius",*,quantized_psd=False):
    """Exact on the entered game. All ties stay eligible for the fallback solver.

    The PSD floor -p/2 is available only for a Gram matrix rounded entrywise
    to the nearest integer from a PSD matrix in the same scaled coordinates.
    """
    p=len(d)
    mask=incumbent(g,d,k)
    a,z=np.flatnonzero(mask),np.flatnonzero(~mask)
    radii=min(k,p-k)
    stats=dict(radii=radii,excluded=0,certified=False,nodes=0)
    if not radii:
        return a,stats
    if method!="bnb":
        floor=(-p/2 if quantized_psd else
               np.min(2*np.diag(g)-np.abs(g).sum(axis=1)))
        # These integer-data bounds are exact when all sums fit in binary64.
        lower=tangent_bounds(g,d,a,z,floor)
        if method!="tangent" and not np.all(lower>0):
            score=_radius_bounds(g,d,a,z,method=="radius")
            lower=np.maximum(lower,score)
        active=np.flatnonzero(lower<=0)+1
        stats["excluded"]=radii-len(active)
        if not len(active):
            stats["certified"]=True
            return a,stats
        # The incumbent already maximizes its one-swap neighbourhood. Ties keep the common warm start.
        if np.max(active)==1:
            stats["certified"]=True
            return a,stats
        unresolved=[int(r) for r in active if r>1]
        remaining=sum(comb(len(a),r)*comb(len(z),r) for r in unresolved)
        if remaining and remaining<=100000 and len(unresolved)<radii-1:
            best=a
            high=value(g,d,a)
            local=g[np.ix_(z,z)]
            for r in unresolved:
                for removed in combinations(a,r):
                    kept=np.setdiff1d(a,removed,assume_unique=True)
                    coef=d[z]-2*g[np.ix_(z,kept)].sum(axis=1)
                    extra=enumerate_best(local,coef,r)
                    candidate=np.sort(np.r_[kept,z[extra]])
                    score=value(g,d,candidate)
                    if score>high:
                        high,best=score,candidate
            stats["shell_sets"]=remaining
            return best,stats
    winner,steps=bnb_best(g,d,k,initial=mask)
    stats["nodes"]=steps["nodes"]
    return winner,stats
