"""Exact singleton-deletion greedy search with cached feature-wise directional bounds."""
from time import perf_counter
import numpy as np
from numba import njit


@njit(cache=True)
def _projections(delta,ap,ai,av,ip,ii):
    p,n=delta.shape
    bins=np.empty((p,n),np.int8)
    den=np.zeros((p,4))
    for j in range(p):
        rms=np.sqrt(np.sum(delta[j]**2)/n)
        for row in range(n):
            d=delta[j,row]
            b=int(d>0)*2+int(abs(d)>rms)
            bins[j,row]=b
            den[j,b]+=d*d
    contributions=np.zeros((len(ii),4))
    for group in range(len(ip)-1):
        for pos in range(ip[group],ip[group+1]):
            j=ii[pos]
            for z in range(ap[group],ap[group+1]):
                row=ai[z]
                contributions[pos,bins[j,row]]+=av[z]*delta[j,row]
    return den,contributions


@njit(cache=True)
def _anchors(delta):
    p,n=delta.shape
    bins=np.empty((p,n),np.int8)
    den=np.zeros((p,4))
    for j in range(p):
        rms=np.sqrt(np.sum(delta[j]**2)/n)
        for row in range(n):
            d=delta[j,row]
            b=int(d>0)*2+int(abs(d)>rms)
            bins[j,row]=b
            den[j,b]+=d*d
    return den,bins


@njit(cache=True)
def _lazy_update(lost,ap,ai,av,ip,ii,delta,bins,acc,known,directions):
    for group in lost:
        for pos in range(ip[group],ip[group+1]):
            j=ii[pos]
            known[j]=False
            if directions:
                sums=np.zeros(4)
                for z in range(ap[group],ap[group+1]):
                    row=ai[z]
                    sums[bins[j,row]]+=av[z]*delta[j,row]
                acc[j]-=sums


@njit(cache=True)
def _cross_update(ap,ai,av,delta,active,cross):
    for row,v in enumerate(delta):
        if v!=0.:
            for z in range(ap[row],ap[row+1]):
                g=ai[z]
                if active[g]:
                    cross[g]+=v*av[z]


@njit(cache=True)
def _aggregate(n,ap,ai,av,active,groups):
    val=np.zeros(n)
    for g in groups:
        if active[g]:
            for z in range(ap[g],ap[g+1]):
                val[ai[z]]+=av[z]
    return val


@njit(cache=True)
def _update(lost,ip,ii,mass,acc,known,directions):
    for g in lost:
        for pos in range(ip[g],ip[g+1]):
            j=ii[pos]
            known[j]=False
            if directions:
                acc[j]-=mass[pos]


def search(game,order,directions=True,rho=.2,lazy=False):
    p=game.p
    n=len(game.full)
    a,inc=game.a,game.incidence
    ap,ai,av=a.indptr,a.indices,a.data
    ip,ii=inc.indptr,inc.indices
    t=perf_counter()
    delta0=np.ascontiguousarray((a@inc).toarray().T)
    squared=np.einsum("ij,ij->i",delta0,delta0)
    bins=np.empty((0,0),np.int8)
    if directions and lazy:
        den,bins=_anchors(delta0)
        mass=np.empty((0,4))
    elif directions:
        den,mass=_projections(delta0,ap,ai,av,ip,ii)
    if directions:
        acc=den.copy()
        inv=np.divide(1.,den,out=np.zeros_like(den),where=den>0)
    else:
        acc=inv=mass=np.empty((0,4))
    if lazy:
        row_matrix=a.tocsr()
        cross=np.zeros(a.shape[1])
    if not lazy or not directions:
        delta0=np.empty((0,0))
    setup=perf_counter()-t
    keep=np.ones(p,bool)
    known=np.ones(p,bool)
    active=np.ones(a.shape[1],bool)
    residual=np.zeros(n)
    ranks=np.argsort(order)
    path=[]
    recalculated=0
    candidates=0
    limit=(rho*game.norm+1e-12*max(1.,game.norm))**2
    numerical_guard=2e-12*max(1.,game.norm**2)
    t=perf_counter()
    while keep.any():
        cross_terms=cross if lazy else a.T@residual
        fixed=residual@residual+2*np.asarray(inc.T@(cross_terms*active)).ravel()
        bound=np.einsum("ij,ij,ij->i",acc,inv,acc) if directions else np.zeros(p)
        lower=fixed+np.where(known,squared,bound)
        lower[~keep]=np.inf
        sorted_ids=np.lexsort((-ranks,lower))
        win=-1
        best=limit
        candidates+=int(keep.sum())
        for j in sorted_ids:
            if lower[j]>best+numerical_guard:
                break
            if not known[j]:
                delta=_aggregate(n,ap,ai,av,active,game.groups[j])
                squared[j]=delta@delta
                known[j]=True
                recalculated+=1
            exact=fixed[j]+squared[j]
            if exact<best or (exact==best and (win<0 or ranks[j]>ranks[win])):
                best=exact
                win=j
        if win<0:
            break
        groups=game.groups[win]
        lost=groups[active[groups]]
        removed=_aggregate(n,ap,ai,av,active,lost)
        residual+=removed
        keep[win]=False
        active[lost]=False
        if lazy:
            _cross_update(row_matrix.indptr,row_matrix.indices,row_matrix.data,removed,active,cross)
            _lazy_update(lost,ap,ai,av,ip,ii,delta0,bins,acc,known,directions)
        else:
            _update(lost,ip,ii,mass,acc,known,directions)
        path.append(int(win))
    return np.flatnonzero(keep),dict(deletion_order=path,setup_s=setup,core_s=perf_counter()-t,
        recomputed=recalculated,known_initial=p,all_candidates=candidates)
