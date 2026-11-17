"""Exact fixed-cardinality quadratic search with an optimistic pair bound."""
import numpy as np
from numba import njit


def _topological(edges, priority):
    if edges is None:
        return np.asarray(priority)
    out=[]
    degree=edges.sum(axis=0).astype(int)
    for _ in range(len(priority)):
        candidates=priority[degree[priority]==0]
        if not len(candidates):
            raise ValueError("Precedence constraints contain a cycle")
        i=candidates[0]
        out.append(i)
        degree[i]=len(priority)+1
        degree-=edges[i]
    return np.array(out)


def _start(pair,d,k,edges):
    p=len(d)
    priority=np.lexsort((np.arange(p),-(d+pair.sum(axis=1)*(k-1)/(2*max(1,p-1)))))
    z=np.zeros(p,dtype=bool)
    z[_topological(edges,priority)[:k]]=True
    floor=128*p*np.finfo(float).eps*max(np.abs(d).max(),np.abs(pair).max(),np.finfo(float).tiny)
    while True:
        ins,outs=np.flatnonzero(~z),np.flatnonzero(z)
        score=d+pair[:,z].sum(axis=1)
        diff=score[ins,None]-score[None,outs]-pair[np.ix_(ins,outs)]
        if edges is not None:
            diff[edges[np.ix_(~z,ins)].any(axis=0),:]=-np.inf
            diff[:,edges[np.ix_(outs,z)].any(axis=1)]=-np.inf
            diff[edges[np.ix_(outs,ins)].T]=-np.inf
        i,j=np.unravel_index(np.argmax(diff),diff.shape)
        if diff[i,j]<=floor:
            return z
        z[ins[i]],z[outs[j]]=True,False


@njit(cache=True)
def _search(pair,d,k,parents_lo,parents_hi,initial,guard):
    p=len(d)
    # In suffix pos:, selecting t items gives each selected item t-1 partners.
    bonus=np.zeros((p,k+1,p))
    for pos in range(p):
        for i in range(pos,p):
            row=np.empty(p-pos-1)
            a=0
            for j in range(pos,p):
                if j!=i:
                    row[a]=pair[i,j]
                    a+=1
            row.sort()
            for t in range(2,min(k,p-pos)+1):
                bonus[pos,t,i]=bonus[pos,t-1,i]+.5*row[len(row)-t+1]

    bits_lo=np.zeros(p,np.uint64)
    bits_hi=np.zeros(p,np.uint64)
    for i in range(p):
        if i<64:
            bits_lo[i]=np.uint64(1)<<np.uint64(i)
        else:
            bits_hi[i]=np.uint64(1)<<np.uint64(i-64)
    past_lo=np.zeros(p+1,np.uint64)
    past_hi=np.zeros(p+1,np.uint64)
    for i in range(p):
        past_lo[i+1]=past_lo[i]|bits_lo[i]
        past_hi[i+1]=past_hi[i]|bits_hi[i]
    incumbent=0.
    winner_lo,winner_hi=np.uint64(0),np.uint64(0)
    for i in range(p):
        if initial[i]:
            incumbent+=d[i]
            winner_lo|=bits_lo[i]
            winner_hi|=bits_hi[i]
            for j in range(i):
                if initial[j]:
                    incumbent+=pair[i,j]
    phase=np.zeros(p+1,np.int8)
    mask_lo=np.zeros(p+1,np.uint64)
    mask_hi=np.zeros(p+1,np.uint64)
    need=np.zeros(p+1,np.int64)
    need[0]=k
    partial=np.zeros(p+1)
    linear=np.empty((p+1,p))
    linear[0]=d
    bounds=np.empty(p)
    pos=0
    visited=0
    pruned=0
    while pos>=0:
        if phase[pos]==0:
            visited+=1
            t=need[pos]
            if t==0:
                if partial[pos]>incumbent:
                    incumbent=partial[pos]
                    winner_lo,winner_hi=mask_lo[pos],mask_hi[pos]
                pos-=1
                continue
            if t>p-pos:
                pos-=1
                continue
            excluded_lo=past_lo[pos] & ~mask_lo[pos]
            excluded_hi=past_hi[pos] & ~mask_hi[pos]
            count=0
            for j in range(pos,p):
                if not (parents_lo[j] & excluded_lo or parents_hi[j] & excluded_hi):
                    bounds[count]=linear[pos,j]+bonus[pos,t,j]
                    count+=1
            if count<t:
                pruned+=1
                pos-=1
                continue
            sorted_bounds=np.sort(bounds[:count])
            upper=partial[pos]+sorted_bounds[count-t:].sum()
            if upper<incumbent-guard:
                pruned+=1
                pos-=1
                continue
            phase[pos]=1
            if not (parents_lo[pos] & ~mask_lo[pos] or parents_hi[pos] & ~mask_hi[pos]):
                phase[pos+1]=0
                need[pos+1]=t-1
                mask_lo[pos+1]=mask_lo[pos] | bits_lo[pos]
                mask_hi[pos+1]=mask_hi[pos] | bits_hi[pos]
                partial[pos+1]=partial[pos]+linear[pos,pos]
                for j in range(pos+1,p):
                    linear[pos+1,j]=linear[pos,j]+pair[pos,j]
                pos+=1
                continue
        if phase[pos]==1:
            phase[pos]=2
            phase[pos+1]=0
            need[pos+1]=need[pos]
            mask_lo[pos+1]=mask_lo[pos]
            mask_hi[pos+1]=mask_hi[pos]
            partial[pos+1]=partial[pos]
            for j in range(pos+1,p):
                linear[pos+1,j]=linear[pos,j]
            pos+=1
        else:
            pos-=1
    out=np.empty(k,np.int64)
    count=0
    for i in range(p):
        if winner_lo & bits_lo[i] or winner_hi & bits_hi[i]:
            out[count]=i
            count+=1
    return out,visited,pruned


def bnb_best(gram,single,k,edges=None,initial=None):
    """Return sorted optimal columns and node counts for at most 128 local players.

    An edge i→j requires selecting i whenever j is selected. An optional feasible
    Boolean initial vector is retained on a tie. The computed objective is the
    sum of single weights minus twice the off-diagonal Gram entries in the set.
    """
    g=np.asarray(gram,dtype=float)
    d=np.asarray(single,dtype=float)
    p=len(d)
    if p>128 or not 0<=k<=p or g.shape!=(p,p):
        raise ValueError("Expected a square game on at most 128 features and feasible width")
    if k==0 or k==p:
        return np.arange(k),dict(nodes=0,pruned=0)
    pair=-2*g
    np.fill_diagonal(pair,0.)
    if edges is not None:
        edges=np.asarray(edges,dtype=bool)
        if edges.shape!=g.shape:
            raise ValueError("Precedence matrix shape differs from the Gram matrix")
    if k>p//2:
        dropped,work=bnb_best(g,-d-pair.sum(axis=1),p-k,None if edges is None else edges.T,
                              None if initial is None else ~np.asarray(initial,dtype=bool))
        return np.setdiff1d(np.arange(p),dropped),work
    z=_start(pair,d,k,edges) if initial is None else np.asarray(initial,dtype=bool)
    if z.shape!=(p,) or z.sum()!=k or (edges is not None and edges[np.ix_(~z,z)].any()):
        raise ValueError("The initial support is infeasible")
    order=_topological(edges,np.lexsort((np.arange(p),-np.abs(d),~z)))
    parents_lo=np.zeros(p,dtype=np.uint64)
    parents_hi=np.zeros(p,dtype=np.uint64)
    if edges is not None:
        for j,i in zip(*np.nonzero(edges[np.ix_(order,order)])):
            if j<64:
                parents_lo[i]|=np.uint64(1)<<np.uint64(j)
            else:
                parents_hi[i]|=np.uint64(1)<<np.uint64(j-64)
    scale=np.abs(d).sum()+.5*np.abs(pair).sum()
    if scale==0:
        return np.flatnonzero(z),dict(nodes=1,pruned=1)
    guard=256*p*np.finfo(float).eps*scale
    best,nodes,pruned=_search(pair[np.ix_(order,order)],d[order],k,parents_lo,parents_hi,z[order],guard)
    return np.sort(order[best]),dict(nodes=nodes,pruned=pruned)
