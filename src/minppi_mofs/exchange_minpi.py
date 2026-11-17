"""Strict exchange votes on fixed-width coalitions of an additive predictor."""
from dataclasses import dataclass
from math import comb
from time import perf_counter

import numpy as np
from numba import njit


@dataclass
class Screen:
    keep: np.ndarray
    free: np.ndarray
    order: np.ndarray
    rounds: int


def integer_game(gram,single):
    """Round a finite game to an integer lattice with exact binary64 score accumulation.

    For width k each unscaled support utility changes by at most k*k/(2*scale).
    """
    p=len(single)
    largest=max(float(np.max(np.abs(gram))),float(np.max(np.abs(single))))
    if not np.isfinite(largest):
        raise ValueError("Nonfinite fitted game")
    if largest==0.:
        return gram.copy(),single.copy(),1.
    bits=min(36,46-int(np.ceil(np.log2(max(1,p*p)))))
    scale=2.**bits/largest
    return np.rint(gram*scale),np.rint(single*scale),scale


@njit(cache=True)
def _dominance(gram,single,k,restricted,guard):
    p=len(single)
    edges=np.zeros((p,p),np.bool_)
    for i in range(p):
        for j in range(i):
            diffs=np.empty(p-2)
            t=0
            pos=0.
            neg=0.
            total=0.
            for c in range(p):
                if c==i or c==j:
                    continue
                z=gram[i,c]-gram[j,c]
                diffs[t]=z
                t+=1
                pos+=max(z,0.)
                neg+=min(z,0.)
                total+=z
            gap=single[i]-single[j]
            # The free-width extremum suffices whenever it already certifies a direction.
            if gap-2*pos>guard:
                edges[i,j]=True
            elif gap-2*neg < -guard:
                edges[j,i]=True
            elif restricted:
                if k==1:
                    upper=lower=0.
                elif k==p-1:
                    upper=lower=total
                else:
                    diffs.sort()
                    upper=diffs[p-k-1:].sum()
                    lower=diffs[:k-1].sum()
                edges[i,j]=gap-2*upper>guard
                edges[j,i]=gap-2*lower < -guard
    return edges


def dominance(gram,single,k,rule="width"):
    """Edge i -> j certifies strictly higher utility for i on every admissible base."""
    p=len(single)
    if not 0<k<p:
        raise ValueError("Exchanges need an interior width")
    scale=max(np.max(np.abs(single)), 2*p*np.max(np.abs(gram)),np.finfo(float).tiny)
    # A conservative buffer for subtraction/summation of the supplied floating-point game.
    guard=128*p*np.finfo(float).eps*scale
    if rule not in ("width","unrestricted"):
        raise ValueError(rule)
    return _dominance(gram,single,k,rule=="width",guard)


def screen(gram,single,k,rule="width"):
    p=len(single)
    free=np.arange(p)
    keep=[]
    rounds=0
    local=single.copy()
    while 0<k<len(free):
        rounds+=1
        edges=dominance(gram[np.ix_(free,free)],local,k,rule)
        inside=edges.sum(axis=1)>=len(free)-k
        outside=edges.sum(axis=0)>=k
        assert not np.any(inside & outside)
        if not np.any(inside|outside):
            return Screen(np.array(keep,dtype=int),free,edges,rounds)
        keep.extend(free[inside].tolist())
        remaining=~(inside|outside)
        local=(local-2*gram[np.ix_(free,free[inside])].sum(axis=1))[remaining]
        free=free[remaining]
        k-=int(inside.sum())
    if k==len(free):
        keep.extend(free.tolist())
    return Screen(np.array(keep,dtype=int),np.empty(0,dtype=int),np.empty((0,0),bool),rounds)


def value(gram,single,columns):
    sub=gram[np.ix_(columns,columns)]
    return float(single[columns].sum()-(sub.sum()-sub.trace()))


@njit(cache=True)
def _plain(gram,single,k,last):
    """Reference enumeration in original column order, without precedence bookkeeping."""
    p=len(single)
    cols=np.empty(k,np.int64)
    partial=np.zeros(k+1)
    level=col=0
    best=-np.inf
    result=np.empty(k,np.int64)
    while level>=0:
        if col>p-k+level:
            level-=1
            if level>=0:
                col=cols[level]+1
            continue
        score=partial[level]+single[col]
        for i in range(level):
            score-=2*gram[col,cols[i]]
        cols[level]=col
        if level==k-1:
            if score>best or (score==best and last):
                best=score
                result[:]=cols
        else:
            partial[level+1]=score
            level+=1
        col+=1
    return result


@njit(cache=True)
def _enumerate(off,single,k,parents,priority,prefer_high):
    """Depth-first enumeration reuses the score of every common prefix."""
    p=len(single)
    cols=np.empty(k,np.int64)
    sums=np.zeros(k+1)
    masks=np.zeros(k+1,np.int64)
    priorities=np.zeros(k+1,np.int64)
    col=0
    level=0
    high=-np.inf
    best_priority=np.int64(0)
    winner=np.zeros(k,np.int64)
    while level>=0:
        if col>p-k+level:
            level-=1
            if level>=0:
                col=cols[level]+1
            continue
        if parents[col] & ~masks[level]:
            col+=1
            continue
        s=sums[level]+single[col]
        for t in range(level):
            s-=2*off[col,cols[t]]
        cols[level]=col
        if level==k-1:
            tie=priorities[level] | priority[col]
            if s>high or (s==high and ((tie>best_priority) if prefer_high else (tie<best_priority))):
                high=s
                best_priority=tie
                winner[:]=cols
        else:
            sums[level+1]=s
            masks[level+1]=masks[level] | (np.int64(1)<<col)
            priorities[level+1]=priorities[level] | priority[col]
            level+=1
        col+=1
    return winner


def enumerate_best(gram,single,k,edges=None):
    """Exact search on at most 62 columns, pruning prefixes outside the certified order."""
    p=len(single)
    complement=k>p//2
    if complement:
        single=2*(gram.sum(axis=1)-gram.diagonal())-single
        edges=None if edges is None else edges.T
        k=p-k
    if k==0:
        return np.arange(p) if complement else np.empty(0,int)
    if edges is None or not np.any(edges) or p>62:
        best=_plain(gram,single,k,complement)
        return np.setdiff1d(np.arange(p),best) if complement else best
    permutation=np.arange(p)
    parents=np.zeros(p,dtype=np.int64)
    if edges is not None and np.any(edges):
        # Ordering by a Kahn traversal also handles a numerically conservative subgraph.
        indegree=edges.sum(axis=0)
        order=[]
        for _ in range(p):
            eligible=np.flatnonzero(indegree==0)
            if not len(eligible):
                raise ArithmeticError("Cyclic strict exchange votes")
            first=eligible[0]
            order.append(first)
            indegree[first]=p+1
            indegree-=edges[first]
        permutation=np.array(order)
        for j in range(p):
            for i in np.flatnonzero(edges[permutation,permutation[j]]):
                parents[j] |= np.int64(1)<<i
    priority=np.left_shift(np.int64(1),p-1-permutation)
    best=(permutation[_enumerate(gram[np.ix_(permutation,permutation)],single[permutation],k,
                                  parents,priority,not complement)] if k else np.empty(0,int))
    return np.setdiff1d(np.arange(p),best) if complement else np.sort(best)


def scip_best(gram,single,k,edges=None,seconds=60.):
    """Off-diagonal product linearization with cardinality lift; native presolve on."""
    from pyscipopt import Model,quicksum
    p=len(single)
    model=Model("fixed_width")
    model.hideOutput()
    model.setRealParam("limits/time",float(seconds))
    model.setIntParam("parallel/maxnthreads",1)
    # Scaling controls SCIP's absolute tolerances on small-risk games.
    scale=max(np.max(np.abs(single)),np.max(np.abs(gram)),1e-20)
    x=[model.addVar(vtype="B") for _ in range(p)]
    model.addCons(quicksum(x)==k)
    products={}
    for i in range(p):
        for j in range(i):
            var=products[i,j]=model.addVar(lb=0.,ub=1.)
            model.addCons(var<=x[i])
            model.addCons(var<=x[j])
            model.addCons(var>=x[i]+x[j]-1)
    for i in range(p):
        model.addCons(quicksum(products[max(i,j),min(i,j)] for j in range(p) if j!=i)==(k-1)*x[i])
    if edges is not None:
        for i,j in zip(*np.nonzero(edges)):
            model.addCons(x[int(i)]>=x[int(j)])
    model.setObjective(quicksum(float(single[i]/scale)*x[i] for i in range(p))-
                       quicksum(float(2*gram[i,j]/scale)*w for (i,j),w in products.items()),"maximize")
    model.optimize()
    if model.getStatus()!="optimal":
        raise TimeoutError(f"SCIP status {model.getStatus()} gap {model.getGap()}")
    return np.array([i for i in range(p) if model.getVal(x[i])>.5])


def solve(gram,single,k,mode="width",backend="auto",seconds=60.,dispatch=True):
    p=len(single)
    if not 0<=k<=p:
        raise ValueError(k)
    start=perf_counter()
    cheap=dispatch and comb(p,k)<=p**3 and backend in ("auto","enum")
    state=(screen(gram,single,k,mode) if mode!="direct" and not cheap else
           Screen(np.empty(0,int),np.arange(p),None,0))
    f,kept=state.free,state.keep
    residual=k-len(kept)
    work=comb(len(f),residual)
    screening=perf_counter()-start
    if residual==0 or residual==len(f):
        selected=kept if residual==0 else np.r_[kept,f]
        used="fixed"
    else:
        reduced=gram[np.ix_(f,f)]
        linear=single[f]-2*gram[np.ix_(f,kept)].sum(axis=1)
        used=("enum" if len(f)<=62 and work<=1000000 else "scip") if backend=="auto" else backend
        idx=(enumerate_best(reduced,linear,residual,state.order) if used=="enum" else
             scip_best(reduced,linear,residual,state.order,seconds))
        selected=np.r_[kept,f[idx]]
    return np.sort(selected),dict(free=len(f),forced_in=len(kept),rounds=state.rounds,
                                  residual_sets=work,screen_s=screening,elapsed_s=perf_counter()-start,
                                  solver=used)
