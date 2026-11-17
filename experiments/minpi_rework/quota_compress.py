"""Minimum-cardinality approximation of fixed additive teacher scores."""
from math import ceil
from time import perf_counter

import numpy as np
from pyscipopt import Model, quicksum


RHO = .20
LIMIT = 10.


def loss(g,keep):
    omitted=~keep
    return float(g[np.ix_(omitted,omitted)].sum())


def feasible(g,keep,rho):
    return loss(g,keep)<=rho*rho+1e-12


def prefix(g,order,rho=RHO):
    keep=np.zeros(len(g),bool)
    for j in order:
        if feasible(g,keep,rho):
            break
        keep[j]=True
    return keep


def greedy(g,rho=RHO):
    """Delete a column, or improve the error at the same width and retry."""
    keep=np.ones(len(g),bool)
    if feasible(g,~keep,rho):
        return ~keep,[]
    path=[]
    while keep.any():
        inside,outside=np.flatnonzero(keep),np.flatnonzero(~keep)
        v=g[:,outside].sum(axis=1)
        current=v[outside].sum()
        cost=2*v+g.diagonal()
        j=inside[np.argmin(cost[inside])]
        if current+cost[j]<=rho*rho+1e-12:
            keep[j]=False
            path.append((~keep).astype(float))
            continue
        if not len(outside):
            break
        # Replace currently retained i by currently omitted j; residual adds a_i-a_j.
        delta=(cost[inside,None] + (g.diagonal()-2*v)[None,outside] -
               2*g[np.ix_(inside,outside)])
        i,j=np.unravel_index(delta.argmin(),delta.shape)
        if delta[i,j]>=-1e-12:
            break
        keep[inside[i]],keep[outside[j]]=False,True
    assert feasible(g,keep,rho)
    return keep,path


def ballots(g,vectors,rho=RHO):
    v=np.asarray(vectors,dtype=float)
    product=v@g
    length=np.sqrt(np.maximum(np.einsum("ij,ij->i",v,product),0.))
    ok=length>1e-14
    weight=product[ok]/length[ok,None]
    return weight,weight.sum(axis=1)-rho


def initial_ballots(g,path,rho):
    p=len(g)
    return ballots(g,[np.ones(p),*np.eye(p),*(-np.eye(p)),*path],rho)


def lower_bound(w,q):
    if not len(w):
        return 0
    sums=np.cumsum(np.sort(np.maximum(w,0.),axis=1)[:,::-1],axis=1)
    margins=256*w.shape[1]*np.finfo(float).eps*(1+abs(w).sum(axis=1))
    needed=(sums<(q-margins)[:,None]).sum(axis=1)+1
    return int(np.max(np.where(q>margins,needed,0)))


def fix(w,q,k):
    """Signs +1/-1 certify presence/absence among all feasible supports of width ≤ k."""
    p=w.shape[1]
    state=np.zeros(p,np.int8)
    guard=256*p*np.finfo(float).eps*(1+abs(w).sum(axis=1))
    while True:
        free=np.flatnonzero(state==0)
        slots=k-int((state==1).sum())
        if slots<0:
            raise ArithmeticError("Infeasible voting budget")
        if not slots:
            state[free]=-1
            break
        if not len(free) or not len(w):
            break
        rows=w[:,free]
        known=w[:,state==1].sum(axis=1)
        positive=np.maximum(rows,0.)
        ordered=np.sort(positive,axis=1)[:,::-1]
        prefix_sum=np.column_stack((np.zeros(len(w)),ordered.cumsum(axis=1)))
        def without(t):
            t=min(t,len(free))
            next_value=ordered[:,t] if t<len(free) else np.zeros(len(w))
            return prefix_sum[:,t,None]-np.maximum(0.,positive-next_value[:,None])
        # At t=0 the largest weight equals next_value, giving zero everywhere.
        must=(known[:,None]+without(slots)<(q-guard)[:,None]).any(axis=0)
        never=(known[:,None]+rows+without(slots-1)<(q-guard)[:,None]).any(axis=0)
        if (must&never).any():
            raise ArithmeticError("Inconsistent ballot bounds at a known feasible upper bound")
        if not (must|never).any():
            break
        state[free[must]]=1
        state[free[never]]=-1
    return state


def setup(p,k):
    model=Model("teacher_quota")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads",1)
    x=[model.addVar(vtype="B") for _ in range(p)]
    count=quicksum(x)
    model.setObjective(count,"minimize")
    model.addCons(count<=int(k))
    return model,x


def start_solution(model,x,keep,extra=()):
    sol=model.createSol()
    for v,z in [*zip(x,keep),*extra]:
        model.setSolVal(sol,v,float(z))
    return bool(model.addSol(sol))


def opt(model,x,deadline):
    available=deadline-perf_counter()
    if available<=.01:
        return None,0,"timelimit"
    model.setRealParam("limits/time",max(.001,available-.005))
    model.optimize()
    sol=model.getBestSol()
    bound=model.getDualbound()
    lb=ceil(bound-1e-5) if np.isfinite(bound) and abs(bound)<1e10 else 0
    z=None if sol is None else np.array([model.getSolVal(sol,v)>.5 for v in x])
    return z,int(lb),str(model.getStatus())


def master(g,incumbent,path,apply_fix,rho=RHO,seconds=LIMIT):
    t=perf_counter()
    deadline=t+seconds
    best=incumbent.copy()
    w,q=initial_ballots(g,path,rho)
    lb=lower_bound(w,q)
    state=np.zeros(len(g),np.int8)
    model,x=setup(len(g),best.sum())
    cuts,rounds=0,0
    status="timelimit"
    while perf_counter()<deadline and lb<int(best.sum()):
        if apply_fix:
            updated=fix(w,q,int(best.sum()))
            assert not ((updated==1)&~best).any() and not ((updated==-1)&best).any()
            for j in np.flatnonzero((state!=updated)&(updated!=0)):
                (model.chgVarLb(x[j],1.) if updated[j]==1 else model.chgVarUb(x[j],0.))
            state=updated
            lb=max(lb,int((state==1).sum()))
        if lb==best.sum():
            break
        for weights,quota in zip(w[cuts:],q[cuts:]):
            model.addCons(quicksum(float(weights[j])*x[j] for j in np.flatnonzero(weights))>=float(quota))
        cuts=len(w)
        start_solution(model,x,best)
        z,lower,status=opt(model,x,deadline)
        lb=max(lb,lower)
        rounds+=1
        if z is None:
            break
        if feasible(g,z,rho):
            if z.sum()<best.sum():
                best=z
            if lb>=best.sum() or status!="optimal":
                break
        if lb>=best.sum() or perf_counter()>=deadline:
            break
        if feasible(g,z,rho):
            raise ArithmeticError("Optimal feasible master solution failed to close cardinality gap")
        a,b=ballots(g,[(~z).astype(float)],rho)
        if not (a@z < b-1e-11).all():
            raise ArithmeticError("Residual did not separate its candidate")
        w=np.vstack((w,a))
        q=np.r_[q,b]
        model.freeTransform()
    # Auditing group sizes at the final incumbent does not enter the solve clock.
    elapsed=perf_counter()-t
    final=fix(w,q,int(best.sum()))
    assert not ((final==1)&~best).any() and not ((final==-1)&best).any()
    assert lb<=best.sum() and feasible(g,best,rho)
    return best,dict(lb=lb,certified=bool(lb==best.sum()),rounds=rounds,ballots=len(w),
        forced_in=int((final==1).sum()),forced_out=int((final==-1).sum()),
        status="optimal" if lb==best.sum() else "timelimit",last_master_status=status,solver_s=elapsed)


def cone(g,incumbent,rho=RHO,seconds=LIMIT):
    t=perf_counter()
    model,x=setup(len(g),incumbent.sum())
    eig,v=np.linalg.eigh(g)
    root=np.sqrt(np.maximum(eig,0.))[:,None]*v.T
    aux=[]
    for r in root:
        y=model.addVar(lb=float(np.minimum(r,0).sum()-1e-10),ub=float(np.maximum(r,0).sum()+1e-10))
        model.addCons(y==float(r.sum())-quicksum(float(s)*a for s,a in zip(r,x)))
        aux.append(y)
    model.addCons(quicksum(y*y for y in aux)<=rho*rho)
    start_solution(model,x,incumbent,zip(aux,root@(~incumbent)))
    z,lb,status=opt(model,x,t+seconds)
    best=incumbent.copy()
    if z is not None and feasible(g,z,rho) and z.sum()<=best.sum():
        best=z
    assert lb<=best.sum()
    return best,dict(lb=lb,certified=bool(lb==best.sum()),status=status,solver_s=perf_counter()-t)
