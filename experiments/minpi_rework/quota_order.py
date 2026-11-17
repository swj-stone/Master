"""Order restrictions for minimum-cardinality approximation of a frozen additive score."""
from time import perf_counter
import numpy as np
from numba import njit
from pyscipopt import quicksum

from experiments.minpi_rework import quota_compress as q


@njit(cache=True)
def _edges(g,d,lo,hi,guard):
    p=len(d)
    edges=np.zeros((p,p),np.bool_)
    for i in range(p):
        for j in range(i):
            diff=np.empty(p-2)
            t=0
            for c in range(p):
                if c!=i and c!=j:
                    diff[t]=g[i,c]-g[j,c]
                    t+=1
            diff.sort()
            pos=np.sum(diff>0.)
            neg=np.sum(diff<0.)
            high_count=min(hi,max(lo,pos))
            low_count=min(hi,max(lo,neg))
            high=diff[p-2-high_count:].sum() if high_count else 0.
            low=diff[:low_count].sum()
            gap=d[i]-d[j]
            edges[i,j]=gap-2*high>guard
            edges[j,i]=gap-2*low < -guard
    return edges


def order_edges(g,d,lower,upper,rule):
    p=len(d)
    if rule not in ("none","free","range"): raise ValueError(rule)
    if rule=="none" or p<2 or upper<1 or lower>p-1:
        return np.zeros((p,p),bool)
    lo,hi=(max(0,lower-1),min(p-2,upper-1)) if rule=="range" else (0,p-2)
    scale=max(np.max(np.abs(d)),2*p*np.max(np.abs(g)),np.finfo(float).tiny)
    return _edges(g,d,lo,hi,128*p*np.finfo(float).eps*scale)


@njit(cache=True)
def _width_votes(g,d,lower,upper,guard):
    """Fixed-cardinality dominance over all admissible sizes, sorting each pair only once."""
    p=len(d)
    out=np.zeros((upper-lower+1,p,p),np.bool_)
    for i in range(p):
        for j in range(i):
            a=np.empty(p-2); t=0
            for c in range(p):
                if c!=i and c!=j:
                    a[t]=g[i,c]-g[j,c]; t+=1
            a.sort()
            high=low=0.
            gap=d[i]-d[j]
            for k in range(1,p):
                if k>=lower and k<=upper:
                    out[k-lower,i,j]=gap-2*high>guard
                    out[k-lower,j,i]=gap-2*low < -guard
                if k+1<p:
                    high+=a[p-k-2]; low+=a[k-1]
    return out


def width_votes(g,d,lo,hi):
    p=len(d)
    scale=max(np.max(np.abs(d)),2*p*np.max(np.abs(g)),np.finfo(float).tiny)
    return _width_votes(g,d,lo,hi,128*p*np.finfo(float).eps*scale)


def closure(edges):
    relation=edges.copy()
    for j in range(len(edges)):
        relation |= relation[:,j,None]&relation[j,None,:]
    if np.diag(relation).any(): raise ArithmeticError("A strict exchange cycle is invalid")
    return relation


def ideal(g,keep,edges):
    keep=keep.copy()
    while True:
        bad=np.argwhere(edges & ~keep[:,None] & keep[None,:])
        if not len(bad): return keep
        i,j=bad[0]
        keep[i],keep[j]=True,False


def propagate(w,quotas,lower,upper,relation):
    """Ballot, cardinality and order implications for the chosen ideal representative."""
    state=np.zeros(relation.shape[0],np.int8)
    while True:
        free=np.flatnonzero(state==0)
        if not len(free): break
        included=int(np.sum(state==1))
        lo,hi=max(0,lower-included),min(len(free),upper-included)
        assert 0<=lo<=hi
        rhs=quotas-w[:,state==1].sum(axis=1)
        fixed=q.fix(w[:,free],rhs,hi)
        sub=relation[np.ix_(free,free)]
        must=(fixed==1)|(sub.sum(axis=1)>=len(free)-lo)
        never=(fixed==-1)|(sub.sum(axis=0)>=hi)
        if (state==1).any(): must|=relation[np.ix_(free,np.flatnonzero(state==1))].any(axis=1)
        if (state==-1).any(): never|=relation[np.ix_(np.flatnonzero(state==-1),free)].any(axis=0)
        assert not (must&never).any()
        if not (must|never).any(): break
        state[free[must]]=1
        state[free[never]]=-1
    return state


def conditional_rows(model,x,g,d,state,global_relation,lower,upper,add_votes):
    """Select the realized width, then impose only the orders valid at that width."""
    widths=np.arange(lower,upper+1)
    witness=[model.addVar(vtype="B") for _ in widths]
    model.addCons(quicksum(witness)==1)
    model.addCons(quicksum(x)==quicksum(int(k)*var for k,var in zip(widths,witness)))
    if not add_votes: return witness,widths,0
    free=np.flatnonzero(state==0)
    nr=len(widths)
    arcs=np.zeros((nr,len(free),len(free)),bool)
    have=np.zeros((nr,len(free)),bool)
    skip=np.zeros_like(have)
    all_votes=width_votes(g,d,lower,upper)
    for h,k in enumerate(widths):
        relation=closure(all_votes[h])
        sub=relation[np.ix_(free,free)]
        slots=int(k)-int((state==1).sum())
        inc=sub.sum(axis=1)>=len(free)-slots
        exc=sub.sum(axis=0)>=slots
        if (state==1).any(): inc|=relation[np.ix_(free,np.flatnonzero(state==1))].any(axis=1)
        if (state==-1).any(): exc|=relation[np.ix_(np.flatnonzero(state==-1),free)].any(axis=0)
        if slots<0 or slots>len(free) or (inc&exc).any() or inc.sum()>slots or exc.sum()>len(free)-slots:
            model.chgVarUb(witness[h],0.)
            continue
        have[h],skip[h]=inc,exc
        sub[inc|exc,:]=False; sub[:,inc|exc]=False
        arcs[h]=sub & ~(sub@sub) & ~global_relation[np.ix_(free,free)]
    num=0
    for i,j in zip(*arcs.any(0).nonzero()):
        allowed=np.flatnonzero(arcs[:,i,j])
        model.addCons(x[free[j]]-x[free[i]]<=1-quicksum(witness[t] for t in allowed)); num+=1
    for i in range(len(free)):
        on=np.flatnonzero(have[:,i]); off=np.flatnonzero(skip[:,i])
        if len(on): model.addCons(x[free[i]]>=quicksum(witness[t] for t in on)); num+=1
        if len(off): model.addCons(x[free[i]]<=1-quicksum(witness[t] for t in off)); num+=1
    return witness,widths,num


def solve(g,start,path,backend,rule,seconds=3.,rho=q.RHO):
    """Solve under a common verified incumbent, time-accounted from initial ballots."""
    tic=perf_counter()
    cutoff=tic+seconds
    upper=int(start.sum())
    w,rhs=q.initial_ballots(g,path,rho)
    lower=q.lower_bound(w,rhs)
    d=2*g.sum(axis=1)-g.diagonal()
    hot=rule.startswith("hot_")
    common_rule=rule.removeprefix("hot_")
    if lower<upper:
        relation=closure(order_edges(g,d,lower,upper,common_rule))
        best=ideal(g,start,order_edges(g,d,upper,upper,"range") if rule=="hot_range" else relation)
        state=propagate(w,rhs,lower,upper,relation)
        lower=max(lower,int((state==1).sum()))
    else:
        relation=np.zeros(g.shape,bool); best=start.copy(); state=np.zeros(len(g),np.int8)
    assert lower<=upper and q.feasible(g,best,rho)
    assert not ((state==1)&~best).any() and not ((state==-1)&best).any()
    prep=perf_counter()-tic
    info=dict(initial_upper=upper,edges=int(relation.sum()),fixed=int(np.sum(state!=0)),
              order_rows=0,conditional_rows=0,nodes=0,rounds=0,prep_s=prep)
    if lower==upper:
        return best,dict(**info,lb=lower,certified=True,status="optimal",seconds=prep)
    model,x=q.setup(len(g),upper)
    model.addCons(quicksum(x)>=lower)
    for j in np.flatnonzero(state==1): model.chgVarLb(x[j],1.)
    for j in np.flatnonzero(state==-1): model.chgVarUb(x[j],0.)
    free=np.flatnonzero(state==0)
    rel=relation[np.ix_(free,free)]
    # Remove arcs implied by paths of at least two arcs.
    cover=rel & ~(rel@rel)
    info["order_rows"]=int(cover.sum())
    for i,j in zip(*cover.nonzero()): model.addCons(x[free[j]]<=x[free[i]])
    witness=[]; widths=[]
    if hot:
        witness,widths,r=conditional_rows(model,x,g,d,state,relation,lower,upper,rule=="hot_range")
        info["conditional_rows"]=r
    def encode(z): return [(v,float(k==z.sum())) for k,v in zip(widths,witness)]
    if backend=="cone":
        eig,v=np.linalg.eigh(g)
        root=np.sqrt(np.maximum(eig,0.))[:,None]*v.T
        aux=[]
        for r in root:
            t=model.addVar(lb=float(np.minimum(r,0).sum()-1e-10),ub=float(np.maximum(r,0).sum()+1e-10))
            model.addCons(t==float(r.sum())-quicksum(float(s)*a for s,a in zip(r,x)))
            aux.append(t)
        model.addCons(quicksum(t*t for t in aux)<=rho*rho)
        q.start_solution(model,x,best,[*zip(aux,root@(~best)),*encode(best)])
        z,lb,status=q.opt(model,x,cutoff)
        lower=max(lower,lb); info["rounds"]=1
        info["nodes"]=int(model.getNNodes()) if model.getStage()>=9 else 0
        if z is not None and q.feasible(g,z,rho) and z.sum()<best.sum(): best=z
    elif backend=="ballot":
        cuts=0; status="timelimit"
        while perf_counter()<cutoff and lower<best.sum():
            for weights,quota in zip(w[cuts:],rhs[cuts:]):
                model.addCons(quicksum(float(weights[j])*x[j] for j in np.flatnonzero(weights))>=float(quota))
            cuts=len(w)
            q.start_solution(model,x,best,encode(best))
            z,lb,status=q.opt(model,x,cutoff)
            info["rounds"]+=1
            info["nodes"]+=int(model.getNNodes()) if model.getStage()>=9 else 0
            lower=max(lower,lb)
            if z is None: break
            if q.feasible(g,z,rho) and z.sum()<best.sum(): best=z
            if lower>=best.sum() or status!="optimal" or perf_counter()>=cutoff: break
            if q.feasible(g,z,rho): raise ArithmeticError("Feasible optimal master did not close gap")
            weights,quota=q.ballots(g,[(~z).astype(float)],rho)
            assert (weights@z < quota-1e-11).all()
            w=np.vstack([w,weights]); rhs=np.r_[rhs,quota]
            model.freeTransform()
            next_state=propagate(w,rhs,lower,int(best.sum()),relation)
            for j in np.flatnonzero((next_state!=0)&(state==0)):
                (model.chgVarLb(x[j],1.) if next_state[j]==1 else model.chgVarUb(x[j],0.))
            assert not ((next_state==1)&~best).any() and not ((next_state==-1)&best).any()
            state=next_state
            lower=max(lower,int(np.sum(state==1)))
    else: raise ValueError(backend)
    assert lower<=best.sum() and q.feasible(g,best,rho)
    return best,dict(**info,lb=lower,certified=bool(lower==best.sum()),
                     status="optimal" if lower==best.sum() else status,seconds=perf_counter()-tic)
