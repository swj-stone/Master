"""Exact signed-ballot bounds for a prescribed error budget on an additive model."""
from time import perf_counter
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numba import njit


@njit(cache=True)
def _project(x,capacity):
    z=np.minimum(1.,np.maximum(0.,x))
    if z.sum()<=capacity:
        return z
    lo=x.min()-1.
    hi=x.max()
    for _ in range(36):
        mid=(lo+hi)/2
        if np.minimum(1.,np.maximum(0.,x-mid)).sum()>capacity:
            lo=mid
        else:
            hi=mid
    return np.minimum(1.,np.maximum(0.,x-hi))


@njit(cache=True)
def _probe(g,state,k,rho,lipschitz,steps,seed_z,roundoff=0.):
    """Return a separating projection, or a primal suggestion for branching."""
    free=np.flatnonzero(state==0)
    known=state==1
    slots=k-known.sum()
    z=known.astype(np.float64)
    if slots<0:
        return True,z,0,z
    z[free]=_project(seed_z[free],slots)
    y=z.copy()
    momentum=1.
    for it in range(steps):
        v=1-y
        w=g @ v
        # Every direction (including extrapolated iterates) gives a valid ballot.
        if it%4==0 or it==steps-1:
            amplitude=max(1.,np.abs(v).max())
            norm=np.sqrt(max(0.,v@w+roundoff*amplitude**2))
            extreme=w[known].sum()
            if slots:
                weights=np.sort(np.maximum(0.,w[free]))
                extreme+=weights[max(0,len(free)-slots):].sum()
            # g was divided by sum(g). The tolerance scales with ballot weights.
            tolerance=1e-10*(1.+np.abs(w).sum())+roundoff*amplitude
            if w.sum()-extreme>rho*np.sqrt(1.+roundoff)*norm+tolerance:
                return True,z,it+1,v
        new=z.copy()
        new[free]=_project(y[free]+w[free]/lipschitz,slots)
        m=(1+np.sqrt(1+4*momentum*momentum))/2
        y=new+(momentum-1)/m*(new-z)
        momentum=m
        z=new
    return False,z,steps,1-z


def propagate(g,state,witness,k,rho,roundoff=0.):
    """One shared direction tests both assignments of every still-free feature."""
    fixed=state==1
    free=np.flatnonzero(state==0)
    slots=k-fixed.sum()
    if not len(free):
        return
    if slots==0:
        state[free]=-1
        return
    w=g@witness
    amplitude=max(1.,np.abs(witness).max())
    threshold=w.sum()-rho*np.sqrt(1+roundoff)*np.sqrt(max(0.,witness@w+roundoff*amplitude**2))
    guard=1e-10*(1.+np.abs(w).sum())+roundoff*amplitude
    known=w[fixed].sum()
    weights=w[free]
    positive=np.maximum(weights,0.)
    ranked=np.sort(positive)[::-1]
    def best_without(capacity):
        t=min(len(free),capacity)
        next_score=ranked[t] if t<len(free) else 0.
        return ranked[:t].sum()-np.maximum(0.,positive-next_score)
    requires=known+best_without(slots)<threshold-guard
    forbids=known+weights+best_without(slots-1)<threshold-guard
    if np.any(requires&forbids):
        raise ArithmeticError("Contradictory requirements at a known attainable budget")
    state[free[requires]]=1
    state[free[forbids]]=-1


def classify(g,initial,rho,L,steps=32,roundoff=0.,deadline=float("inf")):
    """Generate conditional separating directions; reuse every direction across columns."""
    p=len(g)
    k=int(initial.sum())
    state=np.zeros(p,np.int8)
    used=0
    total_steps=0
    seed_z=initial.astype(float)
    # Single-component voters can settle obvious assignments without a relaxation.
    for v in (np.ones(p),1-seed_z):
        propagate(g,state,v,k,rho,roundoff)
    for j in np.argsort(-np.diag(g),kind="stable"):
        if perf_counter()>=deadline:
            break
        if state[j]!=0:
            continue
        counter=state.copy()
        counter[j]=-1 if initial[j] else 1
        if (counter==1).sum()>k:
            state[j]=-1
            continue
        rejected,_,n,v=_probe(g,counter,k,rho,L,steps,seed_z,roundoff)
        used+=1
        total_steps+=n
        propagate(g,state,v,k,rho,roundoff)
        if rejected:
            state[j]=1 if initial[j] else -1
    if np.any((state==1)&~initial) or np.any((state==-1)&initial):
        raise ArithmeticError("A ballot excluded the verified incumbent")
    return state,used,total_steps


def gram_loss(g,mask):
    omitted=~mask
    return float(g[np.ix_(omitted,omitted)].sum())


@dataclass
class FidelityGame:
    """Keep physical contributions for an independent boundary check."""
    factor: np.ndarray
    gram: np.ndarray
    normalizer: float
    roundoff: float
    original: np.ndarray

    @classmethod
    def from_factor(cls, factor):
        a=np.asarray(factor,dtype=np.float64)
        if a.ndim!=2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all():
            raise ValueError("Expected a finite matrix of output-coordinate contributions")
        original=a
        maximum=np.max(np.abs(a))
        if maximum:
            _,exponent=np.frexp(maximum)
            a=np.ldexp(a,-int(exponent))
        full=a.sum(axis=1)
        normalizer=float(full@full)
        size=np.abs(a).sum(axis=1)
        total=float(size@size)
        scale=normalizer if normalizer>0 else (total if total else 1.)
        g=(a.T@a)/scale
        n,p=a.shape
        # Encloses dot, row-sum, normalization and subsequent reductions under binary64.
        # When cancellation is severe this deliberately disables inconclusive Gram tests.
        guard=128*np.finfo(float).eps*(n+2*p+32)*(1.+total/scale)
        if not np.isfinite(g).all() or not np.isfinite(guard):
            raise ValueError("Rescale physical contributions before forming the Gram matrix")
        return cls(a,g,normalizer,guard,original)

    def feasible(self,mask,rho):
        a=self.factor
        if np.all(mask):
            return True
        rem=a[:,~mask].sum(axis=1)
        diff=float(rem@rem)-rho*rho*self.normalizer
        if self.normalizer>0 and abs(diff)>self.roundoff*(1.+rho*rho)*self.normalizer:
            return diff<=0
        # Exact rational arithmetic is used only where the two float intervals overlap.
        slack=Fraction(str(rho))**2
        full_sq=rem_sq=Fraction(0)
        for row in self.original:
            values=[Fraction(float(v)) for v in row]
            full_sq+=sum(values)**2
            rem_sq+=sum(v for v,keep in zip(values,mask) if not keep)**2
        return rem_sq<=slack*full_sq


def improve_upper(g,mask,rho):
    """Feasible deletions and improving swaps, each measured on intact teacher outputs."""
    mask=mask.copy()
    while mask.any():
        ins,out=np.flatnonzero(mask),np.flatnonzero(~mask)
        v=g[:,out].sum(axis=1)
        now=v[out].sum()
        costs=2*v+np.diag(g)
        j=ins[np.argmin(costs[ins])]
        if now+costs[j]<=rho*rho:
            mask[j]=False
            continue
        if not len(out):
            break
        gap=costs[ins,None]+(np.diag(g)-2*v)[out]-2*g[np.ix_(ins,out)]
        i,j=np.unravel_index(gap.argmin(),gap.shape)
        if gap[i,j]>=-1e-12:
            break
        mask[ins[i]],mask[out[j]]=False,True
    return mask


@njit(cache=True)
def _bank_reject(weights,quotas,orders,n,state,k):
    for h in range(n):
        max_score=0.
        slots=k
        for j in range(len(state)):
            if state[j]==1:
                max_score+=weights[h,j]
                slots-=1
        for j in orders[h]:
            if slots==0 or weights[h,j]<=0:
                break
            if state[j]==0:
                max_score+=weights[h,j]
                slots-=1
        if max_score<quotas[h]:
            return True
    return False


def directional_search(game,initial,rho=.2,*,max_seconds=3.,steps=48,shared=False,cache=False):
    """Seek a smaller verified feasible support, stopping only with a bound or a budget."""
    if not isinstance(game,FidelityGame):
        raise ValueError("Supply the original contributions using FidelityGame.from_factor")
    g=game.gram
    if rho<0 or not np.isfinite(rho) or steps<1 or max_seconds<0:
        raise ValueError("Invalid tolerance, iteration count or time budget")
    start=perf_counter()
    p=len(g)
    best=np.asarray(initial,bool).copy()
    if best.shape!=(p,) or not game.feasible(best,rho):
        raise ValueError("Initial support does not satisfy fidelity")
    # Gershgorin gives a safe gradient step bound; no eigenproblem is needed.
    L=max(np.max(np.abs(g).sum(axis=1)),1e-16)
    stop=start+max_seconds
    if shared and perf_counter()<stop:
        root,probes,checks=classify(g,best,rho,L,roundoff=game.roundoff,deadline=stop)
    else:
        root,probes,checks=np.zeros(p,np.int8),0,0
    lower=int((root==1).sum())
    visits=knocks=0
    limit=min(32,p)
    weights=np.zeros((limit,p))
    quotas=np.zeros(limit)
    orders=np.zeros((limit,p),np.int64)
    written=hitrate=0
    # An initial bisection proves a valid cardinality floor even for timed-out jobs.
    hi=int(best.sum())
    while lower<hi and perf_counter()<stop:
        k=(lower+hi)//2
        rejected,_,budget,_=_probe(g,root,k,rho,L,steps,best.astype(float),game.roundoff)
        checks+=budget
        if rejected:
            lower=k+1
        else:
            hi=k
    while lower<int(best.sum()) and perf_counter()-start<max_seconds:
        k=int(best.sum())-1
        stack=[(root,best.astype(float))]
        found=False
        while stack and perf_counter()-start<max_seconds:
            state,warm=stack.pop()
            visits+=1
            if cache and written and _bank_reject(weights,quotas,orders,min(limit,written),state,k):
                hitrate+=1
                continue
            rejected,z,budget,v=_probe(g,state,k,rho,L,steps,warm,game.roundoff)
            checks+=budget
            if rejected:
                knocks+=1
                if cache:
                    anchor=written%limit
                    w=g@v
                    weights[anchor]=w
                    amp=max(1.,np.abs(v).max())
                    norm=np.sqrt(max(0.,v@w+game.roundoff*amp**2))
                    quotas[anchor]=(w.sum()-rho*np.sqrt(1.+game.roundoff)*norm
                                    -1e-10*(1.+np.abs(w).sum())-game.roundoff*amp)
                    orders[anchor]=np.argsort(-w,kind="stable")
                    written+=1
                continue
            free=np.flatnonzero(state==0)
            fixed=state==1
            slots=k-fixed.sum()
            order=free[np.argsort(-z[free],kind="stable")]
            trial=fixed.copy()
            trial[order[:slots]]=True
            if game.feasible(trial,rho):
                improved=improve_upper(g,trial,rho)
                best=improved if game.feasible(improved,rho) else trial
                found=True
                break
            if slots==0 or not len(free):
                continue
            # Branch on an ambiguous coordinate; both children preserve the fixed assignments.
            j=free[np.argmin(abs(z[free]-.5))]
            branch=1 if z[j]>=.5 else -1
            for sign in (-branch,branch):
                child=state.copy()
                child[j]=sign
                stack.append((child,z))
        if not found:
            if not stack:
                lower=k+1
            break
    return best,dict(lb=int(lower),certified=bool(lower==best.sum()),nodes=visits,cut_nodes=knocks,
                     projection_steps=checks,feature_probes=probes,forced_in=int((root==1).sum()),
                     forced_out=int((root==-1).sum()),cached_rejections=hitrate,solver_s=perf_counter()-start)
