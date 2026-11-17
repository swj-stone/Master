"""Coalition-size fidelity votes for feature compression under missing inputs.

Uses the path-increment matrix of a frozen truncated ensemble. All contexts
of each size have equal weight. Each pair of groups changes state at most
twice in a deletion path.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit


@njit(cache=True)
def _overlap(group,other,ip,ii):
    a,b=ip[group],ip[other]
    count=0
    while a<ip[group+1] and b<ip[other+1]:
        if ii[a]==ii[b]:
            count+=1
            a+=1
            b+=1
        elif ii[a]<ii[b]:
            a+=1
        else:
            b+=1
    return count


@njit(cache=True)
def _prepare(gp,gi,gv,ip,ii,p,width):
    size=np.empty(len(gv),np.int16)
    totals=np.zeros(width+1)
    change=np.zeros((p,width+1))
    for g in range(len(ip)-1):
        for edge in range(gp[g],gp[g+1]):
            h=gi[edge]
            k=ip[g+1]-ip[g]+ip[h+1]-ip[h]-_overlap(g,h,ip,ii)
            size[edge]=k
            if h<g:
                continue
            weight=gv[edge] if h==g else 2.*gv[edge]
            totals[k]+=weight
            a,b=ip[g],ip[h]
            while a<ip[g+1] and b<ip[h+1]:
                if ii[a]==ii[b]:
                    change[ii[a],k]+=weight
                    a+=1
                    b+=1
                elif ii[a]<ii[b]:
                    a+=1
                else:
                    b+=1
    return size,totals,change


@njit(cache=True)
def _drop(j,alive,delta,gp,gi,gv,union,ip,ii,jp,ji):
    for pos in range(jp[j],jp[j+1]):
        g=ji[pos]
        if not alive[g]:
            continue
        alive[g]=False
        for edge in range(gp[g],gp[g+1]):
            h=gi[edge]
            w=gv[edge] if h==g else 2.*gv[edge]
            k=union[edge]
            if alive[h]:
                # The pair was covered only by deleting a common member.
                # With g absent, deleting any member of h now suffices.
                a=ip[g]
                for b in range(ip[h],ip[h+1]):
                    f=ii[b]
                    while a<ip[g+1] and ii[a]<f:
                        a+=1
                    if a==ip[g+1] or ii[a]!=f:
                        delta[f,k]+=w
            else:
                for a in range(ip[g],ip[g+1]):
                    delta[ii[a],k]-=w


def probabilities(p,degree):
    out=np.ones((p+1,degree+1))
    s=np.arange(p+1)
    for q in range(1,degree+1):
        out[:,q]=out[:,q-1]*np.maximum(0,s-q+1)/(p-q+1)
    return out


@dataclass
class AvailabilityGame:
    gram: object
    incidence: object
    by_feature: object
    unions: np.ndarray
    totals: np.ndarray
    initial: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_paths(cls,a,incidence):
        a=a.astype(np.float64,copy=False).tocsc()
        inc=incidence.tocsr(copy=False)
        p=inc.shape[1]
        if a.shape[1]!=inc.shape[0] or p<1 or not np.all(inc.data==1) or not inc.has_canonical_format:
            raise ValueError("Expected a canonical binary path dictionary")
        if not np.isfinite(a.data).all():
            raise ValueError("Nonfinite path increment")
        g=(a.T@a).tocsr()
        g.sort_indices()
        width=min(p,2*int(np.diff(inc.indptr).max(initial=0)))
        union,total,delta=_prepare(g.indptr,g.indices,g.data,inc.indptr,inc.indices,p,width)
        return cls(g,inc,inc.tocsc(),union,total,delta,probabilities(p,width))

    def coefficients(self,keep):
        """Direct recount from surviving groups, independent of the deletion recurrence."""
        alive=np.asarray(self.incidence@np.logical_not(keep).astype(int)).ravel()==0
        g=self.gram
        m=(~alive).repeat(np.diff(g.indptr)) & ~alive[g.indices]
        return np.bincount(self.unions[m],weights=g.data[m],minlength=len(self.totals))

    def conditional(self,coeff):
        return self.weights@coeff

    def search(self,rho=.2,*,rule="min",order=None,algorithm="incremental"):
        """Use worst, uniformly pooled or intact-only normalized fidelity; preserve the same quota."""
        if rho<0 or not np.isfinite(rho) or rule not in {"min","pooled","intact"}:
            raise ValueError("Invalid quota or vote")
        if algorithm not in {"incremental","recount"}:
            raise ValueError("Unknown search")
        inc,g=self.incidence,self.gram
        p=inc.shape[1]
        alive,keep=np.ones(inc.shape[0],bool),np.ones(p,bool)
        delta=self.initial.copy()
        current=np.zeros_like(self.totals)
        denominator=self.conditional(self.totals)
        # The zero-column coalition has identically zero discrepancy. A positive
        # error at a zero-energy size is compared on the common numerical scale.
        eps=64*np.finfo(float).eps*max(1,g.nnz)*np.sum(abs(g.data))
        nonzero=denominator>eps
        if np.any(denominator< -eps):
            raise ArithmeticError("Negative reference energy")
        norm_weights=self.weights/np.where(nonzero,denominator,1.)[:,None]
        if rule=="intact":
            score_weights=norm_weights[[-1]]
        elif rule=="pooled":
            score_weights=norm_weights[nonzero].mean(axis=0,keepdims=True) if nonzero.any() else norm_weights[[-1]]
        else:
            score_weights=norm_weights[nonzero] if nonzero.any() else norm_weights[[-1]]
        if order is not None:
            order=np.asarray(order,dtype=int)
            if not np.array_equal(np.sort(order),np.arange(p)):
                raise ValueError("Expected a permutation of features")
        steps=[]
        attempts=0
        while keep.any():
            candidates=np.flatnonzero(keep) if order is None else order[keep[order]]
            before=len(steps)
            if order is None:
                if algorithm=="recount":
                    coeff=np.stack([self.coefficients(keep & (np.arange(p)!=j)) for j in candidates])
                else:
                    coeff=current+delta[candidates]
                scores=(coeff@score_weights.T).max(axis=1)
                if rule=="min" and (~nonzero).any():
                    scores[np.any((coeff@self.weights[~nonzero].T)>eps,axis=1)]=np.inf
                # Costs equal to working precision use the smallest input ID.
                value=scores.min()
                eligible=np.flatnonzero(scores<=value+2e-12)
                if value>rho*rho+2e-12 or not len(eligible):
                    break
                candidates=candidates[eligible[:1]]
            for j in candidates:
                proposed=(self.coefficients(keep & (np.arange(p)!=j)) if algorithm=="recount" else current+delta[j])
                attempts+=1
                if np.max(score_weights@proposed)>rho*rho+2e-12:
                    continue
                if rule=="min" and np.any(self.weights[~nonzero]@proposed>eps):
                    continue
                keep[j]=False
                current=proposed.copy()
                steps.append(int(j))
                if algorithm=="incremental":
                    _drop(j,alive,delta,g.indptr,g.indices,g.data,self.unions,inc.indptr,inc.indices,
                          self.by_feature.indptr,self.by_feature.indices)
            if len(steps)==before:
                break
        direct=self.coefficients(keep)
        np.testing.assert_allclose(current,direct,rtol=1e-8,atol=max(eps,np.finfo(float).tiny))
        risk=self.conditional(direct)
        return np.flatnonzero(keep),dict(path=steps,attempts=attempts,
             conditional_error=np.sqrt(np.maximum(0,np.divide(risk,denominator,out=np.zeros_like(risk),where=nonzero))),
             physical_risk=risk,reference_energy=denominator)
