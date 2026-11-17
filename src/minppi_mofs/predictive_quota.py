"""Coalition-size votes on expected prediction loss of a compressed forest."""
from dataclasses import dataclass
import numpy as np
from numba import njit

from .availability_minpi import AvailabilityGame


@njit(cache=True)
def _both_features(g,h,ip,ii):
    out=np.empty(ip[g+1]-ip[g]+ip[h+1]-ip[h],np.int64)
    a,b=ip[g],ip[h]
    n=0
    while a<ip[g+1] or b<ip[h+1]:
        if b==ip[h+1] or (a<ip[g+1] and ii[a]<ii[b]):
            out[n]=ii[a]
            a+=1
        elif a==ip[g+1] or ii[b]<ii[a]:
            out[n]=ii[b]
            b+=1
        else:
            out[n]=ii[a]
            a+=1
            b+=1
        n+=1
    return out[:n]


@njit(cache=True)
def _start(p,width,linear,ptr,index,weight,ip,ii,unions):
    delta=np.zeros((p,width))
    for g in range(len(ip)-1):
        q=ip[g+1]-ip[g]
        for f in ii[ip[g]:ip[g+1]]:
            delta[f,q]+=linear[g]
        for pos in range(ptr[g],ptr[g+1]):
            h=index[pos]
            if h<g:
                continue
            magnitude=weight[pos] if h==g else 2*weight[pos]
            for f in _both_features(g,h,ip,ii):
                delta[f,unions[pos]]-=magnitude
    return delta


@njit(cache=True)
def _remove(j,alive,delta,linear,ptr,index,weights,ip,ii,jptr,jgroups,unions):
    for g in jgroups[jptr[j]:jptr[j+1]]:
        if not alive[g]:
            continue
        q=ip[g+1]-ip[g]
        for f in ii[ip[g]:ip[g+1]]:
            delta[f,q]-=linear[g]
        for pos in range(ptr[g],ptr[g+1]):
            h=index[pos]
            if alive[h]:
                magnitude=weights[pos] if h==g else 2*weights[pos]
                for f in _both_features(g,h,ip,ii):
                    delta[f,unions[pos]]+=magnitude
        alive[g]=False


@njit(cache=True)
def _path_histogram(order,linear,ptr,index,weights,ip,ii,unions,width):
    """Aggregate each loss term at the first deleted input that disables it."""
    p=len(order)
    ranks=np.empty(p,np.int64)
    for k,j in enumerate(order):
        ranks[j]=k+1
    first=np.empty(len(ip)-1,np.int64)
    hist=np.zeros((p+1,width))
    for g in range(len(first)):
        earliest=p
        for pos in range(ip[g],ip[g+1]):
            earliest=min(earliest,ranks[ii[pos]])
        first[g]=earliest
        hist[first[g],ip[g+1]-ip[g]]+=linear[g]
    for g in range(len(first)):
        for z in range(ptr[g],ptr[g+1]):
            h=index[z]
            if h>=g:
                amount=weights[z] if h==g else 2.*weights[z]
                hist[min(first[g],first[h]),unions[z]]-=amount
    return hist


@njit(cache=True)
def _least_worst(current,delta,keep,w,tol,last_size):
    """Visit the previous worst size and the endpoints, then short-circuit on a veto."""
    sizes=len(w)
    checks=0
    best=np.inf
    values=np.full(len(keep),np.inf)
    worst_sizes=np.zeros(len(keep),np.int64)
    for j in range(len(keep)):
        if not keep[j]:
            continue
        worst=-np.inf
        worst_size=0
        for z in range(sizes+3):
            s=last_size if z==0 else sizes-1 if z==1 else 0 if z==2 else z-3
            if (z==1 and s==last_size) or (z==2 and (s==last_size or s==sizes-1)):
                continue
            if z>=3 and (s==last_size or s==sizes-1 or s==0):
                continue
            val=0.
            for h in range(len(current)):
                val+=(current[h]+delta[j,h])*w[s,h]
            checks+=1
            if val>worst:
                worst,worst_size=val,s
            if worst>best+tol:
                break
        if worst<=best+tol:
            values[j],worst_sizes[j]=worst,worst_size
            best=min(best,worst)
    for j in range(len(keep)):
        if keep[j] and values[j]<=best+tol:
            return j,worst_sizes[j],checks
    raise ValueError("No finite candidate vote")


@dataclass
class PredictiveGame:
    structure: AvailabilityGame
    linear: np.ndarray
    null: float
    initial: np.ndarray

    @classmethod
    def from_paths(cls,a,incidence,target):
        structure=AvailabilityGame.from_paths(a,incidence)
        target=np.ravel(np.asarray(target,dtype=np.float64))
        if len(target)!=a.shape[0] or not np.isfinite(target).all():
            raise ValueError("Target must match the finite calibration scores")
        linear=2*np.asarray(a.T@target).ravel()
        g,inc=structure.gram,structure.incidence
        initial=_start(inc.shape[1],len(structure.totals),linear,g.indptr,g.indices,g.data,
                       inc.indptr,inc.indices,structure.unions)
        return cls(structure,linear,np.dot(target,target),initial)

    def coefficients(self,keep):
        """Excess risk over the full-input forest facing the same availability mask."""
        st=self.structure
        inc,g=st.incidence,st.gram
        dead=np.asarray(inc@np.logical_not(keep).astype(int)).ravel()>0
        linear=np.bincount(np.diff(inc.indptr)[dead],weights=self.linear[dead],minlength=len(st.totals))
        changed=dead.repeat(np.diff(g.indptr))|dead[g.indices]
        quadratic=np.bincount(st.unions[changed],weights=g.data[changed],minlength=len(st.totals))
        return linear-quadratic

    def conditional_risk(self,keep):
        return self.structure.weights@self.coefficients(keep)

    def path_curves(self,order):
        """Rows d=0..p give unnormalized proper-score excess after d fixed deletions."""
        st=self.structure
        inc,g=st.incidence,st.gram
        order=np.asarray(order,dtype=int)
        if not np.array_equal(np.sort(order),np.arange(inc.shape[1])):
            raise ValueError("Need the entire deletion order")
        hist=_path_histogram(order,self.linear,g.indptr,g.indices,g.data,inc.indptr,inc.indices,
                             st.unions,len(st.totals))
        return np.cumsum(hist,axis=0)@st.weights.T

    def ranked_path(self,rule="min",*,screen=True):
        """Trace the greedy order down to zero, without looking at the acceptance sample."""
        st=self.structure
        inc,g=st.incidence,st.gram
        by=st.by_feature
        p=inc.shape[1]
        w=st.weights[1:]*p/np.arange(1,p+1)[:,None]
        if rule=="pooled":
            w=w.mean(axis=0,keepdims=True)
        elif rule=="intact":
            w=w[-1:]
        elif rule!="min":
            raise ValueError("Unknown vote")
        tol=128*np.finfo(float).eps*(np.abs(self.linear).sum()+np.abs(g.data).sum()+self.null)
        current=np.zeros(len(st.totals))
        keep=np.ones(p,bool)
        alive=np.ones(inc.shape[0],bool)
        delta=self.initial.copy()
        order=[]
        checks=0
        last=len(w)-1
        while keep.any():
            if screen:
                j,last,count=_least_worst(current,delta,keep,w,tol,last)
                checks+=count
            else:
                ids=np.flatnonzero(keep)
                loss=((current+delta[ids])@w.T).max(axis=1)
                j=ids[np.flatnonzero(loss<=loss.min()+tol)[0]]
                checks+=len(w)*len(ids)
            current+=delta[j]
            _remove(j,alive,delta,self.linear,g.indptr,g.indices,g.data,inc.indptr,
                    inc.indices,by.indptr,by.indices,st.unions)
            keep[j]=False
            order.append(int(j))
        return np.array(order),checks

    def search(self,epsilon=.02,rule="min",order=None,k=None,*,recount=False):
        """Bound loss per availability rate. k overrides stopping only for equal-width comparisons."""
        st=self.structure
        inc,g=st.incidence,st.gram
        p=inc.shape[1]
        if epsilon<0 or rule not in {"min","pooled","intact"} or (k is not None and (k<0 or k>p)):
            raise ValueError("Invalid risk budget, rule or width")
        w=st.weights[1:]*p/np.arange(1,p+1)[:,None]
        if rule=="intact":
            w=w[-1:]
        elif rule=="pooled":
            w=w.mean(axis=0,keepdims=True)
        budget=epsilon*self.null
        tol=128*np.finfo(float).eps*(np.abs(self.linear).sum()+np.abs(g.data).sum()+self.null)
        alive,keep=np.ones(inc.shape[0],bool),np.ones(p,bool)
        idx=np.arange(p)
        if order is not None and not np.array_equal(np.sort(order),idx):
            raise ValueError("Ranking must be a permutation")
        current=np.zeros(len(st.totals))
        delta=self.initial.copy()
        steps=[]
        while keep.any() and (k is None or keep.sum()>k):
            candidates=idx[keep] if order is None else order[keep[order]]
            before=len(steps)
            if order is None:
                cols=np.stack([self.coefficients(keep&(idx!=j)) for j in candidates]) if recount else current+delta[candidates]
                worst=(cols@w.T).max(axis=1)
                best=worst.min()
                if k is None and best>budget+tol:
                    break
                candidates=candidates[np.flatnonzero(worst<=best+tol)[:1]]
            for j in candidates:
                proposed=self.coefficients(keep&(idx!=j)) if recount else current+delta[j]
                if k is None and (w@proposed).max()>budget+tol:
                    continue
                keep[j]=False
                steps.append(int(j))
                current=proposed.copy()
                if not recount:
                    by=st.by_feature
                    _remove(j,alive,delta,self.linear,g.indptr,g.indices,g.data,inc.indptr,
                            inc.indices,by.indptr,by.indices,st.unions)
                if k is not None and keep.sum()==k:
                    break
            if len(steps)==before:
                break
        direct=self.coefficients(keep)
        np.testing.assert_allclose(current,direct,rtol=1e-8,atol=max(tol*10,np.finfo(float).tiny))
        return np.flatnonzero(keep),steps
