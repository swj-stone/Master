"""Matched analytic and permutation controls for frozen-forest predictive risk."""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import sparse

from minppi_mofs.availability_minpi import probabilities


@dataclass
class Terms:
    incidence: object
    by_feature: object
    mass: np.ndarray
    null: float
    tolerance: float

    @classmethod
    def build(cls, a, membership, target):
        """Aggregate the linear and quadratic terms with the same required feature set."""
        a = a.astype(np.float64, copy=False)
        target = np.asarray(target, float)
        inc = membership.tocsr()
        p = inc.shape[1]
        packed = np.packbits(inc.toarray().astype(bool), axis=1, bitorder="little")
        # Pad byte keys to whole words. np.unique handles a 1-D key per union.
        packed = np.pad(packed, ((0, 0), (0, -packed.shape[1] % 8)))
        gram = sparse.triu(a.T @ a, format="coo")
        pair_mass = gram.data * np.where(gram.row == gram.col, -1., -2.)
        linear = 2 * np.asarray(a.T @ target).ravel()
        joint = np.concatenate((packed, packed[gram.row] | packed[gram.col]))
        uid = np.ascontiguousarray(joint).view(np.dtype((np.void, joint.shape[1]))).ravel()
        unique, index = np.unique(uid, return_inverse=True)
        mass = np.bincount(index, weights=np.r_[linear, pair_mass])
        selected = mass != 0.
        keys = unique[selected].view(np.uint8).reshape(-1, packed.shape[1])
        incidence = sparse.csr_matrix(np.unpackbits(keys, axis=1, count=p, bitorder="little"))
        norm = float(target @ target)
        tol = 128*np.finfo(float).eps*(np.abs(linear).sum()+np.abs(pair_mass).sum()+norm)
        return cls(incidence,incidence.tocsc(),mass[selected],norm,tol)

    @property
    def p(self):
        return self.incidence.shape[1]


def antithetic_ranks(p, count, rng):
    """Return one-based ranks, pairing independent orders with their reversals."""
    if count % 2:
        raise ValueError("The reversed-permutation control uses an even count")
    ranks = np.empty((count,p), np.int32)
    for i in range(0, count, 2):
        ranks[i] = rng.permutation(p) + 1
        ranks[i+1] = p + 1 - ranks[i]
    return ranks


@njit(cache=True)
def _histogram(ptr, ids, ranks, mass):
    p = ranks.shape[1]
    hist = np.zeros((len(mass),p+1),np.float64)
    for u in range(len(mass)):
        for m in range(len(ranks)):
            highest = 0
            for pos in range(ptr[u],ptr[u+1]):
                highest = max(highest, ranks[m,ids[pos]])
            hist[u,highest] += 1.
        hist[u] *= mass[u]/len(ranks)
    return hist


@njit(cache=True)
def _paired_sparse_histogram(ptr, ids, ranks, mass):
    """Compute forward/reverse activation ranks in one traversal and store occupied bins."""
    p = ranks.shape[1]
    capacity = len(mass)*min(p,len(ranks))
    offsets = np.zeros(len(mass)+1,np.int32)
    sizes = np.empty(capacity,np.int32)
    effects = np.empty(capacity)
    counts = np.zeros(p+1,np.int32)
    touched = np.empty(p+1,np.int32)
    cursor=0
    for u in range(len(mass)):
        n=0
        for m in range(0,len(ranks),2):
            lo,hi=p,0
            for pos in range(ptr[u],ptr[u+1]):
                rank=ranks[m,ids[pos]]
                lo,hi=min(lo,rank),max(hi,rank)
            for rank in (hi,p+1-lo):
                if counts[rank]==0:
                    touched[n]=rank
                    n+=1
                counts[rank]+=1
        for i in range(n):
            rank=touched[i]
            sizes[cursor]=rank
            effects[cursor]=mass[u]*counts[rank]/len(ranks)
            cursor+=1
            counts[rank]=0
        offsets[u+1]=cursor
    return offsets,sizes[:cursor],effects[:cursor]


@njit(cache=True)
def _scatter(u, direction, ptr, ids, degree, mass, hist, hp, hi, hv, target):
    for pos in range(ptr[u],ptr[u+1]):
        j = ids[pos]
        if len(hp):
            for q in range(hp[u],hp[u+1]):
                target[j,hi[q]] += direction*hv[q]
        elif hist.shape[1]:
            for size in range(1,hist.shape[1]):
                target[j,size] += direction*hist[u,size]
        else:
            target[j,degree[u]] += direction*mass[u]


@njit(cache=True)
def _initialize(active, ptr, ids, degree, mass, hist, hp, hi, hv, delta):
    delta[:] = 0.
    for u in range(len(mass)):
        if active[u]:
            _scatter(u,1.,ptr,ids,degree,mass,hist,hp,hi,hv,delta)


@njit(cache=True)
def _remove(j, update, active, ptr, ids, fp, terms, degree, mass, hist, hp, hi, hv, delta):
    for i in range(fp[j],fp[j+1]):
        u = terms[i]
        if active[u]:
            active[u] = False
            if update:
                _scatter(u,-1.,ptr,ids,degree,mass,hist,hp,hi,hv,delta)


class MatchedSearch:
    def __init__(self, terms, ranks=None, *, sparse_mc=False):
        self.terms = terms
        inc = terms.incidence
        self.degree = np.diff(inc.indptr)
        self.sampled = ranks is not None
        self.normalizer = terms.p/np.arange(1,terms.p+1)
        self.weights = probabilities(terms.p, int(self.degree.max(initial=0)))[1:].T
        self.hist = np.empty((0,0))
        self.hptr = self.hindex = np.empty(0,np.int32)
        self.hval = np.empty(0)
        self.sparse_hist = None
        if self.sampled and sparse_mc:
            if len(ranks)%2 or not np.array_equal(ranks[::2],1+terms.p-ranks[1::2]):
                raise ValueError("Sparse estimator requires antithetic paired ranks")
            self.hptr,self.hindex,self.hval=_paired_sparse_histogram(
                inc.indptr,inc.indices,ranks,terms.mass)
            self.sparse_hist=sparse.csr_matrix((self.hval,self.hindex,self.hptr),
                                               shape=(len(terms.mass),terms.p+1))
        elif self.sampled:
            self.hist=_histogram(inc.indptr,inc.indices,ranks,terms.mass)

    def _threshold_totals(self,lost):
        hist=self.sparse_hist if self.sparse_hist is not None else self.hist
        return np.asarray(hist[lost].sum(axis=0)).ravel()

    def curve(self, cols):
        """Physical excess over the uncompressed masked teacher, per size and null response."""
        t = self.terms
        missing = np.ones(t.p,np.int32)
        missing[cols] = 0
        lost = np.ravel(t.incidence@missing)>0
        if self.sampled:
            raw = np.cumsum(self._threshold_totals(lost))[1:]
        else:
            by_degree = np.bincount(self.degree[lost], weights=t.mass[lost],
                                    minlength=self.weights.shape[0])
            raw = by_degree @ self.weights
        return raw*self.normalizer/max(t.null,1e-100)

    def run(self, epsilon=.02, recount=False):
        t = self.terms
        inc, by = t.incidence,t.by_feature
        args = inc.indptr, inc.indices, self.degree, t.mass, self.hist, self.hptr,self.hindex,self.hval
        keep = np.ones(t.p,bool)
        active = np.ones(len(t.mass),bool)
        width = t.p+1 if self.sampled else self.weights.shape[0]
        delta = np.zeros((t.p,width))
        current = np.zeros(width)
        path = []
        _initialize(active,*args,delta)
        while keep.any():
            if recount and path:
                _initialize(active,*args,delta)
                lost = ~active
                current = (self._threshold_totals(lost) if self.sampled else
                           np.bincount(self.degree[lost],weights=t.mass[lost],
                                       minlength=width).astype(float,copy=False))
            candidates = np.flatnonzero(keep)
            trial = current+delta[candidates]
            raw = (np.cumsum(trial,axis=1)[:,1:] if self.sampled else trial@self.weights)
            worst = (raw*self.normalizer).max(axis=1)
            value = worst.min()
            if value > epsilon*t.null+t.tolerance:
                break
            j = candidates[np.flatnonzero(worst<=value+t.tolerance)[0]]
            path.append(int(j))
            current += delta[j]
            keep[j] = False
            _remove(j,not recount,active,inc.indptr,inc.indices,by.indptr,by.indices,
                    self.degree,t.mass,self.hist,self.hptr,self.hindex,self.hval,delta)
        return np.flatnonzero(keep),path


def direct_mask_curve(a, incidence, y, keep, ranks):
    """Compute empirical risk by direct nested-mask predictions, without a Gram or union map."""
    p = incidence.shape[1]
    absent = np.ones(p,int)
    absent[keep] = 0
    available = np.ravel(incidence@absent)==0
    val = np.zeros(p)
    rows = np.arange(incidence.shape[0])
    for rank in ranks:
        last = np.maximum.reduceat(rank[incidence.indices],incidence.indptr[:-1])
        shape = (len(rows),p+1)
        steps = sparse.csr_matrix((np.ones(len(rows)),(rows,last)),shape=shape)
        full = np.asarray((a@steps).toarray()).cumsum(axis=1)[:,1:]
        restricted = np.asarray((a@steps.multiply(available[:,None])).toarray()).cumsum(axis=1)[:,1:]
        val += ((y[:,None]-restricted)**2-(y[:,None]-full)**2).sum(axis=0)
    return val*p/np.arange(1,p+1)/len(ranks)/max(float(y@y),1e-100)
