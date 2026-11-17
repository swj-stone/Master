"""All cardinality votes and their joint span on a depth-limited tree dictionary."""
import numpy as np
from numba import njit

from minppi_mofs.stratified_tree import _run


@njit(cache=True)
def _score(j, delta, revision, state):
    (kind, ap, ai, av, ip, gp, gi, weights, bank, denom, unstable,
     rank, cached_revision, cached_score, winners) = state
    if cached_revision[j] == revision:
        return cached_score[j]
    p, depth, _ = bank.shape
    vectors = bank[j]
    if cached_revision[j] < 0:
        vectors[:] = 0.
        for pos in range(gp[j], gp[j+1]):
            g = gi[pos]
            order = ip[g+1]-ip[g]-1
            for k in range(ap[g], ap[g+1]):
                vectors[order,ai[k]] += av[k]
        if kind == 0:
            gram = vectors @ vectors.T
            absgram = np.abs(gram)
            for s in range(p):
                w = weights[s]
                norm2 = np.dot(w,gram@w)
                # Cancellation in w'Gw is possible when order components oppose.
                abs_scale = np.dot(w,absgram@w)
                unstable[j,s] = abs_scale > 0. and norm2 <= 1e-12*abs_scale
                if unstable[j,s]:
                    h = w@vectors
                    norm2 = np.dot(h,h)
                denom[j,s] = np.sqrt(norm2)
        else:
            spans = vectors
            if kind == 1:
                spans = weights @ vectors
            # Direct SVD of row vectors avoids squaring their condition number.
            _, sv, axes = np.linalg.svd(spans, full_matrices=False)
            threshold = np.finfo(np.float64).eps*max(spans.shape)*sv[0] if len(sv) else 0.
            r = min(depth,int(np.sum(sv > threshold)))
            rank[j] = r
            vectors[:r] = axes[:r]
    if kind == 0:
        moments = vectors @ delta
        best = 0.
        fixed_best = 0.
        winner = 0
        for s in range(p):
            if denom[j,s] == 0.:
                continue
            dot = np.dot(weights[s],moments)
            if unstable[j,s]:
                dot = np.dot(weights[s]@vectors,delta)
            ratio = dot / denom[j,s]
            if s==0 or s==(p-1)//2 or s==p-1:
                fixed_best = max(fixed_best, ratio*ratio)
            if ratio*ratio > best:
                best = ratio*ratio
                winner = s
        winners[0 if winner==0 else 3 if winner==p-1 else 1 if winner==(p-1)//2 else 2] += 1
        if best > fixed_best+64*np.finfo(np.float64).eps*max(best,fixed_best):
            winners[4] += 1
    else:
        best = 0.
        for r in range(rank[j]):
            dot = np.dot(vectors[r],delta)
            best += dot*dot
    cached_score[j] = best
    cached_revision[j] = revision
    return best


def state_for(a, inc, name):
    p = inc.shape[1]
    depth = int(np.diff(inc.indptr).max(initial=1))
    kind = {"all_sizes": 0, "three_span": 1, "order_span": 2}[name]
    sizes = np.arange(p) if kind==0 else np.array([0,(p-1)//2,p-1]) if kind==1 else np.array([],int)
    weights = np.ones((len(sizes),depth))
    for q in range(1,depth):
        weights[:,q] = weights[:,q-1]*np.maximum(0,sizes-q+1)/(p-q)
    col = inc.tocsc()
    winners = np.zeros(5,np.int64)
    state = (kind,a.indptr,a.indices,a.data,inc.indptr,col.indptr,col.indices,weights,
             np.empty((p,depth,a.shape[0])),np.empty((p,p)) if kind==0 else np.empty((0,0)),
             np.empty((p,p),bool) if kind==0 else np.empty((0,0),bool),
             np.zeros(p,np.int64),np.full(p,-1,np.int64),np.empty(p),winners)
    return _score,state,winners


def search(game, order, name):
    if name in ("last", "all"):
        return _run(game.a,game.incidence,order,.2,None,name)
    geometry = name.removesuffix("_residual")
    factory = None if geometry=="all" else lambda a,i:state_for(a,i,geometry)
    return _run(game.a,game.incidence,order,.2,None,"all",factory,name.endswith("_residual"))


def check():
    from itertools import combinations
    from scipy.sparse import csc_matrix, csr_matrix
    from minppi_mofs.stratified_tree import marginal_anchors
    rng = np.random.default_rng(8432)
    for p in range(2,9):
        groups = [g for k in range(1,min(p,4)+1) for g in combinations(range(p),k)]
        inc = csr_matrix([[int(j in g) for j in range(p)] for g in groups])
        a = csc_matrix(rng.integers(-2,3,size=(7,len(groups))).astype(float))
        anchors = marginal_anchors(a,inc,range(p),(a@inc).toarray().T)
        for name in ("all_sizes","three_span","order_span"):
            evaluate,state,_ = state_for(a,inc,name)
            for j in range(p):
                d = rng.normal(size=7)
                h = anchors[:,j]
                if name=="all_sizes":
                    norm = np.linalg.norm(h,axis=1)
                    direct = max(np.divide(h@d,norm,out=np.zeros(p),where=norm>0)**2)
                else:
                    rows = h[[0,(p-1)//2,p-1]] if name=="three_span" else h
                    coeff = np.linalg.lstsq(rows.T,d,rcond=None)[0]
                    direct = np.linalg.norm(rows.T@coeff)**2
                np.testing.assert_allclose(evaluate(j,d,0,state),direct,rtol=1e-10,atol=1e-10)
    print("All-cardinality and two joint-span calculations agree with direct projection.")


if __name__=="__main__":
    check()
