"""Exact greedy fidelity deletion screened by coalition-size marginal votes.

Columns of A are output changes for distinct required-feature sets. An increment
survives iff every required feature is present. No calibration labels are used.
"""
import numpy as np
from numba import njit


def containment(p, sizes, orders):
    """Probability all other members of a required set occur in a uniform background."""
    orders = np.asarray(orders, dtype=int)
    if p < 1 or np.any((orders < 1) | (orders > p)):
        raise ValueError("Each path group must require between one and p features")
    weights = np.ones((len(sizes), len(orders)))
    for i, size in enumerate(sizes):
        if size < 0 or size >= p:
            raise ValueError("A background leaves out the candidate feature")
        for h in range(1, int(orders.max(initial=1))):
            weights[i, orders > h] *= max(0, size - h + 1) / (p - h)
    return weights


def marginal_anchors(a, inc, sizes, dense_full):
    """One conditional mean marginal per feature and requested cardinality."""
    p = inc.shape[1]
    weights = containment(p, sizes, np.diff(inc.indptr))
    anchors = np.empty((len(sizes), p, a.shape[0]))
    for s, size in enumerate(sizes):
        if size == p - 1:
            anchors[s] = dense_full
        else:
            anchors[s] = (a @ inc.multiply(weights[s, :, None])).toarray().T
    return anchors


@njit(cache=True)
def _norm(j, colptr, rows, values, jptr, groups, active, work):
    work[:] = 0.
    for q in range(jptr[j], jptr[j+1]):
        group = groups[q]
        if active[group]:
            for z in range(colptr[group], colptr[group+1]):
                work[rows[z]] += values[z]
    return np.dot(work, work)


@njit(cache=True)
def _remove(j, anchors, acc, valid, residual, active, colptr, rows, values,
            iptr, features, jptr, groups):
    for pos in range(jptr[j], jptr[j+1]):
        g = groups[pos]
        if not active[g]:
            continue
        active[g] = False
        for e in range(colptr[g], colptr[g+1]):
            residual[rows[e]] += values[e]
        for h in range(iptr[g], iptr[g+1]):
            other = features[h]
            valid[other] = False
            for s in range(len(anchors)):
                dot = 0.
                for e in range(colptr[g], colptr[g+1]):
                    dot += values[e] * anchors[s,other,rows[e]]
                acc[s,other] -= dot


@njit(cache=True)
def _check(order, tie, cutoff, cushion, fixed, lower, valid, squared, keep,
           colptr, rows, values, jptr, groups, active, work):
    winner = -1
    exact_checks = 0
    for j in order:
        if not keep[j] or lower[j] > cutoff + cushion:
            break
        if not valid[j]:
            squared[j] = _norm(j, colptr, rows, values, jptr, groups, active, work)
            valid[j] = True
            exact_checks += 1
        risk = fixed[j] + squared[j]
        if risk < cutoff or (risk == cutoff and (winner < 0 or tie[j] > tie[winner])):
            winner, cutoff = j, risk
    return winner, cutoff, exact_checks


def greedy_path(a, incidence, gain_order, rho=.2, *, sizes=None, mode="all"):
    """Return retained column IDs, ordered deletions, and actual screening counts.

    A is sparse (calibration-output coordinates × path groups). The sparse
    0/1 incidence matrix is (groups × inputs). Gain order breaks exact ties.
    mode='last' uses only the largest requested size; 'cache' uses no votes.
    """
    a = a.tocsc(copy=False).astype(np.float64,copy=False)
    inc = incidence.tocsr(copy=False)
    p = inc.shape[1]
    order = np.asarray(gain_order, dtype=int)
    if a.shape[1] != inc.shape[0] or not np.array_equal(np.sort(order),np.arange(p)):
        raise ValueError("Incompatible path dictionary or gain ranking")
    if p == 0 or not np.isfinite(rho) or rho < 0 or mode not in {"all","last","cache"}:
        raise ValueError("Invalid feature count, tolerance or voting mode")
    if not np.isfinite(a.data).all() or not np.all(inc.data == 1) or not inc.has_canonical_format:
        raise ValueError("Finite effects and unique binary feature memberships are required")
    nonzero = np.abs(a.data[a.data != 0])
    if len(nonzero):
        lo,hi = np.frexp([nonzero.min(),nonzero.max()])[1]
        if hi-lo>450:
            raise ValueError("Path effects span too many exponents for float64 moment screening")
        if hi>40 or hi< -40 or lo< -450:
            a = a.copy().astype(float)
            a.data = np.ldexp(a.data,-int(hi))
    sizes = tuple(sorted(set((0,(p-1)//2,p-1) if sizes is None else sizes)))
    if not sizes or any(not isinstance(s,(int,np.integer)) or s<0 or s>=p for s in sizes):
        raise ValueError("At least one valid integer background size is required")
    sizes = () if mode == "cache" else sizes[-1:] if mode == "last" else sizes
    dense = np.ascontiguousarray((a @ inc).toarray().T, dtype=float)
    squared = np.einsum("ij,ij->i",dense,dense)
    anchors = marginal_anchors(a, inc, sizes, dense)
    den = np.einsum("ijk,ijk->ij",anchors,anchors)
    acc = np.einsum("ijk,jk->ij",anchors,dense)
    lengths = np.sqrt(den)
    col = inc.tocsc()
    args = (a.indptr,a.indices,a.data,col.indptr,col.indices)
    active, keep = np.ones(a.shape[1],bool), np.ones(p,bool)
    valid = keep.copy()
    residual = np.zeros(a.shape[0])
    work = np.zeros_like(residual)
    full = np.asarray(a.sum(axis=1)).ravel()
    budget = rho*rho*float(full@full)
    # Tolerance applies only to pruning. Equality and feasibility use the exact-candidate calculation.
    magnitude = float(np.linalg.norm(np.asarray(abs(a).sum(axis=1)).ravel()))
    cushion = np.finfo(float).eps * (8+a.shape[0]+a.shape[1]) * magnitude**2
    tie = np.argsort(order)
    path = []
    refreshed = 0
    additional_vetoes = 0
    contenders = 0
    while keep.any():
        cross = np.asarray(a.T @ residual).ravel() * active
        fixed = residual @ residual + 2*np.asarray(inc.T @ cross).ravel()
        projected_length = np.divide(acc,lengths,out=np.zeros_like(acc),where=lengths>0)
        votes = projected_length**2
        projected = votes.max(axis=0) if len(sizes) else np.zeros(p)
        lower = fixed + np.where(valid,squared,projected)
        lower[~keep] = np.inf
        stale = ~valid & keep
        candidates = np.lexsort((-tie,lower))
        winner, bound, evaluations = _check(candidates,tie,budget,cushion,fixed,lower,valid,
                                             squared,keep,*args,active,work)
        refreshed += evaluations
        contenders += int(keep.sum())
        if len(sizes)>1:
            additional_vetoes += int(np.sum(stale & (fixed+votes[-1] <= bound+cushion)
                                             & (lower > bound+cushion)))
        if winner < 0:
            break
        path.append(int(winner))
        keep[winner] = False
        _remove(winner,anchors,acc,valid,residual,active,a.indptr,a.indices,a.data,
                inc.indptr,inc.indices,col.indptr,col.indices)
    return np.flatnonzero(keep),dict(deletion_order=path,initial=p,refreshed=refreshed,
                                     contenders=contenders,extra_stratum_vetoes=additional_vetoes)
