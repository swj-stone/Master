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
def _projection(j, s, weights, active, projected, length, mass, offsets, rows, values,
                member_ptr, groups, memberships, incidence_ptr, scratch):
    """Build one size anchor when it first has a candidate to screen."""
    if length[j, s] < 0.:
        scratch[:] = 0.
        for pos in range(member_ptr[j], member_ptr[j+1]):
            group = groups[pos]
            cardinality = incidence_ptr[group+1] - incidence_ptr[group]
            weight = weights[s, cardinality]
            if weight:
                for k in range(offsets[group], offsets[group+1]):
                    scratch[rows[k]] += weight * values[k]
        length[j, s] = np.sqrt(np.dot(scratch, scratch))
        for pos in range(member_ptr[j], member_ptr[j+1]):
            group = groups[pos]
            if active[group]:
                dot = 0.
                for k in range(offsets[group], offsets[group+1]):
                    dot += values[k] * scratch[rows[k]]
                mass[memberships[pos], s] = dot
                projected[j, s] += dot
    if length[j, s] == 0.:
        return 0.
    v = projected[j, s] / length[j, s]
    return v*v


@njit(cache=True)
def _native(n, offsets, rows, values, incidence_ptr, features, order, weights, rho):
    p = len(order)
    strata = len(weights)
    ng = len(offsets)-1
    full = np.zeros(n)
    envelope = np.zeros(n)
    delta = np.zeros((p, n))
    # Inverse membership retains the group and the corresponding CSR entry.
    member_ptr = np.zeros(p+1, np.int64)
    for j in features:
        member_ptr[j+1] += 1
    member_ptr = np.cumsum(member_ptr)
    next_slot = member_ptr[:-1].copy()
    groups = np.empty(len(features), np.int64)
    memberships = np.empty(len(features), np.int64)
    for g in range(ng):
        for k in range(offsets[g], offsets[g+1]):
            full[rows[k]] += values[k]
            envelope[rows[k]] += abs(values[k])
        for ptr in range(incidence_ptr[g], incidence_ptr[g+1]):
            j = features[ptr]
            pos = next_slot[j]
            groups[pos], memberships[pos] = g, ptr
            next_slot[j] += 1
            for k in range(offsets[g], offsets[g+1]):
                delta[j, rows[k]] += values[k]
    budget = rho*rho*np.dot(full,full)
    cushion = np.finfo(np.float64).eps * (8+n+ng) * np.dot(envelope,envelope)
    squared = np.empty(p)
    for j in range(p):
        squared[j] = np.dot(delta[j],delta[j])

    keep = np.ones(p,np.bool_)
    known = np.ones(p,np.bool_)
    active = np.ones(ng,np.bool_)
    surviving_groups = np.diff(member_ptr)
    length = np.full((p,strata),-1.)
    projected = np.zeros((p,strata))
    mass = np.zeros((len(features),strata))
    scratch = np.empty(n)
    residual = np.zeros(n)
    fixed = np.zeros(p)
    lower = np.empty(p)
    full_lower = np.empty(p)
    stale = np.empty(p,np.bool_)
    tie = np.argsort(order)
    reverse = order[::-1].copy()
    count = checked = extra = contenders = 0
    path = np.empty(p,np.int64)
    rr = 0.
    while count < p:
        contenders += p-count
        best = budget
        win = -1
        for j in reverse:
            stale[j] = keep[j] and not known[j]
            if keep[j]:
                # The unprojected cross term is shared by every size vote.
                cross = np.dot(residual,delta[j]) if rr and surviving_groups[j] else 0.
                fixed[j] = rr + 2.*cross
                if known[j]:
                    cost = fixed[j]+squared[j]
                    if cost < best or (cost == best and win < 0):
                        best, win = cost, j
        for j in range(p):
            lower[j] = np.inf
            full_lower[j] = np.inf
            if stale[j] and fixed[j] <= best+cushion:
                bound = 0.
                for s in range(strata-1,-1,-1):
                    vote = _projection(j,s,weights,active,projected,length,mass,offsets,rows,values,
                                       member_ptr,groups,memberships,incidence_ptr,scratch)
                    bound = max(bound,vote)
                    if s == strata-1:
                        full_lower[j] = fixed[j]+bound
                    if fixed[j]+bound > best+cushion:
                        break
                lower[j] = fixed[j]+bound
        # Stable sorting of reverse gain priority matches ties in equal lower bounds.
        for pos in np.argsort(lower[reverse], kind='mergesort'):
            j = reverse[pos]
            if lower[j] > best+cushion or not stale[j]:
                break
            squared[j] = np.dot(delta[j],delta[j])
            known[j] = True
            checked += 1
            cost = fixed[j]+squared[j]
            if cost < best or (cost == best and (win < 0 or tie[j] > tie[win])):
                best, win = cost, j
        if strata > 1:
            for j in range(p):
                if stale[j] and full_lower[j] <= best+cushion and lower[j] > best+cushion:
                    extra += 1
        if win < 0:
            break
        path[count] = win
        count += 1
        keep[win] = False
        for pos in range(member_ptr[win], member_ptr[win+1]):
            g = groups[pos]
            if not active[g]:
                continue
            active[g] = False
            for k in range(offsets[g],offsets[g+1]):
                residual[rows[k]] += values[k]
            for ptr in range(incidence_ptr[g],incidence_ptr[g+1]):
                j = features[ptr]
                if not keep[j]:
                    continue
                known[j] = False
                surviving_groups[j] -= 1
                if surviving_groups[j] == 0:
                    delta[j,:] = 0.
                else:
                    for k in range(offsets[g],offsets[g+1]):
                        delta[j,rows[k]] -= values[k]
                for s in range(strata):
                    if length[j,s] >= 0.:
                        projected[j,s] -= mass[ptr,s]
        rr = np.dot(residual,residual)
    return path[:count], keep, checked, extra, contenders


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
    cardinality = np.diff(inc.indptr)
    if np.any((cardinality < 1) | (cardinality > p)):
        raise ValueError("Invalid required-feature set size")
    weights = np.zeros((len(sizes), int(cardinality.max(initial=0))+1))
    for s,size in enumerate(sizes):
        if weights.shape[1]>1:
            weights[s,1] = 1.
        for k in range(2,weights.shape[1]):
            weights[s,k] = weights[s,k-1] * (max(0,size-k+2)/(p-k+1))
    path,keep,checked,extra,contenders = _native(a.shape[0],a.indptr,a.indices,a.data,inc.indptr,
                                                inc.indices,order,weights,float(rho))
    return np.flatnonzero(keep),dict(deletion_order=path.tolist(),initial=p,refreshed=checked,
                                     contenders=contenders,extra_stratum_vetoes=extra)
