"""Experimental scalar triangle ballots on fixed-width coalition marginal anchors.

The sparse group dictionary and tie policy match the compiled direct greedy arm.
This is a timing experiment, not a replacement for the production selector.
"""
import numpy as np
from numba import njit
from experiments.minpi_rework.quota_compress_tree_native_exact import initialize


@njit(cache=True)
def _summaries(n, ap, ai, av, ip, ii, p, sizes):
    groups = len(ap) - 1
    q = np.ones((len(sizes), groups))
    mass = np.empty(groups)
    anchors = np.zeros((len(sizes), p, n))
    radius = np.zeros((len(sizes), p))
    for g in range(groups):
        mass_squared = 0.
        for z in range(ap[g], ap[g+1]):
            mass_squared += av[z]**2
        mass[g] = np.sqrt(mass_squared)
        for t, s in enumerate(sizes):
            weight = 1.
            for k in range(ip[g+1]-ip[g]-1):
                weight *= max(0, s-k)/(p-1-k)
            q[t,g] = weight
            for pos in range(ip[g], ip[g+1]):
                j = ii[pos]
                radius[t,j] += (1-weight)*mass[g]
                if weight:
                    for z in range(ap[g], ap[g+1]):
                        anchors[t,j,ai[z]] += weight*av[z]
    lengths = np.zeros((len(sizes), p))
    for t in range(len(sizes)):
        for j in range(p):
            lengths[t,j] = np.sqrt(np.dot(anchors[t,j],anchors[t,j]))
    return q, mass, radius, lengths


@njit(cache=True)
def scalar(n, ap, ai, av, ip, ii, gp, gi, order, limit, sizes, dynamic):
    """Union of size-specific triangle vetoes, optionally with an updated norm.

    For an active group, its distance coefficient from the s-anchor is 1-q;
    after it dies the coefficient is q. Each dead incidence therefore makes
    the scalar radius change by (2q-1) times its group norm. No projection
    inner product is updated. The common dense dictionary stays available for
    contenders that pass these tests. We directly sum their deployment errors.
    """
    p = len(order)
    d = initialize(n, ap, ai, av, ip, ii, p)
    q, mass, radius, lengths = _summaries(n,ap,ai,av,ip,ii,p,sizes)
    residual = np.zeros(n)
    active = np.ones(len(ap)-1, np.bool_)
    keep = np.ones(p, np.bool_)
    cached = np.zeros(p)
    drift = np.zeros(p)
    error_norm = np.zeros(p)
    error_drift = np.zeros(p)
    shared_mass = np.zeros(p)
    if dynamic:
        for j in range(p):
            cached[j] = np.sqrt(np.dot(d[j],d[j]))
            error_norm[j] = cached[j]
    scale = max(1., np.sum(mass))
    cushion = (16+n+len(av))*np.finfo(np.float64).eps*scale**2
    path = np.empty(p, np.int32)
    count = exact = veto = 0
    for _ in range(p):
        best = limit**2
        winner = -1
        residual_norm = np.sqrt(np.dot(residual,residual))
        for pos in range(p-1,-1,-1):
            j = order[pos]
            if not keep[j]:
                continue
            low, high = 0., np.inf
            for t in range(len(sizes)):
                low = max(low,lengths[t,j]-radius[t,j])
                high = min(high,lengths[t,j]+radius[t,j])
            if dynamic:
                low = max(low,cached[j]-drift[j])
                high = min(high,cached[j]+drift[j])
            lower = max(0., low-residual_norm, residual_norm-high)
            if dynamic:
                lower = max(lower,error_norm[j]-error_drift[j])
            if lower**2 > best+cushion:
                veto += 1
                continue
            exact += 1
            cost = exact_norm = 0.
            for row in range(n):
                delta = d[j,row]
                cost += (residual[row]+delta)**2
                if dynamic:
                    exact_norm += delta**2
            if dynamic:
                cached[j], drift[j] = np.sqrt(exact_norm), 0.
                error_norm[j], error_drift[j] = np.sqrt(cost), 0.
            if cost<best or (cost==best and winner<0):
                best, winner = cost, j
        if winner < 0:
            break
        residual += d[winner]
        keep[winner] = False
        path[count] = winner
        count += 1
        deleted_mass = 0.
        shared_mass[:] = 0.
        for ptr in range(gp[winner], gp[winner+1]):
            g = gi[ptr]
            if active[g]:
                active[g] = False
                deleted_mass += mass[g]
                for pos in range(ip[g], ip[g+1]):
                    j = ii[pos]
                    if keep[j]:
                        if dynamic:
                            drift[j] += mass[g]
                            shared_mass[j] += mass[g]
                        for t in range(len(sizes)):
                            radius[t,j] += (2*q[t,g]-1)*mass[g]
                        for z in range(ap[g], ap[g+1]):
                            d[j,ai[z]] -= av[z]
        if dynamic:
            for j in range(p):
                if keep[j]:
                    # The trial error vector changes by the sum of the deleted groups
                    # that do not require j. Either expansion gives a valid radius.
                    error_drift[j] += min(deleted_mass-shared_mass[j], cached[winner]+shared_mass[j])
    return path[:count],exact,veto
