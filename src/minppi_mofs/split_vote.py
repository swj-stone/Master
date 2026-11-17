"""Pivotal votes for coverage of substitutable univariate splits."""

import numpy as np
from scipy.stats import hypergeom


def split_voters(x, y, total_gain, classification, seed, sketch_size=512):
    """Construct threshold voters on a training sketch, including native missingness."""
    n, p = x.shape
    if n > sketch_size:
        ix = np.random.default_rng(seed).choice(n, sketch_size, replace=False)
        x, y = x[ix], y[ix]
    target = np.eye(int(y.max()) + 1)[y.astype(int)] if classification else y[:, None]
    target = target - target.mean(axis=0)
    ballots, owners = [], []
    for j in range(p):
        values = x[:, j]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        cuts = np.unique(np.quantile(values[finite], [.125,.25,.375,.5,.625,.75,.875]))
        bits = [(values <= cut) & finite for cut in cuts]
        if not finite.all():
            bits.append(~finite)
        distinct = np.unique(np.asarray(bits), axis=0)
        for bit in distinct:
            if bit.any() and not bit.all():
                ballots.append(bit)
                owners.append(j)
    if not ballots:
        return np.eye(p, dtype=bool), np.maximum(total_gain, 0.)
    bits = np.asarray(ballots, dtype=np.float32)
    owners = np.asarray(owners)
    m = bits.shape[1]
    mass = bits.sum(axis=1)
    overlap = bits @ bits.T
    dist = mass[:, None] + mass[None, :] - 2 * overlap
    disagree = np.minimum(dist, m-dist)
    # A substitute must halve the mistakes of the constant-majority decision.
    proxies = disagree <= .5 * np.minimum(mass, m-mass)[:, None]
    covers = np.zeros((len(owners), p), dtype=bool)
    for j in np.unique(owners):
        covers[:, j] = proxies[:, owners == j].any(axis=1)
    # Distribute each feature's total training split gain across its own thresholds.
    separation = np.square(bits @ target).sum(axis=1) / (mass*(m-mass))
    weights = np.zeros_like(separation)
    for j in np.unique(owners):
        idx = owners == j
        share = separation[idx]
        weights[idx] = total_gain[j] * (share/share.sum() if share.sum() > 0 else 1./idx.sum())
    return covers, weights


def pivotal_order(covers, weights, gain, rule="min"):
    """Rank by the smallest relative pivotal credit over three coalition sizes.

    A covered voter leaves the game. Conditional on j being outside a uniform
    size-k coalition, its vote is pivotal iff no other representative is present.
    Relative credits are normalized within size to remove the common decay.
    """
    if rule not in {"min", "pooled"}:
        raise ValueError(rule)
    a = np.asarray(covers, dtype=float)
    p = a.shape[1]
    multiplicity = a.sum(axis=1).astype(int)
    if np.any(multiplicity == 0) or np.any(weights < 0):
        raise ValueError("Expected covered voters with nonnegative weights")
    live = np.ones(p, dtype=bool)
    mass = np.asarray(weights, dtype=float).copy()
    order = []
    while live.any():
        n = int(live.sum())
        if n == 1 or not mass.any():
            order.extend(sorted(np.flatnonzero(live), key=lambda j:(gain[j], j), reverse=True))
            break
        sizes = np.unique(np.rint(np.array([1/6, .5, 5/6])*(n-1)).astype(int))
        # The multiplicity of an uncovered voter is unchanged by previous choices.
        d = np.arange(1, n+1)
        hit = np.array([hypergeom.pmf(0, n-1, d-1, k) for k in sizes])
        discounts = hit[:, np.minimum(multiplicity,n)-1]
        credits = (discounts*mass) @ a
        credits[:, ~live] = 0
        totals = credits.sum(axis=1)
        norm = credits[totals > 0] / totals[totals > 0, None]
        if not len(norm):
            order.extend(sorted(np.flatnonzero(live), key=lambda j:(gain[j], j), reverse=True))
            break
        scores = norm.min(axis=0) if rule == "min" else norm.mean(axis=0)
        j = max(np.flatnonzero(live), key=lambda i:(scores[i],gain[i],i))
        order.append(j)
        live[j] = False
        mass[a[:, j] != 0] = 0.
    return np.array(order, dtype=int)
