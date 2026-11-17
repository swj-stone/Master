"""Exact pivotal credit by representation multiplicity on a fixed split sketch."""

from math import comb, lcm

import numpy as np


def split_voters(x, y, total_gain, classification, seed, sketch_size=512):
    """Construct the frozen threshold sketch with byte keys for row deduplication.

    Pack the first sample in the high bit so byte order reproduces the
    original lexicographic order of Boolean sample rows. Unpack-free indexing
    retains those exact Boolean rows for the frozen floating-point gain split.
    """
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
        cuts = np.unique(
            np.quantile(values[finite], [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        )
        bits = (values[None, :] <= cuts[:, None]) & finite
        if not finite.all():
            bits = np.vstack((bits, ~finite))
        packed = np.packbits(bits, axis=1, bitorder="big")
        keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
        _, index = np.unique(keys, return_index=True)
        rows = bits[index]
        rows = rows[rows.any(axis=1) & ~rows.all(axis=1)]
        if len(rows):
            ballots.append(rows)
            owners.extend([j] * len(rows))
    if not ballots:
        return np.eye(p, dtype=bool), np.maximum(total_gain, 0.0)
    bits = np.concatenate(ballots, axis=0).astype(np.float32)
    owners = np.asarray(owners)
    active, starts = np.unique(owners, return_index=True)
    mass = bits.sum(axis=1)
    m = bits.shape[1]
    overlap = bits @ bits.T
    dist = mass[:, None] + mass[None, :] - 2 * overlap
    proxies = np.minimum(dist, m - dist) <= 0.5 * np.minimum(mass, m - mass)[:, None]
    covers = np.zeros((len(owners), p), dtype=bool)
    covers[:, active] = np.logical_or.reduceat(proxies, starts, axis=1)
    separation = np.square(bits @ target).sum(axis=1) / (mass * (m - mass))
    weights = np.zeros_like(separation)
    for j, begin, end in zip(
        active, starts, np.r_[starts[1:], len(owners)], strict=True
    ):
        share = separation[begin:end]
        total = share.sum()
        weights[begin:end] = total_gain[j] * (
            share / total if total > 0 else 1.0 / len(share)
        )
    return covers, weights


def _integer_weights(masses):
    # Every finite binary64 is a dyadic rational. One denominator removes both
    # division and cancellation when the final large or small voter is deleted.
    ratios = [float(m).as_integer_ratio() for m in masses]
    scale = max((d for _, d in ratios), default=1)
    return np.array([v * (scale // d) for v, d in ratios], dtype=object)


def _scaled_coefficients(n, degrees, totals):
    sizes = np.unique(np.rint(np.array([1 / 6, 0.5, 5 / 6]) * (n - 1)).astype(int))
    rows, denominators = [], []
    for k in sizes[::-1]:
        row = np.array([comb(n - int(d), int(k)) for d in degrees], dtype=object)
        denom = row @ totals
        if denom:
            rows.append(row)
            denominators.append(denom)
    common = lcm(*denominators)
    return [
        row * (common // denom) for row, denom in zip(rows, denominators, strict=True)
    ]


def _choose(rows, profile, live, gain, rule, prune, work):
    def best(columns, scores):
        return max(
            zip(columns, scores, strict=True),
            key=lambda pair: (pair[1], gain[pair[0]], pair[0]),
        )[0]

    first = rows[0] @ profile[:, live]
    work["possible"] += len(rows) * len(live)
    work["evaluated"] += len(live)
    candidates = live
    if rule == "min" and prune and len(rows) > 1 and len(live) > 1:
        # A completed candidate's minimum is a lower bound on the winning score.
        # The first share upper-bounds every rival's minimum. Keep exact ties.
        leader = best(live, first)
        lower = min(
            first[np.searchsorted(live, leader)],
            *(row @ profile[:, leader] for row in rows[1:]),
        )
        keep = first >= lower
        work["evaluated"] += len(rows) - 1
        candidates, first = live[keep], first[keep]
        if len(candidates) == 1:
            return int(leader)
        # The incumbent is fully priced already. Avoid counting/evaluating it twice.
        rival = candidates != leader
        candidates, first = candidates[rival], first[rival]
        for row in rows[1:]:
            np.minimum(first, row @ profile[:, candidates], out=first)
            work["evaluated"] += len(candidates)
        ids = np.append(candidates, leader)
        return int(best(ids, np.append(first, lower)))
    for row in rows[1:]:
        next_row = row @ profile[:, live]
        work["evaluated"] += len(live)
        if rule == "min":
            np.minimum(first, next_row, out=first)
        else:
            first += next_row
    return int(best(live, first))


def pivotal_order(covers, weights, gain, rule="min", *, prune=True, work=None):
    """Return the mathematical ranking of finite binary64 masses, breaking ties by gain/index.

    Each live row sums one feature's incident threshold mass within one degree.
    The common size denominator cancels in relative shares. Integer binomial
    weights and integer comparisons give exact minima even at extreme scales.
    """
    covers = np.asarray(covers)
    weights, gain = np.asarray(weights, dtype=float), np.asarray(gain, dtype=float)
    if (
        covers.ndim != 2
        or weights.shape != (len(covers),)
        or gain.shape != (covers.shape[1],)
    ):
        raise ValueError("Unaligned coverage, weight or gain shape")
    if (
        rule not in {"pooled", "min"}
        or not np.all((covers == 0) | (covers == 1))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.all(np.isfinite(gain))
    ):
        raise ValueError(
            "Expected binary incidences, finite nonnegative weights and finite gains"
        )
    covers = covers.astype(bool, copy=False)
    multiplicity = covers.sum(axis=1)
    if np.any(multiplicity == 0):
        raise ValueError("Empty voter")
    positive = weights > 0
    covers, multiplicity = covers[positive], multiplicity[positive]
    degrees, bins = np.unique(multiplicity, return_inverse=True)
    mass = _integer_weights(weights[positive])
    p = covers.shape[1]
    all_cols = np.arange(p)
    available = np.ones(p, dtype=bool)
    unhit = np.ones(len(mass), dtype=bool)
    profile = np.zeros((len(degrees), p), dtype=object)
    degree_total = np.zeros(len(degrees), dtype=object)
    votes, cols = np.nonzero(covers)
    np.add.at(profile, (bins[votes], cols), mass[votes])
    np.add.at(degree_total, bins, mass * degrees[bins])
    if work is None:
        work = {}
    work.update(possible=0, evaluated=0)
    order = []
    while len(order) < p:
        live = all_cols[available]
        active = np.flatnonzero(degree_total)
        rows = (
            _scaled_coefficients(len(live), degrees[active], degree_total[active])
            if len(active)
            else []
        )
        if len(live) == 1 or not rows:
            order.extend(sorted(live, key=lambda j: (gain[j], j), reverse=True))
            break
        if len(active) == 1 and degrees[active[0]] == 1:
            # Private thresholds never interact. Remaining incidence masses stay fixed.
            private = profile[active[0]]
            order.extend(
                sorted(live, key=lambda j: (private[j], gain[j], j), reverse=True)
            )
            break
        chosen = _choose(rows, profile[active], live, gain, rule, prune, work)
        order.append(chosen)
        available[chosen] = False
        newly = np.flatnonzero(unhit & covers[:, chosen])
        unhit[newly] = False
        hit, features = np.nonzero(covers[newly])
        voter = newly[hit]
        np.subtract.at(profile, (bins[voter], features), mass[voter])
        np.subtract.at(degree_total, bins[newly], mass[newly] * degrees[bins[newly]])
    return np.asarray(order, dtype=int)
