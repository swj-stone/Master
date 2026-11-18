"""Certified support for rational pivotal scores using bounded integer arithmetic."""
from math import comb, lcm

import numpy as np

from minppi_mofs.certified_minpi import _validate
from minppi_mofs.exact_minpi import _selection_tiers


def _pivot_count(plus, minus, zero, k, sign):
    """Number of peer subsets for which a ballot of this sign is pivotal."""
    plus -= sign == 1
    minus -= sign == -1
    if plus < 0 or minus < 0:
        return 0
    total = 0
    for margin in (0, -sign):
        for pair in range((k - abs(margin)) // 2 + 1):
            a, b = pair + max(margin, 0), pair + max(-margin, 0)
            total += comb(plus, a) * comb(minus, b) * comb(zero, k-a-b)
    return total


def _layers(votes, tiers, scale):
    p = votes.shape[1]
    pos = np.count_nonzero(votes > 0, axis=1)
    neg = np.count_nonzero(votes < 0, axis=1)
    compositions, row = np.unique(np.column_stack((pos, neg, p-pos-neg)), axis=0, return_inverse=True)
    layers = []
    for k in tiers:
        denom = comb(p-1, k)
        counts = [tuple(_pivot_count(*map(int, c), k, s) * s for s in (1, -1))
                  for c in compositions]
        signed = [counts[i] for i in row]
        if scale is None:
            layers.append((denom, signed, None, 0))
            continue
        lower = np.array([[num * scale // denom for num in pair] for pair in counts], dtype=np.int64)[row]
        # On each row exactly one of its two signed weights or zero can contribute.
        error = sum(any(num * scale % denom for num in pair) for pair in signed)
        layers.append((denom, signed, lower, error))
    return layers


def _bounds(votes, layer):
    _, _, weights, error = layer
    # Exact int64 accumulation. The caller ensures that int64 can hold w * scale + w.
    pos = np.einsum("r,rj->j", weights[:, 0], votes > 0, dtype=np.int64, optimize=False)
    neg = np.einsum("r,rj->j", weights[:, 1], votes < 0, dtype=np.int64, optimize=False)
    lower = pos + neg
    return lower, lower + error


def _rational_scores(votes, layers):
    """Return exact scores multiplied by the common positive integer w*lcm(denominators)."""
    denominator = lcm(*(layer[0] for layer in layers))
    minima = None
    for denom, signed, _, _ in layers:
        totals = np.zeros(votes.shape[1], dtype=object)
        for row, (positive, negative) in zip(votes, signed):
            np.add(totals, positive, out=totals, where=row > 0)
            np.add(totals, negative, out=totals, where=row < 0)
        totals *= denominator // denom
        minima = totals if minima is None else np.minimum(minima, totals)
    return minima


def _descending(scores, count):
    return np.argsort(-scores, kind="stable")[:count]


def _dyadic_anchor(votes, layer):
    lower, upper = _bounds(votes, layer)
    # An abstaining column stays in every peer population. Its own pivot is
    # exactly zero on every layer, so all later minima preserve this endpoint.
    silent = ~np.any(votes, axis=0)
    upper[silent] = 0
    return lower, upper, silent


def _choose_boundary(votes, layers, lower, upper, boundary, quota, silent):
    if len(boundary) == quota:
        return boundary, 0
    if np.all(lower[boundary] == upper[boundary]):
        return boundary[_descending(lower[boundary], quota)], 0
    # Compare the nonzero-ballot columns on the common exact scale. Even on a
    # mixed boundary the known zeros need no arbitrary-precision accumulation.
    active = ~silent[boundary]
    scores = np.zeros(len(boundary), dtype=object)
    scores[active] = _rational_scores(votes[:, boundary[active]], layers)
    return boundary[_descending(scores, quota)], int(np.count_nonzero(active))


def rational_top_from_votes(votes, count, *, bits=None, prune=True, tiers=None):
    """Return the exact weakest-layer top set, breaking ties by column order.

    Columns of the returned support are in input order. The second return value
    reports the fixed-point precision, the number of columns refined past the
    largest layer, and the number scored at full integer precision. With
    prune=False every column is scored at full precision as a reference.
    Supply integer peer counts in [0, p). The default uses 0, 4, and 16 capped at p-1.
    """
    votes = _validate(votes, count)
    count = int(count)
    w, p = votes.shape
    max_bits = 62 - w.bit_length()
    if bits is None:
        bits = min(40, max_bits)
    if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, (int, np.integer)) or not 0 <= bits <= max_bits:
        raise ValueError(f"bits must be an integer between 0 and {max_bits} for these witnesses")
    info = dict(bits=int(bits), refined=0, exact_columns=0)
    tiers = _selection_tiers(tiers, p)
    if prune and count == p:
        return np.arange(p), info
    layers = _layers(votes, tiers, (1 << int(bits)) if prune else None)
    if not prune:
        scores = _rational_scores(votes, layers)
        return np.sort(_descending(scores, count)), dict(bits=int(bits), refined=p, exact_columns=p)

    lower, upper, silent = _dyadic_anchor(votes, layers[0])
    head = np.sort(_descending(upper, count))

    def refine(ids):
        matrix = np.ascontiguousarray(votes[:, ids])
        for layer in layers[1:]:
            left, right = _bounds(matrix, layer)
            lower[ids] = np.minimum(lower[ids], left)
            upper[ids] = np.minimum(upper[ids], right)

    refine(head)
    possible = upper >= lower[head].min()
    possible[head] = False
    frontier = np.flatnonzero(possible)
    if len(frontier):
        refine(frontier)
    possible[head] = True
    eligible = np.flatnonzero(possible)
    info["refined"] = len(eligible)
    if len(eligible) == count:
        return eligible, info

    low_cut = np.partition(lower[eligible], -count)[-count]
    high_cut = np.partition(upper[eligible], -count)[-count]
    inside = eligible[lower[eligible] > high_cut]
    boundary = eligible[(lower[eligible] <= high_cut) & (upper[eligible] >= low_cut)]
    quota = count - len(inside)
    chosen, info["exact_columns"] = _choose_boundary(votes, layers, lower, upper, boundary, quota, silent)
    return np.sort(np.r_[inside, chosen]), info
