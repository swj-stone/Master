"""Study comparator probing a common signed-word scale from observed pivots."""
from math import comb, gcd, lcm

import numpy as np

from minppi_mofs.certified_minpi import _validate
from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.integer_minpi import _pivot_count, rational_top_from_votes
from minppi_mofs.word_minpi import native_scale, word_support


WORD_MAX = 2**63 - 1


def reduced_weights(votes, tiers, max_scale):
    """Return a common scale and per-row int64 weights, or None on certified overflow."""
    p = votes.shape[1]
    positive = np.count_nonzero(votes > 0, axis=1)
    negative = np.count_nonzero(votes < 0, axis=1)
    populations = np.column_stack((positive, negative, p - positive - negative))
    unique, row = np.unique(populations, axis=0, return_inverse=True)
    compositions = unique.tolist()
    layers = []
    common = 1
    for k in tiers:
        denominator = comb(p - 1, k)
        divisor = denominator
        pairs = []
        for plus, minus, zero in compositions:
            pos = _pivot_count(plus, minus, zero, k, 1) if plus else 0
            neg = -_pivot_count(plus, minus, zero, k, -1) if minus else 0
            pairs.append((pos, neg))
            if divisor > 1:
                new_divisor = gcd(divisor, pos, neg)
                if new_divisor != divisor:
                    divisor = new_divisor
                    # Any fraction seen so far must still have an integral weight.
                    if lcm(common, denominator // divisor) > max_scale:
                        return None
        reduced = denominator // divisor
        common = lcm(common, reduced)
        layers.append((divisor, reduced, pairs))
    weights = []
    for divisor, reduced, pairs in layers:
        multiplier = common // reduced
        # Divide and multiply in Python before converting potentially huge counts.
        scaled = [[(a // divisor) * multiplier, (b // divisor) * multiplier]
                  for a, b in pairs]
        weights.append(np.asarray(scaled, dtype=np.int64)[row])
    return common, weights


def word_totals(matrix, weights):
    positive = np.einsum("r,rj->j", weights[:, 0], matrix > 0, dtype=np.int64)
    negative = np.einsum("r,rj->j", weights[:, 1], matrix < 0, dtype=np.int64)
    return positive + negative


def reduced_support(votes, h, weights):
    """Use the same static anchor and inclusive cutoff as the raw word selector."""
    anchor = word_totals(votes, weights[0])
    head = np.sort(np.argsort(-anchor, kind="stable")[:h])
    if len(weights) == 1:
        return head, h

    def complete(indices):
        matrix = np.ascontiguousarray(votes[:, indices])
        minimum = anchor[indices].copy()
        for weight in weights[1:]:
            np.minimum(minimum, word_totals(matrix, weight), out=minimum)
        return minimum

    scores = complete(head)
    eligible = anchor >= scores.min()
    eligible[head] = False
    outside = np.flatnonzero(eligible)
    if not len(outside):
        return head, h
    columns = np.concatenate((head, outside))
    scores = np.concatenate((scores, complete(outside)))
    order = np.lexsort((columns, -scores))
    return np.sort(columns[order[:h]]), len(columns)


def reduced_top_from_votes(votes, count, *, tiers=None):
    votes = _validate(votes, count)
    h = int(count)
    w, p = votes.shape
    tiers = _selection_tiers(tiers, p)
    if h == p:
        return np.arange(p), dict(arithmetic="all", refined=0, exact_columns=0)
    common = native_scale(p, tiers, w)
    if common:
        chosen, refined = word_support(votes, h, tiers, common, prune=True)
        return chosen, dict(arithmetic="int64", refined=refined, exact_columns=refined)
    result = reduced_weights(votes, tiers, WORD_MAX // w)
    if result is not None:
        _, weights = result
        chosen, refined = reduced_support(votes, h, weights)
        return chosen, dict(arithmetic="reduced_int64", refined=refined, exact_columns=refined)
    chosen, info = rational_top_from_votes(votes, h, tiers=tiers)
    return chosen, dict(arithmetic="dyadic", **info)
