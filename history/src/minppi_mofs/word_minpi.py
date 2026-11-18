"""Exact pivotal support with native integer scores whenever the common scale fits."""
from math import comb, lcm

import numpy as np

from minppi_mofs.certified_minpi import _validate
from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.integer_minpi import _layers, rational_top_from_votes


def native_scale(p, tiers, w):
    """Common denominator, or zero if a signed int64 sum could overflow."""
    common = lcm(*(comb(p - 1, k) for k in tiers))
    return common if w * common <= np.iinfo(np.int64).max else 0


def word_support(votes, count, tiers, common, prune):
    """Use validated votes and a proven-safe common scale."""
    p = votes.shape[1]
    weights = []
    for denominator, signed, _, _ in _layers(votes, tiers, None):
        weights.append(np.asarray(signed, dtype=np.int64) * (common // denominator))

    def totals(matrix, weight):
        positive = np.einsum("r,rj->j", weight[:, 0], matrix > 0, dtype=np.int64)
        negative = np.einsum("r,rj->j", weight[:, 1], matrix < 0, dtype=np.int64)
        return positive + negative

    anchor = totals(votes, weights[0])

    def complete(ids):
        matrix = np.ascontiguousarray(votes[:, ids])
        score = anchor[ids].copy()
        for weight in weights[1:]:
            np.minimum(score, totals(matrix, weight), out=score)
        return score

    if not prune:
        scores = complete(np.arange(p))
        chosen = np.argsort(-scores, kind="stable")[:count]
        return np.sort(chosen), p
    head = np.sort(np.argsort(-anchor, kind="stable")[:count])
    if len(weights) == 1:
        return head, count
    score = complete(head)
    eligible = anchor >= score.min()
    eligible[head] = False
    more = np.flatnonzero(eligible)
    if not len(more):
        return head, count
    minima = np.concatenate((score, complete(more)))
    ids = np.concatenate((head, more))
    order = np.lexsort((ids, -minima))
    return np.sort(ids[order[:count]]), len(ids)


def exact_top_from_votes(votes, count, *, tiers=None):
    """Return the exact top set in input order, plus the arithmetic used."""
    votes = _validate(votes, count)
    count = int(count)
    w, p = votes.shape
    tiers = _selection_tiers(tiers, p)
    if count == p:
        return np.arange(p), dict(arithmetic="all", refined=0, exact_columns=0)
    common = native_scale(p, tiers, w)
    if common:
        selected, refined = word_support(votes, count, tiers, common, prune=True)
        return selected, dict(arithmetic="int64", refined=refined, exact_columns=refined)
    selected, info = rational_top_from_votes(votes, count, tiers=tiers)
    return selected, dict(arithmetic="dyadic", **info)

