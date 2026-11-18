"""Select only the top-h pivotal features, using one size as an upper bound."""
from math import ceil

import numpy as np

from minppi_mofs.exact_minpi import _margin_mass, _selection_tiers, witness_margins


def _weights(positive, negative, neutral, p, tiers):
    # Only populations attained by training witnesses occur in the hypergeometric sum.
    # Map them once so the binomial table retains O(w max(K)) entries even for huge p.
    populations = np.unique(np.r_[p-1, positive, np.maximum(0, positive-1),
                                  negative, np.maximum(0, negative-1), neutral])
    log_choose = np.full((len(populations), max(tiers)+1), -np.inf)
    log_choose[:, 0] = 0
    for j in range(1, max(tiers)+1):
        valid = populations >= j
        log_choose[valid, j] = log_choose[valid, j-1] + np.log((populations[valid]-j+1) / j)
    norm = log_choose[np.searchsorted(populations, p-1)]
    zero = np.searchsorted(populations, neutral)
    result = {k: [] for k in tiers if k}
    for sign in (1, -1):
        a = np.searchsorted(populations, np.maximum(0, positive - int(sign == 1)))
        b = np.searchsorted(populations, np.maximum(0, negative - int(sign == -1)))
        for k in result:
            central = _margin_mass(a, b, zero, k, 0, log_choose, norm[k])
            adjacent = _margin_mass(a, b, zero, k, -sign, log_choose, norm[k])
            result[k].append(central + adjacent)
    return result


def _scores(votes, weights):
    # Keep a fixed summation order across columns for ties and threshold comparisons.
    pos, neg = weights
    plus = np.einsum("r,rj->j", pos, votes > 0, dtype=np.float64, optimize=False)
    minus = np.einsum("r,rj->j", neg, votes < 0, dtype=np.float64, optimize=False)
    return (plus - minus) / len(votes)


def _validate(votes, count):
    votes = np.asarray(votes)
    if votes.ndim != 2 or min(votes.shape) < 1:
        raise ValueError("Expected a nonempty witness-by-feature matrix")
    integral = votes.dtype.kind in "biu"
    ternary = (votes.min() >= -1 and votes.max() <= 1) if integral else np.isin(votes, (-1, 0, 1)).all()
    if not ternary:
        raise ValueError("Witness votes must be -1, 0 or +1")
    if not isinstance(count, (int, np.integer)) or not 1 <= count <= votes.shape[1]:
        raise ValueError("Requested feature count must lie in [1,p]")
    return np.ascontiguousarray(votes)


def top_from_votes(votes, count, *, prune=True, tiers=None):
    """Return stable top indices in input order, and the number of tier-column scores evaluated.

    A reference run with prune=False uses the same summation and tie conventions and evaluates
    every specified tier. Counts and inclusion probabilities always use the full population.
    Supply integer peer counts in [0, p). The default uses 0, 4, and 16 capped at p-1.
    """
    votes = _validate(votes, count)
    count = int(count)
    p = votes.shape[1]
    tiers = _selection_tiers(tiers, p)
    phases = len(tiers)
    if prune and count == p:
        return np.arange(p), dict(p=p, h=count, tiers=phases, refined=0, evaluated=0)
    pos = np.count_nonzero(votes > 0, axis=1)
    neg = np.count_nonzero(votes < 0, axis=1)
    zero = p - pos - neg
    weights = _weights(pos, neg, zero, p, tiers)

    def rate(k, matrix):
        return _scores(matrix, weights[k]) if k else matrix.mean(axis=0, dtype=np.float64)

    upper = rate(tiers[0], votes)
    if not prune or phases == 1 or count == p:
        minima = upper.copy()
        for k in tiers[1:]:
            np.minimum(minima, rate(k, votes), out=minima)
        support = np.sort(np.argsort(-minima, kind="stable")[:count])
        return support, dict(p=p, h=count, tiers=phases, refined=p, evaluated=p*phases)

    def refine(ids):
        # C major for identical row-wise accumulation for an arbitrary column subset.
        matrix = np.ascontiguousarray(votes[:, ids])
        minima = upper[ids].copy()
        for k in tiers[1:]:
            np.minimum(minima, rate(k, matrix), out=minima)
        return minima

    head = np.sort(np.argsort(-upper, kind="stable")[:count])
    head_scores = refine(head)
    possible = upper >= head_scores.min()
    possible[head] = False
    frontier = np.flatnonzero(possible)
    eligible = np.zeros(p, dtype=bool)
    eligible[head] = True
    eligible[frontier] = True
    candidates = np.flatnonzero(eligible)
    score = upper.copy()
    score[head] = head_scores
    if len(frontier):
        score[frontier] = refine(frontier)
    support = np.sort(candidates[np.argsort(-score[candidates], kind="stable")[:count]])
    return support, dict(p=p, h=count, tiers=phases, refined=len(candidates),
                         evaluated=p+(phases-1)*len(candidates))


def select_minpi_support(X, y, *, fraction=.1, task="classification", random_state=42, tiers=None):
    """Fit training witnesses and return the certified support as original column positions."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    margins = witness_margins(X, y, task, random_state)
    votes = np.sign(margins).astype(np.int8)
    return top_from_votes(votes, ceil(fraction * votes.shape[1]), tiers=tiers)[0]
