"""Greedy pivotal selection conditional on the votes of features already kept."""
from functools import lru_cache
from math import comb, lcm

import numpy as np

from minppi_mofs.certified_minpi import _validate
from minppi_mofs.exact_minpi import _selection_tiers


def _margin_count(positive, negative, neutral, peers, margin):
    """Number of indexed peer sets with the specified size and signed sum."""
    if abs(margin) > peers:
        return 0
    total = 0
    for pairs in range((peers - abs(margin)) // 2 + 1):
        up = pairs + max(0, margin)
        down = pairs + max(0, -margin)
        zeros = peers - up - down
        if up <= positive and down <= negative and zeros <= neutral:
            total += comb(positive, up) * comb(negative, down) * comb(neutral, zeros)
    return total


def _conditional_order(votes, count, tiers):
    # A cached count depends on the row composition, not on feature identity.
    count_margin = lru_cache(maxsize=16384)(_margin_count)
    fixed = np.zeros(len(votes), dtype=np.int64)
    remaining = np.arange(votes.shape[1])
    selected = []
    for _ in range(count):
        if len(remaining) == 1:
            selected.append(int(remaining[0]))
            break
        pool = votes[:, remaining]
        positive = pool > 0
        negative = pool < 0
        plus = positive.sum(axis=1)
        minus = negative.sum(axis=1)
        states, state_ids = np.unique(
            np.column_stack((plus, minus, len(remaining) - plus - minus, fixed)),
            axis=0, return_inverse=True,
        )
        levels = sorted({min(k, len(remaining) - 1) for k in tiers})
        denominators = [comb(len(remaining) - 1, k) for k in levels]
        scale = lcm(*denominators)
        safe = len(votes) * scale <= np.iinfo(np.int64).max
        minima = None
        for k, denominator in zip(levels, denominators):
            weights = []
            for a, b, c, locked in states.tolist():
                values = []
                for sign in (1, -1):
                    p = a - (sign == 1)
                    n = b - (sign == -1)
                    value = 0
                    if p >= 0 and n >= 0:
                        for target in (-locked, -locked - sign):
                            value += count_margin(p, n, c, k, target)
                    values.append(value * sign)
                weights.append(values)
            dtype = np.int64 if safe else object
            row_weights = np.asarray(weights, dtype=dtype)[state_ids]
            totals = (row_weights[:, 0] @ positive + row_weights[:, 1] @ negative) * (scale // denominator)
            minima = totals if minima is None else np.minimum(minima, totals)
        winner = int(np.argmax(minima))
        chosen = int(remaining[winner])
        selected.append(chosen)
        fixed += votes[:, chosen]
        remaining = np.delete(remaining, winner)
    return selected


def conditional_top_from_votes(votes, count, *, tiers=None):
    """Return greedy choices under exact signed increments given the current selected set.

    Every selected column always votes. At each step the additional peers are
    sampled uniformly from the remaining columns other than the candidate.
    Levels exceeding the remaining peer population are clipped and deduplicated.
    Ties use the smaller original index. The returned support is in input order.
    """
    votes = _validate(votes, count).astype(np.int8, copy=False)
    levels = _selection_tiers(tiers, votes.shape[1])
    order = _conditional_order(votes, count, levels)
    return np.sort(order), {"selection_order": order}
