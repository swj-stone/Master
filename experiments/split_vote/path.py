"""A bounded nested width search shared by the direct and vote rankings."""
from math import ceil, log2

import numpy as np


def select_width(order, evaluate, slack=.005):
    """Probe a geometric grid, then bisect the gap below the conditionally best width.

    Utility may be nonmonotone. Only refits actually observed are eligible.
    """
    p = len(order)
    scores = {}
    def probe(k):
        if k not in scores:
            scores[k] = float(evaluate(tuple(sorted(map(int,order[:k])))))
    def winner():
        bound = max(scores.values())-slack-1e-14
        return min(k for k,v in scores.items() if v >= bound)
    for k in sorted({p, *(min(p, 2**i) for i in range(ceil(log2(p))+1))}):
        probe(k)
    for _ in range(2):
        k = winner()
        lo = max((j for j in scores if j < k), default=0)
        if k-lo <= 1:
            break
        probe((k+lo)//2)
    return winner(), scores


def coverage_width(order, increments, evaluate, slack=.005):
    """Fit prefixes at successive halvings of residual vote mass plus the full set."""
    p = len(order)
    mass = np.cumsum(increments, dtype=float)
    if len(mass) != p or not np.isfinite(mass).all() or np.any(np.asarray(increments)<0):
        raise ValueError("Expected one nonnegative finite credit per retained feature")
    if mass[-1] == 0:
        mass = np.arange(1,p+1,dtype=float)
    widths = {p, *(int(np.searchsorted(mass, mass[-1]*q))+1 for q in (.5,.75,.875,.9375))}
    scores = {k:float(evaluate(tuple(sorted(map(int,order[:k]))))) for k in sorted(widths)}
    def winner():
        return min(k for k,v in scores.items() if v >= max(scores.values())-slack-1e-14)
    hi = winner()
    lo = max((k for k in scores if k < hi),default=0)
    if hi-lo>1:
        k=(hi+lo)//2
        scores[k]=float(evaluate(tuple(sorted(map(int,order[:k])))))
    return winner(),scores
