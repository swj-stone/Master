"""A bounded nested width search shared by the direct and vote rankings."""
from math import ceil, log2


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
