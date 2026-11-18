"""Independent block selector that refits each complete candidate support."""
from itertools import combinations

import numpy as np

from research.pair_checks.verify import ridge_loss


def choose(candidates):
    """Candidates arrive in size/lexicographic order."""
    best_key, best = None, -np.inf
    for key, value in candidates:
        if best_key is None or value - best > 32*np.finfo(float).eps*max(abs(best), abs(value)):
            best_key, best = key, value
    return best_key, best


def block_path(z, response, cap):
    """Log residual ratios are computed from augmented least-squares refits."""
    m, p = z.shape
    empty_loss = float(np.sum(response**2))
    fits = {(): empty_loss}

    def loss(support):
        ids = tuple(sorted(support))
        if ids not in fits:
            fits[ids] = ridge_loss(z, response, ids)
        return fits[ids]

    seed_size = min(2, cap)
    seed, _ = choose((a, empty_loss-loss(a)) for a in combinations(range(p), seed_size))
    path = list(seed)
    blocks = [list(seed)]
    decisions = []
    while len(path) < cap:
        available = [j for j in range(p) if j not in path]
        base = loss(path)
        candidates = []
        for b in range(1, min(2, cap-len(path))+1):
            for a in combinations(available, b):
                after = loss(path + list(a))
                value = m*np.log(base/after) - len(a)*np.log(m)
                candidates.append((a, value))
        winner, criterion = choose(candidates)
        max_single = max(value for a, value in candidates if len(a) == 1)
        decisions.append(dict(suffix=list(winner), score=float(criterion),
                              best_single=float(max_single), accepted=bool(criterion > 0)))
        if criterion <= 0:
            break
        path.extend(winner)
        blocks.append(list(winner))
    return blocks, decisions, len(fits)-1
