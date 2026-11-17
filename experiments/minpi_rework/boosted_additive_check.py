"""Check quadratic block elimination against direct evaluation on tiny games."""
from itertools import combinations

import numpy as np
from scipy.special import softmax

from experiments.minpi_rework.boosted_additive import pair_order, probability_parts
from minppi_mofs.additive_minpi import quadratic


def check():
    rng = np.random.default_rng(1782)
    for p in (4, 5):
        z = rng.normal(size=(9,p,3))
        target = rng.normal(size=(9,3))
        gram, single = quadratic(z,target)
        def value(features):
            return -np.sum((target-z[:,list(features)].sum(axis=1))**2)/len(z)
        for rule in ("min","pooled","point"):
            active = set(range(p))
            path = pair_order(gram,single,rule)[::-1]
            for pos in range(0, p-1, 2):
                scores = {}
                for pair in combinations(active,2):
                    rest = active-set(pair)
                    swings = [value((*s,*pair))-value(s) for k in range(len(rest)+1)
                              for s in combinations(rest,k)]
                    scores[pair] = min(swings) if rule=="min" else (
                        np.mean(swings) if rule=="pooled" else value(active)-value(rest))
                chosen = set(path[pos:pos+2])
                score = next(v for pair,v in scores.items() if set(pair)==chosen)
                np.testing.assert_allclose(score, min(scores.values()),atol=1e-12)
                active -= chosen
    for c in (1, 3):
        bias = rng.normal(size=(9,c))
        raw = rng.normal(size=(9,5,c))
        parts, anchor = probability_parts(raw,bias)
        logits = bias + raw.sum(axis=1)
        if c==1:
            logits = np.hstack((np.zeros_like(logits),logits))
        np.testing.assert_allclose(anchor+parts.sum(axis=1),softmax(logits,axis=1),atol=1e-12)
    print("Block updates match exhaustive marginal votes; probability allocations reconstruct binary and multiclass scores.")


if __name__=="__main__":
    check()
