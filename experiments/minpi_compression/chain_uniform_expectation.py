"""Exact finite-pool expectation of the query-matched random-prefix control."""
from fractions import Fraction
from math import comb

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from experiments.minpi_compression.dictator_audit import jobs
from minppi_mofs.chain_minpi import _brackets, _query, refine_brackets
from minppi_mofs.dictator_minpi import RidgeBallot, sharp_upper_bounds
from minppi_mofs.sampled_linear_minpi import _targets


def detection_probabilities(p, votes, anchors, budget):
    """Conditional probability of seeing any negative edge for each coordinate.

    The two endpoints of distinct edges in one coordinate are disjoint. Each
    negative edge has zero, one or two as-yet-unobserved endpoints. If h edges
    require both endpoints, avoiding them allows choosing at most one member
    per pair. One-endpoint edges can only be avoided by excluding that endpoint.
    """
    n = len(set(votes)-anchors)
    result = []
    for j in range(p):
        bit = 1 << j
        counts = [0,0,0]
        for base,win in votes.items():
            child=base | bit
            if not base & bit and win and child in votes and not votes[child]:
                counts[int(base not in anchors) + int(child not in anchors)] += 1
        if counts[0]:
            result.append(Fraction(1))
            continue
        pairs = counts[2]
        free = n - counts[1] - 2*pairs
        avoid = sum(comb(pairs,k)*2**k*comb(free,budget-k)
                    for k in range(max(0,budget-free), min(pairs,budget)+1))
        result.append(1 - Fraction(avoid,comb(n,budget)))
    return result


def main():
    rows = []
    with threadpool_limits(1):
        for panel,name,seed,(x,y,task,idx) in jobs():
            prep = b.transform(x,idx["fit"])
            fit, vote = idx["fit"],idx["vote"]
            t,v = _targets(y[fit],y[vote],task)
            ballot = RidgeBallot(prep.transform(x[fit]),t,prep.transform(x[vote]),v,1.,.95)
            intervals = _brackets(ballot,4,np.random.default_rng(seed))
            anchors = set(ballot.votes)
            interior = {s for chain in intervals for s in chain[1:-1]}-anchors
            refine_brackets(lambda mask:_query(ballot,mask),intervals)
            q = len(ballot.votes)-len(anchors)
            cut, _ = sharp_upper_bounds(ballot.votes,x.shape[1])
            for mask in sorted(interior):
                _query(ballot,mask)
            chance = detection_probabilities(x.shape[1],ballot.votes,anchors,q)
            expected = sum(chance)
            rows.append(dict(panel=panel,source=name,seed=seed,anchors=len(anchors),interior=len(interior),
                             budget=q,chain=int(sum(cut==-1)),uniform_expected=float(expected),
                             exact_fraction=str(expected)))
    data = pd.DataFrame(rows)
    folder=b.EVIDENCE/"chain-replay"
    folder.mkdir(parents=True,exist_ok=True)
    data.to_csv(folder/"uniform-expectation.csv",index=False)
    print(data.groupby("panel")[["budget","chain","uniform_expected"]].sum())
    print(data[data.budget>0].to_string(index=False))


if __name__=="__main__":
    main()
