"""Conditional feature selection by worst-stratum nearest-neighbour votes."""

import numpy as np
from minppi_mofs.nonlinear_vote import _triplets


def swings(base, delta, cols, limit):
    """Count net changes in strict triplet votes; stable rows contribute zero."""
    active = (base > -limit) & (base <= limit)
    block = delta[np.ix_(active, cols)]
    offset = base[active]
    return np.count_nonzero(block + offset[:, None] > 0, axis=0) - np.count_nonzero(offset > 0)


def next_feature(states, delta, cols, limit, rule):
    """Maximize the stratum minimum, breaking ties by original column ID."""
    if rule == "current":
        states = states[-1:]
    aggregates = swings(states[-1], delta, cols, limit)
    work = len(cols)
    if rule == "lazy":
        # Evaluate the leading candidate completely to establish a valid lower bound.
        leader = int(np.argmax(aggregates))
        bound = aggregates[leader]
        for state in states[:-1]:
            bound = min(bound, swings(state, delta, cols[leader:leader+1], limit)[0])
        work += len(states) - 1
        candidates = np.arange(len(cols))
        for state in states[:-1]:
            candidates = candidates[(aggregates[candidates] > bound) |
                        ((aggregates[candidates] == bound) & (cols[candidates] <= cols[leader]))]
            aggregates[candidates] = np.minimum(aggregates[candidates],
                                                swings(state, delta, cols[candidates], limit))
            work += len(candidates)
        eligible = candidates
        return cols[eligible[np.argmax(aggregates[eligible])]], work
    for state in states[:-1]:
        step = swings(state, delta, cols, limit)
        aggregates = aggregates + step if rule == "pooled" else np.minimum(aggregates, step)
        work += len(cols)
    return cols[np.argmax(aggregates)], work


def rank_triplet_path(x, y, z, truth, *, task, seed, rule="lazy", path_fraction=1.):
    allowed = {"lazy", "minimum", "pooled", "current", "margin"}
    if rule not in allowed:
        raise ValueError(rule)
    diffs = _triplets(x, y, z, truth, seed, task)
    units = 2 * min(len(x), 2048)
    delta = np.rint(diffs * units).astype(np.int32)
    p = delta.shape[1]
    available = np.arange(p)
    if rule == "margin":
        return np.lexsort((available, -delta.sum(axis=0))), {"votes":0}
    order, work = [], 0
    history = [np.zeros(len(delta), dtype=np.int32)]
    # Largest/smallest candidate contribution bounds the possible change on each row.
    limit = np.abs(delta).max(axis=1)
    for k in range(min(p - 1, int(np.ceil(path_fraction*p)))):
        states = [history[j] for j in sorted({k//4, k//2, k})]
        best, queries = next_feature(states, delta, available, limit, rule)
        order.append(best)
        work += queries
        history.append(history[-1] + delta[:, best])
        available = available[available != best]
    return np.r_[order, available].astype(int), {"votes":work, "triplets":len(delta)}
