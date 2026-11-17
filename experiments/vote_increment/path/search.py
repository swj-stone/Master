"""Nominate native Gain prefixes using path-unanimity votes and a common refit guard."""
import json
from math import comb

import numpy as np


def curves(model, order):
    """Expected vote mass lost by a deleted suffix under three coalition sizes."""
    p = len(order)
    rank = np.argsort(order)
    trees = json.loads(model.get_booster().save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees']
    groups = {}
    for tree in trees:
        stack = [(0, frozenset())]
        while stack:
            node, ancestors = stack.pop()
            left = tree['left_children'][node]
            if left < 0:
                continue
            path = ancestors | {int(rank[tree['split_indices'][node]])}
            key = (max(path), len(path))
            groups[key] = groups.get(key, 0.) + max(0., tree['loss_changes'][node])
            stack.extend([(left, path), (tree['right_children'][node], path)])
    votes = []
    for s in sorted({max(1, round(p * q)) for q in (.25, .5, .75)}):
        mass = np.zeros(p)
        for (j, degree), gain in groups.items():
            mass[j] += gain * comb(s, degree) / comb(p, degree)
        votes.append(np.cumsum(mass) / mass.sum() if mass.sum() else np.arange(1, p+1)/p)
    return np.clip(1 - np.array(votes), 0., 1.)


def loss_profile(model, order, arm):
    if arm == 'gain':
        totals = model.get_booster().get_score(importance_type='total_gain')
        mass = np.array([totals.get(f'f{j}', 0.) for j in order])
        return 1 - (np.cumsum(mass)/mass.sum() if mass.sum() else np.arange(1,len(order)+1)/len(order))
    rates = curves(model, order)
    if arm == 'min':
        return rates.min(axis=0)
    if arm == 'mean':
        return rates.mean(axis=0)
    raise ValueError(arm)


def search(profile, measure, tolerance=.005):
    """Calibrate vote loss with one real low-width fit; correct once across the bracket."""
    p = len(profile)
    trace = {p: measure(p)}
    floor = trace[p] - tolerance
    if p == 1:
        return 1, trace
    trace[1] = measure(1)
    if trace[1] >= floor:
        return 1, trace
    for _ in range(3):
        hi = min(k for k, value in trace.items() if value >= floor)
        lo = max(k for k, value in trace.items() if k < hi and value < floor)
        if hi - lo == 1:
            break
        fraction = (floor - trace[lo]) / (trace[hi] - trace[lo])
        target = (1-fraction)*profile[lo-1] + fraction*profile[hi-1]
        candidates = np.arange(lo+1, hi)
        mid = int(candidates[np.argmin(np.abs(profile[candidates-1]-target))])
        trace[mid] = measure(mid)
    return min(k for k, value in trace.items() if value >= floor), trace
