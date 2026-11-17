"""Refine sampled MinPI upper bounds with coalition-level dictator exclusions.

rank_features in sampled_linear_minpi remains the canonical frozen benchmark.
This implementation caches observed coalition votes. The local mode is its
matched ablation: it uses the same cache and only per-feature sampled minima.
"""
from operator import index

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .sampled_linear_minpi import Ranking, _targets, _utility, coalition_batches


class RidgeBallot:
    """Binary quota votes using a shared Cholesky factor within each target batch."""

    def __init__(self, fit, y, vote, truth, ridge, alpha):
        zmean, ymean = fit.mean(axis=0), y.mean(axis=0)
        x, self.z, self.truth = fit - zmean, vote - zmean, truth - ymean
        self.gram = x.T @ x
        self.gram.flat[::x.shape[1]+1] += ridge
        self.xy = x.T @ (y - ymean)
        self.scale = self.loss(self.truth) or 1.
        full = np.arange(x.shape[1])
        self.factors = 0
        _, beta, error = self.fit(full)
        self.tie = np.linalg.norm(beta, axis=1)
        self.quota = alpha * _utility(self.loss(error), self.scale)
        self.bits = [1 << i for i in full.tolist()]
        self.votes = {0: _utility(self.loss(self.truth), self.scale) >= self.quota,
                      (1 << len(full))-1: True}
        self.endpoints = 0
        self.boundaries = 0

    def loss(self, residual):
        return np.einsum("ij,ij->", residual, residual) / len(self.z)

    def fit(self, base):
        if not len(base):
            return None, None, self.truth
        self.factors += 1
        factor = cho_factor(self.gram[np.ix_(base, base)], check_finite=False)
        beta = cho_solve(factor, self.xy[base], check_finite=False)
        return factor, beta, self.truth - self.z[:, base] @ beta

    def base_vote(self, base, mask):
        if mask in self.votes:
            return self.votes[mask], None
        state = self.fit(base)
        self.votes[mask] = _utility(self.loss(state[2]), self.scale) >= self.quota
        return self.votes[mask], state

    def extensions(self, base, mask, targets, state):
        keys = [mask | self.bits[i] for i in targets]
        missing = [pos for pos, key in enumerate(keys) if key not in self.votes]
        if missing:
            active = targets[missing]
            factor, beta, error = self.fit(base) if state is None else state
            denom = self.gram[active, active]
            rhs = self.xy[active]
            direction = self.z[:, active]
            if len(base):
                links = self.gram[np.ix_(base, active)]
                proj = cho_solve(factor, links, check_finite=False)
                denom = denom - np.einsum("ij,ij->j", proj, links)
                rhs = rhs - links.T @ beta
                direction = direction - self.z[:, base] @ proj
            slopes = rhs / denom[:, None]
            dot = np.einsum("ij,ij->i", direction.T @ error, slopes)
            sq = np.einsum("ij,ij->j", direction, direction) * np.einsum("ij,ij->i", slopes, slopes)
            after = _utility(self.loss(error) + (sq - 2*dot) / len(self.z), self.scale)
            for j in np.flatnonzero(np.abs(after-self.quota) <= 64*np.finfo(float).eps):
                cols = np.sort(np.append(base, active[j]))
                after[j] = _utility(self.loss(self.fit(cols)[2]), self.scale)
                self.boundaries += 1
            self.endpoints += len(active)
            for pos, passed in zip(missing, after >= self.quota, strict=True):
                self.votes[keys[pos]] = passed
        return np.array([self.votes[k] for k in keys])


def exclude(bounds, previous, remaining):
    dropped = previous ^ remaining
    while dropped:
        bit = dropped & -dropped
        j = bit.bit_length()-1
        bounds[j] = min(bounds[j], 0)
        dropped ^= bit


def sharp_upper_bounds(votes, p):
    """Coordinatewise largest MinPI attainable by a Boolean completion of votes."""
    possible = unresolved = (1 << p)-1
    edges = {}
    for mask, won in votes.items():
        possible &= mask if won else ~mask
        if won:
            continue
        bits = mask & unresolved
        while bits:
            bit = bits & -bits
            parent = mask ^ bit
            if votes.get(parent, False):
                j = bit.bit_length()-1
                edges[j] = tuple(i for i in range(p) if parent & (1 << i))
                unresolved ^= bit
            bits ^= bit
    upper = np.array([int(bool(possible & (1 << i))) for i in range(p)], dtype=np.int8)
    for i in edges:
        upper[i] = -1
    return upper, edges


def rank_bounds(X_fit, y_fit, X_vote, y_vote, *, task, seed=42, rounds=4, alpha=.95,
                ridge=1., cross_coalition=True, closure=False):
    """Upper-bound each exact MinPI. A -1 always records an actual sampled edge.

    If cross_coalition is True, +1 remains possible only for a feature whose
    membership agreed with every observed coalition vote. This can tighten the
    sampled-edge minimum even on schedules without a zero edge for that feature.
    If closure is True, also inspect every coincident edge in the final cache.
    """
    x, z = np.asarray(X_fit, dtype=float), np.asarray(X_vote, dtype=float)
    if (x.ndim != 2 or z.ndim != 2 or not len(x) or not len(z) or not x.shape[1]
            or x.shape[1] != z.shape[1] or len(y_fit) != len(x) or len(y_vote) != len(z)
            or not np.isfinite(x).all() or not np.isfinite(z).all()):
        raise ValueError("Fitting and voting arrays must have matching lengths, finite values and shared width")
    if not (0 < alpha <= 1 and np.isfinite(ridge) and ridge > 0 and index(rounds) >= 1):
        raise ValueError("Require positive ridge, a quota in (0,1], and at least one round")
    y, truth = _targets(y_fit, y_vote, task)
    ballot = RidgeBallot(x, y, z, truth, ridge, alpha)
    p = len(ballot.bits)
    bounds = np.ones(p, dtype=np.int8)
    witnesses = [None] * p
    possible = 0 if ballot.votes[0] else (1 << p)-1
    if cross_coalition and not possible:
        bounds.fill(0)
    scheduled = used = 0
    for base, targets in coalition_batches(p, rounds, seed):
        scheduled += len(targets)
        active = targets[bounds[targets] >= 0]
        if not len(active):
            continue
        mask = sum(ballot.bits[j] for j in base)
        before, state = ballot.base_vote(base, mask)
        old = possible
        if cross_coalition and possible:
            possible &= mask if before else ~mask
            exclude(bounds, old, possible)
        if not before:
            active = active[bounds[active] > 0]
        if not len(active):
            continue
        used += len(active)
        after = ballot.extensions(base, mask, active, state)
        deltas = after.astype(np.int8) - int(before)
        bounds[active] = np.minimum(bounds[active], deltas)
        for j in active[deltas == -1]:
            witnesses[j] = tuple(map(int, base))
        if cross_coalition and possible:
            old = possible
            # Update the remaining candidates from each observed child vote.
            for j, passed in zip(active, after, strict=True):
                child = mask | ballot.bits[j]
                possible &= child if passed else ~child
                if not possible:
                    break
            exclude(bounds, old, possible)
    extra = 0
    if closure:
        bounds, complete = sharp_upper_bounds(ballot.votes, p)
        for j, base in complete.items():
            if witnesses[j] is None:
                witnesses[j] = base
                extra += 1
    work = dict(scheduled_marginals=scheduled, evaluated_marginals=used,
                new_child_solves=ballot.endpoints, distinct_votes=len(ballot.votes),
                factorizations=ballot.factors, boundary_refits=ballot.boundaries,
                incidental_negatives=extra)
    return Ranking(np.lexsort((np.arange(p), -ballot.tie, -bounds)), bounds, bounds,
                   ballot.tie, work, witnesses)
