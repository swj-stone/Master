"""Locate witnessed negative MinPI edges between nested quota-crossing supports."""
from operator import index

import numpy as np

from .dictator_minpi import RidgeBallot, sharp_upper_bounds
from .sampled_linear_minpi import Ranking, _targets


def _columns(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _query(ballot, mask):
    if mask not in ballot.votes:
        ballot.base_vote(np.fromiter(_columns(mask), dtype=int), mask)
    return ballot.votes[mask]


def _brackets(ballot, rounds, rng):
    p = len(ballot.bits)
    blocks = min(4, p)
    intervals = []
    for _ in range(rounds):
        partition = np.array_split(rng.permutation(p), blocks)
        masks = [sum(ballot.bits[j] for j in group) for group in partition]
        for shift in range(blocks):
            left = 0
            for k in range(blocks):
                group = (shift + k) % blocks
                right = left | masks[group]
                after = _query(ballot, right)
                if ballot.votes[left] and not after:
                    chain = [left]
                    for feature in partition[group]:
                        chain.append(chain[-1] | ballot.bits[feature])
                    intervals.append(chain)
                left = right
    return intervals


def refine_brackets(query, brackets):
    """A winning left and losing right preserve a downward edge after either answer."""
    for chain in brackets:
        left, right = 0, len(chain) - 1
        while right - left > 1:
            mid = (left + right) // 2
            if query(chain[mid]):
                left = mid
            else:
                right = mid


def rank_chain(X_fit, y_fit, X_vote, y_vote, *, task, seed=42, rounds=4, alpha=.95,
               ridge=1., random_queries=None):
    """Return the sharp Boolean-completion upper bounds from a cached ridge ballot.

    A negative always has an observed adjacent pair. random_queries activates the
    diagnostic uniform-prefix control with the supplied distinct-query budget.
    """
    x, z = np.asarray(X_fit, dtype=float), np.asarray(X_vote, dtype=float)
    if (x.ndim != 2 or z.ndim != 2 or not x.shape[1] or z.shape[1] != x.shape[1]
            or len(x) != len(y_fit) or len(z) != len(y_vote) or not min(len(x), len(z))
            or not np.isfinite(x).all() or not np.isfinite(z).all()):
        raise ValueError("Supply finite fitting and voting matrices with shared width and aligned responses")
    if not (np.isfinite(ridge) and ridge > 0 and 0 < alpha <= 1 and index(rounds) >= 1):
        raise ValueError("Require ridge > 0, 0 < alpha <= 1, and at least one round")
    y, truth = _targets(y_fit, y_vote, task)
    ballot = RidgeBallot(x, y, z, truth, ridge, alpha)
    rng = np.random.default_rng(seed)
    intervals = _brackets(ballot, rounds, rng)
    anchors = len(ballot.votes)
    if random_queries is None:
        refine_brackets(lambda mask: _query(ballot, mask), intervals)
    else:
        interior = sorted({mask for chain in intervals for mask in chain[1:-1]} - ballot.votes.keys())
        for pos in rng.choice(len(interior), index(random_queries), replace=False):
            _query(ballot, interior[pos])
    p = len(ballot.bits)
    upper, edges = sharp_upper_bounds(ballot.votes, p)
    work = dict(anchors=anchors, descending_brackets=len(intervals),
                refinement_queries=len(ballot.votes)-anchors,
                distinct_votes=len(ballot.votes), factorizations=ballot.factors)
    return Ranking(np.lexsort((np.arange(p), -ballot.tie, -upper)), upper, upper, ballot.tie,
                   work, [edges.get(i) for i in range(p)])
