"""Conditional minPI using exact unit changes in the refitted ridge objective.

Peer sizes are fixed. A shared random order supplies a uniform peer subset for
each candidate by skipping that candidate. Each class and permutation votes
the decrease in penalized squared error. The score is the smaller of the
mean gains at the two sizes.
"""
from math import ceil, isqrt

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def _gains(X, target, order, size, penalty):
    """All one-feature gains; X, target are centered, X is standardized."""
    n, p = X.shape
    diag = np.einsum("ij,ij->j", X, X)
    cross_y = X.T @ target
    if size == 0:
        return (cross_y**2).sum(axis=1) / (diag+penalty), diag
    anchor = order[:size]
    A = np.ascontiguousarray(X[:, anchor])
    gram = A.T @ A
    gram.flat[::size+1] += penalty
    factor = cho_factor(gram, lower=True, check_finite=False)
    cross = A.T @ X
    projected = solve_triangular(factor[0], cross, lower=True, check_finite=False)
    ratio = np.maximum(0., diag - np.einsum("ij,ij->j", projected, projected))
    residual = cross_y - cross.T @ cho_solve(factor, cross_y[anchor], check_finite=False)
    votes = (residual**2).sum(axis=1) / (ratio+penalty)
    # For members of A use the next peer in the random order instead. All such
    # exact addition gains are deletion costs in a single (s+1)-column system.
    aug = order[:size+1]
    B = np.ascontiguousarray(X[:, aug])
    gram = B.T @ B
    gram.flat[::len(aug)+1] += penalty
    factor = cho_factor(gram, lower=True, check_finite=False)
    beta = cho_solve(factor, cross_y[aug], check_finite=False)
    linv = solve_triangular(factor[0], np.eye(len(aug)), lower=True, check_finite=False)
    invdiag = np.einsum("ij,ij->j", linv, linv)
    votes[anchor] = np.sum(beta[:-1]**2, axis=1) / invdiag[:-1]
    ratio[anchor] = np.maximum(0., 1/invdiag[:-1]-penalty)
    return votes, ratio


def conditional_support(X, y, count=None, *, seed=1129, repetitions=2, row_limit=768):
    X, y = np.asarray(X), np.asarray(y)
    n, p = X.shape
    if min(n,p) < 1 or y.shape != (n,) or (count is not None and
            (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int,np.integer))
             or not 1 <= count <= p)):
        raise ValueError("Invalid matrix, target or width")
    classes, labels = np.unique(y, return_inverse=True)
    if not 2 <= len(classes) <= row_limit or repetitions < 1:
        raise ValueError("Expected class labels and at least one peer draw")
    rng = np.random.default_rng(seed)
    if n > row_limit:
        # One random row per class and a uniform sample of remaining training rows.
        first = np.array([rng.choice(np.flatnonzero(labels == k)) for k in range(len(classes))])
        pool = np.setdiff1d(np.arange(n), first, assume_unique=True)
        rows = np.sort(np.r_[first, rng.choice(pool, row_limit-len(first), replace=False)])
    else:
        rows = np.arange(n)
    sample = np.array(X[rows], dtype=float, order="C")
    if not np.isfinite(sample).all():
        raise ValueError("Requires finite numeric training values")
    sample -= sample.mean(axis=0)
    norm = np.linalg.norm(sample, axis=0)
    sample /= np.where(norm == 0, 1., norm / np.sqrt(len(sample)))
    target = (labels[rows,None] == np.arange(len(classes))).astype(float)
    target -= target.mean(axis=0)
    scale = isqrt(p-1)+1
    sizes = sorted({min(p-1, scale), min(p-1, 2*scale)})
    estimates = np.zeros((len(sizes), p))
    unrepresented = np.zeros(p)
    penalty = np.sqrt(len(sample))
    for _ in range(repetitions):
        order = rng.permutation(p)
        for i, size in enumerate(sizes):
            vote, residual = _gains(sample, target, order, size, penalty)
            estimates[i] += vote
        # The anchors are retained as a dimension allowance, not forced into the support.
        outsiders = order[size:]
        unrepresented[outsiders] += np.minimum(1.,residual[outsiders]/len(sample))
    score = estimates.min(axis=0)
    auto = min(p,max(1,sizes[-1]+ceil(unrepresented.sum()/repetitions)))
    width = auto if count is None else count
    selected = np.sort(np.argsort(-score,kind="stable")[:width])
    return selected, {"tiers":sizes, "rank":score, "tier_scores":estimates/repetitions,
                      "width":int(width), "proxy_dimension":auto, "sketch_size": len(rows)}
