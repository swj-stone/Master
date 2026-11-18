"""Run the frozen dense conditional-vote selector on the same rows of a CSR input."""
from math import ceil, isqrt

import numpy as np
from scipy import sparse

from research.conditional_dense.shared_method import _nested
from research.conditional_dense.vote_width.method import effective_width


def csr_vote_support(X, y, *, seed):
    """Preserve the frozen wide classifier's row sampling, peer order and ridge votes."""
    y = np.asarray(y)
    if not sparse.issparse(X) or X.format != "csr" or min(X.shape) < 1 or y.shape != (X.shape[0],):
        raise ValueError("Expected a nonempty CSR matrix and aligned classification labels")
    n, p = X.shape
    classes, ids = np.unique(y, return_inverse=True)
    limit = min(768, n)
    if p <= 32 or not 2 <= len(classes) <= limit:
        raise ValueError("Expected wide features and at least two classes")
    rng = np.random.default_rng(seed)
    if limit < n:
        first = [rng.choice(np.flatnonzero(ids == c)) for c in range(len(classes))]
        remaining = np.setdiff1d(np.arange(n), first, assume_unique=True)
        rows = np.sort(np.r_[first, rng.choice(remaining, limit-len(first), replace=False)])
    else:
        rows = np.arange(n)
    # Only this sketch crosses the CSR-to-dense boundary. For every later
    # operation, preserve the frozen classifier's data type and operation order.
    Z = X[rows].astype(np.float32, copy=False).toarray(order="C")
    if not np.isfinite(Z).all():
        raise ValueError("Expected finite sampled features")
    Z -= Z.mean(axis=0, dtype=np.float64)
    norms = np.linalg.norm(Z, axis=0)
    Z /= np.where(norms > 0, norms / np.sqrt(np.float32(limit)), 1.)
    Y = (ids[rows, None] == np.arange(len(classes))).astype(np.float32)
    Y -= Y.mean(axis=0)
    high = min(p-1, max(1, p//4), 4*isqrt(limit))
    sizes = sorted({high//2, high})
    order = rng.permutation(p)
    votes, residual = _nested(Z, Y, order, sizes, np.sqrt(limit))
    unexplained = np.sqrt(np.clip(residual[order[high:]]/limit, 0, 1))
    geometric = min(p, max(1, high + ceil(float(unexplained.sum()))))
    scores = votes.min(axis=0)
    width = effective_width(scores, geometric)
    columns = np.sort(np.argsort(-scores, kind="stable")[:width])
    return columns, dict(rank=scores, tier_scores=votes, tiers=sizes, sketch_size=limit,
                         width=geometric, selected_width=width, proxy_dimension=geometric)
