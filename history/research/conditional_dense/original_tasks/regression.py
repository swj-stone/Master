"""Continuous-response extension of the two-tier conditional ridge selector."""

from math import ceil, isqrt

import numpy as np

from research.conditional_dense.shared_method import _nested
from research.conditional_dense.vote_width.method import effective_width


def _regression_ranking(X, y, seed):
    X, y = np.asarray(X), np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),):
        raise ValueError("Expected aligned dense features and a continuous response")
    if not np.isfinite(y).all():
        raise ValueError("Expected finite regression targets")

    n, p = X.shape
    sample_size = min(n, 768)
    rng = np.random.default_rng(seed)
    rows = (np.sort(rng.choice(n, sample_size, replace=False))
            if sample_size < n else np.arange(n))
    sketch = np.array(X[rows], dtype=np.float32, order="C")
    if not np.isfinite(sketch).all():
        raise ValueError("Expected finite selected training features")
    sketch -= sketch.mean(axis=0, dtype=np.float64)
    norms = np.linalg.norm(sketch, axis=0)
    sketch /= np.where(norms > 0, norms / np.sqrt(np.float32(sample_size)), 1.)

    # Scaling by a common factor preserves all gain ratios and support widths.
    # Start at unit amplitude to avoid overflow in the variance calculation.
    response = y[rows]
    amplitude = np.max(np.abs(response))
    if amplitude > 0:
        response = response / amplitude
    response -= response.mean()
    scale = np.sqrt(np.mean(response ** 2))
    if scale > 0:
        response /= scale
    target = response.astype(np.float32).reshape(-1, 1)

    high = min(p - 1, max(1, p // 4), 4 * isqrt(sample_size))
    tiers = sorted({high // 2, high})
    order = rng.permutation(p)
    votes, residual = _nested(sketch, target, order, tiers, np.sqrt(sample_size))
    remaining = np.sqrt(np.clip(residual[order[high:]] / sample_size, 0., 1.))
    geometric_width = min(p, max(1, high + ceil(float(remaining.sum()))))
    return dict(rank=votes.min(axis=0), width=geometric_width, tiers=tiers,
                tier_scores=votes, sketch_size=sample_size)


def regression_geometric_support(X, y, *, seed=1129):
    """Select original columns using the geometric width and continuous-target votes."""
    info = _regression_ranking(X, y, seed)
    info["selected_width"] = info["width"]
    order = np.argsort(-info["rank"], kind="stable")
    return np.sort(order[:info["selected_width"]]), info


def regression_support(X, y, *, seed=1129):
    """Return sorted column indices and both widths from the sampled training data.

    With a constant sampled response, every vote is zero. The concentration
    rule then retains the geometric width, breaking ties by original index.
    """
    info = _regression_ranking(X, y, seed)
    info["selected_width"] = effective_width(info["rank"], info["width"])
    order = np.argsort(-info["rank"], kind="stable")
    return np.sort(order[:info["selected_width"]]), info
