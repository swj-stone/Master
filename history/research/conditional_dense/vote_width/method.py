"""Truncate conditional minPI's own ranking by its effective number of votes."""
from math import ceil

import numpy as np

from research.conditional_dense.shared_method import conditional_support


def effective_width(scores, original_width):
    """Hill number of order two for nonnegative weights, capped at the old width."""
    weights = np.asarray(scores, dtype=np.float64)
    if weights.ndim != 1 or not 1 <= original_width <= len(weights):
        raise ValueError("Expected one weight per feature and a valid original width")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Votes must be finite and nonnegative")
    largest = weights.max()
    if largest == 0:
        return original_width
    weights = weights / largest
    return min(original_width, ceil(float(weights.sum() ** 2 / (weights @ weights))))


def vote_support(X, y, *, seed=1129):
    original, info = conditional_support(X, y, seed=seed)
    width = effective_width(info["rank"], info["width"])
    if width == len(original):
        return original, info
    # The original support was returned in column order. Stable sort within it
    # exactly reproduces the original ranking prefix including tied scores.
    ranking = np.argsort(-info["rank"][original], kind="stable")
    return np.sort(original[ranking[:width]]), info
