"""Peng, Long & Ding (2005), Eq. (7), for integer-coded discrete classification."""

from operator import index

import numpy as np
from sklearn.metrics import mutual_info_score


def _integer_categories(values, name):
    values = np.asarray(values)
    if values.dtype.kind in "biu":
        return values
    if values.dtype.kind == "f" and np.isfinite(values).all() and np.equal(values, np.floor(values)).all():
        return values
    raise ValueError(f"{name} must contain complete integer-coded discrete categories")


def _information(codes, levels, marginal_terms, response, count_log_count):
    """All I(feature; response), in nats, using exact observed cell counts."""
    n = response.size
    response_states = int(response.max()) + 1
    response_term = count_log_count[np.bincount(response)].sum()
    information = np.empty(len(codes))
    # Bound temporary joint tables and index arrays. Sparse contingency counts
    # avoid a dense product of alphabets for an individual high-cardinality pair.
    max_cells = 1_000_000
    for first in range(0, len(codes), max(1, max_cells // n)):
        stop = min(len(codes), first + max(1, max_cells // n))
        sizes = levels[first:stop] * response_states
        if sizes.sum() > max_cells:
            for j in range(first, stop):
                information[j] = mutual_info_score(codes[j], response)
            continue
        offsets = sizes.cumsum() - sizes
        joint_ids = offsets[:, None] + codes[first:stop] * response_states + response
        counts = np.bincount(joint_ids.ravel(), minlength=int(sizes.sum()))
        joint_terms = np.add.reduceat(count_log_count[counts], offsets)
        values = (count_log_count[n] + joint_terms - marginal_terms[first:stop] - response_term) / n
        information[first:stop] = np.maximum(values, 0.0)
    return information


def mrmr_mid_order(X_train, y_train, count):
    """Greedy selection order and winning MID scores; no subsampling or binning.

    X and y are integer category codes (integral floats allowed for stored
    discrete levels). Caller declares their discrete semantics. Symmetric or
    duplicate features can tie. The lowest original index wins differences
    within the documented floating-point comparison tolerance.
    """
    X = _integer_categories(X_train, "features")
    y = _integer_categories(y_train, "labels")
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),):
        raise ValueError("Expected a nonempty matrix with aligned one-dimensional labels")
    if isinstance(count, (bool, np.bool_)) or not 1 <= index(count) <= X.shape[1]:
        raise ValueError("count must be an integer between one and the number of columns")
    response = np.unique(y, return_inverse=True)[1]
    if response.max() == 0:
        raise ValueError("Classification requires at least two observed classes")
    # Relabel each alphabet to consecutive IDs. Empirical MI is invariant to
    # this relabeling, including arbitrary gaps or negative category labels.
    codes = np.array([np.unique(column, return_inverse=True)[1] for column in X.T])
    levels = codes.max(axis=1) + 1
    counts = np.arange(len(X) + 1, dtype=float)
    count_log_count = counts * np.log(np.maximum(counts, 1.0))
    marginals = np.array([count_log_count[np.bincount(column)].sum() for column in codes])
    relevance = _information(codes, levels, marginals, response, count_log_count)
    redundancy = np.zeros(X.shape[1])
    remaining = np.ones(X.shape[1], dtype=bool)
    order, winning_scores = [], []
    tolerance = 32 * np.finfo(float).eps * max(np.log(len(X)), 1.0)
    for step in range(count):
        criterion = relevance.copy() if step == 0 else relevance - redundancy / step
        criterion[~remaining] = -np.inf
        best = np.flatnonzero(criterion >= criterion.max() - tolerance)[0]
        order.append(best)
        winning_scores.append(criterion[best])
        remaining[best] = False
        if step + 1 < count:
            redundancy[remaining] += _information(
                codes[remaining], levels[remaining], marginals[remaining],
                codes[best], count_log_count)
    return np.asarray(order), np.asarray(winning_scores)
