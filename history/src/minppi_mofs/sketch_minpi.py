"""Budget-conditioned minPI using fixed-size feature masks and two training folds."""
from math import ceil
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold

from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.restoration_minpi import _sample_rows
from minppi_mofs.utility_minpi import _endpoint_gains, _utility_inputs


_ROW_CAP = 96


def reduced_width(p, fraction=.75):
    """Round up the deployment fraction, always dropping a column if p >= 2."""
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral) or p < 1:
        raise ValueError("p must be a positive integer")
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real)
            or not 0 < fraction <= 1):
        raise ValueError("fraction must lie in (0, 1]")
    return max(1, min(int(p) - 1, ceil(p * fraction)))


def _budget_coefficients(X, target, count):
    """Fit the expected unit-ridge model over uniformly drawn count-feature masks.

    For M a diagonal mask, minimize E[||target - X M B||² + ||M B||²].
    Let G = X.T X and a = (count-1)/(p-1). Dividing the normal equations by
    Pr(j in M) gives [a G + (1-a) diag(G) + I] B = X.T target.
    The dual form bounds the solve by min(number of sampled rows, p).
    """
    n, p = X.shape
    a = (count - 1) / (p - 1) if p > 1 else 0.
    diag = 1. + (1. - a) * np.einsum("ij,ij->j", X, X)
    if a == 0.:
        return (X.T @ target) / diag[:, None]
    if p <= n:
        gram = a * (X.T @ X)
        gram.flat[::p + 1] += diag
        return solve(gram, X.T @ target, assume_a="pos", check_finite=False)
    scaled = X / diag
    gram = a * (scaled @ X.T)
    gram.flat[::n + 1] += 1.
    return scaled.T @ solve(gram, target, assume_a="pos", check_finite=False)


def _sketch_folds(y, labels, groups, ordered_groups, rng):
    rows = np.arange(len(y))
    if groups is None:
        if len(rows) < 4:
            raise ValueError("Need four training rows for two-fold scoring")
        sample = _sample_rows(rows, 2 * _ROW_CAP, labels, rng)
        sample = rng.permutation(sample)
        if labels is not None:
            # Alternation within class distributes even a pair across the folds.
            sample = sample[np.argsort(labels[sample], kind="stable")]
        left, right = sample[::2], sample[1::2]
        return [(left, right), (right, left)]
    values, codes = np.unique(groups, return_inverse=True)
    if len(values) < 2:
        raise ValueError("Need at least two distinct training groups")
    if ordered_groups:
        pools = [(rows[codes < i], rows[codes == i])
                 for i in range(max(1, len(values) - 2), len(values))]
    else:
        pools = GroupKFold(2).split(rows, y, codes)
    return [(_sample_rows(fit, _ROW_CAP, labels, rng),
             _sample_rows(held, _ROW_CAP, labels, rng)) for fit, held in pools]


def _validate_sampling(y, random_state, groups, ordered_groups):
    if (isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral)
            or not 0 <= random_state < 2**32):
        raise ValueError("Use an integer seed in [0, 2**32)")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape or pd.isna(groups).any():
            raise ValueError("Groups must be complete and aligned with training rows")
    if ordered_groups and groups is None:
        raise ValueError("ordered_groups requires group identifiers")
    return groups


def _prepared_folds(X, y, task, random_state, groups, ordered_groups):
    """Yield indices, standardized feature arrays, and fit-centered target arrays."""
    labels = y if task == "classification" else None
    folds = _sketch_folds(y, labels, groups, ordered_groups, np.random.default_rng(random_state))
    for fit, held in folds:
        train, check = _standardize(np.asarray(X[fit], dtype=float), np.asarray(X[held], dtype=float))
        if labels is None:
            target, query = y[fit, None].astype(float), y[held, None].astype(float)
        else:
            classes = np.unique(y[fit])
            target = (y[fit, None] == classes).astype(float)
            # An unseen held class has a constant omitted coordinate in both losses.
            query = (y[held, None] == classes).astype(float)
        center = target.mean(axis=0)
        centered = target - center
        centered[:, np.ptp(target, axis=0) == 0] = 0.
        yield fit, held, train, check, centered, query - center


def sketch_top_from_training(X, y, count, *, task="classification", random_state=1129,
                             groups=None, ordered_groups=False):
    """Return sorted training-selected column indices and auditable fold diagnostics.

    Each training fold fits one budget-conditioned additive model. Classification
    uses squared error on one-hot labels. For each held row the change in squared
    loss is its signed pivotal vote integrated over the loss threshold. The
    expected vote for k uniform peers interpolates linearly between the zero-
    and all-peer votes. Take the smaller of two near-budget strata after averaging
    within each held fold and weighting the folds equally. Input order breaks ties.

    Ordered groups must sort chronologically. Each of the last two eligible
    groups is assessed using only earlier groups. With just two groups there is
    one forward assessment. For unordered groups the two folds are group-disjoint.
    Each side uses at most 96 rows, sampled proportionally by class when possible.

    For b bounded fit/held rows, d=min(b,p) and c target coordinates, per-fold
    matrix work is O(bp(d+c)+d³) time and O(bp+d²+(p+b)c) space. Sampling also
    inspects the supplied label/group vectors. Column ranking costs O(p log p).
    """
    X, y = _utility_inputs(X, y, count, task)
    p = X.shape[1]
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    low, high = max(0, count - 2), count - 1
    if count == 1 and p > 1:
        high = 1
    info = {"ballot": "budget_conditioned_loss", "budget": int(count),
            "peer_window": (low, high), "row_cap": _ROW_CAP, "folds": [],
            "strategy": "forward" if ordered_groups else "group" if groups is not None else "row"}
    if count == p:
        return np.arange(p), info
    endpoints = np.zeros((2, p))
    folds = _prepared_folds(X, y, task, random_state, groups, ordered_groups)
    for fit, held, train, check, centered, query in folds:
        coef = _budget_coefficients(train, centered, count)
        endpoints += np.asarray(_endpoint_gains(check, query, coef)) / len(held)
        info["folds"].append({"fit_rows": fit, "held_rows": held})
    zero, full = endpoints / len(info["folds"])
    slope = (full - zero) / (p - 1)
    scores = np.minimum(zero + low * slope, zero + high * slope)
    info["scores"] = scores
    return np.sort(np.argsort(-scores, kind="stable")[:count]), info
