"""Pivotal votes integrated over the held-out squared-loss threshold."""
import numpy as np
import pandas as pd
from scipy.linalg import solve

from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.piv2 import _folds


def _utility_inputs(X, y, count, task):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),) or pd.isna(y).any():
        raise ValueError("Expected a nonempty matrix with aligned, complete targets")
    if task not in ("classification", "regression"):
        raise ValueError("Expected classification or regression")
    if task == "regression" and not np.isfinite(y.astype(float)).all():
        raise ValueError("Regression requires finite targets")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= X.shape[1]:
        raise ValueError("count must be an integer in [1,p]")
    return X, y


def _fitted_halves(X, y, task, random_state):
    """Yield held-out inputs, centered held-out targets, and the opposite fold's ridge coefficients."""
    cap = min(1024, max(256, 2 * X.shape[1]))
    folds = _folds(y, task, np.random.default_rng(random_state), row_limit=cap)
    if folds is None or min(map(len, folds)) < 2:
        raise ValueError("Insufficient training observations for cross-fitted utility votes")
    classes = np.unique(y) if task == "classification" else None
    for fit, held in (folds, folds[::-1]):
        train, check = _standardize(np.asarray(X[fit], dtype=float), np.asarray(X[held], dtype=float))
        if task == "classification":
            target = (y[fit, None] == classes).astype(float)
            query = (y[held, None] == classes).astype(float)
        else:
            target = np.asarray(y[fit, None], dtype=float)
            query = np.asarray(y[held, None], dtype=float)
        center = target.mean(axis=0)
        target = target - center
        target[:, np.ptp(target, axis=0) == 0] = 0.
        # Solve the smaller of the primal and dual unit-penalty ridge systems.
        if train.shape[1] <= train.shape[0]:
            gram = train.T @ train
            gram.flat[::len(gram) + 1] += 1.
            coefficients = solve(gram, train.T @ target, assume_a="pos", check_finite=False)
        else:
            gram = train @ train.T
            gram.flat[::len(gram) + 1] += 1.
            coefficients = train.T @ solve(gram, target, assume_a="pos", check_finite=False)
        yield check, query - center, coefficients


def _endpoint_gains(check, centered_target, coefficients):
    """Sum integrated votes for zero peers and for all other features as peers."""
    norm = np.square(check).sum(axis=0) * np.square(coefficients).sum(axis=1)
    zero = 2 * (check * (centered_target @ coefficients.T)).sum(axis=0) - norm
    residual = centered_target - check @ coefficients
    full = 2 * (check * (residual @ coefficients.T)).sum(axis=0) + norm
    return zero, full


def utility_top_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """Rank by worst expected size-conditioned pivotal gain in held-out squared loss.

    A row votes for coalition S at threshold t when its frozen, mean-masked joint
    ridge has squared loss L(S) <= t. Integrating the signed vote change on adding
    j over t >= 0 gives L(S) - L(S+j). For a uniform k-peer set excluding j the
    expected gain interpolates linearly between zero peers and p-1 peers. Thus
    the two endpoints suffice for the minimum over any declared size interval.
    The default covers every coalition size met while assembling count features,
    from zero to count-1. Computation uses double precision.
    """
    X, y = _utility_inputs(X, y, count, task)
    p = X.shape[1]
    if tiers is None:
        low, high = 0, count - 1
    else:
        strata = _selection_tiers(tiers, p)
        low, high = min(strata), max(strata)
    info = {"ballot": "integrated_loss_threshold", "peer_window": (low, high),
            "arithmetic": "float64", "ballot_rows": 0}
    if count == p:
        return np.arange(p), info
    zero, full = np.zeros(p), np.zeros(p)
    for check, target, coefficients in _fitted_halves(X, y, task, random_state):
        empty_gain, full_gain = _endpoint_gains(check, target, coefficients)
        zero += empty_gain
        full += full_gain
        info["ballot_rows"] += len(check)
    slope = (full - zero) / (p - 1)
    worst = np.minimum(zero + low * slope, zero + high * slope) / info["ballot_rows"]
    support = np.argsort(-worst, kind="stable")[:count]
    return np.sort(support), info
