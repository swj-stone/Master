"""Select columns by held-out ridge votes with many peers and check their combined loss."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve

from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.moment_minpi import _draw, _moment_halves
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


@dataclass
class _Validation:
    x: np.ndarray
    target: np.ndarray
    beta: np.ndarray
    full_loss: float
    tolerance: float


def _direction(train, held, target, query):
    x, z = _standardize(train, held)
    mean = target.mean(axis=0)
    y, actual = target - mean, query - mean
    n, p = x.shape
    penalty = max(1., p / n)
    if p <= n:
        gram = x.T @ x
        gram.flat[::p + 1] += penalty
        beta = solve(gram, x.T @ y, assume_a="pos", check_finite=False)
    else:
        gram = x @ x.T
        gram.flat[::n + 1] += penalty
        beta = x.T @ solve(gram, y, assume_a="pos", check_finite=False)
    residual = actual - z @ beta
    # With component j denoted u_j, the two exact gains are 2<y,u_j>-||u_j||²
    # and 2<full residual,u_j>+||u_j||². Contract columns before expanding classes.
    norm = np.einsum("ij,ij->j", z, z) * np.einsum("ij,ij->i", beta, beta)
    zero = 2 * np.einsum("ij,ij->i", z.T @ actual, beta) - norm
    full = 2 * np.einsum("ij,ij->i", z.T @ residual, beta) + norm
    full_loss = float(np.einsum("ij,ij->", residual, residual))
    tolerance = .01 * float(np.einsum("ij,ij->", actual, actual))
    return zero / len(z), full / len(z), _Validation(z, actual, beta, full_loss, tolerance)


def _width(order, checks):
    p = len(order)
    widths = sorted({*(max(1, (p * i + 7) // 8) for i in range(1, 8)),
                     (15 * p + 15) // 16, p - 1} - {p})
    predictions = [np.zeros_like(check.target) for check in checks]
    start = 0
    for width in widths:
        block = order[start:width]
        acceptable = True
        for check, pred in zip(checks, predictions):
            pred += check.x[:, block] @ check.beta[block]
            error = check.target - pred
            loss = np.einsum("ij,ij->", error, error)
            acceptable = acceptable and loss <= check.full_loss + check.tolerance
        if acceptable:
            return width
        start = width
    return p


def ridge_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                groups=None, ordered_groups=False):
    """Fit ridge on training folds and rank columns by the weaker half/full-peer vote.

    Coefficients and the intercept are frozen within a voting game. A coalition's
    score is its negative squared loss on held rows, using only the contributions
    of member columns. A uniformly drawn k-peer coalition includes each other
    feature with probability k/(p-1), so mean addition gain is affine in k.
    The lower of the two endpoint votes is the worst mean gain over the interval.
    Classification uses one-hot targets. Each direction fits at most 96 rows.

    In automatic mode, try ranked prefixes in eighths plus 15/16 and p-1,
    shortest first. The retained frozen components must lose at most 1% of
    intercept-only held squared loss relative to the full fitted predictor in
    every direction. No positive vote, or no acceptable prefix, keeps all.
    This internal check is not a bound on the outer refitted model's score.
    """
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    low, high = (p - 1) // 2, p - 1
    info = {"peer_window": (low, high), "folds": [],
            "strategy": "forward" if ordered_groups else "group" if groups is not None else "row"}
    if p == 1:
        info.update(scores=np.zeros(1), zero_peer_scores=np.zeros(1), budget=1)
        return np.array([0]), info
    rng = np.random.default_rng(random_state)
    left, right = _moment_halves(y, task, groups, ordered_groups, rng)
    left, right = _draw(left, 96, rng), _draw(right, 96, rng)
    labels = np.r_[y[left], y[right]]
    if task == "classification":
        classes = np.unique(labels)
        targets = (labels[:, None] == classes).astype(float)
    else:
        targets = labels.astype(float)[:, None]
    halves = (np.asarray(X[left], dtype=float), np.asarray(X[right], dtype=float))
    responses = (targets[:len(left)], targets[len(left):])
    rows = (left, right)
    zeros, fulls, checks = [], [], []
    for a, b in ([(0, 1)] if ordered_groups else [(0, 1), (1, 0)]):
        zero, full, check = _direction(halves[a], halves[b], responses[a], responses[b])
        zeros.append(zero)
        fulls.append(full)
        checks.append(check)
        info["folds"].append(dict(fit_rows=rows[a], held_rows=rows[b]))
    zero = np.mean(zeros, axis=0)
    full = np.mean(fulls, axis=0)
    scores = np.minimum(zero + (low / high) * (full - zero), full)
    order = np.argsort(-scores, kind="stable")
    if count is None:
        count = _width(order, checks) if scores[order[0]] > 0 else p
    info.update(scores=scores, zero_peer_scores=zero, full_peer_scores=full, budget=int(count))
    return np.sort(order[:count]), info
