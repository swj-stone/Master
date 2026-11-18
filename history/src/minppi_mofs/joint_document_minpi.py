"""Sparse joint-predictor minPI under fixed-coefficient feature coalitions."""
from numbers import Integral

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from minppi_mofs.consensus_document_minpi import _full_split
from minppi_mofs.document_vote_minpi import _counts, _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr
from minppi_mofs.signal_vote_minpi import _row_dot
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _joint_votes(Xfit, target, Xheld, query, n_outer, max_iter, tol):
    scaler = StandardScaler(with_mean=False)
    train = scaler.fit_transform(Xfit)
    alpha = len(target) / n_outer
    model = Ridge(alpha=alpha, solver="lsqr", max_iter=max_iter, tol=tol)
    model.fit(train, target)
    coefficients = np.asarray(model.coef_).T / scaler.scale_[:, None]
    means = scaler.mean_
    target_mean = target.mean(axis=0)
    centered = query - target_mean
    prediction = Xheld @ coefficients - means @ coefficients
    residual = centered - prediction
    term = np.bincount(Xheld.indices, minlength=Xheld.shape[1]).astype(float)
    square = term * (1 - 2 * means) + len(query) * means**2
    norm = square * _row_dot(coefficients, coefficients)
    rhs = Xheld.T @ centered - means[:, None] * centered.sum(axis=0)
    rest = Xheld.T @ residual - means[:, None] * residual.sum(axis=0)
    zero = 2 * _row_dot(rhs, coefficients) - norm
    full = 2 * _row_dot(rest, coefficients) + norm
    details = dict(coefficients=coefficients, means=means, target_mean=target_mean,
                   iterations=np.asarray(model.n_iter_), alpha=alpha)
    return np.stack((zero, full)) / len(query), details


def joint_document_top_from_training(X, y, count=None, *, task="classification",
                                     random_state=1129, groups=None, ordered_groups=False,
                                     max_iter=8, tol=.001):
    """Rank held-row marginal gains of a sparse fixed joint ridge predictor.

    Fit standardized unit-ridge predictions using all rows in two disjoint
    training folds, with alpha=n_fit/n_outer to preserve penalty per row.
    LSQR stops at tol or max_iter iterations. Freeze each fold's coefficients
    and define a coalition by summing only its centered feature contributions.
    The empty-peer and full-peer gains determine the minimum expected gain
    over every uniformly drawn peer-set size. No per-column fit is needed.

    Automatic width keeps 99% of positive minimum gain, with ceil(sqrt(p))
    and ceil(p/2) bounds. Groups stay disjoint, and ordered groups allow only
    the earlier-to-final-group direction. Sparse counts are scored as binary
    presence. This selector's internal game does not refit a coalition.
    """
    if task != "classification":
        raise ValueError("Document presence votes require classification labels")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    X = _presence(_checked_csr(X))
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    n, p = X.shape
    if p == 1:
        return np.array([0]), dict(scores=np.zeros(1), zero_peer_scores=np.zeros(1),
                                  full_peer_scores=np.zeros(1), budget=1, cap=1,
                                  folds=[], peer_window=(0, 0))
    if n < 2:
        raise ValueError("Need at least two training documents")
    classes, labels = np.unique(y, return_inverse=True)
    targets = np.eye(len(classes))[labels]
    left, right = _full_split(labels, groups, ordered_groups, np.random.default_rng(random_state))
    endpoints, detail = _joint_votes(X[left], targets[left], X[right], targets[right], n, max_iter, tol)
    folds = [dict(fit_rows=left, held_rows=right, **detail)]
    if not ordered_groups:
        reverse, detail = _joint_votes(X[right], targets[right], X[left], targets[left], n, max_iter, tol)
        endpoints = (len(right) * endpoints + len(left) * reverse) / n
        folds.append(dict(fit_rows=right, held_rows=left, **detail))
    zero, full = endpoints
    scores = np.minimum(zero, full)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    counts = _counts(X, labels, len(classes))
    marginal = _row_dot(counts.rhs, counts.rhs) / (counts.square + 1.)
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    return np.sort(rank[:count]), dict(scores=scores, zero_peer_scores=zero,
                                      full_peer_scores=full, cap=cap, budget=int(count),
                                      folds=folds, peer_window=(0, p - 1))
