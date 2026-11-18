"""All-row term-presence votes with penalties matching standardized ridge."""
from numbers import Integral

import numpy as np

from minppi_mofs.consensus_document_minpi import _cooccurrences, _full_split
from minppi_mofs.document_vote_minpi import _counts, _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr
from minppi_mofs.signal_vote_minpi import _row_dot
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _standard_votes(fit, held, max_peers):
    n, p = fit.matrix.shape
    held_n = held.matrix.shape[0]
    mean, prior = fit.term / n, fit.classes / n
    # StandardScaler uses population variance. Ridge alpha=1 in standardized
    # coordinates is a diagonal penalty variance_j in binary coordinates.
    # A training-constant coordinate has StandardScaler scale=1.
    penalty = np.where(fit.square > 0, fit.square / n, 1.)
    root = np.sqrt(fit.square + penalty)
    single = fit.rhs / root[:, None]
    strength = _row_dot(single, single)
    anchors = np.argsort(-strength, kind="stable")[:min(p, max_peers + 1)]
    peer_count = min(max_peers, p - 1)
    square = (held.term * (1 - 2 * mean) + held_n * mean**2) / root**2
    rhs = (held.joint - held.term[:, None] * prior -
           mean[:, None] * (held.classes - held_n * prior)) / root[:, None]
    zero = 2 * _row_dot(rhs, single) - square * _row_dot(single, single)
    one = np.zeros(p)
    columns, used = np.arange(p), np.zeros(p, dtype=int)
    for anchor in anchors:
        eligible = (columns != anchor) & (used < peer_count)
        r = (_cooccurrences(fit.matrix, anchor) - mean * fit.term[anchor]) / root / root[anchor]
        cross = (_cooccurrences(held.matrix, anchor) - mean * held.term[anchor] -
                 mean[anchor] * held.term + held_n * mean * mean[anchor]) / root / root[anchor]
        beta = (single - r[:, None] * single[anchor]) / (1 - r**2)[:, None]
        partial_rhs = rhs - r[:, None] * rhs[anchor] - (
            cross - r * square[anchor])[:, None] * single[anchor]
        residual_norm = square - 2 * r * cross + r**2 * square[anchor]
        gain = 2 * _row_dot(partial_rhs, beta) - residual_norm * _row_dot(beta, beta)
        one[eligible] += gain[eligible]
        used[eligible] += 1
    return np.stack((zero, one / peer_count)) / held_n, anchors


def standard_document_top_from_training(X, y, count=None, *, task="classification",
                                        random_state=1129, groups=None, ordered_groups=False,
                                        max_peers=4):
    """Use all training rows and the exact standardized zero/one-peer ridge game.

    Two disjoint training folds supply predictions evaluated on opposite
    folds. Each tested column is averaged over up to max_peers strongest
    distinct fitting-fold peers. Select by the minimum of this expectation
    and the zero-peer gain. The two directions are weighted by held count.

    The per-feature raw penalty equals fitting population variance, matching
    StandardScaler followed by unit-penalty Ridge with an intercept. For
    automatic width retain 99% of positive mass, with a sqrt(p) floor and
    half-width ceiling. Ordered groups use only the forward prediction.
    """
    if task != "classification":
        raise ValueError("Document presence votes require classification labels")
    if isinstance(max_peers, (bool, np.bool_)) or not isinstance(max_peers, Integral) or max_peers < 1:
        raise ValueError("max_peers must be a positive integer")
    X = _presence(_checked_csr(X))
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    n, p = X.shape
    if p == 1:
        return np.array([0]), dict(scores=np.zeros(1), zero_peer_scores=np.zeros(1),
                                  one_peer_scores=np.zeros(1), budget=1, cap=1, folds=[],
                                  peer_count=0, peer_window=(0, 0))
    if n < 2:
        raise ValueError("Need at least two training documents")
    classes, labels = np.unique(y, return_inverse=True)
    left, right = _full_split(labels, groups, ordered_groups, np.random.default_rng(random_state))
    a = _counts(X[left], labels[left], len(classes))
    b = _counts(X[right], labels[right], len(classes))
    endpoints, anchors = _standard_votes(a, b, max_peers)
    folds = [dict(fit_rows=left, held_rows=right, anchors=anchors)]
    if not ordered_groups:
        reverse, anchors = _standard_votes(b, a, max_peers)
        endpoints = (len(right) * endpoints + len(left) * reverse) / n
        folds.append(dict(fit_rows=right, held_rows=left, anchors=anchors))
    zero, one = endpoints
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    term, freq = a.term + b.term, a.classes + b.classes
    rhs = a.joint + b.joint - term[:, None] * (freq / n)
    square = term * (1 - term / n)
    marginal = _row_dot(rhs, rhs) / (square + np.where(square > 0, square / n, 1.))
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    return np.sort(rank[:count]), dict(scores=scores, zero_peer_scores=zero,
                                      one_peer_scores=one, cap=cap, budget=int(count),
                                      folds=folds, peer_count=min(max_peers, p - 1),
                                      peer_window=(0, 1))
