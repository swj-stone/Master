"""All-row cross-fitted minPI with an expected vote over several strong peers."""
from numbers import Integral

import numpy as np

from minppi_mofs.document_vote_minpi import _counts, _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr
from minppi_mofs.signal_vote_minpi import _row_dot
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _full_split(labels, groups, ordered, rng):
    if groups is None:
        rows = rng.permutation(len(labels))
        rows = rows[np.argsort(labels[rows], kind="stable")]
        return rows[::2], rows[1::2]
    values = np.unique(groups)
    if len(values) < 2:
        raise ValueError("Need at least two distinct training groups")
    held = groups == values[-1] if ordered else np.isin(groups, rng.permutation(values)[::2])
    return np.flatnonzero(~held), np.flatnonzero(held)


def _cooccurrences(X, anchor):
    active = X[:, [anchor]].nonzero()[0]
    return np.bincount(X[active].indices, minlength=X.shape[1]).astype(float)


def _peer_votes(fit, held, max_peers):
    n, p = fit.matrix.shape
    held_n = held.matrix.shape[0]
    mean, prior = fit.term / n, fit.classes / n
    root = np.sqrt(fit.square + 1.)
    single = fit.rhs / root[:, None]
    strength = _row_dot(single, single)
    anchors = np.argsort(-strength, kind="stable")[:min(p, max_peers + 1)]
    # The extra candidate supplies the sole peer when max_peers=1 and the
    # tested feature is the strongest. Every column gets the same peer count.
    peer_count = min(max_peers, p - 1)
    square = (held.term * (1 - 2 * mean) + held_n * mean**2) / root**2
    rhs = (held.joint - held.term[:, None] * prior -
           mean[:, None] * (held.classes - held_n * prior)) / root[:, None]
    zero = 2 * _row_dot(rhs, single) - square * _row_dot(single, single)
    one = np.zeros(p)
    columns = np.arange(p)
    used = np.zeros(p, dtype=int)
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


def consensus_document_top_from_training(X, y, count=None, *, task="classification",
                                         random_state=1129, groups=None, ordered_groups=False,
                                         max_peers=4):
    """Rank term-presence columns by cross-fitted zero/one-peer minimum gain.

    Every training row belongs to one of two disjoint folds. Each fitting
    fold selects its strongest peer_count + 1 columns by marginal unit-ridge
    strength. A tested column is evaluated with the first peer_count other
    columns from that pool. The one-peer stratum averages their signed
    held-row loss gains, then the minimum with zero-peer gain ranks columns.
    The unit ridge penalty applies to unstandardized binary term indicators.
    Class targets are one-hot encoded, including in binary classification.

    Groups remain disjoint. Ordered groups hold the final group out and use
    only the forward direction. Other fits average the two directions by
    held-row count. Automatic width keeps 99% of positive vote mass with a
    ceil(sqrt(p)) floor and ceil(p/2) ceiling. A requested count overrides it.
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
    endpoints, anchors = _peer_votes(a, b, max_peers)
    folds = [dict(fit_rows=left, held_rows=right, anchors=anchors)]
    if not ordered_groups:
        reverse, anchors = _peer_votes(b, a, max_peers)
        endpoints = (len(right) * endpoints + len(left) * reverse) / n
        folds.append(dict(fit_rows=right, held_rows=left, anchors=anchors))
    zero, one = endpoints
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    term, freq = a.term + b.term, a.classes + b.classes
    rhs = a.joint + b.joint - term[:, None] * (freq / n)
    marginal = _row_dot(rhs, rhs) / (term * (1 - term / n) + 1.)
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
