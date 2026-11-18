"""Cross-fitted term-presence loss votes on sparse document classification data."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from minppi_mofs.signal_vote_minpi import _row_dot, _split
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _presence(X):
    if not sparse.issparse(X) or X.ndim != 2:
        raise ValueError("Expected a sparse document-by-term matrix")
    X = X.tocsr(copy=False)
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()
    if X.data.size and not np.all(X.data == 1):
        if not np.all(np.isfinite(X.data)) or (X.data < 0).any():
            raise ValueError("Term entries must be finite nonnegative counts")
        X = X.copy()
        X.eliminate_zeros()
        X.data[:] = 1
    return X


@dataclass
class _Counts:
    matrix: object
    term: np.ndarray
    classes: np.ndarray
    joint: np.ndarray
    rhs: np.ndarray
    square: np.ndarray


def _counts(X, y, num_classes):
    n, p = X.shape
    term = np.bincount(X.indices, minlength=p).astype(float)
    freq = np.bincount(y, minlength=num_classes).astype(float)
    # Each CSR row contains each term at most once. One bincount per class
    # avoids constructing n_documents by n_classes dense indicator matrices.
    joint = np.empty((p, num_classes), dtype=float)
    for k in range(num_classes - 1):
        joint[:, k] = np.bincount(X[y == k].indices, minlength=p)
    joint[:, -1] = term - joint[:, :-1].sum(axis=1)
    rhs = joint - term[:, None] * (freq / n)
    return _Counts(X, term, freq, joint, rhs, term * (1 - term/n))


def _intersections(X, anchor, alternate):
    active = X[:, [anchor]].nonzero()[0]
    counts = np.bincount(X[active].indices, minlength=X.shape[1]).astype(float)
    counts[anchor] = counts[alternate]
    return counts


def _votes(fit, held):
    n, p = fit.matrix.shape
    held_n = held.matrix.shape[0]
    mean, prior = fit.term/n, fit.classes/n
    # Unit ridge on binary indicators acts as a one-document shrinkage term.
    root = np.sqrt(fit.square + 1.)
    single = fit.rhs/root[:, None]
    strength = _row_dot(single, single)
    anchor = int(np.argmax(strength))
    strength[anchor] = -np.inf
    alternate = int(np.argmax(strength))
    peer = np.full(p, anchor)
    peer[anchor] = alternate
    cross_fit = _intersections(fit.matrix, anchor, alternate) - mean * fit.term[peer]
    r = cross_fit / root / root[peer]
    cross = (_intersections(held.matrix, anchor, alternate) - mean * held.term[peer] -
             mean[peer] * held.term + held_n * mean * mean[peer]) / root / root[peer]
    square = (held.term * (1 - 2*mean) + held_n * mean**2) / root**2
    rhs = (held.joint - held.term[:, None] * prior -
           mean[:, None] * (held.classes - held_n * prior)) / root[:, None]
    zero = 2 * _row_dot(rhs, single) - square * _row_dot(single, single)
    beta = (single - r[:, None] * single[peer]) / (1 - r**2)[:, None]
    partial_rhs = rhs - r[:, None] * rhs[peer] - (cross - r*square[peer])[:, None] * single[peer]
    residual_norm = square - 2*r*cross + r**2 * square[peer]
    one = 2 * _row_dot(partial_rhs, beta) - residual_norm * _row_dot(beta, beta)
    return np.stack((zero, one))/held_n, (anchor, alternate)


def document_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                    groups=None, ordered_groups=False):
    """Rank terms by worst held presence gain in the zero/one-anchor contexts.

    Accept CSR-compatible sparse nonnegative document counts. Nonzero counts
    signify presence. Each fitting sketch chooses the strongest term and its
    alternate by ridge fitting strength. Binary contingency counts yield the
    exact held-row loss gain of refitting unstandardized unit-penalty ridge
    on the intercept plus the term, or on the intercept plus anchor and term.
    Dense temporaries scale with vocabulary size times number of classes.

    The adaptive budget covers 99% of positive worst-context gain, bounded
    below by ceil(sqrt(p)) and above by ceil(p/2). No positive votes use the
    upper bound. Ties use full-training marginal ridge strength then column
    index. A fixed caller-specified count overrides both bounds.
    """
    if task != "classification":
        raise ValueError("Document presence votes require classification labels")
    X = _presence(X)
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    n, p = X.shape
    if p == 1:
        return np.array([0]), dict(scores=np.zeros(1), zero_peer_scores=np.zeros(1),
                                   one_peer_scores=np.zeros(1), budget=1, cap=1, folds=[], peer_window=(0, 0))
    if n < 2:
        raise ValueError("Need at least two training documents")
    classes, labels = np.unique(y, return_inverse=True)
    left, right = _split(labels, task, groups, ordered_groups, np.random.default_rng(random_state))
    a, b = _counts(X[left], labels[left], len(classes)), _counts(X[right], labels[right], len(classes))
    endpoints, peers = _votes(a, b)
    folds = [dict(fit_rows=left, held_rows=right, peers=peers)]
    if not ordered_groups:
        reverse, peers = _votes(b, a)
        endpoints = (endpoints + reverse) / 2.
        folds.append(dict(fit_rows=right, held_rows=left, peers=peers))
    zero, one = endpoints
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    full = _counts(X, labels, len(classes))
    marginal = _row_dot(full.rhs, full.rhs)/(full.square + 1.)
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    return np.sort(rank[:count]), dict(scores=scores, zero_peer_scores=zero, one_peer_scores=one,
                                       cap=cap, budget=int(count), folds=folds, peer_window=(0, 1))
