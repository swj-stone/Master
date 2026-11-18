"""Balanced leave-one-out posterior log-loss votes for binary term presence."""
import numpy as np

from minppi_mofs.document_vote_minpi import _presence, document_vote_top_from_training
from minppi_mofs.fast_document_vote_minpi import _checked_csr
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _loo_log_loss(joint, class_counts):
    """Jeffreys-smoothed conditional LOO log loss, equally weighted by class."""
    total = joint.sum(axis=-1, keepdims=True)
    present = joint > 0
    probability = np.divide(
        joint - .5, total - 1. + .5 * len(class_counts),
        out=np.ones_like(joint), where=present,
    )
    return -np.sum(joint * np.log(probability) / class_counts, axis=-1) / len(class_counts)


def _term_class_counts(X, labels, class_count):
    return np.stack([
        np.bincount(X[labels == c].indices, minlength=X.shape[1]) for c in range(class_count)
    ], axis=1).astype(float)


def predictive_document_top_from_training(
    X, y, count=None, *, task="classification", random_state=1129,
    groups=None, ordered_groups=False,
):
    """Select terms using balanced LOO posterior gains with zero and one peer.

    Each binary feature cell has a Jeffreys Dirichlet prior of one half per
    observed training class. Exactly leave out each target document from its
    cell's class count and total count before predicting that document. Average
    log losses equally across training classes, using every supplied row.

    Choose a strong peer from the full-training zero-peer LOO gain, keeping
    its identity fixed throughout the pair-cell LOO computation. The strongest
    term receives the runner-up as its peer. The minimum over the zero/one-peer
    gains defines the minPI score.

    Without groups, random_state has no effect. Grouped inputs use the earlier
    document cross-fit rule to preserve held-group isolation and report that
    fallback in diagnostics. A fixed count overrides the default 99% positive
    vote-mass prefix with a square-root lower bound and half-width cap.
    """
    X = _checked_csr(X)
    if task != "classification":
        raise ValueError("Document presence votes require classification labels")
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    if groups is not None or ordered_groups:
        cols, info = document_vote_top_from_training(
            X, y, count, task=task, random_state=random_state,
            groups=groups, ordered_groups=ordered_groups,
        )
        return cols, dict(info, estimator="group_crossfit")
    X = _presence(X)
    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least two training documents")
    classes, labels, class_counts = np.unique(y, return_inverse=True, return_counts=True)
    class_counts = class_counts.astype(float)
    joint = _term_class_counts(X, labels, len(classes))
    loss0 = float(_loo_log_loss(class_counts, class_counts))
    single_loss = _loo_log_loss(joint, class_counts) + _loo_log_loss(class_counts - joint, class_counts)
    zero = loss0 - single_loss
    if p == 1:
        return np.array([0]), dict(scores=zero, zero_peer_scores=zero, one_peer_scores=zero,
                                   budget=1, cap=1, folds=[], peer_window=(0, 0), estimator="posterior_loo")
    order = np.argsort(-zero, kind="stable")
    strongest, alternate = int(order[0]), int(order[1])
    peer = np.full(p, strongest)
    peer[strongest] = alternate
    active = X[:, [strongest]].nonzero()[0]
    both = _term_class_counts(X[active], labels[active], len(classes))
    both[strongest] = both[alternate]
    cells = (
        class_counts - joint - joint[peer] + both,
        joint - both,
        joint[peer] - both,
        both,
    )
    pair_loss = sum(_loo_log_loss(cell, class_counts) for cell in cells)
    one = single_loss[peer] - pair_loss
    constant = np.isin(joint.sum(axis=1), (0, n))
    zero[constant] = 0.
    one[constant] = 0.
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    rank = np.lexsort((np.arange(p), -zero, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    return np.sort(rank[:count]), dict(
        scores=scores, zero_peer_scores=zero, one_peer_scores=one, intercept_loss=loss0,
        cap=cap, budget=int(count), folds=[], peers=(strongest, alternate),
        peer_window=(0, 1), estimator="posterior_loo_fixed_peers", rows_used=n,
    )
