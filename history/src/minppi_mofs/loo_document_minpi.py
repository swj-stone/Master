"""All-document leave-one-out loss votes with fixed zero/one-peer contexts."""
import numpy as np

from minppi_mofs.document_vote_minpi import _presence, document_vote_top_from_training
from minppi_mofs.fast_document_vote_minpi import _checked_csr
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _single_loss(n, positives, frequency, positive_frequency):
    """Exact LOO loss of intercept plus a binary term, with unit slope penalty."""
    mean = frequency / n
    prior = positives / n
    gram = frequency * (1. - mean) + 1.
    beta = (positive_frequency - frequency * prior) / gram
    loss = np.zeros_like(frequency)
    for value in (0., 1.):
        total = frequency if value else n - frequency
        positive = positive_frequency if value else positives - positive_frequency
        centered = value - mean
        prediction = prior + centered * beta
        leverage = 1. / n + centered**2 / gram
        squared = (total - positive) * prediction**2 + positive * (1. - prediction)**2
        loss += np.divide(squared, (1. - leverage)**2,
                          out=np.zeros_like(squared), where=total > 0)
    # Binary one-hot squared loss has two equal coordinate losses.
    return 2. * loss / n


def _pair_loss(n, positives, frequency, positive_frequency, peer, both, positive_both):
    """Exact LOO loss of intercept and two binary terms from four cell counts."""
    prior = positives / n
    mean = frequency / n
    peer_mean = mean[peer]
    gram = frequency * (1. - mean) + 1.
    cross = both - frequency * peer_mean
    determinant = gram * gram[peer] - cross**2
    rhs = positive_frequency - frequency * prior
    first_beta = (gram[peer] * rhs - cross * rhs[peer]) / determinant
    peer_beta = (gram * rhs[peer] - cross * rhs) / determinant
    cells = (
        (0., 0., n - frequency - frequency[peer] + both,
         positives - positive_frequency - positive_frequency[peer] + positive_both),
        (1., 0., frequency - both, positive_frequency - positive_both),
        (0., 1., frequency[peer] - both, positive_frequency[peer] - positive_both),
        (1., 1., both, positive_both),
    )
    loss = np.zeros_like(frequency)
    for value, peer_value, total, positive in cells:
        first = value - mean
        second = peer_value - peer_mean
        prediction = prior + first * first_beta + second * peer_beta
        leverage = 1. / n + (
            gram[peer] * first**2 - 2. * cross * first * second + gram * second**2
        ) / determinant
        squared = (total - positive) * prediction**2 + positive * (1. - prediction)**2
        loss += np.divide(squared, (1. - leverage)**2,
                          out=np.zeros_like(squared), where=total > 0)
    return 2. * loss / n


def loo_document_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                   groups=None, ordered_groups=False):
    """Rank binary terms by exact LOO gains with fixed strong-peer contexts.

    Fit an unpenalized intercept and unit-penalty term slopes. The leave-one-out
    formula refits these coefficients exactly, keeping the peer identities
    chosen on all supplied training rows fixed. It does not redo the peer
    search for each omitted row. The two peer-count strata are zero and one.

    For ordinary ungrouped binary labels all rows contribute, and the seed
    has no effect on the result. Multiclass and grouped inputs use the earlier
    cross-fitted document votes, preserving its held-group isolation. A
    requested fixed count overrides the original automatic 99% vote-mass
    budget with its square-root floor and half-width cap.
    """
    X = _checked_csr(X)
    if task != "classification":
        raise ValueError("Document presence votes require classification labels")
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    classes, labels = np.unique(y, return_inverse=True)
    if groups is not None or ordered_groups or len(classes) != 2:
        cols, info = document_vote_top_from_training(
            X, y, count, task=task, random_state=random_state,
            groups=groups, ordered_groups=ordered_groups)
        return cols, dict(info, estimator="group_or_multiclass_crossfit")
    X = _presence(X)
    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least two training documents")
    frequency = np.bincount(X.indices, minlength=p).astype(float)
    positive_frequency = np.bincount(X[labels == 1].indices, minlength=p).astype(float)
    positives = int(labels.sum())
    prior = positives / n
    rhs = positive_frequency - frequency * prior
    marginal = 2. * rhs**2 / (frequency * (1. - frequency / n) + 1.)
    single_loss = _single_loss(n, positives, frequency, positive_frequency)
    intercept_loss = 2. * prior * (1. - prior) / (1. - 1. / n)**2
    zero = intercept_loss - single_loss
    if p == 1:
        return np.array([0]), dict(scores=zero, zero_peer_scores=zero, one_peer_scores=zero,
                                   budget=1, cap=1, folds=[], peer_window=(0, 0), estimator="loo")
    order = np.argsort(-marginal, kind="stable")
    strongest, alternate = int(order[0]), int(order[1])
    peer = np.full(p, strongest)
    peer[strongest] = alternate
    active = X[:, [strongest]].nonzero()[0]
    both = np.bincount(X[active].indices, minlength=p).astype(float)
    positive_both = np.bincount(X[active[labels[active] == 1]].indices, minlength=p).astype(float)
    both[strongest] = both[alternate]
    positive_both[strongest] = positive_both[alternate]
    pair_loss = _pair_loss(n, positives, frequency, positive_frequency, peer, both, positive_both)
    one = single_loss[peer] - pair_loss
    constant = (frequency == 0) | (frequency == n)
    zero[constant] = 0.
    one[constant] = 0.
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, decimals=12) if scale > 0 else scores
    rank = np.lexsort((np.arange(p), -marginal, -relative))
    cap = (p + 1) // 2
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(mass, .99 * mass[-1]) if mass[-1] > 0 else cap
        count = min(cap, max(int(np.ceil(np.sqrt(p))), needed))
    return np.sort(rank[:count]), dict(
        scores=scores, zero_peer_scores=zero, one_peer_scores=one, marginal_scores=marginal,
        cap=cap, budget=int(count), folds=[], peers=(strongest, alternate),
        peer_window=(0, 1), estimator="loo_fixed_peers", rows_used=n,
    )
