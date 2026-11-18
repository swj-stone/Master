"""Capped adaptive minPI with cross-fitted low- and multi-peer ridge votes."""
import numpy as np

from minppi_mofs.moment_minpi import _draw, _moment_halves
from minppi_mofs.signal_vote_minpi import _half, _row_dot, _square
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _gain(x, z, fit_rhs, query, held_square, candidates, peers):
    """Loss gain from adding each candidate to the same ridge model and refitting."""
    a, b = x[:, peers], z[:, peers]
    gram = a.T @ a
    # Columns have already been divided by sqrt(fit sum of squares + penalty).
    np.fill_diagonal(gram, 1.)
    inverse = np.linalg.inv(gram)
    base = inverse @ fit_rhs[peers]
    cross = a.T @ x
    adjustment = inverse @ cross[:, candidates]
    den = 1. - np.einsum("ij,ij->j", cross[:, candidates], adjustment)
    beta = (fit_rhs[candidates] - cross[:, candidates].T @ base) / den[:, None]

    residual = query - b @ base
    candidate_z = z[:, candidates]
    held_cross = b.T @ candidate_z
    norm = (held_square[candidates] - 2 * np.einsum("ij,ij->j", adjustment, held_cross) +
            np.einsum("ij,ij->j", adjustment, (b.T @ b) @ adjustment))
    rhs = candidate_z.T @ residual - adjustment.T @ (b.T @ residual)
    return (2 * _row_dot(rhs, beta) - norm * _row_dot(beta, beta)) / len(z)


def _votes(fit, held, partition, k):
    root = np.sqrt(np.where(fit.square > 0, fit.square * (1. + 1. / len(fit.x)), 1.))
    x = fit.x / root
    dx = held.mean - fit.mean
    if held.observed is None:
        z = held.x + dx
    else:
        z = np.add(held.x, dx, out=np.zeros_like(held.x), where=held.observed)
    z /= root
    fit_rhs = fit.rhs / root[:, None]
    query = held.target + (held.target_mean - fit.target_mean)
    norm = _square(z)
    strength = _row_dot(fit_rhs, fit_rhs)
    zero = (2 * _row_dot(z.T @ query, fit_rhs) - norm * strength) / len(z)
    low, high, chosen = zero.copy(), np.empty_like(zero), []
    for side in (0, 1):
        candidates = partition[side]
        pool = partition[1-side]
        peers = pool[np.argsort(-strength[pool], kind="stable")[:k]]
        if k > 1:
            low[candidates] = _gain(x, z, fit_rhs, query, norm, candidates, peers[:1])
        high[candidates] = _gain(x, z, fit_rhs, query, norm, candidates, peers)
        chosen.append(peers)
    return np.stack((zero, low, high)), chosen


def capped_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                  groups=None, ordered_groups=False):
    """Rank by the weakest held gain with one and up to four strong peers.

    A seeded bipartition of columns ensures each cohort is scored against
    peer predictors from its complement. Select peers using fit-side target
    associations only. With fewer than two available peers, use zero and one.
    Ridge fits have an unpenalized intercept and unit penalty on standardized
    columns. Classification uses one-hot targets. Scores are evaluated on
    held training rows and averaged over valid fitting directions.

    By default keep the shortest prefix that covers 99% of positive worst-
    context gain, subject to a hard ceil(p/2) maximum. If no positive gain
    exists, use that maximum. Explicit fixed counts override this cap.
    """
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    cap = (p + 1) // 2
    info = dict(cap=cap, folds=[], peer_window=(0, 0))
    if p == 1:
        return np.array([0]), dict(info, budget=1, zero_peer_scores=np.zeros(1), scores=np.zeros(1))
    rng = np.random.default_rng(random_state)
    left, right = _moment_halves(y, task, groups, ordered_groups, rng)
    left, right = _draw(left, 96, rng), _draw(right, 96, rng)
    order = rng.permutation(p)
    partition = (np.sort(order[::2]), np.sort(order[1::2]))
    k = min(4, len(partition[1]))
    info.update(peer_window=(1 if k > 1 else 0, k), column_halves=partition)
    labels = y[np.r_[left, right]]
    target = ((labels[:, None] == np.unique(labels)).astype(float) if task == "classification"
              else labels[:, None].astype(float))
    a = _half(np.asarray(X[left], dtype=float), target[:len(left)])
    b = _half(np.asarray(X[right], dtype=float), target[len(left):])
    votes, peers = _votes(a, b, partition, k)
    info["folds"].append(dict(fit_rows=left, held_rows=right, peers=peers))
    if not ordered_groups:
        reverse, peers = _votes(b, a, partition, k)
        votes = (votes + reverse) / 2.
        info["folds"].append(dict(fit_rows=right, held_rows=left, peers=peers))
    zero, low, high = votes
    scores = np.minimum(low, high)
    scale = np.abs(scores).max()
    relative = np.round(scores / scale, decimals=12) if scale else scores
    rank = np.argsort(-relative, kind="stable")
    if count is None:
        mass = np.maximum(relative[rank], 0.).cumsum()
        count = min(cap, 1 + np.searchsorted(mass, .99 * mass[-1])) if mass[-1] else cap
    info.update(budget=int(count), zero_peer_scores=zero, low_peer_scores=low,
                high_peer_scores=high, scores=scores)
    return np.sort(rank[:count]), info
