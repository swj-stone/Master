"""Paired minPI with one training sketch and reusable per-half loss moments."""
from dataclasses import dataclass

import numpy as np

from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _draw(rows, cap, rng):
    return rows if len(rows) <= cap else rng.choice(rows, cap, replace=False)


def _moment_halves(y, task, groups, ordered, rng):
    if ordered:
        latest = groups == groups[groups.argmax()]
        previous, current = np.flatnonzero(~latest), np.flatnonzero(latest)
        if not len(previous):
            raise ValueError("Need at least two distinct training groups")
        return _draw(previous, 96, rng), _draw(current, 96, rng)
    if len(y) < 2:
        raise ValueError("Need at least two training rows")
    sample = rng.choice(len(y), min(192, len(y)), replace=False)
    if groups is None:
        if task == "classification":
            sample = sample[np.argsort(y[sample], kind="stable")]
        return sample[::2], sample[1::2]
    group_ids = np.unique(groups[sample])
    if len(group_ids) == 1:
        others = np.flatnonzero(groups != group_ids[0])
        if not len(others):
            raise ValueError("Need at least two distinct training groups")
        second_id = groups[rng.choice(others)]
        second = _draw(np.flatnonzero(groups == second_id), 96, rng)
        return sample[:192 - len(second)], second
    left_groups = rng.permutation(group_ids)[::2]
    left = np.isin(groups[sample], left_groups)
    return sample[left], sample[~left]


@dataclass
class _Moments:
    n: int
    mean: np.ndarray
    target_mean: np.ndarray
    square: np.ndarray
    pair: np.ndarray
    rhs: np.ndarray
    # Extra contractions for hold-side mean imputation when entries are missing.
    missing_terms: tuple | None


def _moments(matrix, target, mate):
    n, p = matrix.shape
    finite = np.isfinite(matrix)
    complete = finite.all()
    if complete:
        anchor = matrix[0]
        centered = matrix - anchor
        shift = centered.mean(axis=0)
        centered -= shift
    else:
        anchor = matrix[np.argmax(finite, axis=0), np.arange(p)]
        anchor = np.where(np.isfinite(anchor), anchor, 0.)
        centered = np.subtract(matrix, anchor, out=np.zeros_like(matrix), where=finite)
        observed = finite.sum(axis=0)
        shift = centered.sum(axis=0) / np.maximum(observed, 1)
        np.subtract(centered, shift, out=centered, where=finite)
    # Anchoring at an observed value makes every constant column exactly zero.
    target_centered = target - target[0]
    target_shift = target_centered.mean(axis=0)
    target_centered -= target_shift
    missing_terms = None
    if not complete:
        mate_observed = finite[:, mate]
        missing_terms = (observed, finite.T @ target_centered,
                         np.einsum("ij,ij->j", centered, mate_observed),
                         (finite & mate_observed).sum(axis=0))
    return _Moments(n, anchor + shift, target[0] + target_shift,
                    np.einsum("ij,ij->j", centered, centered),
                    np.einsum("ij,ij->j", centered, centered[:, mate]),
                    centered.T @ target_centered, missing_terms)


def _held_moments(fit, held, mate):
    # Shift held-half moments to the fitting means. Only observed features
    # shift: mean-imputed coordinates must have exactly zero fitted increment.
    dx = held.mean - fit.mean
    dy = held.target_mean - fit.target_mean
    if held.missing_terms is None:
        return (held.square + held.n * dx**2,
                held.pair + held.n * dx * dx[mate],
                held.rhs + (held.n * dx)[:, None] * dy)
    count, observed_y, centered_with_mate, both = held.missing_terms
    product = centered_with_mate * dx[mate]
    return (held.square + count * dx**2,
            held.pair + product + product[mate] + both * dx * dx[mate],
            held.rhs + dx[:, None] * (observed_y + count[:, None] * dy))


def _moment_gains(fit, held, mate):
    # On the raw feature scale, unit ridge on a standardized column adds its
    # fitting variance to its Gram diagonal. An unobserved/constant column
    # has zero cross-products and zero coefficient.
    diag = np.where(fit.square > 0, fit.square * (1. + 1. / fit.n), 1.)
    root = np.sqrt(diag)
    # Divide out the diagonal before the 2-by-2 solve so differently scaled
    # features never require multiplying two potentially large Gram entries.
    correlation = fit.pair / root / root[mate]
    single = fit.rhs / root[:, None]
    pair = (single - correlation[:, None] * single[mate]) / (1. - correlation**2)[:, None]
    norm, cross, rhs_held = _held_moments(fit, held, mate)
    norm /= diag
    cross = cross / root / root[mate]
    rhs_held /= root[:, None]

    def dot(a, b):
        return np.einsum("ij,ij->i", a, b)

    alone = 2 * dot(rhs_held, single) - norm * dot(single, single)
    part = 2 * dot(rhs_held, pair) - norm * dot(pair, pair)
    joint = part + part[mate] - 2 * cross * dot(pair, pair[mate])
    conditional = joint - alone[mate]
    singleton = mate == np.arange(len(mate))
    conditional[singleton] = alone[singleton]
    return np.asarray([alone, conditional]) / held.n


def moment_top_from_training(X, y, count, *, task="classification", random_state=1129,
                             groups=None, ordered_groups=False):
    """Select sorted feature indices by the minimum of two size-conditioned votes.

    A seeded random matching partitions features into pairs and possibly one
    singleton. Each pair's value is its reduction in held squared loss after
    fitting unit ridge on the active members using fit-side means/scales.
    Other pairs cancel when taking a feature's marginal. At k uniform peers,
    its mate is included with probability k/(p-1). Average the held-row votes
    within each direction, weight directions equally, then take the minimum
    at zero peers and max(1, count-1) peers. Input order resolves exact ties.

    All statistics use at most 192 training rows. Row splits stratify the
    sampled labels for classification. Unordered group splits hold out entire
    sampled groups in each direction. Ordered groups must sort chronologically
    and use only the latest group for assessment, once. Each half's feature
    squares, paired products and feature-target products are computed once.
    With b sampled rows, p columns and c target coordinates this costs
    O(bpc + p log p) time and O(bp + (b+p)c) space after sampling.
    """
    X, y = _utility_inputs(X, y, count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    high = min(p - 1, max(1, count - 1))
    info = {"ballot": "paired_moment_loss", "budget": int(count),
            "peer_window": (0, high), "folds": []}
    if count == p:
        return np.arange(p), info
    # Separate generators keep the matching independent of sample size/strategy.
    split_seed, pair_seed = np.random.SeedSequence(random_state).spawn(2)
    left, right = _moment_halves(y, task, groups, ordered_groups, np.random.default_rng(split_seed))
    order = np.random.default_rng(pair_seed).permutation(p)
    pairs = order[:2 * (p // 2)].reshape(-1, 2)
    mate = np.arange(p)
    mate[pairs[:, 0]], mate[pairs[:, 1]] = pairs[:, 1], pairs[:, 0]
    sample = np.concatenate((left, right))
    if task == "classification":
        codes = np.unique(y[sample], return_inverse=True)[1]
        target = (codes[:, None] == np.arange(codes.max() + 1)).astype(float)
    else:
        target = np.asarray(y[sample, None], dtype=float)
    a = _moments(np.asarray(X[left], dtype=float), target[:len(left)], mate)
    b = _moments(np.asarray(X[right], dtype=float), target[len(left):], mate)
    gains = _moment_gains(a, b, mate)
    info["folds"].append({"fit_rows": left, "held_rows": right, "mate": mate})
    if not ordered_groups:
        gains = (gains + _moment_gains(b, a, mate)) / 2
        info["folds"].append({"fit_rows": right, "held_rows": left, "mate": mate})
    alone, conditional = gains
    scores = np.minimum(alone, alone + (conditional - alone) * (high / (p - 1)))
    info["scores"] = scores
    return np.sort(np.argsort(-scores, kind="stable")[:count]), info
