"""Adaptive minPI from cross-fitted votes with an influential peer."""
from dataclasses import dataclass

import numpy as np

from minppi_mofs.moment_minpi import _draw
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


_HALF_CAP = 256
_RETAINED_MASS = .99


def _split(y, task, groups, ordered, rng):
    if groups is None:
        sample = rng.choice(len(y), min(len(y), 2 * _HALF_CAP), replace=False)
        if task == "classification":
            sample = sample[np.argsort(y[sample], kind="stable")]
        return sample[::2], sample[1::2]
    values = np.unique(groups)
    if len(values) < 2:
        raise ValueError("Need at least two distinct training groups")
    if ordered:
        held = groups == values[-1]
    else:
        held = np.isin(groups, rng.permutation(values)[::2])
    return (_draw(np.flatnonzero(~held), _HALF_CAP, rng),
            _draw(np.flatnonzero(held), _HALF_CAP, rng))


def _row_dot(A, B):
    return np.einsum("ij,ij->i", A, B)


def _square(X):
    return np.einsum("ij,ij->j", X, X)


@dataclass
class _Half:
    x: np.ndarray
    mean: np.ndarray
    observed: np.ndarray | None
    target: np.ndarray
    target_mean: np.ndarray
    square: np.ndarray
    rhs: np.ndarray


def _half(X, target):
    observed = np.isfinite(X)
    if observed.all():
        observed = None
        anchor = X[0]
        x = X - anchor
        shift = x.mean(axis=0)
        x -= shift
    else:
        # Anchor first to make constant nonbinary floating-point values exact zeros.
        anchor = X[observed.argmax(axis=0), np.arange(X.shape[1])]
        anchor = np.where(np.isfinite(anchor), anchor, 0.)
        x = np.subtract(X, anchor, out=np.zeros_like(X), where=observed)
        shift = x.sum(axis=0) / np.maximum(1, observed.sum(axis=0))
        np.subtract(x, shift, out=x, where=observed)
    cy = target - target[0]
    target_shift = cy.mean(axis=0)
    cy -= target_shift
    return _Half(x, anchor + shift, observed, cy, target[0] + target_shift, _square(x), x.T @ cy)


def _peer_products(x, strongest, runner_up):
    # Every feature has the strongest peer except the strongest feature itself.
    cross = np.einsum("ij,i->j", x, x[:, strongest])
    cross[strongest] = cross[runner_up]
    return cross


def _votes(fit, held):
    root = np.sqrt(np.where(fit.square > 0., fit.square * (1. + 1. / len(fit.x)), 1.))
    single = fit.rhs / root[:, None]
    strength = _row_dot(single, single)
    strongest = int(strength.argmax())
    strength[strongest] = -np.inf
    runner_up = int(strength.argmax())
    peer = np.full(len(root), strongest, dtype=np.intp)
    peer[strongest] = runner_up
    r = _peer_products(fit.x, strongest, runner_up) / root / root[peer]

    dx = held.mean - fit.mean
    dy = held.target_mean - fit.target_mean
    n = len(held.x)
    if held.observed is None:
        square = held.square + n * dx**2
        cross = _peer_products(held.x, strongest, runner_up) + n * dx * dx[peer]
        rhs = held.rhs + (n * dx)[:, None] * dy
    else:
        check = np.add(held.x, dx, out=np.zeros_like(held.x), where=held.observed)
        square = _square(check)
        cross = _peer_products(check, strongest, runner_up)
        rhs = check.T @ (held.target + dy)
    square = square / root**2
    cross = cross / root / root[peer]
    rhs = rhs / root[:, None]
    zero = 2 * _row_dot(rhs, single) - square * _row_dot(single, single)

    # Schur complement of the normalized two-column ridge Gram. Adding j
    # changes the peer's coefficient by -r[j] times the new j coefficient.
    beta = (single - r[:, None] * single[peer]) / (1. - r**2)[:, None]
    rhs_residual = rhs - r[:, None] * rhs[peer] - (cross - r * square[peer])[:, None] * single[peer]
    residual_norm = square - 2 * r * cross + r**2 * square[peer]
    one = 2 * _row_dot(rhs_residual, beta) - residual_norm * _row_dot(beta, beta)
    return np.stack((zero, one)) / n, (strongest, runner_up)


def signal_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                  groups=None, ordered_groups=False):
    """Rank by the worse cross-fitted gain in the zero- and one-peer strata.

    A fitting half picks the strongest non-self peer by one-variable ridge
    improvement in penalized fitting loss. Votes on held rows measure squared
    loss decrease after adding a column and refitting the active coefficients.
    Each fit uses its own means and unit ridge on standardized features.
    Classification uses one-hot targets. Folds have equal weight; time-ordered
    groups allow only the past-to-latest direction.

    With count=None, retain the shortest ranked prefix covering 99% of the
    positive worst-context vote mass. With no positive votes, retain all.
    This surrogate budget does not bound downstream test-score loss.
    """
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    folds = []
    info = {"ballot": "strong_peer_held_loss", "peer_window": (0, min(1, p - 1)),
            "row_cap_per_half": _HALF_CAP, "retained_positive_mass": _RETAINED_MASS,
            "strategy": "forward" if ordered_groups else "group" if groups is not None else "row",
            "folds": folds}
    if p == 1:
        endpoints = np.zeros((2, 1))
    else:
        if len(y) < 2:
            raise ValueError("Need at least two training rows")
        left, right = _split(y, task, groups, ordered_groups, np.random.default_rng(random_state))
        labels = y[np.concatenate((left, right))]
        if task == "classification":
            codes = np.unique(labels, return_inverse=True)[1]
            target = (codes[:, None] == np.arange(codes.max() + 1)).astype(float)
        else:
            target = labels.astype(float)[:, None]
        a = _half(np.asarray(X[left], dtype=float), target[:len(left)])
        b = _half(np.asarray(X[right], dtype=float), target[len(left):])
        endpoints, peers = _votes(a, b)
        folds.append({"fit_rows": left, "held_rows": right, "peers": peers})
        if not ordered_groups:
            reverse, peers = _votes(b, a)
            endpoints = (endpoints + reverse) / 2.
            folds.append({"fit_rows": right, "held_rows": left, "peers": peers})
    scores = endpoints.min(axis=0)
    order = np.argsort(-scores, kind="stable")
    if count is None:
        mass = np.maximum(scores[order], 0.).cumsum()
        count = int(np.searchsorted(mass, _RETAINED_MASS * mass[-1]) + 1) if mass[-1] > 0 else p
    info.update(budget=int(count), zero_peer_scores=endpoints[0], one_peer_scores=endpoints[1], scores=scores)
    return np.sort(order[:count]), info
