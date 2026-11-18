"""Proxy-group pivotal votes with finite-observation anchoring."""
import numpy as np

from minppi_mofs.moment_minpi import _moment_halves
from minppi_mofs.pilot_minpi import _coverage_scores, _proxy_groups, _target
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _unit_columns(values):
    """Unit length for varying columns, zero for constant or entirely absent columns."""
    centered = np.array(values, dtype=float, copy=True)
    valid = np.isfinite(centered)
    if valid.all():
        centered -= centered[0]
        centered -= centered.mean(axis=0)
    else:
        first = centered[np.argmax(valid, axis=0), np.arange(centered.shape[1])]
        first = np.where(np.isfinite(first), first, 0.)
        np.subtract(centered, first, where=valid, out=centered)
        centered[~valid] = 0.
        mean = centered.sum(axis=0) / np.maximum(1, valid.sum(axis=0))
        np.subtract(centered, mean, where=valid, out=centered)
    scale = np.abs(centered).max(axis=0)
    centered /= np.where(scale > 0, scale, 1.)
    norm = np.sqrt(np.einsum("ij,ij->j", centered, centered))
    centered /= np.where(norm > 0, norm, 1.)
    return centered


def _target_direction(y, classes):
    if classes is None and np.all(y == y[0]):
        return np.zeros((len(y), 1))
    return _target(y, classes)


def proxy_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                            groups=None, ordered_groups=False):
    """Retain a representative per confirmed group or use a fixed column budget.

    Proxy proposals, both-sample correlation tests, random sampling and the
    weakest-layer group utility follow the original pilot. Anchor every
    column at an observed value before centering, including columns with
    missing observations. Constant sampled targets have zero direction.
    """
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    ids = np.arange(p)
    if p == 1 or count == p:
        return ids, dict(budget=p, folds=[], parent=ids, scores=np.zeros(p), groups_found=p)
    rng = np.random.default_rng(random_state)
    left, right = _moment_halves(y, task, groups, ordered_groups, rng)
    cap = min(96, max(12, 32768 // p))
    if len(left) > cap:
        left = rng.choice(left, cap, replace=False)
    if len(right) > cap:
        right = rng.choice(right, cap, replace=False)
    a, b = _unit_columns(X[left]), _unit_columns(X[right])
    classes = np.unique(np.concatenate((y[left], y[right]))) if task == "classification" else None
    fit, held = a.T @ _target_direction(y[left], classes), b.T @ _target_direction(y[right], classes)
    evidence = np.einsum("ij,ij->i", fit, held)
    parent = _proxy_groups(a, b, evidence, rng)
    leaders = parent == ids
    group_count = int(leaders.sum())
    budget = min(group_count, p - 1) if count is None else count
    scores = _coverage_scores(evidence, parent)
    order = np.lexsort((ids, -evidence, -scores, ~leaders))
    return np.sort(order[:budget]), dict(
        budget=int(budget), groups_found=group_count, peer_window=(0, p - 1), parent=parent,
        folds=[{"fit_rows": left, "held_rows": right}], scores=scores)
