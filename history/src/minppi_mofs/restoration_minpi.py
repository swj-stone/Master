"""Signed restoration votes under training-held feature permutation."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier

from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.utility_minpi import _utility_inputs


def _sample_rows(rows, cap, labels, rng):
    """Draw without replacement, allocating proportional class quotas when possible."""
    if len(rows) <= cap:
        return rows
    if labels is None:
        return rng.choice(rows, cap, replace=False)
    values, codes, counts = np.unique(labels[rows], return_inverse=True, return_counts=True)
    if cap < len(values):
        return rng.choice(rows, cap, replace=False)
    available = counts - 1
    extra = (cap - len(counts)) * available / available.sum()
    sizes = 1 + np.floor(extra).astype(int)
    largest = np.argsort(-(extra - np.floor(extra)), kind="stable")[:cap - sizes.sum()]
    sizes[largest] += 1
    return np.concatenate([rng.choice(rows[codes == c], size, replace=False)
                           for c, size in enumerate(sizes)])


def _screen_rows(y, task, rng, groups, ordered_groups):
    """Disjoint fitting and assessment rows drawn strictly inside the training set."""
    labels = y if task == "classification" else None
    rows = np.arange(len(y))
    if groups is None:
        if ordered_groups:
            raise ValueError("ordered_groups requires training groups")
        eligible = rows
        if labels is not None:
            # Preserve a fit-fold example even for a class that occurs only once.
            protected = [rng.choice(rows[y == label]) for label in np.unique(y)]
            eligible = np.setdiff1d(rows, protected)
        assess = _sample_rows(eligible, min(256, max(1, len(y) // 4)), labels, rng)
        fit = np.setdiff1d(rows, assess)
    else:
        groups = np.asarray(groups)
        if groups.shape != y.shape or pd.isna(groups).any():
            raise ValueError("Expected complete groups aligned with the training labels")
        unique = np.unique(groups)
        if len(unique) < 2:
            raise ValueError("Need at least two distinct training groups")
        held = unique[-1:] if ordered_groups else rng.permutation(unique)[:max(1, len(unique) // 4)]
        mask = np.isin(groups, held)
        fit, assess = rows[~mask], _sample_rows(rows[mask], 256, labels, rng)
    fit = _sample_rows(fit, 1024, labels, rng)
    if len(fit) < 2 or len(assess) < 2:
        raise ValueError("Need at least two fitting rows and two held rows for permutation")
    if labels is not None and len(np.unique(labels[fit])) < 2:
        raise ValueError("Need at least two fitting classes")
    return fit, assess


def _class_ids(scores):
    return (scores[..., 0] > 0).astype(int) if scores.shape[-1] == 1 else scores.argmax(axis=-1)


def _layer_scores(delta, prediction, coefficients, feature_orders, k, target, weights):
    """Average signed gain for one donor shuffle, using k other corrupted inputs per j."""
    n, p = delta.shape
    inside = np.zeros((n, p), dtype=bool)
    rows = np.arange(n)
    if k:
        inside[rows[:, None], feature_orders[:, :k]] = True
        background = prediction + (delta * inside) @ coefficients
        swap = feature_orders[:, k]
        replacement = delta[rows, swap, None] * coefficients[swap]
    else:
        background, replacement = prediction, np.zeros_like(prediction)
    # For j outside the prefix, the restored background is the k-prefix.
    # For j inside it, move j out and use the next prefix item as the kth peer.
    corrupted_inside = background + replacement
    if weights is not None:
        correct_outside = (_class_ids(background) == target)[:, None]
        correct_inside = (_class_ids(corrupted_inside) == target)[:, None]
    scores = np.empty(p)
    # Bound peak memory for the samples × candidates × classes tensor.
    block = max(1, min(p, 1_000_000 // (n * prediction.shape[1])))
    for start in range(0, p, block):
        cols = slice(start, start + block)
        effect = delta[:, cols, None] * coefficients[cols]
        member = inside[:, cols]
        if weights is None:
            effect = effect[:, :, 0]
            restored_residual = ((background[:, 0] - target)[:, None] +
                                 member * (replacement[:, 0, None] - effect))
            scores[cols] = (effect * (2 * restored_residual + effect)).mean(axis=0)
        else:
            before = _class_ids(background[:, None, :] + effect) == target[:, None]
            votes = correct_outside.astype(int) - before
            if k:
                after = _class_ids(corrupted_inside[:, None, :] - effect) == target[:, None]
                votes = np.where(member, after.astype(int) - correct_inside, votes)
            scores[cols] = (weights[:, None] * votes).sum(axis=0)
    return scores


def restoration_top_from_training(X, y, count, *, task="classification", random_state=42,
                                  tiers=None, groups=None, ordered_groups=False):
    """Rank by the smallest signed restoration gain across declared corruption sizes.

    For each assessed row and donor permutation, uniformly corrupt exactly k
    *other* features, then compare the observed j with donor j. The number
    of other clean peers is p-1-k. Default k=0 and p-count compare the fully
    observed background and a mask with the final feature budget. The zero
    layer equals ordinary permutation importance on the same ridge and rows.
    Three independent donor permutations pair the scenarios. Classification
    votes count corrected minus newly wrong predictions with balanced class
    weights; regression votes measure the decrease in squared error.
    """
    X, y = _utility_inputs(X, y, count, task)
    p = X.shape[1]
    levels = sorted(set((0, p - count))) if tiers is None else sorted(_selection_tiers(tiers, p))
    seed_rng = np.random.default_rng(random_state)
    fit, assess = _screen_rows(y, task, seed_rng, groups, ordered_groups)
    train, check = _standardize(np.asarray(X[fit], dtype=float), np.asarray(X[assess], dtype=float))
    classification = task == "classification"
    cls = RidgeClassifier if classification else Ridge
    model = cls(alpha=1., solver="lsqr", tol=.001, max_iter=250).fit(train, y[fit])
    coefficients = model.coef_.reshape(-1, p).T
    target = y[assess]
    weights = None
    if classification:
        prediction = model.decision_function(check)
        labels, codes, counts = np.unique(target, return_inverse=True, return_counts=True)
        weights = 1. / (len(labels) * counts[codes])
        class_lookup = {key: i for i, key in enumerate(model.classes_)}
        target = np.array([class_lookup.get(key, -1) for key in target])
    else:
        prediction = model.predict(check)
    prediction = prediction.reshape(len(check), -1)
    rng = np.random.default_rng(random_state)
    donors = [rng.permutation(len(check)) for _ in range(3)]
    layer_scores = np.zeros((len(levels), p))
    for donor in donors:
        delta = check[donor] - check
        order = (rng.permuted(np.broadcast_to(np.arange(p), check.shape), axis=1)
                 if max(levels) else None)
        for i, k in enumerate(levels):
            layer_scores[i] += _layer_scores(delta, prediction, coefficients, order, k, target, weights)
    layer_scores /= len(donors)
    scores = layer_scores.min(axis=0)
    selected = np.argsort(-scores, kind="stable")[:count]
    return np.sort(selected), dict(fit_rows=fit, assessment_rows=assess, scores=scores,
                                   layer_scores=layer_scores, tiers=levels)
