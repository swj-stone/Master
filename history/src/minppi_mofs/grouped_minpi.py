"""Budgeted pivotal voting on supervised, correlation-filtered representatives."""
import numpy as np
import pandas as pd

from minppi_mofs.exact_minpi import _selection_tiers, _witness_margins_from_folds
from minppi_mofs.piv2 import _folds
from minppi_mofs.word_minpi import exact_top_from_votes


def _relevance_and_vectors(X, y, task):
    """Mean-impute on the training sketch, then center and unit-normalize columns."""
    data = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(data)
    data = np.where(finite, data, 0.)
    means = data.sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
    data = np.where(finite, data - means, 0.)
    lengths = np.linalg.norm(data, axis=0)
    vectors = data / np.where(lengths > 0, lengths, 1.)
    if task == "regression":
        target = y.astype(float)
        target -= target.mean()
        norm = np.linalg.norm(target)
        relevance = (target @ vectors / (norm or 1.)) ** 2
    else:
        relevance = np.zeros(X.shape[1])
        for group in np.unique(y):
            subset = vectors[y == group]
            relevance += subset.sum(axis=0) ** 2 / len(subset)
    # Silent columns cannot distinguish witnesses. Reserve them for budget fill.
    relevance[lengths == 0] = -np.inf
    return relevance, vectors


def _representatives(X, y, task, seed, count, tiers):
    X, y = np.asarray(X), np.asarray(y)
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),) or pd.isna(y).any():
        raise ValueError("Expected a nonempty matrix with aligned, complete labels")
    if task == "regression" and not np.isfinite(y.astype(float)).all():
        raise ValueError("Regression targets must be finite")
    p = X.shape[1]
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= p:
        raise ValueError("count must be an integer in [1,p]")
    tiers = _selection_tiers(tiers, p)
    if count == p:
        return np.arange(p), tiers, {"representatives": p, "screened": p, "budget_fill": 0}, None
    folds = _folds(y, task, np.random.default_rng(seed), row_limit=256)
    if folds is None or min(map(len, folds)) < 2:
        raise ValueError("Insufficient training observations for hit/miss witnesses")
    rows = np.concatenate(folds)
    score, vectors = _relevance_and_vectors(X[rows], y[rows], task)
    order = np.argsort(-score, kind="stable")
    # Limit candidate-pair work and leave alternatives for the pivotal cutoff.
    shortlist = order[:max(128, 4 * count)]
    target = min(len(shortlist), max(2 * count, max(tiers) + 1))
    data = np.ascontiguousarray(vectors[:, shortlist].T)
    alive = np.isfinite(score[shortlist])
    chosen = []
    # Compute in small blocks: no p-by-p correlation matrix is materialized.
    for start in range(0, len(shortlist), 32):
        if len(chosen) >= target:
            break
        block = np.flatnonzero(alive[start:start + 32]) + start
        if not len(block):
            continue
        correlated = np.abs(data[block] @ data[start:].T) >= .90
        for pos, neighbours in zip(block, correlated):
            if alive[pos]:
                chosen.append(shortlist[pos])
                alive[start:] &= ~neighbours
                if len(chosen) == target:
                    break
    fill = max(0, count - len(chosen))
    if fill:
        unused = ~np.isin(order, chosen)
        chosen.extend(order[unused][:fill])
    info = {"representatives": len(chosen), "screened": len(shortlist), "budget_fill": fill}
    return np.asarray(chosen, dtype=int), tiers, info, folds


def grouped_control_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """Select the first h representatives by relevance, omitting witness voting."""
    reps, _, info, _ = _representatives(X, y, task, random_state, count, tiers)
    return np.sort(reps[:count]), info


def grouped_top_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """Construct neighbors and majority votes on a reduced set of representatives.

    A 256-row training sketch ranks columns by between-class variance fraction
    or squared correlation with the regression response. In that order, skip
    candidates with absolute correlation at least 0.90 to a retained column.
    Search up to max(128, 4h) candidates for max(2h, max(tiers)+1) representatives.
    If fewer than h survive, fill the budget in relevance order. Peer counts
    are capped at the remaining pool size minus one. The final pivotal cutoff
    uses the original column ID for ties.
    """
    reps, tiers, info, folds = _representatives(X, y, task, random_state, count, tiers)
    reps = np.sort(reps)
    if len(reps) == count:
        return reps, dict(info, arithmetic="all")
    margins = _witness_margins_from_folds(np.asarray(X), np.asarray(y), task, *folds, columns=reps)
    votes = np.sign(margins).astype(np.int8)
    effective = sorted({min(k, len(reps) - 1) for k in tiers})
    support, arithmetic = exact_top_from_votes(votes, count, tiers=effective)
    return reps[support], dict(info, **arithmetic)
