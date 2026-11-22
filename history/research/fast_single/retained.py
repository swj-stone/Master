"""Opt-in paper selector with minority-panel repair and compact sparse calculation.

This entry point uses the repaired panel's natural width. The original paper
selector remains available as research.fast_single.selector.select_fast.
"""
import numpy as np
from scipy import sparse

from research.fast_single import selector as paper
from research.vote_frontier.fastpath import select_arch


def select_retained(X, y, *, task, seed=1129):
    """Select columns on the training partition and return the vote diagnostics."""
    if task == "regression":
        return select_arch(X, y, task=task, seed=seed)
    if task != "classification":
        raise ValueError("Expected classification or regression")
    X = X.tocsr() if sparse.issparse(X) else np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (X.shape[0],):
        raise ValueError("Expected a nonempty matrix and aligned response")
    n, p = X.shape
    m = min(n, 768 if p <= paper.MAX_FEATURES else paper.ROWS)
    rows = np.arange(m, dtype=np.int64)
    if m < n:
        paper.prepare_sampler().sample_rows(n, m, seed, rows.ctypes.data)
    first_label = y[rows[0]]
    if np.any(y[rows] != first_label):
        return select_arch(X, y, task=task, seed=seed)

    labels, first, counts = np.unique(y, return_index=True, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Classification requires at least two training classes")
    if len(labels) > m:
        zeros = np.zeros((2, p))
        return np.arange(p), dict(rank=zeros[0], tier_scores=zeros, width=p,
                                  selected_width=p, sketch_size=m, null_scale=0.,
                                  repair_status="neutral")
    quota = min(int(np.ceil(np.sqrt(m))), m // len(labels))
    rng = np.random.default_rng([int(seed), 1996227])
    added = []
    for i in np.argsort(first):
        if labels[i] != first_label:
            population = np.flatnonzero(y == labels[i])
            added.append(rng.choice(population, min(int(counts[i]), quota), replace=False))
    extra = np.concatenate(added)
    kept = rng.choice(rows, m-len(extra), replace=False)
    panel = np.sort(np.concatenate((kept, extra)))
    cols, info = select_arch(X[panel], y[panel], task=task, seed=seed)
    return cols, dict(info, repaired_rows=panel, repair_status="repaired")
