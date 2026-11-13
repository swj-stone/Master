"""PIv2 training-only prefilter for numeric feature matrices."""

from math import ceil, sqrt

import numpy as np
import pandas as pd


ROW_LIMIT = 1024
N_BINS = 8
PRIOR_WEIGHT = 8
REPEATS = 4
TIE_SCALE = 0.1


def _folds(y, task, rng):
    """Sample balanced class quotas, sharing unfilled slots among larger classes."""
    if task == "regression":
        sample = rng.choice(len(y), min(ROW_LIMIT, len(y)), replace=False)
        return np.array_split(sample, 2)
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2 or len(values) > ROW_LIMIT // 2 or min(counts) < 2:
        return None
    budget = min(len(y), ROW_LIMIT)
    quotas = np.zeros(len(values), dtype=int)
    for j, idx in enumerate(np.argsort(counts, kind="stable")):
        quotas[idx] = min(counts[idx], budget // (len(counts) - j))
        budget -= quotas[idx]
    folds = [[], []]
    offset = 0
    for value, quota in zip(values, quotas, strict=True):
        rows = rng.choice(np.flatnonzero(y == value), quota, replace=False)
        folds[offset].extend(rows[::2])
        folds[1-offset].extend(rows[1::2])
        offset = (offset + quota) % 2
    return [np.asarray(indices, dtype=int) for indices in folds]


def _bins(train, heldout):
    """Up to eight ordered bins plus a separate bin for missing values."""
    values, counts = np.unique(train[np.isfinite(train)], return_counts=True)
    cuts = np.empty(0)
    if len(values) > 1:
        targets = np.arange(1, N_BINS) * counts.sum() / N_BINS
        positions = np.minimum(np.searchsorted(np.cumsum(counts), targets), len(values)-2)
        cuts = values[np.unique(positions)]
    def encode(x):
        return np.where(np.isfinite(x), np.searchsorted(cuts, x), len(cuts)+1)
    return encode(train), encode(heldout)


def _fold_signals(x_train, x_check, y_train, y_check, task, outputs):
    if task == "regression":
        scale = np.std(y_train) or 1.0
        target = (y_train[:, None] - y_train.mean()) / scale
        check = (y_check[:, None] - y_train.mean()) / scale
    else:
        target = (y_train[:, None] == outputs).astype(float)
        prior = target.mean(axis=0)
        target -= prior
        check = (y_check[:, None] == outputs).astype(float) - prior
    width = x_train.shape[1]
    signals = np.empty((width, len(x_check), target.shape[1]), dtype=float)
    for j in range(width):
        fit_bins, check_bins = _bins(x_train[:, j], x_check[:, j])
        totals = np.zeros((N_BINS + 1, target.shape[1]))
        np.add.at(totals, fit_bins, target)
        counts = np.bincount(fit_bins, minlength=N_BINS + 1)
        signals[j] = (totals / (counts[:, None] + PRIOR_WEIGHT))[check_bins]
    return signals.reshape(width, -1), check.ravel()


def _changes(signals, target, norms, order, k):
    """Mean squared error decrease on adding i to a random k-set excluding i.

    A single random permutation provides each i with its own uniform k-set:
    members of the first k are replaced by the next element when scored.
    Predictions are means of centered single-feature signals. Expanding both
    squares lets all columns reuse the same two coalition sums.
    """
    total = signals[order[:k]].sum(axis=0)
    result = np.empty(len(order))
    for inside, indices in ((False, order[k:]), (True, order[:k])):
        common = total + signals[order[k]] if inside else total
        before = target - common / k
        after = target - common / (k+1)
        shared = (before @ before - after @ after) / target.size
        if inside:
            # For i inside, its signal is absent before and present after.
            dot = np.einsum("ij,j->i", signals, before) / target.size
            result[indices] = shared + 2*dot[indices]/k + norms[indices]/k**2
        else:
            dot = np.einsum("ij,j->i", signals, after) / target.size
            result[indices] = shared + 2*dot[indices]/(k+1) - norms[indices]/(k+1)**2
    return result


def _selection(net, standalone):
    width = len(standalone)
    rates = net.sum(axis=0) / (net.shape[0] * REPEATS)
    worst = rates.min(axis=0)
    corroborated = np.any(np.all(net < 0, axis=0), axis=0)
    protected = np.argsort(-standalone, kind="stable")[:ceil(sqrt(width))]
    eligible = (worst < 0) & corroborated
    eligible[protected] = False
    weakest = np.lexsort((np.arange(width), standalone, worst))
    removed = weakest[eligible[weakest]][:width//2]
    return rates, worst, protected, removed


def prefilter(train, labels, *, task, seed=42):
    """Return original column names retained for a downstream selector and diagnostics.

    The fitted surrogate uses only rows of the supplied training partition.
    Classification uses squared probability errors; regression uses squared
    errors in units of the target standard deviation of each encoding fold.
    """
    if not isinstance(train, pd.DataFrame) or train.columns.has_duplicates or not train.shape[1]:
        raise ValueError("Expected a DataFrame with nonempty, unique feature columns")
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(train) or not len(y) or pd.isna(y).any():
        raise ValueError("Training labels must be a nonempty, complete aligned vector")
    if task not in {"binary", "multiclass", "regression"}:
        raise ValueError("Expected binary, multiclass, or regression")
    if task == "regression" and not np.isfinite(y).all():
        raise ValueError("Regression labels must be finite")
    columns = train.columns.tolist()
    width = len(columns)
    diagnostics = dict(columns=columns, retained=columns, input_count=width, sample_rows=0,
                       tier_sizes=[], fold_tier_net=[], tier_rates=[], min_vote=[], standalone=[],
                       protected=[], removed=[], bypass=None)
    row_seed, order_seed = np.random.SeedSequence(seed).spawn(2)
    if width < 3 or len(y) < 16:
        diagnostics["bypass"] = "insufficient width or rows"
        return columns, diagnostics
    folds = _folds(y, task, np.random.default_rng(row_seed))
    if folds is None:
        diagnostics["bypass"] = "insufficient class support for two stratified folds"
        return columns, diagnostics
    # Slice rows before converting all columns to a dense numeric matrix.
    sampled = [train.iloc[positions].to_numpy(dtype=float) for positions in folds]
    outcomes = [y[positions] for positions in folds]
    outputs = np.unique(outcomes[0]) if task != "regression" else None
    tiers = np.clip(np.rint((width-1)*np.array([1/6, 1/2, 5/6])).astype(int), 1, width-1)
    rng = np.random.default_rng(order_seed)
    net = np.zeros((2, 3, width), dtype=int)
    standalone = np.zeros(width)
    for fold in range(2):
        signals, target = _fold_signals(sampled[fold], sampled[1-fold], outcomes[fold], outcomes[1-fold],
                                        task, outputs)
        norms = np.einsum("ij,ij->i", signals, signals) / target.size
        standalone += (2*np.einsum("ij,j->i", signals, target)/target.size - norms) / 2
        for h, k in enumerate(tiers):
            for _ in range(REPEATS):
                changes = _changes(signals, target, norms, rng.permutation(width), int(k))
                residual = changes - np.median(changes)
                margin = max(1e-12, TIE_SCALE*np.median(np.abs(residual)))
                net[fold, h] += (residual > margin).astype(int) - (residual < -margin).astype(int)
    rates, worst, protected, removed = _selection(net, standalone)
    drop_set = set(removed)
    retained = [c for i,c in enumerate(columns) if i not in drop_set]
    diagnostics.update(retained=retained, sample_rows=sum(map(len, folds)), tier_sizes=tiers.tolist(),
                       fold_tier_net=net.tolist(), tier_rates=rates.tolist(), min_vote=worst.tolist(),
                       standalone=standalone.tolist(), protected=[columns[i] for i in protected],
                       removed=[columns[i] for i in sorted(drop_set)])
    return retained, diagnostics
