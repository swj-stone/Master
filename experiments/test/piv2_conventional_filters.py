"""Training-only conventional filters for the extended paired PIv2 test."""

from math import ceil, sqrt

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split


ROW_LIMIT = 4096
N_BINS = 8
FILTERS = ("spearman", "mrmr", "chi2", "variance")


def _sample_rows(y, regression, seed):
    rows = np.arange(len(y))
    if len(rows) > ROW_LIMIT:
        rows, _ = train_test_split(rows, train_size=ROW_LIMIT, random_state=seed,
                                   stratify=None if regression else y)
        rows.sort()
    return rows


def _spearman(x, y, regression):
    ranked = rankdata(x, axis=0)
    ranked -= ranked.mean(axis=0)
    scale = np.linalg.norm(ranked, axis=0)
    classes, counts = np.unique(y, return_counts=True)
    if regression or len(classes) <= 2:
        outcomes = [rankdata(y)]
        weights = [1.]
    else:
        outcomes = [(y == value).astype(float) for value in classes]
        weights = counts / len(y)
    score = np.zeros(x.shape[1])
    for response, weight in zip(outcomes, weights, strict=True):
        response -= response.mean()
        denom = scale * np.linalg.norm(response)
        corr = np.divide(np.einsum("ij,i->j", ranked, response), denom,
                         where=denom > 0, out=np.zeros_like(scale))
        score += weight * np.abs(corr)
    return score


def _quantile_codes(values):
    levels, encoded, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(levels) <= N_BINS:
        return encoded
    positions = np.searchsorted(np.cumsum(counts), np.arange(1, N_BINS)*len(values)/N_BINS)
    cuts = np.unique(np.minimum(positions, len(levels)-2))
    return np.searchsorted(cuts, encoded)


def _association(a, b, statistic):
    # The encoded levels are consecutive. Dropped duplicate quantiles never create empty bins.
    size_b = int(b.max()) + 1
    observed = np.bincount(a * size_b + b, minlength=(int(a.max())+1)*size_b).reshape(-1, size_b)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(a)
    if statistic == "chi2":
        return np.sum((observed - expected)**2 / expected)
    nonzero = observed > 0
    return np.sum(observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])) / len(a)


def _mrmr(columns, relevance, k):
    """Greedy maximal relevance minus mean redundancy (empirical mutual information, nats)."""
    selected = []
    criterion = []
    redundancy = np.zeros(len(columns))
    for step in range(k):
        scores = relevance.copy() if step == 0 else relevance - redundancy / step
        scores[selected] = -np.inf
        best = int(np.argmax(scores))
        selected.append(best)
        criterion.append(float(scores[best]))
        if step + 1 < k:
            for j, column in enumerate(columns):
                if j not in selected:
                    redundancy[j] += _association(column, columns[best], "mi")
    return np.asarray(selected), criterion


def choose_filter(backend, train, yt, policy, *, original_width, full_order=False):
    """Return retained names in input order and the unrounded training ranking scores."""
    width = train.shape[1]
    if backend not in FILTERS or not 1 <= width <= original_width or not train.columns.is_unique:
        raise ValueError("Invalid filter or original feature width")
    y = np.asarray(yt)
    if y.ndim != 1 or not len(y) or len(y) != len(train):
        raise ValueError("Unaligned training labels")
    regression = policy["task"] == "regression"
    rows = _sample_rows(y, regression, policy["seed"])
    x = train.iloc[rows].to_numpy(dtype=float)
    y = y[rows]
    if not np.isfinite(x).all():
        raise ValueError("Filters require finite preprocessed features")
    k = width if full_order else min(width, ceil(sqrt(original_width)))
    info = dict(k=k, ranking_sample_rows=len(rows), row_limit=ROW_LIMIT)
    if backend == "variance":
        scores = np.var(x, axis=0)
        info["scale"] = "variance on the frozen input scale"
    elif backend == "spearman":
        scores = _spearman(x, y, regression)
    else:
        codes = [_quantile_codes(col) for col in x.T]
        response = _quantile_codes(y) if regression else np.unique(y, return_inverse=True)[1]
        scores = np.array([_association(col, response, "chi2" if backend == "chi2" else "mi")
                           for col in codes])
        info.update(max_bins=N_BINS, target_binned=regression)
    if backend == "mrmr":
        order, info["mid_by_step"] = _mrmr(codes, scores, k)
    else:
        order = np.argsort(-scores, kind="stable")[:k]
    selected = train.columns[np.sort(order)].tolist()
    info.update(selected=selected, ranking_scores=scores.tolist(), choice_order=train.columns[order].tolist())
    return selected, info
