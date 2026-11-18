"""Optimized binary LOO endpoint sums and optional one-standard-error screening."""
import numpy as np
from scipy.special import expit
from research.text_peer.vote import presences


def endpoint_statistics(X, y):
    X = presences(X)
    y = np.asarray(y)
    if y.shape != (X.shape[0],) or not X.shape[1]:
        raise ValueError("Expected an aligned binary label vector and at least one column")
    classes, y, sizes = np.unique(y, return_inverse=True, return_counts=True)
    if len(classes) != 2 or sizes.min() < 2:
        raise ValueError("Each of two classes needs at least two training rows")
    positive = X[y == 1]
    counts = np.empty((2, X.shape[1]), dtype=np.int64)
    counts[1] = np.bincount(positive.indices, minlength=X.shape[1])
    counts[0] = np.bincount(X.indices, minlength=X.shape[1]) - counts[1]
    gain = np.zeros_like(counts, dtype=float)
    var = np.zeros_like(gain)
    for label, block in ((1, positive), (0, X[y == 0])):
        n = sizes[label]
        prior = (n - 1) / (len(y) - 1)
        component = np.log1p(np.maximum(0,counts[label] - 1) / prior) - np.log1p(counts[1-label] / (1-prior))
        single = np.log(2.) - np.logaddexp(0., -component)
        row_logit = block @ component
        residual = np.repeat(expit(-row_logit), np.diff(block.indptr))
        ratio = np.expm1(component)
        # For negative components, log1p is stable except when the ratio
        # rounds to -1. Finite count bounds keep this above -1 on task data.
        context = np.log1p(residual * ratio[block.indices])
        sums = np.stack([counts[label] * single, np.bincount(block.indices, weights=context, minlength=X.shape[1])])
        squares = np.stack([counts[label] * single**2, np.bincount(block.indices, weights=context**2, minlength=X.shape[1])])
        gain += sums / (2 * n)
        var += np.maximum(0., squares - sums**2 / n) / (4 * n * (n-1))
    return gain, np.sqrt(var)


def rank_words(X,y, *, count=None, rule="both", penalty=0.):
    gain, uncertainty = endpoint_statistics(X, y)
    if rule not in ("both", "zero", "all") or penalty < 0:
        raise ValueError("Unknown rule or negative penalty")
    lower = gain - penalty * uncertainty
    score = {"both":lower.min(axis=0), "zero":lower[0], "all":lower[1]}[rule]
    if count is None:
        count = max(1, np.count_nonzero(score > 1e-15))
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or not 1 <= count <= len(score):
        raise ValueError("Invalid width")
    return np.sort(np.lexsort((np.arange(len(score)), -np.round(gain[0],14), -np.round(score,14)))[:count])
