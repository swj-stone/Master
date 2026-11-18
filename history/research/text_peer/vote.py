"""Binary document minPI with all-document leave-one-out evidence votes."""
import numpy as np
from scipy import sparse


def presences(X):
    if not sparse.issparse(X):
        raise ValueError("Expected a sparse matrix of binary document features")
    X = X.tocsr(copy=False)
    if not X.has_canonical_format or not np.all(X.data == 1):
        raise ValueError("Expected canonical binary CSR without explicit zeros")
    return X


def evidence_votes(X, y, *, row_normalization="sqrt"):
    """Evaluate equal-class LOO log-payoff margins for k=0 and k=p-1.

    Within one held document, per-word evidence coefficients and document
    normalization are fixed across all coalitions. A word that is absent in
    that document abstains. Probabilities have a one-observation prior with
    the class distribution of the remaining documents.
    """
    X = presences(X)
    y = np.asarray(y)
    if y.shape != (X.shape[0],) or not X.shape[1]:
        raise ValueError("Expected an aligned label vector and at least one column")
    classes, labels, sizes = np.unique(y, return_inverse=True, return_counts=True)
    if len(classes) != 2 or min(sizes) < 2:
        raise ValueError("Two classes with at least two documents each required")
    rows = np.repeat(np.arange(len(y)), np.diff(X.indptr))
    freq = np.bincount(X.indices, minlength=X.shape[1])
    positive = np.bincount(X.indices, weights=labels[rows], minlength=X.shape[1])
    counts = np.stack([freq - positive, positive])
    zero, peers = np.zeros((2, X.shape[1]), dtype=float)
    for label in (0, 1):
        row_mask = labels[rows] == label
        active = rows[row_mask]
        cols = X.indices[row_mask]
        prior = sizes.astype(float)
        prior[label] -= 1
        prior /= prior.sum()
        same = np.maximum(0, counts[label] - 1)
        other = counts[1 - label]
        word = np.log1p(same / prior[label]) - np.log1p(other / prior[1 - label])
        length = np.maximum(np.diff(X.indptr), 1)
        if row_normalization == "sqrt":
            scale = np.sqrt(length)
        elif row_normalization == "unit":
            scale = np.ones_like(length)
        else:
            raise ValueError("Unknown normalization")
        component = word[cols] / scale[active]
        total = np.bincount(active, weights=component, minlength=len(y))[active]
        single_gain = np.log(2.) - np.logaddexp(0, -component)
        contextual_gain = np.logaddexp(0, component - total) - np.logaddexp(0, -total)
        zero += np.bincount(cols, weights=single_gain, minlength=X.shape[1]) / (2 * sizes[label])
        peers += np.bincount(cols, weights=contextual_gain, minlength=X.shape[1]) / (2 * sizes[label])
    return zero, peers


def support(X, y, *, count=None, rule="both", row_normalization="sqrt"):
    zero, all_peers = evidence_votes(X, y, row_normalization=row_normalization)
    if rule not in ("both", "zero", "all"):
        raise ValueError("Unknown vote rule")
    scores = {"both": np.minimum(zero, all_peers), "zero": zero, "all": all_peers}[rule]
    # A strictly positive LOO payoff in both games is required for auto width.
    if count is None:
        count = max(1, np.count_nonzero(scores > 1e-15))
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= len(scores):
        raise ValueError("Invalid width")
    order = np.lexsort((np.arange(len(scores)), -np.round(zero, 14), -np.round(scores, 14)))
    return np.sort(order[:count])
