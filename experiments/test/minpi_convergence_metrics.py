"""Reuse each class's score ordering across all votes on the same masked predictions."""

import numpy as np
from sklearn.metrics import average_precision_score

from minppi_mofs.piv3_compact_witness import _measure


def make_scorer(y, weights, task, utility, rare):
    """Return one score dict for each fixed, nonempty panel (or bootstrap draw).

    The binary AUC is the weighted fraction of positive/negative pairs in
    descending quality order, with half credit for ties. Zero-weight rows
    exclude an observation from that vote. Multiclass uses the macro OvR mean.
    """
    labels = np.asarray(y)
    weights = np.asarray(weights, dtype=float)
    if (labels.ndim != 1 or weights.ndim != 2 or weights.shape[1] != len(labels)
            or not len(weights) or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Expected aligned labels and nonnegative finite panel weights")
    masses = weights.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("Every panel needs positive observation mass")
    if task == "regression":
        return lambda predictions: [_measure(labels, predictions, w, task, rare) for w in weights]

    class_ids = np.unique(labels)
    if not np.array_equal(class_ids, np.arange(len(class_ids))) or len(class_ids) < 2:
        raise ValueError("Expected consecutive class IDs starting at zero for all model outputs")
    outputs = [1] if task == "binary" else class_ids
    positives = labels[None, :] == np.asarray(outputs)[:, None]
    positive_mass = weights @ positives.T
    negative_mass = weights @ (~positives).T
    if np.any(positive_mass <= 0) or np.any(negative_mass <= 0):
        raise ValueError("Each voting panel needs positive and negative mass for every AUC")
    pair_mass = positive_mass * negative_mass

    def score(predictions):
        pred = np.asarray(predictions)
        if pred.shape != (len(labels), len(class_ids)) or not np.isfinite(pred).all():
            raise ValueError("Expected one finite prediction per observation and class")
        accuracy = (weights @ (pred.argmax(axis=1) == labels)) / masses
        numerator = np.empty_like(pair_mass)
        for j, column in enumerate(outputs):
            order = np.argsort(pred[:, column], kind="stable")
            sorted_scores = pred[order, column]
            # Collapse equal scores before the cumulative sum so row order cannot break a tie.
            starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
            sorted_weights = weights[:, order]
            pos = positives[j, order]
            group_pos = np.add.reduceat(sorted_weights * pos, starts, axis=1)
            group_neg = np.add.reduceat(sorted_weights * (~pos), starts, axis=1)
            credit = np.cumsum(group_neg, axis=1) - .5 * group_neg
            numerator[:, j] = (group_pos * credit).sum(axis=1)
        auc = (numerator / pair_mass).mean(axis=1)
        result = [dict(accuracy=float(a), auc=float(b)) for a, b in zip(accuracy, auc, strict=True)]
        if task == "binary" and rare:
            for metrics, panel in zip(result, weights, strict=True):
                metrics["ap"] = float(average_precision_score(labels, pred[:, 1], sample_weight=panel))
        return result

    return score
