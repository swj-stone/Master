"""
Task-agnostic utility functions — decoupled from model training.

Every function accepts ground-truth labels and predictions, returns a
float in ``[0, 1]`` (or higher-is-better range).  The caller is
responsible for training the model and producing predictions; this
module only answers "how good are these predictions under metric *M*?"

Metric identifiers
------------------
Classification (``y_score`` = probability matrix of shape (n, n_classes)):
  - ``"roc_auc"``      — binary ROC-AUC (uses ``y_score[:, 1]``)
  - ``"ovr_auc"``      — macro OVR ROC-AUC (multi_class="ovr", average="macro")
  - ``"accuracy"``      — argmax accuracy
  - ``"pr_auc"``        — binary PR-AUC (average_precision)

Regression (``y_pred`` = predicted continuous values, shape (n,)):
  - ``"r2"``            — R² score (clamped to ≥ 0)
  - ``"rmse"``          — RMSE normalised to [0,1] via 1/(1+RMSE)

Usage::

    from minppi_mofs.utility import compute_utility

    auc = compute_utility(y_val, y_proba, metric="roc_auc")
    r2  = compute_utility(y_val, y_pred,  metric="r2")
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CLASSIFICATION_METRICS = frozenset({"roc_auc", "ovr_auc", "accuracy", "pr_auc"})
_REGRESSION_METRICS = frozenset({"r2", "rmse"})
_ALL_METRICS = _CLASSIFICATION_METRICS | _REGRESSION_METRICS


def compute_utility(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "roc_auc",
) -> float:
    """Compute a utility score from ground truth and (probability) predictions.

    Args:
        y_true: Ground-truth labels (int for classification, float for regression).
        y_score: For classification: probability matrix ``(n_samples, n_classes)``.
            For regression: continuous predictions ``(n_samples,)``.
        metric: One of ``"roc_auc"``, ``"ovr_auc"``, ``"accuracy"``, ``"pr_auc"``,
            ``"r2"``, ``"rmse"``.

    Returns:
        Utility value — always in ``[0, 1]`` (higher is better).
    """
    if metric not in _ALL_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}.  Choose from {sorted(_ALL_METRICS)}."
        )

    if metric in ("r2", "rmse"):
        return _regression_utility(y_true, y_score, metric=metric)

    # Classification — y_score is probability matrix
    classes = np.unique(y_true)
    n_cls = len(classes)

    if metric == "roc_auc":
        return _binary_roc_auc(y_true, y_score)
    if metric == "ovr_auc":
        return _ovr_auc(y_true, y_score, n_cls)
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_score.argmax(axis=1)))
    # pr_auc
    return _binary_pr_auc(y_true, y_score)


def is_classification_metric(metric: str) -> bool:
    """Return True if *metric* is a classification utility."""
    return metric in _CLASSIFICATION_METRICS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        return 0.5
    if y_score.shape[1] == 2:
        return float(roc_auc_score(y_true, y_score[:, 1]))
    # multi-class but metric=roc_auc → use OVR macro as fallback
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def _ovr_auc(y_true: np.ndarray, y_score: np.ndarray, n_cls: int) -> float:
    if n_cls < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def _binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        return 0.5
    if y_score.shape[1] == 2:
        return float(average_precision_score(y_true, y_score[:, 1]))
    # fallback: OVR PR-AUC
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=np.arange(y_score.shape[1]))
    aps = []
    for c in range(y_score.shape[1]):
        if y_bin[:, c].sum() == 0:
            continue
        aps.append(average_precision_score(y_bin[:, c], y_score[:, c]))
    return float(np.mean(aps)) if aps else 0.5


def _regression_utility(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "r2",
) -> float:
    if metric == "r2":
        val = float(r2_score(y_true, y_pred))
        return max(val, 0.0)
    # rmse → normalised to [0, 1]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return float(1.0 / (1.0 + rmse))
