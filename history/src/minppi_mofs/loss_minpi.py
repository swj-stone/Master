"""Pivotal minPI with cross-fitted univariate prediction-loss ballots."""
import numpy as np
import pandas as pd

from minppi_mofs.piv2 import _folds
from minppi_mofs.word_minpi import exact_top_from_votes


def held_loss_votes(X, y, task, seed):
    """Return two-fold out-of-fold signs of intercept loss minus singleton loss.

    At most 256 training observations enter the two disjoint folds. Each feature
    fits a unit-penalty linear ridge on the opposite fold, using reference-only
    standardization and mean imputation. Classification uses squared loss over
    one-hot class targets, regression uses squared error. Zero gain is abstention.
    Rows are in sampled fold order and columns remain in input order.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or not X.shape[1] or y.shape != (len(X),) or pd.isna(y).any():
        raise ValueError("Expected a nonempty feature matrix with complete, aligned targets")
    if task not in {"classification", "regression"}:
        raise ValueError("Expected classification or regression")
    if task == "regression" and not np.isfinite(y.astype(float)).all():
        raise ValueError("Regression requires finite targets")
    folds = _folds(y, task, np.random.default_rng(seed), row_limit=256)
    if folds is None or min(map(len, folds)) < 2:
        raise ValueError("Insufficient training observations for cross-fitted loss ballots")
    left, right = folds
    sample = np.concatenate((left, right))
    if task == "classification":
        classes = np.unique(y[sample])
        target = (y[sample, None] == classes).astype(float)
    else:
        target = y[sample, None].astype(float)
    matrix = np.asarray(X[sample], dtype=float)
    halves = (slice(0, len(left)), slice(len(left), len(sample)))
    votes = np.empty(matrix.shape, dtype=np.int8)
    for reference, query in (halves, halves[::-1]):
        ref = matrix[reference]
        finite = np.isfinite(ref)
        center = np.where(finite, ref, 0).sum(axis=0) / np.maximum(1, finite.sum(axis=0))
        centered = np.where(finite, ref - center, 0)
        scale = np.sqrt(np.square(centered).mean(axis=0))
        # A floating constant's computed mean can differ from the original value.
        low = np.where(finite, ref, np.inf).min(axis=0)
        high = np.where(finite, ref, -np.inf).max(axis=0)
        constant = (high <= low) | (scale == 0)
        centered[:, constant] = 0
        scale[constant] = 1
        z_ref = centered / scale
        z_query = np.where(np.isfinite(matrix[query]), matrix[query] - center, 0) / scale
        z_query[:, constant] = 0
        intercept = target[reference].mean(axis=0)
        centered_target = target[reference] - intercept
        centered_target[:, np.ptp(target[reference], axis=0) == 0] = 0
        coefficient = (z_ref.T @ centered_target) / (
            np.square(z_ref).sum(axis=0)[:, None] + 1
        )
        # A fitted increment d changes squared error by 2*(y-intercept).d - d.d.
        gain = (2 * z_query * ((target[query] - intercept) @ coefficient.T)
                - np.square(z_query) * np.square(coefficient).sum(axis=1))
        votes[query] = np.sign(gain).astype(np.int8)
    return votes


def loss_top_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """Return the exact weakest-layer support for cross-fitted loss ballots."""
    votes = held_loss_votes(X, y, task, random_state)
    columns, info = exact_top_from_votes(votes, count, tiers=tiers)
    return columns, {**info, "ballot": "held_loss", "ballot_rows": len(votes)}
