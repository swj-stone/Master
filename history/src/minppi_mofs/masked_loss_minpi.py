"""Pivotal minPI votes from masking a coordinate in a cross-fitted joint ridge."""
import numpy as np
import pandas as pd
from scipy.linalg import solve

from minppi_mofs.piv2 import _folds
from minppi_mofs.word_minpi import exact_top_from_votes


def _standardize(reference, query):
    """Center, mean-impute, and scale both matrices from the reference only."""
    observed = np.isfinite(reference)
    mean = np.where(observed, reference, 0).sum(axis=0) / np.maximum(observed.sum(axis=0), 1)
    centered = np.where(observed, reference - mean, 0)
    scale = np.sqrt(np.square(centered).mean(axis=0))
    low = np.where(observed, reference, np.inf).min(axis=0)
    high = np.where(observed, reference, -np.inf).max(axis=0)
    constant = (high <= low) | (scale == 0)
    scale[constant] = 1
    centered[:, constant] = 0
    centered_query = np.where(np.isfinite(query), query - mean, 0)
    centered_query[:, constant] = 0
    return centered / scale, centered_query / scale


def _cross_fitted_values(X, y, task, seed, *, singleton):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or not X.shape[1] or y.shape != (len(X),) or pd.isna(y).any():
        raise ValueError("Expected a nonempty matrix and complete aligned targets")
    if task not in {"classification", "regression"}:
        raise ValueError("Expected classification or regression")
    if task == "regression" and not np.isfinite(y.astype(float)).all():
        raise ValueError("Regression requires finite targets")
    folds = _folds(y, task, np.random.default_rng(seed), row_limit=256)
    if folds is None or min(map(len, folds)) < 2:
        raise ValueError("Insufficient training observations for cross-fitting")
    sample = np.concatenate(folds)
    X = np.asarray(X[sample], dtype=float)
    if task == "classification":
        target = (y[sample, None] == np.unique(y[sample])).astype(float)
    else:
        target = np.asarray(y[sample, None], dtype=float)
    halves = (slice(0, len(folds[0])), slice(len(folds[0]), len(sample)))
    values = np.empty(X.shape, dtype=float if singleton else np.int8)
    for reference, query in (halves, halves[::-1]):
        ref, check = _standardize(X[reference], X[query])
        intercept = target[reference].mean(axis=0)
        centered = target[reference] - intercept
        centered[:, np.ptp(target[reference], axis=0) == 0] = 0
        # The system is at most the size of the reference fold even for wide X.
        gram = ref @ ref.T
        gram.flat[::len(gram) + 1] += 1
        coefficients = ref.T @ solve(gram, centered, assume_a="pos", check_finite=False)
        squared = np.square(check) * np.square(coefficients).sum(axis=1)
        if singleton:
            values[query] = 2 * check * ((target[query] - intercept) @ coefficients.T) - squared
        else:
            residual = target[query] - (intercept + check @ coefficients)
            gain = 2 * check * (residual @ coefficients.T) + squared
            values[query] = np.sign(gain).astype(np.int8)
    return values


def masked_loss_votes(X, y, task, seed):
    """Compare held-out squared loss with each coordinate masked versus observed.

    Masking replaces one coordinate by its training-fold mean; all coefficients
    remain fixed. Use opposite-fold ridge with unit penalty and an intercept.
    A positive vote means the observed coordinate reduces the query's squared
    loss, summed over one-hot targets for classification. The split and 256-row
    cap match the singleton-loss ballots.
    """
    return _cross_fitted_values(X, y, task, seed, singleton=False)


def joint_ridge_cis_scores(X, y, task, seed):
    """CIS order for the cross-fit negative squared-loss game with mean masking.

    For v(S) equal to negative held-out squared loss, the CIS correction is
    common to all features. Rank by v({j}) - v(empty), leaving only feature j
    observed in the already fitted joint model; there is no singleton refit.
    """
    return _cross_fitted_values(X, y, task, seed, singleton=True).mean(axis=0)


def masked_loss_top_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """Compute the exact weakest-layer support for joint-ridge masking ballots."""
    votes = masked_loss_votes(X, y, task, random_state)
    columns, info = exact_top_from_votes(votes, count, tiers=tiers)
    return columns, {**info, "ballot": "masked_loss", "ballot_rows": len(votes)}
