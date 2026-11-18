"""Choose columns by the weakest held-out loss gain across nested ridge refits."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.piv2 import _folds


def _refit_votes(fit, check, fit_target, check_target, selected):
    """Sum the held-row squared-error reductions of all one-column augmentations.

    Inputs are centered using fitting-fold statistics. Ridge penalty is one.
    The Schur complement gives the fitted update for every candidate using a
    single factorization of the selected-set ridge system.
    """
    diagonal = np.einsum("ij,ij->j", fit, fit) + 1
    rhs = fit.T @ fit_target
    if len(selected):
        active = fit[:, selected]
        active_check = check[:, selected]
        gram = active.T @ active
        gram.flat[::len(selected) + 1] += 1
        factor = cho_factor(gram, check_finite=False)
        coefficients = cho_solve(factor, rhs[selected], check_finite=False)
        cross = active.T @ fit
        projection = cho_solve(factor, cross, check_finite=False)
        rhs -= cross.T @ coefficients
        diagonal -= np.einsum("ij,ij->j", cross, projection)
        check = check - active_check @ projection
        check_target = check_target - active_check @ coefficients
    update = rhs / diagonal[:, None]
    linear = np.einsum("ij,ij->i", check.T @ check_target, update)
    quadratic = np.einsum("ij,ij->j", check, check) * np.einsum("ij,ij->i", update, update)
    return 2 * linear - quadratic


def refit_top_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """Add features in blocks, ranking by worst mean gain across visited support sizes.

    Each row votes its signed decrease in held-out squared loss when j is
    added to the current support and the coefficients are refitted. The
    selection path visits sizes 0, 1, 2, 4, ... < count by default. Explicit
    sizes must contain 0 and be below count. At each size the weakest mean
    vote for each available column determines the next block. Thus each size
    uses the realized selected coalition, not a draw from all equally sized
    subsets. Training folds share the path, independently of outer test rows.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),) or pd.isna(y).any():
        raise ValueError("Expected a nonempty matrix and aligned complete targets")
    if task not in ("classification", "regression"):
        raise ValueError("Expected classification or regression")
    if task == "regression" and not np.isfinite(y.astype(float)).all():
        raise ValueError("Regression targets must be finite")
    p = X.shape[1]
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= p:
        raise ValueError("count must be an integer in [1,p]")
    if tiers is None:
        sizes = [0]
        size = 1
        while size < count:
            sizes.append(size)
            size *= 2
    else:
        sizes = sorted(_selection_tiers(tiers, p))
        if sizes[0] != 0 or sizes[-1] >= count:
            raise ValueError("Refit tiers must include zero and be smaller than the budget")
    if count == p:
        return np.arange(p), {"sizes": [], "ballot_rows": 0}
    folds = _folds(y, task, np.random.default_rng(random_state), row_limit=min(1024, max(256, 2*p)))
    if folds is None or min(map(len, folds)) < 2:
        raise ValueError("Insufficient training observations for cross-fitting")
    classes = np.unique(y) if task == "classification" else None
    data = []
    for ref, query in (folds, folds[::-1]):
        fit, check = _standardize(X[ref], X[query])
        if classes is None:
            fit_target = y[ref, None].astype(float)
            check_target = y[query, None].astype(float)
        else:
            fit_target = (y[ref, None] == classes).astype(float)
            check_target = (y[query, None] == classes).astype(float)
        intercept = fit_target.mean(axis=0)
        data.append((fit, check, fit_target - intercept, check_target - intercept))
    selected = np.empty(0, dtype=int)
    available = np.ones(p, dtype=bool)
    minimum = np.full(p, np.inf)
    for size, next_size in zip(sizes, sizes[1:] + [count]):
        summed = sum(_refit_votes(*fold, selected) for fold in data)
        minimum = np.minimum(minimum, summed)
        pool = np.flatnonzero(available)
        chosen = pool[np.argsort(-minimum[pool], kind="stable")[:next_size-size]]
        selected = np.concatenate((selected, chosen))
        available[chosen] = False
    return np.sort(selected), {"sizes": sizes, "ballot_rows": sum(map(len, folds))}
