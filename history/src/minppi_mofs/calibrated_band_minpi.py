"""Training-validated dyadic refinement of minPI-voted spectral intervals."""
import numpy as np
from scipy.linalg import solve

from minppi_mofs.band_vote_minpi import band_vote_top_from_training
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.moment_minpi import _draw, _moment_halves


def _refine(starts, stops, support, scores, limit):
    """Bisect wide intervals, preserving the previous representative in its child."""
    if (stops - starts).max() <= limit:
        return starts, stops, support
    p = len(scores)
    indices = np.arange(p)
    original_start = np.repeat(starts, stops - starts)
    starts = np.flatnonzero((indices - original_start) % limit == 0)
    stops = np.r_[starts[1:], p]
    block = np.repeat(np.arange(len(starts)), stops - starts)
    distance = np.abs(2 * indices - (starts + stops - 1)[block])
    other = np.ones(p, dtype=bool)
    other[support] = False
    # Every parent representative has the top score within its own child.
    # Favor it in a tie before applying the child's usual center tie rule.
    order = np.lexsort((indices, distance, other, -scores, block))
    return starts, stops, order[starts]


def _calibration_rows(y, task, groups, ordered, seed, votes):
    used = np.zeros(len(y), dtype=bool)
    used[votes[0]["fit_rows"]] = used[votes[0]["held_rows"]] = True
    remaining = np.flatnonzero(~used)
    if len(remaining) < 8:
        return []
    subset_groups = None if groups is None else np.asarray(groups)[remaining]
    if subset_groups is not None:
        if len(np.unique(subset_groups)) < 2 or (ordered and subset_groups.max() != np.max(groups)):
            return []
    rng = np.random.default_rng([seed, 91672])
    left, right = _moment_halves(y[remaining], task, subset_groups, ordered, rng)
    left, right = remaining[_draw(left, 48, rng)], remaining[_draw(right, 48, rng)]
    if min(len(left), len(right)) < 4:
        return []
    folds = [dict(fit_rows=left, held_rows=right)]
    if not ordered:
        folds.append(dict(fit_rows=right, held_rows=left))
    return folds


def _refit_loss(x, z, centered_target, held_target, alpha):
    n, p = x.shape
    if p < n:
        gram = x.T @ x
        gram.flat[::p + 1] += alpha
        pred = z @ solve(gram, x.T @ centered_target, assume_a="pos", check_finite=False)
    else:
        gram = x @ x.T
        gram.flat[::n + 1] += alpha
        dual = solve(gram, centered_target, assume_a="pos", check_finite=False)
        pred = z @ (x.T @ dual)
    residual = pred - held_target
    return float(np.vdot(residual, residual))


def _validation_data(X, y, task, folds):
    classes = np.unique(np.r_[y[folds[0]["fit_rows"]], y[folds[0]["held_rows"]]]) if task == "classification" else None
    checks = []
    for fold in folds:
        fit, held = fold["fit_rows"], fold["held_rows"]
        x, z = _standardize(np.asarray(X[fit], dtype=float), np.asarray(X[held], dtype=float))
        if classes is None:
            target, query = y[fit, None].astype(float), y[held, None].astype(float)
        else:
            target, query = (y[fit, None] == classes).astype(float), (y[held, None] == classes).astype(float)
        mean = target.mean(axis=0)
        target -= mean
        held_target = query - mean
        alpha = len(fit) / len(y)
        full = _refit_loss(x, z, target, held_target, alpha)
        variance = query - query.mean(axis=0)
        checks.append((x, z, target, held_target, alpha, full, .01 * float(np.vdot(variance, variance))))
    return checks


def calibrated_band_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                      groups=None, ordered_groups=False, backend="auto"):
    """Validate real ridge refits at progressively finer voted spectral widths.

    Calibration uses training rows disjoint from vote construction in every
    direction. Standardize each feature on the calibration fit half. The
    unit full-data ridge penalty is scaled by n_fit/n_outer to preserve the
    penalty per observation. Use the coarsest candidate for which held loss
    is within 1% of held target variation of the full model in each direction.
    The bound describes this internal sketch, not deployment score.
    """
    X, y = np.asarray(X), np.asarray(y)
    coarse, info = band_vote_top_from_training(X, y, count, task=task, random_state=random_state,
                                               groups=groups, ordered_groups=ordered_groups, backend=backend)
    p = X.shape[1]
    info.update(calibration_folds=[], checked_widths=[], excess_loss=[], allowed_excess=[])
    if count is not None or len(coarse) == p:
        return coarse, info
    folds = _calibration_rows(y, task, groups, ordered_groups, random_state, info["folds"])
    info["calibration_folds"] = folds
    if folds:
        checks = _validation_data(X, y, task, folds)
        info["allowed_excess"] = [check[-1] for check in checks]
        starts, stops, support = info["band_starts"], info["band_stops"], coarse
        scores = info["scores"]
        scale = np.max(np.abs(scores))
        ordering_scores = np.round(scores / scale, decimals=12) if scale else scores
        for limit in (8, 4, 2):
            starts, stops, support = _refine(starts, stops, support, ordering_scores, limit)
            if len(support) in info["checked_widths"] or len(support) == p:
                continue
            excess = [_refit_loss(x[:, support], z[:, support], target, query, alpha) - full
                      for x, z, target, query, alpha, full, _ in checks]
            info["checked_widths"].append(len(support))
            info["excess_loss"].append(excess)
            if all(loss <= limit_loss for loss, limit_loss in zip(excess, info["allowed_excess"])):
                info.update(budget=len(support), band_starts=starts, band_stops=stops)
                return support, info
    cols = np.arange(p)
    info.update(budget=p, band_starts=cols, band_stops=cols + 1)
    return cols, info
