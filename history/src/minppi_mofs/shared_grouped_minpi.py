"""Exploratory grouped minPI with one ranking and shared prefix calibration.

This separate prototype supports finite dense classification inputs without
deployment groups. Calibration is an empirical pilot check, not a guarantee.
The selected original columns are refitted using the outer learner by callers.
"""
from math import ceil
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from minppi_mofs.certified_minpi import _scores, _weights
from minppi_mofs.exact_minpi import _selection_tiers, _witness_margins_from_folds
from minppi_mofs.grouped_minpi import _representatives


def _pilot_rows(y, seed, *, fit_limit=512, validation_limit=256):
    """Disjoint stratified pilot rows; all unselected training rows stay unused."""
    n = len(y)
    n_fit = min(fit_limit, 2 * n // 3)
    n_validation = min(validation_limit, n - n_fit)
    fit, held = train_test_split(np.arange(n), train_size=n_fit, test_size=n_validation,
                                stratify=y, random_state=seed)
    labels, counts = np.unique(y[fit], return_counts=True)
    if (counts.min() < 2 or len(labels) != len(np.unique(y))
            or not np.array_equal(np.unique(y[held]), labels)):
        raise ValueError("Insufficient class rows in disjoint pilots")
    return np.sort(fit), np.sort(held)


def _candidate_widths(p, representatives):
    """Fixed eighths of ORIGINAL width, plus all representatives if reduced."""
    widths = {ceil(p * numerator / 8) for numerator in range(1, 8)}
    widths = {width for width in widths if width <= representatives and width < p}
    if representatives < p:
        widths.add(representatives)
    return sorted(widths)


def _grouped_ranking(X, y, seed, *, voting=True):
    """Build the budget-independent half-width pool, then rank its entire game."""
    p = X.shape[1]
    representatives, _, pool_info, folds = _representatives(
        X, y, "classification", seed, ceil(p / 2), None)
    relevance_order = representatives.copy()
    representatives = np.sort(representatives)
    info = dict(pool_info, representatives=representatives,
                relevance_order=relevance_order, voting=bool(voting))
    if not voting or folds is None:
        info.update(votes=np.empty((0, len(representatives)), dtype=np.int8),
                    pivotal_scores=np.empty(0), tiers=[], witness_fit_rows=np.empty(0, dtype=int),
                    witness_query_rows=np.empty(0, dtype=int))
        return relevance_order, info
    margins = _witness_margins_from_folds(X, y, "classification", *folds,
                                         columns=representatives)
    votes = np.sign(margins).astype(np.int8)
    r = len(representatives)
    tiers = _selection_tiers(None, r)
    pos, neg = (np.count_nonzero(votes == value, axis=1) for value in (1, -1))
    weights = _weights(pos, neg, r - pos - neg, r, tiers)
    score = np.full(r, np.inf)
    for tier in tiers:
        values = _scores(votes, weights[tier]) if tier else votes.mean(axis=0, dtype=np.float64)
        np.minimum(score, values, out=score)
    order = representatives[np.argsort(-score, kind="stable")]
    info.update(votes=votes, pivotal_scores=score, tiers=tiers,
                witness_fit_rows=folds[0], witness_query_rows=folds[1])
    return order, info


def _prefix_ridge(X, y, Xt, order, widths):
    """Unit ridge using shared scaling and shape-appropriate sufficient statistics.

    These are the solutions of StandardScaler + RidgeClassifier(alpha=1,
    fit_intercept=True), including -1/+1 one-vs-rest targets and class ties.
    The pilot mean and variance of each column do not depend on the prefix.
    """
    order, widths = np.asarray(order, dtype=int), list(widths)
    if (not widths or widths != sorted(set(widths)) or widths[0] < 1
            or widths[-1] > len(order) or len(np.unique(order)) != len(order)
            or order.min() < 0 or order.max() >= X.shape[1]):
        raise ValueError("Expected increasing valid widths and unique original columns")
    scaler = StandardScaler().fit(X)
    fit, held = scaler.transform(X), scaler.transform(Xt)
    # Ridge centers the transformed predictors again for its intercept.
    offset = fit.mean(axis=0)
    fit -= offset
    held -= offset
    encoder = LabelBinarizer(neg_label=-1, pos_label=1)
    target = encoder.fit_transform(y).astype(np.float64)
    intercept = target.mean(axis=0)
    target -= intercept
    primal = len(order) <= len(fit)
    if primal:
        ordered_fit = fit[:, order]
        gram = ordered_fit.T @ ordered_fit
        gram.flat[::len(order)+1] += 1.
        rhs = ordered_fit.T @ target
    else:
        gram = np.eye(len(fit), dtype=np.float64)
        cross = np.zeros((len(held), len(fit)), dtype=np.float64)
    previous = 0
    decisions, predictions = [], []
    for width in widths:
        if primal:
            coefficient = cho_solve(cho_factor(gram[:width, :width], lower=True, check_finite=False),
                                    rhs[:width], check_finite=False)
            decision = held[:, order[:width]] @ coefficient + intercept
        else:
            columns = order[previous:width]
            A, B = fit[:, columns], held[:, columns]
            gram += A @ A.T
            cross += B @ A.T
            dual = cho_solve(cho_factor(gram, lower=True, check_finite=False),
                             target, check_finite=False)
            decision = cross @ dual + intercept
        if len(encoder.classes_) == 2:
            decision = decision.ravel()
            pred = encoder.classes_[(decision > 0).astype(int)]
        else:
            pred = encoder.classes_[decision.argmax(axis=1)]
        decisions.append(decision)
        predictions.append(pred)
        previous = width
    return decisions, predictions


def shared_grouped_top_from_training(X, y, *, random_state=1129,
                                     ranking="pivotal", tolerance=.02,
                                     groups=None, ordered_groups=False):
    """Choose the shortest pilot-acceptable grouped prefix, or full on failure.

    ``ranking='representatives'`` disables voting with the identical pool.
    ``ranking='F'`` independently calibrates a pilot-fit ANOVA ranking over all
    columns with the same disjoint rows, eighth-width grid and unit-ridge model.
    No calibration rows enter a ranking. Groups require a future separate
    split protocol and are explicitly rejected instead of silently mixed.
    """
    if groups is not None or ordered_groups:
        raise NotImplementedError("This prototype does not yet support grouped or ordered pilots")
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    if (X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),)
            or not np.isfinite(X).all() or pd.isna(y).any()):
        raise ValueError("Expected a nonempty finite dense matrix and complete aligned labels")
    if ranking not in ("pivotal", "representatives", "F"):
        raise ValueError("ranking must be pivotal, representatives or F")
    if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("tolerance must lie in [0,1)")
    p = X.shape[1]
    labels, counts = np.unique(y, return_counts=True)
    fallback = dict(reason="insufficient_class_rows", reduced=False, ranking_kind=ranking,
                    tolerance=float(tolerance))
    if len(labels) < 2 or counts.min() < 3:
        return np.arange(p), fallback
    try:
        fit, held = _pilot_rows(y, random_state)
    except ValueError as error:
        return np.arange(p), dict(fallback, detail=str(error))
    if ranking == "F":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            scores = f_classif(X[fit], y[fit])[0]
        order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
        info = dict(representatives=np.arange(p), F_scores=scores, voting=False)
    else:
        order, info = _grouped_ranking(X[fit], y[fit], random_state,
                                       voting=ranking == "pivotal")
    widths = _candidate_widths(p, len(order))
    # Appending the remaining columns gives the full pilot without rebuilding
    # its Gram matrix. Those columns never enter the candidate ranking.
    complete_order = np.r_[order, np.setdiff1d(np.arange(p), order, assume_unique=True)]
    evaluated = widths + [p]
    decisions, predictions = _prefix_ridge(X[fit], y[fit], X[held], complete_order, evaluated)
    scores = np.asarray([balanced_accuracy_score(y[held], pred) for pred in predictions])
    acceptable = np.flatnonzero(scores[:-1] >= scores[-1] - tolerance - 1e-12)
    count = widths[acceptable[0]] if len(acceptable) else p
    support = np.sort(order[:count]) if count < p else np.arange(p)
    info.update(reason="pilot_acceptable" if count < p else "no_acceptable_prefix",
                reduced=bool(count < p), ranking_kind=ranking, tolerance=float(tolerance),
                pilot_fit_rows=fit, calibration_rows=held, ranking=order,
                candidate_widths=np.asarray(widths), evaluated_widths=np.asarray(evaluated),
                calibration_scores=scores, calibration_predictions=np.asarray(predictions),
                calibration_decisions=np.asarray(decisions), full_pilot_score=float(scores[-1]))
    return support, info
