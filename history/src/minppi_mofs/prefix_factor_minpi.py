"""One primal Cholesky factor for all fixed-ranking pilot prefixes."""
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, solve_triangular
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from minppi_mofs.bestcal_grouped_minpi import best_calibrated_support
from minppi_mofs.scaled_pilot_grouped_minpi import _extended_fit_rows
from minppi_mofs.shared_grouped_minpi import (
    _candidate_widths, _grouped_ranking, _pilot_rows,
    _prefix_ridge as _reference_prefix_ridge,
)


def _factorized_prefix_ridge(X, y, Xt, order, widths):
    """Return exact-model prefix decisions with one factor when p<=n.

    If G=Z.T@Z+I=L@L.T, its leading k-by-k principal block is
    L[:k,:k]@L[:k,:k]. Positive diagonal uniqueness makes that block the
    Cholesky factor of the prefix ridge Gram. Shared column scaling and
    response centering therefore permit two triangular solves per prefix.
    Floating-point decisions need numerical comparison, not bitwise claims.
    For more ranked columns than fit rows, retain the original dual routine;
    this branch still performs a factorization for each candidate width.
    """
    order, widths = np.asarray(order, dtype=int), list(widths)
    if (not widths or widths != sorted(set(widths)) or widths[0] < 1
            or widths[-1] > len(order) or len(np.unique(order)) != len(order)
            or order.min() < 0 or order.max() >= X.shape[1]):
        raise ValueError("Expected increasing valid widths and unique original columns")
    if len(order) > len(X):
        return _reference_prefix_ridge(X, y, Xt, order, widths)
    scaler = StandardScaler().fit(X)
    fit, held = scaler.transform(X), scaler.transform(Xt)
    offset = fit.mean(axis=0)
    fit -= offset
    held -= offset
    encoder = LabelBinarizer(neg_label=-1, pos_label=1)
    target = encoder.fit_transform(y).astype(np.float64)
    intercept = target.mean(axis=0)
    target -= intercept
    ordered_fit = fit[:, order]
    gram = ordered_fit.T @ ordered_fit
    gram.flat[::len(order) + 1] += 1.
    rhs = ordered_fit.T @ target
    lower = cho_factor(gram, lower=True, check_finite=False)[0]
    decisions, predictions = [], []
    for width in widths:
        factor = lower[:width, :width]
        intermediate = solve_triangular(factor, rhs[:width], lower=True, check_finite=False)
        coefficient = solve_triangular(factor.T, intermediate, lower=False, check_finite=False)
        decision = held[:, order[:width]] @ coefficient + intercept
        if len(encoder.classes_) == 2:
            decision = decision.ravel()
            prediction = encoder.classes_[(decision > 0).astype(int)]
        else:
            prediction = encoder.classes_[decision.argmax(axis=1)]
        decisions.append(decision)
        predictions.append(prediction)
    return decisions, predictions


def factorized_grouped_top_from_training(
    X, y, *, random_state=1129, ranking="pivotal", tolerance=.02,
    groups=None, ordered_groups=False, budget_rule="bestcal",
):
    """Preserve scaled-pilot sampling/ranking and reuse one primal factor.

    All three rankings (pivotal, representatives, F) pay the same calibration
    procedure. Default bestcal chooses the best calibration BA among genuinely
    reduced prefixes, exact ties preferring fewer columns, provided it is no
    more than tolerance below the full pilot. first_acceptable preserves the
    previous shortest-acceptable rule for comparisons. Full fallback is failure.
    Neither rule guarantees the score of the downstream refitted classifier.
    """
    if budget_rule not in ("bestcal", "first_acceptable"):
        raise ValueError("budget_rule must be bestcal or first_acceptable")
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
                    tolerance=float(tolerance), budget_rule=budget_rule)
    if len(labels) < 2 or counts.min() < 3:
        return np.arange(p), fallback
    try:
        rank_rows, held = _pilot_rows(y, random_state)
    except ValueError as error:
        return np.arange(p), dict(fallback, detail=str(error))
    fit = _extended_fit_rows(y, rank_rows, held, max(512, 2*p), random_state)
    if ranking == "F":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            values = f_classif(X[rank_rows], y[rank_rows])[0]
        order = np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="stable")
        info = dict(representatives=np.arange(p), F_scores=values, voting=False)
    else:
        order, info = _grouped_ranking(X[rank_rows], y[rank_rows], random_state,
                                       voting=ranking == "pivotal")
    widths = _candidate_widths(p, len(order))
    complete_order = np.r_[order, np.setdiff1d(np.arange(p), order, assume_unique=True)]
    evaluated = widths + [p]
    decisions, predictions = _factorized_prefix_ridge(X[fit], y[fit], X[held], complete_order, evaluated)
    scores = np.asarray([balanced_accuracy_score(y[held], prediction) for prediction in predictions])
    acceptable = np.flatnonzero(scores[:-1] >= scores[-1] - tolerance - 1e-12)
    count = widths[acceptable[0]] if len(acceptable) else p
    support = np.sort(order[:count]) if count < p else np.arange(p)
    info.update(reason="pilot_acceptable" if count < p else "no_acceptable_prefix",
                reduced=bool(count < p), ranking_kind=ranking, tolerance=float(tolerance),
                ranking_fit_rows=rank_rows, pilot_fit_rows=fit, calibration_rows=held,
                fit_target=max(512, 2*p), ranking=order,
                candidate_widths=np.asarray(widths), evaluated_widths=np.asarray(evaluated),
                calibration_scores=scores, calibration_predictions=np.asarray(predictions),
                calibration_decisions=np.asarray(decisions), full_pilot_score=float(scores[-1]),
                budget_rule=budget_rule,
                calibration_solver="single_primal_factor" if p <= len(fit) else "reference_dual")
    return best_calibrated_support(p, info) if budget_rule == "bestcal" else (support, info)
