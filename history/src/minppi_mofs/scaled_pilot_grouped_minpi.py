"""Dimension-scaled calibration, with the previous 512-row ranking unchanged.

The fixed factor two only keeps a sufficiently large available pilot above
the original feature dimension. It is not an accuracy guarantee or a tuned
per-dataset factor. This module preserves the previous prototype verbatim.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from minppi_mofs.shared_grouped_minpi import (
    _candidate_widths, _grouped_ranking, _pilot_rows, _prefix_ridge,
)


def _extended_fit_rows(y, ranking_rows, calibration_rows, target, seed):
    occupied = np.r_[ranking_rows, calibration_rows]
    available = np.setdiff1d(np.arange(len(y)), occupied, assume_unique=True)
    additional = min(len(available), max(0, int(target) - len(ranking_rows)))
    if additional == len(available):
        extra = available
    elif additional == 0:
        extra = np.empty(0, dtype=int)
    else:
        labels, counts = np.unique(y[available], return_counts=True)
        split_seed = int(np.random.default_rng([seed, 71023]).integers(0, 2**31-1))
        if (counts.min() >= 2 and additional >= len(labels)
                and len(available) - additional >= len(labels)):
            extra, _ = train_test_split(available, train_size=additional,
                                        stratify=y[available], random_state=split_seed)
        else:
            extra = np.random.default_rng(split_seed).choice(available, additional, replace=False)
    return np.sort(np.r_[ranking_rows, extra])


def scaled_pilot_grouped_top_from_training(X, y, *, random_state=1129,
                                          ranking="pivotal", tolerance=.02,
                                          groups=None, ordered_groups=False):
    """Calibrate on at least max(512,2p) fit rows when available, excluding held rows.

    The previous ranking rows and calibration rows are kept exactly unchanged.
    Added rows affect the pilot predictor only, including for the F/no-vote
    controls. Grouped/time-ordered deployments are not implemented here.
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
    decisions, predictions = _prefix_ridge(X[fit], y[fit], X[held], complete_order, evaluated)
    scores = np.asarray([balanced_accuracy_score(y[held], pred) for pred in predictions])
    acceptable = np.flatnonzero(scores[:-1] >= scores[-1] - tolerance - 1e-12)
    count = widths[acceptable[0]] if len(acceptable) else p
    support = np.sort(order[:count]) if count < p else np.arange(p)
    info.update(reason="pilot_acceptable" if count < p else "no_acceptable_prefix",
                reduced=bool(count < p), ranking_kind=ranking, tolerance=float(tolerance),
                ranking_fit_rows=rank_rows, pilot_fit_rows=fit, calibration_rows=held,
                fit_target=max(512, 2*p), ranking=order,
                candidate_widths=np.asarray(widths), evaluated_widths=np.asarray(evaluated),
                calibration_scores=scores, calibration_predictions=np.asarray(predictions),
                calibration_decisions=np.asarray(decisions), full_pilot_score=float(scores[-1]))
    return support, info
