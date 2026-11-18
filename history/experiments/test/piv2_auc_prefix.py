"""Select the best nonempty ranking prefix by full validation AUC on Madelon."""

from time import monotonic

import numpy as np

from minppi_mofs.modeling import fit_predict
from minppi_mofs.utility import compute_utility


TIE_TOLERANCE = 1e-14


def select(backend, train, yt, val, yv, policy, metric, progress, *, original_width, original_columns):
    """Return the winning columns, their validation predictions, fit count and full curve."""
    from experiments.test.piv2_prefix_rankings import rank_features

    width = train.shape[1]
    if (not width or not train.columns.is_unique or not train.columns.equals(val.columns)
            or original_width != len(original_columns)
            or list(train.columns) != [c for c in original_columns if c in train.columns]):
        raise ValueError("Expected aligned nonempty subsets in original column order")
    if metric != "roc_auc":
        raise ValueError("This Madelon benchmark selects k by binary validation ROC-AUC")
    progress("ranking")
    start = monotonic()
    order, ranking_fits, full_prediction, ranking = rank_features(
        backend, train, yt, val, yv, policy, metric, progress, original_columns=original_columns,
    )
    ranking_s = monotonic() - start
    order = np.asarray(order)
    if (order.shape != (width,) or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order), np.arange(width))):
        raise ValueError("The ranking must be a permutation of the candidate indices")
    if ranking_fits < 0 or (full_prediction is not None and ranking_fits < 1):
        raise ValueError("Inconsistent ranking fit count or full-model prediction")

    checked = 0
    best_score = -np.inf
    best_k = None
    winner_prediction = None
    curve, seconds_by_k = [], []
    keep = np.zeros(width, dtype=bool)
    scan_start = monotonic()
    progress("validation_prefixes", checked=0, candidates=width, fits=ranking_fits)
    for k, feature_index in enumerate(order, start=1):
        point_start = monotonic()
        keep[feature_index] = True
        if k == width and full_prediction is not None:
            prediction = full_prediction
        else:
            prediction = fit_predict(train.iloc[:, keep], yt, val.iloc[:, keep], **policy)
            checked += 1
        score = float(compute_utility(yv, prediction, metric=metric))
        if not np.isfinite(score):
            raise ValueError(f"Nonfinite validation AUC for {backend} k={k}")
        curve.append(score)
        seconds_by_k.append(monotonic() - point_start)
        if score > best_score + TIE_TOLERANCE:
            best_score, best_k, winner_prediction = score, k, prediction
        progress("validation_prefixes", checked=k, candidates=width, fits=ranking_fits+checked,
                 best_k=best_k, best_validation_auc=best_score,
                 eta_s=(monotonic()-scan_start)*(width-k)/k)

    selected = train.columns[np.sort(order[:best_k])].tolist()
    diagnostics = dict(order=train.columns[order].tolist(), curve=curve, seconds_by_k=seconds_by_k,
                       selected_k=best_k, ranking=ranking, ranking_fits=ranking_fits, prefix_fits=checked,
                       reused_full_validation=full_prediction is not None, ranking_s=ranking_s,
                       scan_s=monotonic()-scan_start)
    return selected, winner_prediction, ranking_fits+checked, diagnostics
