"""Training-only calibration of one environment-aware minPI path and its width."""
from math import ceil
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from minppi_mofs.environment_minpi import InsufficientEnvironmentsError, environment_path_from_training
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.restoration_minpi import _sample_rows
from minppi_mofs.utility_minpi import _utility_inputs


def _partition(y, groups, ordered_groups, task, random_state):
    """Reserve up to three disjoint validation environments entirely outside discovery."""
    rng = np.random.default_rng(random_state)
    ids = np.arange(len(y))
    if groups is not None:
        labels, group_ids = np.unique(groups, return_inverse=True)
        if len(labels) < 3:
            raise InsufficientEnvironmentsError("Need two discovery groups and a validation group")
        if ordered_groups:
            held_count = min(3, max(1, len(labels) // 3))
            held_parts = np.arange(len(labels) - held_count, len(labels))[:, None]
        else:
            held_count = max(1, len(labels) // 4)
            reserved = rng.permutation(len(labels))[:held_count]
            held_parts = np.array_split(reserved, min(3, held_count))
        valid = [ids[np.isin(group_ids, part)] for part in held_parts]
    else:
        labels = y if task == "classification" else np.zeros(len(y), dtype=int)
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if task == "classification" and len(counts) < 2:
            raise InsufficientEnvironmentsError("Need at least two classes for calibration")
        # Keep at least two fitting examples per class. Each validation block
        # of iid classification data includes an example of every class.
        held_counts = np.minimum(counts - 2, np.maximum(1, counts // 4))
        if np.min(held_counts) < 1:
            raise InsufficientEnvironmentsError("Insufficient rows to separate discovery and validation")
        blocks = min(3, np.min(held_counts) if task == "classification" else held_counts[0] // 2)
        if blocks < 1:
            raise InsufficientEnvironmentsError("Need at least two rows for validation")
        parts = [[] for _ in range(blocks)]
        for label_idx, held_count in enumerate(held_counts):
            held = rng.permutation(ids[inverse == label_idx])[:held_count]
            for block, indices in zip(parts, np.array_split(held, blocks)):
                block.append(indices)
        valid = [np.sort(np.concatenate(block)) for block in parts]
    discovery = np.setdiff1d(ids, np.concatenate(valid))
    if len(discovery) < 4 or min(map(len, valid)) < 2:
        raise InsufficientEnvironmentsError("Insufficient discovery or validation rows")
    if task == "classification" and len(np.unique(y[discovery])) != len(np.unique(y)):
        raise InsufficientEnvironmentsError("A validation class is absent from discovery")
    return discovery, valid


def _ridge_scores(fit, check, train_target, intercept, target, parts, task):
    """Evaluate a unit-penalty ridge with an unpenalized intercept on each environment."""
    if fit.shape[1] <= len(fit):
        gram = fit.T @ fit
        gram.flat[::len(gram) + 1] += 1
        coef = solve(gram, fit.T @ train_target, assume_a="pos", check_finite=False)
    else:
        gram = fit @ fit.T
        gram.flat[::len(gram) + 1] += 1
        coef = fit.T @ solve(gram, train_target, assume_a="pos", check_finite=False)
    out = check @ coef + intercept
    prediction = out.argmax(axis=1) if task == "classification" else out[:, 0]
    return _environment_quality(target, prediction, parts, task)


def _environment_quality(target, prediction, parts, task):
    scores = []
    for part in parts:
        gold, pred = target[part], prediction[part]
        if task == "classification":
            _, code, counts = np.unique(gold, return_inverse=True, return_counts=True)
            score = np.mean(np.bincount(code, weights=(gold == pred), minlength=len(counts)) / counts)
        else:
            spread = np.sum((gold - gold.mean()) ** 2)
            if spread == 0:
                raise InsufficientEnvironmentsError("A validation environment has constant regression targets")
            score = 1 - np.sum((gold - pred) ** 2) / spread
        scores.append(float(score))
    return np.array(scores)


def _impute_validation(train, query):
    """Impute from the fitting rows, preserving their floating-point precision."""
    if train.dtype.kind not in "fiu":
        train, query = train.astype(float), query.astype(float)
    if np.isfinite(train).all() and np.isfinite(query).all():
        return train, query
    valid = np.isfinite(train)
    mean = np.where(valid, train, 0).sum(axis=0, dtype=np.float64) / np.maximum(valid.sum(axis=0), 1)
    mean = mean.astype(train.dtype)
    return np.where(valid, train, mean), np.where(np.isfinite(query), query, mean)


def _deployment_scores(train, query, fit_y, held_y, parts, task):
    scaler = StandardScaler()
    fit = scaler.fit_transform(train)
    check = scaler.transform(query)
    estimator = RidgeClassifier if task == "classification" else Ridge
    model = estimator(alpha=1., solver="lsqr", tol=.001, max_iter=250)
    predictions = model.fit(fit, fit_y).predict(check)
    if task == "regression":
        if any(np.ptp(held_y[part]) == 0 for part in parts):
            raise InsufficientEnvironmentsError("A validation environment has constant regression targets")
        return np.array([r2_score(held_y[part], predictions[part]) for part in parts])
    return _environment_quality(held_y, predictions, parts, task)


def _calibrate_supports(X, y, discovery, validation, supports, task, random_state, tolerance,
                        rule="environment", bounded_calibration=False):
    """Choose the smallest actual discovery support passing every held-environment comparison."""
    rng = np.random.default_rng(random_state)
    labels = y if task == "classification" else None
    fit_rows, held = discovery, validation
    if bounded_calibration:
        fit_rows = _sample_rows(discovery, 512, labels, rng)
        held = [_sample_rows(part, 256, labels, rng) for part in validation]
    query = np.concatenate(held)
    cuts = np.cumsum([0, *(len(rows) for rows in held)])
    parts = [slice(start, end) for start, end in zip(cuts[:-1], cuts[1:])]
    fit_y, query_y = y[fit_rows], y[query]
    if task == "classification" and not np.isin(query_y, np.unique(fit_y)).all():
        raise InsufficientEnvironmentsError("A validation class is absent from the calibration fit")
    if bounded_calibration:
        if task == "classification":
            classes = np.unique(fit_y)
            fit_target = (fit_y[:, None] == classes).astype(float)
            target = np.searchsorted(classes, query_y)
        else:
            fit_target = fit_y[:, None].astype(float)
            target = query_y.astype(float)
        intercept = fit_target.mean(axis=0)
        fit_target -= intercept
        fit, check = _standardize(np.asarray(X[fit_rows], dtype=float), np.asarray(X[query], dtype=float))
        def evaluate(columns):
            return _ridge_scores(fit[:, columns], check[:, columns], fit_target, intercept, target, parts, task)
    else:
        fit, check = _impute_validation(X[fit_rows], X[query])
        def evaluate(columns):
            return _deployment_scores(fit[:, columns], check[:, columns], fit_y, query_y, parts, task)
    full_scores = evaluate(slice(None))
    chosen = np.arange(X.shape[1])
    candidates = []
    for width, support in sorted(supports.items()):
        if width == X.shape[1]:
            continue
        scores = evaluate(support)
        gaps = scores - full_scores
        eligible = bool(np.all(gaps >= -tolerance))
        candidates.append(dict(rule=rule, width=int(width), scores=scores.tolist(),
                               gaps=gaps.tolist(), eligible=eligible))
        if eligible and len(chosen) == X.shape[1]:
            chosen = support
    fallback = len(chosen) == X.shape[1]
    info = dict(rule="full" if fallback else rule, width=len(chosen),
                reason="no_eligible_support" if fallback else "within_tolerance_on_every_environment",
                calibration_mode="bounded" if bounded_calibration else "deployment",
                validation_candidates=candidates, full_scores=full_scores.tolist(),
                calibration_fit_rows=fit_rows.tolist(), calibration_validation_rows=[h.tolist() for h in held])
    return chosen, info


def stable_top_from_training(X, y, *, task="classification", random_state=42, tolerance=.02,
                             fractions=(.10, .50, .75), groups=None, ordered_groups=False, bounded_calibration=False):
    """Fit a minPI path on discovery rows, validate its fixed supports and return a width.

    Pass ordered training-group IDs for prospective deployment. The largest
    IDs are reserved for calibration, before calling the ranking routine.
    Unordered groups are kept intact. Iid classification splits are stratified.
    Selection minimizes retained width subject to each observed validation
    gap against full ridge being at least -tolerance. Default calibration
    uses full discovery and validation rows with the deployment ridge's LSQR
    solver and input precision. Bounded calibration uses a capped sketch and
    a direct double-precision ridge solve. The tolerance is an
    empirical calibration criterion. It gives no bound on future populations.
    """
    X, y = _utility_inputs(X, y, 1, task)
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be a nonnegative integer below 2**32")
    if not isinstance(bounded_calibration, (bool, np.bool_)):
        raise ValueError("bounded_calibration must be boolean")
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative finite number")
    if not isinstance(fractions, (list, tuple, np.ndarray)) or len(fractions) < 1:
        raise ValueError("fractions must be a nonempty sequence")
    if any(isinstance(f, (bool, np.bool_)) or not isinstance(f, Real) or not 0 < f <= 1 for f in fractions):
        raise ValueError("Each fraction must be in (0, 1]")
    if ordered_groups and groups is None:
        raise ValueError("ordered_groups requires training groups")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape or pd.isna(groups).any():
            raise ValueError("Expected complete training groups aligned with y")
    p = X.shape[1]
    widths = sorted({ceil(f * p) for f in fractions} - {p})
    info = dict(split="ordered" if ordered_groups else ("iid" if groups is None else "group"),
                tolerance=float(tolerance), discovery_rows=[], validation_rows=[], full_scores=[],
                calibration_mode="bounded" if bounded_calibration else "deployment",
                path_row_index_basis="relative_to_discovery_rows",
                validation_candidates=[], rule="full", width=p, reason="no_compressed_width")
    if not widths:
        return np.arange(p), info
    try:
        discovery, valid = _partition(y, groups, ordered_groups, task, random_state)
        info.update(discovery_rows=discovery.tolist(), validation_rows=[v.tolist() for v in valid])
        supports, diagnostics = environment_path_from_training(
            X[discovery], y[discovery], widths, task=task, random_state=random_state,
            groups=None if groups is None else groups[discovery], ordered_groups=ordered_groups)
        info["environment_path"] = dict(diagnostics, folds=[
            {key: values.tolist() for key, values in fold.items()} for fold in diagnostics["folds"]])
        chosen, calibrated = _calibrate_supports(
            X, y, discovery, valid, supports, task, random_state, tolerance, bounded_calibration=bounded_calibration)
    except InsufficientEnvironmentsError as exc:
        info.update(reason="insufficient_environments", detail=str(exc))
        return np.arange(p), info
    info.update(calibrated)
    return chosen, info


class StableMinPISelector(SelectorMixin, BaseEstimator):
    """Training-only, deterministic selector of an environment-aware minPI support."""

    def __init__(self, *, task="classification", random_state=42, tolerance=.02,
                 fractions=(.10, .50, .75), bounded_calibration=False):
        self.task = task
        self.random_state = random_state
        self.tolerance = tolerance
        self.fractions = fractions
        self.bounded_calibration = bounded_calibration

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y, *, groups=None, ordered_groups=False):
        if hasattr(X, "columns") and not X.columns.is_unique:
            raise ValueError("Feature names must be unique")
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        selected, self.diagnostics_ = stable_top_from_training(
            X, y, task=self.task, random_state=self.random_state, tolerance=self.tolerance,
            fractions=self.fractions, groups=groups, ordered_groups=ordered_groups,
            bounded_calibration=self.bounded_calibration)
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[selected] = True
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_
