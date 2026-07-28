"""Canonical XGBoost prediction boundary for experiment evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from minppi_mofs.utility import compute_utility


_TASKS = frozenset({"binary", "multiclass", "regression"})
_BACKENDS = frozenset({"cpu", "cuda"})
_POLICY_PARAMETERS = frozenset({"device", "tree_method", "random_state", "seed"})
_MODEL_PARAMETERS = frozenset(
    {"n_estimators", "max_depth", "learning_rate", "n_jobs"}
)


def _validated_policy(
    *, task: str, model_params: Mapping[str, object], backend: str, seed: int
) -> dict[str, object]:
    if task not in _TASKS:
        raise ValueError(f"task must be one of {sorted(_TASKS)}")
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(model_params, Mapping):
        raise TypeError("model_params must be a mapping")
    if not all(isinstance(key, str) and key for key in model_params):
        raise ValueError("model_params keys must be non-empty strings")
    forbidden = _POLICY_PARAMETERS.intersection(model_params)
    if forbidden:
        raise ValueError(
            "model_params cannot override canonical policy keys: "
            + ", ".join(sorted(forbidden))
        )
    unknown = set(model_params) - _MODEL_PARAMETERS
    if unknown:
        raise ValueError("unknown model_params keys: " + ", ".join(sorted(unknown)))
    for key, value in model_params.items():
        if key in {"n_estimators", "max_depth", "n_jobs"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"model_params[{key!r}] must be a positive integer")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"model_params[{key!r}] must be a positive finite number")
    return {
        **dict(model_params),
        "tree_method": "hist",
        "device": "cuda" if backend == "cuda" else "cpu",
        "random_state": seed,
    }


def _validated_frame(value: object, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if value.columns.duplicated().any():
        raise ValueError(f"{name} columns must be unique")
    try:
        numeric = value.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite values")
    return numeric


def _validated_target(value: object, *, row_count: int, task: str) -> np.ndarray:
    target = np.asarray(value)
    if target.ndim != 1 or len(target) != row_count or not np.isfinite(target).all():
        raise ValueError("target must be a finite one-dimensional training vector")
    if task == "regression":
        return target.astype(float)
    if not np.equal(target, np.floor(target)).all():
        raise ValueError("classification labels must be integers")
    labels = target.astype(int)
    classes = np.unique(labels)
    expected = np.arange(len(classes))
    if len(classes) < 2 or not np.array_equal(classes, expected):
        raise ValueError("classification labels must be zero-based and contiguous")
    if task == "binary" and len(classes) != 2:
        raise ValueError("binary labels must contain exactly two classes")
    return labels


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    *,
    task: str,
    model_params: Mapping[str, object],
    backend: str,
    seed: int,
) -> np.ndarray:
    """Fit the canonical task model once and return raw predictions."""
    model = make_model(
        task=task, model_params=model_params, backend=backend, seed=seed
    )
    train = _validated_frame(X_train, name="X_train")
    evaluation = _validated_frame(X_eval, name="X_eval")
    if not tuple(train.columns) or tuple(evaluation.columns) != tuple(train.columns):
        raise ValueError("non-empty train/evaluation feature columns must match")
    target = _validated_target(y_train, row_count=len(train), task=task)
    safe_columns = [f"f{index}" for index in range(train.shape[1])]
    train = train.set_axis(safe_columns, axis="columns")
    evaluation = evaluation.set_axis(safe_columns, axis="columns")
    model.fit(train, target)
    predictions = (
        model.predict(evaluation)
        if task == "regression"
        else model.predict_proba(evaluation)
    )
    result = np.asarray(predictions, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("model predictions must be finite")
    expected_ndim = 1 if task == "regression" else 2
    if result.ndim != expected_ndim or len(result) != len(evaluation):
        raise ValueError("model predictions have an invalid shape")
    return result


def fit_predict_checkpoints(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    *,
    task: str,
    model_params: Mapping[str, object],
    checkpoints: Sequence[int],
    backend: str,
    seed: int,
) -> dict[int, np.ndarray]:
    """Fit the maximum boosting round once and predict at deterministic prefixes.

    XGBoost builds trees sequentially, so the first ``k`` trees of a model
    trained for ``K >= k`` rounds are the same deterministic model obtained by
    stopping training after ``k`` rounds.  This boundary is used only for
    validation calibration; formal experiments still fit the selected frozen
    round count directly.
    """
    rounds = tuple(checkpoints)
    if (
        not rounds
        or tuple(sorted(rounds)) != rounds
        or len(set(rounds)) != len(rounds)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in rounds
        )
    ):
        raise ValueError("checkpoints must be unique increasing positive integers")
    params = _validated_policy(
        task=task, model_params=model_params, backend=backend, seed=seed
    )
    if model_params.get("n_estimators") != rounds[-1]:
        raise ValueError(
            "model_params n_estimators must equal the largest checkpoint"
        )
    train = _validated_frame(X_train, name="X_train")
    evaluation = _validated_frame(X_eval, name="X_eval")
    if not tuple(train.columns) or tuple(evaluation.columns) != tuple(train.columns):
        raise ValueError("non-empty train/evaluation feature columns must match")
    target = _validated_target(y_train, row_count=len(train), task=task)
    safe_columns = [f"f{index}" for index in range(train.shape[1])]
    train = train.set_axis(safe_columns, axis="columns")
    evaluation = evaluation.set_axis(safe_columns, axis="columns")
    model = XGBRegressor(**params) if task == "regression" else XGBClassifier(**params)
    model.fit(train, target)
    expected_ndim = 1 if task == "regression" else 2
    predictions: dict[int, np.ndarray] = {}
    for checkpoint in rounds:
        raw = (
            model.predict(evaluation, iteration_range=(0, checkpoint))
            if task == "regression"
            else model.predict_proba(
                evaluation, iteration_range=(0, checkpoint)
            )
        )
        result = np.asarray(raw, dtype=float)
        if (
            not np.isfinite(result).all()
            or result.ndim != expected_ndim
            or len(result) != len(evaluation)
        ):
            raise ValueError("model checkpoint predictions have an invalid shape")
        predictions[checkpoint] = result
    return predictions


def make_model(
    *,
    task: str,
    model_params: Mapping[str, object],
    backend: str,
    seed: int,
) -> XGBClassifier | XGBRegressor:
    """Construct, but do not fit, the canonical model for a validated policy."""
    params = _validated_policy(
        task=task, model_params=model_params, backend=backend, seed=seed
    )
    return XGBRegressor(**params) if task == "regression" else XGBClassifier(**params)


def _empty_subset_predictions(
    y_train: np.ndarray, *, task: str, row_count: int
) -> np.ndarray:
    target = _validated_target(y_train, row_count=len(y_train), task=task)
    if task == "regression":
        return np.full(row_count, float(np.mean(target)))
    counts = np.bincount(target, minlength=int(target.max()) + 1).astype(float)
    priors = counts / counts.sum()
    return np.tile(priors, (row_count, 1))


def evaluate_fixed_subset(
    X_train_val: pd.DataFrame,
    y_train_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features: Sequence[str],
    *,
    task: str,
    utility: str,
    model_params: Mapping[str, object],
    backend: str,
    seed: int,
) -> float:
    """Evaluate an already-frozen ordered subset through ``compute_utility``."""
    params = _validated_policy(
        task=task, model_params=model_params, backend=backend, seed=seed
    )
    del params  # validation must happen even when no model is fitted
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise TypeError("features must be an ordered sequence")
    selected = tuple(features)
    if len(set(selected)) != len(selected) or not all(
        isinstance(feature, str) for feature in selected
    ):
        raise ValueError("features must contain unique strings")
    train = _validated_frame(X_train_val, name="X_train_val")
    test = _validated_frame(X_test, name="X_test")
    if not set(selected).issubset(train.columns) or not set(selected).issubset(
        test.columns
    ):
        raise ValueError("features are outside the supplied frames")
    target = _validated_target(y_train_val, row_count=len(train), task=task)
    if selected:
        predictions = fit_predict(
            train.loc[:, list(selected)],
            target,
            test.loc[:, list(selected)],
            task=task,
            model_params=model_params,
            backend=backend,
            seed=seed,
        )
    else:
        predictions = _empty_subset_predictions(
            target, task=task, row_count=len(test)
        )
    value = float(compute_utility(np.asarray(y_test), predictions, metric=utility))
    if not math.isfinite(value):
        raise ValueError("utility must be finite")
    return value
