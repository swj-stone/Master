"""
Split-first data protocol — leak-free feature selection and standardisation.

**Problem it solves (P0-2).**  The seven original preprocess scripts fit
``mutual_info_classif`` / ``mutual_info_regression`` on the full dataset
*before* splitting, so test labels influenced which features were selected.
``prepare_split()`` reverses the order: split, then fit every supervised
and unsupervised transform on the training set only.

**Processing order (guaranteed leak-free):**

    raw data
      → train_test_split(test_size=test_frac)     → X_temp, X_test
      → train_test_split(test_size=val_frac/(train+val))  → X_train, X_val
      → VarianceThreshold.fit(train)               → transform all
      → mutual_info_*.fit(train)                   → select top-k cols
      → StandardScaler.fit(train)                  → transform all
      → DataSplit

Usage::

    from minppi_mofs.data_protocol import DataSplit, prepare_split

    split = prepare_split(X, y, task="binary", seed=42, max_features=100)
    split = prepare_split(X, y, train_frac=0.64, val_frac=0.16, test_frac=0.20)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_VALID_TASKS = frozenset({"binary", "multiclass", "regression"})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSplit:
    """Train / validation / test partition with leak-free feature selection.

    All feature matrices are **standardised** (zero mean, unit variance,
    fitted on training data only).  ``selected_features`` reports which
    columns survived the VT + MI pipeline.

    Attributes:
        X_train / X_val / X_test: Scaled DataFrames (same column subset).
        y_train / y_val / y_test: Target arrays.
        selected_features: Column names kept after VT and MI filtering.
        seed: Random seed that produced this split.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    selected_features: tuple[str, ...]
    seed: int


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(
    n_samples: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    task: str,
    max_features: Optional[int],
    variance_threshold: float,
) -> None:
    """Raise ``ValueError`` on invalid parameters."""
    for name, val in [("train_frac", train_frac), ("val_frac", val_frac),
                      ("test_frac", test_frac)]:
        if not isinstance(val, (int, float)) or not (0.0 < val < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {val!r}")

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(
            f"train_frac + val_frac + test_frac = {train_frac} + {val_frac} "
            f"+ {test_frac} = {total}, must equal 1.0"
        )
    if task not in _VALID_TASKS:
        raise ValueError(
            f"task must be one of {sorted(_VALID_TASKS)}, got {task!r}"
        )
    if max_features is not None:
        if not isinstance(max_features, int) or max_features < 1:
            raise ValueError(
                f"max_features must be None or int >= 1, got {max_features!r}"
            )
    if not isinstance(variance_threshold, (int, float)) or variance_threshold < 0.0:
        raise ValueError(
            f"variance_threshold must be >= 0, got {variance_threshold!r}"
        )
    if n_samples < 3:
        raise ValueError(
            f"need at least 3 samples for train/val/test splits, got {n_samples}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str = "binary",
    seed: int = 42,
    max_features: Optional[int] = 100,
    train_frac: float = 0.64,
    val_frac: float = 0.16,
    test_frac: float = 0.20,
    variance_threshold: float = 0.0,
) -> DataSplit:
    """Split first, then fit every transform on training data only.

    **Split formula** (exact three-way)::

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_frac, ...)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac / (train_frac + val_frac), ...)

    **Leak-free transform pipeline (train-only fit, apply to all):**

    1. ``VarianceThreshold.fit(train)`` → ``transform(train/val/test)``
    2. ``mutual_info_*.fit(train)`` → select top-*max_features* columns
    3. ``StandardScaler.fit(train)`` → ``transform(train/val/test)``

    Args:
        X: Feature DataFrame (all samples).
        y: Target array.
        task: ``"binary"``, ``"multiclass"``, or ``"regression"``.
        seed: Random seed for splitting and MI.
        max_features: Cap to this many MI top-k columns; ``None`` = keep all.
        train_frac: Training fraction, in ``(0, 1)``  (default 0.64).
        val_frac: Validation fraction, in ``(0, 1)`` (default 0.16).
        test_frac: Test fraction, in ``(0, 1)``       (default 0.20).
            The three fractions must sum to 1.0.
        variance_threshold: Min variance for ``VarianceThreshold``
            (default 0.0 = keep all non-constant columns).

    Returns:
        ``DataSplit`` with standardised DataFrames and row indices preserved.
    """
    _validate(len(X), train_frac, val_frac, test_frac, task, max_features,
              variance_threshold)

    rng = np.random.default_rng(seed)

    # ── 1. Split: test from full, then val from temp ───────────────────
    stratify_arr: Optional[np.ndarray] = y if task != "regression" else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_frac,
        random_state=int(rng.integers(0, 2**31)),
        stratify=stratify_arr,
    )

    stratify_temp: Optional[np.ndarray] = y_temp if task != "regression" else None
    val_ratio = val_frac / (train_frac + val_frac)  # relative to temp
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=int(rng.integers(0, 2**31)),
        stratify=stratify_temp,
    )

    # ── 2. VarianceThreshold (train-only fit, transform all) ───────────
    all_cols = list(X_train.columns)
    if variance_threshold > 0.0:
        vt = VarianceThreshold(threshold=variance_threshold)
        X_tr_vt = vt.fit_transform(X_train[all_cols].values)
        kept_mask = vt.get_support()
        vt_cols = [all_cols[i] for i in range(len(all_cols)) if kept_mask[i]]
        X_train = pd.DataFrame(X_tr_vt, columns=vt_cols, index=X_train.index)
        X_val = X_val[vt_cols]
        X_test = X_test[vt_cols]
        all_cols = vt_cols

    # ── 3. Feature selection (train-only MI, same columns for all) ───
    if max_features is not None and max_features < len(all_cols):
        if task == "regression":
            mi = mutual_info_regression(
                X_train.values, y_train, random_state=seed
            )
        else:
            mi = mutual_info_classif(
                X_train.values, y_train, random_state=seed
            )
        scores = pd.Series(mi, index=all_cols)
        top_cols: list[str] = (
            scores.abs().sort_values(ascending=False).head(max_features)
            .index.tolist()
        )
    else:
        top_cols = all_cols

    # ── 4. Standardise (train-only fit, transform all) ─────────────────
    scaler = StandardScaler()
    X_tr_vals = scaler.fit_transform(X_train[top_cols].values)
    X_va_vals = scaler.transform(X_val[top_cols].values)
    X_te_vals = scaler.transform(X_test[top_cols].values)

    # ── 5. Wrap (preserve original row indices for auditability) ───────
    return DataSplit(
        X_train=pd.DataFrame(X_tr_vals, columns=top_cols, index=X_train.index),
        X_val=pd.DataFrame(X_va_vals, columns=top_cols, index=X_val.index),
        X_test=pd.DataFrame(X_te_vals, columns=top_cols, index=X_test.index),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        selected_features=tuple(top_cols),
        seed=seed,
    )
