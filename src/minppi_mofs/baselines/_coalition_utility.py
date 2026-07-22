"""Deterministic cached utility evaluator shared by coalition baselines."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from minppi_mofs.modeling import evaluate_fixed_subset


def default_utility(task: str) -> str:
    return {"binary": "roc_auc", "multiclass": "ovr_auc", "regression": "r2"}.get(
        task, ""
    )


class CoalitionUtility:
    """Evaluate ordered coalitions on one frozen internal holdout split."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        task: str,
        utility: str,
        model_params: Mapping[str, object],
        backend: str,
        seed: int,
        test_size: float = 0.2,
        max_samples: int = 5000,
        diagnostics: MutableMapping[str, int] | None = None,
    ) -> None:
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise ValueError("X must be a DataFrame with unique columns")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError("max_samples must be an integer of at least two")
        features = tuple(X.columns)
        target = np.asarray(y)
        if len(X) != len(target):
            raise ValueError("X and y row counts must match")
        if len(X) > max_samples:
            material = f"{seed}\0" + "\0".join(features)
            stable = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
            indices = np.random.default_rng(stable).choice(
                len(X), size=max_samples, replace=False
            )
            X = X.iloc[indices].reset_index(drop=True)
            target = target[indices]
        positions = np.arange(len(X))
        stratify = target if task != "regression" else None
        try:
            train_pos, eval_pos = train_test_split(
                positions,
                test_size=test_size,
                random_state=seed,
                stratify=stratify,
            )
        except ValueError:
            train_pos, eval_pos = train_test_split(
                positions, test_size=test_size, random_state=seed
            )
        self._features = features
        self._X_train = X.iloc[train_pos].reset_index(drop=True)
        self._X_eval = X.iloc[eval_pos].reset_index(drop=True)
        self._y_train = target[train_pos]
        self._y_eval = target[eval_pos]
        self._task = task
        self._utility = utility
        self._model_params = dict(model_params)
        self._backend = backend
        self._seed = seed
        self._cache: dict[tuple[str, ...], float] = {}
        self._diagnostics = diagnostics
        self._update_diagnostics()

    def _update_diagnostics(self) -> None:
        if self._diagnostics is not None:
            self._diagnostics["coalition_count"] = len(self._cache)
            self._diagnostics["model_fit_count"] = sum(bool(key) for key in self._cache)

    def __call__(self, coalition: set[str]) -> float:
        if not isinstance(coalition, set) or not coalition.issubset(self._features):
            raise ValueError("coalition contains an unknown feature")
        key = tuple(feature for feature in self._features if feature in coalition)
        if key not in self._cache:
            self._cache[key] = evaluate_fixed_subset(
                self._X_train,
                self._y_train,
                self._X_eval,
                self._y_eval,
                key,
                task=self._task,
                utility=self._utility,
                model_params=self._model_params,
                backend=self._backend,
                seed=self._seed,
            )
            self._update_diagnostics()
        return self._cache[key]
