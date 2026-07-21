"""
Utility-Threshold Voting Game for feature selection (§4.1).

Defines the binary winning function  v_alpha: 2^{N_1} → {0, 1}:

    v_alpha(S) = 𝟙[U(M_S) ≥ α · U_full]

where:
- S ⊆ N_1 is a feature subset (coalition)
- M_S is a base learner trained on features S
- α ∈ (0, 1] is the utility quota
- U is a pluggable utility (see ``minppi_mofs.utility``)

**Prediction cache (2026-07-21).**  ``AccFullCache`` caches raw predictions
(probabilities for classification, continuous values for regression), NOT
scalar utilities.  Multiple ``UtilityView`` instances can read the same
predictions under different metrics — this is the structural guarantee that
the ISOLET utility ablation compares metrics on truly identical model outputs.

Usage::

    from minppi_mofs.voting_game import AccFullCache, UtilityView

    cache = AccFullCache(X_train, X_val, y_train, y_val, task="multiclass")
    view_acc = UtilityView(cache, utility="accuracy", alpha=0.95)
    view_auc = UtilityView(cache, utility="ovr_auc", alpha=0.95)
    # Both views share the SAME trained models and predictions.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from minppi_mofs.utility import _ALL_METRICS, compute_utility, is_classification_metric

_VALID_TASKS = frozenset({"binary", "multiclass", "regression"})

_DEFAULT_UTILITY: dict[str, str] = {
    "binary": "roc_auc",
    "multiclass": "ovr_auc",
    "regression": "r2",
}


def _validate_task_utility(task: str, utility: str) -> None:
    """Reject impossible (task, utility) pairs before any model is built.

    Raises ValueError with the field name in the message so tests can
    ``pytest.raises(ValueError, match="task"|"classification-only")``.
    """
    if task not in _VALID_TASKS:
        raise ValueError(
            f"task must be one of {sorted(_VALID_TASKS)}, got {task!r}"
        )

    is_cls_metric = is_classification_metric(utility)
    is_reg_task = (task == "regression")

    if is_reg_task and is_cls_metric:
        raise ValueError(
            f"Utility {utility!r} is classification-only; "
            f"use 'r2' or 'rmse' for task={task!r}."
        )
    if (not is_reg_task) and (not is_cls_metric):
        raise ValueError(
            f"Utility {utility!r} is regression-only (r2/rmse); "
            f"cannot be used with classification task={task!r}."
        )


# ---------------------------------------------------------------------------
# Prediction cache (model → predictions, shared across utility views)
# ---------------------------------------------------------------------------


class AccFullCache:
    """Trains XGBoost once per coalition and caches the raw predictions.

    The cache key is the sorted tuple of feature names.  The cached value
    is a numpy array: probabilities ``(n_val, n_classes)`` for classification
    or continuous predictions ``(n_val,)`` for regression.

    Multiple :class:`UtilityView` instances can wrap the same cache and
    compute different metrics from the identical predictions — this is the
    structural guarantee for controlled utility-function ablation.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: np.ndarray,
        y_val: np.ndarray,
        *,
        task: str = "binary",
        seed: int = 42,
        utility: Optional[str] = None,
        model_params: Optional[dict] = None,
    ) -> None:
        self._X_train = X_train
        self._X_val = X_val
        self._y_train = y_train
        self._y_val = y_val
        self._task = task
        self._seed = seed

        # ── predictions cache: key → np.ndarray ──────────────────────
        self._preds: dict[tuple[str, ...], np.ndarray] = {}

        # ── default utility ────────────────────────────────────────────
        if utility is None:
            utility = _DEFAULT_UTILITY.get(task, "roc_auc")
        if utility not in _ALL_METRICS:
            raise ValueError(
                f"Unknown utility {utility!r}. Choose from {sorted(_ALL_METRICS)}."
            )
        _validate_task_utility(task, utility)
        self._utility = utility

        # ── model (reused across coalitions) ──────────────────────────
        mp = model_params or {}
        self._model_params = {
            "n_estimators": mp.get("n_estimators", 15),
            "max_depth": mp.get("max_depth", 2),
            "learning_rate": mp.get("learning_rate", 0.3),
            "n_jobs": mp.get("n_jobs", 1),
        }
        if task == "regression":
            self._model = XGBRegressor(
                verbosity=0, random_state=seed, **self._model_params,
            )
        else:
            self._model = XGBClassifier(
                verbosity=0, random_state=seed, **self._model_params,
            )

        # ── compute U_full from default utility ───────────────────────
        self.acc_full = self.evaluate(set(X_train.columns))
        self.threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Core: train once, cache predictions
    # ------------------------------------------------------------------

    def get_predictions(self, S: set[str], *, _seed_base: int = 42) -> np.ndarray:
        """Return cached predictions for coalition *S*, training if needed.

        Args:
            S: Feature subset.
            _seed_base: Seed offset for the model (ensures determinism).

        Returns:
            For classification: probability matrix ``(n_val, n_classes)``.
            For regression: continuous predictions ``(n_val,)``.
        """
        key = tuple(sorted(S))
        if key in self._preds:
            return self._preds[key]

        cols = sorted(S)
        if len(cols) == 0:
            # empty coalition → training-set class priors (no model trained).
            # These are valid "predictions" for every metric via the standard
            # compute_utility path: accuracy → majority-class proportion,
            # PR-AUC → positive-class prior, OVR AUC / ROC-AUC → 0.5, etc.
            if self._task == "regression":
                preds = np.full(len(self._y_val), float(np.mean(self._y_train)))
            else:
                cls, cnt = np.unique(self._y_train, return_counts=True)
                priors = cnt.astype(float) / cnt.sum()
                n_cls = len(cls)
                preds = np.tile(priors, (len(self._y_val), 1))
                # ensure column count matches number of unique classes
                if preds.shape[1] < max(2, n_cls):
                    preds = np.full((len(self._y_val), max(2, n_cls)), 0.0, dtype=float)
                    for i, p in enumerate(priors):
                        preds[:, i] = p
            self._preds[key] = preds
            return preds

        Xtr = self._X_train[cols].values
        Xva = self._X_val[cols].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(Xtr, self._y_train)

        if self._task == "regression":
            preds = self._model.predict(Xva)
        else:
            preds = self._model.predict_proba(Xva)

        self._preds[key] = preds
        return preds

    def evaluate(self, S: set[str], *, utility: Optional[str] = None) -> float:
        """Utility of coalition *S* under ``utility`` (or ``self._utility``).

        Reads from the shared prediction cache — no model retraining.
        """
        metric = utility if utility is not None else self._utility
        if metric not in _ALL_METRICS:
            raise ValueError(
                f"Unknown utility {metric!r}. Choose from {sorted(_ALL_METRICS)}."
            )
        _validate_task_utility(self._task, metric)
        preds = self.get_predictions(S)
        return compute_utility(self._y_val, preds, metric=metric)

    # ------------------------------------------------------------------
    # Threshold / winning (use self._utility)
    # ------------------------------------------------------------------

    def set_threshold(self, alpha: float) -> None:
        self.threshold = self.acc_full * alpha

    def is_winning(self, S: set[str]) -> bool:
        if self.threshold is None:
            raise RuntimeError("Call set_threshold(alpha) before is_winning().")
        return self.evaluate(S) >= self.threshold

    def eval_marginal(self, S: set[str], i: str) -> int:
        v_before = 1 if self.is_winning(S) else 0
        v_after = 1 if self.is_winning(S | {i}) else 0
        return v_after - v_before


# ---------------------------------------------------------------------------
# Utility view — a metric-specific lens on a shared prediction cache
# ---------------------------------------------------------------------------


class UtilityView:
    """A lightweight metric-specific view of an :class:`AccFullCache`.

    Shares the **same** trained models and cached predictions as the parent
    cache, but computes :meth:`evaluate`, :meth:`is_winning`, and
    :meth:`eval_marginal` under a different utility metric.

    This is duck-type compatible with the ``acc_cache`` parameter of every
    ``estimate_minppi_*`` function.

    Usage::

        cache = AccFullCache(X_tr, X_va, y_tr, y_va)
        view_acc = UtilityView(cache, utility="accuracy", alpha=0.95)
        view_auc = UtilityView(cache, utility="ovr_auc",  alpha=0.95)
        # Both read from cache._preds — models trained exactly once.
    """

    def __init__(
        self,
        cache: AccFullCache,
        *,
        utility: str,
        alpha: float,
    ) -> None:
        if utility not in _ALL_METRICS:
            raise ValueError(
                f"Unknown utility {utility!r}. Choose from {sorted(_ALL_METRICS)}."
            )
        _validate_task_utility(cache._task, utility)
        self._cache = cache
        self._utility = utility
        self.acc_full = cache.evaluate(
            set(cache._X_train.columns), utility=utility,
        )
        self.threshold: float = self.acc_full * alpha

    def evaluate(self, S: set[str]) -> float:
        return self._cache.evaluate(S, utility=self._utility)

    def set_threshold(self, alpha: float) -> None:
        self.threshold = self.acc_full * alpha

    def is_winning(self, S: set[str]) -> bool:
        return self.evaluate(S) >= self.threshold

    def eval_marginal(self, S: set[str], i: str) -> int:
        v_before = 1 if self.is_winning(S) else 0
        v_after = 1 if self.is_winning(S | {i}) else 0
        return v_after - v_before


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_acc_full_cache(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    task: str = "binary",
    seed: int = 42,
    utility: Optional[str] = None,
    model_params: Optional[dict] = None,
) -> AccFullCache:
    """Create an :class:`AccFullCache` with ``U_full`` pre-computed."""
    return AccFullCache(
        X_train, X_val, y_train, y_val,
        task=task, seed=seed, utility=utility, model_params=model_params,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_v_alpha(
    S: set[str],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float = 0.95,
    task: str = "binary",
    utility: Optional[str] = None,
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
) -> int:
    """Evaluate the binary winning function v_alpha(S)."""
    if acc_cache is None:
        acc_cache = make_acc_full_cache(
            X_train, X_val, y_train, y_val,
            task=task, seed=seed, utility=utility,
        )
    acc_cache.set_threshold(alpha)
    return 1 if acc_cache.is_winning(S) else 0


def classify_minppi(minppi_value: int) -> str:
    """Map MinPPI score → label: +1=decisive, 0=neutral, -1=harmful."""
    return {+1: "decisive", 0: "neutral", -1: "harmful"}[minppi_value]


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print(f"\n{'='*72}\n{t}\n{'='*72}")

    _demo_banner("v_alpha voting game — synthetic demo")

    rng = np.random.default_rng(42)
    n = 300
    X_demo = pd.DataFrame({
        "f_signal": rng.normal(0, 1, n),
        "f_noise1": rng.normal(0, 1, n),
        "f_noise2": rng.normal(0, 1, n),
    })
    y_demo = np.where(X_demo["f_signal"] > 0, 1, 0).astype(int)

    n_tr = int(0.7 * n)
    X_tr, X_val = X_demo.iloc[:n_tr], X_demo.iloc[n_tr:]
    y_tr, y_val = y_demo[:n_tr], y_demo[n_tr:]

    cache = make_acc_full_cache(X_tr, X_val, y_tr, y_val, seed=42)
    print(f"Full-feature utility: {cache.acc_full:.4f}  ({cache._utility})")

    for alpha in [0.95, 0.85, 0.99]:
        cache.set_threshold(alpha)
        print(f"\nα = {alpha}  → threshold = {cache.threshold:.4f}")
        for S_desc, S in [
            ("{f_signal}", {"f_signal"}),
            ("{f_noise1, f_noise2}", {"f_noise1", "f_noise2"}),
            ("all features", {"f_signal", "f_noise1", "f_noise2"}),
        ]:
            is_win = cache.is_winning(S)
            print(f"  v_α({S_desc:30s}) = {int(is_win)}  "
                  f"(U={cache.evaluate(S):.4f})")

    _demo_banner("Non-monotonicity demo")
    cache.set_threshold(0.99)
    S_base = {"f_signal"}
    v_base = cache.is_winning(S_base)
    v_with_noise = cache.is_winning(S_base | {"f_noise1"})
    print(f"v_α({{f_signal}})          = {int(v_base)}  "
          f"(U={cache.evaluate(S_base):.4f})")
    print(f"v_α({{f_signal, f_noise1}}) = {int(v_with_noise)}  "
          f"(U={cache.evaluate(S_base | {'f_noise1'}):.4f})")
    if v_base and not v_with_noise:
        print("Observation: Non-monotonicity confirmed.")
    else:
        print("Observation: Not observed at this α level.")
