"""
Utility-Threshold Voting Game for feature selection (§4.1).

Defines the binary winning function  v_alpha: 2^{N_1} → {0, 1}:

    v_alpha(S) = 𝟙[U(M_S) ≥ α · U_full]

where:
- S ⊆ N_1 is a feature subset (coalition)
- M_S is a base learner trained on features S
- α ∈ (0, 1] is the utility quota
- U is a pluggable utility (see ``minppi_mofs.utility``)

Model and utility are **decoupled**: classification always uses XGBoost
and produces probabilities; the *utility* parameter selects which metric
is computed from those probabilities.  This makes ISOLET utility ablation
valid (same model, different metric).

Tasks:
- ``"binary"``    → default ``utility="roc_auc"``
- ``"multiclass"`` → default ``utility="ovr_auc"``
- ``"regression"`` → default ``utility="r2"``

Usage::

    from minppi_mofs.voting_game import compute_v_alpha, make_acc_full_cache
    cache = make_acc_full_cache(X_train, X_val, y_train, y_val, seed=42)
    v = compute_v_alpha({"f0", "f2"}, X_train, X_val, y_train, y_val,
                         alpha=0.95, acc_cache=cache, seed=42)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from minppi_mofs.utility import _ALL_METRICS, compute_utility, is_classification_metric

# ── default utility per task ────────────────────────────────────────────
_DEFAULT_UTILITY: dict[str, str] = {
    "binary": "roc_auc",
    "multiclass": "ovr_auc",
    "regression": "r2",
}


# ---------------------------------------------------------------------------
# Cached utility evaluation
# ---------------------------------------------------------------------------


class AccFullCache:
    """Cache for full-feature utility and per-subset evaluations.

    Avoids re-evaluating the same coalition across multiple calls — critical
    because each evaluation requires training an XGBoost model.

    **Classification** (binary / multiclass): always uses ``XGBClassifier``
    and stores probability predictions.  The *utility* parameter selects
    which metric is computed (``"roc_auc"``, ``"ovr_auc"``, ``"accuracy"``,
    ``"pr_auc"``).

    **Regression**: uses ``XGBRegressor`` with ``"r2"`` or ``"rmse"``.

    Attributes:
        acc_full: Utility of the model on all features.
        threshold: ``alpha * acc_full`` — the winning threshold.
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
        """Compute U_full and initialise the subset cache.

        Args:
            X_train / X_val / y_train / y_val: Data arrays.
            task: ``"binary"``, ``"multiclass"``, or ``"regression"``.
            seed: Random seed for XGBoost.
            utility: Metric identifier (see ``minppi_mofs.utility``).
                If ``None``, defaults are: ``binary`` → ``"roc_auc"``,
                ``multiclass`` → ``"ovr_auc"``, ``regression`` → ``"r2"``.
            model_params: Override for XGBoost params, e.g.
                ``{"n_estimators": 50, "max_depth": 3}``.
        """
        self._X_train = X_train
        self._X_val = X_val
        self._y_train = y_train
        self._y_val = y_val
        self._task = task
        self._seed = seed
        self._cache: dict[tuple[str, ...], float] = {}

        # Resolve utility
        if utility is None:
            utility = _DEFAULT_UTILITY.get(task, "roc_auc")
        if utility not in _ALL_METRICS:
            raise ValueError(
                f"Unknown utility {utility!r}. Choose from {sorted(_ALL_METRICS)}."
            )
        if task == "regression" and is_classification_metric(utility):
            raise ValueError(
                f"Utility {utility!r} is classification-only; "
                f"use 'r2' or 'rmse' for regression."
            )
        self._utility = utility

        # ── Build model (unified: XGBoost for all classification) ─────
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
        # ─────────────────────────────────────────────────────────────

        # Compute U_full
        self.acc_full = self._evaluate_subset(set(X_train.columns), _seed_base=seed)
        self.threshold: Optional[float] = None

    def _evaluate_subset(self, S: set[str], *, _seed_base: int = 42) -> float:
        """Train on *S*, produce predictions, compute utility.

        Reuses ``self._model`` — ``.fit()`` replaces internal state.
        """
        key = tuple(sorted(S))
        if key in self._cache:
            return self._cache[key]

        cols = list(S)
        # ── empty coalition fallback ─────────────────────────────────
        if len(cols) == 0:
            val = 0.5 if self._task != "regression" else 0.0
            self._cache[key] = val
            return val

        Xtr = self._X_train[cols].values
        Xva = self._X_val[cols].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(Xtr, self._y_train)

        if self._task == "regression":
            y_pred = self._model.predict(Xva)
        else:
            y_pred = self._model.predict_proba(Xva)

        val = compute_utility(self._y_val, y_pred, metric=self._utility)
        self._cache[key] = val
        return val

    def evaluate(self, S: set[str]) -> float:
        """Return U(M_S) for coalition *S*."""
        return self._evaluate_subset(S)

    def set_threshold(self, alpha: float) -> None:
        """Set the winning threshold = alpha * Acc_full."""
        self.threshold = self.acc_full * alpha

    def is_winning(self, S: set[str]) -> bool:
        """True iff U(M_S) >= alpha * U_full.

        Raises RuntimeError if ``set_threshold`` hasn't been called.
        """
        if self.threshold is None:
            raise RuntimeError("Call set_threshold(alpha) before is_winning().")
        return self.evaluate(S) >= self.threshold

    def eval_marginal(self, S: set[str], i: str) -> int:
        """Marginal contribution: v_alpha(S∪{i}) - v_alpha(S) ∈ {-1,0,+1}."""
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
    """Create an :class:`AccFullCache` with ``U_full`` pre-computed.

    Args:
        X_train / X_val / y_train / y_val: Data arrays.
        task: ``"binary"``, ``"multiclass"``, or ``"regression"``.
        seed: Random seed.
        utility: Metric override (defaults per *task*).
        model_params: Optional XGBoost param overrides.

    Returns:
        Initialised cache.
    """
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
    """Evaluate the binary winning function v_alpha(S).

    Returns 1 if U(M_S) >= α·U_full, else 0.
    """
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
