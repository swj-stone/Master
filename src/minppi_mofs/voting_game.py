"""
Weighted Voting Game formulation for feature selection (§4.1).

Defines the binary winning function  v_alpha: 2^{N_1} → {0, 1}:

    v_alpha(S) = 𝟙[Acc(ℳ_S) ≥ α · Acc_full]

where:
- S ⊆ N_1 is a feature subset (coalition)
- ℳ_S is a base learner (XGBoost default) trained on features S
- α ∈ (0, 1] is the accuracy quota
- Acc_full = Acc(ℳ_{N_1}) is the full-feature accuracy

Key properties (distinguishing from WVG-FS):
- **No additivity assumption**: v_alpha is evaluated by actual model training
- **Non-monotonic**: adding features can decrease accuracy (noise/redundancy),
  i.e. ∃S ⊂ T: v_alpha(S)=1, v_alpha(T)=0 — the basis for MinPPI's
  three-way classification in Theorem 1.

Usage::

    from minppi_mofs.voting_game import compute_v_alpha, make_acc_full_cache
    acc_cache = make_acc_full_cache(X_train, X_val, y_train, y_val, seed=42)
    v = compute_v_alpha({"f0", "f2"}, X_train, X_val, y_train, y_val,
                         alpha=0.95, acc_cache=acc_cache, seed=42)
    # v → 0 or 1
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Cached accuracy evaluation
# ---------------------------------------------------------------------------


class AccFullCache:
    """Cache for full-feature accuracy and per-subset evaluations.

    Avoids re-evaluating the same coalition across multiple calls — critical
    because each evaluation requires training an XGBoost model.

    Attributes:
        acc_full: AUC of the model trained on all features.
        alpha: The quota parameter.
        threshold: ``alpha * acc_full`` — the winning threshold.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: np.ndarray,
        y_val: np.ndarray,
        *,
        seed: int = 42,
    ) -> None:
        """Compute Acc_full and initialise the subset cache.

        Args:
            X_train: Training feature matrix.
            X_val: Validation feature matrix.
            y_train: Training target.
            y_val: Validation target.
            seed: Random seed for XGBoost.
        """
        self._X_train = X_train
        self._X_val = X_val
        self._y_train = y_train
        self._y_val = y_val
        self._seed = seed
        self._cache: dict[tuple[str, ...], float] = {}

        # Compute Acc_full
        self.acc_full = self._evaluate_subset(set(X_train.columns), _seed_base=seed)
        self.threshold: Optional[float] = None  # set later when alpha is chosen

    def _evaluate_subset(self, S: set[str], *, _seed_base: int = 42) -> float:
        """Train XGBoost on S and return AUC on validation set.

        Results are cached by sorted-tuple key for reuse across calls with
        different α values.
        """
        key = tuple(sorted(S))
        if key in self._cache:
            return self._cache[key]

        cols = list(S)
        if len(cols) == 0:
            val = 0.5
        else:
            X_tr_sub = self._X_train[cols].values
            X_val_sub = self._X_val[cols].values
            y_val_classes = np.unique(self._y_val)

            model = XGBClassifier(
                n_estimators=15,
                max_depth=2,
                learning_rate=0.3,
                verbosity=0,
                random_state=_seed_base,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_sub, self._y_train)

            y_prob = model.predict_proba(X_val_sub)
            if len(y_val_classes) < 2:
                val = 0.5
            elif y_prob.shape[1] == 2:
                val = float(roc_auc_score(self._y_val, y_prob[:, 1]))
            else:
                val = float(roc_auc_score(self._y_val, y_prob, multi_class="ovr"))
        self._cache[key] = val
        return val

    def evaluate(self, S: set[str]) -> float:
        """Return AUC(ℳ_S) for coalition *S*.

        Args:
            S: Feature subset.

        Returns:
            AUC value in [0, 1].
        """
        return self._evaluate_subset(S)

    def set_threshold(self, alpha: float) -> None:
        """Set the winning threshold = alpha * Acc_full.

        Args:
            alpha: Quota parameter in (0, 1].
        """
        self.threshold = self.acc_full * alpha

    def is_winning(self, S: set[str]) -> bool:
        """Check whether coalition *S* wins.

        Args:
            S: Feature subset.

        Returns:
            True iff AUC(ℳ_S) >= alpha * Acc_full.

        Raises:
            RuntimeError: If ``set_threshold`` has not been called.
        """
        if self.threshold is None:
            raise RuntimeError("Call set_threshold(alpha) before is_winning().")
        return self.evaluate(S) >= self.threshold

    def eval_marginal(self, S: set[str], i: str) -> int:
        """Compute marginal contribution of feature *i* to coalition *S*.

        Args:
            S: Coalition (must not contain *i*).
            i: Feature to add.

        Returns:
            v_alpha(S ∪ {i}) - v_alpha(S) ∈ {-1, 0, +1}.
        """
        v_before = 1 if self.is_winning(S) else 0
        v_after = 1 if self.is_winning(S | {i}) else 0
        return v_after - v_before


def make_acc_full_cache(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 42,
) -> AccFullCache:
    """Create an :class:`AccFullCache` with Acc_full pre-computed.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        seed: Random seed.

    Returns:
        Initialised cache with ``acc_full`` available.
    """
    return AccFullCache(X_train, X_val, y_train, y_val, seed=seed)


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
    acc_cache: Optional[AccFullCache] = None,
    seed: int = 42,
) -> int:
    """Evaluate the binary winning function v_alpha(S).

    Args:
        S: Feature subset (coalition).
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        y_train: Training target.
        y_val: Validation target.
        alpha: Quota parameter in (0, 1].
        acc_cache: Optional pre-built cache. Created on first call if None.
        seed: Random seed for XGBoost (ignored if *acc_cache* is provided).

    Returns:
        1 if AUC(ℳ_S) >= α · Acc_full, else 0.
    """
    if acc_cache is None:
        acc_cache = make_acc_full_cache(X_train, X_val, y_train, y_val, seed=seed)
    acc_cache.set_threshold(alpha)
    return 1 if acc_cache.is_winning(S) else 0


def classify_minppi(
    minppi_value: int,
) -> str:
    """Map a MinPPI value to its three-way classification label.

    Args:
        minppi_value: The MinPPI score ∈ {-1, 0, +1}.

    Returns:
        One of ``"harmful"``, ``"neutral"``, ``"decisive"``.
    """
    return {+1: "decisive", 0: "neutral", -1: "harmful"}[minppi_value]


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("v_alpha voting game — synthetic demo")

    rng = np.random.default_rng(42)
    n = 300
    X_demo = pd.DataFrame(
        {
            "f_signal": rng.normal(0, 1, n),
            "f_noise1": rng.normal(0, 1, n),
            "f_noise2": rng.normal(0, 1, n),
        }
    )
    y_demo = np.where(X_demo["f_signal"] > 0, 1, 0).astype(int)

    # Simple train/val split for the demo
    n_tr = int(0.7 * n)
    X_tr, X_val = X_demo.iloc[:n_tr], X_demo.iloc[n_tr:]
    y_tr, y_val = y_demo[:n_tr], y_demo[n_tr:]

    cache = make_acc_full_cache(X_tr, X_val, y_tr, y_val, seed=42)
    print(f"Full-feature AUC (Acc_full): {cache.acc_full:.4f}")

    for alpha in [0.95, 0.85, 0.99]:
        cache.set_threshold(alpha)
        print(f"\nα = {alpha}  → threshold = {cache.threshold:.4f}")

        # Single "signal" feature should still give reasonable AUC
        for S_desc, S in [
            ("{f_signal}", {"f_signal"}),
            ("{f_noise1, f_noise2}", {"f_noise1", "f_noise2"}),
            ("all features", {"f_signal", "f_noise1", "f_noise2"}),
        ]:
            is_win = cache.is_winning(S)
            auc_val = cache.evaluate(S)
            print(
                f"  v_α({S_desc:30s}) = {int(is_win)}  "
                f"(AUC={auc_val:.4f}, threshold={cache.threshold:.4f})"
            )

    # Demonstrate non-monotonicity: adding a noise feature can hurt
    _demo_banner("Non-monotonicity demo")
    cache.set_threshold(0.99)
    S_base = {"f_signal"}
    v_base = cache.is_winning(S_base)
    v_with_noise = cache.is_winning(S_base | {"f_noise1"})
    print(
        f"v_α({{f_signal}})               = {int(v_base)}  "
        f"(AUC={cache.evaluate(S_base):.4f})"
    )
    print(
        f"v_α({{f_signal, f_noise1}})      = {int(v_with_noise)}  "
        f"(AUC={cache.evaluate(S_base | {'f_noise1'}):.4f})"
    )
    if v_base and not v_with_noise:
        print(
            "Observation: Non-monotonicity confirmed — adding a noise feature "
            "can turn a winning coalition into a losing one.  This is the "
            "foundation for Theorem 1's harmful-feature detection."
        )
    else:
        print(
            "Observation: With this synthetic data and α level, non-monotonicity "
            "was not observed. It becomes more pronounced with noisier data or "
            "higher α values."
        )
