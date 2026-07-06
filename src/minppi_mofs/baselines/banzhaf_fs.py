"""
Banzhaf feature importance baseline.

Banzhaf for Feature Selection (Karavas et al. 2017, arXiv:1711.04992) frames
features as voters in a weighted voting game and uses the **Banzhaf power
index** to rank them.  Unlike the Shapley value (which weights marginal
contributions by subset-size combinatorics), the Banzhaf index is an
unweighted average of marginals over all 2^{n-1} coalitions::

    beta_i = (1 / 2^{n-1}) * sum_{S ⊆ N\\{i}} [v(S ∪ {i}) - v(S)]

where  v(S) = AUC(XGBoost(S))  (continuous characteristic function).

The continuous v(·) avoids the alpha-threshold sensitivity of binary v_alpha;
it also makes the estimator consistent with the MinShap baseline (same v,
different aggregation operator: mean vs min).

Estimation uses **shared-coalition Monte Carlo**:

1. Sample *M* random coalitions S ⊆ N.
2. For each S, evaluate v(S) once.
3. For a random subset of features not in S (up to *per_coalition*), evaluate
   v(S ∪ {i}) and accumulate the marginal.
4. Divide each feature's accumulated marginals by its evaluation count.

Usage::

    from minppi_mofs.baselines.banzhaf_fs import compute_banzhaf_importance
    scores = compute_banzhaf_importance(X_train, y_train)

Run (stand-alone)::

    python src/minppi_mofs/baselines/banzhaf_fs.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Cached continuous-AUC characteristic function
# ---------------------------------------------------------------------------


class _CachedAUC:
    """v(S) = AUC of a shallow XGBoost trained on feature subset S.

    Results are cached by sorted-tuple key.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        seed: int = 42,
        test_size: float = 0.2,
        max_samples: int = 5000,
        task: str = "binary",
    ) -> None:
        self._X = X
        self._y = y
        self._seed = seed
        self._test_size = test_size
        self._max_samples = max_samples
        self._task = task
        self._cache: dict[tuple[str, ...], float] = {}

    def __call__(self, S: set[str]) -> float:
        key = tuple(sorted(S))
        if key in self._cache:
            return self._cache[key]

        cols = list(S)
        if len(cols) == 0:
            val = 0.0 if self._task == "regression" else 0.5
        else:
            X_sub = self._X[cols].values
            n_rows = X_sub.shape[0]
            if n_rows > self._max_samples:
                rng = np.random.default_rng(self._seed + hash(key) % 2**31)
                idx = rng.choice(n_rows, size=self._max_samples, replace=False)
                X_sub = X_sub[idx]
                y_sub = self._y[idx]
            else:
                y_sub = self._y
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub,
                    y_sub,
                    test_size=self._test_size,
                    random_state=self._seed,
                    stratify=y_sub if self._task != "regression" else None,
                )
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub,
                    y_sub,
                    test_size=self._test_size,
                    random_state=self._seed,
                )
            if self._task == "regression":
                model = XGBRegressor(
                    n_estimators=10,
                    max_depth=2,
                    learning_rate=0.3,
                    verbosity=0,
                    random_state=self._seed,
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                val = float(r2_score(y_te, y_pred))
                if val < 0.0:
                    val = 0.0
            elif self._task == "multiclass":
                model = LogisticRegression(
                    max_iter=200,
                    solver="lbfgs",
                    random_state=self._seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_tr)
                val = float(accuracy_score(y_te, model.predict(X_te)))
            else:
                model = XGBClassifier(
                    n_estimators=10,
                    max_depth=2,
                    learning_rate=0.3,
                    verbosity=0,
                    random_state=self._seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_tr)
                y_prob = model.predict_proba(X_te)
                y_test_classes = np.unique(y_te)
                if len(y_test_classes) < 2:
                    val = 0.5
                elif y_prob.shape[1] == 2:
                    val = float(roc_auc_score(y_te, y_prob[:, 1]))
                else:
                    val = float(roc_auc_score(y_te, y_prob, multi_class="ovr"))
        self._cache[key] = val
        return val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_banzhaf_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_coalitions: int = 200,
    per_coalition: int = 5,
    seed: int = 42,
    verbose: bool = False,
    task: str = "binary",
) -> dict[str, float]:
    """Compute Banzhaf feature importance via shared-coalition Monte Carlo.

    Uses continuous  v(S) = AUC(XGBoost(S)) → marginals in [−1, 1].

    Args:
        X: Feature DataFrame.
        y: Target array.
        n_coalitions: Number of random coalitions to sample.
        per_coalition: Max features to test for marginal per coalition.
        seed: Random seed.
        verbose: Print progress every 50 coalitions.
        task: ``"binary"`` for classification (AUC), ``"regression"`` for R^2.

    Returns:
        Dict ``{feature_name: banzhaf_value}`` sorted descending.
    """
    rng = np.random.default_rng(seed)
    features = list(X.columns)
    n = len(features)
    all_set = set(features)

    v = _CachedAUC(X, y, seed=seed, task=task)
    full_auc = v(all_set)
    if verbose:
        print(f"  Full-feature AUC: {full_auc:.4f}", flush=True)

    accum = {name: 0.0 for name in features}
    counts = {name: 0 for name in features}

    for k in range(n_coalitions):
        if verbose and (k + 1) % 50 == 0:
            done = sum(1 for c in counts.values() if c > 0)
            print(
                f"  Banzhaf: {k + 1}/{n_coalitions}  ({done}/{n} features seen)",
                flush=True,
            )

        s_size = rng.integers(0, max(1, n // 3) + 1)
        S_list = rng.choice(features, size=s_size, replace=False).tolist()
        S = set(S_list)
        v_S = v(S)

        outside = list(all_set - S)
        rng.shuffle(outside)
        test_feats = outside[:per_coalition]

        for feat in test_feats:
            v_Si = v(S | {feat})
            accum[feat] += float(v_Si - v_S)
            counts[feat] += 1

        # Early stop: every feature evaluated at least twice.
        # With feature reduction (max 200 feats) the coupon-collector
        # overhead is negligible, and 2 samples give the mean estimator
        # a reasonable variance reduction.
        min_eval = 2
        all_covered = all(c >= min_eval for c in counts.values())
        if all_covered and k >= 20:
            if verbose:
                print(
                    f"  Early stop at coalition {k + 1}/{n_coalitions} "
                    f"(every feature has >= {min_eval} evals)",
                    flush=True,
                )
            break

    if verbose:
        min_c = min(counts.values())
        max_c = max(counts.values())
        n_pos = sum(1 for a in accum.values() if a > 0)
        n_neg = sum(1 for a in accum.values() if a < 0)
        print(
            f"  Complete: {n}/{n} features  evals/feat: {min_c}-{max_c}  "
            f"(pos={n_pos}, neg={n_neg})",
            flush=True,
        )

    result = {name: accum[name] / max(counts[name], 1) for name in features}
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


def rank_features(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Convert a score dict to a ranked list (highest first)."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("Banzhaf baseline — synthetic demo")
    rng = np.random.default_rng(42)
    n_samp = 200
    X_demo = pd.DataFrame(
        {
            "f0": rng.normal(0, 1, n_samp),
            "f1": rng.normal(0, 1, n_samp),
            "f2": rng.normal(0, 1, n_samp),
        }
    )
    y_demo = np.where(X_demo["f0"] + 0.1 * X_demo["f1"] > 0, 1, 0)

    scores = compute_banzhaf_importance(X_demo, y_demo, n_coalitions=100, verbose=True)
    print("Feature ranking:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.6f}")
    print("\nObservation: f0 should score highest.")
