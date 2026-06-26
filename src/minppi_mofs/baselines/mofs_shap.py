"""
MOFS-SHAP baseline for tabular feature importance.

MOFS-SHAP is the ablation baseline for the MOFS-MinPPI framework (§4.4).
In the full NSGA-II algorithm, MinPPI power indices guide mutation and
crossover (§4.4.1–4.4.2).  Replacing MinPPI with **mean |SHAP| values**
yields the ablation control: the search operators use SHAP importance
instead of min-marginal power indices as the bias signal.

For the baseline ranking we run a **SHAP-guided stochastic subset search**:

1. Train XGBoost → extract per-feature gain importance (SHAP proxy).
2. Generate *N* candidate subsets by biased random sampling — each
   feature's inclusion probability is proportional to its gain score.
3. Evaluate each subset via actual XGBoost training (AUC or Accuracy).
4. Keep the Pareto-optimal subsets (objectives: |F|, 1-AUC).
5. Score each feature by its frequency in the elite front × its SHAP
   weight.

This differs from SHAP-Top-k in the *aggregation* step: SHAP-Top-k ranks
by pure gain; MOFS-SHAP combines gain × empirical selection frequency,
capturing interactions that raw SHAP misses.

Usage::

    from minppi_mofs.baselines.mofs_shap import compute_mofs_shap_importance
    scores = compute_mofs_shap_importance(X_train, y_train)

Run (stand-alone)::

    python src/minppi_mofs/baselines/mofs_shap.py
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_mofs_shap_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_samples: int = 100,
    seed: int = 42,
    verbose: bool = False,
    task: str = "binary",
    _result_box: Optional[list] = None,
) -> dict[str, float]:
    """Compute MOFS-SHAP via SHAP-guided stochastic subset search.

    Args:
        X: Feature DataFrame.
        y: Target array.
        n_samples: Number of candidate subsets to evaluate.
        seed: Random seed.
        verbose: Print progress.
        task: ``"binary"`` (default) uses XGBClassifier + AUC;
            ``"regression"`` uses XGBRegressor + (1−R²).
        _result_box: If a list, ``{"fitness": ndarray, "masks": ndarray}`` is
            appended so callers can compute HV / compression without re-running.

    Returns:
        Dict ``{feature_name: mofs_shap_score}`` sorted descending.
        Score = gain_importance × elite-front frequency.
    """
    rng = np.random.default_rng(seed)
    features = list(X.columns)
    n_features = len(features)
    n_classes = len(np.unique(y))

    # ------------------------------------------------------------------
    # 1.  SHAP / gain importance
    # ------------------------------------------------------------------
    safe_map = {c: f"f{i}" for i, c in enumerate(features)}
    X_safe = X.rename(columns=safe_map)

    if task == "regression":
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=seed,
        )
    else:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=seed,
        )
    model.fit(X_safe, y)

    import shap

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_safe)
    if isinstance(shap_vals, list):
        per_class = []
        for s in shap_vals:
            s_abs = np.abs(s)
            if s_abs.ndim == 3:
                per_class.append(s_abs.mean(axis=0).mean(axis=-1))
            else:
                per_class.append(s_abs.mean(axis=0))
        mean_shap = np.mean(per_class, axis=0)
    else:
        mean_shap = np.abs(shap_vals).mean(axis=0)
        if mean_shap.ndim > 1:  # multiclass 3D array
            mean_shap = mean_shap.mean(axis=-1)

    if mean_shap.max() == 0:
        return {f: 0.0 for f in features}
    prob = mean_shap / mean_shap.sum()  # inclusion probability per feature

    if verbose:
        nz = int((mean_shap > 0).sum())
        print(
            f"  Mean |SHAP|: {nz}/{n_features} non-zero, sampling {n_samples} subsets",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 2.  SHAP-biased subset sampling + fitness evaluation
    # ------------------------------------------------------------------
    masks = np.zeros((n_samples, n_features), dtype=np.int8)
    fitness = np.full((n_samples, 2), np.inf, dtype=float)  # (|F|, 1-AUC)

    for i in range(n_samples):
        # Biased inclusion: prob proportional to gain score
        n_feat_in = max(1, rng.binomial(n_features, 0.3))
        chosen = rng.choice(n_features, size=n_feat_in, replace=False, p=prob)
        mask = np.zeros(n_features, dtype=np.int8)
        mask[chosen] = 1
        masks[i] = mask

        cols = [features[j] for j in range(n_features) if mask[j]]
        X_sub = X[cols].values
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub,
                y,
                test_size=0.2,
                random_state=seed,
                stratify=y,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub,
                y,
                test_size=0.2,
                random_state=seed,
            )
        m = XGBClassifier(
            n_estimators=10,
            max_depth=2,
            learning_rate=0.3,
            verbosity=0,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr, y_tr)
        y_prob = m.predict_proba(X_te)
        if task == "regression":
            m = XGBRegressor(
                n_estimators=10,
                max_depth=2,
                learning_rate=0.3,
                verbosity=0,
                random_state=seed,
            )
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            quality = float(r2_score(y_te, y_pred))
            if quality < 0.0:
                quality = 0.0
            perf = 1.0 - quality
        elif n_classes == 2:
            perf = 1.0 - float(roc_auc_score(y_te, y_prob[:, 1]))
        else:
            perf = 1.0 - float(accuracy_score(y_te, m.predict(X_te)))
        fitness[i] = [float(len(cols)), float(perf)]

        if verbose and (i + 1) % 50 == 0:
            print(f"  MOFS-SHAP subsamples: {i + 1}/{n_samples}", flush=True)

    # ------------------------------------------------------------------
    # 3.  Pareto-optimal elite front
    # ------------------------------------------------------------------
    n = n_samples
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                dominated[i] = True
                break
    elite_idx = np.where(~dominated)[0]

    if verbose:
        elite_feat_counts = [int(masks[i].sum()) for i in elite_idx]
        print(
            f"  Elite front: {len(elite_idx)}/{n_samples} subsets, "
            f"feat counts: {sorted(elite_feat_counts)[:10]}...",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 3b. Side-channel for HV / compression
    # ------------------------------------------------------------------
    if _result_box is not None and len(elite_idx) > 0:
        # Pad to 3D: (|F|, 1-AUC, 0) — MOFS-SHAP doesn't measure time
        elite_fit = fitness[elite_idx]
        _result_box.append({
            "fitness": np.column_stack([
                elite_fit[:, 0],
                elite_fit[:, 1],
                np.zeros(len(elite_idx)),
            ]),
            "masks": masks[elite_idx],
        })

    # ------------------------------------------------------------------
    # 4.  Score = gain × elite frequency
    # ------------------------------------------------------------------
    elite_freq = (
        masks[elite_idx].mean(axis=0) if len(elite_idx) > 0 else np.ones(n_features)
    )
    scores = mean_shap * elite_freq

    result = dict(zip(features, scores.astype(float)))
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

    _demo_banner("MOFS-SHAP baseline — synthetic demo")
    rng = np.random.default_rng(42)
    n = 200
    X_demo = pd.DataFrame(
        {
            "f0": rng.normal(0, 1, n),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
        },
    )
    y_demo = np.where(X_demo["f0"] + 0.1 * X_demo["f1"] > 0, 1, 0)

    scores = compute_mofs_shap_importance(X_demo, y_demo, n_samples=50, verbose=True)
    print("Feature importance ranking:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.6f}")
    print("\nObservation: f0 should score highest; ranking may differ from pure gain.")
