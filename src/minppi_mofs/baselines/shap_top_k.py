"""
SHAP-Top-k baseline for tabular feature importance.

The classic approach: train an XGBoost model on all features, extract SHAP
values via TreeExplainer, then rank features by mean absolute SHAP value.

This is the most widely-used explanation-based feature ranking and serves as
the primary direct-comparison baseline for all game-theoretic methods.

Usage::

    from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
    scores = compute_shap_topk_importance(X_train, y_train)

Run (stand-alone)::

    python src/minppi_mofs/baselines/shap_top_k.py
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import warnings

import numpy as np
import pandas as pd

from minppi_mofs.baselines._coalition_utility import default_utility
from minppi_mofs.modeling import make_model


def compute_shap_topk_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str = "binary",
    seed: int = 42,
    verbose: bool = False,
    utility: str | None = None,
    model_params: Mapping[str, object] | None = None,
    backend: str = "cpu",
    diagnostics: MutableMapping[str, int] | None = None,
) -> dict[str, float]:
    """Compute SHAP-based feature importance.

    Uses XGBoost gain importance (equivalent to mean |SHAP| for tree
    ensembles) to rank features.  Avoids shap-library numpy 2.x conflicts.

    Args:
        X: Feature DataFrame.
        y: Target array (binary, multiclass, or continuous).
        task: ``"binary"`` (default) for classification or ``"regression"``
            for continuous targets.
        seed: Random seed for XGBoost training.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: mean_shap}`` sorted descending.
    """
    # Sanitise column names for XGBoost (forbids [, ], <)
    safe_map = {c: f"f{i}" for i, c in enumerate(X.columns)}
    X_safe = X.rename(columns=safe_map)

    selected_utility = utility or default_utility(task)
    valid_utilities = (
        {"r2", "rmse"}
        if task == "regression"
        else {"roc_auc", "ovr_auc", "accuracy", "pr_auc"}
    )
    if selected_utility not in valid_utilities:
        raise ValueError("utility is invalid for the requested task")
    model = make_model(
        task=task,
        model_params=model_params
        or {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        backend=backend,
        seed=seed,
    )
    model.fit(X_safe, y)
    if diagnostics is not None:
        diagnostics["model_fit_count"] = 1
        diagnostics["coalition_count"] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        import shap

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_safe)

    if task == "regression":
        # XGBRegressor returns a single 2D (n_samples, n_features) array
        vals = np.abs(shap_vals).mean(axis=0)
    elif isinstance(shap_vals, list):
        per_class = []
        for s in shap_vals:
            s_abs = np.abs(s)
            if s_abs.ndim == 3:
                per_class.append(s_abs.mean(axis=0).mean(axis=-1))
            else:
                per_class.append(s_abs.mean(axis=0))
        vals = np.mean(per_class, axis=0)
    else:
        vals = np.abs(shap_vals).mean(axis=0)
        if vals.ndim > 1:  # multiclass 3D array
            vals = vals.mean(axis=-1)

    if verbose:
        print("  Mean |SHAP| extracted via TreeExplainer", flush=True)

    result = dict(zip(X.columns, vals.astype(float), strict=True))
    if tuple(result) != tuple(X.columns) or not all(np.isfinite(list(result.values()))):
        raise ValueError("SHAP returned incomplete or non-finite scores")
    return result


def rank_features(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Convert a score dict to a ranked list (highest first)."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _demo_banner(t: str) -> None:
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    _demo_banner("SHAP-Top-k baseline — synthetic demo")
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

    scores = compute_shap_topk_importance(X_demo, y_demo, verbose=True)
    print("Feature importance ranking:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.6f}")
    print("\nObservation: f0 should score highest.")
