"""SHAP-Top-k baseline for tabular feature importance.

Rank features by mean absolute, exact TreeSHAP contributions to the
XGBoost model's raw output, excluding its bias term.

Usage::

    from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
    scores = compute_shap_topk_importance(X_train, y_train)

Run (stand-alone)::

    ../venv/Scripts/python.exe src/minppi_mofs/baselines/shap_top_k.py
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from numbers import Integral

import numpy as np
import pandas as pd

from minppi_mofs.baselines._coalition_utility import default_utility
from minppi_mofs.modeling import make_model


DEFAULT_EXPLANATION_ROWS = 1024
_MAX_CONTRIBUTIONS_PER_BATCH = 4_000_000


def _explanation_sample(frame, seed, max_rows):
    if frame.empty:
        raise ValueError("TreeSHAP requires a nonempty training sample")
    if max_rows is None:
        return frame
    if isinstance(max_rows, bool) or not isinstance(max_rows, Integral) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer or None")
    if len(frame) <= max_rows:
        return frame
    positions = np.random.default_rng(seed).choice(len(frame), max_rows, replace=False)
    return frame.iloc[np.sort(positions)]


def fitted_tree_shap_importance(model, X_safe: pd.DataFrame, *, seed=42,
                                max_rows=DEFAULT_EXPLANATION_ROWS) -> np.ndarray:
    """Mean |TreeSHAP| from one fitted booster, over sampled training rows and model outputs."""
    from xgboost import DMatrix

    sample = _explanation_sample(X_safe, seed, max_rows)
    booster = model.get_booster()
    classes = getattr(model, "n_classes_", 1)
    groups = classes if classes > 2 else 1
    # XGBoost's float32 output includes a bias term in each output group.
    # Bound each contribution tensor to ~16 MB (or one row for very wide models).
    chunk_rows = max(1, _MAX_CONTRIBUTIONS_PER_BATCH // (groups * (sample.shape[1] + 1)))
    sums = np.zeros(sample.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(sample), chunk_rows):
        batch = sample.iloc[start:start + chunk_rows]
        values = booster.predict(DMatrix(batch, nthread=1), pred_contribs=True,
                                 approx_contribs=False, strict_shape=True)
        if values.ndim != 3 or values.shape[0] != len(batch) or values.shape[2] != len(sums) + 1:
            raise ValueError("Unexpected TreeSHAP dimensions for training features")
        sums += np.abs(values[:, :, :-1]).sum(axis=(0, 1), dtype=np.float64)
        count += len(batch) * values.shape[1]
    importance = sums / count
    if not np.isfinite(importance).all():
        raise ValueError("Non-finite TreeSHAP importance")
    return importance


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
    max_rows: int | None = DEFAULT_EXPLANATION_ROWS,
) -> dict[str, float]:
    """Compute SHAP-based feature importance.

    Trains on every supplied row and averages exact TreeSHAP contributions
    in absolute value over the explanation sample and the model's outputs.

    Args:
        X: Feature DataFrame.
        y: Target array (binary, multiclass, or continuous).
        task: ``"binary"``, ``"multiclass"`` or ``"regression"``.
        seed: Random seed for XGBoost training and uniform sampling without replacement.
        max_rows: Maximum training rows to explain. Pass None to explain all rows.
        verbose: Print progress.

    Returns:
        Dict ``{feature_name: mean_shap}`` in input column order.
    """
    # Sanitise column names for XGBoost (forbids [, ], <)
    safe_map = {c: f"f{i}" for i, c in enumerate(X.columns)}
    X_safe = X.rename(columns=safe_map)
    sample = _explanation_sample(X_safe, seed, max_rows)

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

    vals = fitted_tree_shap_importance(model, sample, max_rows=None)

    if verbose:
        print(f"  Mean |SHAP| over {len(sample)}/{len(X_safe)} training rows", flush=True)

    result = dict(zip(X.columns, vals.astype(float), strict=True))
    if tuple(result) != tuple(X.columns) or not all(np.isfinite(list(result.values()))):
        raise ValueError("SHAP returned incomplete or non-finite scores")
    return result


def rank_features(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Convert a score dict to a ranked list (highest first)."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def select(context):
    """Rank using exact TreeSHAP on the fitted reference for subsequent validation prefix sizing."""
    from minppi_mofs.baselines._selection_context import Selection

    sample = _explanation_sample(context.train, context.seed, DEFAULT_EXPLANATION_ROWS)
    names = context.full_model.get_booster().feature_names
    if names is None:
        names = [f"f{i}" for i in range(sample.shape[1])]
    sample = sample.set_axis(names, axis=1)
    context.progress("tree_shap", explained_rows=len(sample))
    scores = fitted_tree_shap_importance(context.full_model, sample, max_rows=None)
    return Selection(scores=scores, diagnostics=dict(
        explained_rows=len(sample), training_rows=len(context.train),
        max_explanation_rows=DEFAULT_EXPLANATION_ROWS,
        explanation_sampling="uniform without replacement on training rows",
        ranking="mean absolute exact raw-margin TreeSHAP across rows and outputs, excluding bias",
    ))


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
