"""
Powershap baseline for tabular feature importance.

Powershap (Verhaeghe et al. 2022, *ECML PKDD*) identifies informative features
by comparing their SHAP value magnitudes against those of randomly permuted
"shadow" features via a statistical power test.  This is a thin wrapper around
the official ``powershap`` PyPI package (v0.1.0.1) that exposes continuous
importance scores rather than the boolean selection mask provided by
``sklearn.feature_selection.SelectorMixin``.

Usage:

    from minppi_mofs.baselines.powershap import compute_powershap_importance
    scores = compute_powershap_importance(X_train, y_train, power_iterations=20)
    # scores: dict {feature_name: score}  (sorted descending)

Run (stand-alone):

    python src/minppi_mofs/baselines/powershap.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from powershap import PowerShap
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def compute_powershap_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    power_iterations: int = 20,
    alpha: float = 0.01,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute Powershap feature importance using the official PyPI package.

    Args:
        X: Feature DataFrame.
        y: Target array.
        power_iterations: Number of shadow-feature shuffles.
        alpha: Significance threshold for the statistical test.
        seed: Random seed (passed to CatBoost internally via
            ``random_seed``-aware model re-creation in PowerShap).
        verbose: If True, enable PowerShap console output.

    Returns:
        Dict ``{feature_name: importance_score}`` sorted descending.
        The score is the mean absolute SHAP value (``impact`` column from
        the internal ``_processed_shaps_df``).
    """
    ps = PowerShap(
        power_iterations=power_iterations,
        power_alpha=alpha,
        include_all=True,
        automatic=False,
        show_progress=verbose,
        verbose=verbose,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ps.fit(X, y)

    # Extract continuous importance from internal statistics
    proc_df: pd.DataFrame = ps._processed_shaps_df  # noqa: SLF001
    feature_names = list(X.columns)
    result: dict[str, float] = {}
    for name in feature_names:
        if name in proc_df.index:
            result[name] = float(proc_df.loc[name, "impact"])
        else:
            result[name] = 0.0

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

    _demo_banner("Powershap baseline — synthetic demo")
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

    scores = compute_powershap_importance(
        X_demo,
        y_demo,
        power_iterations=5,
        seed=42,
        verbose=False,
    )
    print("Feature importance ranking:")
    for feat, val in scores.items():
        print(f"  {feat}: {val:.6f}")
    print("\nObservation: f0 should score highest.")
