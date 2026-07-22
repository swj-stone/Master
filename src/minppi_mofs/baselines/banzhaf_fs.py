"""Banzhaf feature importance with deterministic shared-coalition sampling."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import numpy as np
import pandas as pd

from minppi_mofs.baselines._coalition_utility import CoalitionUtility, default_utility


def compute_banzhaf_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_coalitions: int = 200,
    per_coalition: int = 5,
    seed: int = 42,
    verbose: bool = False,
    task: str = "binary",
    utility: str | None = None,
    model_params: Mapping[str, object] | None = None,
    backend: str = "cpu",
    diagnostics: MutableMapping[str, int] | None = None,
) -> dict[str, float]:
    """Return mean utility marginals in original feature order."""
    if isinstance(n_coalitions, bool) or not isinstance(n_coalitions, int) or n_coalitions < 1:
        raise ValueError("n_coalitions must be a positive integer")
    if isinstance(per_coalition, bool) or not isinstance(per_coalition, int) or per_coalition < 1:
        raise ValueError("per_coalition must be a positive integer")
    rng = np.random.default_rng(seed)
    features = tuple(X.columns)
    evaluator = CoalitionUtility(
        X,
        y,
        task=task,
        utility=utility or default_utility(task),
        model_params=model_params or {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.3},
        backend=backend,
        seed=seed,
        diagnostics=diagnostics,
    )
    evaluator(set(features))
    totals = {feature: 0.0 for feature in features}
    counts = {feature: 0 for feature in features}
    for index in range(n_coalitions):
        size = int(rng.integers(0, max(1, len(features) // 3) + 1))
        coalition = set(rng.choice(features, size=size, replace=False).tolist())
        base = evaluator(coalition)
        outside = [feature for feature in features if feature not in coalition]
        rng.shuffle(outside)
        for feature in outside[:per_coalition]:
            totals[feature] += evaluator(coalition | {feature}) - base
            counts[feature] += 1
        if all(count >= 2 for count in counts.values()) and index >= 20:
            break
    result = {
        feature: float(totals[feature] / max(counts[feature], 1))
        for feature in features
    }
    if set(result) != set(features) or not all(np.isfinite(list(result.values()))):
        raise ValueError("Banzhaf returned incomplete or non-finite scores")
    if verbose:
        print(f"  Banzhaf evaluated {len(features)} features", flush=True)
    return result


def rank_features(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Rank features by descending score, preserving input order for ties."""
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
