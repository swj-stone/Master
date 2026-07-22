"""
Quick-test: NSGA-II baseline only — single dataset, single seed.

Use before launching the full ``run_main_table.py`` (which takes 4-5 h)
to verify that the NSGA-II baseline works correctly on real data.

Run::

    python -u tests/test_nsga2_quick.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import minppi_mofs.baselines.nsga2_fs as nsga2_module
import minppi_mofs.modeling as modeling_module
from minppi_mofs.baselines.nsga2_fs import (
    _evaluate_individual,
    nsga2_feature_selection,
    nsga2_ranking,
)

# ── config ──────────────────────────────────────────────────────────
DATASET: tuple[str, str, str] = ("spambase", "is_spam", "binary")
SEED: int = 42
POP_SIZE: int = 10
MAX_GEN: int = 10
# ─────────────────────────────────────────────────────────────────────

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def test_individual_uses_explicit_utility_and_backend_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, object]] = []
    utilities: list[str] = []

    class FakeClassifier:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

        def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
            assert list(X.columns) == ["f0"]

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.tile([0.25, 0.75], (len(X), 1))

    monkeypatch.setattr(modeling_module, "XGBClassifier", FakeClassifier)
    monkeypatch.setattr(
        modeling_module,
        "compute_utility",
        lambda y_true, predictions, *, metric: utilities.append(metric) or 0.7,
    )
    X_train = pd.DataFrame({"signal": [0.0, 1.0, 0.0, 1.0]})
    X_val = pd.DataFrame({"signal": [0.0, 1.0]})

    objective = _evaluate_individual(
        np.array([1]),
        X_train,
        np.array([0, 1, 0, 1]),
        X_val,
        np.array([0, 1]),
        task="binary",
        utility="pr_auc",
        model_params={
            "n_estimators": 3,
            "max_depth": 2,
            "learning_rate": 0.2,
            "n_jobs": 1,
        },
        backend="cuda",
        seed=17,
    )

    assert objective == pytest.approx((1.0, 0.3))
    assert utilities == ["pr_auc"]
    assert constructed == [
        {
            "n_estimators": 3,
            "max_depth": 2,
            "learning_rate": 0.2,
            "n_jobs": 1,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": 17,
        }
    ]


def test_individual_selects_regressor_family_only_from_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, object]] = []
    utilities: list[str] = []

    class FakeRegressor:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

        def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
            assert list(X.columns) == ["f0"]

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X))

    monkeypatch.setattr(modeling_module, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        modeling_module,
        "XGBClassifier",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("regression task constructed a classifier")
        ),
    )
    monkeypatch.setattr(
        modeling_module,
        "compute_utility",
        lambda y_true, predictions, *, metric: utilities.append(metric) or 0.6,
    )
    X_train = pd.DataFrame({"signal": [0.0, 1.0, 2.0, 3.0]})
    X_val = pd.DataFrame({"signal": [4.0, 5.0]})

    objective = _evaluate_individual(
        np.array([1]),
        X_train,
        np.array([0.0, 1.0, 2.0, 3.0]),
        X_val,
        np.array([4.0, 5.0]),
        task="regression",
        utility="rmse",
        model_params={
            "n_estimators": 3,
            "max_depth": 2,
            "learning_rate": 0.2,
            "n_jobs": 1,
        },
        backend="cpu",
        seed=17,
    )

    assert objective == pytest.approx((1.0, 0.4))
    assert utilities == ["rmse"]
    assert constructed[0]["device"] == "cpu"
    assert constructed[0]["random_state"] == 17


def test_prefiltered_cardinality_uses_original_feature_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nsga2_module, "evaluate_fixed_subset", lambda *args, **kwargs: 0.75)
    X_train = pd.DataFrame({"f0": [0.0, 1.0], "f2": [1.0, 0.0]})

    objective = _evaluate_individual(
        np.array([1, 0]),
        X_train,
        np.array([0, 1]),
        X_train,
        np.array([0, 1]),
        task="binary",
        utility="roc_auc",
        model_params={"n_estimators": 1, "max_depth": 1, "learning_rate": 0.3, "n_jobs": 1},
        backend="cpu",
        seed=42,
        cardinality_denominator=4,
    )

    assert objective == pytest.approx((0.25, 0.25))


def main() -> None:
    prefix, target, task = DATASET

    train = pd.read_csv(PROCESSED / f"{prefix}_s{SEED}_train.csv")
    test = pd.read_csv(PROCESSED / f"{prefix}_s{SEED}_test.csv")
    X_tr = train.drop(columns=[target])
    y_tr = train[target].values
    X_te = test.drop(columns=[target])
    y_te = test[target].values

    print(f"Dataset: {prefix}  ({X_tr.shape[1]} features, {X_tr.shape[0]} samples)")
    print(f"Task: {task}")
    print(f"NSGA-II: pop={POP_SIZE}, gen={MAX_GEN}, seed={SEED}")
    print()

    # ── Full NSGA-II ────────────────────────────────────────────────
    t0 = time.perf_counter()
    result = nsga2_feature_selection(
        X_tr, y_tr, X_te, y_te,
        task=task, seed=SEED,
        pop_size=POP_SIZE, max_gen=MAX_GEN,
        verbose=True,
    )
    t1 = time.perf_counter() - t0
    print(f"\nDone: {t1:.1f}s")
    print(f"  HV       = {result.hv:.4f}")
    print(f"  Front    = {len(result.pareto_front)} non-dominated solutions")
    print(f"  Converged at gen {result.converged_gen}")
    print(f"  HV curve = {[round(h, 4) for h in result.hv_history]}")

    # ── Ranking adapter ─────────────────────────────────────────────
    t0 = time.perf_counter()
    ranking = nsga2_ranking(
        X_tr, y_tr, task=task, seed=SEED,
        pop_size=POP_SIZE, max_gen=MAX_GEN, verbose=False,
    )
    t2 = time.perf_counter() - t0
    print(f"\nRanking adapter: {t2:.1f}s")
    print("  Top-10 features:")
    for i, (feat, val) in enumerate(ranking[:10], 1):
        print(f"    {i:2d}. {feat:30s}  {val:.6f}")

    print("\nAll OK — ready for run_main_table.py.")


if __name__ == "__main__":
    main()
