"""Task 9 contracts for canonical modeling and ranking baselines."""

from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import experiments.icomp2026_stages as stages


class _Classifier:
    instances: list["_Classifier"] = []

    def __init__(self, **params: object) -> None:
        self.params = params
        self.fit_columns: tuple[str, ...] | None = None
        self.__class__.instances.append(self)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_Classifier":
        self.fit_columns = tuple(X.columns)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        count = len(self.classes_)
        return np.full((len(X), count), 1.0 / count)


class _Regressor:
    instances: list["_Regressor"] = []

    def __init__(self, **params: object) -> None:
        self.params = params
        self.fit_columns: tuple[str, ...] | None = None
        self.__class__.instances.append(self)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_Regressor":
        self.fit_columns = tuple(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.arange(len(X), dtype=float)


@pytest.fixture
def fake_xgboost(monkeypatch: pytest.MonkeyPatch):
    modeling = importlib.import_module("minppi_mofs.modeling")
    _Classifier.instances.clear()
    _Regressor.instances.clear()
    monkeypatch.setattr(modeling, "XGBClassifier", _Classifier)
    monkeypatch.setattr(modeling, "XGBRegressor", _Regressor)
    return modeling


def test_task9_public_model_api_exists() -> None:
    assert callable(stages.fit_predict)
    assert callable(stages.evaluate_fixed_subset)


@pytest.mark.parametrize(("task", "class_count"), [("binary", 2), ("multiclass", 3)])
def test_fit_predict_uses_classifier_probabilities_and_frozen_settings(
    fake_xgboost, task: str, class_count: int
) -> None:
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [3.0, 4.0, 5.0]})
    y = np.arange(len(X)) % class_count

    predictions = stages.fit_predict(
        X,
        y,
        X.iloc[:2],
        task=task,
        model_params={"n_estimators": 17, "max_depth": 3},
        backend="cuda",
        seed=91,
    )

    assert predictions.shape == (2, class_count)
    assert len(_Classifier.instances) == 1
    assert not _Regressor.instances
    assert _Classifier.instances[0].params == {
        "n_estimators": 17,
        "max_depth": 3,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 91,
    }


def test_fit_predict_uses_regressor_continuous_predictions(fake_xgboost) -> None:
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]})
    predictions = stages.fit_predict(
        X,
        np.array([0.5, 1.5]),
        X,
        task="regression",
        model_params={"n_estimators": 9},
        backend="cpu",
        seed=12,
    )
    assert predictions.tolist() == [0.0, 1.0]
    assert len(_Regressor.instances) == 1
    assert _Regressor.instances[0].params["device"] == "cpu"
    assert not _Classifier.instances


def test_fit_predict_sanitizes_xgboost_forbidden_feature_names(fake_xgboost) -> None:
    X = pd.DataFrame({"value[0]": [0.0, 1.0], "x<limit": [2.0, 3.0]})
    stages.fit_predict(
        X,
        np.array([0, 1]),
        X,
        task="binary",
        model_params={"n_estimators": 2},
        backend="cpu",
        seed=4,
    )
    assert _Classifier.instances[0].fit_columns == ("f0", "f1")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": "ranking", "model_params": {}, "backend": "cpu"},
        {"task": "binary", "model_params": {}, "backend": "tpu"},
        {"task": "binary", "model_params": {"device": "cuda"}, "backend": "cpu"},
        {"task": "binary", "model_params": {"tree_method": "approx"}, "backend": "cpu"},
        {"task": "binary", "model_params": {"random_state": 8}, "backend": "cpu"},
        {"task": "binary", "model_params": {"not_an_xgb_parameter": 8}, "backend": "cpu"},
    ],
)
def test_fit_predict_rejects_invalid_policy_before_fit(
    fake_xgboost, kwargs: dict[str, object]
) -> None:
    X = pd.DataFrame({"a": [0.0, 1.0]})
    with pytest.raises((TypeError, ValueError)):
        stages.fit_predict(X, np.array([0, 1]), X, seed=7, **kwargs)
    assert not _Classifier.instances
    assert not _Regressor.instances


@pytest.mark.parametrize(
    ("task", "utility", "y", "expected_shape"),
    [
        ("binary", "roc_auc", np.array([0, 1, 1]), (2, 2)),
        ("multiclass", "accuracy", np.array([0, 1, 2]), (2, 3)),
        ("regression", "rmse", np.array([2.0, 4.0, 6.0]), (2,)),
    ],
)
def test_empty_subset_uses_training_prior_without_model_fit(
    monkeypatch: pytest.MonkeyPatch,
    fake_xgboost,
    task: str,
    utility: str,
    y: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    X_train = pd.DataFrame(index=range(3))
    X_eval = pd.DataFrame(index=range(2))
    seen: dict[str, object] = {}

    def fake_utility(y_true: np.ndarray, predictions: np.ndarray, *, metric: str) -> float:
        seen.update(predictions=predictions.copy(), metric=metric)
        return 0.625

    monkeypatch.setattr(fake_xgboost, "compute_utility", fake_utility)
    score = stages.evaluate_fixed_subset(
        X_train,
        y,
        X_eval,
        y[:2],
        (),
        task=task,
        utility=utility,
        model_params={"n_estimators": 4},
        backend="cpu",
        seed=3,
    )
    assert score == 0.625
    assert seen["metric"] == utility
    assert np.asarray(seen["predictions"]).shape == expected_shape
    assert not _Classifier.instances
    assert not _Regressor.instances


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("minppi_mofs.baselines.min_shap", "compute_minshap_importance"),
        ("minppi_mofs.baselines.banzhaf_fs", "compute_banzhaf_importance"),
    ],
)
def test_baseline_scores_are_hash_seed_independent(
    module_name: str, function_name: str
) -> None:
    script = f"""
import json
import numpy as np
import pandas as pd
from {module_name} import {function_name}
X = pd.DataFrame({{name: np.arange(30, dtype=float) + i for i, name in enumerate(['z','a','q'])}})
y = np.array([0, 1] * 15)
print(json.dumps({function_name}(X, y, n_coalitions=8, seed=5, model_params={{'n_estimators': 2}}, backend='cpu'), sort_keys=True))
"""
    root = Path(__file__).parents[1]
    outputs = []
    for hash_seed in ("11", "97"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=root, env=env,
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(completed.stdout.strip()))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("minppi_mofs.baselines.min_shap", "compute_minshap_importance"),
        ("minppi_mofs.baselines.banzhaf_fs", "compute_banzhaf_importance"),
    ],
)
def test_coalition_baselines_propagate_multiclass_policy_and_keep_feature_order(
    monkeypatch: pytest.MonkeyPatch, module_name: str, function_name: str
) -> None:
    coalition_module = importlib.import_module(
        "minppi_mofs.baselines._coalition_utility"
    )
    seen: list[dict[str, object]] = []

    def fake_evaluate(*args, **kwargs) -> float:
        seen.append(kwargs)
        return 0.75

    monkeypatch.setattr(coalition_module, "evaluate_fixed_subset", fake_evaluate)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    X = pd.DataFrame({"z": np.arange(18), "a": np.arange(18) + 1, "q": np.arange(18) + 2})
    diagnostics: dict[str, int] = {}
    scores = function(
        X,
        np.array([0, 1, 2] * 6),
        task="multiclass",
        utility="ovr_auc",
        model_params={"n_estimators": 6, "n_jobs": 1},
        backend="cuda",
        n_coalitions=3,
        seed=8,
        diagnostics=diagnostics,
    )
    assert tuple(scores) == ("z", "a", "q")
    assert seen
    assert all(item["task"] == "multiclass" for item in seen)
    assert all(item["utility"] == "ovr_auc" for item in seen)
    assert all(item["model_params"] == {"n_estimators": 6, "n_jobs": 1} for item in seen)
    assert all(item["backend"] == "cuda" for item in seen)
    assert diagnostics["coalition_count"] >= 1


def test_shap_baseline_uses_canonical_multiclass_model_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("minppi_mofs.baselines.shap_top_k")
    seen: dict[str, object] = {}

    class FakeModel:
        def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
            seen["fit_columns"] = tuple(X.columns)

    def fake_make_model(**kwargs):
        seen.update(kwargs)
        return FakeModel()

    class FakeExplainer:
        def __init__(self, model: object) -> None:
            pass

        def shap_values(self, X: pd.DataFrame) -> np.ndarray:
            return np.ones((len(X), X.shape[1], 3))

    monkeypatch.setattr(module, "make_model", fake_make_model)
    monkeypatch.setitem(sys.modules, "shap", type("FakeShap", (), {"TreeExplainer": FakeExplainer}))
    X = pd.DataFrame({"unsafe[0]": np.arange(9), "b": np.arange(9) + 1})
    diagnostics: dict[str, int] = {}
    scores = module.compute_shap_topk_importance(
        X,
        np.array([0, 1, 2] * 3),
        task="multiclass",
        utility="ovr_auc",
        model_params={"n_estimators": 5},
        backend="cuda",
        seed=22,
        diagnostics=diagnostics,
    )
    assert tuple(scores) == tuple(X.columns)
    assert seen == {
        "task": "multiclass",
        "model_params": {"n_estimators": 5},
        "backend": "cuda",
        "seed": 22,
        "fit_columns": ("f0", "f1"),
    }
    assert diagnostics == {"model_fit_count": 1, "coalition_count": 1}
