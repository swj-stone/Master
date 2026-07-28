"""Task 9 contracts for canonical modeling and ranking baselines."""

from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import FrozenInstanceError

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


def test_checkpoint_predictions_match_independent_boosting_prefixes() -> None:
    from minppi_mofs.modeling import (
        fit_predict,
        fit_predict_checkpoints,
    )

    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = (X["a"].to_numpy() + 0.2 * X["b"].to_numpy() > 0).astype(int)
    train = X.iloc[:60]
    evaluation = X.iloc[60:]
    params = {
        "n_estimators": 4,
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_jobs": 1,
    }

    checkpoints = fit_predict_checkpoints(
        train,
        y[:60],
        evaluation,
        task="binary",
        model_params=params,
        checkpoints=(2, 4),
        backend="cpu",
        seed=11,
    )
    independently_fitted = {
        rounds: fit_predict(
            train,
            y[:60],
            evaluation,
            task="binary",
            model_params={**params, "n_estimators": rounds},
            backend="cpu",
            seed=11,
        )
        for rounds in (2, 4)
    }

    assert tuple(checkpoints) == (2, 4)
    for rounds in checkpoints:
        assert np.allclose(
            checkpoints[rounds],
            independently_fitted[rounds],
            rtol=0.0,
            atol=1e-12,
        )


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
def test_legacy_none_coalition_utility_accepts_empty_feature_frame(
    module_name: str, function_name: str
) -> None:
    function = getattr(importlib.import_module(module_name), function_name)
    X = pd.DataFrame(index=range(20))

    result = function(
        X,
        np.array([0, 1] * 10),
        n_coalitions=2,
        per_coalition=1,
        model_params={"n_estimators": 2},
        backend="cpu",
        seed=8,
        coalition_utility=None,
    )

    assert result == {}


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("minppi_mofs.baselines.min_shap", "compute_minshap_importance"),
        ("minppi_mofs.baselines.banzhaf_fs", "compute_banzhaf_importance"),
    ],
)
def test_legacy_none_coalition_utility_accepts_unique_non_string_columns(
    module_name: str, function_name: str
) -> None:
    function = getattr(importlib.import_module(module_name), function_name)
    X = pd.DataFrame(
        {
            10: np.arange(20, dtype=float),
            20: np.arange(20, dtype=float) + 1,
        }
    )

    result = function(
        X,
        np.array([0, 1] * 10),
        n_coalitions=2,
        per_coalition=1,
        model_params={"n_estimators": 2},
        backend="cpu",
        seed=8,
        coalition_utility=None,
    )

    assert tuple(result) == (10, 20)
    assert np.isfinite(list(result.values())).all()


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("minppi_mofs.baselines.min_shap", "compute_minshap_importance"),
        ("minppi_mofs.baselines.banzhaf_fs", "compute_banzhaf_importance"),
    ],
)
def test_legacy_none_creates_sampling_rng_before_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    function_name: str,
) -> None:
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    events: list[str] = []
    original_default_rng = module.np.random.default_rng

    class RecordingLegacyEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            events.append("evaluator")

        def __call__(self, coalition: set[str]) -> float:
            return float(len(coalition))

    def recording_rng(*args, **kwargs):
        events.append("rng")
        return original_default_rng(*args, **kwargs)

    monkeypatch.setattr(module, "CoalitionUtility", RecordingLegacyEvaluator)
    monkeypatch.setattr(module.np.random, "default_rng", recording_rng)
    X = pd.DataFrame(
        {"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) + 1}
    )

    function(
        X,
        np.array([0, 1] * 10),
        n_coalitions=2,
        per_coalition=1,
        model_params={"n_estimators": 2},
        backend="cpu",
        seed=8,
        coalition_utility=None,
    )

    assert events[:2] == ["rng", "evaluator"]


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


class _SuppliedCoalitionUtility:
    def __init__(
        self,
        feature_order: tuple[str, ...],
        *,
        task: str = "binary",
        utility: str = "roc_auc",
        model_params: dict[str, object] | None = None,
        backend: str = "cpu",
        seed: int = 8,
        test_size: float = 0.2,
        max_samples: int = 5000,
    ) -> None:
        self.feature_order = feature_order
        self.task = task
        self.utility = utility
        self.model_params = model_params or {
            "n_estimators": 6,
            "n_jobs": 1,
        }
        self.backend = backend
        self.seed = seed
        self.test_size = test_size
        self.max_samples = max_samples
        self.coalitions: list[frozenset[str]] = []

    def __call__(self, coalition: set[str]) -> float:
        self.coalitions.append(frozenset(coalition))
        return float(len(coalition))


class _CanonicalRecordingEvaluator:
    def __init__(self, identity, prepared_split) -> None:
        from minppi_mofs.baselines._coalition_utility import (
            CoalitionUtilityDiagnostics,
        )

        self.identity = identity
        self.prepared_split = prepared_split
        self.coalitions: list[frozenset[str]] = []
        self._seen: set[tuple[str, ...]] = set()
        self._logical = 0
        self._physical = 0
        self._hits = 0
        self._diagnostics_type = CoalitionUtilityDiagnostics

    @property
    def diagnostics_snapshot(self):
        return self._diagnostics_type(
            logical_evaluation_count=self._logical,
            logical_unique_coalition_count=len(self._seen),
            physical_model_fit_count=self._physical,
            cache_hit_count=self._hits,
        )

    def __call__(self, coalition: set[str]) -> float:
        key = tuple(
            feature
            for feature in self.identity.ordered_features
            if feature in coalition
        )
        self.coalitions.append(frozenset(coalition))
        self._logical += 1
        if key in self._seen:
            self._hits += 1
        else:
            self._seen.add(key)
            self._physical += int(bool(key))
        return float(len(coalition))


def _canonical_recording_evaluator(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str = "binary",
    utility: str = "roc_auc",
    model_params: dict[str, object] | None = None,
    backend: str = "cpu",
    seed: int = 8,
):
    from minppi_mofs.baselines._coalition_utility import (
        _prepare_baseline_internal_split,
        make_baseline_utility_identity,
    )

    resolved_model_params = model_params or {
        "n_estimators": 6,
        "n_jobs": 1,
    }
    prepared = _prepare_baseline_internal_split(
        X, y, task=task, seed=seed
    )
    identity = make_baseline_utility_identity(
        X,
        y,
        prepared=prepared,
        task=task,
        utility=utility,
        model_params=resolved_model_params,
        backend=backend,
        seed=seed,
        store_identity_digest="b" * 64,
        library_versions={"numpy": np.__version__, "xgboost": "test"},
    )
    return _CanonicalRecordingEvaluator(identity, prepared)


def test_prepared_baseline_split_and_identity_are_frozen_authority() -> None:
    from minppi_mofs.baselines._coalition_utility import (
        BaselineUtilityIdentity,
        PreparedBaselineSplit,
        _prepare_baseline_internal_split,
        make_baseline_utility_identity,
    )
    from minppi_mofs.coalition_cache import (
        hash_numeric_array,
        hash_numeric_frame,
    )

    X = pd.DataFrame(
        {
            "a": np.arange(20, dtype=float),
            "b": np.arange(20, dtype=float) + 1,
        }
    )
    y = np.array([0, 1] * 10)
    prepared = _prepare_baseline_internal_split(
        X, y, task="binary", seed=8
    )
    identity = make_baseline_utility_identity(
        X,
        y,
        prepared=prepared,
        task="binary",
        utility="roc_auc",
        model_params={"n_estimators": 2},
        backend="cpu",
        seed=8,
        store_identity_digest="b" * 64,
        library_versions={"numpy": np.__version__, "xgboost": "test"},
    )

    assert isinstance(prepared, PreparedBaselineSplit)
    assert isinstance(identity, BaselineUtilityIdentity)
    assert identity.raw_features_sha256 == hash_numeric_frame(X)
    assert identity.raw_target_sha256 == hash_numeric_array(y)
    assert identity.train_features_sha256 == prepared.train_features_sha256
    assert identity.train_target_sha256 == prepared.train_target_sha256
    assert identity.eval_features_sha256 == prepared.eval_features_sha256
    assert identity.eval_target_sha256 == prepared.eval_target_sha256
    assert identity.ordered_features == ("a", "b")
    assert identity.namespace == "baseline_internal"
    assert identity.store_identity_digest == "b" * 64
    assert prepared.sample_indices == tuple(range(len(X)))
    assert prepared.train_positions == (
        10, 16, 2, 19, 9, 4, 0, 12, 15, 14, 5, 17, 18, 3, 7, 13
    )
    assert prepared.eval_positions == (8, 6, 1, 11)
    assert (
        prepared.train_features_sha256
        == "a076b3ef11c13661734e45a99d62bc9735538a89978890d73b0e35e02ca52bec"
    )
    assert (
        prepared.train_target_sha256
        == "1d495f874964c68fa32d64ed091e5ce9b7141f1b03a5779752a702954d06243a"
    )
    assert (
        prepared.eval_features_sha256
        == "6e3be03054ee77ba4881547979560238959a3a6a6c189112f8875b8121431c96"
    )
    assert (
        prepared.eval_target_sha256
        == "9f3b9cd5087cc361af1eca6b7ffb79fb81425e2088473334f5fdf81dbaedf3af"
    )
    with pytest.raises(FrozenInstanceError):
        identity.seed = 9
    with pytest.raises(ValueError, match="read-only"):
        prepared.y_train[0] = 99


def test_supplied_evaluator_requires_canonical_identity_and_diagnostics_before_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minppi_mofs.baselines import banzhaf_fs

    X = pd.DataFrame(
        {"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) + 1}
    )
    evaluator = _SuppliedCoalitionUtility(tuple(X.columns))
    monkeypatch.setattr(
        banzhaf_fs.np.random,
        "default_rng",
        lambda *_args, **_kwargs: pytest.fail(
            "sampling RNG was created before canonical protocol validation"
        ),
    )

    with pytest.raises(TypeError, match="identity|diagnostics|canonical|frozen"):
        banzhaf_fs.compute_banzhaf_importance(
            X,
            np.array([0, 1] * 10),
            n_coalitions=2,
            per_coalition=1,
            task="binary",
            utility="roc_auc",
            model_params={"n_estimators": 6, "n_jobs": 1},
            backend="cpu",
            seed=8,
            coalition_utility=evaluator,
        )
    assert not evaluator.coalitions


def test_supplied_evaluator_rejects_wrong_raw_data_before_sampling_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minppi_mofs.baselines import banzhaf_fs

    X = pd.DataFrame(
        {"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) + 1}
    )
    y = np.array([0, 1] * 10)
    evaluator = _canonical_recording_evaluator(X, y)
    changed = X.copy()
    changed.iloc[0, 0] += 0.5
    monkeypatch.setattr(
        banzhaf_fs.np.random,
        "default_rng",
        lambda *_args, **_kwargs: pytest.fail(
            "sampling RNG was created before raw-data validation"
        ),
    )

    with pytest.raises(ValueError, match="raw|authority|features"):
        banzhaf_fs.compute_banzhaf_importance(
            changed,
            y,
            n_coalitions=2,
            per_coalition=1,
            task="binary",
            utility="roc_auc",
            model_params={"n_estimators": 6, "n_jobs": 1},
            backend="cpu",
            seed=8,
            coalition_utility=evaluator,
        )
    assert not evaluator.coalitions


def test_supplied_evaluator_rejects_untrusted_diagnostics_before_sampling_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minppi_mofs.baselines import banzhaf_fs

    X = pd.DataFrame(
        {"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) + 1}
    )
    y = np.array([0, 1] * 10)
    canonical = _canonical_recording_evaluator(X, y)

    class NoDiagnostics:
        identity = canonical.identity
        prepared_split = canonical.prepared_split

        def __call__(self, coalition: set[str]) -> float:
            pytest.fail("untrusted evaluator was called")

    monkeypatch.setattr(
        banzhaf_fs.np.random,
        "default_rng",
        lambda *_args, **_kwargs: pytest.fail(
            "sampling RNG was created before diagnostics validation"
        ),
    )

    with pytest.raises(TypeError, match="diagnostics"):
        banzhaf_fs.compute_banzhaf_importance(
            X,
            y,
            n_coalitions=2,
            per_coalition=1,
            task="binary",
            utility="roc_auc",
            model_params={"n_estimators": 6, "n_jobs": 1},
            backend="cpu",
            seed=8,
            coalition_utility=NoDiagnostics(),
        )


def test_supplied_evaluator_rejects_duplicate_columns_before_sampling_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minppi_mofs.baselines import banzhaf_fs

    X = pd.DataFrame(
        {"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) + 1}
    )
    y = np.array([0, 1] * 10)
    evaluator = _canonical_recording_evaluator(X, y)
    duplicated = X.copy()
    duplicated.columns = ["a", "a"]
    monkeypatch.setattr(
        banzhaf_fs.np.random,
        "default_rng",
        lambda *_args, **_kwargs: pytest.fail(
            "sampling RNG was created before duplicate-column validation"
        ),
    )

    with pytest.raises(ValueError, match="unique|duplicate|universe"):
        banzhaf_fs.compute_banzhaf_importance(
            duplicated,
            y,
            n_coalitions=2,
            per_coalition=1,
            task="binary",
            utility="roc_auc",
            model_params={"n_estimators": 6, "n_jobs": 1},
            backend="cpu",
            seed=8,
            coalition_utility=evaluator,
        )
    assert not evaluator.coalitions


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("minppi_mofs.baselines.min_shap", "compute_minshap_importance"),
        ("minppi_mofs.baselines.banzhaf_fs", "compute_banzhaf_importance"),
    ],
)
def test_coalition_baselines_use_one_supplied_evaluator(
    module_name: str, function_name: str
) -> None:
    function = getattr(importlib.import_module(module_name), function_name)
    X = pd.DataFrame(
        {
            "z": np.arange(18, dtype=float),
            "a": np.arange(18, dtype=float) + 1,
            "q": np.arange(18, dtype=float) + 2,
        }
    )
    policy = {
        "task": "binary",
        "utility": "roc_auc",
        "model_params": {"n_estimators": 6, "n_jobs": 1},
        "backend": "cpu",
        "seed": 8,
    }
    y = np.array([0, 1] * 9)
    evaluator = _canonical_recording_evaluator(X, y, **policy)

    scores = function(
        X,
        y,
        n_coalitions=8,
        per_coalition=2,
        coalition_utility=evaluator,
        **policy,
    )

    assert tuple(scores) == tuple(X.columns)
    assert evaluator.coalitions
    assert evaluator.coalitions[0] == frozenset(X.columns)


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("minppi_mofs.baselines.min_shap", "compute_minshap_importance"),
        ("minppi_mofs.baselines.banzhaf_fs", "compute_banzhaf_importance"),
    ],
)
def test_supplied_evaluator_rejects_wrong_universe_before_sampling(
    monkeypatch: pytest.MonkeyPatch, module_name: str, function_name: str
) -> None:
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    X = pd.DataFrame(
        {
            "z": np.arange(18, dtype=float),
            "a": np.arange(18, dtype=float) + 1,
            "q": np.arange(18, dtype=float) + 2,
        }
    )
    evaluator = _canonical_recording_evaluator(
        X.loc[:, ["z", "a"]],
        np.array([0, 1] * 9),
        model_params={"n_estimators": 6, "n_jobs": 1},
    )
    monkeypatch.setattr(
        module.np.random,
        "default_rng",
        lambda *_args, **_kwargs: pytest.fail(
            "sampling RNG was created before evaluator-universe validation"
        ),
    )

    with pytest.raises(ValueError, match="feature|universe|order"):
        function(
            X,
            np.array([0, 1] * 9),
            n_coalitions=8,
            per_coalition=2,
            task="binary",
            utility="roc_auc",
            model_params={"n_estimators": 6, "n_jobs": 1},
            backend="cpu",
            seed=8,
            coalition_utility=evaluator,
        )
    assert not evaluator.coalitions


@pytest.mark.parametrize(
    ("field", "conflicting_value"),
    [
        ("task", "multiclass"),
        ("utility", "accuracy"),
        ("model_params", {"n_estimators": 7, "n_jobs": 1}),
        ("backend", "cuda"),
        ("seed", 9),
        ("test_size", 0.25),
        ("max_samples", 4999),
    ],
)
def test_supplied_evaluator_rejects_policy_conflict_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    conflicting_value: object,
) -> None:
    from minppi_mofs.baselines import banzhaf_fs

    X = pd.DataFrame(
        {
            "z": np.arange(18, dtype=float),
            "a": np.arange(18, dtype=float) + 1,
            "q": np.arange(18, dtype=float) + 2,
        }
    )
    from dataclasses import replace

    evaluator = _canonical_recording_evaluator(
        X,
        np.array([0, 1] * 9),
        model_params={"n_estimators": 6, "n_jobs": 1},
    )
    evaluator.identity = replace(
        evaluator.identity, **{field: conflicting_value}
    )
    monkeypatch.setattr(
        banzhaf_fs.np.random,
        "default_rng",
        lambda *_args, **_kwargs: pytest.fail(
            "sampling RNG was created before evaluator-policy validation"
        ),
    )

    with pytest.raises(ValueError, match=field):
        banzhaf_fs.compute_banzhaf_importance(
            X,
            np.array([0, 1] * 9),
            n_coalitions=8,
            per_coalition=2,
            task="binary",
            utility="roc_auc",
            model_params={"n_estimators": 6, "n_jobs": 1},
            backend="cpu",
            seed=8,
            coalition_utility=evaluator,
        )
    assert not evaluator.coalitions


def test_shared_cache_backed_baselines_match_legacy_and_reduce_real_fits(
    tmp_path: Path,
) -> None:
    from minppi_mofs.baselines._coalition_utility import (
        _prepare_baseline_internal_split,
        make_shared_baseline_utility,
    )
    from minppi_mofs.baselines.banzhaf_fs import compute_banzhaf_importance
    from minppi_mofs.baselines.min_shap import compute_minshap_importance
    from minppi_mofs.coalition_cache import (
        CACHE_SCHEMA_VERSION,
        CoalitionCacheIdentity,
        SQLiteCoalitionStore,
    )

    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        rng.normal(size=(60, 3)), columns=("signal", "noise_1", "noise_2")
    )
    y = (X["signal"].to_numpy() + 0.1 * rng.normal(size=len(X)) > 0).astype(int)
    policy = {
        "task": "binary",
        "utility": "roc_auc",
        "model_params": {
            "n_estimators": 2,
            "max_depth": 2,
            "learning_rate": 0.3,
            "n_jobs": 1,
        },
        "backend": "cpu",
        "seed": 8,
    }
    legacy_min_diag: dict[str, int] = {}
    legacy_banzhaf_diag: dict[str, int] = {}
    legacy_min = compute_minshap_importance(
        X,
        y,
        n_coalitions=8,
        per_coalition=2,
        diagnostics=legacy_min_diag,
        **policy,
    )
    legacy_banzhaf = compute_banzhaf_importance(
        X,
        y,
        n_coalitions=8,
        per_coalition=2,
        diagnostics=legacy_banzhaf_diag,
        **policy,
    )

    prepared = _prepare_baseline_internal_split(
        X,
        y,
        task="binary",
        seed=8,
    )
    identity = CoalitionCacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        protocol_hash="a" * 64,
        dataset_id="toy",
        seed=8,
        namespace="baseline_internal",
        train_features_sha256=prepared.train_features_sha256,
        train_target_sha256=prepared.train_target_sha256,
        eval_features_sha256=prepared.eval_features_sha256,
        eval_target_sha256=prepared.eval_target_sha256,
        ordered_features=tuple(X.columns),
        task="binary",
        class_labels=(0, 1),
        model_params=policy["model_params"],
        backend="cpu",
        model_seed=8,
        library_versions={"numpy": np.__version__, "xgboost": "test"},
    )
    shared_diag: dict[str, int] = {}
    shared_min_diag: dict[str, int] = {}
    shared_banzhaf_diag: dict[str, int] = {}
    with SQLiteCoalitionStore(
        tmp_path / "baseline.sqlite3",
        identity,
        commit_every_fits=2,
    ) as store:
        evaluator = make_shared_baseline_utility(
            X,
            y,
            store=store,
            diagnostics=shared_diag,
            prepared=prepared,
            **policy,
        )
        shared_min = compute_minshap_importance(
            X,
            y,
            n_coalitions=8,
            per_coalition=2,
            diagnostics=shared_min_diag,
            coalition_utility=evaluator,
            **policy,
        )
        shared_banzhaf = compute_banzhaf_importance(
            X,
            y,
            n_coalitions=8,
            per_coalition=2,
            diagnostics=shared_banzhaf_diag,
            coalition_utility=evaluator,
            **policy,
        )

    for feature in X.columns:
        assert shared_min[feature] == pytest.approx(
            legacy_min[feature], abs=1e-12
        )
        assert shared_banzhaf[feature] == pytest.approx(
            legacy_banzhaf[feature], abs=1e-12
        )
    assert shared_diag["physical_model_fit_count"] < (
        legacy_min_diag["model_fit_count"]
        + legacy_banzhaf_diag["model_fit_count"]
    )
    assert shared_banzhaf_diag["physical_model_fit_count"] == 0
    assert shared_banzhaf_diag["cache_hit_count"] == shared_banzhaf_diag[
        "logical_evaluation_count"
    ]
    assert shared_banzhaf_diag["model_fit_count"] > 0
    assert shared_diag["logical_evaluation_count"] > 0
    assert shared_diag["logical_unique_coalition_count"] > 0
    assert shared_diag["cache_hit_count"] > 0
    assert shared_diag["model_fit_count"] == shared_diag[
        "physical_model_fit_count"
    ]
    assert shared_diag["coalition_count"] == shared_diag[
        "logical_unique_coalition_count"
    ]
    assert (
        shared_diag["logical_evaluation_count"]
        == shared_diag["logical_unique_coalition_count"]
        + shared_diag["cache_hit_count"]
    )


def test_warm_reopened_shared_baseline_reports_logical_and_physical_costs(
    tmp_path: Path,
) -> None:
    from minppi_mofs.baselines._coalition_utility import (
        _prepare_baseline_internal_split,
        make_shared_baseline_utility,
    )
    from minppi_mofs.baselines.banzhaf_fs import compute_banzhaf_importance
    from minppi_mofs.coalition_cache import (
        CACHE_SCHEMA_VERSION,
        CoalitionCacheIdentity,
        SQLiteCoalitionStore,
    )

    X = pd.DataFrame(
        {"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float) + 1}
    )
    y = np.array([0, 1] * 10)
    policy = {
        "task": "binary",
        "utility": "roc_auc",
        "model_params": {"n_estimators": 2, "n_jobs": 1},
        "backend": "cpu",
        "seed": 8,
    }
    prepared = _prepare_baseline_internal_split(
        X, y, task="binary", seed=8
    )
    identity = CoalitionCacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        protocol_hash="c" * 64,
        dataset_id="toy",
        seed=8,
        namespace="baseline_internal",
        train_features_sha256=prepared.train_features_sha256,
        train_target_sha256=prepared.train_target_sha256,
        eval_features_sha256=prepared.eval_features_sha256,
        eval_target_sha256=prepared.eval_target_sha256,
        ordered_features=("a", "b"),
        task="binary",
        class_labels=(0, 1),
        model_params=policy["model_params"],
        backend="cpu",
        model_seed=8,
        library_versions={"numpy": np.__version__, "xgboost": "test"},
    )
    path = tmp_path / "warm.sqlite3"
    cold_method: dict[str, int] = {}
    with SQLiteCoalitionStore(
        path, identity, commit_every_fits=1
    ) as store:
        evaluator = make_shared_baseline_utility(
            X, y, store=store, diagnostics={}, prepared=prepared, **policy
        )
        compute_banzhaf_importance(
            X,
            y,
            n_coalitions=2,
            per_coalition=1,
            diagnostics=cold_method,
            coalition_utility=evaluator,
            **policy,
        )

    assert cold_method == {
        "coalition_count": 4,
        "model_fit_count": 3,
        "logical_evaluation_count": 5,
        "logical_unique_coalition_count": 4,
        "physical_model_fit_count": 3,
        "cache_hit_count": 1,
    }

    aggregate: dict[str, int] = {}
    method: dict[str, int] = {}
    with SQLiteCoalitionStore(
        path, identity, commit_every_fits=1
    ) as reopened:
        evaluator = make_shared_baseline_utility(
            X,
            y,
            store=reopened,
            diagnostics=aggregate,
            prepared=prepared,
            **policy,
        )
        compute_banzhaf_importance(
            X,
            y,
            n_coalitions=2,
            per_coalition=1,
            diagnostics=method,
            coalition_utility=evaluator,
            **policy,
        )

    assert method == {
        "coalition_count": 4,
        "model_fit_count": 3,
        "logical_evaluation_count": 5,
        "logical_unique_coalition_count": 4,
        "physical_model_fit_count": 0,
        "cache_hit_count": 5,
    }
    assert aggregate == method


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
