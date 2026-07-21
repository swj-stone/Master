"""
Unit tests for ``minppi_mofs.utility`` — pure utility functions.

Verify all classification and regression metrics with known inputs.
Run::

    python -m pytest tests/test_utility.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from minppi_mofs.utility import compute_utility, is_classification_metric


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestRocAuc:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        assert compute_utility(y_true, y_proba, metric="roc_auc") == pytest.approx(1.0)

    def test_random_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=float)
        assert compute_utility(y_true, y_proba, metric="roc_auc") == pytest.approx(0.5, abs=0.1)

    def test_single_class_returns_half(self) -> None:
        y_true = np.array([0, 0, 0])
        y_proba = np.array([[1, 0], [1, 0], [1, 0]], dtype=float)
        assert compute_utility(y_true, y_proba, metric="roc_auc") == pytest.approx(0.5)


class TestOvrAuc:
    def test_perfect_multiclass(self) -> None:
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        assert compute_utility(y_true, y_proba, metric="ovr_auc") == pytest.approx(1.0)

    def test_single_class_returns_half(self) -> None:
        y_true = np.array([0, 0, 0])
        y_proba = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        assert compute_utility(y_true, y_proba, metric="ovr_auc") == pytest.approx(0.5)

    def test_binary_ovr_auc_on_two_column_proba(self) -> None:
        """Binary task with ovr_auc metric: XGBoost outputs 2-column proba
        but scikit-learn `roc_auc_score(multi_class='ovr')` wants 1d.
        The n_cls==2 shortcut delegates to binary ROC-AUC."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        assert compute_utility(y_true, y_proba, metric="ovr_auc") == pytest.approx(1.0)


class TestAccuracy:
    def test_perfect(self) -> None:
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        assert compute_utility(y_true, y_proba, metric="accuracy") == pytest.approx(1.0)

    def test_half_correct(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([
            [1, 0],
            [1, 0],  # wrong
            [1, 0],
            [1, 0],  # wrong
        ], dtype=float)
        assert compute_utility(y_true, y_proba, metric="accuracy") == pytest.approx(0.5)


class TestPrAuc:
    def test_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        assert compute_utility(y_true, y_proba, metric="pr_auc") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestR2:
    def test_perfect(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert compute_utility(y_true, y_pred, metric="r2") == pytest.approx(1.0)

    def test_mean_baseline(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # always predicts mean
        assert compute_utility(y_true, y_pred, metric="r2") == pytest.approx(0.0)

    def test_worse_than_mean_clamped_to_zero(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([100.0, -50.0, 0.0])
        assert compute_utility(y_true, y_pred, metric="r2") == pytest.approx(0.0)


class TestRmse:
    def test_perfect(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        # rmse=0 → 1/(1+0) = 1.0
        assert compute_utility(y_true, y_pred, metric="rmse") == pytest.approx(1.0)

    def test_rmse_one(self) -> None:
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, -1.0])
        # rmse = sqrt((1+1)/2) = 1 → 1/(1+1) = 0.5
        assert compute_utility(y_true, y_pred, metric="rmse") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestUnknownMetric:
    def test_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            compute_utility(np.array([0, 1]), np.array([[1, 0], [0, 1]]), metric="f1")


class TestIsClassificationMetric:
    def test_classification_metrics(self) -> None:
        assert is_classification_metric("roc_auc")
        assert is_classification_metric("ovr_auc")
        assert is_classification_metric("accuracy")
        assert is_classification_metric("pr_auc")

    def test_regression_metrics(self) -> None:
        assert not is_classification_metric("r2")
        assert not is_classification_metric("rmse")


_X = None  # lazy fixture
_y_cls = None
_y_reg = None


def _make_data():
    global _X, _y_cls, _y_reg
    if _X is not None:
        return _X, _y_cls, _y_reg
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 20
    _X = pd.DataFrame({"f": rng.normal(0, 1, n)})
    _y_cls = rng.integers(0, 2, n)
    _y_reg = rng.normal(0, 1, n)
    return _X, _y_cls, _y_reg


def _cache(**kw):
    from minppi_mofs.voting_game import AccFullCache
    X, y_cls, y_reg = _make_data()
    task = kw.get("task", "binary")
    y = y_reg if task == "regression" else y_cls
    return AccFullCache(X, X, y, y, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# AccFullCache: task × utility validation (bidirectional)
# ═══════════════════════════════════════════════════════════════════════════


class TestAccFullCacheValidatesTask:
    """task must be one of {binary, multiclass, regression}."""

    def test_invalid_task_raises(self) -> None:
        with pytest.raises(ValueError, match="task"):
            _cache(task="poisson")

    def test_empty_task_raises(self) -> None:
        with pytest.raises(ValueError, match="task"):
            _cache(task="")

    def test_none_task_raises(self) -> None:
        with pytest.raises(ValueError, match="task"):
            _cache(task=None)


class TestAccFullCacheRejectsClassificationTaskWithRegressionUtility:
    """binary/multiclass + r2/rmse → ValueError *before* model training."""

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    @pytest.mark.parametrize("utility", ["r2", "rmse"])
    def test_raises(self, task: str, utility: str) -> None:
        with pytest.raises(ValueError, match="regression-only|classification-only"):
            _cache(task=task, utility=utility)


class TestAccFullCacheRejectsRegressionTaskWithClassificationUtility:
    """regression + roc_auc/ovr_auc/accuracy/pr_auc → ValueError."""

    @pytest.mark.parametrize("utility", ["roc_auc", "ovr_auc", "accuracy", "pr_auc"])
    def test_raises(self, utility: str) -> None:
        with pytest.raises(ValueError, match="classification-only"):
            _cache(task="regression", utility=utility)


class TestAccFullCacheAcceptsValidTaskUtilityPairs:
    """All valid (task, utility) combinations must construct without error."""

    @pytest.mark.parametrize("task, utility", [
        ("binary", "roc_auc"),
        ("binary", "pr_auc"),
        ("binary", "accuracy"),
        ("binary", "ovr_auc"),
        ("multiclass", "ovr_auc"),
        ("multiclass", "accuracy"),
        ("multiclass", "roc_auc"),
        ("regression", "r2"),
        ("regression", "rmse"),
    ])
    def test_accepts(self, task: str, utility: str) -> None:
        c = _cache(task=task, utility=utility)
        assert c._task == task
        assert c._utility == utility


# ═══════════════════════════════════════════════════════════════════════════
# UtilityView: same validation
# ═══════════════════════════════════════════════════════════════════════════


class TestUtilityViewValidatesTaskAndUtility:
    """UtilityView inherits the task from its parent cache and must also
    reject impossible (task, utility) combinations."""

    def test_view_rejects_classification_cache_with_regression_utility(self) -> None:
        from minppi_mofs.voting_game import UtilityView
        cache = _cache(task="binary", utility="roc_auc")
        with pytest.raises(ValueError, match="regression-only|classification-only"):
            UtilityView(cache, utility="r2", alpha=0.95)

    def test_view_rejects_regression_cache_with_classification_utility(self) -> None:
        from minppi_mofs.voting_game import UtilityView
        cache = _cache(task="regression", utility="r2")
        with pytest.raises(ValueError, match="regression-only|classification-only"):
            UtilityView(cache, utility="roc_auc", alpha=0.95)

    def test_view_rejects_unknown_utility(self) -> None:
        from minppi_mofs.voting_game import UtilityView
        cache = _cache(task="binary")
        with pytest.raises(ValueError, match="Unknown utility"):
            UtilityView(cache, utility="f1", alpha=0.95)

    @pytest.mark.parametrize("task, utility", [
        ("binary", "accuracy"),
        ("binary", "pr_auc"),
        ("multiclass", "ovr_auc"),
        ("regression", "r2"),
    ])
    def test_view_accepts_valid_pairs(self, task: str, utility: str) -> None:
        from minppi_mofs.voting_game import UtilityView
        default = "roc_auc" if task != "regression" else "r2"
        cache = _cache(task=task, utility=default)
        v = UtilityView(cache, utility=utility, alpha=0.95)
        assert v._utility == utility


# ═══════════════════════════════════════════════════════════════════════════
# Structural guarantee: one fit + one predict_proba per coalition
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictionCacheIsShared:
    """Two UtilityViews reading the same coalition MUST get identical
    numpy arrays and must NOT trigger a second model training."""

    def test_two_views_receive_same_prediction_array(self) -> None:
        from minppi_mofs.voting_game import AccFullCache, UtilityView
        import pandas as pd
        rng = np.random.default_rng(7)
        n = 50
        X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        y = rng.integers(0, 3, n)

        cache = AccFullCache(X, X, y, y, task="multiclass")
        n_after_ctor = len(cache._preds)  # full-feature coalition trained by __init__

        view_acc = UtilityView(cache, utility="accuracy", alpha=0.95)
        view_auc = UtilityView(cache, utility="ovr_auc", alpha=0.95)

        S = {"a"}
        _va = view_acc.evaluate(S)
        _vo = view_auc.evaluate(S)

        key = tuple(sorted(S))
        assert key in cache._preds, "coalition should be cached after first view"
        # Construction trained the full set; our S adds exactly 1 new coalition
        assert len(cache._preds) == n_after_ctor + 1, (
            f"expected {n_after_ctor + 1} coalitions (ctor full-set + {{'a'}}), "
            f"got {len(cache._preds)}"
        )

        # Both views read the identical numpy array
        preds_acc = cache.get_predictions(S)
        preds_auc = cache.get_predictions(S)
        assert preds_acc is preds_auc, (
            "get_predictions returned different objects — predictions not shared"
        )

    def test_fit_called_once_per_coalition_across_views(self) -> None:
        """Fit must be called exactly once per coalition even when two
        UtilityViews evaluate it under different metrics."""
        from unittest import mock
        from minppi_mofs.voting_game import AccFullCache, UtilityView
        import pandas as pd
        rng = np.random.default_rng(8)
        n = 40
        X = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "z": rng.normal(0, 1, n),
        })
        y = rng.integers(0, 2, n)

        cache = AccFullCache(X, X, y, y, task="binary")
        view_acc = UtilityView(cache, utility="accuracy", alpha=0.95)
        view_auc = UtilityView(cache, utility="roc_auc", alpha=0.95)

        # Spy after construction so we only count our test coalition
        with mock.patch.object(cache._model, "fit", wraps=cache._model.fit) as spy_fit:
            S = {"x", "z"}
            view_acc.evaluate(S)
            view_auc.evaluate(S)

            # Model fitted exactly once for this coalition
            assert spy_fit.call_count == 1, (
                f"fit() called {spy_fit.call_count} times for {{'x','z'}} "
                f"across two views — expected 1"
            )
        # Cache has only the ctor's full-set entry plus our one coalition
        # (second view was a cache hit — no new entry)
        assert len(cache._preds) == 2, (
            f"expected 2 entries (full-set + {{'x','z'}}), got {len(cache._preds)}"
        )


class TestBinaryOvrAucEndToEnd:
    """binary + ovr_auc must work from cache construction through evaluation."""

    def test_binary_ovr_auc_full_pipeline(self) -> None:
        from minppi_mofs.voting_game import AccFullCache, UtilityView
        import pandas as pd
        rng = np.random.default_rng(9)
        n = 40
        X = pd.DataFrame({"f": rng.normal(0, 1, n)})
        y = (X["f"].values > 0).astype(int)

        cache = AccFullCache(X, X, y, y, task="binary", utility="roc_auc")
        view = UtilityView(cache, utility="ovr_auc", alpha=0.95)

        u_full = view.acc_full
        u_subset = view.evaluate({"f"})

        assert 0.0 <= u_full <= 1.0
        assert 0.0 <= u_subset <= 1.0
        # Full-feature should have reasonable AUC on training data
        assert u_full > 0.5, f"ovr_auc on binary full-feature too low: {u_full:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# Empty-coalition baselines per metric
# ═══════════════════════════════════════════════════════════════════════════


class TestEmptyCoalitionBaselines:
    """Empty coalition returns training-set class priors, producing
    metric-appropriate baselines (not a hard-coded 0.5)."""

    def test_multiclass_accuracy_equals_majority_class_proportion(self) -> None:
        import pandas as pd
        from minppi_mofs.voting_game import AccFullCache
        rng = np.random.default_rng(10)
        n = 100
        X = pd.DataFrame({"f": rng.normal(0, 1, n)})
        # 3 classes: 50, 30, 20
        y = np.concatenate([
            np.zeros(50, dtype=int),
            np.ones(30, dtype=int),
            np.full(20, 2, dtype=int),
        ])
        cache = AccFullCache(X, X, y, y, task="multiclass")

        acc = cache.evaluate(set())
        # majority class is 0 with proportion 0.50
        assert acc == pytest.approx(0.50, abs=0.02), (
            f"multiclass empty-coalition accuracy={acc:.4f}, expected ~0.50"
        )

    def test_binary_pr_auc_equals_positive_class_prior(self) -> None:
        import pandas as pd
        from minppi_mofs.voting_game import AccFullCache
        rng = np.random.default_rng(11)
        n = 200
        X = pd.DataFrame({"f": rng.normal(0, 1, n)})
        y = np.concatenate([np.zeros(160, dtype=int),
                            np.ones(40, dtype=int)])
        cache = AccFullCache(X, X, y, y, task="binary")

        pr = cache.evaluate(set(), utility="pr_auc")
        # PR-AUC of constant-prior predictor = positive class prevalence
        assert pr == pytest.approx(0.20, abs=0.02), (
            f"binary PR-AUC empty-coalition={pr:.4f}, expected ~0.20"
        )

    def test_binary_accuracy_equals_majority_prior(self) -> None:
        import pandas as pd
        from minppi_mofs.voting_game import AccFullCache
        rng = np.random.default_rng(12)
        n = 100
        X = pd.DataFrame({"f": rng.normal(0, 1, n)})
        y = np.concatenate([np.zeros(70, dtype=int),
                            np.ones(30, dtype=int)])
        cache = AccFullCache(X, X, y, y, task="binary")

        acc = cache.evaluate(set(), utility="accuracy")
        # predicting the majority class (0) every time → 0.70
        assert acc == pytest.approx(0.70, abs=0.02)


# ═══════════════════════════════════════════════════════════════════════════
# Protocol: train ≠ validation
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainValidationSeparation:
    """AccFullCache stores training labels and validation labels separately;
    predictions are always shaped for the validation set, not the training set."""

    def test_predictions_have_val_row_count(self) -> None:
        import pandas as pd
        from minppi_mofs.voting_game import AccFullCache
        rng = np.random.default_rng(13)
        X_tr = pd.DataFrame({"f": rng.normal(0, 1, 60)})
        X_va = pd.DataFrame({"f": rng.normal(0, 1, 15)})
        y_tr = rng.integers(0, 2, 60)
        y_va = rng.integers(0, 2, 15)

        cache = AccFullCache(X_tr, X_va, y_tr, y_va, task="binary")
        preds = cache.get_predictions({"f"})

        # Predictions are for the VALIDATION set, not the training set
        assert preds.shape[0] == len(y_va), (
            f"predictions have {preds.shape[0]} rows, expected {len(y_va)} (val)"
        )

    def test_utility_computed_on_val_labels(self) -> None:
        """When train ≠ val, utility must be computed against val labels
        (not train labels).  We verify by constructing train/val with
        deliberately different label distributions."""
        import pandas as pd
        from minppi_mofs.voting_game import AccFullCache
        rng = np.random.default_rng(14)
        X_tr = pd.DataFrame({"f": rng.normal(0, 1, 50)})
        X_va = pd.DataFrame({"f": rng.normal(0, 1, 20)})
        y_tr = rng.integers(0, 2, 50)
        # Make val labels deliberately different from train
        y_va = 1 - y_tr[:20]

        cache = AccFullCache(X_tr, X_va, y_tr, y_va, task="binary")
        u = cache.evaluate({"f"})

        # If utility were computed against y_tr by mistake, it would be
        # the training AUC (optimistically high).  With y_va being the
        # antithesis, utility should be low.
        assert 0.0 <= u <= 1.0
        # The key check: predictions shape matches val, not train
        preds = cache.get_predictions({"f"})
        assert preds.shape[0] == 20  # val size, not 50 (train size)


class TestEvaluateRejectsUtilityOverride:
    """Explicit metric overrides must respect the cache task."""

    def test_classification_cache_rejects_regression_override(self) -> None:
        cache = _cache(task="binary", utility="roc_auc")

        with pytest.raises(ValueError, match="regression-only"):
            cache.evaluate({"f"}, utility="r2")

    def test_regression_cache_rejects_classification_override(self) -> None:
        cache = _cache(task="regression", utility="r2")

        with pytest.raises(ValueError, match="classification-only"):
            cache.evaluate({"f"}, utility="accuracy")

    def test_rejects_unknown_override_as_unknown(self) -> None:
        cache = _cache(task="binary", utility="roc_auc")

        with pytest.raises(ValueError, match="Unknown utility"):
            cache.evaluate({"f"}, utility="f1")


class TestPredictionKeyGuard:
    """E4 must refuse to label divergent cache keys as shared predictions."""

    def test_accepts_identical_key_sets(self) -> None:
        from experiments.e4_isolet_utility_ablation import (
            _assert_same_prediction_keys,
        )

        keys = {("f0",), ("f0", "f1")}
        _assert_same_prediction_keys(keys, set(keys))

    @pytest.mark.parametrize(
        "before, after, expected",
        [
            ({("f0",)}, {("f0",), ("f1",)}, "added=1, missing=0"),
            ({("f0",), ("f1",)}, {("f0",)}, "added=0, missing=1"),
        ],
    )
    def test_rejects_changed_key_sets(
        self,
        before: set[tuple[str, ...]],
        after: set[tuple[str, ...]],
        expected: str,
    ) -> None:
        from experiments.e4_isolet_utility_ablation import (
            _assert_same_prediction_keys,
        )

        with pytest.raises(RuntimeError, match=expected):
            _assert_same_prediction_keys(before, after)


class TestFullRunCacheConsistency:
    """Two complete utility runs must reuse one fixed evaluation plan."""

    def test_same_plan_reuses_all_predictions_without_refitting(self) -> None:
        import pandas as pd
        from unittest import mock

        from experiments.e4_isolet_utility_ablation import (
            _assert_same_prediction_keys,
        )
        from minppi_mofs.min_ppi import (
            estimate_minppi_random_mc,
            make_shared_coalition_plan,
        )
        from minppi_mofs.voting_game import AccFullCache, UtilityView

        rng = np.random.default_rng(2)
        n_rows, n_features = 75, 8
        X = pd.DataFrame(
            rng.normal(size=(n_rows, n_features)),
            columns=[f"f{i}" for i in range(n_features)],
        )
        logits = np.column_stack([
            1.2 * X["f0"] - 0.5 * X["f3"],
            1.1 * X["f1"] + 0.4 * X["f4"],
            1.0 * X["f2"] - 0.4 * X["f5"],
        ]) + rng.normal(scale=0.9, size=(n_rows, 3))
        y = logits.argmax(axis=1)
        X_train, X_val = X.iloc[:50], X.iloc[50:]
        y_train, y_val = y[:50], y[50:]

        cache = AccFullCache(
            X_train,
            X_val,
            y_train,
            y_val,
            task="multiclass",
            seed=42,
            model_params={"n_estimators": 5, "max_depth": 2},
        )
        view_accuracy = UtilityView(cache, utility="accuracy", alpha=0.95)
        view_ovr_auc = UtilityView(cache, utility="ovr_auc", alpha=0.95)
        shared_plan = make_shared_coalition_plan(
            X_train.columns,
            K=8,
            per_coalition=2,
            seed=42,
        )
        common = {
            "alpha": 0.95,
            "K": 8,
            "shared_coalitions": True,
            "per_coalition": 2,
            "shared_plan": shared_plan,
            "seed": 42,
            "task": "multiclass",
        }

        estimate_minppi_random_mc(
            X_train, X_val, y_train, y_val,
            utility="accuracy", acc_cache=view_accuracy, **common,
        )
        keys_after_accuracy = set(cache._preds)
        with mock.patch.object(
            cache._model, "fit", wraps=cache._model.fit,
        ) as spy_fit:
            estimate_minppi_random_mc(
                X_train, X_val, y_train, y_val,
                utility="ovr_auc", acc_cache=view_ovr_auc, **common,
            )
            assert spy_fit.call_count == 0
        keys_after_ovr_auc = set(cache._preds)

        assert keys_after_ovr_auc == keys_after_accuracy
        _assert_same_prediction_keys(keys_after_accuracy, keys_after_ovr_auc)
