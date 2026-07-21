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


class TestAccFullCacheRejectsWrongUtilityForTask:
    """Regression task + classification utility → ValueError."""

    def test_regression_with_roc_auc_raises(self) -> None:
        from minppi_mofs.voting_game import AccFullCache
        import pandas as pd
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"f": rng.normal(0, 1, 20)})
        y = rng.normal(0, 1, 20)
        with pytest.raises(ValueError, match="classification-only"):
            AccFullCache(X, X, y, y, task="regression", utility="roc_auc")
