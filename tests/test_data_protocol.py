"""
Leak-detection and correctness tests for the split-first data protocol.

Each test verifies that a specific transform (MI, VarianceThreshold,
StandardScaler) is fitted on **training data only** and never touches
validation or test rows.  Parameter validation and content-level checks
are included.

Run::

    python -m pytest tests/test_data_protocol.py -v
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from minppi_mofs import data_protocol
from minppi_mofs.data_protocol import prepare_split

# Default split: 64 / 16 / 20
_DEF = {"train_frac": 0.64, "val_frac": 0.16}


# ═══════════════════════════════════════════════════════════════════════════
# MI 收到的是训练集内容
# ═══════════════════════════════════════════════════════════════════════════


class TestMIReceivesExactTrainContent:
    """MI receives the actual training X and y arrays."""

    def test_mi_classif_X_received_has_train_row_count(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(8)})
        y = rng.integers(0, 2, n)
        X.index = range(n)

        captured_X: list[np.ndarray] = []
        captured_y: list[np.ndarray] = []

        _orig = data_protocol.mutual_info_classif
        def _spy(X_arr, y_arr, **kw):
            captured_X.append(np.asarray(X_arr).copy())
            captured_y.append(np.asarray(y_arr).copy())
            return _orig(X_arr, y_arr, **kw)

        with mock.patch.object(data_protocol, "mutual_info_classif", _spy):
            split = prepare_split(X, y, task="binary", seed=42, max_features=5,
                                  **_DEF)

        assert len(captured_X) == 1
        mi_X, mi_y = captured_X[0], captured_y[0]
        assert mi_X.shape[0] == len(split.y_train)
        assert len(mi_y) == len(split.y_train)
        assert mi_X.shape[0] < n

    def test_mi_regression_X_received_has_train_row_count(self) -> None:
        rng = np.random.default_rng(43)
        n = 250
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(10)})
        y = rng.normal(0, 1, n)

        captured: list[tuple[np.ndarray, np.ndarray]] = []
        _orig = data_protocol.mutual_info_regression
        def _spy(X_arr, y_arr, **kw):
            captured.append((np.asarray(X_arr).copy(), np.asarray(y_arr).copy()))
            return _orig(X_arr, y_arr, **kw)

        with mock.patch.object(data_protocol, "mutual_info_regression", _spy):
            split = prepare_split(X, y, task="regression", seed=43, max_features=4,
                                  **_DEF)

        assert len(captured) == 1
        mi_X, mi_y = captured[0]
        assert mi_X.shape[0] == len(split.y_train)
        assert mi_X.shape[0] < n

    def test_mi_y_is_train_labels_not_full_labels(self) -> None:
        rng = np.random.default_rng(77)
        n = 200
        X = pd.DataFrame({f"c{i}": rng.normal(0, 1, n) for i in range(5)})
        y = rng.integers(0, 2, n)

        captured_y: list[np.ndarray] = []
        _orig = data_protocol.mutual_info_classif
        def _spy(X_arr, y_arr, **kw):
            captured_y.append(np.asarray(y_arr).copy())
            return _orig(X_arr, y_arr, **kw)

        with mock.patch.object(data_protocol, "mutual_info_classif", _spy):
            split = prepare_split(X, y, task="binary", seed=77, max_features=2,
                                  **_DEF)

        assert len(captured_y) == 1
        assert np.array_equal(captured_y[0], split.y_train), (
            "MI y differs from y_train — may have seen val/test labels"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scaler 参数必须等于训练集统计量
# ═══════════════════════════════════════════════════════════════════════════


class TestScalerParametersEqualTrainStatistics:
    """scaler.mean_ / scale_ equal train column-wise stats."""

    def test_scaler_mean_equals_train_column_means(self) -> None:
        rng = np.random.default_rng(51)
        n = 200
        X = pd.DataFrame({"a": rng.normal(5, 2, n), "b": rng.normal(-3, 1, n)})
        y = rng.integers(0, 2, n)

        captured_X: list[np.ndarray] = []
        captured_scaler: list[StandardScaler] = []
        _orig_ft = StandardScaler.fit_transform
        def _cap_ft(self, X_arr, *a, **kw):
            captured_X.append(np.asarray(X_arr).copy())
            r = _orig_ft(self, X_arr, *a, **kw)
            captured_scaler.append(self)
            return r

        with mock.patch.object(StandardScaler, "fit_transform", _cap_ft):
            split = prepare_split(X, y, task="binary", seed=51, max_features=None,
                                  **_DEF)

        assert len(captured_X) == 1
        scaler = captured_scaler[0]
        pre_mean = captured_X[0].mean(axis=0)
        assert np.allclose(scaler.mean_, pre_mean, atol=1e-10)
        assert captured_X[0].shape[0] == len(split.y_train)
        assert captured_X[0].shape[0] < n

    def test_scaler_scale_equals_train_column_std_population(self) -> None:
        rng = np.random.default_rng(52)
        n = 150
        X = pd.DataFrame({"x": rng.normal(10, 3, n), "y": rng.normal(0, 0.5, n)})
        y = rng.integers(0, 2, n)

        scaler_inst: list[StandardScaler] = []
        _orig_ft = StandardScaler.fit_transform
        def _cap(self, X_arr, *a, **kw):
            r = _orig_ft(self, X_arr, *a, **kw)
            scaler_inst.append(self)
            return r

        with mock.patch.object(StandardScaler, "fit_transform", _cap):
            split = prepare_split(X, y, task="binary", seed=52, max_features=None,
                                  **_DEF)

        scaler = scaler_inst[0]
        recon = split.X_train.values * scaler.scale_ + scaler.mean_
        assert np.allclose(recon.mean(axis=0), scaler.mean_, atol=1e-10)
        assert np.allclose(recon.std(axis=0, ddof=0), scaler.scale_, atol=1e-10)

    def test_scaler_fit_row_count_is_train_only(self) -> None:
        rng = np.random.default_rng(53)
        n = 300
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
            "c": rng.normal(-2, 0.5, n),
        })
        y = rng.integers(0, 2, n)

        captured_X: list[np.ndarray] = []
        captured_scaler: list[StandardScaler] = []
        _orig_ft = StandardScaler.fit_transform
        def _cap_ft(self, X_arr, *a, **kw):
            captured_X.append(np.asarray(X_arr).copy())
            r = _orig_ft(self, X_arr, *a, **kw)
            captured_scaler.append(self)
            return r

        with mock.patch.object(StandardScaler, "fit_transform", _cap_ft):
            split = prepare_split(X, y, task="binary", seed=53, max_features=None,
                                  **_DEF)

        scaler = captured_scaler[0]
        pre_X = captured_X[0]
        pre_std_pop = pre_X.std(axis=0, ddof=0)
        assert np.allclose(scaler.scale_, pre_std_pop, atol=1e-10)
        assert pre_X.shape[0] == len(split.y_train)


# ═══════════════════════════════════════════════════════════════════════════
# VarianceThreshold 只能在训练集拟合
# ═══════════════════════════════════════════════════════════════════════════


class TestVarianceThresholdFittedOnTrainOnly:
    """VT.fit receives only training rows."""

    def test_vt_fit_receives_only_train_rows(self) -> None:
        rng = np.random.default_rng(61)
        n = 300
        X = pd.DataFrame({
            "keep": rng.normal(0, 1, n),
            "drop": np.full(n, 7.0),   # constant everywhere
        })
        y = rng.integers(0, 2, n)

        captured_rows: list[int] = []
        _orig_fit = VarianceThreshold.fit
        def _spy_fit(self, X_arr, y=None):
            captured_rows.append(X_arr.shape[0])
            return _orig_fit(self, X_arr, y)

        with mock.patch.object(VarianceThreshold, "fit", _spy_fit):
            split = prepare_split(
                X, y, task="binary", seed=61, max_features=None,
                variance_threshold=0.01, **_DEF,
            )

        assert len(captured_rows) == 1
        assert captured_rows[0] == len(split.y_train)
        assert captured_rows[0] < n

    def test_vt_drops_truly_constant_column(self) -> None:
        rng = np.random.default_rng(62)
        n = 200
        X = pd.DataFrame({
            "varying": rng.normal(0, 1, n),
            "always_const": np.full(n, 5.0),
        })
        y = rng.integers(0, 2, n)

        split = prepare_split(
            X, y, task="binary", seed=62, max_features=None,
            variance_threshold=0.01, **_DEF,
        )
        assert "always_const" not in split.selected_features
        assert "varying" in split.selected_features

    def test_vt_default_zero_keeps_all_nonzero_variance(self) -> None:
        rng = np.random.default_rng(63)
        n = 100
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(5)})
        y = rng.integers(0, 2, n)

        split = prepare_split(
            X, y, task="binary", seed=63, max_features=None, **_DEF,
        )
        assert len(split.selected_features) == 5

    def test_vt_zero_still_drops_constant_train_column(self) -> None:
        rng = np.random.default_rng(630)
        n = 200
        X = pd.DataFrame({
            "varying": rng.normal(0, 1, n),
            "constant": np.full(n, 9.0),
        })
        y = rng.integers(0, 2, n)

        split = prepare_split(
            X,
            y,
            task="binary",
            seed=630,
            max_features=None,
            variance_threshold=0.0,
            **_DEF,
        )

        assert split.variance_features == ("varying",)
        assert split.selected_features == ("varying",)

    def test_split_exposes_immutable_feature_aligned_fitted_metadata(self) -> None:
        rng = np.random.default_rng(631)
        n = 200
        X = pd.DataFrame({
            "wide": rng.normal(10, 5, n),
            "narrow": rng.normal(-2, 0.25, n),
            "constant": np.full(n, 3.0),
        })
        y = rng.integers(0, 2, n)

        split = prepare_split(
            X,
            y,
            task="binary",
            seed=631,
            max_features=None,
            variance_threshold=0.0,
            **_DEF,
        )

        assert split.variance_features == ("wide", "narrow")
        assert isinstance(split.scaler_mean, tuple)
        assert isinstance(split.scaler_scale, tuple)
        assert len(split.scaler_mean) == len(split.selected_features)
        assert len(split.scaler_scale) == len(split.selected_features)
        train_raw = X.loc[split.X_train.index, list(split.selected_features)]
        assert np.allclose(split.scaler_mean, train_raw.mean(axis=0))
        assert np.allclose(split.scaler_scale, train_raw.std(axis=0, ddof=0))


# ═══════════════════════════════════════════════════════════════════════════
# 非法参数必须报错
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidParametersRaiseErrors:
    """Every invalid parameter must raise ValueError with a message naming it."""

    # ── train_frac ─────────────────────────────────────────────────────

    def test_train_frac_zero_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="train_frac"):
            prepare_split(X, y, train_frac=0.0)

    def test_train_frac_one_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="train_frac"):
            prepare_split(X, y, train_frac=1.0)

    def test_train_frac_negative_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="train_frac"):
            prepare_split(X, y, train_frac=-0.5)

    def test_train_frac_above_one_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="train_frac"):
            prepare_split(X, y, train_frac=1.5)

    # ── val_frac ───────────────────────────────────────────────────────

    def test_val_frac_zero_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="val_frac"):
            prepare_split(X, y, val_frac=0.0)

    def test_val_frac_one_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="val_frac"):
            prepare_split(X, y, val_frac=1.0)

    def test_val_frac_negative_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="val_frac"):
            prepare_split(X, y, val_frac=-0.1)

    # ── test_frac ──────────────────────────────────────────────────────

    def test_test_frac_zero_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="test_frac"):
            prepare_split(X, y, test_frac=0.0)

    def test_test_frac_one_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="test_frac"):
            prepare_split(X, y, test_frac=1.0)

    def test_test_frac_negative_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="test_frac"):
            prepare_split(X, y, test_frac=-0.1)

    # ── sum ≠ 1.0 ──────────────────────────────────────────────────────

    def test_fractions_not_summing_to_one_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = np.array([0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="1\\.0"):   # "must equal 1.0"
            prepare_split(X, y, train_frac=0.5, val_frac=0.3, test_frac=0.3)

    def test_fractions_exceed_one_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = np.array([0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="1\\.0"):
            prepare_split(X, y, train_frac=0.8, val_frac=0.3, test_frac=0.2)

    # ── task ───────────────────────────────────────────────────────────

    def test_invalid_task_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(ValueError, match="task"):
            prepare_split(X, y, task="poisson")

    # ── max_features ───────────────────────────────────────────────────

    def test_max_features_zero_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(ValueError, match="max_features"):
            prepare_split(X, y, max_features=0)

    def test_max_features_negative_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(ValueError, match="max_features"):
            prepare_split(X, y, max_features=-1)

    # ── samples ────────────────────────────────────────────────────────

    def test_too_few_samples_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0]})
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="samples"):
            prepare_split(X, y, train_frac=0.64, val_frac=0.16, test_frac=0.20)

    # ── variance_threshold ─────────────────────────────────────────────

    def test_variance_threshold_negative_raises(self) -> None:
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(ValueError, match="variance_threshold"):
            prepare_split(X, y, variance_threshold=-0.01)


# ═══════════════════════════════════════════════════════════════════════════
# 基础行为
# ═══════════════════════════════════════════════════════════════════════════


class TestMIFittedOnTrainOnly:
    """MI receives only train data (row count + row provenance)."""

    def test_mi_classif_receives_only_train_rows(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(10)})
        y = rng.integers(0, 2, n)

        with mock.patch.object(
            data_protocol, "mutual_info_classif",
            wraps=data_protocol.mutual_info_classif,
        ) as spy:
            split = prepare_split(X, y, task="binary", seed=42, max_features=5,
                                  **_DEF)

        spy.assert_called_once()
        call_X, call_y = spy.call_args[0]
        assert call_X.shape[0] == len(split.y_train)
        assert call_X.shape[0] < n

    def test_mi_regression_receives_only_train_rows(self) -> None:
        rng = np.random.default_rng(43)
        n = 250
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(12)})
        y = rng.normal(0, 1, n)

        with mock.patch.object(
            data_protocol, "mutual_info_regression",
            wraps=data_protocol.mutual_info_regression,
        ) as spy:
            split = prepare_split(X, y, task="regression", seed=43, max_features=6,
                                  **_DEF)

        spy.assert_called_once()
        call_X, call_y = spy.call_args[0]
        assert call_X.shape[0] == len(split.y_train)
        assert call_X.shape[0] < n

    def test_mi_not_called_when_max_features_none(self) -> None:
        rng = np.random.default_rng(44)
        X = pd.DataFrame({"f": rng.normal(0, 1, 100)})
        y = rng.integers(0, 2, 100)

        with mock.patch.object(data_protocol, "mutual_info_classif") as spy:
            prepare_split(X, y, task="binary", seed=44, max_features=None, **_DEF)
        spy.assert_not_called()


class TestScalerFittedOnTrainOnly:
    def test_train_has_unit_standardisation(self) -> None:
        rng = np.random.default_rng(123)
        n = 200
        X = pd.DataFrame({"a": rng.normal(5, 2, n), "b": rng.normal(-3, 1, n)})
        y = rng.integers(0, 2, n)

        split = prepare_split(X, y, task="binary", seed=123, max_features=None, **_DEF)
        train_arr = split.X_train.values
        assert np.allclose(train_arr.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(train_arr.std(axis=0, ddof=0), 1.0, atol=1e-6)

    def test_full_data_not_perfectly_standardised(self) -> None:
        rng = np.random.default_rng(124)
        n = 200
        X = pd.DataFrame({
            "a": np.concatenate([rng.normal(5, 2, 100), rng.normal(0, 2, 100)]),
        })
        y = rng.integers(0, 2, n)

        split = prepare_split(X, y, task="binary", seed=124, max_features=None, **_DEF)
        full = np.vstack([
            split.X_train.values, split.X_val.values, split.X_test.values,
        ])
        assert not np.allclose(full.mean(axis=0), 0.0, atol=1e-12)
        assert len(split.X_train) + len(split.X_val) + len(split.X_test) == n


class TestSplitNonOverlapping:
    def test_splits_are_disjoint_and_complete(self) -> None:
        rng = np.random.default_rng(99)
        n = 150
        X = pd.DataFrame({"f0": rng.normal(0, 1, n)})
        y = rng.integers(0, 2, n)
        X.index = range(n)

        split = prepare_split(X, y, task="binary", seed=99, max_features=None, **_DEF)

        train_idx = set(split.X_train.index)
        val_idx = set(split.X_val.index)
        test_idx = set(split.X_test.index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert train_idx | val_idx | test_idx == set(range(n))

    def test_default_split_fractions_are_64_16_20(self) -> None:
        rng = np.random.default_rng(98)
        n = 200
        X = pd.DataFrame({"f": rng.normal(0, 1, n)})
        y = rng.integers(0, 2, n)

        split = prepare_split(X, y, task="binary", seed=98, max_features=None)
        assert len(split.y_train) == pytest.approx(n * 0.64, abs=1)
        assert len(split.y_val) == pytest.approx(n * 0.16, abs=1)
        assert len(split.y_test) == pytest.approx(n * 0.20, abs=1)

    def test_stratified_split_preserves_class_balance(self) -> None:
        rng = np.random.default_rng(77)
        n = 500
        X = pd.DataFrame({"f": rng.normal(0, 1, n)})
        y = rng.integers(0, 2, n)

        split = prepare_split(X, y, task="binary", seed=77, max_features=None, **_DEF)
        full_pos = y.mean()
        for name, y_split in [
            ("train", split.y_train),
            ("val", split.y_val),
            ("test", split.y_test),
        ]:
            assert abs(y_split.mean() - full_pos) < 0.15


class TestRegressionSplit:
    def test_regression_split_no_stratify(self) -> None:
        rng = np.random.default_rng(55)
        n = 200
        X = pd.DataFrame({"f": rng.normal(0, 1, n)})
        y = rng.normal(0, 1, n)

        split = prepare_split(
            X, y, task="regression", seed=55, max_features=None, **_DEF,
        )
        assert split.X_train.shape[1] == 1
        assert len(split.y_train) + len(split.y_val) + len(split.y_test) == n


class TestMulticlass:
    def test_multiclass_split(self) -> None:
        rng = np.random.default_rng(33)
        n = 300
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(5)})
        y = rng.integers(0, 5, n)

        split = prepare_split(
            X, y, task="multiclass", seed=33, max_features=3, **_DEF,
        )
        assert len(split.selected_features) == 3
        assert split.X_train.shape[1] == 3


class TestDeterminism:
    def test_same_seed_reproduces(self) -> None:
        rng = np.random.default_rng(11)
        X = pd.DataFrame({"f": rng.normal(0, 1, 200)})
        y = rng.integers(0, 2, 200)

        s1 = prepare_split(X, y, task="binary", seed=11, max_features=None)
        s2 = prepare_split(X, y, task="binary", seed=11, max_features=None)

        assert np.array_equal(s1.X_train.values, s2.X_train.values)
        assert np.array_equal(s1.y_train, s2.y_train)

    def test_different_seed_differs(self) -> None:
        rng = np.random.default_rng(22)
        X = pd.DataFrame({"f": rng.normal(0, 1, 200)})
        y = rng.integers(0, 2, 200)

        s1 = prepare_split(X, y, task="binary", seed=22, max_features=None)
        s2 = prepare_split(X, y, task="binary", seed=44, max_features=None)

        arrays_differ = (
            not np.array_equal(s1.X_train.values, s2.X_train.values)
            or not np.array_equal(s1.X_val.values, s2.X_val.values)
        )
        assert arrays_differ, "Different seeds produced identical splits"


class TestMaxFeatures:
    def test_max_features_none_keeps_all(self) -> None:
        rng = np.random.default_rng(66)
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 100) for i in range(8)})
        y = rng.integers(0, 2, 100)
        split = prepare_split(
            X, y, task="binary", seed=66, max_features=None, **_DEF,
        )
        assert len(split.selected_features) == 8

    def test_max_features_caps(self) -> None:
        rng = np.random.default_rng(88)
        X = pd.DataFrame({f"f{i}": rng.normal(0, 1, 100) for i in range(20)})
        y = rng.integers(0, 2, 100)
        split = prepare_split(
            X, y, task="binary", seed=88, max_features=5, **_DEF,
        )
        assert len(split.selected_features) == 5
