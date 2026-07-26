from dataclasses import replace
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from minppi_mofs.coalition_cache import (
    CACHE_SCHEMA_VERSION,
    CoalitionCacheIdentity,
    CoalitionUtilityCache,
    CoalitionUtilityRecord,
    CoalitionUtilityView,
    SQLiteCoalitionStore,
    _hash_numeric_payload,
    canonical_coalition,
    hash_numeric_array,
    hash_numeric_frame,
    prediction_digest,
)
from minppi_mofs.utility import compute_utility
from minppi_mofs.modeling import evaluate_fixed_subset


def identity(**changes):
    value = CoalitionCacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        protocol_hash="0" * 64,
        dataset_id="toy",
        seed=42,
        namespace="selection_validation",
        train_features_sha256="a" * 64,
        train_target_sha256="b" * 64,
        eval_features_sha256="c" * 64,
        eval_target_sha256="d" * 64,
        ordered_features=("a", "b", "c"),
        task="binary",
        class_labels=(0, 1),
        model_params={
            "learning_rate": 0.1,
            "max_depth": 4,
            "n_estimators": 100,
            "n_jobs": 1,
        },
        backend="cuda",
        model_seed=42,
        library_versions={"numpy": "2", "xgboost": "3"},
    )
    return replace(value, **changes)


def bound_identity(X_train, y_train, X_eval, y_eval, **changes):
    return identity(
        train_features_sha256=hash_numeric_frame(X_train),
        train_target_sha256=hash_numeric_array(np.asarray(y_train)),
        eval_features_sha256=hash_numeric_frame(X_eval),
        eval_target_sha256=hash_numeric_array(np.asarray(y_eval)),
        **changes,
    )


def make_multiclass_cache(tmp_path):
    X_train = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]}
    )
    X_eval = pd.DataFrame(
        {"a": [0.5, 1.5, 2.5], "b": [2.5, 1.5, 0.5]}
    )
    y_train = np.array([0, 1, 2, 0])
    y_eval = np.array([0, 1, 2])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a", "b"),
        task="multiclass",
        class_labels=(0, 1, 2),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )
    store = SQLiteCoalitionStore(
        tmp_path / "multiclass.sqlite3",
        cache_identity,
        commit_every_fits=1,
    )
    return (
        CoalitionUtilityCache(
            cache_identity, X_train, y_train, X_eval, y_eval, store
        ),
        store,
        y_eval,
    )


def test_multiclass_metrics_share_one_prediction(monkeypatch, tmp_path):
    calls = []
    predictions = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
    )

    def fake_fit_predict(*args, **kwargs):
        calls.append(kwargs)
        return predictions.copy()

    monkeypatch.setattr("minppi_mofs.coalition_cache.fit_predict", fake_fit_predict)
    cache, store, y_eval = make_multiclass_cache(tmp_path)
    try:
        accuracy = cache.evaluate({"a"}, "accuracy")
        auc = cache.evaluate({"a"}, "ovr_auc")
        record = cache.record({"a"})

        assert len(calls) == 1
        assert accuracy == pytest.approx(
            compute_utility(y_eval, predictions, metric="accuracy"), abs=1e-12
        )
        assert auc == pytest.approx(
            compute_utility(y_eval, predictions, metric="ovr_auc"), abs=1e-12
        )
        assert record.prediction_sha256 == prediction_digest(predictions)
        assert cache.stats.physical_model_fit_count == 1
        assert cache.stats.cache_hit_count == 2
    finally:
        store.close()


def test_metric_views_account_for_logical_coalitions_independently(tmp_path):
    cache, store, _ = make_multiclass_cache(tmp_path)
    try:
        accuracy = CoalitionUtilityView(cache, utility="accuracy", alpha=1.0)
        auc = CoalitionUtilityView(cache, utility="ovr_auc", alpha=0.9)

        assert accuracy.model_fit_count == auc.model_fit_count == 1
        assert accuracy.coalition_count == auc.coalition_count == 1
        assert cache.stats.physical_model_fit_count == 1
        assert cache.stats.cache_hit_count == 1

        accuracy.evaluate({"a"})
        auc.evaluate({"a"})
        accuracy.evaluate(set())

        assert accuracy.model_fit_count == 2
        assert accuracy.coalition_count == 3
        assert auc.model_fit_count == 2
        assert auc.coalition_count == 2
        assert cache.stats.physical_model_fit_count == 2
        assert cache.stats.cache_hit_count == 2
    finally:
        store.close()


def test_metric_view_exposes_immutable_public_feature_order(monkeypatch, tmp_path):
    predictions = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
    )
    monkeypatch.setattr(
        "minppi_mofs.coalition_cache.fit_predict",
        lambda *args, **kwargs: predictions.copy(),
    )
    cache, store, _ = make_multiclass_cache(tmp_path)
    try:
        view = CoalitionUtilityView(cache, utility="accuracy", alpha=0.9)

        assert view.feature_order == ("a", "b")
        assert isinstance(view.feature_order, tuple)
        with pytest.raises(AttributeError):
            view.feature_order = ("changed",)
    finally:
        store.close()


def test_binary_cache_materializes_all_metrics_and_cacheable_empty_subset(
    monkeypatch, tmp_path
):
    X_train = pd.DataFrame({"a": [0.0, 1.0, 0.1, 0.9], "b": [1.0, 0.0, 0.8, 0.2]})
    X_eval = pd.DataFrame({"a": [0.2, 0.8], "b": [0.9, 0.1]})
    y_train = np.array([0, 1, 0, 1])
    y_eval = np.array([0, 1])
    predictions = np.array([[0.8, 0.2], [0.2, 0.8]])
    fit_calls = 0

    def fake_fit_predict(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return predictions.copy()

    monkeypatch.setattr("minppi_mofs.coalition_cache.fit_predict", fake_fit_predict)
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a", "b"),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )
    with SQLiteCoalitionStore(
        tmp_path / "binary.sqlite3", cache_identity, commit_every_fits=1
    ) as store:
        cache = CoalitionUtilityCache(
            cache_identity, X_train, y_train, X_eval, y_eval, store
        )
        fitted = cache.record({"a"})
        empty = cache.record(set())
        empty_again = cache.record(set())

        assert set(fitted.utilities) == {"accuracy", "pr_auc", "roc_auc"}
        for metric in ("accuracy", "pr_auc", "roc_auc"):
            assert fitted.utilities[metric] == pytest.approx(
                compute_utility(y_eval, predictions, metric=metric), abs=1e-12
            )
        assert set(empty.utilities) == {"accuracy", "pr_auc", "roc_auc"}
        assert fitted.fitted is True
        assert empty.fitted is False
        assert empty_again == empty
        assert empty.prediction_sha256 == prediction_digest(
            np.tile(np.array([0.5, 0.5]), (2, 1))
        )
        assert cache.stats.physical_model_fit_count == 1
        assert cache.stats.cache_hit_count == 1
        assert cache.stats.coalition_count == 2
        assert fit_calls == 1


def test_regression_cache_matches_uncached_project_utilities(tmp_path):
    X_train = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0, 0.0]}
    )
    X_eval = pd.DataFrame({"a": [0.5, 2.5, 3.5], "b": [3.5, 1.5, 0.5]})
    y_train = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    y_eval = np.array([0.25, 6.25, 12.25])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a", "b"),
        task="regression",
        class_labels=(),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )
    with SQLiteCoalitionStore(
        tmp_path / "regression.sqlite3", cache_identity, commit_every_fits=1
    ) as store:
        cache = CoalitionUtilityCache(
            cache_identity, X_train, y_train, X_eval, y_eval, store
        )
        record = cache.record({"a"})

        assert set(record.utilities) == {"r2", "rmse"}
        for metric in ("r2", "rmse"):
            uncached = evaluate_fixed_subset(
                X_train,
                y_train,
                X_eval,
                y_eval,
                ("a",),
                task="regression",
                utility=metric,
                model_params=cache_identity.model_params,
                backend="cpu",
                seed=cache_identity.model_seed,
            )
            assert cache.evaluate({"a"}, metric) == pytest.approx(
                uncached, abs=1e-12
            )
        assert cache.stats.physical_model_fit_count == 1
        assert cache.stats.cache_hit_count == 2


def test_reopened_cache_reuses_identical_record_without_fit(monkeypatch, tmp_path):
    path = tmp_path / "reopen.sqlite3"
    X_train = pd.DataFrame({"a": [0.0, 1.0, 0.1, 0.9]})
    X_eval = pd.DataFrame({"a": [0.2, 0.8]})
    y_train = np.array([0, 1, 0, 1])
    y_eval = np.array([0, 1])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a",),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )
    with SQLiteCoalitionStore(path, cache_identity, commit_every_fits=1) as store:
        first_cache = CoalitionUtilityCache(
            cache_identity, X_train, y_train, X_eval, y_eval, store
        )
        first = first_cache.record({"a"})
        assert first_cache.stats.physical_model_fit_count == 1

    def fail_if_fitted(*args, **kwargs):
        raise AssertionError("reopened record must not fit")

    monkeypatch.setattr(
        "minppi_mofs.coalition_cache.fit_predict", fail_if_fitted
    )
    with SQLiteCoalitionStore(path, cache_identity, commit_every_fits=1) as store:
        reopened = CoalitionUtilityCache(
            cache_identity, X_train, y_train, X_eval, y_eval, store
        )
        assert reopened.stats.physical_model_fit_count == 0
        assert reopened.stats.cache_hit_count == 0
        assert reopened.stats.coalition_count == 1

        second = reopened.record({"a"})

        assert second == first
        assert second.prediction_sha256 == first.prediction_sha256
        assert reopened.stats.physical_model_fit_count == 0
        assert reopened.stats.cache_hit_count == 1


def test_final_test_identity_cannot_open_selection_validation_store(tmp_path):
    path = tmp_path / "namespace.sqlite3"
    SQLiteCoalitionStore(path, identity(), commit_every_fits=1).close()

    with pytest.raises(ValueError, match="identity"):
        SQLiteCoalitionStore(
            path,
            identity(namespace="final_test"),
            commit_every_fits=1,
        )


def test_cache_rejects_target_length_mismatch_during_construction(tmp_path):
    X_train = pd.DataFrame({"a": [0.0, 1.0]})
    X_eval = pd.DataFrame({"a": [0.5]})
    cache_identity = bound_identity(
        X_train,
        np.array([0, 1]),
        X_eval,
        np.array([0]),
        ordered_features=("a",),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )
    with SQLiteCoalitionStore(
        tmp_path / "invalid-target.sqlite3",
        cache_identity,
        commit_every_fits=1,
    ) as store:
        with pytest.raises(ValueError, match="target|row"):
            CoalitionUtilityCache(
                cache_identity,
                X_train,
                np.array([0]),
                X_eval,
                np.array([0]),
                store,
            )


@pytest.mark.parametrize(
    "mismatched_input",
    ["X_train", "y_train", "X_eval", "y_eval"],
)
def test_cache_rejects_same_shape_content_identity_mismatch(
    tmp_path, mismatched_input
):
    X_train = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]}
    )
    X_eval = pd.DataFrame(
        {"a": [0.5, 1.5, 2.5], "b": [2.5, 1.5, 0.5]}
    )
    y_train = np.array([0, 1, 2, 0])
    y_eval = np.array([0, 1, 2])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a", "b"),
        task="multiclass",
        class_labels=(0, 1, 2),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )
    supplied = {
        "X_train": X_train.copy(),
        "y_train": y_train.copy(),
        "X_eval": X_eval.copy(),
        "y_eval": y_eval.copy(),
    }
    if mismatched_input == "X_train":
        supplied["X_train"].iloc[0, 0] = 987654.0
    elif mismatched_input == "X_eval":
        supplied["X_eval"].iloc[0, 0] = 987654.0
    elif mismatched_input == "y_train":
        supplied["y_train"][0] = 1
    else:
        supplied["y_eval"][:2] = [1, 0]

    with SQLiteCoalitionStore(
        tmp_path / f"{mismatched_input}.sqlite3",
        cache_identity,
        commit_every_fits=1,
    ) as store:
        with pytest.raises(
            ValueError, match=rf"{mismatched_input}.*identity.*hash"
        ) as error:
            CoalitionUtilityCache(
                cache_identity,
                supplied["X_train"],
                supplied["y_train"],
                supplied["X_eval"],
                supplied["y_eval"],
                store,
            )

        assert "987654" not in str(error.value)
        assert store.entry_count == 0


def test_regression_cache_rejects_single_eval_row_without_fit(monkeypatch, tmp_path):
    X_train = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    X_eval = pd.DataFrame({"a": [0.5]})
    y_train = np.array([0.0, 1.0, 4.0])
    y_eval = np.array([0.25])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a",),
        task="regression",
        class_labels=(),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )

    def fail_if_fitted(*args, **kwargs):
        raise AssertionError("invalid regression evaluation must not fit")

    monkeypatch.setattr(
        "minppi_mofs.coalition_cache.fit_predict", fail_if_fitted
    )
    with SQLiteCoalitionStore(
        tmp_path / "single-regression-row.sqlite3",
        cache_identity,
        commit_every_fits=1,
    ) as store:
        with pytest.raises(ValueError, match="at least two"):
            CoalitionUtilityCache(
                cache_identity,
                X_train,
                y_train,
                X_eval,
                y_eval,
                store,
            )
        assert store.entry_count == 0


def test_cache_validates_model_policy_before_empty_evaluation(tmp_path):
    X_train = pd.DataFrame({"a": [0.0, 1.0]})
    X_eval = pd.DataFrame({"a": [0.5]})
    y_train = np.array([0, 1])
    y_eval = np.array([0])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a",),
        backend="cpu",
        model_params={"unknown_parameter": 1},
    )
    with SQLiteCoalitionStore(
        tmp_path / "invalid-policy.sqlite3",
        cache_identity,
        commit_every_fits=1,
    ) as store:
        with pytest.raises(ValueError, match="unknown model_params"):
            CoalitionUtilityCache(
                cache_identity,
                X_train,
                y_train,
                X_eval,
                y_eval,
                store,
            )
        assert store.entry_count == 0


def test_multiclass_cache_rejects_eval_target_missing_an_identity_class(tmp_path):
    X_train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    X_eval = pd.DataFrame({"a": [0.5, 1.5]})
    y_train = np.array([0, 1, 2, 0])
    y_eval = np.array([0, 1])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a",),
        task="multiclass",
        class_labels=(0, 1, 2),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )
    with SQLiteCoalitionStore(
        tmp_path / "missing-eval-class.sqlite3",
        cache_identity,
        commit_every_fits=1,
    ) as store:
        with pytest.raises(ValueError, match="classes"):
            CoalitionUtilityCache(
                cache_identity,
                X_train,
                y_train,
                X_eval,
                y_eval,
                store,
            )


def test_cache_rejects_reopened_record_with_nonnumeric_prediction_dtype(tmp_path):
    X_train = pd.DataFrame({"a": [0.0, 1.0]})
    X_eval = pd.DataFrame({"a": [0.5, 0.7]})
    y_train = np.array([0, 1])
    y_eval = np.array([0, 1])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a",),
        backend="cpu",
        model_params={"n_estimators": 2, "n_jobs": 1},
    )
    with SQLiteCoalitionStore(
        tmp_path / "invalid-dtype.sqlite3",
        cache_identity,
        commit_every_fits=1,
    ) as store:
        store.put(
            CoalitionUtilityRecord(
                coalition=("a",),
                utilities={"accuracy": 0.5, "pr_auc": 0.5, "roc_auc": 0.5},
                prediction_shape=(2, 2),
                prediction_dtype="object",
                prediction_sha256="f" * 64,
                fitted=True,
            )
        )
        cache = CoalitionUtilityCache(
            cache_identity,
            X_train,
            y_train,
            X_eval,
            y_eval,
            store,
        )

        with pytest.raises(ValueError, match="dtype"):
            cache.record({"a"})


@pytest.mark.parametrize("alpha", [True, 0.0, 1.01, float("nan")])
def test_view_rejects_invalid_thresholds_without_evaluation(tmp_path, alpha):
    cache, store, _ = make_multiclass_cache(tmp_path)
    try:
        with pytest.raises(ValueError, match="alpha"):
            CoalitionUtilityView(cache, utility="accuracy", alpha=alpha)
        assert cache.stats.coalition_count == 0
    finally:
        store.close()


def test_cache_and_view_reject_wrong_metric_features_and_frame_order(tmp_path):
    cache, store, _ = make_multiclass_cache(tmp_path)
    try:
        with pytest.raises(ValueError, match="utility"):
            cache.evaluate({"a"}, "roc_auc")
        with pytest.raises(ValueError, match="utility"):
            CoalitionUtilityView(cache, utility="roc_auc", alpha=1.0)
        view = CoalitionUtilityView(cache, utility="accuracy", alpha=1.0)
        with pytest.raises(ValueError, match="outside"):
            view.evaluate({"unknown"})
        with pytest.raises(ValueError, match="outside"):
            view.eval_marginal(set(), "unknown")
    finally:
        store.close()

    X_train = pd.DataFrame({"b": [0.0, 1.0], "a": [1.0, 0.0]})
    X_eval = pd.DataFrame({"a": [0.5], "b": [0.5]})
    y_train = np.array([0, 1])
    y_eval = np.array([0])
    cache_identity = bound_identity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        ordered_features=("a", "b"),
        backend="cpu",
    )
    with SQLiteCoalitionStore(
        tmp_path / "wrong-order.sqlite3", cache_identity, commit_every_fits=1
    ) as store:
        with pytest.raises(ValueError, match="column order"):
            CoalitionUtilityCache(
                cache_identity,
                X_train,
                y_train,
                X_eval,
                y_eval,
                store,
            )


def test_sqlite_store_persists_one_canonical_record(tmp_path):
    path = tmp_path / "cache.sqlite3"
    record = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75, "roc_auc": 0.8, "pr_auc": 0.7},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=True,
    )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        assert store.get(("a",)) is None
        store.put(record)
        assert store.get(("a",)) == record
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as reopened:
        assert reopened.get(("a",)) == record


def test_sqlite_store_rejects_foreign_identity(tmp_path):
    path = tmp_path / "cache.sqlite3"
    SQLiteCoalitionStore(path, identity(), commit_every_fits=1).close()
    with pytest.raises(ValueError, match="identity"):
        SQLiteCoalitionStore(
            path, identity(eval_features_sha256="9" * 64), commit_every_fits=1
        )


def test_sqlite_store_rejects_corrupt_payload_checksum(tmp_path):
    path = tmp_path / "cache.sqlite3"
    record = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=True,
    )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        store.put(record)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE coalition_utility SET payload_sha256 = ?", ("0" * 64,)
        )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        with pytest.raises(ValueError, match="checksum"):
            store.get(("a",))


def test_sqlite_store_rejects_noncanonical_payload(tmp_path):
    path = tmp_path / "cache.sqlite3"
    record = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=True,
    )
    noncanonical_payload = '{"utilities":{"accuracy":0.75},"coalition":["a"],"prediction_shape":[4,2],"prediction_dtype":"float64","prediction_sha256":"' + "f" * 64 + '","fitted":true}'
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        store.put(record)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE coalition_utility SET payload_json = ?, payload_sha256 = ?",
            (
                noncanonical_payload,
                hashlib.sha256(noncanonical_payload.encode("utf-8")).hexdigest(),
            ),
        )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        with pytest.raises(ValueError, match="canonical"):
            store.get(("a",))


def test_sqlite_store_rejects_corrupt_metadata(tmp_path):
    path = tmp_path / "cache.sqlite3"
    SQLiteCoalitionStore(path, identity(), commit_every_fits=1).close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'", ("bad",)
        )
    with pytest.raises(ValueError, match="metadata"):
        SQLiteCoalitionStore(path, identity(), commit_every_fits=1)


@pytest.mark.parametrize(
    "identity_changes",
    [
        {},
        {"namespace": "final_test", "eval_features_sha256": "9" * 64},
    ],
)
def test_sqlite_store_rejects_missing_metadata_when_rows_exist(
    tmp_path, identity_changes
):
    path = tmp_path / "cache.sqlite3"
    record = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=True,
    )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        store.put(record)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM metadata")

    with pytest.raises(ValueError, match="metadata"):
        SQLiteCoalitionStore(
            path, identity(**identity_changes), commit_every_fits=1
        )

    with sqlite3.connect(path, timeout=0.1) as connection:
        assert connection.execute("SELECT * FROM metadata").fetchall() == []
        connection.execute("BEGIN EXCLUSIVE")
        connection.rollback()


def test_sqlite_store_flush_persists_pending_fitted_batch(tmp_path):
    path = tmp_path / "cache.sqlite3"
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=50) as store:
        for index in range(49):
            store.put(
                CoalitionUtilityRecord(
                    coalition=(f"feature-{index}",),
                    utilities={"accuracy": 0.75},
                    prediction_shape=(4, 2),
                    prediction_dtype="float64",
                    prediction_sha256=f"{index:064x}",
                    fitted=True,
                )
            )
        assert store.pending_write_count == 49
        assert store.entry_count == 49
        store.flush()
        assert store.pending_write_count == 0
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=50) as reopened:
        assert [reopened.get((f"feature-{index}",)) for index in range(49)] == [
            CoalitionUtilityRecord(
                coalition=(f"feature-{index}",),
                utilities={"accuracy": 0.75},
                prediction_shape=(4, 2),
                prediction_dtype="float64",
                prediction_sha256=f"{index:064x}",
                fitted=True,
            )
            for index in range(49)
        ]


def test_sqlite_store_rejects_conflicting_write_and_invalid_intervals(tmp_path):
    path = tmp_path / "cache.sqlite3"
    first = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=True,
    )
    with pytest.raises(ValueError, match="positive integer"):
        SQLiteCoalitionStore(path, identity(), commit_every_fits=True)
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=2) as store:
        store.put(first)
        with pytest.raises(ValueError, match="conflicts"):
            store.put(replace(first, utilities={"accuracy": 0.5}))


@pytest.mark.parametrize("column", ["payload_json", "payload_sha256"])
def test_sqlite_store_fails_closed_for_blob_payload_columns(tmp_path, column):
    path = tmp_path / "cache.sqlite3"
    record = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=True,
    )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        store.put(record)
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"UPDATE coalition_utility SET {column} = ?", (sqlite3.Binary(b"bad"),)
        )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        with pytest.raises(ValueError, match="payload|checksum|schema"):
            store.get(("a",))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coalition", [1]),
        ("utilities", {"accuracy": "0.75"}),
        ("prediction_shape", [True]),
        ("prediction_dtype", 64),
        ("prediction_sha256", 12),
        ("fitted", 1),
    ],
)
def test_sqlite_store_rejects_malformed_reconstructed_fields(tmp_path, field, value):
    path = tmp_path / "cache.sqlite3"
    record = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=True,
    )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        store.put(record)
    with sqlite3.connect(path) as connection:
        payload_json = connection.execute(
            "SELECT payload_json FROM coalition_utility"
        ).fetchone()[0]
        payload = json.loads(payload_json)
        payload[field] = value
        corrupt_json = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        connection.execute(
            "UPDATE coalition_utility SET payload_json = ?, payload_sha256 = ?",
            (
                corrupt_json,
                hashlib.sha256(corrupt_json.encode("utf-8")).hexdigest(),
            ),
        )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        with pytest.raises(ValueError, match="cache"):
            store.get(("a",))


def test_sqlite_store_unfitted_writes_do_not_advance_pending_fit_count(tmp_path):
    path = tmp_path / "cache.sqlite3"
    record = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=False,
    )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        store.put(record)
        assert store.pending_write_count == 0
        store.flush()
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as reopened:
        assert reopened.get(("a",)) == record


def test_sqlite_store_rejects_symlink_path(tmp_path):
    target = tmp_path / "target.sqlite3"
    symlink = tmp_path / "cache.sqlite3"
    try:
        symlink.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    with pytest.raises(ValueError, match="symlink"):
        SQLiteCoalitionStore(symlink, identity(), commit_every_fits=1)


def test_sqlite_store_close_and_flush_are_idempotent(tmp_path):
    store = SQLiteCoalitionStore(tmp_path / "cache.sqlite3", identity(), commit_every_fits=1)
    store.close()
    store.flush()
    store.close()


def test_sqlite_store_context_does_not_mask_body_exception_after_manual_close(tmp_path):
    with pytest.raises(RuntimeError, match="body failure"):
        with SQLiteCoalitionStore(
            tmp_path / "cache.sqlite3", identity(), commit_every_fits=1
        ) as store:
            store.close()
            raise RuntimeError("body failure")


def test_identity_is_canonical_and_field_sensitive():
    first = identity()
    assert first.canonical_json() == identity().canonical_json()
    assert first.digest == identity().digest
    for field, value in {
        "namespace": "final_test",
        "eval_features_sha256": "e" * 64,
        "backend": "cpu",
        "model_seed": 123,
    }.items():
        assert replace(first, **{field: value}).digest != first.digest


def test_canonical_coalition_uses_frozen_feature_order():
    assert canonical_coalition(("b", "a", "c"), {"a", "b"}) == ("b", "a")
    with pytest.raises(ValueError, match="outside"):
        canonical_coalition(("a",), {"unknown"})


def test_numeric_hashes_bind_order_shape_dtype_and_values():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert hash_numeric_frame(frame) == hash_numeric_frame(frame.copy())
    assert hash_numeric_frame(frame[["b", "a"]]) != hash_numeric_frame(frame)
    assert hash_numeric_array(np.array([0, 1])) != hash_numeric_array(
        np.array([1, 0])
    )


def test_numeric_hashes_bind_dtype_and_frame_shape():
    assert hash_numeric_array(np.array([1, 2], dtype=np.int32)) != hash_numeric_array(
        np.array([1, 2], dtype=np.int64)
    )
    single_column = pd.DataFrame({"a": [1, 2, 3, 4]})
    two_columns = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert hash_numeric_frame(single_column) != hash_numeric_frame(two_columns)


def test_identity_owns_mutable_feature_order():
    feature_order = ["a", "b", "c"]
    value = identity(ordered_features=feature_order)
    original_digest = value.digest

    feature_order[0] = "changed"

    assert value.ordered_features == ("a", "b", "c")
    assert value.digest == original_digest


def test_identity_canonicalizes_mapping_order_and_exposes_immutable_mappings():
    first = identity(
        model_params={"nested": {"beta": [2, None], "alpha": 1}},
        library_versions={"xgboost": "3", "numpy": "2"},
    )
    second = identity(
        model_params={"nested": {"alpha": 1, "beta": [2, None]}},
        library_versions={"numpy": "2", "xgboost": "3"},
    )

    assert first.digest == second.digest
    with pytest.raises(TypeError):
        first.model_params["new"] = 1
    with pytest.raises(TypeError):
        first.library_versions["new"] = "1"


def test_identity_owns_and_deeply_freezes_nested_model_params():
    caller_params = {"tuning": {"layers": [3, {"learning_rate": 0.1}]}}
    value = identity(model_params=caller_params)
    original_json = value.canonical_json()
    original_digest = value.digest

    caller_params["tuning"]["layers"][1]["learning_rate"] = 0.2

    assert value.canonical_json() == original_json
    assert value.digest == original_digest
    with pytest.raises(TypeError):
        value.model_params["tuning"]["layers"][1]["learning_rate"] = 0.3


def test_numeric_payload_hash_binds_shape_with_identical_bytes_and_dtype():
    matrix = np.array([[1, 2], [3, 4]], dtype=np.int16)
    vector = np.array([1, 2, 3, 4], dtype=np.int16)

    assert matrix.tobytes() == vector.tobytes()
    assert _hash_numeric_payload(matrix, labels=("same",)) != _hash_numeric_payload(
        vector, labels=("same",)
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "unknown", "schema"),
        ("namespace", "test", "namespace"),
        ("backend", "auto", "backend"),
        ("ordered_features", ("a", "a"), "unique"),
        ("class_labels", (1, 2), "zero-based"),
    ],
)
def test_identity_rejects_invalid_authority(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        identity(**{field: value})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task": "regression", "class_labels": (0,)}, "regression"),
        ({"task": "binary", "class_labels": (0, 1, 2)}, "zero-based"),
        ({"model_params": {1: 0.1}}, "model parameter"),
        ({"model_params": {"learning_rate": float("nan")}}, "finite"),
        ({"library_versions": {1: "2"}}, "library version"),
    ],
)
def test_identity_rejects_invalid_class_or_mapping_values(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        identity(**changes)


@pytest.mark.parametrize(
    "model_params",
    [
        {"nested": {"value": float("nan")}},
        {"nested": [0.1, {"value": float("inf")} ]},
        {"unsupported": {"not", "json"}},
    ],
)
def test_identity_rejects_nested_nonfinite_or_non_json_model_params(model_params):
    with pytest.raises((TypeError, ValueError), match="finite|JSON-compatible"):
        identity(model_params=model_params)
