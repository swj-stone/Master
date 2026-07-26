"""Canonical identities and payloads for coalition-utility cache entries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from types import MappingProxyType

import numpy as np
import pandas as pd

from minppi_mofs.modeling import (
    _empty_subset_predictions,
    _validated_policy,
    fit_predict,
)
from minppi_mofs.utility import compute_utility


CACHE_SCHEMA_VERSION = "coalition_utility_v1"
_NAMESPACES = frozenset({"selection_validation", "baseline_internal", "final_test"})
_TASKS = frozenset({"binary", "multiclass", "regression"})
_BACKENDS = frozenset({"cpu", "cuda"})
_TASK_METRICS = {
    "binary": ("accuracy", "pr_auc", "roc_auc"),
    "multiclass": ("accuracy", "ovr_auc"),
    "regression": ("r2", "rmse"),
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS coalition_utility (
    coalition_key TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
);
"""


def _freeze_json_compatible_model_param(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("model parameter numeric values must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("model parameter keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_json_compatible_model_param(value[key])
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_compatible_model_param(item) for item in value)
    raise TypeError("model parameter values must be JSON-compatible")


def _thaw_json_compatible_model_param(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw_json_compatible_model_param(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json_compatible_model_param(item) for item in value]
    return value


@dataclass(frozen=True)
class CoalitionCacheIdentity:
    schema_version: str
    protocol_hash: str
    dataset_id: str
    seed: int
    namespace: str
    train_features_sha256: str
    train_target_sha256: str
    eval_features_sha256: str
    eval_target_sha256: str
    ordered_features: tuple[str, ...]
    task: str
    class_labels: tuple[int, ...]
    model_params: Mapping[str, object]
    backend: str
    model_seed: int
    library_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        try:
            ordered_features = tuple(self.ordered_features)
        except TypeError as error:
            raise TypeError("cache identity features must be a sequence") from error
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError("cache identity schema is invalid")
        if self.namespace not in _NAMESPACES:
            raise ValueError("cache identity namespace is invalid")
        if self.task not in _TASKS:
            raise ValueError("cache identity task is invalid")
        if self.backend not in _BACKENDS:
            raise ValueError("cache identity backend is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.seed, self.model_seed)
        ):
            raise ValueError("cache identity seeds must be non-negative integers")
        if (
            not ordered_features
            or len(set(ordered_features)) != len(ordered_features)
            or not all(isinstance(item, str) and item for item in ordered_features)
        ):
            raise ValueError("cache identity features must be non-empty unique strings")
        hash_fields = (
            self.protocol_hash,
            self.train_features_sha256,
            self.train_target_sha256,
            self.eval_features_sha256,
            self.eval_target_sha256,
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hash_fields
        ):
            raise ValueError("cache identity hashes must be lowercase SHA-256")
        if self.task == "regression" and self.class_labels:
            raise ValueError("regression cache identity cannot have class labels")
        if self.task == "binary" and self.class_labels != (0, 1):
            raise ValueError("binary class labels must be zero-based and contiguous")
        if self.task == "multiclass" and self.class_labels != tuple(
            range(len(self.class_labels))
        ):
            raise ValueError("multiclass labels must be zero-based and contiguous")
        if not isinstance(self.model_params, Mapping) or any(
            not isinstance(key, str) for key in self.model_params
        ):
            raise TypeError("model parameter keys must be strings")
        frozen_model_params = {
            key: _freeze_json_compatible_model_param(value)
            for key, value in sorted(self.model_params.items())
        }
        if not isinstance(self.library_versions, Mapping) or any(
            not isinstance(key, str) for key in self.library_versions
        ):
            raise TypeError("library version keys must be strings")
        if not all(isinstance(value, str) for value in self.library_versions.values()):
            raise TypeError("library version values must be strings")
        object.__setattr__(
            self,
            "ordered_features",
            ordered_features,
        )
        object.__setattr__(
            self,
            "model_params",
            MappingProxyType(frozen_model_params),
        )
        object.__setattr__(
            self,
            "library_versions",
            MappingProxyType(dict(sorted(self.library_versions.items()))),
        )

    def canonical_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "protocol_hash": self.protocol_hash,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "namespace": self.namespace,
            "train_features_sha256": self.train_features_sha256,
            "train_target_sha256": self.train_target_sha256,
            "eval_features_sha256": self.eval_features_sha256,
            "eval_target_sha256": self.eval_target_sha256,
            "ordered_features": list(self.ordered_features),
            "task": self.task,
            "class_labels": list(self.class_labels),
            "model_params": _thaw_json_compatible_model_param(self.model_params),
            "backend": self.backend,
            "model_seed": self.model_seed,
            "library_versions": dict(self.library_versions),
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CoalitionUtilityRecord:
    coalition: tuple[str, ...]
    utilities: Mapping[str, float]
    prediction_shape: tuple[int, ...]
    prediction_dtype: str
    prediction_sha256: str
    fitted: bool


@dataclass(frozen=True)
class CoalitionCacheStats:
    physical_model_fit_count: int
    cache_hit_count: int
    coalition_count: int


def _record_from_payload(payload: object) -> CoalitionUtilityRecord:
    expected_keys = {
        "coalition",
        "utilities",
        "prediction_shape",
        "prediction_dtype",
        "prediction_sha256",
        "fitted",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("coalition cache payload schema is invalid")

    coalition = payload["coalition"]
    utilities = payload["utilities"]
    prediction_shape = payload["prediction_shape"]
    prediction_dtype = payload["prediction_dtype"]
    prediction_sha256 = payload["prediction_sha256"]
    fitted = payload["fitted"]
    if not isinstance(coalition, list) or not all(
        isinstance(feature, str) and feature for feature in coalition
    ):
        raise ValueError("coalition cache coalition is invalid")
    if not isinstance(utilities, dict) or not all(
        isinstance(name, str)
        and name
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(value)
        for name, value in utilities.items()
    ):
        raise ValueError("coalition cache utilities are invalid")
    if not isinstance(prediction_shape, list) or not all(
        not isinstance(size, bool) and isinstance(size, int) and size >= 0
        for size in prediction_shape
    ):
        raise ValueError("coalition cache prediction shape is invalid")
    if not isinstance(prediction_dtype, str) or not prediction_dtype:
        raise ValueError("coalition cache prediction dtype is invalid")
    if (
        not isinstance(prediction_sha256, str)
        or len(prediction_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prediction_sha256)
    ):
        raise ValueError("coalition cache prediction checksum is invalid")
    if not isinstance(fitted, bool):
        raise ValueError("coalition cache fitted flag is invalid")
    return CoalitionUtilityRecord(
        coalition=tuple(coalition),
        utilities=MappingProxyType(dict(utilities)),
        prediction_shape=tuple(prediction_shape),
        prediction_dtype=prediction_dtype,
        prediction_sha256=prediction_sha256,
        fitted=fitted,
    )


class SQLiteCoalitionStore:
    def __init__(
        self,
        path: Path,
        identity: CoalitionCacheIdentity,
        *,
        commit_every_fits: int,
    ) -> None:
        if (
            isinstance(commit_every_fits, bool)
            or not isinstance(commit_every_fits, int)
            or commit_every_fits < 1
        ):
            raise ValueError("commit_every_fits must be a positive integer")
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("cache path must not be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._commit_every_fits = commit_every_fits
        self._pending_write_count = 0
        self._connection: sqlite3.Connection | None = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(SCHEMA_SQL)
        rows = dict(self._connection.execute("SELECT key, value FROM metadata"))
        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "identity_json": identity.canonical_json(),
            "identity_sha256": identity.digest,
        }
        if not rows:
            persistent_row = self._connection.execute(
                "SELECT 1 FROM coalition_utility LIMIT 1"
            ).fetchone()
            if persistent_row is not None:
                self._connection.close()
                self._connection = None
                raise ValueError("cache metadata is missing for persistent rows")
            self._connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                tuple(expected.items()),
            )
            self._connection.commit()
        elif rows != expected:
            self._connection.close()
            raise ValueError("cache metadata conflicts with identity")

    def __enter__(self) -> SQLiteCoalitionStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                pass

    @property
    def entry_count(self) -> int:
        return self._connection.execute(
            "SELECT COUNT(*) FROM coalition_utility"
        ).fetchone()[0]

    @property
    def pending_write_count(self) -> int:
        return self._pending_write_count

    def validate_all(self) -> tuple[CoalitionUtilityRecord, ...]:
        """Return every structurally validated persistent record."""
        records: list[CoalitionUtilityRecord] = []
        try:
            rows = self._connection.execute(
                "SELECT coalition_key FROM coalition_utility"
            ).fetchall()
            for (coalition_key,) in rows:
                if not isinstance(coalition_key, str):
                    raise ValueError("coalition cache key is invalid")
                try:
                    parsed = json.loads(coalition_key)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "coalition cache key JSON is invalid"
                    ) from error
                if (
                    not isinstance(parsed, list)
                    or not all(
                        isinstance(feature, str) and feature
                        for feature in parsed
                    )
                    or json.dumps(
                        parsed,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    != coalition_key
                ):
                    raise ValueError(
                        "coalition cache key is not canonical"
                    )
                coalition = tuple(parsed)
                if (
                    len(set(coalition)) != len(coalition)
                    or canonical_coalition(
                        self.identity.ordered_features, set(coalition)
                    )
                    != coalition
                ):
                    raise ValueError(
                        "coalition cache key conflicts with identity"
                    )
                record = self.get(coalition)
                if record is None:
                    raise RuntimeError("persistent cache row disappeared")
                records.append(record)
        except (TypeError, ValueError, sqlite3.DatabaseError) as error:
            raise ValueError(
                "coalition cache persistent rows are invalid"
            ) from error
        return tuple(records)

    def get(self, coalition: tuple[str, ...]) -> CoalitionUtilityRecord | None:
        coalition_key = json.dumps(
            list(coalition), ensure_ascii=False, separators=(",", ":")
        )
        row = self._connection.execute(
            "SELECT payload_json, payload_sha256 FROM coalition_utility "
            "WHERE coalition_key = ?",
            (coalition_key,),
        ).fetchone()
        if row is None:
            return None
        payload_json, expected_sha256 = row
        if not isinstance(payload_json, str) or not isinstance(expected_sha256, str):
            raise ValueError("coalition cache payload columns are invalid")
        actual_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("coalition cache payload checksum is invalid")
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("coalition cache payload JSON is invalid") from error
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical != payload_json:
            raise ValueError("coalition cache payload is not canonical")
        record = _record_from_payload(payload)
        if record.coalition != coalition:
            raise ValueError("coalition cache key conflicts with payload")
        return record

    def put(self, record: CoalitionUtilityRecord) -> None:
        coalition_key = json.dumps(
            list(record.coalition), ensure_ascii=False, separators=(",", ":")
        )
        payload = {
            "coalition": list(record.coalition),
            "utilities": dict(record.utilities),
            "prediction_shape": list(record.prediction_shape),
            "prediction_dtype": record.prediction_dtype,
            "prediction_sha256": record.prediction_sha256,
            "fitted": record.fitted,
        }
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        existing = self.get(record.coalition)
        if existing is not None and existing != record:
            raise ValueError("coalition cache write conflicts with existing row")
        if existing is None:
            self._connection.execute(
                "INSERT INTO coalition_utility"
                "(coalition_key, payload_json, payload_sha256) VALUES (?, ?, ?)",
                (coalition_key, payload_json, payload_sha256),
            )
            self._pending_write_count += int(record.fitted)
            if self._pending_write_count >= self._commit_every_fits:
                self.flush()

    def flush(self) -> None:
        if self._connection is None:
            return
        self._connection.commit()
        self._pending_write_count = 0

    def close(self) -> None:
        if self._connection is None:
            return
        try:
            self.flush()
        finally:
            connection = self._connection
            self._connection = None
            connection.close()


def canonical_coalition(
    feature_order: Sequence[str], coalition: AbstractSet[str]
) -> tuple[str, ...]:
    ordered = tuple(feature_order)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("feature order must be non-empty and unique")
    if not isinstance(coalition, AbstractSet):
        raise TypeError("coalition must be a set-like value")
    unknown = set(coalition) - set(ordered)
    if unknown:
        raise ValueError(f"coalition contains features outside authority: {sorted(unknown)}")
    return tuple(feature for feature in ordered if feature in coalition)


def prediction_digest(predictions: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(predictions)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()


def _validated_cache_frame(
    value: object, *, name: str, feature_order: tuple[str, ...]
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if value.columns.duplicated().any():
        raise ValueError(f"{name} columns must be unique")
    if tuple(value.columns) != feature_order:
        raise ValueError(f"{name} column order must match cache identity")
    if len(value) < 1:
        raise ValueError(f"{name} must contain at least one row")
    try:
        numeric = value.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values") from error
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite values")
    return numeric.copy(deep=True)


def _validated_cache_target(
    value: object,
    *,
    name: str,
    row_count: int,
    identity: CoalitionCacheIdentity,
    require_all_classes: bool,
) -> np.ndarray:
    target = np.asarray(value)
    if target.ndim != 1 or len(target) != row_count:
        raise ValueError(f"{name} target row count must match its frame")
    try:
        finite = np.isfinite(target).all()
    except TypeError as error:
        raise ValueError(f"{name} target must be numeric and finite") from error
    if not finite:
        raise ValueError(f"{name} target must be numeric and finite")
    if identity.task == "regression":
        return target.astype(float, copy=True)
    if not np.equal(target, np.floor(target)).all():
        raise ValueError(f"{name} classification target must contain integers")
    labels = target.astype(int, copy=True)
    observed = set(map(int, np.unique(labels)))
    expected = set(identity.class_labels)
    if not observed.issubset(expected) or (
        require_all_classes and observed != expected
    ):
        raise ValueError(f"{name} target classes conflict with cache identity")
    return labels


def _validated_predictions(
    value: object, *, identity: CoalitionCacheIdentity, row_count: int
) -> np.ndarray:
    predictions = np.asarray(value)
    if not np.issubdtype(predictions.dtype, np.number):
        raise ValueError("model predictions must be numeric")
    expected_shape = (
        (row_count,)
        if identity.task == "regression"
        else (row_count, len(identity.class_labels))
    )
    if predictions.shape != expected_shape:
        raise ValueError("model predictions conflict with target/task shape")
    numeric = predictions.astype(float, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError("model predictions must be finite")
    if identity.task != "regression":
        if np.any(numeric < 0.0) or np.any(numeric > 1.0):
            raise ValueError("classification predictions must be probabilities")
        if not np.allclose(numeric.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("classification prediction rows must sum to one")
    return np.array(predictions, copy=True)


class CoalitionUtilityCache:
    def __init__(
        self,
        identity: CoalitionCacheIdentity,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_eval: pd.DataFrame,
        y_eval: np.ndarray,
        store: SQLiteCoalitionStore,
    ) -> None:
        if not isinstance(identity, CoalitionCacheIdentity):
            raise TypeError("identity must be a CoalitionCacheIdentity")
        if not isinstance(store, SQLiteCoalitionStore):
            raise TypeError("store must be a SQLiteCoalitionStore")
        feature_order = identity.ordered_features
        if store.identity.digest != identity.digest:
            raise ValueError("store identity conflicts with cache identity")
        _validated_policy(
            task=identity.task,
            model_params=identity.model_params,
            backend=identity.backend,
            seed=identity.model_seed,
        )
        self.identity = identity
        self.feature_order = feature_order
        self._X_train = _validated_cache_frame(
            X_train, name="X_train", feature_order=feature_order
        )
        self._X_eval = _validated_cache_frame(
            X_eval, name="X_eval", feature_order=feature_order
        )
        self._y_train = _validated_cache_target(
            y_train,
            name="y_train",
            row_count=len(self._X_train),
            identity=identity,
            require_all_classes=True,
        )
        self._y_eval = _validated_cache_target(
            y_eval,
            name="y_eval",
            row_count=len(self._X_eval),
            identity=identity,
            require_all_classes=identity.task == "multiclass",
        )
        actual_hashes = {
            "X_train": hash_numeric_frame(self._X_train),
            "y_train": hash_numeric_array(np.asarray(y_train)),
            "X_eval": hash_numeric_frame(self._X_eval),
            "y_eval": hash_numeric_array(np.asarray(y_eval)),
        }
        expected_hashes = {
            "X_train": identity.train_features_sha256,
            "y_train": identity.train_target_sha256,
            "X_eval": identity.eval_features_sha256,
            "y_eval": identity.eval_target_sha256,
        }
        for name, actual_hash in actual_hashes.items():
            if actual_hash != expected_hashes[name]:
                raise ValueError(f"{name} identity hash mismatch")
        if identity.task == "regression" and len(self._X_eval) < 2:
            raise ValueError(
                "regression evaluation must contain at least two rows"
            )
        self._store = store
        self._records: dict[tuple[str, ...], CoalitionUtilityRecord] = {}
        self._physical_model_fit_count = 0
        self._cache_hit_count = 0

    @property
    def stats(self) -> CoalitionCacheStats:
        return CoalitionCacheStats(
            physical_model_fit_count=self._physical_model_fit_count,
            cache_hit_count=self._cache_hit_count,
            coalition_count=self._store.entry_count,
        )

    def validate_persistent_records(self) -> int:
        """Apply task-level cache semantics to every persistent row."""
        records = self._store.validate_all()
        for record in records:
            key = canonical_coalition(
                self.feature_order, set(record.coalition)
            )
            self._validate_record(record, key)
        return len(records)

    def record(self, coalition: AbstractSet[str]) -> CoalitionUtilityRecord:
        key = canonical_coalition(self.feature_order, coalition)
        cached = self._records.get(key)
        if cached is None:
            cached = self._store.get(key)
        if cached is not None:
            self._validate_record(cached, key)
            self._records[key] = cached
            self._cache_hit_count += 1
            return cached
        if key:
            predictions = fit_predict(
                self._X_train.loc[:, list(key)],
                self._y_train,
                self._X_eval.loc[:, list(key)],
                task=self.identity.task,
                model_params=self.identity.model_params,
                backend=self.identity.backend,
                seed=self.identity.model_seed,
            )
            fitted = True
            self._physical_model_fit_count += 1
        else:
            predictions = _empty_subset_predictions(
                self._y_train,
                task=self.identity.task,
                row_count=len(self._X_eval),
            )
            fitted = False
        predictions = _validated_predictions(
            predictions,
            identity=self.identity,
            row_count=len(self._X_eval),
        )
        utilities = {
            metric: float(compute_utility(self._y_eval, predictions, metric=metric))
            for metric in _TASK_METRICS[self.identity.task]
        }
        result = CoalitionUtilityRecord(
            coalition=key,
            utilities=MappingProxyType(utilities),
            prediction_shape=predictions.shape,
            prediction_dtype=predictions.dtype.str,
            prediction_sha256=prediction_digest(predictions),
            fitted=fitted,
        )
        self._store.put(result)
        self._records[key] = result
        return result

    def _validate_record(
        self, record: CoalitionUtilityRecord, key: tuple[str, ...]
    ) -> None:
        expected_shape = (
            (len(self._X_eval),)
            if self.identity.task == "regression"
            else (len(self._X_eval), len(self.identity.class_labels))
        )
        if (
            record.coalition != key
            or set(record.utilities) != set(_TASK_METRICS[self.identity.task])
            or record.prediction_shape != expected_shape
            or record.fitted is not bool(key)
        ):
            raise ValueError("cached record conflicts with cache identity or task")
        try:
            prediction_dtype = np.dtype(record.prediction_dtype)
        except TypeError as error:
            raise ValueError("cached record prediction dtype is invalid") from error
        if not np.issubdtype(prediction_dtype, np.number):
            raise ValueError("cached record prediction dtype must be numeric")

    def evaluate(self, coalition: AbstractSet[str], utility: str) -> float:
        if utility not in _TASK_METRICS[self.identity.task]:
            raise ValueError(
                f"utility must be one of {list(_TASK_METRICS[self.identity.task])}"
            )
        return self.record(coalition).utilities[utility]


class CoalitionUtilityView:
    def __init__(
        self, cache: CoalitionUtilityCache, *, utility: str, alpha: float
    ) -> None:
        if utility not in _TASK_METRICS[cache.identity.task]:
            raise ValueError(
                f"utility must be one of {list(_TASK_METRICS[cache.identity.task])}"
            )
        self._cache = cache
        self._utility = utility
        self._seen: set[tuple[str, ...]] = set()
        self.set_threshold(alpha)

    @property
    def feature_order(self) -> tuple[str, ...]:
        return self._cache.feature_order

    def set_threshold(self, alpha: float) -> None:
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or not 0.0 < float(alpha) <= 1.0
        ):
            raise ValueError("alpha must be in (0, 1]")
        self.acc_full = self.evaluate(set(self._cache.feature_order))
        self.threshold = self.acc_full * float(alpha)

    def evaluate(self, coalition: AbstractSet[str]) -> float:
        self._seen.add(
            canonical_coalition(self._cache.feature_order, coalition)
        )
        return self._cache.evaluate(coalition, self._utility)

    def is_winning(self, coalition: AbstractSet[str]) -> bool:
        return self.evaluate(coalition) >= self.threshold

    def eval_marginal(self, coalition: set[str], feature: str) -> int:
        if feature not in self._cache.feature_order:
            raise ValueError("feature is outside cache authority")
        return int(self.is_winning(coalition | {feature})) - int(
            self.is_winning(coalition)
        )

    @property
    def model_fit_count(self) -> int:
        return sum(bool(key) for key in self._seen)

    @property
    def coalition_count(self) -> int:
        return len(self._seen)


def _hash_numeric_payload(
    values: np.ndarray, *, labels: Sequence[str] = ()
) -> str:
    contiguous = np.ascontiguousarray(values)
    metadata = json.dumps(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "labels": list(labels),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(metadata + b"\0" + contiguous.tobytes()).hexdigest()


def hash_numeric_frame(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("frame must have unique columns")
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(values.astype(float)).all():
        raise ValueError("frame must be finite")
    return _hash_numeric_payload(values, labels=tuple(map(str, frame.columns)))


def hash_numeric_array(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.ndim != 1 or not np.isfinite(array.astype(float)).all():
        raise ValueError("array must be finite and one-dimensional")
    return _hash_numeric_payload(array)
