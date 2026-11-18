"""Deterministic cached utility evaluator shared by coalition baselines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from minppi_mofs.coalition_cache import (
    CoalitionUtilityCache as PersistentCoalitionUtilityCache,
    SQLiteCoalitionStore,
    hash_numeric_array,
    hash_numeric_frame,
)
from minppi_mofs.modeling import evaluate_fixed_subset


def default_utility(task: str) -> str:
    return {"binary": "roc_auc", "multiclass": "ovr_auc", "regression": "r2"}.get(
        task, ""
    )


_IDENTITY_SCHEMA = "baseline_utility_v1"
_NAMESPACE = "baseline_internal"
_TASKS = frozenset({"binary", "multiclass", "regression"})
_BACKENDS = frozenset({"cpu", "cuda"})


def _hash_is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("baseline model parameters must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("baseline model parameter keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_json_value(value[key])
                for key in sorted(value)
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError("baseline model parameters must be JSON-compatible")


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw_json_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _baseline_library_versions() -> Mapping[str, str]:
    try:
        xgboost_version = metadata.version("xgboost")
    except metadata.PackageNotFoundError:
        xgboost_version = "unavailable"
    return MappingProxyType(
        {"numpy": np.__version__, "xgboost": xgboost_version}
    )


def _read_only_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frozen = frame.copy(deep=True)
    for block in frozen._mgr.blocks:  # pandas has no public read-only frame API
        block.values.setflags(write=False)
    return frozen


@dataclass(frozen=True, eq=False)
class PreparedBaselineSplit:
    """One immutable materialization of the legacy baseline split policy."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_eval: pd.DataFrame
    y_eval: np.ndarray
    ordered_features: tuple[str, ...]
    task: str
    seed: int
    test_size: float
    max_samples: int
    raw_features_sha256: str
    raw_target_sha256: str
    sample_indices: tuple[int, ...]
    train_positions: tuple[int, ...]
    eval_positions: tuple[int, ...]
    train_features_sha256: str = ""
    train_target_sha256: str = ""
    eval_features_sha256: str = ""
    eval_target_sha256: str = ""

    def __post_init__(self) -> None:
        X_train = _read_only_frame(self.X_train)
        X_eval = _read_only_frame(self.X_eval)
        y_train = np.array(self.y_train, copy=True)
        y_eval = np.array(self.y_eval, copy=True)
        y_train.setflags(write=False)
        y_eval.setflags(write=False)
        object.__setattr__(self, "X_train", X_train)
        object.__setattr__(self, "X_eval", X_eval)
        object.__setattr__(self, "y_train", y_train)
        object.__setattr__(self, "y_eval", y_eval)
        object.__setattr__(
            self, "ordered_features", tuple(self.ordered_features)
        )
        object.__setattr__(self, "sample_indices", tuple(self.sample_indices))
        object.__setattr__(
            self, "train_positions", tuple(self.train_positions)
        )
        object.__setattr__(
            self, "eval_positions", tuple(self.eval_positions)
        )
        actual_hashes = {
            "train_features_sha256": hash_numeric_frame(X_train),
            "train_target_sha256": hash_numeric_array(y_train),
            "eval_features_sha256": hash_numeric_frame(X_eval),
            "eval_target_sha256": hash_numeric_array(y_eval),
        }
        for field, actual in actual_hashes.items():
            supplied = getattr(self, field)
            if supplied and supplied != actual:
                raise ValueError(
                    f"prepared baseline {field} conflicts with content"
                )
            object.__setattr__(self, field, actual)
        self.validate()

    def validate(self) -> None:
        if (
            self.task not in _TASKS
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples < 2
            or isinstance(self.test_size, bool)
            or not isinstance(self.test_size, (int, float))
            or not 0.0 < float(self.test_size) < 1.0
        ):
            raise ValueError("prepared baseline split policy is invalid")
        if (
            not self.ordered_features
            or len(set(self.ordered_features)) != len(self.ordered_features)
            or not all(
                isinstance(feature, str) and feature
                for feature in self.ordered_features
            )
            or tuple(self.X_train.columns) != self.ordered_features
            or tuple(self.X_eval.columns) != self.ordered_features
        ):
            raise ValueError("prepared baseline feature universe is invalid")
        if (
            len(self.X_train) != len(self.y_train)
            or len(self.X_eval) != len(self.y_eval)
            or not self.sample_indices
            or len(set(self.sample_indices)) != len(self.sample_indices)
            or any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                for position in self.sample_indices
            )
            or not self.train_positions
            or not self.eval_positions
            or set(self.train_positions) & set(self.eval_positions)
            or sorted(self.train_positions + self.eval_positions)
            != list(range(len(self.sample_indices)))
        ):
            raise ValueError("prepared baseline split indices are invalid")
        hashes = (
            self.raw_features_sha256,
            self.raw_target_sha256,
            self.train_features_sha256,
            self.train_target_sha256,
            self.eval_features_sha256,
            self.eval_target_sha256,
        )
        if not all(_hash_is_sha256(value) for value in hashes):
            raise ValueError("prepared baseline hashes are invalid")
        actual_hashes = (
            hash_numeric_frame(self.X_train),
            hash_numeric_array(self.y_train),
            hash_numeric_frame(self.X_eval),
            hash_numeric_array(self.y_eval),
        )
        if actual_hashes != hashes[2:]:
            raise ValueError("prepared baseline read-only content changed")
        if (
            self.y_train.flags.writeable
            or self.y_eval.flags.writeable
            or any(
                block.values.flags.writeable
                for frame in (self.X_train, self.X_eval)
                for block in frame._mgr.blocks
            )
        ):
            raise ValueError("prepared baseline arrays must remain read-only")

    @property
    def digest(self) -> str:
        document = {
            "raw_features_sha256": self.raw_features_sha256,
            "raw_target_sha256": self.raw_target_sha256,
            "ordered_features": list(self.ordered_features),
            "task": self.task,
            "seed": self.seed,
            "test_size": self.test_size,
            "max_samples": self.max_samples,
            "sample_indices": list(self.sample_indices),
            "train_positions": list(self.train_positions),
            "eval_positions": list(self.eval_positions),
            "train_features_sha256": self.train_features_sha256,
            "train_target_sha256": self.train_target_sha256,
            "eval_features_sha256": self.eval_features_sha256,
            "eval_target_sha256": self.eval_target_sha256,
        }
        canonical = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BaselineUtilityIdentity:
    schema_version: str
    namespace: str
    raw_features_sha256: str
    raw_target_sha256: str
    train_features_sha256: str
    train_target_sha256: str
    eval_features_sha256: str
    eval_target_sha256: str
    prepared_split_digest: str
    ordered_features: tuple[str, ...]
    task: str
    utility: str
    model_params: Mapping[str, object]
    backend: str
    seed: int
    test_size: float
    max_samples: int
    library_versions: Mapping[str, str]
    store_identity_digest: str
    sample_indices: tuple[int, ...]
    train_positions: tuple[int, ...]
    eval_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != _IDENTITY_SCHEMA
            or self.namespace != _NAMESPACE
            or self.task not in _TASKS
            or self.backend not in _BACKENDS
        ):
            raise ValueError("baseline utility identity policy is invalid")
        ordered = tuple(self.ordered_features)
        if (
            not ordered
            or len(set(ordered)) != len(ordered)
            or not all(isinstance(feature, str) and feature for feature in ordered)
        ):
            raise ValueError("baseline utility feature universe is invalid")
        hashes = (
            self.raw_features_sha256,
            self.raw_target_sha256,
            self.train_features_sha256,
            self.train_target_sha256,
            self.eval_features_sha256,
            self.eval_target_sha256,
            self.prepared_split_digest,
            self.store_identity_digest,
        )
        if not all(_hash_is_sha256(value) for value in hashes):
            raise ValueError("baseline utility identity hashes are invalid")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or isinstance(self.test_size, bool)
            or not isinstance(self.test_size, (int, float))
            or not 0.0 < float(self.test_size) < 1.0
            or isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples < 2
        ):
            raise ValueError("baseline utility split policy is invalid")
        if not isinstance(self.utility, str) or not self.utility:
            raise ValueError("baseline utility metric is invalid")
        if (
            not self.sample_indices
            or len(set(self.sample_indices)) != len(self.sample_indices)
            or any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                for positions in (
                    self.sample_indices,
                    self.train_positions,
                    self.eval_positions,
                )
                for position in positions
            )
            or not self.train_positions
            or not self.eval_positions
            or set(self.train_positions) & set(self.eval_positions)
            or sorted(self.train_positions + self.eval_positions)
            != list(range(len(self.sample_indices)))
        ):
            raise ValueError("baseline utility split indices are invalid")
        if not isinstance(self.model_params, Mapping):
            raise TypeError("baseline utility model_params must be a mapping")
        if not isinstance(self.library_versions, Mapping) or not all(
            isinstance(key, str)
            and isinstance(value, str)
            and key
            and value
            for key, value in self.library_versions.items()
        ):
            raise TypeError("baseline utility library_versions are invalid")
        object.__setattr__(self, "ordered_features", ordered)
        object.__setattr__(
            self,
            "model_params",
            _freeze_json_value(self.model_params),
        )
        object.__setattr__(
            self,
            "library_versions",
            MappingProxyType(dict(sorted(self.library_versions.items()))),
        )
        object.__setattr__(
            self, "sample_indices", tuple(self.sample_indices)
        )
        object.__setattr__(
            self, "train_positions", tuple(self.train_positions)
        )
        object.__setattr__(
            self, "eval_positions", tuple(self.eval_positions)
        )

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "namespace": self.namespace,
                "raw_features_sha256": self.raw_features_sha256,
                "raw_target_sha256": self.raw_target_sha256,
                "train_features_sha256": self.train_features_sha256,
                "train_target_sha256": self.train_target_sha256,
                "eval_features_sha256": self.eval_features_sha256,
                "eval_target_sha256": self.eval_target_sha256,
                "prepared_split_digest": self.prepared_split_digest,
                "ordered_features": list(self.ordered_features),
                "task": self.task,
                "utility": self.utility,
                "model_params": _thaw_json_value(self.model_params),
                "backend": self.backend,
                "seed": self.seed,
                "test_size": self.test_size,
                "max_samples": self.max_samples,
                "library_versions": dict(self.library_versions),
                "store_identity_digest": self.store_identity_digest,
                "sample_indices": list(self.sample_indices),
                "train_positions": list(self.train_positions),
                "eval_positions": list(self.eval_positions),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()


def _raw_baseline_authority(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[tuple[str, ...], str, str]:
    if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
        raise ValueError("X must be a DataFrame with unique columns")
    features = tuple(X.columns)
    if (
        not features
        or not all(isinstance(feature, str) and feature for feature in features)
    ):
        raise ValueError("X feature universe must be non-empty strings")
    target = np.asarray(y)
    if len(X) != len(target):
        raise ValueError("X and y row counts must match")
    return (
        features,
        hash_numeric_frame(X),
        hash_numeric_array(target),
    )


def _prepare_legacy_baseline_internal_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    seed: int,
    test_size: float = 0.2,
    max_samples: int = 5000,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Preserve the pre-shared-cache baseline input domain and split behavior."""
    if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
        raise ValueError("X must be a DataFrame with unique columns")
    if (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples < 2
    ):
        raise ValueError("max_samples must be an integer of at least two")
    features = tuple(X.columns)
    target = np.asarray(y)
    if len(X) != len(target):
        raise ValueError("X and y row counts must match")
    sampled = X
    if len(sampled) > max_samples:
        material = f"{seed}\0" + "\0".join(features)
        stable = int.from_bytes(
            hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"
        )
        indices = np.random.default_rng(stable).choice(
            len(sampled), size=max_samples, replace=False
        )
        sampled = sampled.iloc[indices].reset_index(drop=True)
        target = target[indices]
    positions = np.arange(len(sampled))
    stratify = target if task != "regression" else None
    try:
        train_pos, eval_pos = train_test_split(
            positions,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_pos, eval_pos = train_test_split(
            positions, test_size=test_size, random_state=seed
        )
    return (
        sampled.iloc[train_pos].reset_index(drop=True),
        target[train_pos],
        sampled.iloc[eval_pos].reset_index(drop=True),
        target[eval_pos],
    )


def _prepare_baseline_internal_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    seed: int,
    test_size: float = 0.2,
    max_samples: int = 5000,
) -> PreparedBaselineSplit:
    """Reproduce and freeze the legacy sample cap and internal holdout once."""
    features, raw_features_sha256, raw_target_sha256 = (
        _raw_baseline_authority(X, y)
    )
    if (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples < 2
    ):
        raise ValueError("max_samples must be an integer of at least two")
    target = np.asarray(y)
    sample_indices = np.arange(len(X))
    if len(X) > max_samples:
        material = f"{seed}\0" + "\0".join(features)
        stable = int.from_bytes(
            hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"
        )
        sample_indices = np.random.default_rng(stable).choice(
            len(X), size=max_samples, replace=False
        )
    sampled = X.iloc[sample_indices].reset_index(drop=True)
    sampled_target = target[sample_indices]
    positions = np.arange(len(sampled))
    stratify = sampled_target if task != "regression" else None
    try:
        train_pos, eval_pos = train_test_split(
            positions,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_pos, eval_pos = train_test_split(
            positions, test_size=test_size, random_state=seed
        )
    return PreparedBaselineSplit(
        X_train=sampled.iloc[train_pos].reset_index(drop=True),
        y_train=sampled_target[train_pos],
        X_eval=sampled.iloc[eval_pos].reset_index(drop=True),
        y_eval=sampled_target[eval_pos],
        ordered_features=features,
        task=task,
        seed=seed,
        test_size=test_size,
        max_samples=max_samples,
        raw_features_sha256=raw_features_sha256,
        raw_target_sha256=raw_target_sha256,
        sample_indices=tuple(map(int, sample_indices)),
        train_positions=tuple(map(int, train_pos)),
        eval_positions=tuple(map(int, eval_pos)),
    )


def make_baseline_utility_identity(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    prepared: PreparedBaselineSplit,
    task: str,
    utility: str,
    model_params: Mapping[str, object],
    backend: str,
    seed: int,
    store_identity_digest: str,
    library_versions: Mapping[str, str] | None = None,
    test_size: float = 0.2,
    max_samples: int = 5000,
) -> BaselineUtilityIdentity:
    if not isinstance(prepared, PreparedBaselineSplit):
        raise TypeError("prepared must be a PreparedBaselineSplit")
    prepared.validate()
    features, raw_features_sha256, raw_target_sha256 = (
        _raw_baseline_authority(X, y)
    )
    if (
        features != prepared.ordered_features
        or raw_features_sha256 != prepared.raw_features_sha256
        or raw_target_sha256 != prepared.raw_target_sha256
        or task != prepared.task
        or seed != prepared.seed
        or test_size != prepared.test_size
        or max_samples != prepared.max_samples
    ):
        raise ValueError("prepared baseline split conflicts with raw authority")
    return BaselineUtilityIdentity(
        schema_version=_IDENTITY_SCHEMA,
        namespace=_NAMESPACE,
        raw_features_sha256=raw_features_sha256,
        raw_target_sha256=raw_target_sha256,
        train_features_sha256=prepared.train_features_sha256,
        train_target_sha256=prepared.train_target_sha256,
        eval_features_sha256=prepared.eval_features_sha256,
        eval_target_sha256=prepared.eval_target_sha256,
        prepared_split_digest=prepared.digest,
        ordered_features=features,
        task=task,
        utility=utility,
        model_params=model_params,
        backend=backend,
        seed=seed,
        test_size=test_size,
        max_samples=max_samples,
        library_versions=library_versions or _baseline_library_versions(),
        store_identity_digest=store_identity_digest,
        sample_indices=prepared.sample_indices,
        train_positions=prepared.train_positions,
        eval_positions=prepared.eval_positions,
    )


@dataclass(frozen=True)
class CoalitionUtilityDiagnostics:
    logical_evaluation_count: int
    logical_unique_coalition_count: int
    physical_model_fit_count: int
    cache_hit_count: int

    def __post_init__(self) -> None:
        values = (
            self.logical_evaluation_count,
            self.logical_unique_coalition_count,
            self.physical_model_fit_count,
            self.cache_hit_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ) or self.logical_unique_coalition_count > self.logical_evaluation_count:
            raise ValueError("coalition utility diagnostics are invalid")


def supplied_evaluator_diagnostics(
    evaluator: Callable[[set[str]], float],
) -> CoalitionUtilityDiagnostics:
    try:
        value = evaluator.diagnostics_snapshot  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError(
            "coalition_utility must expose canonical diagnostics_snapshot"
        ) from error
    if not isinstance(value, CoalitionUtilityDiagnostics):
        raise TypeError(
            "coalition_utility diagnostics_snapshot has an invalid protocol"
        )
    return value


def validated_supplied_coalition_utility(
    coalition_utility: Callable[[set[str]], float],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    utility: str,
    model_params: Mapping[str, object],
    backend: str,
    seed: int,
    test_size: float = 0.2,
    max_samples: int = 5000,
) -> Callable[[set[str]], float]:
    """Validate immutable input, split, policy, store, and diagnostics authority."""
    if not callable(coalition_utility):
        raise TypeError("coalition_utility must be callable")
    features, raw_features_sha256, raw_target_sha256 = (
        _raw_baseline_authority(X, y)
    )
    try:
        identity = coalition_utility.identity  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError(
            "coalition_utility must expose a canonical frozen identity"
        ) from error
    if not isinstance(identity, BaselineUtilityIdentity):
        raise TypeError(
            "coalition_utility identity must be BaselineUtilityIdentity"
        )
    try:
        prepared = coalition_utility.prepared_split  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError(
            "coalition_utility must expose its frozen PreparedBaselineSplit"
        ) from error
    if not isinstance(prepared, PreparedBaselineSplit):
        raise TypeError(
            "coalition_utility prepared_split must be PreparedBaselineSplit"
        )
    prepared.validate()
    prepared_authority = {
        "raw_features_sha256": prepared.raw_features_sha256,
        "raw_target_sha256": prepared.raw_target_sha256,
        "train_features_sha256": prepared.train_features_sha256,
        "train_target_sha256": prepared.train_target_sha256,
        "eval_features_sha256": prepared.eval_features_sha256,
        "eval_target_sha256": prepared.eval_target_sha256,
        "prepared_split_digest": prepared.digest,
        "ordered_features": prepared.ordered_features,
        "task": prepared.task,
        "seed": prepared.seed,
        "test_size": prepared.test_size,
        "max_samples": prepared.max_samples,
        "sample_indices": prepared.sample_indices,
        "train_positions": prepared.train_positions,
        "eval_positions": prepared.eval_positions,
    }
    for field, expected in prepared_authority.items():
        if getattr(identity, field) != expected:
            raise ValueError(
                f"coalition_utility prepared {field} conflicts with identity"
            )
    expected_model_params = _freeze_json_value(model_params)
    conflicts = {
        "namespace": (identity.namespace, _NAMESPACE),
        "raw_features_sha256": (
            identity.raw_features_sha256,
            raw_features_sha256,
        ),
        "raw_target_sha256": (
            identity.raw_target_sha256,
            raw_target_sha256,
        ),
        "feature universe/order": (identity.ordered_features, features),
        "task": (identity.task, task),
        "utility": (identity.utility, utility),
        "model_params": (identity.model_params, expected_model_params),
        "backend": (identity.backend, backend),
        "seed": (identity.seed, seed),
        "test_size": (identity.test_size, test_size),
        "max_samples": (identity.max_samples, max_samples),
    }
    for name, (actual, expected) in conflicts.items():
        if actual != expected:
            raise ValueError(
                f"coalition_utility {name} conflicts with requested authority"
            )
    if not all(
        _hash_is_sha256(value)
        for value in (
            identity.train_features_sha256,
            identity.train_target_sha256,
            identity.eval_features_sha256,
            identity.eval_target_sha256,
            identity.prepared_split_digest,
            identity.store_identity_digest,
        )
    ):
        raise ValueError("coalition_utility split/store identity is invalid")
    supplied_evaluator_diagnostics(coalition_utility)
    return coalition_utility


def update_supplied_diagnostics(
    diagnostics: MutableMapping[str, int] | None,
    *,
    evaluator: Callable[[set[str]], float],
    before: CoalitionUtilityDiagnostics,
    logical_evaluation_count: int,
    seen: set[tuple[str, ...]],
) -> None:
    after = supplied_evaluator_diagnostics(evaluator)
    physical_fits = (
        after.physical_model_fit_count - before.physical_model_fit_count
    )
    cache_hits = after.cache_hit_count - before.cache_hit_count
    if (
        after.logical_evaluation_count - before.logical_evaluation_count
        != logical_evaluation_count
        or physical_fits < 0
        or cache_hits < 0
    ):
        raise ValueError(
            "coalition_utility diagnostics conflict with method calls"
        )
    if diagnostics is not None:
        diagnostics.update(
            coalition_count=len(seen),
            model_fit_count=sum(bool(key) for key in seen),
            logical_evaluation_count=logical_evaluation_count,
            logical_unique_coalition_count=len(seen),
            physical_model_fit_count=physical_fits,
            cache_hit_count=cache_hits,
        )


class CoalitionUtility:
    """Evaluate ordered coalitions on one frozen internal holdout split."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        task: str,
        utility: str,
        model_params: Mapping[str, object],
        backend: str,
        seed: int,
        test_size: float = 0.2,
        max_samples: int = 5000,
        diagnostics: MutableMapping[str, int] | None = None,
    ) -> None:
        X_train, y_train, X_eval, y_eval = (
            _prepare_legacy_baseline_internal_split(
                X,
                y,
                task=task,
                seed=seed,
                test_size=test_size,
                max_samples=max_samples,
            )
        )
        features = tuple(X.columns)
        model_features = (
            features
            if all(isinstance(feature, str) for feature in features)
            else tuple(f"f{index}" for index in range(len(features)))
        )
        if model_features != features:
            X_train = X_train.copy()
            X_eval = X_eval.copy()
            X_train.columns = model_features
            X_eval.columns = model_features
        self._features = features
        self._model_features = model_features
        self._X_train = X_train
        self._X_eval = X_eval
        self._y_train = y_train
        self._y_eval = y_eval
        self._task = task
        self._utility = utility
        self._model_params = dict(model_params)
        self._backend = backend
        self._seed = seed
        self._test_size = test_size
        self._max_samples = max_samples
        self._cache: dict[tuple[str, ...], float] = {}
        self._persistent_cache: PersistentCoalitionUtilityCache | None = None
        self._seen: set[tuple[str, ...]] = set()
        self._logical_evaluation_count = 0
        self._diagnostics = diagnostics
        self._prepared_split: PreparedBaselineSplit | None = None
        self._identity: BaselineUtilityIdentity | None = None
        self._update_diagnostics()

    @property
    def identity(self) -> BaselineUtilityIdentity:
        if self._identity is None:
            raise TypeError(
                "legacy coalition utility has no canonical shared identity"
            )
        return self._identity

    @property
    def prepared_split(self) -> PreparedBaselineSplit:
        if self._prepared_split is None:
            raise TypeError(
                "legacy coalition utility has no frozen shared split"
            )
        return self._prepared_split

    @property
    def feature_order(self) -> tuple[str, ...]:
        return self._features

    @property
    def task(self) -> str:
        return self._task

    @property
    def utility(self) -> str:
        return self._utility

    @property
    def model_params(self) -> Mapping[str, object]:
        return self._model_params

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def test_size(self) -> float:
        return self._test_size

    @property
    def max_samples(self) -> int:
        return self._max_samples

    @property
    def diagnostics_snapshot(self) -> CoalitionUtilityDiagnostics:
        if self._persistent_cache is None:
            return CoalitionUtilityDiagnostics(
                logical_evaluation_count=self._logical_evaluation_count,
                logical_unique_coalition_count=len(self._seen),
                physical_model_fit_count=sum(bool(key) for key in self._cache),
                cache_hit_count=(
                    self._logical_evaluation_count - len(self._seen)
                ),
            )
        stats = self._persistent_cache.stats
        return CoalitionUtilityDiagnostics(
            logical_evaluation_count=self._logical_evaluation_count,
            logical_unique_coalition_count=len(self._seen),
            physical_model_fit_count=stats.physical_model_fit_count,
            cache_hit_count=stats.cache_hit_count,
        )

    def _update_diagnostics(self) -> None:
        if self._diagnostics is not None:
            snapshot = self.diagnostics_snapshot
            self._diagnostics["coalition_count"] = (
                snapshot.logical_unique_coalition_count
            )
            self._diagnostics["model_fit_count"] = sum(
                bool(key) for key in self._seen
            )
            if self._persistent_cache is not None:
                self._diagnostics["logical_evaluation_count"] = (
                    snapshot.logical_evaluation_count
                )
                self._diagnostics["logical_unique_coalition_count"] = (
                    snapshot.logical_unique_coalition_count
                )
                self._diagnostics["physical_model_fit_count"] = (
                    snapshot.physical_model_fit_count
                )
                self._diagnostics["cache_hit_count"] = snapshot.cache_hit_count

    def __call__(self, coalition: set[str]) -> float:
        if not isinstance(coalition, set) or not coalition.issubset(self._features):
            raise ValueError("coalition contains an unknown feature")
        key = tuple(feature for feature in self._features if feature in coalition)
        self._logical_evaluation_count += 1
        self._seen.add(key)
        if self._persistent_cache is not None:
            result = self._persistent_cache.evaluate(
                coalition, self._utility
            )
            self._update_diagnostics()
            return result
        if key not in self._cache:
            model_key = tuple(
                self._model_features[index]
                for index, feature in enumerate(self._features)
                if feature in coalition
            )
            self._cache[key] = evaluate_fixed_subset(
                self._X_train,
                self._y_train,
                self._X_eval,
                self._y_eval,
                model_key,
                task=self._task,
                utility=self._utility,
                model_params=self._model_params,
                backend=self._backend,
                seed=self._seed,
            )
        self._update_diagnostics()
        return self._cache[key]

    @classmethod
    def _from_persistent_cache(
        cls,
        cache: PersistentCoalitionUtilityCache,
        *,
        identity: BaselineUtilityIdentity,
        prepared: PreparedBaselineSplit,
        diagnostics: MutableMapping[str, int] | None,
    ) -> CoalitionUtility:
        self = cls.__new__(cls)
        self._features = cache.feature_order
        self._model_features = cache.feature_order
        self._X_train = None
        self._X_eval = None
        self._y_train = None
        self._y_eval = None
        self._task = cache.identity.task
        self._utility = identity.utility
        self._model_params = cache.identity.model_params
        self._backend = cache.identity.backend
        self._seed = cache.identity.model_seed
        self._test_size = identity.test_size
        self._max_samples = identity.max_samples
        self._cache = {}
        self._persistent_cache = cache
        self._seen = set()
        self._logical_evaluation_count = 0
        self._diagnostics = diagnostics
        self._identity = identity
        self._prepared_split = prepared
        self._update_diagnostics()
        return self
def make_shared_baseline_utility(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    utility: str,
    model_params: Mapping[str, object],
    backend: str,
    seed: int,
    store: SQLiteCoalitionStore,
    diagnostics: MutableMapping[str, int] | None,
    test_size: float = 0.2,
    max_samples: int = 5000,
    prepared: PreparedBaselineSplit | None = None,
) -> CoalitionUtility:
    """Create one cache-backed evaluator over the legacy internal split."""
    if not isinstance(store, SQLiteCoalitionStore):
        raise TypeError("store must be a SQLiteCoalitionStore")
    if store.identity.namespace != "baseline_internal":
        raise ValueError("shared baseline store must use baseline_internal namespace")
    if prepared is None:
        prepared = _prepare_baseline_internal_split(
            X,
            y,
            task=task,
            seed=seed,
            test_size=test_size,
            max_samples=max_samples,
        )
    identity = make_baseline_utility_identity(
        X,
        y,
        prepared=prepared,
        task=task,
        utility=utility,
        model_params=model_params,
        backend=backend,
        seed=seed,
        store_identity_digest=store.identity.digest,
        library_versions=store.identity.library_versions,
        test_size=test_size,
        max_samples=max_samples,
    )
    cache = PersistentCoalitionUtilityCache(
        store.identity,
        prepared.X_train,
        prepared.y_train,
        prepared.X_eval,
        prepared.y_eval,
        store,
    )
    cache.validate_persistent_records()
    evaluator = CoalitionUtility._from_persistent_cache(
        cache,
        identity=identity,
        prepared=prepared,
        diagnostics=diagnostics,
    )
    validated_supplied_coalition_utility(
        evaluator,
        X,
        y,
        task=task,
        utility=utility,
        model_params=model_params,
        backend=backend,
        seed=seed,
        test_size=test_size,
        max_samples=max_samples,
    )
    return evaluator
