"""Immutable identities and canonical schemas for grouped estimator bundles."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Set as AbstractSet, Sequence
from dataclasses import dataclass, replace
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from minppi_mofs.coalition_cache import (
    CACHE_SCHEMA_VERSION,
    CoalitionCacheIdentity,
    CoalitionCacheStats,
    CoalitionUtilityCache,
    CoalitionUtilityView,
    SQLiteCoalitionStore,
    canonical_coalition,
    hash_numeric_array,
    hash_numeric_frame,
)
from minppi_mofs.experiment_protocol import (
    _validate_data_manifest_row,
    atomic_write_json,
    read_canonical_json,
    resolved_protocol_hash,
)
from minppi_mofs.min_ppi import estimate_minppi_exact
from minppi_mofs.nested_estimation import (
    NestedProgress,
    NestedGuidedSchedule,
    NestedRandomSchedule,
    derive_hybrid_checkpoints,
    estimate_guided_checkpoints,
    estimate_random_checkpoints,
    make_nested_guided_schedule,
    make_nested_random_schedule,
    materialize_exact_checkpoints,
    nested_guided_schedule_digest,
    nested_random_schedule_digest,
)
from minppi_mofs.shapg_integration import build_mi_correlation_guidance

if __package__:
    from .icomp2026_progress import (
        EstimatorProgressEvent,
        EstimatorProgressState,
        estimate_eta_s,
        progress_from_document,
        progress_to_document,
    )
else:
    from icomp2026_progress import (  # type: ignore[no-redef]
        EstimatorProgressEvent,
        EstimatorProgressState,
        estimate_eta_s,
        progress_from_document,
        progress_to_document,
    )

if TYPE_CHECKING:
    from experiments.icomp2026_stages import StageContext


ESTIMATOR_BUNDLE_SCHEMA_VERSION = 1

_CANONICAL_METHODS = (
    "exact",
    "random_mc",
    "mi_correlation_guided",
    "hybrid",
)
_METHODS = frozenset(_CANONICAL_METHODS)
_TASK_UTILITIES = {
    "binary": frozenset({"accuracy", "pr_auc", "roc_auc"}),
    "multiclass": frozenset({"accuracy", "ovr_auc"}),
    "regression": frozenset({"r2", "rmse"}),
}
_BACKENDS = frozenset({"cpu", "cuda"})
_GROUPED_EXECUTION_MODE = "grouped_nested_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_DATASET_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?"
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "request_digest",
        "checkpoints",
        "physical_model_fit_count",
        "cache_hit_count",
        "group_runtime_s",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "method",
        "budget",
        "scores",
        "logical_evaluation_count",
        "logical_unique_coalition_count",
        "physical_model_fit_count",
        "cache_hit_count",
        "schedule_digest",
        "component_runtime_s",
    }
)


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _utf8_text(value: str, context: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{context} must be valid UTF-8") from exc
    return value


def _safe_dataset_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("dataset_id is not a safe identifier")
    identifier = _utf8_text(value, "dataset_id")
    reserved_base = identifier.split(".", maxsplit=1)[0].upper()
    if (
        _SAFE_DATASET_RE.fullmatch(identifier) is None
        or identifier in {".", ".."}
        or reserved_base in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError("dataset_id is not a safe identifier")
    return identifier


def _validate_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, context: str) -> int:
    if not _is_exact_int(value) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: object, context: str) -> int:
    if not _is_exact_int(value) or value < 0:
        raise ValueError(f"{context} counter must be a non-negative integer")
    return value


def _nonnegative_finite_float(value: object, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{context} runtime must be finite and non-negative")
    return float(value)


def _freeze_json_value(value: object, context: str) -> object:
    if isinstance(value, str):
        return _utf8_text(value, context)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} numeric values must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{context} keys must be strings")
        keys = tuple(
            _utf8_text(key, f"{context} key") for key in value
        )
        return MappingProxyType(
            {
                key: _freeze_json_value(value[key], context)
                for key in sorted(keys)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, context) for item in value)
    raise TypeError(f"{context} values must be JSON-compatible")


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _owned_string_tuple(
    value: object, context: str, *, nonempty: bool = True
) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise ValueError(f"{context} must be an immutable string sequence")
    result = tuple(value)
    if (
        (nonempty and not result)
        or not all(isinstance(item, str) and item for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{context} must be non-empty unique strings")
    return tuple(_utf8_text(item, context) for item in result)


def _canonical_methods(value: object) -> tuple[str, ...]:
    methods = _owned_string_tuple(value, "methods")
    unknown_methods = sorted(set(methods) - _METHODS)
    if unknown_methods:
        raise ValueError(f"methods contain unknown values: {unknown_methods}")
    canonical = tuple(
        method for method in _CANONICAL_METHODS if method in methods
    )
    if methods != canonical:
        raise ValueError("methods must follow the frozen canonical method order")
    return methods


def _owned_budgets(value: object) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise ValueError("budgets must be an immutable integer sequence")
    budgets = tuple(value)
    if (
        not budgets
        or any(not _is_exact_int(item) or item <= 0 for item in budgets)
        or any(
            current >= following
            for current, following in zip(budgets, budgets[1:])
        )
    ):
        raise ValueError(
            "budgets must be positive, unique, and strictly increasing"
        )
    return budgets


@dataclass(frozen=True)
class EstimatorGroupRequest:
    """Complete frozen authority for one dataset/seed estimator group."""

    protocol_hash: str
    dataset_id: str
    seed: int
    task: str
    utility: str
    ordered_features: tuple[str, ...]
    split_hashes: Mapping[str, str]
    model_params: Mapping[str, object]
    backend: str
    methods: tuple[str, ...]
    budgets: tuple[int, ...]
    exact_max_features: int
    alpha: float
    cache_schema: str
    checkpoint_every_fits: int
    progress_every_seconds: int

    def __post_init__(self) -> None:
        _validate_sha256(self.protocol_hash, "protocol_hash")
        dataset_id = _safe_dataset_id(self.dataset_id)
        if not _is_exact_int(self.seed) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.task not in _TASK_UTILITIES:
            raise ValueError("task is invalid")
        if self.utility not in _TASK_UTILITIES[self.task]:
            raise ValueError("utility is invalid for task")

        features = _owned_string_tuple(
            self.ordered_features, "ordered features"
        )
        if (
            not isinstance(self.split_hashes, Mapping)
            or set(self.split_hashes) != {"train", "validation"}
        ):
            raise ValueError(
                "split_hashes must contain exactly train and validation"
            )
        split_hashes = MappingProxyType(
            {
                key: _validate_sha256(
                    self.split_hashes[key], f"split_hashes.{key}"
                )
                for key in ("train", "validation")
            }
        )
        if not isinstance(self.model_params, Mapping):
            raise TypeError("model_params must be a mapping")
        model_params = _freeze_json_value(self.model_params, "model parameters")
        if not isinstance(model_params, Mapping):
            raise TypeError("model_params must be a mapping")
        if self.backend not in _BACKENDS:
            raise ValueError("backend is invalid")

        methods = _canonical_methods(self.methods)
        budgets = _owned_budgets(self.budgets)
        exact_max_features = _positive_int(
            self.exact_max_features, "exact_max_features"
        )
        if (
            "exact" in methods
            and len(features) > exact_max_features
        ):
            raise ValueError(
                "exact method is not authorized for the prepared feature count"
            )
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(float(self.alpha))
            or not 0.0 < float(self.alpha) <= 1.0
        ):
            raise ValueError("alpha must be finite and in (0, 1]")
        if self.cache_schema != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"cache_schema must be {CACHE_SCHEMA_VERSION!r}"
            )
        _positive_int(
            self.checkpoint_every_fits, "checkpoint_every_fits"
        )
        _positive_int(
            self.progress_every_seconds, "progress_every_seconds"
        )

        object.__setattr__(self, "ordered_features", features)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "split_hashes", split_hashes)
        object.__setattr__(self, "model_params", model_params)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 identity of every request field."""
        encoded = json.dumps(
            group_request_to_document(self),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EstimatorCheckpoint:
    """One method/budget checkpoint inside a completed group bundle."""

    method: str
    budget: int
    scores: tuple[tuple[str, int], ...]
    logical_evaluation_count: int
    logical_unique_coalition_count: int
    physical_model_fit_count: int
    cache_hit_count: int
    schedule_digest: str | None
    component_runtime_s: float

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError("checkpoint method is invalid")
        _positive_int(self.budget, "checkpoint budget")
        if (
            isinstance(self.scores, (str, bytes))
            or not isinstance(self.scores, Sequence)
        ):
            raise ValueError("checkpoint scores must be a sequence")
        scores: list[tuple[str, int]] = []
        for item in self.scores:
            if (
                isinstance(item, (str, bytes))
                or not isinstance(item, Sequence)
                or len(item) != 2
            ):
                raise ValueError("checkpoint score entries are invalid")
            feature, score = item
            if not isinstance(feature, str) or not feature:
                raise ValueError("checkpoint score features are invalid")
            feature = _utf8_text(
                feature, "checkpoint score feature"
            )
            if not _is_exact_int(score) or score not in {-1, 0, 1}:
                raise ValueError(
                    "checkpoint score values must be integers in {-1, 0, 1}"
                )
            scores.append((feature, score))
        if not scores or len({item[0] for item in scores}) != len(scores):
            raise ValueError(
                "checkpoint score features must be non-empty and unique"
            )

        logical_evaluations = _nonnegative_int(
            self.logical_evaluation_count, "checkpoint"
        )
        logical_unique = _nonnegative_int(
            self.logical_unique_coalition_count, "checkpoint"
        )
        physical_fits = _nonnegative_int(
            self.physical_model_fit_count, "checkpoint"
        )
        cache_hits = _nonnegative_int(
            self.cache_hit_count, "checkpoint"
        )
        if logical_evaluations == 0 or logical_unique == 0:
            raise ValueError("checkpoint logical counters must be positive")
        if logical_unique > logical_evaluations:
            raise ValueError(
                "checkpoint logical unique counter exceeds "
                "logical evaluation counter"
            )
        if physical_fits > logical_unique:
            raise ValueError(
                "checkpoint physical fit counter exceeds logical coalitions"
            )
        if self.schedule_digest is not None:
            _validate_sha256(
                self.schedule_digest, "checkpoint schedule digest"
            )
        if self.method == "exact" and self.schedule_digest is not None:
            raise ValueError("exact checkpoint schedule digest must be null")
        if (
            self.method in {"random_mc", "mi_correlation_guided"}
            and self.schedule_digest is None
        ):
            raise ValueError(
                "scheduled checkpoint schedule digest is required"
            )
        if self.method == "hybrid" and physical_fits != 0:
            raise ValueError(
                "hybrid checkpoint physical fit counter must be zero"
            )
        runtime = _nonnegative_finite_float(
            self.component_runtime_s, "checkpoint component"
        )

        object.__setattr__(self, "scores", tuple(scores))
        object.__setattr__(
            self, "logical_evaluation_count", logical_evaluations
        )
        object.__setattr__(
            self, "logical_unique_coalition_count", logical_unique
        )
        object.__setattr__(
            self, "physical_model_fit_count", physical_fits
        )
        object.__setattr__(self, "cache_hit_count", cache_hits)
        object.__setattr__(self, "component_runtime_s", runtime)


@dataclass(frozen=True)
class EstimatorBundle:
    """Canonical completed estimator-group artifact."""

    schema_version: int
    request_digest: str
    checkpoints: tuple[EstimatorCheckpoint, ...]
    physical_model_fit_count: int
    cache_hit_count: int
    group_runtime_s: float

    def __post_init__(self) -> None:
        if (
            not _is_exact_int(self.schema_version)
            or self.schema_version != ESTIMATOR_BUNDLE_SCHEMA_VERSION
        ):
            raise ValueError("estimator bundle schema version is invalid")
        _validate_sha256(self.request_digest, "bundle request digest")
        if (
            isinstance(self.checkpoints, (str, bytes))
            or not isinstance(self.checkpoints, Sequence)
        ):
            raise ValueError("bundle checkpoints must be a sequence")
        checkpoints = tuple(self.checkpoints)
        if not checkpoints or not all(
            isinstance(item, EstimatorCheckpoint) for item in checkpoints
        ):
            raise ValueError(
                "bundle checkpoints must be non-empty EstimatorCheckpoint values"
            )
        physical_fits = _nonnegative_int(
            self.physical_model_fit_count, "bundle"
        )
        cache_hits = _nonnegative_int(self.cache_hit_count, "bundle")
        runtime = _nonnegative_finite_float(
            self.group_runtime_s, "bundle group"
        )

        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(
            self, "physical_model_fit_count", physical_fits
        )
        object.__setattr__(self, "cache_hit_count", cache_hits)
        object.__setattr__(self, "group_runtime_s", runtime)


def group_request_to_document(
    request: EstimatorGroupRequest,
) -> dict[str, object]:
    """Convert a validated request to its canonical JSON-compatible payload."""
    if not isinstance(request, EstimatorGroupRequest):
        raise TypeError("request must be an EstimatorGroupRequest")
    return {
        "protocol_hash": request.protocol_hash,
        "dataset_id": request.dataset_id,
        "seed": request.seed,
        "task": request.task,
        "utility": request.utility,
        "ordered_features": list(request.ordered_features),
        "split_hashes": dict(request.split_hashes),
        "model_params": _thaw_json_value(request.model_params),
        "backend": request.backend,
        "methods": list(request.methods),
        "budgets": list(request.budgets),
        "exact_max_features": request.exact_max_features,
        "alpha": request.alpha,
        "cache_schema": request.cache_schema,
        "checkpoint_every_fits": request.checkpoint_every_fits,
        "progress_every_seconds": request.progress_every_seconds,
    }


def _validated_feature_order(
    feature_order: object,
) -> tuple[str, ...]:
    return _owned_string_tuple(feature_order, "prepared features")


def _manifest_features(row: Mapping[str, object]) -> tuple[str, ...]:
    value = row.get("selected_features_json")
    if not isinstance(value, str):
        raise ValueError(
            "prepared manifest selected feature order is missing"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "prepared manifest selected feature order is invalid"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError("prepared manifest selected feature order is invalid")
    features = _validated_feature_order(parsed)
    count = row.get("selected_feature_count")
    if not _is_exact_int(count) or count != len(features):
        raise ValueError(
            "prepared manifest selected feature count is invalid"
        )
    return features


def group_request_from_context(
    context: StageContext,
    prepared_manifest_row: Mapping[str, object],
    feature_order: Sequence[str],
) -> EstimatorGroupRequest:
    """Build one group request from frozen stage and prepared-data authority."""
    try:
        protocol = context.protocol
        job = context.job
        layout = context.layout
    except AttributeError as exc:
        raise TypeError("context must be a StageContext") from exc
    if job.stage != "estimator":
        raise ValueError("group request requires an estimator job")
    if job.dataset_id not in {
        dataset.dataset_id for dataset in protocol.datasets
    }:
        raise ValueError("estimator job dataset is outside protocol authority")
    dataset = next(
        item
        for item in protocol.datasets
        if item.dataset_id == job.dataset_id
    )
    if job.seed not in protocol.seeds:
        raise ValueError("estimator job seed is outside protocol authority")
    protocol_hash = resolved_protocol_hash(protocol)
    if job.protocol_hash != protocol_hash:
        raise ValueError("estimator job protocol hash conflicts with protocol")
    if (
        job.method not in protocol.estimator_methods
        or set(job.parameters) != {"task", "utility", "budget"}
        or job.parameters["task"] != dataset.task
        or job.parameters["utility"] != dataset.utility
        or job.parameters["budget"] not in protocol.estimator_budgets
    ):
        raise ValueError("estimator job conflicts with protocol authority")
    if (
        layout.cache_root != layout.root / "cache"
        or layout.bundle_root
        != layout.root / "estimator" / "_bundles"
    ):
        raise ValueError("run layout cache or bundle root is not canonical")
    if (
        protocol.execution.mode != _GROUPED_EXECUTION_MODE
        or protocol.execution.cache_schema != CACHE_SCHEMA_VERSION
    ):
        raise ValueError("execution mode or cache schema is unsupported")
    if not isinstance(prepared_manifest_row, Mapping):
        raise TypeError("prepared manifest row must be a mapping")
    if prepared_manifest_row.get("dataset_id") != job.dataset_id:
        raise ValueError(
            "prepared manifest dataset conflicts with estimator authority"
        )
    if prepared_manifest_row.get("seed") != job.seed:
        raise ValueError(
            "prepared manifest seed conflicts with estimator authority"
        )
    if not {
        "train_file_sha256",
        "val_file_sha256",
    } <= set(prepared_manifest_row):
        raise ValueError("prepared manifest split hash is missing")
    validated_row = _validate_data_manifest_row(
        dict(prepared_manifest_row), protocol
    )
    if (
        validated_row["dataset_id"] != job.dataset_id
        or validated_row["seed"] != job.seed
    ):
        raise ValueError(
            "prepared manifest dataset or seed conflicts with estimator authority"
        )
    features = _validated_feature_order(feature_order)
    if _manifest_features(validated_row) != features:
        raise ValueError(
            "prepared manifest selected feature order conflicts with data"
        )

    try:
        manifest = tuple(context.status.manifest)
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "frozen status manifest authority is invalid"
        ) from exc
    if job not in manifest:
        raise ValueError("estimator job is outside the frozen status manifest")
    protocol_methods = _canonical_methods(protocol.estimator_methods)
    protocol_budgets = _owned_budgets(protocol.estimator_budgets)
    exact_authorized = (
        len(features) <= protocol.exact_max_features
    )
    expected_methods = tuple(
        method
        for method in protocol_methods
        if method != "exact" or exact_authorized
    )
    expected_pairs = tuple(
        (method, budget)
        for method in expected_methods
        for budget in protocol_budgets
    )
    try:
        group_jobs = tuple(
            candidate
            for candidate in manifest
            if candidate.stage == "estimator"
            and candidate.dataset_id == job.dataset_id
            and candidate.seed == job.seed
        )
    except AttributeError as exc:
        raise ValueError(
            "frozen status manifest authority is invalid"
        ) from exc
    actual_pairs: list[tuple[str, int]] = []
    for candidate in group_jobs:
        if (
            candidate.protocol_hash != protocol_hash
            or set(candidate.parameters) != {"task", "utility", "budget"}
            or candidate.parameters["task"] != dataset.task
            or candidate.parameters["utility"] != dataset.utility
            or candidate.method not in expected_methods
            or not _is_exact_int(candidate.parameters["budget"])
        ):
            raise ValueError(
                "estimator group manifest conflicts with protocol authority"
            )
        actual_pairs.append(
            (candidate.method, candidate.parameters["budget"])
        )
    if tuple(actual_pairs) != expected_pairs:
        raise ValueError(
            "estimator group manifest pairs conflict with protocol authority"
        )

    methods = tuple(dict.fromkeys(method for method, _ in actual_pairs))
    budgets = tuple(
        budget
        for method, budget in actual_pairs
        if method == methods[0]
    )
    split_hashes = {
        "train": _validate_sha256(
            validated_row["train_file_sha256"],
            "prepared manifest train hash",
        ),
        "validation": _validate_sha256(
            validated_row["val_file_sha256"],
            "prepared manifest validation hash",
        ),
    }
    return EstimatorGroupRequest(
        protocol_hash=protocol_hash,
        dataset_id=job.dataset_id,
        seed=job.seed,
        task=dataset.task,
        utility=dataset.utility,
        ordered_features=features,
        split_hashes=split_hashes,
        model_params=protocol.model_params,
        backend=protocol.backend,
        methods=methods,
        budgets=budgets,
        exact_max_features=protocol.exact_max_features,
        alpha=protocol.alpha_default,
        cache_schema=protocol.execution.cache_schema,
        checkpoint_every_fits=protocol.execution.checkpoint_every_fits,
        progress_every_seconds=protocol.execution.progress_every_seconds,
    )


@dataclass(frozen=True)
class _UtilityCallTrace:
    """One logical utility lookup and its observed shared-cache contribution."""

    coalition: tuple[str, ...]
    feature: str | None
    trial_ordinal: int | None
    physical_model_fit_count: int
    cache_hit_count: int
    elapsed_s: float


@dataclass(frozen=True)
class _ScheduleAuthority:
    random_schedule: NestedRandomSchedule | None
    guided_schedule: NestedGuidedSchedule | None
    random_digest: str | None
    guided_digest: str | None
    master_digest: str
    trial_upper_bound: int


def _schedule_authority(
    request: EstimatorGroupRequest,
) -> _ScheduleAuthority:
    feature_count = len(request.ordered_features)
    hybrid_requested = "hybrid" in request.methods
    exact_needed = feature_count <= 20 and (
        "exact" in request.methods or hybrid_requested
    )
    random_needed = (
        "random_mc" in request.methods
        or (hybrid_requested and feature_count > 75)
    )
    guided_needed = (
        "mi_correlation_guided" in request.methods
        or (hybrid_requested and feature_count > 20)
    )
    random_schedule = (
        make_nested_random_schedule(
            request.ordered_features,
            budgets=request.budgets,
            seed=request.seed,
        )
        if random_needed
        else None
    )
    guided_schedule = (
        make_nested_guided_schedule(
            request.ordered_features,
            budgets=request.budgets,
            seed=request.seed,
        )
        if guided_needed
        else None
    )
    random_digest = (
        nested_random_schedule_digest(random_schedule)
        if random_schedule is not None
        else None
    )
    guided_digest = (
        nested_guided_schedule_digest(guided_schedule)
        if guided_schedule is not None
        else None
    )
    exact_trial_upper_bound = (
        feature_count * (2 ** (feature_count - 1))
        if exact_needed
        else 0
    )
    nested_trial_upper_bound = (
        feature_count
        * request.budgets[-1]
        * (int(random_needed) + int(guided_needed))
    )
    trial_upper_bound = (
        exact_trial_upper_bound + nested_trial_upper_bound
    )
    payload = {
        "schema_version": 1,
        "request_digest": request.digest,
        "feature_count": feature_count,
        "max_budget": request.budgets[-1],
        "exact_trial_upper_bound": exact_trial_upper_bound,
        "random_schedule_digest": random_digest,
        "guided_schedule_digest": guided_digest,
        "trial_upper_bound": trial_upper_bound,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _ScheduleAuthority(
        random_schedule=random_schedule,
        guided_schedule=guided_schedule,
        random_digest=random_digest,
        guided_digest=guided_digest,
        master_digest=hashlib.sha256(encoded).hexdigest(),
        trial_upper_bound=trial_upper_bound,
    )


@dataclass(frozen=True)
class _TraceAccounting:
    logical_evaluation_count: int
    logical_coalitions: frozenset[tuple[str, ...]]
    physical_model_fit_count: int
    cache_hit_count: int
    component_runtime_s: float


class _TracingCoalitionUtilityView(CoalitionUtilityView):
    """Method-local view that records calls without changing evaluator order."""

    def __init__(
        self,
        cache: CoalitionUtilityCache,
        *,
        utility: str,
        alpha: float,
        algorithm: str,
        fit_hook: Callable[[int], None] | None = None,
        cache_observer: (
            Callable[
                [str, int, int, CoalitionCacheStats, int], None
            ]
            | None
        ) = None,
        trial_observer: (
            Callable[[str, int, int, int, CoalitionCacheStats], None]
            | None
        ) = None,
    ) -> None:
        self._traced_cache = cache
        self._call_trace: list[_UtilityCallTrace] = []
        self._active_feature: str | None = None
        self._active_trial_ordinal: int | None = None
        self._feature_trial_counts: dict[str, int] = {}
        self._algorithm = algorithm
        self._fit_hook = fit_hook
        self._cache_observer = cache_observer
        self._trial_observer = trial_observer
        super().__init__(cache, utility=utility, alpha=alpha)

    @property
    def call_trace(self) -> tuple[_UtilityCallTrace, ...]:
        return tuple(self._call_trace)

    def evaluate(self, coalition: AbstractSet[str]) -> float:
        key = canonical_coalition(self.feature_order, coalition)
        before = self._traced_cache.stats
        started = time.perf_counter()
        result = super().evaluate(coalition)
        elapsed = time.perf_counter() - started
        after = self._traced_cache.stats
        fit_delta = (
            after.physical_model_fit_count
            - before.physical_model_fit_count
        )
        hit_delta = after.cache_hit_count - before.cache_hit_count
        if fit_delta < 0 or hit_delta < 0:
            raise RuntimeError("shared cache counters moved backwards")
        self._call_trace.append(
            _UtilityCallTrace(
                coalition=key,
                feature=self._active_feature,
                trial_ordinal=self._active_trial_ordinal,
                physical_model_fit_count=fit_delta,
                cache_hit_count=hit_delta,
                elapsed_s=elapsed,
            )
        )
        if self._cache_observer is not None:
            feature_index = (
                0
                if self._active_feature is None
                else self.feature_order.index(self._active_feature) + 1
            )
            self._cache_observer(
                self._algorithm,
                feature_index,
                self._active_trial_ordinal or 0,
                after,
                fit_delta,
            )
        if fit_delta and self._fit_hook is not None:
            self._fit_hook(after.physical_model_fit_count)
        return result

    def eval_marginal(self, coalition: set[str], feature: str) -> int:
        ordinal = self._feature_trial_counts.get(feature, 0) + 1
        self._feature_trial_counts[feature] = ordinal
        previous_feature = self._active_feature
        previous_ordinal = self._active_trial_ordinal
        self._active_feature = feature
        self._active_trial_ordinal = ordinal
        try:
            result = super().eval_marginal(coalition, feature)
            if self._trial_observer is not None:
                feature_index = self.feature_order.index(feature) + 1
                self._trial_observer(
                    self._algorithm,
                    feature_index,
                    ordinal,
                    len(self.feature_order),
                    self._traced_cache.stats,
                )
            return result
        finally:
            self._active_feature = previous_feature
            self._active_trial_ordinal = previous_ordinal


def _validate_builder_authority(
    request: EstimatorGroupRequest,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    store: SQLiteCoalitionStore,
    progress_sink: Callable[[NestedProgress], None],
) -> CoalitionCacheIdentity:
    if not isinstance(request, EstimatorGroupRequest):
        raise TypeError("request must be an EstimatorGroupRequest")
    if not callable(progress_sink):
        raise TypeError("progress_sink must be callable")
    try:
        identity = store.identity
    except AttributeError as exc:
        raise TypeError("store must expose a coalition cache identity") from exc
    authority_fields = {
        "schema_version": request.cache_schema,
        "protocol_hash": request.protocol_hash,
        "dataset_id": request.dataset_id,
        "seed": request.seed,
        "namespace": "selection_validation",
        "ordered_features": request.ordered_features,
        "task": request.task,
        "backend": request.backend,
        "model_seed": request.seed,
    }
    if any(
        getattr(identity, field, None) != expected
        for field, expected in authority_fields.items()
    ):
        raise ValueError("store identity conflicts with estimator request")
    try:
        identity_model_params = dict(identity.model_params)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("store model identity is invalid") from exc
    if identity_model_params != dict(request.model_params):
        raise ValueError("store model identity conflicts with estimator request")
    data_hashes = {
        "train_features_sha256": hash_numeric_frame(X_train),
        "train_target_sha256": hash_numeric_array(np.asarray(y_train)),
        "eval_features_sha256": hash_numeric_frame(X_val),
        "eval_target_sha256": hash_numeric_array(np.asarray(y_val)),
    }
    if any(
        getattr(identity, field, None) != expected
        for field, expected in data_hashes.items()
    ):
        raise ValueError("store data identity conflicts with estimator frames")
    return identity


def _trace_accounting(
    view: _TracingCoalitionUtilityView,
    *,
    component_runtime_s: float,
    max_trial_ordinal: int | None,
) -> _TraceAccounting:
    traces = view.call_trace
    selected = tuple(
        item
        for item in traces
        if (
            item.feature is None
            or max_trial_ordinal is None
            or (
                item.trial_ordinal is not None
                and item.trial_ordinal <= max_trial_ordinal
            )
        )
    )
    if not selected:
        raise RuntimeError("component trace must contain threshold work")
    if len(selected) == len(traces):
        checkpoint_runtime = component_runtime_s
    else:
        traced_elapsed = math.fsum(item.elapsed_s for item in traces)
        selected_elapsed = math.fsum(item.elapsed_s for item in selected)
        ratio = (
            selected_elapsed / traced_elapsed
            if traced_elapsed > 0.0
            else len(selected) / len(traces)
        )
        checkpoint_runtime = min(
            component_runtime_s,
            component_runtime_s * ratio,
        )
    return _TraceAccounting(
        logical_evaluation_count=len(selected),
        logical_coalitions=frozenset(
            item.coalition for item in selected
        ),
        physical_model_fit_count=sum(
            item.physical_model_fit_count for item in selected
        ),
        cache_hit_count=sum(item.cache_hit_count for item in selected),
        component_runtime_s=checkpoint_runtime,
    )


def _hybrid_schedule_digest(
    *,
    feature_count: int,
    random_digest: str | None,
    guided_digest: str | None,
) -> str | None:
    if feature_count <= 20:
        return None
    if feature_count <= 75:
        return guided_digest
    if random_digest is None or guided_digest is None:
        raise RuntimeError("hybrid schedule components are missing")
    encoded = json.dumps(
        {
            "random_mc": random_digest,
            "mi_correlation_guided": guided_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _hybrid_accounting(
    *,
    feature_count: int,
    budget: int,
    exact_view: _TracingCoalitionUtilityView | None,
    exact_runtime_s: float,
    random_view: _TracingCoalitionUtilityView | None,
    random_runtime_s: float,
    guided_view: _TracingCoalitionUtilityView | None,
    guided_runtime_s: float,
    hybrid_runtime_s: float,
) -> _TraceAccounting:
    if feature_count <= 20:
        if exact_view is None:
            raise RuntimeError("hybrid exact trace is missing")
        source = (
            _trace_accounting(
                exact_view,
                component_runtime_s=exact_runtime_s,
                max_trial_ordinal=None,
            ),
        )
    elif feature_count <= 75:
        if guided_view is None:
            raise RuntimeError("hybrid guided trace is missing")
        source = (
            _trace_accounting(
                guided_view,
                component_runtime_s=guided_runtime_s,
                max_trial_ordinal=budget,
            ),
        )
    else:
        if random_view is None or guided_view is None:
            raise RuntimeError("hybrid scheduled traces are missing")
        source = (
            _trace_accounting(
                random_view,
                component_runtime_s=random_runtime_s,
                max_trial_ordinal=budget,
            ),
            _trace_accounting(
                guided_view,
                component_runtime_s=guided_runtime_s,
                max_trial_ordinal=budget,
            ),
        )
    return _TraceAccounting(
        logical_evaluation_count=sum(
            item.logical_evaluation_count for item in source
        ),
        logical_coalitions=frozenset().union(
            *(item.logical_coalitions for item in source)
        ),
        physical_model_fit_count=0,
        cache_hit_count=0,
        component_runtime_s=hybrid_runtime_s,
    )


def build_estimator_bundle(
    request: EstimatorGroupRequest,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    store: SQLiteCoalitionStore,
    progress_sink: Callable[[NestedProgress], None],
    fit_hook: Callable[[int], None] | None = None,
    cache_observer: (
        Callable[
            [str, int, int, CoalitionCacheStats, int], None
        ]
        | None
    ) = None,
    trial_observer: (
        Callable[[str, int, int, int, CoalitionCacheStats], None] | None
    ) = None,
    _schedules: _ScheduleAuthority | None = None,
) -> EstimatorBundle:
    """Build every manifest-authorized estimator checkpoint exactly once."""
    group_started = time.perf_counter()
    identity = _validate_builder_authority(
        request,
        X_train,
        y_train,
        X_val,
        y_val,
        store,
        progress_sink,
    )
    if fit_hook is not None and not callable(fit_hook):
        raise TypeError("fit_hook must be callable or None")
    if cache_observer is not None and not callable(cache_observer):
        raise TypeError("cache_observer must be callable or None")
    if trial_observer is not None and not callable(trial_observer):
        raise TypeError("trial_observer must be callable or None")
    feature_count = len(request.ordered_features)
    if "exact" in request.methods and feature_count > 20:
        raise ValueError("public exact estimation requires at most 20 features")

    cache = CoalitionUtilityCache(
        identity,
        X_train,
        y_train,
        X_val,
        y_val,
        store,
    )
    group_stats_before = cache.stats
    hybrid_requested = "hybrid" in request.methods
    exact_needed = feature_count <= 20 and (
        "exact" in request.methods or hybrid_requested
    )
    random_needed = (
        "random_mc" in request.methods
        or (hybrid_requested and feature_count > 75)
    )
    guided_needed = (
        "mi_correlation_guided" in request.methods
        or (hybrid_requested and feature_count > 20)
    )

    schedules = (
        _schedule_authority(request)
        if _schedules is None
        else _schedules
    )
    random_schedule = schedules.random_schedule
    guided_schedule = schedules.guided_schedule
    random_digest = schedules.random_digest
    guided_digest = schedules.guided_digest
    if random_needed != (random_schedule is not None):
        raise ValueError("random schedule authority conflicts with request")
    if guided_needed != (guided_schedule is not None):
        raise ValueError("guided schedule authority conflicts with request")
    if guided_needed:
        mi_scores, correlation_neighbors = build_mi_correlation_guidance(
            X_train,
            np.asarray(y_train),
            task=request.task,
            seed=request.seed,
        )
    else:
        mi_scores = None
        correlation_neighbors = None

    exact_scores: dict[str, int] | None = None
    exact_view: _TracingCoalitionUtilityView | None = None
    exact_runtime_s = 0.0
    if exact_needed:
        component_started = time.perf_counter()
        exact_view = _TracingCoalitionUtilityView(
            cache,
            utility=request.utility,
            alpha=request.alpha,
            algorithm="exact",
            fit_hook=fit_hook,
            cache_observer=cache_observer,
            trial_observer=trial_observer,
        )
        exact_scores = estimate_minppi_exact(
            X_train,
            X_val,
            np.asarray(y_train),
            np.asarray(y_val),
            alpha=request.alpha,
            task=request.task,
            utility=request.utility,
            acc_cache=exact_view,
            seed=request.seed,
        )
        exact_runtime_s = time.perf_counter() - component_started

    random_scores: dict[int, dict[str, int]] | None = None
    random_view: _TracingCoalitionUtilityView | None = None
    random_runtime_s = 0.0
    if random_needed:
        if random_schedule is None:
            raise RuntimeError("random schedule is missing")
        component_started = time.perf_counter()
        random_view = _TracingCoalitionUtilityView(
            cache,
            utility=request.utility,
            alpha=request.alpha,
            algorithm="random",
            fit_hook=fit_hook,
            cache_observer=cache_observer,
            trial_observer=trial_observer,
        )
        random_scores = estimate_random_checkpoints(
            random_schedule,
            random_view,
            progress=progress_sink,
        )
        random_runtime_s = time.perf_counter() - component_started

    guided_scores: dict[int, dict[str, int]] | None = None
    guided_view: _TracingCoalitionUtilityView | None = None
    guided_runtime_s = 0.0
    if guided_needed:
        if guided_schedule is None:
            raise RuntimeError("guided schedule is missing")
        component_started = time.perf_counter()
        guided_view = _TracingCoalitionUtilityView(
            cache,
            utility=request.utility,
            alpha=request.alpha,
            algorithm="guided",
            fit_hook=fit_hook,
            cache_observer=cache_observer,
            trial_observer=trial_observer,
        )
        guided_scores = estimate_guided_checkpoints(
            guided_schedule,
            guided_view,
            mi_scores=mi_scores,
            correlation_neighbors=correlation_neighbors,
            progress=progress_sink,
        )
        guided_runtime_s = time.perf_counter() - component_started

    hybrid_scores: dict[int, dict[str, int]] | None = None
    hybrid_runtime_s = 0.0
    if hybrid_requested:
        component_started = time.perf_counter()
        hybrid_scores = derive_hybrid_checkpoints(
            feature_count=feature_count,
            checkpoints=request.budgets,
            exact_scores=exact_scores,
            random_scores=random_scores,
            guided_scores=guided_scores,
        )
        hybrid_runtime_s = time.perf_counter() - component_started

    score_maps: dict[str, Mapping[int, Mapping[str, int]]] = {}
    if "exact" in request.methods:
        if exact_scores is None:
            raise RuntimeError("public exact scores are missing")
        score_maps["exact"] = materialize_exact_checkpoints(
            request.budgets, exact_scores
        )
    if "random_mc" in request.methods:
        if random_scores is None:
            raise RuntimeError("public random scores are missing")
        score_maps["random_mc"] = random_scores
    if "mi_correlation_guided" in request.methods:
        if guided_scores is None:
            raise RuntimeError("public guided scores are missing")
        score_maps["mi_correlation_guided"] = guided_scores
    if hybrid_requested:
        if hybrid_scores is None:
            raise RuntimeError("public hybrid scores are missing")
        score_maps["hybrid"] = hybrid_scores

    checkpoints: list[EstimatorCheckpoint] = []
    for method in request.methods:
        for budget in request.budgets:
            if method == "exact":
                if exact_view is None:
                    raise RuntimeError("exact accounting trace is missing")
                accounting = _trace_accounting(
                    exact_view,
                    component_runtime_s=exact_runtime_s,
                    max_trial_ordinal=None,
                )
                schedule_digest = None
            elif method == "random_mc":
                if random_view is None:
                    raise RuntimeError("random accounting trace is missing")
                accounting = _trace_accounting(
                    random_view,
                    component_runtime_s=random_runtime_s,
                    max_trial_ordinal=budget,
                )
                schedule_digest = random_digest
            elif method == "mi_correlation_guided":
                if guided_view is None:
                    raise RuntimeError("guided accounting trace is missing")
                accounting = _trace_accounting(
                    guided_view,
                    component_runtime_s=guided_runtime_s,
                    max_trial_ordinal=budget,
                )
                schedule_digest = guided_digest
            else:
                accounting = _hybrid_accounting(
                    feature_count=feature_count,
                    budget=budget,
                    exact_view=exact_view,
                    exact_runtime_s=exact_runtime_s,
                    random_view=random_view,
                    random_runtime_s=random_runtime_s,
                    guided_view=guided_view,
                    guided_runtime_s=guided_runtime_s,
                    hybrid_runtime_s=hybrid_runtime_s,
                )
                schedule_digest = _hybrid_schedule_digest(
                    feature_count=feature_count,
                    random_digest=random_digest,
                    guided_digest=guided_digest,
                )
            scores = score_maps[method][budget]
            checkpoints.append(
                EstimatorCheckpoint(
                    method=method,
                    budget=budget,
                    scores=tuple(
                        (feature, scores[feature])
                        for feature in request.ordered_features
                    ),
                    logical_evaluation_count=(
                        accounting.logical_evaluation_count
                    ),
                    logical_unique_coalition_count=len(
                        accounting.logical_coalitions
                    ),
                    physical_model_fit_count=(
                        accounting.physical_model_fit_count
                    ),
                    cache_hit_count=accounting.cache_hit_count,
                    schedule_digest=schedule_digest,
                    component_runtime_s=accounting.component_runtime_s,
                )
            )

    group_stats_after = cache.stats
    bundle = EstimatorBundle(
        schema_version=ESTIMATOR_BUNDLE_SCHEMA_VERSION,
        request_digest=request.digest,
        checkpoints=tuple(checkpoints),
        physical_model_fit_count=(
            group_stats_after.physical_model_fit_count
            - group_stats_before.physical_model_fit_count
        ),
        cache_hit_count=(
            group_stats_after.cache_hit_count
            - group_stats_before.cache_hit_count
        ),
        group_runtime_s=time.perf_counter() - group_started,
    )
    return validate_estimator_bundle(
        bundle, request, _schedules=schedules
    )


def bundle_relpath(request: EstimatorGroupRequest) -> str:
    """Return the canonical completed-bundle path relative to a run root."""
    if not isinstance(request, EstimatorGroupRequest):
        raise TypeError("request must be an EstimatorGroupRequest")
    return (
        f"estimator/_bundles/{request.dataset_id}/"
        f"s{request.seed}/bundle.json"
    )


def progress_relpath(request: EstimatorGroupRequest) -> str:
    """Return the canonical progress path relative to a run root."""
    if not isinstance(request, EstimatorGroupRequest):
        raise TypeError("request must be an EstimatorGroupRequest")
    return (
        f"estimator/_bundles/{request.dataset_id}/"
        f"s{request.seed}/progress.json"
    )


def cache_relpath(request: EstimatorGroupRequest) -> str:
    """Return the canonical selection/validation cache path."""
    if not isinstance(request, EstimatorGroupRequest):
        raise TypeError("request must be an EstimatorGroupRequest")
    return (
        f"cache/selection_validation/{request.dataset_id}/"
        f"s{request.seed}/coalitions.sqlite3"
    )


def _expected_pairs(
    request: EstimatorGroupRequest,
) -> tuple[tuple[str, int], ...]:
    exact_authorized = (
        len(request.ordered_features) <= request.exact_max_features
    )
    return tuple(
        (method, budget)
        for method in request.methods
        for budget in request.budgets
        if method != "exact" or exact_authorized
    )


def validate_estimator_bundle(
    bundle: EstimatorBundle,
    request: EstimatorGroupRequest,
    *,
    _schedules: _ScheduleAuthority | None = None,
) -> EstimatorBundle:
    """Validate a bundle against the exact request and manifest pair set."""
    if not isinstance(bundle, EstimatorBundle):
        raise TypeError("bundle must be an EstimatorBundle")
    if not isinstance(request, EstimatorGroupRequest):
        raise TypeError("request must be an EstimatorGroupRequest")
    if bundle.request_digest != request.digest:
        raise ValueError("bundle request digest conflicts with request")

    actual_pairs = tuple(
        (checkpoint.method, checkpoint.budget)
        for checkpoint in bundle.checkpoints
    )
    if actual_pairs != _expected_pairs(request):
        raise ValueError(
            "bundle checkpoint method/budget pairs or checkpoint order "
            "conflict with request authority"
        )

    expected_features = request.ordered_features
    schedules = (
        _schedule_authority(request)
        if _schedules is None
        else _schedules
    )
    expected_schedule_by_method = {
        "exact": None,
        "random_mc": schedules.random_digest,
        "mi_correlation_guided": schedules.guided_digest,
        "hybrid": _hybrid_schedule_digest(
            feature_count=len(expected_features),
            random_digest=schedules.random_digest,
            guided_digest=schedules.guided_digest,
        ),
    }
    schedule_by_method: dict[str, str | None] = {}
    previous_by_method: dict[str, EstimatorCheckpoint] = {}
    for checkpoint in bundle.checkpoints:
        actual_features = tuple(feature for feature, _ in checkpoint.scores)
        if set(actual_features) != set(expected_features):
            raise ValueError(
                "bundle checkpoint score feature conflicts with request"
            )
        if actual_features != expected_features:
            raise ValueError(
                "bundle checkpoint score feature order conflicts with request"
            )
        frozen_schedule = schedule_by_method.setdefault(
            checkpoint.method, checkpoint.schedule_digest
        )
        if checkpoint.schedule_digest != frozen_schedule:
            raise ValueError(
                "bundle checkpoint schedule digest conflicts within method"
            )
        if (
            checkpoint.schedule_digest
            != expected_schedule_by_method[checkpoint.method]
        ):
            raise ValueError(
                "bundle checkpoint schedule digest conflicts with "
                "deterministic schedule authority"
            )
        if (
            checkpoint.physical_model_fit_count
            > bundle.physical_model_fit_count
            or checkpoint.cache_hit_count > bundle.cache_hit_count
        ):
            raise ValueError(
                "bundle checkpoint counter exceeds group counter"
            )
        if checkpoint.component_runtime_s > bundle.group_runtime_s:
            raise ValueError(
                "bundle checkpoint runtime exceeds group runtime"
            )
        previous = previous_by_method.get(checkpoint.method)
        if previous is not None and any(
            current < earlier
            for current, earlier in (
                (
                    checkpoint.logical_evaluation_count,
                    previous.logical_evaluation_count,
                ),
                (
                    checkpoint.logical_unique_coalition_count,
                    previous.logical_unique_coalition_count,
                ),
                (
                    checkpoint.physical_model_fit_count,
                    previous.physical_model_fit_count,
                ),
                (
                    checkpoint.cache_hit_count,
                    previous.cache_hit_count,
                ),
                (
                    checkpoint.component_runtime_s,
                    previous.component_runtime_s,
                ),
            )
        ):
            raise ValueError(
                "bundle checkpoint counters or runtime are not monotonic"
            )
        previous_by_method[checkpoint.method] = checkpoint

    final_components = tuple(
        previous_by_method[method] for method in request.methods
    )
    expected_physical_fits = sum(
        checkpoint.physical_model_fit_count
        for checkpoint in final_components
    )
    if bundle.physical_model_fit_count < expected_physical_fits:
        raise ValueError(
            "bundle physical model fit total is below final components"
        )
    expected_cache_hits = sum(
        checkpoint.cache_hit_count
        for checkpoint in final_components
    )
    if bundle.cache_hit_count < expected_cache_hits:
        raise ValueError(
            "bundle cache hit total is below final components"
        )
    component_runtime = math.fsum(
        checkpoint.component_runtime_s
        for checkpoint in final_components
    )
    if bundle.group_runtime_s + 1e-12 < component_runtime:
        raise ValueError(
            "bundle group runtime is less than final component runtimes"
        )
    return bundle


def _checkpoint_to_document(
    checkpoint: EstimatorCheckpoint,
) -> dict[str, object]:
    return {
        "method": checkpoint.method,
        "budget": checkpoint.budget,
        "scores": [
            [feature, score] for feature, score in checkpoint.scores
        ],
        "logical_evaluation_count": checkpoint.logical_evaluation_count,
        "logical_unique_coalition_count": (
            checkpoint.logical_unique_coalition_count
        ),
        "physical_model_fit_count": checkpoint.physical_model_fit_count,
        "cache_hit_count": checkpoint.cache_hit_count,
        "schedule_digest": checkpoint.schedule_digest,
        "component_runtime_s": checkpoint.component_runtime_s,
    }


def bundle_to_document(
    bundle: EstimatorBundle, request: EstimatorGroupRequest
) -> dict[str, object]:
    """Convert a request-validated bundle to canonical JSON-compatible data."""
    validate_estimator_bundle(bundle, request)
    return {
        "schema_version": bundle.schema_version,
        "request_digest": bundle.request_digest,
        "checkpoints": [
            _checkpoint_to_document(checkpoint)
            for checkpoint in bundle.checkpoints
        ],
        "physical_model_fit_count": bundle.physical_model_fit_count,
        "cache_hit_count": bundle.cache_hit_count,
        "group_runtime_s": bundle.group_runtime_s,
    }


def _checkpoint_from_document(
    value: object, *, index: int
) -> EstimatorCheckpoint:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_FIELDS:
        raise ValueError(
            f"estimator bundle checkpoint[{index}] schema is invalid"
        )
    scores = value["scores"]
    if not isinstance(scores, list):
        raise ValueError(
            f"estimator bundle checkpoint[{index}] scores are invalid"
        )
    return EstimatorCheckpoint(
        method=value["method"],
        budget=value["budget"],
        scores=tuple(scores),
        logical_evaluation_count=value["logical_evaluation_count"],
        logical_unique_coalition_count=value[
            "logical_unique_coalition_count"
        ],
        physical_model_fit_count=value["physical_model_fit_count"],
        cache_hit_count=value["cache_hit_count"],
        schedule_digest=value["schedule_digest"],
        component_runtime_s=value["component_runtime_s"],
    )


def bundle_from_document(
    value: object, request: EstimatorGroupRequest
) -> EstimatorBundle:
    """Parse and strictly validate canonical bundle document content."""
    if not isinstance(value, dict) or set(value) != _BUNDLE_FIELDS:
        raise ValueError("estimator bundle schema is invalid")
    checkpoints = value["checkpoints"]
    if not isinstance(checkpoints, list):
        raise ValueError("estimator bundle checkpoints schema is invalid")
    bundle = EstimatorBundle(
        schema_version=value["schema_version"],
        request_digest=value["request_digest"],
        checkpoints=tuple(
            _checkpoint_from_document(item, index=index)
            for index, item in enumerate(checkpoints)
        ),
        physical_model_fit_count=value["physical_model_fit_count"],
        cache_hit_count=value["cache_hit_count"],
        group_runtime_s=value["group_runtime_s"],
    )
    return validate_estimator_bundle(bundle, request)


def _validated_frames(
    frames: object,
    request: EstimatorGroupRequest,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    names = ("X_train", "y_train", "X_val", "y_val")
    if isinstance(frames, Mapping):
        if set(frames) != set(names):
            raise ValueError("frames mapping schema is invalid")
        values = tuple(frames[name] for name in names)
    elif isinstance(frames, (tuple, list)) and len(frames) == 4:
        values = tuple(frames)
    elif all(hasattr(frames, name) for name in names):
        values = tuple(getattr(frames, name) for name in names)
    else:
        raise TypeError(
            "frames must provide X_train, y_train, X_val, and y_val"
        )
    X_train, y_train_value, X_val, y_val_value = values
    if not isinstance(X_train, pd.DataFrame) or not isinstance(
        X_val, pd.DataFrame
    ):
        raise TypeError("training and validation features must be DataFrames")
    expected_features = request.ordered_features
    for name, frame in (("X_train", X_train), ("X_val", X_val)):
        if (
            tuple(frame.columns) != expected_features
            or frame.columns.duplicated().any()
            or len(frame) == 0
        ):
            raise ValueError(
                f"{name} feature order conflicts with estimator request"
            )
        try:
            numeric = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f"{name} must contain finite values")
    y_train = np.asarray(y_train_value)
    y_val = np.asarray(y_val_value)
    for name, target, frame in (
        ("y_train", y_train, X_train),
        ("y_val", y_val, X_val),
    ):
        if (
            target.ndim != 1
            or len(target) != len(frame)
            or not np.issubdtype(target.dtype, np.number)
            or not np.isfinite(target.astype(float, copy=False)).all()
        ):
            raise ValueError(f"{name} is invalid for its feature frame")
    if request.task == "regression":
        if len(X_val) < 2:
            raise ValueError("regression validation requires two rows")
    else:
        if (
            not np.equal(y_train, np.floor(y_train)).all()
            or not np.equal(y_val, np.floor(y_val)).all()
        ):
            raise ValueError("classification targets must be integer labels")
        train_classes = tuple(
            int(value) for value in np.unique(y_train)
        )
        validation_classes = {
            int(value) for value in np.unique(y_val)
        }
        if request.task == "binary" and train_classes != (0, 1):
            raise ValueError("binary training classes must be exactly {0,1}")
        if request.task == "multiclass" and (
            len(train_classes) < 2
            or train_classes != tuple(range(len(train_classes)))
        ):
            raise ValueError(
                "multiclass training classes must be zero-based contiguous"
            )
        if not validation_classes.issubset(train_classes):
            raise ValueError("validation target contains an unseen class")
        y_train = y_train.astype(int, copy=False)
        y_val = y_val.astype(int, copy=False)
    return X_train, y_train, X_val, y_val


def _library_versions() -> dict[str, str]:
    try:
        xgboost_version = metadata.version("xgboost")
    except metadata.PackageNotFoundError:
        xgboost_version = "unavailable"
    return {
        "numpy": np.__version__,
        "xgboost": xgboost_version,
    }


def _cache_identity(
    request: EstimatorGroupRequest,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> CoalitionCacheIdentity:
    class_labels = (
        ()
        if request.task == "regression"
        else tuple(int(value) for value in np.unique(y_train))
    )
    return CoalitionCacheIdentity(
        schema_version=request.cache_schema,
        protocol_hash=request.protocol_hash,
        dataset_id=request.dataset_id,
        seed=request.seed,
        namespace="selection_validation",
        train_features_sha256=hash_numeric_frame(X_train),
        train_target_sha256=hash_numeric_array(y_train),
        eval_features_sha256=hash_numeric_frame(X_val),
        eval_target_sha256=hash_numeric_array(y_val),
        ordered_features=request.ordered_features,
        task=request.task,
        class_labels=class_labels,
        model_params=request.model_params,
        backend=request.backend,
        model_seed=request.seed,
        library_versions=_library_versions(),
    )


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError as exc:
        raise ValueError(f"cannot inspect path safety: {path}") from exc


def _reject_link_components(path: Path) -> None:
    absolute = path.absolute()
    for candidate in reversed((absolute, *absolute.parents)):
        if _is_link(candidate):
            raise ValueError(f"path contains a symbolic link: {candidate}")


def _canonical_run_path(
    run_root: Path,
    relpath: str,
    *,
    context: str,
) -> Path:
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be a Path")
    if (
        not relpath
        or "\\" in relpath
        or Path(relpath).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(relpath).parts)
    ):
        raise ValueError(f"{context} relative path is unsafe")
    _reject_link_components(run_root)
    root = run_root.resolve(strict=False)
    candidate = root.joinpath(*Path(relpath).parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context} path escapes run root") from exc
    _reject_link_components(candidate)
    return candidate


def _prepare_artifact_paths(
    request: EstimatorGroupRequest,
    run_root: Path,
) -> tuple[Path, Path, Path]:
    bundle_path = _canonical_run_path(
        run_root, bundle_relpath(request), context="bundle"
    )
    progress_path = _canonical_run_path(
        run_root, progress_relpath(request), context="progress"
    )
    cache_path = _canonical_run_path(
        run_root, cache_relpath(request), context="cache"
    )
    for path in (bundle_path, progress_path, cache_path):
        if path.exists() and not path.is_file():
            raise ValueError(f"artifact path is not a regular file: {path}")
        _reject_link_components(path)
    for suffix in ("-wal", "-shm"):
        sidecar = cache_path.with_name(cache_path.name + suffix)
        if _is_link(sidecar):
            raise ValueError("cache sidecar must not be a symbolic link")
    return bundle_path, progress_path, cache_path


def _read_progress(
    path: Path,
    request: EstimatorGroupRequest,
    schedules: _ScheduleAuthority,
) -> EstimatorProgressState | None:
    if not path.exists():
        return None
    state = progress_from_document(
        read_canonical_json(path, context="estimator progress")
    )
    event = state.event
    if (
        event.request_digest != request.digest
        or event.schedule_digest != schedules.master_digest
        or event.dataset_id != request.dataset_id
        or event.seed != request.seed
        or event.feature_count != len(request.ordered_features)
        or event.max_budget
        not in {
            request.budgets[-1],
            2 ** (len(request.ordered_features) - 1),
        }
        or state.last_completed_trial_index
        > schedules.trial_upper_bound
    ):
        raise ValueError(
            "estimator progress conflicts with request or schedule digest"
        )
    return state


def _progress_proves_zero_work(
    state: EstimatorProgressState,
) -> bool:
    return (
        state.event.physical_model_fit_count == 0
        and state.event.cache_hit_count == 0
        and state.last_completed_trial_index == 0
    )


def _complete_progress_event(
    request: EstimatorGroupRequest,
    schedules: _ScheduleAuthority,
    bundle: EstimatorBundle,
) -> EstimatorProgressEvent:
    return EstimatorProgressEvent(
        request_digest=request.digest,
        schedule_digest=schedules.master_digest,
        dataset_id=request.dataset_id,
        seed=request.seed,
        algorithm="complete",
        feature_index=len(request.ordered_features),
        feature_count=len(request.ordered_features),
        budget=request.budgets[-1],
        max_budget=request.budgets[-1],
        physical_model_fit_count=bundle.physical_model_fit_count,
        cache_hit_count=bundle.cache_hit_count,
        elapsed_s=bundle.group_runtime_s,
        eta_s=0.0,
    )


def _validate_completed_recovery_progress(
    state: EstimatorProgressState,
    request: EstimatorGroupRequest,
    schedules: _ScheduleAuthority,
    bundle: EstimatorBundle,
) -> EstimatorProgressEvent:
    event = state.event
    if (
        event.physical_model_fit_count
        != bundle.physical_model_fit_count
        or event.cache_hit_count != bundle.cache_hit_count
        or event.elapsed_s != bundle.group_runtime_s
    ):
        raise ValueError(
            "estimator progress conflicts with completed bundle totals"
        )
    canonical = _complete_progress_event(
        request, schedules, bundle
    )
    if event.algorithm == "complete" and event != canonical:
        raise ValueError(
            "completed estimator progress conflicts with bundle state"
        )
    return canonical


class _ProgressCoordinator:
    def __init__(
        self,
        *,
        request: EstimatorGroupRequest,
        schedules: _ScheduleAuthority,
        path: Path,
        sink: Callable[[EstimatorProgressEvent], None],
        previous: EstimatorProgressState | None,
    ) -> None:
        self._request = request
        self._schedules = schedules
        self._path = path
        self._sink = sink
        self._started = time.perf_counter()
        self._last_emit_at = self._started
        self._recent_fit_timestamps: deque[float] = deque(maxlen=200)
        self._current_trial_count = 0
        self._early_stopped_reduction = 0
        self._closed_features: set[tuple[str, int]] = set()
        self._trial_counts: dict[tuple[str, int], int] = {}
        self._current_physical_fits = 0
        self._current_cache_hits = 0
        self._base_physical_fits = (
            0
            if previous is None
            else previous.event.physical_model_fit_count
        )
        self._base_cache_hits = (
            0 if previous is None else previous.event.cache_hit_count
        )
        self._base_elapsed_s = (
            0.0 if previous is None else previous.event.elapsed_s
        )
        self._resume_trial_index = (
            0 if previous is None else previous.last_completed_trial_index
        )
        if previous is None:
            algorithm = (
                "exact"
                if len(request.ordered_features) <= 20
                and (
                    "exact" in request.methods
                    or "hybrid" in request.methods
                )
                else (
                    "random"
                    if schedules.random_schedule is not None
                    else "guided"
                )
            )
            self._algorithm = algorithm
            self._feature_index = 0
            self._budget = 0
            self._max_budget = self._algorithm_max_budget(algorithm)
        else:
            self._algorithm = previous.event.algorithm
            self._feature_index = previous.event.feature_index
            self._budget = previous.event.budget
            self._max_budget = previous.event.max_budget

    def _algorithm_max_budget(self, algorithm: str) -> int:
        if algorithm == "exact":
            return 2 ** (len(self._request.ordered_features) - 1)
        return self._request.budgets[-1]

    def observe_cache(
        self,
        algorithm: str,
        feature_index: int,
        trial_ordinal: int,
        stats: CoalitionCacheStats,
        fit_delta: int,
        *,
        committed: bool,
    ) -> None:
        self._algorithm = algorithm
        self._feature_index = feature_index
        self._budget = trial_ordinal
        self._max_budget = self._algorithm_max_budget(algorithm)
        self._current_physical_fits = stats.physical_model_fit_count
        self._current_cache_hits = stats.cache_hit_count
        now = time.perf_counter()
        for _ in range(fit_delta):
            self._recent_fit_timestamps.append(now)
        if fit_delta and committed:
            self.persist(emit_boundary=False)

    def observe_trial(
        self,
        algorithm: str,
        feature_index: int,
        trial_ordinal: int,
        feature_count: int,
        stats: CoalitionCacheStats,
    ) -> None:
        if feature_count != len(self._request.ordered_features):
            raise ValueError("trial progress feature count conflicts")
        self._algorithm = algorithm
        self._feature_index = feature_index
        self._budget = trial_ordinal
        self._max_budget = self._algorithm_max_budget(algorithm)
        self._current_physical_fits = stats.physical_model_fit_count
        self._current_cache_hits = stats.cache_hit_count
        self._current_trial_count += 1
        key = (algorithm, feature_index)
        self._trial_counts[key] = self._trial_counts.get(key, 0) + 1
        if algorithm == "exact" and trial_ordinal == self._max_budget:
            self.persist(emit_boundary=True)

    def on_nested_progress(self, progress: NestedProgress) -> None:
        if not isinstance(progress, NestedProgress):
            raise TypeError("nested progress is invalid")
        self._algorithm = progress.algorithm
        self._feature_index = progress.feature_index
        self._budget = progress.budget
        self._max_budget = progress.max_budget
        if progress.budget == progress.max_budget:
            key = (progress.algorithm, progress.feature_index)
            if key not in self._closed_features:
                completed = self._trial_counts.get(key, 0)
                self._early_stopped_reduction += max(
                    0, progress.max_budget - completed
                )
                self._closed_features.add(key)
        self.persist(emit_boundary=True)

    def _elapsed_s(self) -> float:
        return self._base_elapsed_s + (
            time.perf_counter() - self._started
        )

    def _last_completed_trial_index(self) -> int:
        return max(
            self._resume_trial_index, self._current_trial_count
        )

    def event(self, *, complete: bool = False) -> EstimatorProgressEvent:
        elapsed = self._elapsed_s()
        last_completed = self._last_completed_trial_index()
        remaining = max(
            0,
            self._schedules.trial_upper_bound
            - self._early_stopped_reduction
            - last_completed,
        )
        if complete:
            algorithm = "complete"
            feature_index = len(self._request.ordered_features)
            budget = self._request.budgets[-1]
            max_budget = budget
            eta = 0.0
        else:
            algorithm = self._algorithm
            feature_index = self._feature_index
            budget = self._budget
            max_budget = self._max_budget
            eta = estimate_eta_s(
                tuple(self._recent_fit_timestamps), remaining
            )
        return EstimatorProgressEvent(
            request_digest=self._request.digest,
            schedule_digest=self._schedules.master_digest,
            dataset_id=self._request.dataset_id,
            seed=self._request.seed,
            algorithm=algorithm,
            feature_index=feature_index,
            feature_count=len(self._request.ordered_features),
            budget=budget,
            max_budget=max_budget,
            physical_model_fit_count=(
                self._base_physical_fits
                + self._current_physical_fits
            ),
            cache_hit_count=(
                self._base_cache_hits + self._current_cache_hits
            ),
            elapsed_s=elapsed,
            eta_s=eta,
        )

    def cumulative_bundle(
        self, bundle: EstimatorBundle
    ) -> EstimatorBundle:
        """Return the bundle counters accumulated across the full lineage."""
        return replace(
            bundle,
            physical_model_fit_count=(
                self._base_physical_fits
                + bundle.physical_model_fit_count
            ),
            cache_hit_count=(
                self._base_cache_hits + bundle.cache_hit_count
            ),
            group_runtime_s=(
                self._base_elapsed_s + bundle.group_runtime_s
            ),
        )

    def persist_bundle(
        self,
        bundle: EstimatorBundle,
        *,
        complete: bool,
    ) -> EstimatorProgressEvent:
        """Persist a progress event with the bundle's canonical totals."""
        event = replace(
            self.event(complete=complete),
            physical_model_fit_count=bundle.physical_model_fit_count,
            cache_hit_count=bundle.cache_hit_count,
            elapsed_s=bundle.group_runtime_s,
        )
        atomic_write_json(
            self._path,
            progress_to_document(
                event,
                last_completed_trial_index=(
                    self._last_completed_trial_index()
                ),
            ),
        )
        self._sink(event)
        self._last_emit_at = time.perf_counter()
        return event

    def persist(
        self,
        *,
        emit_boundary: bool,
        complete: bool = False,
    ) -> EstimatorProgressEvent:
        event = self.event(complete=complete)
        atomic_write_json(
            self._path,
            progress_to_document(
                event,
                last_completed_trial_index=(
                    self._last_completed_trial_index()
                ),
            ),
        )
        now = time.perf_counter()
        if (
            emit_boundary
            or complete
            or now - self._last_emit_at
            >= self._request.progress_every_seconds
        ):
            self._sink(event)
            self._last_emit_at = now
        return event


def ensure_estimator_bundle(
    request: EstimatorGroupRequest,
    frames: object,
    run_root: Path,
    progress_sink: Callable[[EstimatorProgressEvent], None],
    *,
    fit_hook: Callable[[int], None] | None = None,
) -> EstimatorBundle:
    """Return or atomically build one request-bound resumable bundle."""
    if not isinstance(request, EstimatorGroupRequest):
        raise TypeError("request must be an EstimatorGroupRequest")
    if not callable(progress_sink):
        raise TypeError("progress_sink must be callable")
    if fit_hook is not None and not callable(fit_hook):
        raise TypeError("fit_hook must be callable or None")
    X_train, y_train, X_val, y_val = _validated_frames(frames, request)
    identity = _cache_identity(
        request, X_train, y_train, X_val, y_val
    )
    schedules = _schedule_authority(request)
    bundle_path, progress_path, cache_path = _prepare_artifact_paths(
        request, run_root
    )
    previous = _read_progress(progress_path, request, schedules)
    completed: EstimatorBundle | None = None
    if bundle_path.exists():
        completed = bundle_from_document(
            read_canonical_json(
                bundle_path, context="estimator bundle"
            ),
            request,
        )
    cache_preexisted = cache_path.exists()
    if completed is not None and not cache_preexisted:
        raise ValueError(
            "completed estimator bundle is missing its coalition cache"
        )
    if (
        previous is not None
        and not cache_preexisted
        and not _progress_proves_zero_work(previous)
    ):
        raise ValueError(
            "estimator progress exists without its coalition cache"
        )

    with SQLiteCoalitionStore(
        cache_path,
        identity,
        commit_every_fits=request.checkpoint_every_fits,
    ) as store:
        audit_cache = CoalitionUtilityCache(
            identity,
            X_train,
            y_train,
            X_val,
            y_val,
            store,
        )
        audit_cache.validate_persistent_records()
        if previous is None and cache_preexisted and store.entry_count != 0:
            raise ValueError(
                "estimator cache exists without validated progress authority"
            )
        if completed is not None:
            if previous is None or store.entry_count == 0:
                raise ValueError(
                    "completed estimator bundle recovery state is incomplete"
                )
            fitted_entry_count = sum(
                int(record.fitted) for record in store.validate_all()
            )
            if (
                fitted_entry_count
                != completed.physical_model_fit_count
            ):
                raise ValueError(
                    "completed estimator bundle conflicts with cache totals"
                )
            complete_event = _validate_completed_recovery_progress(
                previous, request, schedules, completed
            )
            if previous.event.algorithm != "complete":
                atomic_write_json(
                    progress_path,
                    progress_to_document(
                        complete_event,
                        last_completed_trial_index=(
                            previous.last_completed_trial_index
                        ),
                    ),
                )
                progress_sink(complete_event)
            return completed
        if previous is not None:
            fitted_entry_count = sum(
                int(record.fitted) for record in store.validate_all()
            )
            if (
                fitted_entry_count
                != previous.event.physical_model_fit_count
            ):
                raise ValueError(
                    "estimator progress conflicts with cache totals"
                )

        coordinator = _ProgressCoordinator(
            request=request,
            schedules=schedules,
            path=progress_path,
            sink=progress_sink,
            previous=previous,
        )
        if previous is None:
            coordinator.persist(emit_boundary=False)

        def observe_cache(
            algorithm: str,
            feature_index: int,
            trial_ordinal: int,
            stats: CoalitionCacheStats,
            fit_delta: int,
        ) -> None:
            coordinator.observe_cache(
                algorithm,
                feature_index,
                trial_ordinal,
                stats,
                fit_delta,
                committed=store.pending_write_count == 0,
            )

        try:
            bundle = build_estimator_bundle(
                request,
                X_train,
                y_train,
                X_val,
                y_val,
                store=store,
                progress_sink=coordinator.on_nested_progress,
                fit_hook=fit_hook,
                cache_observer=observe_cache,
                trial_observer=coordinator.observe_trial,
                _schedules=schedules,
            )
        except BaseException:
            store.flush()
            coordinator.persist(emit_boundary=True)
            raise
        store.flush()
        bundle = coordinator.cumulative_bundle(bundle)
        coordinator.persist_bundle(bundle, complete=False)

    validated = validate_estimator_bundle(
        bundle, request, _schedules=schedules
    )
    atomic_write_json(
        bundle_path, bundle_to_document(validated, request)
    )
    reopened = bundle_from_document(
        read_canonical_json(bundle_path, context="estimator bundle"),
        request,
    )
    coordinator.persist_bundle(reopened, complete=True)
    return reopened


__all__ = [
    "ESTIMATOR_BUNDLE_SCHEMA_VERSION",
    "EstimatorBundle",
    "EstimatorCheckpoint",
    "EstimatorGroupRequest",
    "bundle_from_document",
    "cache_relpath",
    "bundle_relpath",
    "bundle_to_document",
    "ensure_estimator_bundle",
    "group_request_from_context",
    "group_request_to_document",
    "progress_relpath",
    "validate_estimator_bundle",
]
