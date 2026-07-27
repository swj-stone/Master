"""Synchronous, manifest-authoritative ICOMP 2026 stage execution."""

from __future__ import annotations

import hashlib
import csv
import io
from importlib import metadata
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence
from uuid import uuid4

from minppi_mofs.coalition_cache import (
    CoalitionCacheIdentity,
    CoalitionUtilityCache,
    CoalitionUtilityView,
    SQLiteCoalitionStore,
    canonical_coalition,
    hash_numeric_array,
    hash_numeric_frame,
)
from minppi_mofs.experiment_protocol import (
    AuditLog,
    IncompleteRunError,
    JobSpec,
    ResolvedProtocol,
    ResultRecord,
    RunLayout,
    StatusStore,
    assert_stage_complete,
    atomic_write_json,
    preflight_data_manifest,
    publish_json_exclusive,
    read_canonical_json,
    read_rich_data_manifest,
    read_result_record,
    resolved_protocol_hash,
    upsert_data_manifest,
)
from minppi_mofs.modeling import evaluate_fixed_subset, fit_predict as fit_predict
from minppi_mofs.baselines.nsga2_fs import (
    _compute_hv,
    _converged_gen,
    _non_dominated_sort,
    nsga2_feature_selection,
)

if __package__:
    from .summarize_icomp2026 import (
        AGGREGATE_COLUMNS,
        FRONT_COLUMNS,
        PAIRED_COLUMNS,
        RAW_COLUMNS,
        aggregate_canonical_records,
        csv_bytes,
        deduplicate_estimator_group_metrics,
        estimator_projection_reporting_metrics,
        generate_latex_macros,
        provenance_bytes,
        validate_estimator_reporting_schema,
    )
else:
    from summarize_icomp2026 import (  # type: ignore[no-redef]
        AGGREGATE_COLUMNS,
        FRONT_COLUMNS,
        PAIRED_COLUMNS,
        RAW_COLUMNS,
        aggregate_canonical_records,
        csv_bytes,
        deduplicate_estimator_group_metrics,
        estimator_projection_reporting_metrics,
        generate_latex_macros,
        provenance_bytes,
        validate_estimator_reporting_schema,
    )


STAGE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "validate": (),
    "preprocess": ("validate",),
    "estimator": ("preprocess",),
    "removal": ("preprocess", "estimator"),
    "alpha": ("preprocess", "estimator"),
    "nsga2": ("preprocess", "removal"),
    "ablation": ("preprocess", "estimator"),
    "summarize": ("estimator", "removal", "alpha", "nsga2", "ablation"),
    "figures": ("summarize",),
}
_CLASSIFICATION_UTILITIES = frozenset(
    {"roc_auc", "ovr_auc", "accuracy", "pr_auc"}
)
_REGRESSION_UTILITIES = frozenset({"r2", "rmse"})


@dataclass(frozen=True)
class StageContext:
    """Immutable authority and services for exactly one manifest job."""

    protocol: ResolvedProtocol
    layout: RunLayout
    job: JobSpec
    status: StatusStore
    audit: AuditLog


StageExecutor = Callable[[StageContext], ResultRecord]

# Tasks 7--11 register computational handlers here. Keeping this module-level
# boundary lightweight preserves import-safe CLI help in Task 6.
STAGE_EXECUTORS: dict[str, StageExecutor] = {}


@dataclass(frozen=True)
class StageRunSummary:
    """Outcome of one synchronous stage dispatch."""

    stage: str
    succeeded_job_ids: tuple[str, ...] = ()
    skipped_job_ids: tuple[str, ...] = ()
    failed_job_ids: tuple[str, ...] = ()
    blocked_job_ids: tuple[str, ...] = ()
    dependency_issues: tuple[str, ...] = ()
    stopped_early: bool = False

    @property
    def exit_code(self) -> int:
        if self.stopped_early:
            return 75
        if self.failed_job_ids or self.blocked_job_ids or self.dependency_issues:
            return 1
        return 0


def execute_stage(
    stage: str,
    *,
    context: Sequence[StageContext],
    executors: Mapping[str, StageExecutor],
    stop_after_jobs: int | None = None,
) -> StageRunSummary:
    """Execute one stage sequentially in the supplied manifest order."""
    if stage not in STAGE_DEPENDENCIES:
        raise ValueError(f"unknown stage: {stage}")
    if not isinstance(context, Sequence) or isinstance(context, (str, bytes)):
        raise TypeError("context must be an ordered sequence of StageContext values")
    contexts = tuple(context)
    if not contexts or not all(isinstance(item, StageContext) for item in contexts):
        raise ValueError("context must contain StageContext values")
    if isinstance(stop_after_jobs, bool) or (
        stop_after_jobs is not None
        and (not isinstance(stop_after_jobs, int) or stop_after_jobs < 1)
    ):
        raise ValueError("stop_after_jobs must be a positive integer")

    protocol = contexts[0].protocol
    layout = contexts[0].layout
    status = contexts[0].status
    manifest = tuple(item.job for item in contexts)
    if len({job.job_id for job in manifest}) != len(manifest):
        raise ValueError("context contains duplicate manifest jobs")
    if any(
        item.protocol != protocol or item.layout != layout or item.status is not status
        for item in contexts
    ):
        raise ValueError("all contexts must share one protocol, layout, and status store")
    if layout != status.layout:
        raise ValueError("context layout does not match StatusStore run authority")
    authority = status.manifest
    if manifest != authority:
        raise ValueError("context jobs do not match StatusStore manifest authority")
    if (
        resolved_protocol_hash(protocol) != status.protocol_hash
        or any(job.protocol_hash != status.protocol_hash for job in authority)
    ):
        raise ValueError("context protocol does not match StatusStore authority")
    if stop_after_jobs is not None and protocol.run_kind != "smoke":
        raise ValueError("stop_after_jobs is allowed only for a smoke protocol")
    manifest = authority
    stage_jobs = tuple(job for job in manifest if job.stage == stage)
    if not stage_jobs:
        raise ValueError(f"manifest contains no jobs for stage {stage}")

    # Rebind every dispatched context to the current complete trusted manifest.
    audit = AuditLog.from_manifest(layout.audit_path, manifest)
    dependencies = STAGE_DEPENDENCIES[stage]
    if dependencies:
        try:
            assert_stage_complete(layout, manifest, dependencies)
        except IncompleteRunError as exc:
            return StageRunSummary(stage=stage, dependency_issues=exc.issues)

    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    blocked: list[str] = []
    executor = executors.get(stage)
    for job in stage_jobs:
        if any(
            _dependency_state(status, dependency) != "success"
            for dependency in job.depends_on
        ):
            blocked.append(job.job_id)
            continue
        resume_context = StageContext(protocol, layout, job, status, audit)
        result_validator = (
            (lambda record, bound=resume_context: _validate_task11_result_record(bound, record))
            if job.stage in {"summarize", "figures"}
            else None
        )
        decision = status.decide(
            job.job_id, result_validator=result_validator
        )
        if decision.action == "skip":
            skipped.append(job.job_id)
            continue
        try:
            status.start(job.job_id)
        except Exception as exc:
            reopened = StatusStore(layout, manifest)
            if reopened.entry(job.job_id)["state"] != "running":
                raise exc
            status = reopened
        item = StageContext(protocol, layout, job, status, audit)
        try:
            if executor is None:
                raise RuntimeError(f"no executor registered for stage {stage}")
            result = executor(item)
            if not isinstance(result, ResultRecord):
                raise TypeError("stage executor must return ResultRecord")
            atomic_write_json(layout.root / job.output_relpath, asdict(result))
            if job.stage in {"summarize", "figures"}:
                _validate_task11_result_record(item, result)
        except Exception as exc:
            status = _record_failure(status, layout, manifest, job.job_id, exc)
            failed.append(job.job_id)
            continue
        try:
            status.succeed(job.job_id)
        except Exception as exc:
            reopened = StatusStore(layout, manifest)
            state = reopened.entry(job.job_id)["state"]
            if state == "success":
                status = reopened
            elif state == "running":
                status = _record_failure(
                    reopened, layout, manifest, job.job_id, exc
                )
                failed.append(job.job_id)
                continue
            else:
                raise exc
        succeeded.append(job.job_id)
        if stop_after_jobs is not None and len(succeeded) == stop_after_jobs:
            return StageRunSummary(
                stage=stage,
                succeeded_job_ids=tuple(succeeded),
                skipped_job_ids=tuple(skipped),
                failed_job_ids=tuple(failed),
                blocked_job_ids=tuple(blocked),
                stopped_early=True,
            )
    return StageRunSummary(
        stage=stage,
        succeeded_job_ids=tuple(succeeded),
        skipped_job_ids=tuple(skipped),
        failed_job_ids=tuple(failed),
        blocked_job_ids=tuple(blocked),
    )


def _dependency_state(status: StatusStore, job_id: str) -> object:
    try:
        return status.entry(job_id)["state"]
    except ValueError:
        return None


def _record_failure(
    status: StatusStore,
    layout: RunLayout,
    manifest: Sequence[JobSpec],
    job_id: str,
    exception: Exception,
) -> StatusStore:
    try:
        status.fail(job_id, exception)
        return status
    except Exception as persistence_error:
        reopened = StatusStore(layout, manifest)
        if reopened.entry(job_id)["state"] == "failed":
            return reopened
        raise persistence_error from exception


def run_validate_job(context: StageContext) -> ResultRecord:
    """Validate frozen inputs and environment without downloading or fitting."""
    started_at = _timestamp()
    if context.job.stage != "validate":
        raise ValueError("validate executor requires a validate manifest job")
    if context.job.protocol_hash != resolved_protocol_hash(context.protocol):
        raise ValueError("validate job has a foreign protocol hash")
    raw_datasets: list[str] = []
    raw_authority: dict[str, dict[str, object]] = {}
    for dataset in context.protocol.datasets:
        compatible = (
            dataset.utility in _REGRESSION_UTILITIES
            if dataset.task == "regression"
            else dataset.task in {"binary", "multiclass"}
            and dataset.utility in _CLASSIFICATION_UTILITIES
        )
        if not compatible:
            raise ValueError(
                f"task and utility are not compatible for {dataset.dataset_id}"
            )
        if not dataset.raw_path.is_file():
            raise FileNotFoundError(
                f"configured raw cache for {dataset.dataset_id} is missing: "
                f"{dataset.raw_path}"
            )
        raw_payload = dataset.raw_path.read_bytes()
        raw_authority[dataset.dataset_id] = {
            "byte_count": len(raw_payload),
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
        }
        raw_datasets.append(dataset.dataset_id)
    if not context.layout.root.is_dir() or not os.access(context.layout.root, os.W_OK):
        raise PermissionError(f"run directory is not writable: {context.layout.root}")
    authority_relpath = "validate/raw_authority.json"
    raw_authority = _freeze_raw_authority(
        context, context.layout.root / authority_relpath, raw_authority
    )

    environment_path = context.layout.root / "environment.json"
    try:
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen environment metadata is missing or malformed") from exc
    required_environment = {
        "python", "numpy", "pandas", "scikit_learn", "xgboost",
        "xgboost_build_info", "cuda_build_info", "backend",
    }
    if not isinstance(environment, dict) or not required_environment.issubset(environment):
        raise ValueError("frozen environment metadata is incomplete")
    if environment["backend"] != context.protocol.backend:
        raise ValueError("frozen environment backend does not match the protocol")
    for package in ("python", "numpy", "pandas", "scikit_learn", "xgboost"):
        if (
            not isinstance(environment[package], str)
            or not environment[package].strip()
            or environment[package].strip().casefold() == "unavailable"
        ):
            raise ValueError(f"frozen environment package version is invalid: {package}")
    _validate_xgboost_build_info(
        environment["xgboost_build_info"], context.protocol.backend
    )

    required_executors = {"validate", "preprocess"}
    required_executors.update(job.stage for job in context.status.manifest)
    missing_executors = sorted(required_executors - set(STAGE_EXECUTORS))
    if missing_executors:
        raise RuntimeError(
            f"registered stage executor is missing for: {missing_executors}"
        )
    cuda_visibility = _cuda_visibility(context.protocol.backend)
    checks = {
        "backend": context.protocol.backend,
        "cuda_visibility": cuda_visibility,
        "environment": environment,
        "raw_datasets": raw_datasets,
        "raw_authority": raw_authority,
        "registered_stages": sorted(STAGE_EXECUTORS),
        "run_directory_writable": True,
    }
    checks_relpath = "validate/validation_checks.json"
    atomic_write_json(context.layout.root / checks_relpath, checks)
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics={
            "checks_passed": True,
            "dataset_count": len(raw_datasets),
            "backend": context.protocol.backend,
        },
        artifacts={"checks": checks_relpath, "raw_authority": authority_relpath},
        started_at=started_at,
        finished_at=_timestamp(),
    )


def run_preprocess_job(context: StageContext) -> ResultRecord:
    """Create one leak-free audited split from an existing raw CSV cache."""
    started_at = _timestamp()
    if context.job.stage != "preprocess":
        raise ValueError("preprocess executor requires a preprocess manifest job")
    dataset = next(
        (item for item in context.protocol.datasets if item.dataset_id == context.job.dataset_id),
        None,
    )
    if dataset is None or context.job.seed not in context.protocol.seeds:
        raise ValueError("preprocess job is outside the protocol dataset/seed scope")
    if context.job.parameters != {"task": dataset.task, "utility": dataset.utility}:
        raise ValueError("preprocess job parameters do not match dataset authority")
    if not dataset.raw_path.is_file():
        raise FileNotFoundError(
            f"configured raw cache for {dataset.dataset_id} is missing: {dataset.raw_path}"
        )

    from io import BytesIO

    import numpy as np
    import pandas as pd

    from minppi_mofs.data_protocol import prepare_split

    raw_bytes = dataset.raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    _assert_frozen_raw_authority(
        context, dataset.dataset_id, raw_sha256, len(raw_bytes)
    )
    try:
        frame = pd.read_csv(BytesIO(raw_bytes))
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"could not parse configured raw CSV: {dataset.raw_path}") from exc
    if dataset.target not in frame.columns:
        raise ValueError(f"raw CSV is missing target column {dataset.target!r}")
    if len(frame) < 3:
        raise ValueError("raw CSV has too few rows for a three-way split")
    raw_features = frame.drop(columns=[dataset.target])
    if raw_features.shape[1] < 1 or raw_features.columns.duplicated().any():
        raise ValueError("raw CSV feature columns are empty or duplicated")
    try:
        features = raw_features.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw CSV features must be numeric") from exc
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("raw CSV features must be finite")
    targets, target_encoding = _coerce_target(frame[dataset.target], dataset.task)

    split = prepare_split(
        features,
        targets,
        task=dataset.task,
        seed=context.job.seed,
        max_features=dataset.max_features,
        train_frac=dataset.train_frac,
        val_frac=dataset.val_frac,
        test_frac=dataset.test_frac,
        variance_threshold=dataset.variance_threshold,
    )
    prefix = f"preprocess/{dataset.dataset_id}/s{context.job.seed}"
    partition_values = {
        "train": (split.X_train, split.y_train),
        "val": (split.X_val, split.y_val),
        "test": (split.X_test, split.y_test),
    }
    partition_bytes: dict[str, bytes] = {}
    partition_hashes: dict[str, str] = {}
    label_summaries: dict[str, dict[str, object]] = {}
    index_values: dict[str, list[object]] = {}
    index_hashes: dict[str, str] = {}
    for name, (partition_features, partition_targets) in partition_values.items():
        partition = partition_features.copy(deep=True)
        partition[dataset.target] = partition_targets
        payload = partition.to_csv(index=False, lineterminator="\n").encode("utf-8")
        partition_bytes[name] = payload
        partition_hashes[name] = hashlib.sha256(payload).hexdigest()
        label_summaries[name] = _label_summary(partition_targets, dataset.task)
        safe_indices = [_json_safe_index(value) for value in partition_features.index]
        index_values[name] = safe_indices
        index_hashes[name] = hashlib.sha256(_canonical_json_bytes(safe_indices)).hexdigest()

    transform_metadata = {
        "scaler_mean": list(split.scaler_mean),
        "scaler_scale": list(split.scaler_scale),
        "selected_features": list(split.selected_features),
        "target_encoding": target_encoding,
        "variance_features": list(split.variance_features),
    }
    split_metadata = {
        "dataset_id": dataset.dataset_id,
        "seed": context.job.seed,
        "target": dataset.target,
        "task": dataset.task,
        "selected_features": list(split.selected_features),
        "variance_features": list(split.variance_features),
        "scaler_mean": list(split.scaler_mean),
        "scaler_scale": list(split.scaler_scale),
        "train_indices": index_values["train"],
        "val_indices": index_values["val"],
        "test_indices": index_values["test"],
        "train_index_sha256": index_hashes["train"],
        "val_index_sha256": index_hashes["val"],
        "test_index_sha256": index_hashes["test"],
        "target_encoding": target_encoding,
    }
    relpaths = {name: f"{prefix}/{name}.csv" for name in partition_values}
    metadata_relpath = f"{prefix}/split_metadata.json"
    metadata_bytes = _artifact_json_bytes(split_metadata)

    manifest_row: dict[str, object] = {
        "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id,
        "dataset_id": dataset.dataset_id,
        "seed": context.job.seed,
        "task": dataset.task,
        "utility": dataset.utility,
        "target": dataset.target,
        "raw_file_sha256": raw_sha256,
        "raw_row_count": len(frame),
        "raw_feature_count": raw_features.shape[1],
        "selected_feature_count": len(split.selected_features),
        "selected_features_json": _canonical_json_text(list(split.selected_features)),
        "train_relpath": relpaths["train"],
        "val_relpath": relpaths["val"],
        "test_relpath": relpaths["test"],
        "train_row_count": len(split.X_train),
        "val_row_count": len(split.X_val),
        "test_row_count": len(split.X_test),
        "train_index_sha256": index_hashes["train"],
        "val_index_sha256": index_hashes["val"],
        "test_index_sha256": index_hashes["test"],
        "train_file_sha256": partition_hashes["train"],
        "val_file_sha256": partition_hashes["val"],
        "test_file_sha256": partition_hashes["test"],
        "train_label_summary_json": _canonical_json_text(label_summaries["train"]),
        "val_label_summary_json": _canonical_json_text(label_summaries["val"]),
        "test_label_summary_json": _canonical_json_text(label_summaries["test"]),
        "transform_metadata_json": _canonical_json_text(transform_metadata),
    }
    manifest_path = context.layout.root / "data_manifest.csv"
    action = preflight_data_manifest(
        manifest_path, context.protocol, manifest_row
    )
    _publish_preprocess_snapshot(
        context,
        action=action,
        manifest_path=manifest_path,
        manifest_row=manifest_row,
        partition_bytes=partition_bytes,
        metadata_bytes=metadata_bytes,
    )
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics={
            "raw_row_count": len(frame),
            "raw_feature_count": raw_features.shape[1],
            "selected_feature_count": len(split.selected_features),
            "train_row_count": len(split.X_train),
            "val_row_count": len(split.X_val),
            "test_row_count": len(split.X_test),
        },
        artifacts={
            "train": relpaths["train"],
            "val": relpaths["val"],
            "test": relpaths["test"],
            "split_metadata": metadata_relpath,
            "data_manifest": "data_manifest.csv",
        },
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _load_prepared_train_validation(context: StageContext):
    """Load verified prepared train/validation partitions without test access."""
    from io import BytesIO

    import numpy as np
    import pandas as pd

    rows = read_rich_data_manifest(
        context.layout.root / "data_manifest.csv", context.protocol
    )
    row = next(
        (
            item
            for item in rows
            if item["dataset_id"] == context.job.dataset_id
            and item["seed"] == context.job.seed
        ),
        None,
    )
    if row is None:
        raise ValueError("prepared data manifest has no row for this job")
    selected = json.loads(str(row["selected_features_json"]))
    expected_columns = [*selected, row["target"]]
    loaded: list[tuple[pd.DataFrame, np.ndarray]] = []
    for partition in ("train", "val"):
        path = context.layout.root / str(row[f"{partition}_relpath"])
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"prepared {partition} CSV is missing") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row[f"{partition}_file_sha256"]:
            raise ValueError(f"prepared {partition} CSV SHA-256 drift")
        try:
            frame = pd.read_csv(BytesIO(payload))
        except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"prepared {partition} CSV is malformed") from exc
        if list(frame.columns) != expected_columns:
            raise ValueError(f"prepared {partition} feature column order is invalid")
        if len(frame) != row[f"{partition}_row_count"]:
            raise ValueError(f"prepared {partition} row count drift")
        if frame.columns.duplicated().any():
            raise ValueError(f"prepared {partition} columns are duplicated")
        try:
            numeric = frame.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prepared {partition} data must be numeric") from exc
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"prepared {partition} data must be finite")
        features = numeric[selected].astype(float)
        target = numeric[str(row["target"])].to_numpy()
        if row["task"] != "regression":
            if not np.equal(target, np.floor(target)).all():
                raise ValueError(f"prepared {partition} target must use integer labels")
            target = target.astype(int)
        else:
            target = target.astype(float)
        loaded.append((features, target))
    (X_train, y_train), (X_val, y_val) = loaded
    for partition, target in (("train", y_train), ("val", y_val)):
        expected_summary = json.loads(str(row[f"{partition}_label_summary_json"]))
        actual_summary = _label_summary(target, str(row["task"]))
        if not _label_summaries_match(
            actual_summary, expected_summary, task=str(row["task"])
        ):
            raise ValueError(
                f"prepared {partition} label summary conflicts with manifest"
            )
    if row["task"] in {"binary", "multiclass"}:
        train_classes = tuple(int(value) for value in np.unique(y_train))
        val_classes = set(int(value) for value in np.unique(y_val))
        if row["task"] == "binary":
            if train_classes != (0, 1):
                raise ValueError("prepared binary train classes must be exactly {0,1}")
        elif len(train_classes) < 2 or train_classes != tuple(
            range(len(train_classes))
        ):
            raise ValueError(
                "prepared multiclass train classes must be zero-based contiguous labels"
            )
        if not val_classes.issubset(train_classes):
            raise ValueError(
                "prepared validation labels contain an unseen class outside train"
            )
    return X_train, y_train, X_val, y_val, row


class _OpenedCoalitionCache:
    """Own one immediately opened persistent store and its utility cache."""

    def __init__(
        self, store: SQLiteCoalitionStore, cache: CoalitionUtilityCache
    ) -> None:
        self._store = store
        self._cache = cache

    def __enter__(self) -> CoalitionUtilityCache:
        return self._cache

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __getattr__(self, name: str) -> object:
        return getattr(self._cache, name)

    def close(self) -> None:
        self._store.close()


class _OpenedSharedBaselineUtility:
    """Own one shared baseline evaluator and its persistent store."""

    def __init__(self, store: SQLiteCoalitionStore, evaluator: Callable) -> None:
        self._store = store
        self._evaluator = evaluator

    def __enter__(self) -> Callable:
        return self._evaluator

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._store.close()


def _validated_cache_manifest_authority(
    context: StageContext, data_row: Mapping[str, object]
) -> tuple[dict[str, object], object]:
    """Validate one supplied rich row against frozen run and protocol authority."""
    if not isinstance(context, StageContext):
        raise TypeError("context must be a StageContext")
    if not isinstance(data_row, Mapping):
        raise TypeError("data_row must be a rich manifest mapping")
    protocol_hash = resolved_protocol_hash(context.protocol)
    if (
        context.layout != context.status.layout
        or context.job not in context.status.manifest
        or context.job.protocol_hash != protocol_hash
        or context.status.protocol_hash != protocol_hash
    ):
        raise ValueError("cache context conflicts with frozen run authority")
    rows = read_rich_data_manifest(
        context.layout.root / "data_manifest.csv", context.protocol
    )
    matches = tuple(
        row
        for row in rows
        if row["dataset_id"] == context.job.dataset_id
        and row["seed"] == context.job.seed
    )
    if len(matches) != 1:
        raise ValueError("cache rich manifest authority is missing or duplicated")
    row = matches[0]
    if dict(data_row) != row:
        raise ValueError("cache data row conflicts with rich manifest authority")
    dataset = next(
        (
            item
            for item in context.protocol.datasets
            if item.dataset_id == context.job.dataset_id
        ),
        None,
    )
    if (
        dataset is None
        or row["task"] != dataset.task
        or row["utility"] != dataset.utility
        or row["target"] != dataset.target
    ):
        raise ValueError("cache data row conflicts with dataset authority")
    return row, dataset


def _validated_cache_frames(
    row: Mapping[str, object],
    X_train,
    y_train,
    X_eval,
    y_eval,
    *,
    train_row_count: int,
    eval_row_count: int,
):
    import numpy as np
    import pandas as pd

    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_eval, pd.DataFrame):
        raise TypeError("cache feature frames must be pandas DataFrames")
    try:
        selected = tuple(json.loads(str(row["selected_features_json"])))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("cache manifest feature order is invalid") from exc
    if (
        tuple(X_train.columns) != selected
        or tuple(X_eval.columns) != selected
        or X_train.columns.duplicated().any()
        or X_eval.columns.duplicated().any()
    ):
        raise ValueError("cache feature order conflicts with rich manifest")
    train_target = np.asarray(y_train)
    eval_target = np.asarray(y_eval)
    if (
        len(X_train) != train_row_count
        or len(X_eval) != eval_row_count
        or train_target.ndim != 1
        or eval_target.ndim != 1
        or len(train_target) != len(X_train)
        or len(eval_target) != len(X_eval)
    ):
        raise ValueError("cache frame shape conflicts with rich manifest")
    hash_numeric_frame(X_train)
    hash_numeric_frame(X_eval)
    hash_numeric_array(train_target)
    hash_numeric_array(eval_target)
    task = str(row["task"])
    if task != "regression":
        if (
            not np.equal(train_target, np.floor(train_target)).all()
            or not np.equal(eval_target, np.floor(eval_target)).all()
        ):
            raise ValueError("cache classification targets must be integer labels")
        train_labels = tuple(int(value) for value in np.unique(train_target))
        eval_labels = {int(value) for value in np.unique(eval_target)}
        if task == "binary" and train_labels != (0, 1):
            raise ValueError("cache binary train classes must be exactly {0,1}")
        if task == "multiclass" and train_labels != tuple(
            range(len(train_labels))
        ):
            raise ValueError(
                "cache multiclass train classes must be zero-based contiguous"
            )
        if not eval_labels.issubset(train_labels):
            raise ValueError("cache evaluation labels exceed training authority")
    return X_train, train_target, X_eval, eval_target


def _coalition_library_versions() -> dict[str, str]:
    import numpy as np

    try:
        xgboost_version = metadata.version("xgboost")
    except metadata.PackageNotFoundError:
        xgboost_version = "unavailable"
    return {"numpy": np.__version__, "xgboost": xgboost_version}


def _cache_identity_from_validated_frames(
    context: StageContext,
    row: Mapping[str, object],
    X_train,
    y_train,
    X_eval,
    y_eval,
    *,
    namespace: str,
) -> CoalitionCacheIdentity:
    import numpy as np

    class_labels = (
        ()
        if row["task"] == "regression"
        else tuple(int(value) for value in np.unique(y_train))
    )
    return CoalitionCacheIdentity(
        schema_version=context.protocol.execution.cache_schema,
        protocol_hash=context.job.protocol_hash,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        namespace=namespace,
        train_features_sha256=hash_numeric_frame(X_train),
        train_target_sha256=hash_numeric_array(y_train),
        eval_features_sha256=hash_numeric_frame(X_eval),
        eval_target_sha256=hash_numeric_array(y_eval),
        ordered_features=tuple(str(feature) for feature in X_train.columns),
        task=str(row["task"]),
        class_labels=class_labels,
        model_params=context.protocol.model_params,
        backend=context.protocol.backend,
        model_seed=context.job.seed,
        library_versions=_coalition_library_versions(),
    )


def _selection_cache_identity(
    context: StageContext,
    data_row: Mapping[str, object],
    X_train,
    y_train,
    X_val,
    y_val,
) -> CoalitionCacheIdentity:
    """Bind selection reuse to frozen authority and loaded numeric content."""
    row, _dataset = _validated_cache_manifest_authority(context, data_row)
    frames = _validated_cache_frames(
        row,
        X_train,
        y_train,
        X_val,
        y_val,
        train_row_count=int(row["train_row_count"]),
        eval_row_count=int(row["val_row_count"]),
    )
    return _cache_identity_from_validated_frames(
        context, row, *frames, namespace="selection_validation"
    )


def _cache_path(context: StageContext, namespace: str) -> Path:
    if context.layout.cache_root != context.layout.root / "cache":
        raise ValueError("run cache root is not canonical")
    if namespace not in {
        "selection_validation",
        "baseline_internal",
        "final_test",
    }:
        raise ValueError("coalition cache namespace is invalid")
    if __package__:
        from .icomp2026_estimator_bundle import _canonical_run_path
    else:
        from icomp2026_estimator_bundle import (  # type: ignore[no-redef]
            _canonical_run_path,
        )

    return _canonical_run_path(
        context.layout.root,
        (
            f"cache/{namespace}/{context.job.dataset_id}/"
            f"s{context.job.seed}/coalitions.sqlite3"
        ),
        context="coalition cache",
    )


def _open_bound_cache(
    context: StageContext,
    identity: CoalitionCacheIdentity,
    frames: tuple[object, object, object, object],
) -> _OpenedCoalitionCache:
    path = _cache_path(context, identity.namespace)
    store = SQLiteCoalitionStore(
        path,
        identity,
        commit_every_fits=context.protocol.execution.checkpoint_every_fits,
    )
    try:
        cache = CoalitionUtilityCache(identity, *frames, store)
        cache.validate_persistent_records()
    except BaseException:
        store.close()
        raise
    return _OpenedCoalitionCache(store, cache)


def _open_selection_cache(
    context: StageContext,
    frames: object,
    data_row: Mapping[str, object],
) -> _OpenedCoalitionCache:
    """Immediately reopen the run-scoped training/validation cache."""
    row, _dataset = _validated_cache_manifest_authority(context, data_row)
    if (
        not isinstance(frames, (tuple, list))
        or len(frames) != 4
    ):
        raise TypeError("selection cache frames must contain four values")
    validated = _validated_cache_frames(
        row,
        *frames,
        train_row_count=int(row["train_row_count"]),
        eval_row_count=int(row["val_row_count"]),
    )
    identity = _cache_identity_from_validated_frames(
        context, row, *validated, namespace="selection_validation"
    )
    return _open_bound_cache(context, identity, validated)


def _open_shared_baseline_utility(
    context: StageContext,
    X,
    y,
    data_row: Mapping[str, object],
    diagnostics: dict[str, int],
) -> _OpenedSharedBaselineUtility:
    """Open one baseline-internal evaluator shared by MinShap and Banzhaf."""
    from minppi_mofs.baselines._coalition_utility import (
        _prepare_baseline_internal_split,
        make_shared_baseline_utility,
    )

    row, dataset = _validated_cache_manifest_authority(context, data_row)
    if (
        str(row["task"]) != dataset.task
        or str(row["utility"]) != dataset.utility
    ):
        raise ValueError("baseline cache policy conflicts with dataset authority")
    prepared = _prepare_baseline_internal_split(
        X,
        y,
        task=dataset.task,
        seed=context.job.seed,
    )
    identity = _cache_identity_from_validated_frames(
        context,
        row,
        prepared.X_train,
        prepared.y_train,
        prepared.X_eval,
        prepared.y_eval,
        namespace="baseline_internal",
    )
    store = SQLiteCoalitionStore(
        _cache_path(context, "baseline_internal"),
        identity,
        commit_every_fits=context.protocol.execution.checkpoint_every_fits,
    )
    try:
        evaluator = make_shared_baseline_utility(
            X,
            y,
            task=dataset.task,
            utility=dataset.utility,
            model_params=context.protocol.model_params,
            backend=context.protocol.backend,
            seed=context.job.seed,
            store=store,
            diagnostics=diagnostics,
            prepared=prepared,
        )
    except BaseException:
        store.close()
        raise
    return _OpenedSharedBaselineUtility(store, evaluator)


def _open_final_test_cache(
    context: StageContext,
    train_val: object,
    test: object,
    data_row: Mapping[str, object],
    reason: str,
) -> _OpenedCoalitionCache:
    """Audit the test read, then open its separately identified cache."""
    row, _dataset = _validated_cache_manifest_authority(context, data_row)
    if (
        not isinstance(train_val, (tuple, list))
        or len(train_val) != 2
    ):
        raise TypeError("final cache train_val must contain features and target")
    audited_test = _task9_load_test(context, row, reason=reason)
    if test is not None:
        if not isinstance(test, (tuple, list)) or len(test) != 2:
            raise TypeError("final cache test must contain features and target")
        if (
            hash_numeric_frame(test[0]) != hash_numeric_frame(audited_test[0])
            or hash_numeric_array(test[1]) != hash_numeric_array(audited_test[1])
        ):
            raise ValueError("supplied test data conflicts with audited test data")
    frames = _validated_cache_frames(
        row,
        train_val[0],
        train_val[1],
        audited_test[0],
        audited_test[1],
        train_row_count=(
            int(row["train_row_count"]) + int(row["val_row_count"])
        ),
        eval_row_count=int(row["test_row_count"]),
    )
    identity = _cache_identity_from_validated_frames(
        context, row, *frames, namespace="final_test"
    )
    return _open_bound_cache(context, identity, frames)


def _label_summaries_match(
    actual: Mapping[str, object], expected: object, *, task: str
) -> bool:
    """Compare frozen summaries, tolerating only CSV float round-trip noise."""
    if not isinstance(expected, dict) or set(actual) != set(expected):
        return False
    if task != "regression":
        return actual == expected
    if actual["count"] != expected["count"]:
        return False
    return all(
        isinstance(expected[field], (int, float))
        and not isinstance(expected[field], bool)
        and __import__("math").isclose(
            float(actual[field]),
            float(expected[field]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for field in ("max", "mean", "min", "std")
    )


def group_request_from_context(
    context: StageContext,
    prepared_manifest_row: Mapping[str, object],
    feature_order: Sequence[str],
):
    """Load the grouped request builder only for grouped estimator execution."""
    if __package__:
        from .icomp2026_estimator_bundle import (
            group_request_from_context as build_request,
        )
    else:
        from icomp2026_estimator_bundle import (  # type: ignore[no-redef]
            group_request_from_context as build_request,
        )

    return build_request(context, prepared_manifest_row, feature_order)


def ensure_estimator_bundle(
    request: object,
    frames: object,
    run_root: Path,
    progress_sink: Callable[[object], None],
    *,
    build_owner_job_id: str,
):
    """Load the grouped bundle builder only for grouped estimator execution."""
    if __package__:
        from .icomp2026_estimator_bundle import (
            ensure_estimator_bundle as ensure_bundle,
        )
    else:
        from icomp2026_estimator_bundle import (  # type: ignore[no-redef]
            ensure_estimator_bundle as ensure_bundle,
        )

    return ensure_bundle(
        request,
        frames,
        run_root,
        progress_sink,
        build_owner_job_id=build_owner_job_id,
    )


def _validate_grouped_estimator_bundle(bundle: object, request: object):
    if __package__:
        from .icomp2026_estimator_bundle import validate_estimator_bundle
    else:
        from icomp2026_estimator_bundle import (  # type: ignore[no-redef]
            validate_estimator_bundle,
        )

    return validate_estimator_bundle(bundle, request)


def _emit_estimator_progress(log_path: Path, event: object) -> None:
    if __package__:
        from .icomp2026_progress import emit_progress
    else:
        from icomp2026_progress import emit_progress  # type: ignore[no-redef]

    emit_progress(log_path, event)


def run_estimator_job(context: StageContext) -> ResultRecord:
    """Dispatch one estimator job solely by its frozen execution mode."""
    if context.protocol.execution.mode == "legacy":
        return _run_legacy_estimator_job(context)
    if context.protocol.execution.mode == "grouped_nested_v1":
        return _run_grouped_estimator_job(context)
    raise ValueError("unsupported execution mode")


def _run_grouped_estimator_job(context: StageContext) -> ResultRecord:
    """Project one canonical estimator job from its completed group bundle."""
    started_at = _timestamp()
    X_train, y_train, X_val, y_val, manifest_row = (
        _load_prepared_train_validation(context)
    )
    feature_order = tuple(str(feature) for feature in X_train.columns)
    request = group_request_from_context(context, manifest_row, feature_order)
    bundle = ensure_estimator_bundle(
        request=request,
        frames=(X_train, y_train, X_val, y_val),
        run_root=context.layout.root,
        progress_sink=lambda event: _emit_estimator_progress(
            context.layout.log_path, event
        ),
        build_owner_job_id=context.job.job_id,
    )
    bundle = _validate_grouped_estimator_bundle(bundle, request)

    budget = context.job.parameters.get("budget")
    if (
        request.protocol_hash != context.job.protocol_hash
        or request.dataset_id != context.job.dataset_id
        or request.seed != context.job.seed
        or context.job.method not in request.methods
        or budget not in request.budgets
    ):
        raise ValueError("estimator job conflicts with grouped request authority")
    matches = tuple(
        checkpoint
        for checkpoint in bundle.checkpoints
        if checkpoint.method == context.job.method
        and checkpoint.budget == budget
    )
    if len(matches) != 1:
        raise ValueError(
            "grouped estimator checkpoint is missing or duplicated"
        )
    checkpoint = matches[0]
    checkpoint_features = tuple(
        feature for feature, _score in checkpoint.scores
    )
    if (
        checkpoint_features != feature_order
        or checkpoint_features != request.ordered_features
    ):
        raise ValueError(
            "grouped estimator checkpoint feature order is invalid"
        )

    ordered_scores = [
        {"feature": feature, "score": score}
        for feature, score in checkpoint.scores
    ]
    logical_unique = checkpoint.logical_unique_coalition_count
    score_relpath = (
        f"estimator/{context.job.dataset_id}/s{context.job.seed}/"
        f"{context.job.method}/b{budget}/scores.json"
    )
    score_document = {
        "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id,
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "method": context.job.method,
        "budget": budget,
        "scores": ordered_scores,
        "model_fit_count": logical_unique,
        "coalition_count": logical_unique + 1,
        "logical_evaluation_count": checkpoint.logical_evaluation_count,
        "logical_unique_coalition_count": logical_unique,
        "physical_model_fit_count": checkpoint.physical_model_fit_count,
        "cache_hit_count": checkpoint.cache_hit_count,
        "group_request_digest": request.digest,
        "group_physical_model_fit_count": bundle.physical_model_fit_count,
        "group_cache_hit_count": bundle.cache_hit_count,
        "group_runtime_s": bundle.group_runtime_s,
    }
    if checkpoint.schedule_digest is not None:
        score_document["schedule_digest"] = checkpoint.schedule_digest
    _publish_idempotent_json(
        context.layout.root / score_relpath,
        score_document,
        context="estimator score artifact",
    )

    metrics: dict[str, float | int | str | bool | None] = {
        "budget": budget,
        "model_fit_count": logical_unique,
        "coalition_count": logical_unique + 1,
        "logical_evaluation_count": checkpoint.logical_evaluation_count,
        "logical_unique_coalition_count": logical_unique,
        "physical_model_fit_count": checkpoint.physical_model_fit_count,
        "cache_hit_count": checkpoint.cache_hit_count,
        "runtime_s": checkpoint.component_runtime_s,
        "group_request_digest": request.digest,
        "group_physical_model_fit_count": bundle.physical_model_fit_count,
        "group_cache_hit_count": bundle.cache_hit_count,
        "group_runtime_s": bundle.group_runtime_s,
    }
    if checkpoint.schedule_digest is not None:
        metrics["schedule_digest"] = checkpoint.schedule_digest
    scores = dict(checkpoint.scores)
    exact_scores = (
        scores
        if context.job.method == "exact"
        else _completed_exact_scores(
            context,
            budget=budget,
            expected_features=feature_order,
        )
    )
    if exact_scores is not None:
        metrics.update(_agreement_metrics(scores, exact_scores))
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics=metrics,
        artifacts={"scores": score_relpath},
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _run_legacy_estimator_job(context: StageContext) -> ResultRecord:
    """Run one manifest-authorized MinPPI estimator on train/validation only."""
    import math

    from minppi_mofs.min_ppi import (
        algorithm_b_schedule_digest,
        estimate_minppi_exact,
        estimate_minppi_hybrid,
        estimate_minppi_mi_correlation_guided,
        estimate_minppi_random_mc,
        make_algorithm_b_perturbation_schedule,
    )
    from minppi_mofs.shapg_integration import build_mi_correlation_guidance
    from minppi_mofs.voting_game import AccFullCache

    started_at = _timestamp()
    runtime_start = time.perf_counter()
    if context.job.stage != "estimator":
        raise ValueError("estimator executor requires an estimator manifest job")
    if context.job not in context.status.manifest:
        raise ValueError("estimator job is outside the frozen status manifest")
    dataset = next(
        (item for item in context.protocol.datasets if item.dataset_id == context.job.dataset_id),
        None,
    )
    if dataset is None:
        raise ValueError("estimator job has an unknown dataset")
    expected_parameter_keys = {"task", "utility", "budget"}
    if set(context.job.parameters) != expected_parameter_keys:
        raise ValueError("estimator job parameters have an invalid schema")
    if (
        context.job.parameters["task"] != dataset.task
        or context.job.parameters["utility"] != dataset.utility
    ):
        raise ValueError("estimator job parameters do not match dataset authority")
    budget = context.job.parameters["budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("estimator budget must be a positive integer")
    if budget not in context.protocol.estimator_budgets:
        raise ValueError("estimator budget is outside protocol authority")
    method = context.job.method
    if method not in {"exact", "random_mc", "mi_correlation_guided", "hybrid"}:
        raise ValueError("estimator method is not canonical")

    X_train, y_train, X_val, y_val, _ = _load_prepared_train_validation(context)
    mi_scores = None
    correlation_neighbors = None
    perturbation_schedule = None
    schedule_digest = None
    if method in {"mi_correlation_guided", "hybrid"}:
        perturbation_schedule = make_algorithm_b_perturbation_schedule(
            tuple(X_train.columns), K_adv=budget, seed=context.job.seed
        )
        schedule_digest = algorithm_b_schedule_digest(perturbation_schedule)
        mi_scores, correlation_neighbors = build_mi_correlation_guidance(
            X_train,
            y_train,
            task=dataset.task,
            seed=context.job.seed,
        )
    model_params = dict(context.protocol.model_params)
    model_params["tree_method"] = "hist"
    model_params["device"] = "cuda" if context.protocol.backend == "cuda" else "cpu"
    cache = AccFullCache(
        X_train,
        X_val,
        y_train,
        y_val,
        task=dataset.task,
        utility=dataset.utility,
        seed=context.job.seed,
        model_params=model_params,
    )
    common = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "alpha": context.protocol.alpha_default,
        "task": dataset.task,
        "utility": dataset.utility,
        "acc_cache": cache,
        "seed": context.job.seed,
    }
    if method == "exact":
        scores = estimate_minppi_exact(**common)
    elif method == "random_mc":
        scores = estimate_minppi_random_mc(K=budget, **common)
    elif method == "mi_correlation_guided":
        scores = estimate_minppi_mi_correlation_guided(
            K_adv=budget,
            mi_scores=mi_scores,
            correlation_neighbors=correlation_neighbors,
            perturbation_schedule=perturbation_schedule,
            **common,
        )
    else:
        scores = estimate_minppi_hybrid(
            K=budget,
            K_adv=budget,
            mi_scores=mi_scores,
            correlation_neighbors=correlation_neighbors,
            perturbation_schedule=perturbation_schedule,
            **common,
        )
    runtime_s = time.perf_counter() - runtime_start
    if not math.isfinite(runtime_s) or runtime_s < 0:
        raise RuntimeError("estimator runtime is invalid")
    ordered_scores = [
        {"feature": feature, "score": int(scores[feature])}
        for feature in X_train.columns
    ]
    score_relpath = (
        f"estimator/{dataset.dataset_id}/s{context.job.seed}/"
        f"{method}/b{budget}/scores.json"
    )
    score_document = {
        "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id,
        "dataset_id": dataset.dataset_id,
        "seed": context.job.seed,
        "method": method,
        "budget": budget,
        "scores": ordered_scores,
        "model_fit_count": cache.model_fit_count,
        "coalition_count": cache.coalition_count,
    }
    if schedule_digest is not None:
        score_document["schedule_digest"] = schedule_digest
    _publish_idempotent_json(
        context.layout.root / score_relpath,
        score_document,
        context="estimator score artifact",
    )
    metrics: dict[str, float | int | str | bool | None] = {
        "budget": budget,
        "model_fit_count": cache.model_fit_count,
        "coalition_count": cache.coalition_count,
        "runtime_s": runtime_s,
    }
    exact_scores = scores if method == "exact" else _completed_exact_scores(
        context, budget=budget, expected_features=tuple(X_train.columns)
    )
    if exact_scores is not None:
        metrics.update(_agreement_metrics(scores, exact_scores))
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics=metrics,
        artifacts={"scores": score_relpath},
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _completed_exact_scores(
    context: StageContext, *, budget: int, expected_features: tuple[str, ...]
) -> dict[str, int] | None:
    exact_job = next(
        (
            job
            for job in context.status.manifest
            if job.stage == "estimator"
            and job.dataset_id == context.job.dataset_id
            and job.seed == context.job.seed
            and job.method == "exact"
            and job.parameters.get("budget") == budget
        ),
        None,
    )
    if exact_job is None or context.status.entry(exact_job.job_id)["state"] != "success":
        return None
    record = read_result_record(context.layout.root / exact_job.output_relpath, exact_job)
    if set(record.artifacts) != {"scores"}:
        raise ValueError("exact result record has invalid score artifacts")
    expected_relpath = (
        f"estimator/{exact_job.dataset_id}/s{exact_job.seed}/"
        f"exact/b{budget}/scores.json"
    )
    if record.artifacts["scores"] != expected_relpath:
        raise ValueError("exact result score artifact path is invalid")
    return _read_score_artifact(
        context.layout.root / expected_relpath,
        expected_job=exact_job,
        budget=budget,
        expected_features=expected_features,
        expected_metrics=record.metrics,
        expected_execution_mode=context.protocol.execution.mode,
    )


def _completed_estimator_checkpoint(
    context: StageContext,
    method: str,
    budget: int,
    expected_features: Sequence[str],
):
    """Return one fully validated grouped-estimator checkpoint authority."""
    if not isinstance(context, StageContext):
        raise TypeError("context must be a StageContext")
    if (
        isinstance(expected_features, (str, bytes))
        or not isinstance(expected_features, Sequence)
    ):
        raise TypeError("expected_features must be an ordered sequence")
    features = tuple(expected_features)
    if (
        not features
        or any(
            not isinstance(feature, str) or not feature
            for feature in features
        )
        or len(set(features)) != len(features)
    ):
        raise ValueError("expected estimator features are invalid")
    if (
        not isinstance(method, str)
        or method not in context.protocol.estimator_methods
    ):
        raise ValueError("estimator checkpoint method is invalid")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget not in context.protocol.estimator_budgets
    ):
        raise ValueError("estimator checkpoint budget is invalid")
    protocol_hash = resolved_protocol_hash(context.protocol)
    if (
        context.protocol.execution.mode != "grouped_nested_v1"
        or context.layout != context.status.layout
        or context.job not in context.status.manifest
        or context.job.protocol_hash != protocol_hash
        or context.status.protocol_hash != protocol_hash
    ):
        raise ValueError(
            "completed estimator lookup requires grouped run authority"
        )
    dataset = next(
        (
            item
            for item in context.protocol.datasets
            if item.dataset_id == context.job.dataset_id
        ),
        None,
    )
    if dataset is None or context.job.seed not in context.protocol.seeds:
        raise ValueError("completed estimator dataset or seed is invalid")

    matches = tuple(
        job
        for job in context.status.manifest
        if job.stage == "estimator"
        and job.dataset_id == context.job.dataset_id
        and job.seed == context.job.seed
        and job.method == method
        and job.parameters.get("budget") == budget
    )
    if len(matches) != 1:
        raise ValueError(
            "completed estimator authority is missing or duplicated"
        )
    job = matches[0]
    if (
        job.protocol_hash != protocol_hash
        or set(job.parameters) != {"task", "utility", "budget"}
        or job.parameters["task"] != dataset.task
        or job.parameters["utility"] != dataset.utility
        or job.parameters["budget"] != budget
    ):
        raise ValueError("completed estimator job conflicts with authority")
    if context.status.entry(job.job_id)["state"] != "success":
        raise ValueError("completed estimator authority is not successful")

    result_path = context.layout.root / job.output_relpath
    read_canonical_json(result_path, context="estimator ResultRecord")
    record = read_result_record(result_path, job)
    expected_score_relpath = (
        f"estimator/{job.dataset_id}/s{job.seed}/"
        f"{job.method}/b{budget}/scores.json"
    )
    if (
        set(record.artifacts) != {"scores"}
        or record.artifacts["scores"] != expected_score_relpath
    ):
        raise ValueError(
            "completed estimator ResultRecord score artifact is invalid"
        )
    scores = _read_score_artifact(
        context.layout.root / expected_score_relpath,
        expected_job=job,
        budget=budget,
        expected_features=features,
        expected_metrics=record.metrics,
        expected_execution_mode="grouped_nested_v1",
    )

    estimator_context = StageContext(
        context.protocol, context.layout, job, context.status, context.audit
    )
    request = group_request_from_context(
        estimator_context, _task9_manifest_row(context), features
    )
    if __package__:
        from .icomp2026_estimator_bundle import (
            bundle_from_document,
            bundle_relpath,
        )
    else:
        from icomp2026_estimator_bundle import (  # type: ignore[no-redef]
            bundle_from_document,
            bundle_relpath,
        )

    bundle = bundle_from_document(
        read_canonical_json(
            context.layout.root / bundle_relpath(request),
            context="estimator bundle",
        ),
        request,
    )
    checkpoints = tuple(
        checkpoint
        for checkpoint in bundle.checkpoints
        if checkpoint.method == method and checkpoint.budget == budget
    )
    if len(checkpoints) != 1:
        raise ValueError(
            "completed estimator bundle checkpoint is missing or duplicated"
        )
    checkpoint = checkpoints[0]
    checkpoint_scores = dict(checkpoint.scores)
    expected_metrics: dict[str, object] = {
        "budget": checkpoint.budget,
        "model_fit_count": checkpoint.logical_unique_coalition_count,
        "coalition_count": checkpoint.logical_unique_coalition_count + 1,
        "logical_evaluation_count": checkpoint.logical_evaluation_count,
        "logical_unique_coalition_count": (
            checkpoint.logical_unique_coalition_count
        ),
        "physical_model_fit_count": checkpoint.physical_model_fit_count,
        "cache_hit_count": checkpoint.cache_hit_count,
        "runtime_s": checkpoint.component_runtime_s,
        "group_request_digest": request.digest,
        "group_physical_model_fit_count": bundle.physical_model_fit_count,
        "group_cache_hit_count": bundle.cache_hit_count,
        "group_runtime_s": bundle.group_runtime_s,
    }
    if checkpoint.schedule_digest is not None:
        expected_metrics["schedule_digest"] = checkpoint.schedule_digest
    if (
        tuple(checkpoint_scores) != features
        or checkpoint_scores != scores
        or not set(expected_metrics) <= set(record.metrics)
        or any(
            record.metrics[key] != value
            for key, value in expected_metrics.items()
        )
        or (
            checkpoint.schedule_digest is None
            and "schedule_digest" in record.metrics
        )
    ):
        raise ValueError(
            "completed estimator ResultRecord, score artifact, and bundle "
            "checkpoint conflict"
        )
    return checkpoint


def _read_score_artifact(
    path: Path,
    *,
    expected_job: JobSpec,
    budget: int,
    expected_features: tuple[str, ...],
    expected_metrics: Mapping[str, object] | None = None,
    expected_execution_mode: str | None = None,
) -> dict[str, int]:
    try:
        document = read_canonical_json(path, context="estimator score artifact")
    except ValueError as exc:
        raise ValueError(
            "estimator score artifact is missing, malformed, or noncanonical JSON"
        ) from exc
    base_keys = {
        "protocol_hash", "job_id", "dataset_id", "seed", "method", "budget",
        "scores", "model_fit_count", "coalition_count",
    }
    grouped_fields = {
        "logical_evaluation_count",
        "logical_unique_coalition_count",
        "physical_model_fit_count",
        "cache_hit_count",
        "group_request_digest",
        "group_physical_model_fit_count",
        "group_cache_hit_count",
        "group_runtime_s",
    }
    expected_grouped_fields = (
        set()
        if expected_metrics is None
        else grouped_fields & set(expected_metrics)
    )
    if expected_grouped_fields and expected_grouped_fields != grouped_fields:
        raise ValueError(
            "estimator ResultRecord grouped metric schema is incomplete"
        )
    metrics_grouped = expected_grouped_fields == grouped_fields
    if expected_execution_mode not in {
        None, "legacy", "grouped_nested_v1"
    }:
        raise ValueError("estimator expected execution mode is invalid")
    if expected_execution_mode is None:
        grouped = metrics_grouped
    else:
        grouped = expected_execution_mode == "grouped_nested_v1"
        if expected_metrics is not None and grouped != metrics_grouped:
            raise ValueError(
                "estimator ResultRecord metrics conflict with execution mode"
            )
    expected_keys = base_keys | (grouped_fields if grouped else set())
    legacy_scheduled = expected_job.method in {
        "mi_correlation_guided", "hybrid"
    }
    grouped_schedule_required = expected_job.method in {
        "random_mc", "mi_correlation_guided"
    } or (
        expected_job.method == "hybrid"
        and len(expected_features) > 20
    )
    if (
        (grouped and grouped_schedule_required)
        or (not grouped and legacy_scheduled)
    ):
        expected_keys.add("schedule_digest")
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise ValueError("estimator score artifact schema is invalid")
    for field in ("protocol_hash", "job_id", "dataset_id", "seed", "method"):
        if document[field] != getattr(expected_job, field):
            raise ValueError(f"estimator score artifact {field} is invalid")
    if document["budget"] != budget or not isinstance(document["scores"], list):
        raise ValueError("estimator score artifact budget or scores are invalid")
    normalized: dict[str, int] = {}
    for item in document["scores"]:
        score = item.get("score") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or set(item) != {"feature", "score"}
            or not isinstance(item["feature"], str)
            or isinstance(score, bool)
            or not isinstance(score, int)
            or score not in {-1, 0, 1}
        ):
            raise ValueError("estimator score artifact scores are invalid")
        feature = item["feature"]
        if feature in normalized:
            raise ValueError("estimator score artifact features are duplicated")
        normalized[feature] = score
    if (
        len(document["scores"]) != len(expected_features)
        or tuple(normalized) != expected_features
    ):
        raise ValueError("estimator score artifact feature order is invalid")
    model_fit_count = document["model_fit_count"]
    coalition_count = document["coalition_count"]
    if (
        isinstance(model_fit_count, bool)
        or not isinstance(model_fit_count, int)
        or model_fit_count < 1
        or isinstance(coalition_count, bool)
        or not isinstance(coalition_count, int)
        or coalition_count < model_fit_count
        or coalition_count > model_fit_count + 1
        or (
            expected_job.method == "exact"
            and coalition_count != model_fit_count + 1
        )
    ):
        raise ValueError("estimator score artifact cost counters are invalid")
    if grouped:
        logical_evaluations = document["logical_evaluation_count"]
        logical_unique = document["logical_unique_coalition_count"]
        physical_fits = document["physical_model_fit_count"]
        cache_hits = document["cache_hit_count"]
        group_fits = document["group_physical_model_fit_count"]
        group_hits = document["group_cache_hit_count"]
        group_runtime = document["group_runtime_s"]
        request_digest = document["group_request_digest"]
        exact_nonnegative_ints = (
            logical_evaluations,
            logical_unique,
            physical_fits,
            cache_hits,
            group_fits,
            group_hits,
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in exact_nonnegative_ints
            )
            or logical_evaluations < 1
            or logical_unique < 1
            or logical_unique > logical_evaluations
            or model_fit_count != logical_unique
            or coalition_count != logical_unique + 1
            or physical_fits > logical_unique
            or physical_fits > group_fits
            or cache_hits > group_hits
            or isinstance(group_runtime, bool)
            or not isinstance(group_runtime, (int, float))
            or not __import__("math").isfinite(float(group_runtime))
            or group_runtime < 0
            or not isinstance(request_digest, str)
            or len(request_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in request_digest
            )
        ):
            raise ValueError("grouped estimator score counters are invalid")
        if "schedule_digest" in document:
            digest = document["schedule_digest"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise ValueError(
                    "grouped estimator score schedule digest is invalid"
                )
            if (
                expected_metrics is None
                or expected_metrics.get("schedule_digest") != digest
            ):
                raise ValueError(
                    "grouped estimator score schedule digest conflicts "
                    "with ResultRecord authority"
                )
        elif (
            expected_metrics is not None
            and "schedule_digest" in expected_metrics
        ):
            raise ValueError(
                "grouped estimator ResultRecord schedule digest conflicts "
                "with score artifact"
            )
    elif legacy_scheduled:
        from minppi_mofs.min_ppi import (
            algorithm_b_schedule_digest,
            make_algorithm_b_perturbation_schedule,
        )

        expected_digest = algorithm_b_schedule_digest(
            make_algorithm_b_perturbation_schedule(
                expected_features, K_adv=budget, seed=expected_job.seed
            )
        )
        if document["schedule_digest"] != expected_digest:
            raise ValueError("estimator score artifact schedule digest is invalid")
    if expected_metrics is not None:
        metric_fields = {"budget", "model_fit_count", "coalition_count"}
        if grouped:
            metric_fields.update(grouped_fields)
        if any(
            expected_metrics.get(field) != document[field]
            for field in metric_fields
        ):
            raise ValueError(
                "estimator ResultRecord metrics conflict with score artifact"
            )
        runtime = expected_metrics.get("runtime_s")
        if (
            isinstance(runtime, bool)
            or not isinstance(runtime, (int, float))
            or not __import__("math").isfinite(float(runtime))
            or runtime < 0
        ):
            raise ValueError("estimator ResultRecord runtime is invalid")
    return normalized


def _agreement_metrics(
    predicted: Mapping[str, int], truth: Mapping[str, int]
) -> dict[str, float]:
    if tuple(predicted) != tuple(truth):
        raise ValueError("estimator scores cannot be compared across different features")
    predicted_harmful = {feature for feature, score in predicted.items() if score == -1}
    true_harmful = {feature for feature, score in truth.items() if score == -1}
    overlap = len(predicted_harmful & true_harmful)
    if not predicted_harmful:
        precision = 1.0 if not true_harmful else 0.0
    else:
        precision = overlap / len(predicted_harmful)
    recall = 1.0 if not true_harmful else overlap / len(true_harmful)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    agreement = sum(
        predicted[feature] == truth[feature] for feature in predicted
    ) / len(predicted)
    return {
        "harmful_precision": precision,
        "harmful_recall": recall,
        "harmful_f1": f1,
        "score_agreement": agreement,
    }


def run_ablation_job(context: StageContext) -> ResultRecord:
    """Run one frozen ablation value from the manifest."""
    if context.job.stage != "ablation":
        raise ValueError("ablation executor requires an ablation manifest job")
    if context.job not in context.status.manifest:
        raise ValueError("ablation job is outside the frozen status manifest")
    if set(context.job.parameters) != {"task", "utility", "axis", "value"}:
        raise ValueError("ablation job parameters have an invalid schema")
    if context.job.method != context.job.parameters["axis"]:
        raise ValueError("ablation axis does not match manifest method")
    if context.job.parameters["axis"] == "utility":
        return _run_utility_ablation(context)
    if context.job.parameters["axis"] == "guidance":
        return _run_guidance_ablation(context)
    if context.job.parameters["axis"] == "dispatcher":
        return _run_dispatcher_ablation(context)
    raise ValueError("ablation axis is not frozen in the protocol")


def _run_utility_ablation(context: StageContext) -> ResultRecord:
    from minppi_mofs.coalition_cache import prediction_digest
    from minppi_mofs.min_ppi import (
        estimate_minppi_random_mc,
        make_shared_coalition_plan,
    )
    from minppi_mofs.voting_game import AccFullCache, UtilityView

    started_at = _timestamp()
    if context.job.dataset_id != "isolet":
        raise ValueError("utility ablation is restricted to ISOLET")
    value = context.job.parameters["value"]
    if value not in {"accuracy", "ovr_auc"}:
        raise ValueError("utility ablation value is not frozen")
    budget = max(context.protocol.estimator_budgets)
    group_id = f"utility:{context.job.dataset_id}:s{context.job.seed}:b{budget}"
    relpath = (
        f"ablation/{context.job.dataset_id}/s{context.job.seed}/"
        f"utility/paired/b{budget}/scores.json"
    )
    path = context.layout.root / relpath
    if path.exists():
        document = _read_utility_pair_artifact(context, path, budget, group_id)
    else:
        X_train, y_train, X_val, y_val, _ = _load_prepared_train_validation(context)
        plan = make_shared_coalition_plan(
            tuple(X_train.columns), K=budget, per_coalition=5, seed=context.job.seed
        )
        model_params = dict(context.protocol.model_params)
        model_params["tree_method"] = "hist"
        model_params["device"] = (
            "cuda" if context.protocol.backend == "cuda" else "cpu"
        )
        cache = AccFullCache(
            X_train,
            X_val,
            y_train,
            y_val,
            task="multiclass",
            utility="accuracy",
            seed=context.job.seed,
            model_params=model_params,
        )
        views = {
            metric: UtilityView(
                cache, utility=metric, alpha=context.protocol.alpha_default
            )
            for metric in ("accuracy", "ovr_auc")
        }
        scores: dict[str, list[dict[str, object]]] = {}
        primary = estimate_minppi_random_mc(
            X_train,
            X_val,
            y_train,
            y_val,
            alpha=context.protocol.alpha_default,
            task="multiclass",
            K=budget,
            shared_coalitions=True,
            shared_plan=plan,
            utility="accuracy",
            acc_cache=views["accuracy"],
            seed=context.job.seed,
        )
        accuracy_records = [
            {
                "coalition": list(coalition),
                "prediction_sha256": prediction_digest(predictions),
            }
            for coalition, predictions in sorted(cache._preds.items())
        ]
        before_secondary = (cache.model_fit_count, cache.coalition_count)
        secondary = estimate_minppi_random_mc(
            X_train,
            X_val,
            y_train,
            y_val,
            alpha=context.protocol.alpha_default,
            task="multiclass",
            K=budget,
            shared_coalitions=True,
            shared_plan=plan,
            utility="ovr_auc",
            acc_cache=views["ovr_auc"],
            seed=context.job.seed,
        )
        ovr_auc_records = [
            {
                "coalition": list(coalition),
                "prediction_sha256": prediction_digest(predictions),
            }
            for coalition, predictions in sorted(cache._preds.items())
        ]
        if ovr_auc_records != accuracy_records:
            raise RuntimeError(
                "utility ablation metrics did not share prediction records"
            )
        records_payload = json.dumps(
            accuracy_records,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        shared_prediction_digest = hashlib.sha256(
            records_payload
        ).hexdigest()
        scores["accuracy"] = [
            {"feature": feature, "score": int(primary[feature])}
            for feature in X_train.columns
        ]
        scores["ovr_auc"] = [
            {"feature": feature, "score": int(secondary[feature])}
            for feature in X_train.columns
        ]
        document = {
            "protocol_hash": context.job.protocol_hash,
            "dataset_id": context.job.dataset_id,
            "seed": context.job.seed,
            "axis": "utility",
            "budget": budget,
            "shared_group_id": group_id,
            "scores": scores,
            "plan_step_count": len(plan),
            "shared_group_model_fit_count": cache.model_fit_count,
            "shared_group_coalition_count": cache.coalition_count,
            "secondary_model_fit_delta": cache.model_fit_count - before_secondary[0],
            "secondary_coalition_delta": cache.coalition_count - before_secondary[1],
            "prediction_records": accuracy_records,
            "prediction_digests": {
                "accuracy": shared_prediction_digest,
                "ovr_auc": shared_prediction_digest,
            },
        }
        _publish_idempotent_json(path, document, context="utility paired artifact")
        document = _read_utility_pair_artifact(context, path, budget, group_id)
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics={
            "axis": "utility",
            "value": str(value),
            "budget": budget,
            "shared_group_id": group_id,
            "shared_group_model_fit_count": int(
                document["shared_group_model_fit_count"]
            ),
            "shared_group_coalition_count": int(
                document["shared_group_coalition_count"]
            ),
        },
        artifacts={"paired_scores": relpath},
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _read_utility_pair_artifact(
    context: StageContext, path: Path, budget: int, group_id: str
) -> dict[str, object]:
    try:
        document = read_canonical_json(path, context="utility paired artifact")
    except ValueError as exc:
        raise ValueError(
            "utility paired artifact is missing, malformed, or noncanonical JSON"
        ) from exc
    expected_keys = {
        "protocol_hash", "dataset_id", "seed", "axis", "budget",
        "shared_group_id", "scores", "plan_step_count",
        "shared_group_model_fit_count", "shared_group_coalition_count",
        "secondary_model_fit_delta", "secondary_coalition_delta",
        "prediction_records", "prediction_digests",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise ValueError("utility paired artifact schema is invalid")
    identity = {
        "protocol_hash": context.job.protocol_hash,
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "axis": "utility",
        "budget": budget,
        "shared_group_id": group_id,
    }
    if any(document[field] != expected for field, expected in identity.items()):
        raise ValueError("utility paired artifact identity conflicts with this job")
    scores = document["scores"]
    if not isinstance(scores, dict) or set(scores) != {"accuracy", "ovr_auc"}:
        raise ValueError("utility paired artifact scores are invalid")
    feature_orders: list[tuple[str, ...]] = []
    for metric in ("accuracy", "ovr_auc"):
        entries = scores[metric]
        if not isinstance(entries, list):
            raise ValueError("utility paired artifact scores are invalid")
        features: list[str] = []
        for entry in entries:
            score = entry.get("score") if isinstance(entry, dict) else None
            if (
                not isinstance(entry, dict)
                or set(entry) != {"feature", "score"}
                or not isinstance(entry["feature"], str)
                or isinstance(score, bool)
                or not isinstance(score, int)
                or score not in {-1, 0, 1}
            ):
                raise ValueError("utility paired artifact scores are invalid")
            features.append(entry["feature"])
        if len(features) != len(set(features)):
            raise ValueError("utility paired artifact features are duplicated")
        feature_orders.append(tuple(features))
    if feature_orders[0] != feature_orders[1]:
        raise ValueError("utility paired artifact feature order differs by metric")
    manifest_rows = read_rich_data_manifest(
        context.layout.root / "data_manifest.csv", context.protocol
    )
    manifest_row = next(
        (
            row
            for row in manifest_rows
            if row["dataset_id"] == context.job.dataset_id
            and row["seed"] == context.job.seed
        ),
        None,
    )
    if manifest_row is None:
        raise ValueError("utility paired artifact has no rich manifest authority")
    expected_features = tuple(
        json.loads(str(manifest_row["selected_features_json"]))
    )
    if feature_orders[0] != expected_features:
        raise ValueError("utility paired artifact feature order conflicts with manifest")
    for field in (
        "plan_step_count", "shared_group_model_fit_count",
        "shared_group_coalition_count", "secondary_model_fit_delta",
        "secondary_coalition_delta",
    ):
        if (
            isinstance(document[field], bool)
            or not isinstance(document[field], int)
            or document[field] < 0
        ):
            raise ValueError("utility paired artifact costs are invalid")
    if document["secondary_model_fit_delta"] != 0 or document[
        "secondary_coalition_delta"
    ] != 0:
        raise ValueError("utility paired artifact did not reuse the shared cache")
    prediction_records = document["prediction_records"]
    prediction_digests = document["prediction_digests"]
    if (
        not isinstance(prediction_records, list)
        or len(prediction_records) != document["shared_group_coalition_count"]
        or not isinstance(prediction_digests, dict)
        or set(prediction_digests) != {"accuracy", "ovr_auc"}
    ):
        raise ValueError(
            "utility paired prediction coalition cost evidence is invalid"
        )
    seen_coalitions: set[tuple[str, ...]] = set()
    for record in prediction_records:
        if (
            not isinstance(record, dict)
            or set(record) != {"coalition", "prediction_sha256"}
            or not isinstance(record["coalition"], list)
            or any(
                not isinstance(feature, str)
                for feature in record["coalition"]
            )
            or tuple(record["coalition"]) in seen_coalitions
            or not isinstance(record["prediction_sha256"], str)
            or len(record["prediction_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record["prediction_sha256"]
            )
        ):
            raise ValueError("utility paired prediction evidence is invalid")
        seen_coalitions.add(tuple(record["coalition"]))
    expected_prediction_digest = hashlib.sha256(
        json.dumps(
            prediction_records,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        prediction_digests["accuracy"] != expected_prediction_digest
        or prediction_digests["ovr_auc"] != expected_prediction_digest
    ):
        raise ValueError("utility paired prediction digests are invalid")

    from minppi_mofs.min_ppi import make_shared_coalition_plan

    plan = make_shared_coalition_plan(
        expected_features, K=budget, per_coalition=5, seed=context.job.seed
    )
    coalition_keys = {frozenset(expected_features)}
    for coalition, evaluated in plan:
        coalition_keys.add(coalition)
        coalition_keys.update(coalition | {feature} for feature in evaluated)
    expected_costs = {
        "plan_step_count": len(plan),
        "shared_group_model_fit_count": sum(bool(key) for key in coalition_keys),
        "shared_group_coalition_count": len(coalition_keys),
    }
    if any(document[field] != expected for field, expected in expected_costs.items()):
        raise ValueError("utility paired artifact plan or shared-cache costs are invalid")
    return document


def _run_guidance_ablation(context: StageContext) -> ResultRecord:
    from minppi_mofs.nested_estimation import estimate_guided_checkpoints
    from minppi_mofs.shapg_integration import (
        compute_mi_guidance_signal,
        get_correlation_neighbors,
    )

    started_at = _timestamp()
    value = context.job.parameters["value"]
    allowed = {"random", "mi", "correlation", "mi_correlation"}
    if value not in allowed:
        raise ValueError("guidance ablation value is not frozen")
    budget = max(context.protocol.estimator_budgets)
    manifest_row = _task9_manifest_row(context)
    features = tuple(
        json.loads(str(manifest_row["selected_features_json"]))
    )
    checkpoint, _request, schedules = _ablation_estimator_authority(
        context, "mi_correlation_guided", features
    )
    if value == "mi_correlation":
        return _publish_single_ablation(
            context,
            started_at=started_at,
            value=str(value),
            estimator_method="mi_correlation_guided",
            budget=budget,
            scores=dict(checkpoint.scores),
            model_fit_count=checkpoint.logical_unique_coalition_count,
            coalition_count=checkpoint.logical_unique_coalition_count + 1,
            logical_evaluation_count=checkpoint.logical_evaluation_count,
            logical_unique_coalition_count=(
                checkpoint.logical_unique_coalition_count
            ),
            physical_model_fit_count=0,
            cache_hit_count=0,
            schedule_digest=checkpoint.schedule_digest,
        )
    X_train, y_train, X_val, y_val, _ = _load_prepared_train_validation(
        context
    )
    if schedules.guided_schedule is None or schedules.guided_digest is None:
        raise RuntimeError("guidance ablation schedule authority is missing")
    task = str(context.job.parameters["task"])
    mi_scores = None
    correlation_neighbors = None
    if value == "mi":
        mi_scores = compute_mi_guidance_signal(
            X_train, y_train, task=task, seed=context.job.seed
        )
    elif value == "correlation":
        correlation_neighbors = get_correlation_neighbors(
            X_train, seed=context.job.seed
        )
    with _open_selection_cache(
        context,
        (X_train, y_train, X_val, y_val),
        manifest_row,
    ) as cache:
        before = cache.stats
        view = _AlphaCoalitionUtilityView(
            cache,
            utility=str(context.job.parameters["utility"]),
            alpha=context.protocol.alpha_default,
        )
        scores = estimate_guided_checkpoints(
            schedules.guided_schedule,
            view,
            mi_scores=mi_scores,
            correlation_neighbors=correlation_neighbors,
        )[budget]
        after = cache.stats
        logical_coalitions = view.logical_coalitions
    unique_coalitions = set(logical_coalitions)
    return _publish_single_ablation(
        context,
        started_at=started_at,
        value=str(value),
        estimator_method="mi_correlation_guided",
        budget=budget,
        scores=scores,
        model_fit_count=sum(bool(item) for item in unique_coalitions),
        coalition_count=len(unique_coalitions),
        logical_evaluation_count=len(logical_coalitions),
        logical_unique_coalition_count=len(unique_coalitions),
        physical_model_fit_count=(
            after.physical_model_fit_count
            - before.physical_model_fit_count
        ),
        cache_hit_count=after.cache_hit_count - before.cache_hit_count,
        schedule_digest=schedules.guided_digest,
    )


def _run_dispatcher_ablation(context: StageContext) -> ResultRecord:
    started_at = _timestamp()
    value = context.job.parameters["value"]
    allowed = {"exact", "random_mc", "mi_correlation_guided", "hybrid"}
    if value not in allowed:
        raise ValueError("dispatcher ablation value is not frozen")
    if value == "exact" and not any(
        job == context.job
        and job.stage == "ablation"
        and job.method == "dispatcher"
        and job.parameters.get("value") == "exact"
        for job in context.status.manifest
    ):
        raise ValueError("exact dispatcher job is absent from the manifest")
    budget = max(context.protocol.estimator_budgets)
    manifest_row = _task9_manifest_row(context)
    features = tuple(
        json.loads(str(manifest_row["selected_features_json"]))
    )
    checkpoint, _request, _schedules = _ablation_estimator_authority(
        context, str(value), features
    )
    return _publish_single_ablation(
        context,
        started_at=started_at,
        value=str(value),
        estimator_method=str(value),
        budget=budget,
        scores=dict(checkpoint.scores),
        model_fit_count=checkpoint.logical_unique_coalition_count,
        coalition_count=checkpoint.logical_unique_coalition_count + 1,
        logical_evaluation_count=checkpoint.logical_evaluation_count,
        logical_unique_coalition_count=(
            checkpoint.logical_unique_coalition_count
        ),
        physical_model_fit_count=0,
        cache_hit_count=0,
        schedule_digest=checkpoint.schedule_digest,
    )


def _ablation_estimator_authority(
    context: StageContext,
    method: str,
    features: tuple[str, ...],
):
    """Return a completed checkpoint plus recomputed schedule authority."""
    budget = max(context.protocol.estimator_budgets)
    checkpoint = _completed_estimator_checkpoint(
        context, method, budget, features
    )
    estimator_jobs = tuple(
        job
        for job in context.status.manifest
        if job.stage == "estimator"
        and job.dataset_id == context.job.dataset_id
        and job.seed == context.job.seed
        and job.method == method
        and job.parameters.get("budget") == budget
    )
    if len(estimator_jobs) != 1:
        raise ValueError(
            "ablation estimator group authority is missing or duplicated"
        )
    estimator_context = StageContext(
        context.protocol,
        context.layout,
        estimator_jobs[0],
        context.status,
        context.audit,
    )
    request = group_request_from_context(
        estimator_context, _task9_manifest_row(context), features
    )
    if __package__:
        from .icomp2026_estimator_bundle import (
            _hybrid_schedule_digest,
            _schedule_authority,
        )
    else:
        from icomp2026_estimator_bundle import (  # type: ignore[no-redef]
            _hybrid_schedule_digest,
            _schedule_authority,
        )

    schedules = _schedule_authority(request)
    expected_digest = {
        "exact": None,
        "random_mc": schedules.random_digest,
        "mi_correlation_guided": schedules.guided_digest,
        "hybrid": _hybrid_schedule_digest(
            feature_count=len(features),
            random_digest=schedules.random_digest,
            guided_digest=schedules.guided_digest,
        ),
    }[method]
    if checkpoint.schedule_digest != expected_digest:
        raise ValueError(
            "ablation checkpoint schedule digest conflicts with "
            "deterministic schedule authority"
        )
    return checkpoint, request, schedules


def _make_job_cache(context: StageContext, X_train, X_val, y_train, y_val):
    from minppi_mofs.voting_game import AccFullCache

    model_params = dict(context.protocol.model_params)
    model_params["tree_method"] = "hist"
    model_params["device"] = "cuda" if context.protocol.backend == "cuda" else "cpu"
    return AccFullCache(
        X_train,
        X_val,
        y_train,
        y_val,
        task=str(context.job.parameters["task"]),
        utility=str(context.job.parameters["utility"]),
        seed=context.job.seed,
        model_params=model_params,
    )


def _publish_single_ablation(
    context: StageContext,
    *,
    started_at: str,
    value: str,
    estimator_method: str,
    budget: int,
    scores: Mapping[str, int],
    model_fit_count: int,
    coalition_count: int,
    logical_evaluation_count: int,
    logical_unique_coalition_count: int,
    physical_model_fit_count: int,
    cache_hit_count: int,
    schedule_digest: str | None = None,
) -> ResultRecord:
    axis = str(context.job.parameters["axis"])
    relpath = (
        f"ablation/{context.job.dataset_id}/s{context.job.seed}/"
        f"{axis}/{value}/b{budget}/scores.json"
    )
    document = {
        "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id,
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "axis": axis,
        "value": value,
        "method": estimator_method,
        "budget": budget,
        "schedule_seed": context.job.seed,
        "scores": [
            {"feature": feature, "score": int(score)}
            for feature, score in scores.items()
        ],
        "model_fit_count": model_fit_count,
        "coalition_count": coalition_count,
        "logical_evaluation_count": logical_evaluation_count,
        "logical_unique_coalition_count": logical_unique_coalition_count,
        "physical_model_fit_count": physical_model_fit_count,
        "cache_hit_count": cache_hit_count,
    }
    if schedule_digest is not None:
        document["schedule_digest"] = schedule_digest
    path = context.layout.root / relpath
    if path.exists():
        existing = read_canonical_json(
            path, context=f"{axis} ablation artifact"
        )
        if (
            not isinstance(existing, Mapping)
            or existing.get("schedule_digest") != schedule_digest
        ):
            raise ValueError(
                f"{axis} ablation schedule digest is invalid"
            )
    _publish_idempotent_json(
        path,
        document,
        context=f"{axis} ablation artifact",
    )
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics={
            "axis": axis,
            "value": value,
            "budget": budget,
            "model_fit_count": model_fit_count,
            "coalition_count": coalition_count,
            "logical_evaluation_count": logical_evaluation_count,
            "logical_unique_coalition_count": logical_unique_coalition_count,
            "physical_model_fit_count": physical_model_fit_count,
            "cache_hit_count": cache_hit_count,
        },
        artifacts={"scores": relpath},
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _publish_idempotent_json(
    path: Path, document: Mapping[str, object], *, context: str
) -> None:
    """Publish JSON atomically; identical retries reuse and conflicts fail closed."""
    publish_json_exclusive(path, document, context=context)


def _coerce_target(series: object, task: str):
    import numpy as np
    import pandas as pd

    if not isinstance(series, pd.Series) or series.isna().any():
        raise ValueError("raw CSV target must not contain missing values")
    if task == "regression":
        try:
            values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("regression target must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError("regression target must be finite")
        return values, {"kind": "regression_numeric"}
    tokens = [_canonical_json_text(_json_safe_index(value)) for value in series]
    classes = sorted(set(tokens))
    expected = 2 if task == "binary" else None
    if len(classes) < 2 or (expected is not None and len(classes) != expected):
        raise ValueError(f"{task} target has an invalid number of classes")
    mapping = {token: index for index, token in enumerate(classes)}
    return np.asarray([mapping[token] for token in tokens], dtype=int), {
        "kind": "classification_codes",
        "classes": classes,
    }


def _assert_frozen_raw_authority(
    context: StageContext,
    dataset_id: str,
    sha256: str,
    byte_count: int,
) -> None:
    authority = _read_raw_authority(
        context, context.layout.root / "validate" / "raw_authority.json"
    )
    frozen = authority[dataset_id]
    if frozen != {"byte_count": byte_count, "sha256": sha256}:
        raise RuntimeError(f"raw authority drift detected for {dataset_id}")


def _freeze_raw_authority(
    context: StageContext,
    path: Path,
    candidate: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    document = {
        "protocol_hash": context.job.protocol_hash,
        "datasets": {dataset_id: dict(entry) for dataset_id, entry in candidate.items()},
    }
    if not path.exists():
        atomic_write_json(path, document)
    frozen = _read_raw_authority(context, path)
    if frozen != document["datasets"]:
        raise RuntimeError("frozen raw authority conflicts with current raw datasets")
    return frozen


def _read_raw_authority(
    context: StageContext, path: Path
) -> dict[str, dict[str, object]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {
            "protocol_hash", "datasets",
        }:
            raise ValueError("raw authority document is invalid")
        if document["protocol_hash"] != context.job.protocol_hash:
            raise ValueError("raw authority protocol hash is invalid")
        datasets = document["datasets"]
        expected_ids = {dataset.dataset_id for dataset in context.protocol.datasets}
        if not isinstance(datasets, dict) or set(datasets) != expected_ids:
            raise ValueError("raw authority scope is invalid")
        normalized: dict[str, dict[str, object]] = {}
        for dataset_id, entry in datasets.items():
            if not isinstance(entry, dict) or set(entry) != {"byte_count", "sha256"}:
                raise ValueError("raw authority entry is invalid")
            byte_count = entry["byte_count"]
            sha256 = entry["sha256"]
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise ValueError("raw authority entry is invalid")
            normalized[dataset_id] = {
                "byte_count": byte_count,
                "sha256": sha256,
            }
        return normalized
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise RuntimeError("frozen raw authority is missing or invalid") from exc


def _label_summary(values: object, task: str) -> dict[str, object]:
    import numpy as np

    array = np.asarray(values)
    if task == "regression":
        return {
            "count": int(array.size),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "min": float(array.min()),
            "std": float(array.std(ddof=0)),
        }
    labels, counts = np.unique(array, return_counts=True)
    return {
        "classes": [
            {"count": int(count), "label": int(label)}
            for label, count in zip(labels, counts, strict=True)
        ],
        "count": int(array.size),
    }


def _json_safe_index(value: object) -> object:
    try:
        value = value.item()  # type: ignore[union-attr]
    except AttributeError:
        pass
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not __import__("math").isfinite(value):
            raise ValueError("split row indices must be finite")
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[union-attr]
    raise ValueError("split row indices must be JSON-safe")


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json_text(value).encode("utf-8")


def _artifact_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _publish_preprocess_snapshot(
    context: StageContext,
    *,
    action: str,
    manifest_path: Path,
    manifest_row: Mapping[str, object],
    partition_bytes: Mapping[str, bytes],
    metadata_bytes: bytes,
) -> None:
    parent = (
        context.layout.root
        / "preprocess"
        / context.job.dataset_id
    )
    final = parent / f"s{context.job.seed}"
    staging = parent / f".s{context.job.seed}.staging"
    payloads = {
        "train.csv": partition_bytes["train"],
        "val.csv": partition_bytes["val"],
        "test.csv": partition_bytes["test"],
        "split_metadata.json": metadata_bytes,
    }
    if action == "identical":
        if staging.exists():
            raise RuntimeError("orphan preprocess staging snapshot exists")
        _verify_preprocess_snapshot(final, payloads)
        return
    if action != "insert":
        raise ValueError("data manifest preflight action is invalid")
    if final.exists() or staging.exists():
        raise RuntimeError("orphan preprocess snapshot exists without manifest authority")
    parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    published = False
    try:
        for name, payload in payloads.items():
            _atomic_write_bytes(staging / name, payload)
        _verify_staged_snapshot_shape(staging, payloads)
        os.replace(staging, final)
        published = True
        try:
            upsert_data_manifest(manifest_path, context.protocol, manifest_row)
        except Exception:
            shutil.rmtree(final)
            published = False
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if published and not manifest_path.exists():
            shutil.rmtree(final)


def _verify_preprocess_snapshot(
    directory: Path, payloads: Mapping[str, bytes]
) -> None:
    if not directory.is_dir():
        raise RuntimeError("frozen preprocess snapshot is missing")
    entries = tuple(directory.iterdir())
    if (
        {item.name for item in entries} != set(payloads)
        or not all(item.is_file() and not item.is_symlink() for item in entries)
        or any(item.stat().st_size != len(payloads[item.name]) for item in entries)
    ):
        raise RuntimeError("frozen preprocess snapshot has invalid artifacts")
    for name in ("train.csv", "val.csv", "split_metadata.json"):
        expected = payloads[name]
        if (directory / name).read_bytes() != expected:
            raise RuntimeError("frozen preprocess snapshot does not match manifest row")


def _verify_staged_snapshot_shape(
    directory: Path, payloads: Mapping[str, bytes]
) -> None:
    entries = tuple(directory.iterdir())
    if (
        {item.name for item in entries} != set(payloads)
        or not all(item.is_file() for item in entries)
        or any(item.stat().st_size != len(payloads[item.name]) for item in entries)
    ):
        raise RuntimeError("staged preprocess snapshot is incomplete")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cuda_visibility(backend: str) -> str:
    if backend != "cuda":
        return "not_requested"
    import subprocess

    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True,
            check=False, timeout=5, encoding="utf-8", errors="replace",
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("CUDA backend requested but nvidia-smi is unavailable") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError("CUDA backend requested but no visible GPU was reported")
    return completed.stdout.strip()


def _validate_xgboost_build_info(value: object, backend: str) -> None:
    if not isinstance(value, dict) or not value or _contains_unavailable(value):
        raise ValueError("frozen XGBoost build info is empty or unavailable")
    if backend == "cuda" and not _xgboost_build_supports_cuda(value):
        raise ValueError("frozen XGBoost build lacks explicit CUDA capability")


def _contains_unavailable(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() == "unavailable"
    if isinstance(value, Mapping):
        return any(
            str(key).strip().casefold() == "unavailable"
            or _contains_unavailable(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_unavailable(item) for item in value)
    return False


def _xgboost_build_supports_cuda(build_info: Mapping[object, object]) -> bool:
    for key, value in build_info.items():
        normalized = "".join(character for character in str(key).upper() if character.isalnum())
        if "CUDA" not in normalized:
            continue
        if any(marker in normalized for marker in ("USE", "ENABLE", "BUILT", "HAVE")):
            if _build_flag_enabled(value):
                return True
        elif "VERSION" in normalized and _build_flag_enabled(value):
            return True
    return False


def _build_flag_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value > 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        return normalized not in {"", "0", "false", "no", "off", "none", "unavailable"}
    if isinstance(value, (list, tuple)):
        return bool(value) and any(_build_flag_enabled(item) for item in value)
    return False


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


_REMOVAL_METHODS = (
    "full",
    "minppi",
    "shap_bottom",
    "minshap_bottom",
    "banzhaf_bottom",
    "random_matched",
)


def _task9_authority(context: StageContext, *, stage: str) -> object:
    if context.job.stage != stage:
        raise ValueError(f"{stage} executor requires a {stage} manifest job")
    if context.job not in context.status.manifest:
        raise ValueError(f"{stage} job is outside the frozen status manifest")
    dataset = next(
        (
            item
            for item in context.protocol.datasets
            if item.dataset_id == context.job.dataset_id
        ),
        None,
    )
    if dataset is None:
        raise ValueError(f"{stage} job has an unknown dataset")
    if context.job.protocol_hash != resolved_protocol_hash(context.protocol):
        raise ValueError(f"{stage} job protocol hash conflicts with authority")
    return dataset


def _task9_manifest_row(context: StageContext) -> dict[str, object]:
    """Return rich prepared-data authority without touching split files."""
    rows = read_rich_data_manifest(
        context.layout.root / "data_manifest.csv", context.protocol
    )
    matches = tuple(
        row
        for row in rows
        if row["dataset_id"] == context.job.dataset_id
        and row["seed"] == context.job.seed
    )
    if len(matches) != 1:
        raise ValueError("Task 9 rich manifest authority is missing or duplicated")
    row = matches[0]
    dataset = next(
        item
        for item in context.protocol.datasets
        if item.dataset_id == context.job.dataset_id
    )
    if row["task"] != dataset.task:
        raise ValueError("Task 9 rich manifest task conflicts with dataset authority")
    return row


def _task9_feature_hash(features: Sequence[str]) -> str:
    payload = json.dumps(
        list(features), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task9_subset_row(
    features: tuple[str, ...], removed: Sequence[str], *, trial: int | None = None,
    rng_seed: int | None = None,
) -> dict[str, object]:
    removed_set = set(removed)
    ordered_removed = [feature for feature in features if feature in removed_set]
    retained = [feature for feature in features if feature not in removed_set]
    return {
        "trial": trial,
        "rng_seed": rng_seed,
        "retained_features": retained,
        "removed_features": ordered_removed,
        "retained_hash": _task9_feature_hash(retained),
    }


def _task9_validate_subset(
    value: object,
    *,
    features: tuple[str, ...],
    expected_removed_count: int,
    random_trial: bool,
) -> dict[str, object]:
    expected = {
        "trial", "rng_seed", "retained_features", "removed_features", "retained_hash"
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("selection subset schema is invalid")
    retained = value["retained_features"]
    removed = value["removed_features"]
    if (
        not isinstance(retained, list)
        or not isinstance(removed, list)
        or len(set(retained)) != len(retained)
        or len(set(removed)) != len(removed)
        or any(not isinstance(item, str) for item in retained + removed)
    ):
        raise ValueError("selection subset features are invalid")
    if len(removed) != expected_removed_count:
        raise ValueError("selection subset removal count is invalid")
    removed_set = set(removed)
    if retained != [feature for feature in features if feature not in removed_set]:
        raise ValueError("selection retained order or complement is invalid")
    if removed != [feature for feature in features if feature in removed_set]:
        raise ValueError("selection removed order is invalid")
    if set(retained).union(removed) != set(features):
        raise ValueError("selection subset does not cover the frozen features")
    if value["retained_hash"] != _task9_feature_hash(retained):
        raise ValueError("selection retained feature hash is invalid")
    trial = value["trial"]
    rng_seed = value["rng_seed"]
    if random_trial:
        if (
            isinstance(trial, bool)
            or not isinstance(trial, int)
            or trial < 0
            or isinstance(rng_seed, bool)
            or not isinstance(rng_seed, int)
            or rng_seed < 0
        ):
            raise ValueError("random selection trial identity is invalid")
    elif trial is not None or rng_seed is not None:
        raise ValueError("non-random selection has random trial metadata")
    return value


def _task9_scores(scores: Mapping[str, object], features: tuple[str, ...]) -> list[dict[str, object]]:
    import math

    if tuple(scores) != features or set(scores) != set(features):
        raise ValueError("selection scores do not match frozen feature order")
    ordered = []
    for feature in features:
        score = scores[feature]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("selection scores must be finite numbers")
        ordered.append({"feature": feature, "score": float(score)})
    return ordered


def _task9_manifest_plan(context: StageContext, *, stage: str) -> JobSpec:
    matches = tuple(
        job
        for job in context.status.manifest
        if job.stage == stage
        and job.dataset_id == context.job.dataset_id
        and job.seed == context.job.seed
        and job.method == "selection_plan"
    )
    if len(matches) != 1:
        raise ValueError(f"{stage} manifest must contain exactly one selection plan")
    return matches[0]


def _validate_removal_scope(context: StageContext) -> None:
    if tuple(context.protocol.removal.get("controls", ())) != _REMOVAL_METHODS:
        raise ValueError("removal controls are not the canonical complete method set")
    jobs = tuple(
        job
        for job in context.status.manifest
        if job.stage == "removal"
        and job.dataset_id == context.job.dataset_id
        and job.seed == context.job.seed
    )
    if tuple(job.method for job in jobs) != ("selection_plan", *_REMOVAL_METHODS):
        raise ValueError("removal manifest scope is incomplete or out of order")
    plan = jobs[0]
    if any(job.depends_on != (plan.job_id,) for job in jobs[1:]):
        raise ValueError("removal dependencies do not reference the frozen plan")


def _removal_plan_relpath(context: StageContext) -> str:
    return f"removal/{context.job.dataset_id}/s{context.job.seed}/selection_plan.json"


def _validate_removal_plan(context: StageContext, document: object) -> dict[str, object]:
    import numpy as np

    expected_keys = {
        "schema_version", "protocol_hash", "job_id", "dataset_id", "seed",
        "task", "utility", "alpha", "budget", "features", "k", "methods",
        "scores", "selection_model_fit_count", "selection_coalition_count",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise ValueError("removal selection plan schema is invalid")
    dataset = _task9_authority(context, stage="removal")
    if (
        document["schema_version"] != 1
        or document["protocol_hash"] != context.job.protocol_hash
        or document["job_id"] != _task9_manifest_plan(context, stage="removal").job_id
        or document["dataset_id"] != context.job.dataset_id
        or document["seed"] != context.job.seed
        or document["task"] != dataset.task
        or document["utility"] != dataset.utility
        or document["alpha"] != context.protocol.alpha_default
        or document["budget"] != max(context.protocol.estimator_budgets)
    ):
        raise ValueError("removal selection plan identity is invalid")
    features_value = document["features"]
    if not isinstance(features_value, list) or len(set(features_value)) != len(features_value) or any(
        not isinstance(feature, str) for feature in features_value
    ):
        raise ValueError("removal plan features are invalid")
    features = tuple(features_value)
    manifest_row = _task9_manifest_row(context)
    manifest_features = tuple(
        json.loads(str(manifest_row["selected_features_json"]))
    )
    if features != manifest_features:
        raise ValueError("removal plan features conflict with rich manifest authority")
    k = document["k"]
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= len(features):
        raise ValueError("removal plan k is invalid")
    methods = document["methods"]
    if not isinstance(methods, list) or [item.get("method") if isinstance(item, dict) else None for item in methods] != list(_REMOVAL_METHODS):
        raise ValueError("removal plan methods are incomplete or out of order")
    repeats = context.protocol.removal["random_repeats"]
    minppi_removed: list[str] | None = None
    for item in methods:
        method = item.get("method") if isinstance(item, dict) else None
        expected_method_keys = {
            "method", "trials", "model_fit_count", "coalition_count"
        }
        if method == "minppi":
            expected_method_keys.update(
                {
                    "logical_unique_coalition_count",
                    "physical_model_fit_count",
                }
            )
        if (
            not isinstance(item, dict)
            or set(item) != expected_method_keys
            or not isinstance(item["trials"], list)
        ):
            raise ValueError("removal plan method schema is invalid")
        for field in ("model_fit_count", "coalition_count"):
            if (
                isinstance(item[field], bool)
                or not isinstance(item[field], int)
                or item[field] < 0
            ):
                raise ValueError("removal method cost counters are invalid")
        if method in {"full", "random_matched"} and (
            item["model_fit_count"] != 0 or item["coalition_count"] != 0
        ):
            raise ValueError("non-model removal method has selection costs")
        if method == "minppi" and (
            isinstance(item["logical_unique_coalition_count"], bool)
            or not isinstance(
                item["logical_unique_coalition_count"], int
            )
            or item["logical_unique_coalition_count"] < 1
            or item["coalition_count"]
            != item["logical_unique_coalition_count"]
            or item["model_fit_count"] != 0
            or item["physical_model_fit_count"] != 0
        ):
            raise ValueError(
                "reused MinPPI logical or physical counters are invalid"
            )
        expected_trials = repeats if method == "random_matched" else 1
        if len(item["trials"]) != expected_trials:
            raise ValueError("removal plan trial count is invalid")
        expected_removed = 0 if method == "full" else k
        for trial_index, row in enumerate(item["trials"]):
            checked = _task9_validate_subset(
                row, features=features, expected_removed_count=expected_removed,
                random_trial=method == "random_matched",
            )
            if method == "random_matched" and checked["trial"] != trial_index:
                raise ValueError("random removal trials are not complete and ordered")
        if method == "minppi":
            minppi_removed = item["trials"][0]["removed_features"]
    score_groups = document["scores"]
    if not isinstance(score_groups, list) or [item.get("method") if isinstance(item, dict) else None for item in score_groups] != ["minppi", "shap_bottom", "minshap_bottom", "banzhaf_bottom"]:
        raise ValueError("removal plan score groups are invalid")
    for group in score_groups:
        if set(group) != {"method", "values"} or not isinstance(group["values"], list):
            raise ValueError("removal plan scores schema is invalid")
        if [item.get("feature") if isinstance(item, dict) else None for item in group["values"]] != list(features):
            raise ValueError("removal plan score feature order is invalid")
        if any(
            not isinstance(item.get("score"), (int, float))
            or isinstance(item.get("score"), bool)
            or not __import__("math").isfinite(float(item["score"]))
            for item in group["values"]
        ):
            raise ValueError("removal plan scores must be finite")
    minppi_values = {item["feature"]: item["score"] for item in score_groups[0]["values"]}
    expected_harmful = [feature for feature in features if minppi_values[feature] == -1]
    if len(expected_harmful) != k or expected_harmful != minppi_removed:
        raise ValueError("removal plan harmful set conflicts with MinPPI scores")
    methods_by_name = {item["method"]: item for item in methods}
    feature_index = {feature: position for position, feature in enumerate(features)}
    for group in score_groups[1:]:
        scores = {item["feature"]: item["score"] for item in group["values"]}
        expected_removed = sorted(
            features,
            key=lambda feature: (float(scores[feature]), feature_index[feature]),
        )[:k]
        actual_removed = methods_by_name[group["method"]]["trials"][0][
            "removed_features"
        ]
        if actual_removed != [
            feature for feature in features if feature in set(expected_removed)
        ]:
            raise ValueError("removal control subset conflicts with its frozen scores")
    for trial, row in enumerate(methods_by_name["random_matched"]["trials"]):
        expected_seed = int(
            np.random.SeedSequence([context.job.seed, trial]).generate_state(1)[0]
        )
        sampled = np.random.default_rng(expected_seed).choice(
            features, size=k, replace=False
        ).tolist()
        expected_removed = [feature for feature in features if feature in set(sampled)]
        if row["rng_seed"] != expected_seed or row["removed_features"] != expected_removed:
            raise ValueError("random removal trial conflicts with its deterministic seed")
    for field in ("selection_model_fit_count", "selection_coalition_count"):
        if isinstance(document[field], bool) or not isinstance(document[field], int) or document[field] < 0:
            raise ValueError("removal plan cost counters are invalid")
    if document["selection_model_fit_count"] != sum(
        item["model_fit_count"] for item in methods
    ) or document["selection_coalition_count"] != sum(
        item["coalition_count"] for item in methods
    ):
        raise ValueError("removal aggregate costs conflict with method costs")
    return document


def _read_removal_plan(context: StageContext) -> dict[str, object]:
    path = context.layout.root / _removal_plan_relpath(context)
    return _validate_removal_plan(
        context, read_canonical_json(path, context="removal selection plan")
    )


def _selection_record(
    context: StageContext,
    *,
    started_at: str,
    relpath: str,
    metrics: Mapping[str, float | int],
) -> ResultRecord:
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics=dict(metrics),
        artifacts={"selection_plan": relpath},
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _removal_selection_metrics(
    document: Mapping[str, object], times: Mapping[str, float]
) -> dict[str, float | int]:
    methods = {item["method"]: item for item in document["methods"]}
    metrics: dict[str, float | int] = {
        "selection_time_s": sum(float(times[method]) for method in _REMOVAL_METHODS),
        "selection_model_fit_count": int(document["selection_model_fit_count"]),
        "selection_coalition_count": int(document["selection_coalition_count"]),
    }
    for method in _REMOVAL_METHODS:
        metrics[f"{method}_selection_time_s"] = float(times[method])
        metrics[f"{method}_selection_model_fit_count"] = int(
            methods[method]["model_fit_count"]
        )
        metrics[f"{method}_selection_coalition_count"] = int(
            methods[method]["coalition_count"]
        )
    minppi = methods["minppi"]
    metrics["minppi_selection_logical_unique_coalition_count"] = int(
        minppi["logical_unique_coalition_count"]
    )
    metrics["minppi_selection_physical_model_fit_count"] = int(
        minppi["physical_model_fit_count"]
    )
    return metrics


def _alpha_selection_metrics(
    document: Mapping[str, object], times: Mapping[str, float]
) -> dict[str, float | int | str]:
    return {
        "selection_time_s": float(document["selection_group_runtime_s"]),
        "selection_model_fit_count": int(document["selection_model_fit_count"]),
        "selection_coalition_count": int(document["selection_coalition_count"]),
        "selection_logical_evaluation_count": int(
            document["selection_logical_evaluation_count"]
        ),
        "selection_logical_unique_coalition_count": int(
            document["selection_logical_unique_coalition_count"]
        ),
        "selection_physical_model_fit_count": int(
            document["selection_physical_model_fit_count"]
        ),
        "selection_cache_hit_count": int(
            document["selection_cache_hit_count"]
        ),
        "selection_group_runtime_s": float(
            document["selection_group_runtime_s"]
        ),
        "group_request_digest": str(document["group_request_digest"]),
        **{
            f"{entry['alpha_token']}_selection_time_s": float(
                times[str(entry["alpha_token"])]
            )
            for entry in document["entries"]
        },
    }


def _run_removal_plan(context: StageContext) -> ResultRecord:
    import numpy as np

    from minppi_mofs.baselines.banzhaf_fs import compute_banzhaf_importance
    from minppi_mofs.baselines.min_shap import compute_minshap_importance
    from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance

    started_at = _timestamp()
    relpath = _removal_plan_relpath(context)
    path = context.layout.root / relpath
    if path.exists():
        document = _read_removal_plan(context)
        return _selection_record(
            context,
            started_at=started_at,
            relpath=relpath,
            metrics=_removal_selection_metrics(
                document, {method: 0.0 for method in _REMOVAL_METHODS}
            ),
        )
    dataset = _task9_authority(context, stage="removal")
    X_train, y_train, X_val, y_val, data_row = (
        _load_prepared_train_validation(context)
    )
    features = tuple(X_train.columns)
    budget = max(context.protocol.estimator_budgets)
    selection_times = {method: 0.0 for method in _REMOVAL_METHODS}
    minppi_started = time.perf_counter()
    checkpoint = _completed_estimator_checkpoint(
        context, "hybrid", budget, features
    )
    minppi = dict(checkpoint.scores)
    ordered_minppi = _task9_scores(minppi, features)
    harmful = [feature for feature in features if float(minppi[feature]) == -1.0]
    k = len(harmful)
    selection_times["minppi"] = time.perf_counter() - minppi_started
    diagnostics: dict[str, dict[str, int]] = {}
    baseline_values: dict[str, Mapping[str, float]] = {}
    def compute_baseline(
        method: str,
        function: Callable,
        extra: Mapping[str, object],
        *,
        coalition_utility: Callable | None = None,
    ) -> None:
        diag: dict[str, int] = {}
        baseline_started = time.perf_counter()
        kwargs: dict[str, object] = dict(
            task=dataset.task,
            utility=dataset.utility,
            model_params=context.protocol.model_params,
            backend=context.protocol.backend,
            seed=context.job.seed,
            diagnostics=diag,
            **extra,
        )
        if coalition_utility is not None:
            kwargs["coalition_utility"] = coalition_utility
        scores = function(
            X_train,
            y_train,
            **kwargs,
        )
        selection_times[method] = time.perf_counter() - baseline_started
        _task9_scores(scores, features)
        diagnostics[method] = diag
        baseline_values[method] = scores

    compute_baseline("shap_bottom", compute_shap_topk_importance, {})
    shared_diagnostics: dict[str, int] = {}
    with _open_shared_baseline_utility(
        context,
        X_train,
        y_train,
        data_row,
        shared_diagnostics,
    ) as shared_utility:
        compute_baseline(
            "minshap_bottom",
            compute_minshap_importance,
            {"n_coalitions": budget, "per_coalition": 5},
            coalition_utility=shared_utility,
        )
        compute_baseline(
            "banzhaf_bottom",
            compute_banzhaf_importance,
            {"n_coalitions": budget, "per_coalition": 5},
            coalition_utility=shared_utility,
        )
    methods: list[dict[str, object]] = []
    index = {feature: position for position, feature in enumerate(features)}
    for method in _REMOVAL_METHODS:
        if method == "full":
            trials = [_task9_subset_row(features, ())]
        elif method == "minppi":
            trials = [_task9_subset_row(features, harmful)]
        elif method == "random_matched":
            random_started = time.perf_counter()
            trials = []
            for trial in range(int(context.protocol.removal["random_repeats"])):
                rng_seed = int(
                    np.random.SeedSequence([context.job.seed, trial]).generate_state(1)[0]
                )
                removed = np.random.default_rng(rng_seed).choice(
                    features, size=k, replace=False
                ).tolist()
                trials.append(
                    _task9_subset_row(
                        features, removed, trial=trial, rng_seed=rng_seed
                    )
                )
            selection_times[method] = time.perf_counter() - random_started
        else:
            scores = baseline_values[method]
            removed = sorted(
                features, key=lambda feature: (float(scores[feature]), index[feature])
            )[:k]
            trials = [_task9_subset_row(features, removed)]
        if method == "minppi":
            model_fit_count = 0
            coalition_count = int(
                checkpoint.logical_unique_coalition_count
            )
        elif method in diagnostics:
            model_fit_count = int(diagnostics[method].get("model_fit_count", 0))
            coalition_count = int(diagnostics[method].get("coalition_count", 0))
        else:
            model_fit_count = coalition_count = 0
        method_document: dict[str, object] = {
            "method": method,
            "trials": trials,
            "model_fit_count": model_fit_count,
            "coalition_count": coalition_count,
        }
        if method == "minppi":
            method_document.update(
                logical_unique_coalition_count=(
                    checkpoint.logical_unique_coalition_count
                ),
                physical_model_fit_count=0,
            )
        methods.append(method_document)
    baseline_fit_count = sum(
        item.get("model_fit_count", 0) for item in diagnostics.values()
    )
    baseline_coalitions = sum(
        item.get("coalition_count", 0) for item in diagnostics.values()
    )
    document = {
        "schema_version": 1,
        "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id,
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "task": dataset.task,
        "utility": dataset.utility,
        "alpha": context.protocol.alpha_default,
        "budget": budget,
        "features": list(features),
        "k": k,
        "methods": methods,
        "scores": [
            {"method": "minppi", "values": ordered_minppi},
            *[
                {"method": method, "values": _task9_scores(baseline_values[method], features)}
                for method in ("shap_bottom", "minshap_bottom", "banzhaf_bottom")
            ],
        ],
        "selection_model_fit_count": int(baseline_fit_count),
        "selection_coalition_count": int(
            checkpoint.logical_unique_coalition_count
            + baseline_coalitions
        ),
    }
    _validate_removal_plan(context, document)
    _publish_idempotent_json(path, document, context="removal selection plan")
    return _selection_record(
        context,
        started_at=started_at,
        relpath=relpath,
        metrics=_removal_selection_metrics(document, selection_times),
    )


def _task9_plan_result(context: StageContext, *, stage: str) -> ResultRecord:
    import math

    plan = _task9_manifest_plan(context, stage=stage)
    if context.job.depends_on != (plan.job_id,):
        raise ValueError(f"{stage} job dependency does not match its selection plan")
    entry = context.status.entry(plan.job_id)
    if entry["state"] != "success":
        raise ValueError(f"{stage} selection plan dependency is not successful")
    record = read_result_record(context.layout.root / plan.output_relpath, plan)
    canonical_relpath = (
        _removal_plan_relpath(context)
        if stage == "removal"
        else _alpha_plan_relpath(context)
    )
    if record.artifacts != {"selection_plan": canonical_relpath}:
        raise ValueError(f"{stage} selection plan ResultRecord artifact is invalid")
    document = (
        _read_removal_plan(context) if stage == "removal" else _read_alpha_plan(context)
    )
    if stage == "removal":
        expected_keys = {
            "selection_time_s",
            "selection_model_fit_count",
            "selection_coalition_count",
            *(
                f"{method}_{suffix}"
                for method in _REMOVAL_METHODS
                for suffix in (
                    "selection_time_s",
                    "selection_model_fit_count",
                    "selection_coalition_count",
                )
            ),
            "minppi_selection_logical_unique_coalition_count",
            "minppi_selection_physical_model_fit_count",
        }
    else:
        expected_keys = {
            "selection_time_s",
            "selection_model_fit_count",
            "selection_coalition_count",
            "selection_logical_evaluation_count",
            "selection_logical_unique_coalition_count",
            "selection_physical_model_fit_count",
            "selection_cache_hit_count",
            "selection_group_runtime_s",
            "group_request_digest",
            *(
                f"{item['alpha_token']}_selection_time_s"
                for item in document["entries"]
            ),
        }
    if set(record.metrics) != expected_keys:
        raise ValueError(f"{stage} selection plan ResultRecord metric schema is invalid")
    for key, value in record.metrics.items():
        if key == "group_request_digest":
            if value != document["group_request_digest"]:
                raise ValueError(
                    "alpha selection request digest conflicts with plan"
                )
        elif key.endswith(("_time_s", "_runtime_s")):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{stage} selection runtime metric is invalid")
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{stage} selection counter metric {key!r} is invalid"
            )
    if (
        record.metrics["selection_model_fit_count"]
        != document["selection_model_fit_count"]
        or record.metrics["selection_coalition_count"]
        != document["selection_coalition_count"]
    ):
        raise ValueError(f"{stage} selection aggregate counters conflict with plan")
    if stage == "removal":
        methods = {item["method"]: item for item in document["methods"]}
        for method in _REMOVAL_METHODS:
            if (
                record.metrics[f"{method}_selection_model_fit_count"]
                != methods[method]["model_fit_count"]
                or record.metrics[f"{method}_selection_coalition_count"]
                != methods[method]["coalition_count"]
            ):
                raise ValueError("removal selection method counters conflict with plan")
        if (
            record.metrics[
                "minppi_selection_logical_unique_coalition_count"
            ]
            != methods["minppi"]["logical_unique_coalition_count"]
            or record.metrics[
                "minppi_selection_physical_model_fit_count"
            ]
            != methods["minppi"]["physical_model_fit_count"]
        ):
            raise ValueError(
                "removal reused MinPPI counters conflict with plan"
            )
        expected_time = sum(
            float(record.metrics[f"{method}_selection_time_s"])
            for method in _REMOVAL_METHODS
        )
    else:
        for field in (
            "selection_logical_evaluation_count",
            "selection_logical_unique_coalition_count",
            "selection_physical_model_fit_count",
            "selection_cache_hit_count",
        ):
            if record.metrics[field] != document[field]:
                raise ValueError(
                    "alpha selection grouped counters conflict with plan"
                )
        if (
            record.metrics["selection_group_runtime_s"]
            != document["selection_group_runtime_s"]
        ):
            raise ValueError(
                "alpha selection grouped runtime conflicts with plan"
            )
        for item in document["entries"]:
            if (
                record.metrics[
                    f"{item['alpha_token']}_selection_time_s"
                ]
                != item["component_runtime_s"]
            ):
                raise ValueError(
                    "alpha selection component runtime conflicts with plan"
                )
        expected_time = float(document["selection_group_runtime_s"])
    if record.metrics["selection_time_s"] != expected_time:
        raise ValueError(f"{stage} aggregate selection runtime is invalid")
    return record


def _task9_load_test(
    context: StageContext,
    row: Mapping[str, object],
    *,
    reason: str,
):
    import numpy as np

    from minppi_mofs.experiment_protocol import AuditedSplitAccess, PreparedSplitPaths

    paths = PreparedSplitPaths(
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        target=str(row["target"]),
        train_csv=context.layout.root / str(row["train_relpath"]),
        val_csv=context.layout.root / str(row["val_relpath"]),
        test_csv=context.layout.root / str(row["test_relpath"]),
    )
    access = AuditedSplitAccess(
        paths, stage=context.job.stage, job_id=context.job.job_id, audit=context.audit
    )
    X_test, y_test = access.load_test(
        reason=reason, expected_sha256=str(row["test_file_sha256"])
    )
    expected_features = json.loads(str(row["selected_features_json"]))
    if list(X_test.columns) != expected_features or len(X_test) != row["test_row_count"]:
        raise ValueError("audited test partition conflicts with manifest shape")
    if not np.isfinite(X_test.to_numpy(dtype=float)).all() or not np.isfinite(y_test).all():
        raise ValueError("audited test partition must be finite numeric data")
    expected_summary = json.loads(str(row["test_label_summary_json"]))
    actual_summary = _label_summary(y_test, str(row["task"]))
    if not _label_summaries_match(actual_summary, expected_summary, task=str(row["task"])):
        raise ValueError("audited test label summary conflicts with manifest")
    if str(row["task"]) != "regression":
        classes = np.unique(y_test)
        if not np.equal(classes, np.floor(classes)).all() or any(
            int(value) < 0 for value in classes
        ):
            raise ValueError("audited classification test labels are invalid")
        train_summary = json.loads(str(row["train_label_summary_json"]))
        train_classes = {
            int(item["label"]) for item in train_summary.get("classes", ())
        }
        if not set(int(value) for value in classes).issubset(train_classes):
            raise ValueError("audited test labels exceed training class authority")
    return X_test, y_test


def _task9_result_existing(
    context: StageContext, relpath: str, *, validator: Callable[[StageContext, object], dict[str, object]]
) -> dict[str, object] | None:
    path = context.layout.root / relpath
    if not path.exists():
        return None
    return validator(
        context, read_canonical_json(path, context=f"{context.job.stage} result artifact")
    )


def _removal_result_relpath(context: StageContext) -> str:
    return (
        f"removal/{context.job.dataset_id}/s{context.job.seed}/"
        f"{context.job.method}/results.json"
    )


def _validate_removal_result(context: StageContext, value: object) -> dict[str, object]:
    expected = {
        "schema_version", "protocol_hash", "job_id", "dataset_id", "seed",
        "method", "k", "rows", "selection_time_s", "evaluation_time_s",
        "selection_model_fit_count", "selection_coalition_count",
        "evaluation_model_fit_count",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("removal result artifact schema is invalid")
    if (
        value["schema_version"] != 1
        or value["protocol_hash"] != context.job.protocol_hash
        or value["job_id"] != context.job.job_id
        or value["dataset_id"] != context.job.dataset_id
        or value["seed"] != context.job.seed
        or value["method"] != context.job.method
    ):
        raise ValueError("removal result artifact identity is invalid")
    plan = _read_removal_plan(context)
    if value["k"] != plan["k"]:
        raise ValueError("removal result k conflicts with selection plan")
    method_plan = next(item for item in plan["methods"] if item["method"] == context.job.method)
    plan_record = _task9_plan_result(context, stage="removal")
    expected_selection_time = float(
        plan_record.metrics[f"{context.job.method}_selection_time_s"]
    )
    expected_evaluation_fits = sum(
        bool(features)
        for features in {
            tuple(plan["features"]),
            *(tuple(item["retained_features"]) for item in method_plan["trials"]),
        }
    )
    if (
        value["selection_time_s"] != expected_selection_time
        or value["selection_model_fit_count"] != method_plan["model_fit_count"]
        or value["selection_coalition_count"] != method_plan["coalition_count"]
        or value["evaluation_model_fit_count"] != expected_evaluation_fits
    ):
        raise ValueError("removal result counters or selection time are not exact")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != len(method_plan["trials"]):
        raise ValueError("removal result raw rows are incomplete")
    required_row = {
        "trial", "retained_features", "removed_features", "retained_hash", "k",
        "full_utility", "selected_utility", "delta_utility", "compression",
        "selection_time_s", "evaluation_time_s", "selection_model_fit_count",
        "selection_coalition_count", "evaluation_model_fit_count",
    }
    for result_row, planned in zip(rows, method_plan["trials"], strict=True):
        if not isinstance(result_row, dict) or set(result_row) != required_row:
            raise ValueError("removal raw row schema is invalid")
        for field in ("trial", "retained_features", "removed_features", "retained_hash"):
            if result_row[field] != planned[field]:
                raise ValueError("removal result did not reuse the frozen subset")
        if result_row["k"] != plan["k"]:
            raise ValueError("removal raw row k is invalid")
        expected_compression = plan["k"] / len(plan["features"]) if plan["features"] else 0.0
        if result_row["compression"] != expected_compression:
            raise ValueError("removal result compression is invalid")
        numeric = (
            "full_utility", "selected_utility", "delta_utility", "compression",
            "selection_time_s", "evaluation_time_s",
        )
        if any(
            isinstance(result_row[field], bool)
            or not isinstance(result_row[field], (int, float))
            or not __import__("math").isfinite(float(result_row[field]))
            for field in numeric
        ):
            raise ValueError("removal raw metrics must be finite")
        if result_row["delta_utility"] != result_row["selected_utility"] - result_row["full_utility"]:
            raise ValueError("removal utility delta is invalid")
        for field in (
            "selection_time_s", "evaluation_time_s", "selection_model_fit_count",
            "selection_coalition_count", "evaluation_model_fit_count",
        ):
            if result_row[field] != value[field]:
                raise ValueError("removal row counters or runtimes conflict with job totals")
    for field in ("selection_time_s", "evaluation_time_s"):
        if isinstance(value[field], bool) or not isinstance(value[field], (int, float)) or not __import__("math").isfinite(float(value[field])) or value[field] < 0:
            raise ValueError("removal runtime is invalid")
    for field in (
        "selection_model_fit_count", "selection_coalition_count",
        "evaluation_model_fit_count",
    ):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
            raise ValueError("removal fit count is invalid")
    if len({row["full_utility"] for row in rows}) != 1:
        raise ValueError("removal rows do not share one cached full utility")
    return value


def _result_record_from_artifact(
    context: StageContext, *, started_at: str, relpath: str, document: Mapping[str, object]
) -> ResultRecord:
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics={
            key: value
            for key, value in document.items()
            if key in {
                "k", "harmful_count", "selection_time_s", "evaluation_time_s",
                "selection_model_fit_count", "selection_coalition_count",
                "selection_logical_evaluation_count",
                "selection_logical_unique_coalition_count",
                "selection_physical_model_fit_count",
                "selection_cache_hit_count",
                "evaluation_model_fit_count",
            }
        },
        artifacts={"results": relpath},
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _run_removal_evaluation(context: StageContext) -> ResultRecord:
    import numpy as np
    import pandas as pd

    started_at = _timestamp()
    relpath = _removal_result_relpath(context)
    plan_record = _task9_plan_result(context, stage="removal")
    existing = _task9_result_existing(context, relpath, validator=_validate_removal_result)
    if existing is not None:
        return _result_record_from_artifact(
            context, started_at=started_at, relpath=relpath, document=existing
        )
    plan = _read_removal_plan(context)
    X_train, y_train, X_val, y_val, row = _load_prepared_train_validation(context)
    X_test, y_test = _task9_load_test(
        context, row, reason="final_removal_evaluation"
    )
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])
    dataset = _task9_authority(context, stage="removal")
    method_plan = next(item for item in plan["methods"] if item["method"] == context.job.method)
    evaluation_started = time.perf_counter()
    utility_cache: dict[tuple[str, ...], float] = {}

    def evaluate(features: Sequence[str]) -> float:
        key = tuple(features)
        if key not in utility_cache:
            utility_cache[key] = evaluate_fixed_subset(
                X_train_val, y_train_val, X_test, y_test, key,
                task=dataset.task, utility=dataset.utility,
                model_params=context.protocol.model_params,
                backend=context.protocol.backend, seed=context.job.seed,
            )
        return utility_cache[key]

    full = evaluate(tuple(plan["features"]))
    rows = []
    selection_time = float(
        plan_record.metrics[f"{context.job.method}_selection_time_s"]
    )
    selection_fits = int(method_plan["model_fit_count"])
    selection_coalitions = int(method_plan["coalition_count"])
    for planned in method_plan["trials"]:
        selected = evaluate(tuple(planned["retained_features"]))
        rows.append(
            {
                "trial": planned["trial"],
                "retained_features": planned["retained_features"],
                "removed_features": planned["removed_features"],
                "retained_hash": planned["retained_hash"],
                "k": plan["k"],
                "full_utility": full,
                "selected_utility": selected,
                "delta_utility": selected - full,
                "compression": plan["k"] / len(plan["features"]),
                "selection_time_s": selection_time,
                "evaluation_time_s": 0.0,
                "selection_model_fit_count": selection_fits,
                "selection_coalition_count": selection_coalitions,
                "evaluation_model_fit_count": sum(bool(key) for key in utility_cache),
            }
        )
    evaluation_time = time.perf_counter() - evaluation_started
    evaluation_fit_count = sum(bool(key) for key in utility_cache)
    for raw_row in rows:
        raw_row["evaluation_time_s"] = evaluation_time
        raw_row["evaluation_model_fit_count"] = evaluation_fit_count
    document = {
        "schema_version": 1,
        "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id,
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "method": context.job.method,
        "k": plan["k"],
        "rows": rows,
        "selection_time_s": selection_time,
        "evaluation_time_s": evaluation_time,
        "selection_model_fit_count": selection_fits,
        "selection_coalition_count": selection_coalitions,
        "evaluation_model_fit_count": evaluation_fit_count,
    }
    _validate_removal_result(context, document)
    _publish_idempotent_json(
        context.layout.root / relpath, document, context="removal result artifact"
    )
    return _result_record_from_artifact(
        context, started_at=started_at, relpath=relpath, document=document
    )


def run_removal_job(context: StageContext) -> ResultRecord:
    """Run one frozen matched-removal plan or final evaluation job."""
    dataset = _task9_authority(context, stage="removal")
    _validate_removal_scope(context)
    expected = {"task", "utility", "random_repeats"} if context.job.method == "selection_plan" else {"task", "utility"}
    if set(context.job.parameters) != expected or context.job.parameters["task"] != dataset.task or context.job.parameters["utility"] != dataset.utility:
        raise ValueError("removal job parameters conflict with dataset authority")
    if context.job.method == "selection_plan":
        if context.job.parameters["random_repeats"] != context.protocol.removal["random_repeats"]:
            raise ValueError("removal random repeat count conflicts with protocol")
        return _run_removal_plan(context)
    if context.job.method not in _REMOVAL_METHODS:
        raise ValueError("removal method is not canonical")
    return _run_removal_evaluation(context)


def _alpha_token(alpha: float) -> str:
    return float(alpha).hex().replace("-", "m").replace("+", "p").replace(".", "d")


def _alpha_plan_relpath(context: StageContext) -> str:
    return f"alpha/{context.job.dataset_id}/s{context.job.seed}/selection_plan.json"


class _AlphaCoalitionUtilityView(CoalitionUtilityView):
    """Record alpha-local logical calls while sharing one persistent cache."""

    def __init__(
        self,
        cache: CoalitionUtilityCache,
        *,
        utility: str,
        alpha: float,
    ) -> None:
        self._logical_coalitions: list[tuple[str, ...]] = []
        super().__init__(cache, utility=utility, alpha=alpha)

    def evaluate(self, coalition) -> float:
        self._logical_coalitions.append(
            canonical_coalition(self.feature_order, coalition)
        )
        return super().evaluate(coalition)

    @property
    def logical_coalitions(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._logical_coalitions)


def _alpha_estimator_authority(
    context: StageContext,
    data_row: Mapping[str, object],
    features: tuple[str, ...],
):
    """Bind alpha work to the completed maximum-budget grouped schedule."""
    budget = max(context.protocol.estimator_budgets)
    checkpoint = _completed_estimator_checkpoint(
        context, "hybrid", budget, features
    )
    estimator_jobs = tuple(
        job
        for job in context.status.manifest
        if job.stage == "estimator"
        and job.dataset_id == context.job.dataset_id
        and job.seed == context.job.seed
        and job.method == "hybrid"
        and job.parameters.get("budget") == budget
    )
    if len(estimator_jobs) != 1:
        raise ValueError(
            "alpha estimator group authority is missing or duplicated"
        )
    estimator_context = StageContext(
        context.protocol,
        context.layout,
        estimator_jobs[0],
        context.status,
        context.audit,
    )
    request = group_request_from_context(
        estimator_context, data_row, features
    )
    if __package__:
        from .icomp2026_estimator_bundle import (
            _hybrid_schedule_digest,
            _schedule_authority,
        )
    else:
        from icomp2026_estimator_bundle import (  # type: ignore[no-redef]
            _hybrid_schedule_digest,
            _schedule_authority,
        )

    schedules = _schedule_authority(request)
    expected_schedule_digest = _hybrid_schedule_digest(
        feature_count=len(features),
        random_digest=schedules.random_digest,
        guided_digest=schedules.guided_digest,
    )
    if checkpoint.schedule_digest != expected_schedule_digest:
        raise ValueError(
            "alpha hybrid checkpoint schedule digest conflicts with "
            "deterministic schedule authority"
        )
    return checkpoint, request, schedules


def _validate_alpha_scope(context: StageContext) -> None:
    jobs = tuple(
        job for job in context.status.manifest
        if job.stage == "alpha" and job.dataset_id == context.job.dataset_id and job.seed == context.job.seed
    )
    if tuple(job.method for job in jobs) != ("selection_plan", *("alpha_evaluation" for _ in context.protocol.alpha_grid)):
        raise ValueError("alpha manifest scope is incomplete or out of order")
    if [job.parameters.get("alpha") for job in jobs[1:]] != list(context.protocol.alpha_grid):
        raise ValueError("alpha manifest grid conflicts with protocol order")
    if any(job.depends_on != (jobs[0].job_id,) for job in jobs[1:]):
        raise ValueError("alpha dependencies do not reference the frozen plan")


def _validate_alpha_plan(context: StageContext, value: object) -> dict[str, object]:
    expected = {
        "schema_version", "protocol_hash", "job_id", "dataset_id", "seed", "task",
        "utility", "budget", "features", "entries", "selection_model_fit_count",
        "selection_coalition_count", "initial_model_fit_count",
        "initial_coalition_count", "group_request_digest", "schedule_digest",
        "selection_logical_evaluation_count",
        "selection_logical_unique_coalition_count",
        "selection_physical_model_fit_count", "selection_cache_hit_count",
        "selection_group_runtime_s",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("alpha selection plan schema is invalid")
    dataset = _task9_authority(context, stage="alpha")
    if (
        value["schema_version"] != 2
        or value["protocol_hash"] != context.job.protocol_hash
        or value["job_id"] != _task9_manifest_plan(context, stage="alpha").job_id
        or value["dataset_id"] != context.job.dataset_id
        or value["seed"] != context.job.seed
        or value["task"] != dataset.task
        or value["utility"] != dataset.utility
        or value["budget"] != max(context.protocol.estimator_budgets)
    ):
        raise ValueError("alpha selection plan identity is invalid")
    features_value = value["features"]
    if not isinstance(features_value, list) or len(set(features_value)) != len(features_value) or any(not isinstance(item, str) for item in features_value):
        raise ValueError("alpha plan features are invalid")
    features = tuple(features_value)
    manifest_row = _task9_manifest_row(context)
    manifest_features = tuple(
        json.loads(str(manifest_row["selected_features_json"]))
    )
    if features != manifest_features:
        raise ValueError("alpha plan features conflict with rich manifest authority")
    checkpoint, request, _schedules = _alpha_estimator_authority(
        context, manifest_row, features
    )
    if (
        value["group_request_digest"] != request.digest
        or value["schedule_digest"] != checkpoint.schedule_digest
    ):
        raise ValueError(
            "alpha plan request or schedule digest conflicts with "
            "estimator authority"
        )
    entries = value["entries"]
    if not isinstance(entries, list) or [item.get("alpha") if isinstance(item, dict) else None for item in entries] != list(context.protocol.alpha_grid):
        raise ValueError("alpha selection plan grid is incomplete or out of order")
    expected_entry = {
        "alpha", "alpha_token", "harmful_count", "compression", "validation_utility",
        "retained_features", "removed_features", "retained_hash", "model_fit_count",
        "coalition_count", "scores", "logical_evaluation_count",
        "logical_unique_coalition_count", "physical_model_fit_count",
        "cache_hit_count", "component_runtime_s",
    }
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected_entry:
            raise ValueError("alpha plan entry schema is invalid")
        alpha = entry["alpha"]
        if entry["alpha_token"] != _alpha_token(alpha):
            raise ValueError("alpha plan token is invalid")
        harmful_count = entry["harmful_count"]
        if (
            isinstance(harmful_count, bool)
            or not isinstance(harmful_count, int)
            or harmful_count < 0
        ):
            raise ValueError(
                "alpha plan harmful count must be a non-negative integer"
            )
        subset = {
            "trial": None, "rng_seed": None,
            "retained_features": entry["retained_features"],
            "removed_features": entry["removed_features"],
            "retained_hash": entry["retained_hash"],
        }
        _task9_validate_subset(
            subset, features=features, expected_removed_count=harmful_count,
            random_trial=False,
        )
        scores = entry["scores"]
        if (
            not isinstance(scores, list)
            or [item.get("feature") if isinstance(item, dict) else None for item in scores]
            != list(features)
            or any(
                isinstance(item.get("score"), bool)
                or not isinstance(item.get("score"), (int, float))
                or not __import__("math").isfinite(float(item["score"]))
                or float(item["score"]) not in (-1.0, 0.0, 1.0)
                for item in scores
            )
        ):
            raise ValueError("alpha plan score evidence is invalid")
        expected_removed = [
            item["feature"] for item in scores if float(item["score"]) == -1.0
        ]
        if entry["removed_features"] != expected_removed:
            raise ValueError("alpha harmful set conflicts with frozen scores")
        if entry["compression"] != harmful_count / len(features):
            raise ValueError("alpha plan compression is invalid")
        if not __import__("math").isfinite(float(entry["validation_utility"])):
            raise ValueError("alpha validation utility is invalid")
        for field in (
            "model_fit_count", "coalition_count",
            "logical_evaluation_count", "logical_unique_coalition_count",
            "physical_model_fit_count", "cache_hit_count",
        ):
            if isinstance(entry[field], bool) or not isinstance(entry[field], int) or entry[field] < 0:
                raise ValueError("alpha plan cost is invalid")
        if (
            entry["model_fit_count"] > entry["coalition_count"]
            or entry["physical_model_fit_count"] != entry["model_fit_count"]
            or entry["logical_unique_coalition_count"]
            > entry["logical_evaluation_count"]
            or entry["physical_model_fit_count"]
            > entry["logical_unique_coalition_count"]
            or entry["cache_hit_count"] > entry["logical_evaluation_count"]
        ):
            raise ValueError("alpha plan model-fit delta exceeds coalition delta")
        component_runtime = entry["component_runtime_s"]
        if (
            isinstance(component_runtime, bool)
            or not isinstance(component_runtime, (int, float))
            or not __import__("math").isfinite(float(component_runtime))
            or component_runtime < 0
        ):
            raise ValueError("alpha plan component runtime is invalid")
    for field in (
        "initial_model_fit_count", "initial_coalition_count",
        "selection_model_fit_count", "selection_coalition_count",
        "selection_logical_evaluation_count",
        "selection_logical_unique_coalition_count",
        "selection_physical_model_fit_count", "selection_cache_hit_count",
    ):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
            raise ValueError("alpha plan aggregate cost is invalid")
    if (
        value["initial_model_fit_count"] != 0
        or value["initial_coalition_count"] != 0
    ):
        raise ValueError("alpha plan initial full-coalition costs are invalid")
    if value["selection_model_fit_count"] != value["initial_model_fit_count"] + sum(
        entry["model_fit_count"] for entry in entries
    ):
        raise ValueError("alpha plan aggregate model-fit count is invalid")
    if value["selection_coalition_count"] != value["initial_coalition_count"] + sum(
        entry["coalition_count"] for entry in entries
    ):
        raise ValueError("alpha plan aggregate coalition count is invalid")
    if (
        value["selection_physical_model_fit_count"]
        != value["selection_model_fit_count"]
        or value["selection_physical_model_fit_count"]
        != sum(entry["physical_model_fit_count"] for entry in entries)
        or value["selection_cache_hit_count"]
        != sum(entry["cache_hit_count"] for entry in entries)
        or value["selection_logical_evaluation_count"]
        != sum(entry["logical_evaluation_count"] for entry in entries)
        or value["selection_logical_unique_coalition_count"]
        > value["selection_logical_evaluation_count"]
        or value["selection_physical_model_fit_count"]
        > value["selection_logical_unique_coalition_count"]
        or value["selection_cache_hit_count"]
        > value["selection_logical_evaluation_count"]
        or value["selection_logical_unique_coalition_count"]
        > sum(
            entry["logical_unique_coalition_count"] for entry in entries
        )
    ):
        raise ValueError("alpha plan grouped accounting is invalid")
    group_runtime = value["selection_group_runtime_s"]
    if (
        isinstance(group_runtime, bool)
        or not isinstance(group_runtime, (int, float))
        or not __import__("math").isfinite(float(group_runtime))
        or group_runtime < 0
        or group_runtime + 1e-12
        < __import__("math").fsum(
            float(entry["component_runtime_s"]) for entry in entries
        )
    ):
        raise ValueError("alpha plan grouped runtime is invalid")
    return value


def _read_alpha_plan(context: StageContext) -> dict[str, object]:
    return _validate_alpha_plan(
        context,
        read_canonical_json(
            context.layout.root / _alpha_plan_relpath(context),
            context="alpha selection plan",
        ),
    )


def _derive_alpha_max_budget_scores(
    *,
    feature_count: int,
    budgets: Sequence[int],
    exact_scores: Mapping[str, int] | None,
    random_scores: Mapping[int, Mapping[str, int]] | None,
    guided_scores: Mapping[int, Mapping[str, int]] | None,
) -> dict[str, int]:
    """Derive every authorized checkpoint, then select the maximum budget."""
    from minppi_mofs.nested_estimation import derive_hybrid_checkpoints

    checkpoints = tuple(budgets)
    derived = derive_hybrid_checkpoints(
        feature_count=feature_count,
        checkpoints=checkpoints,
        exact_scores=exact_scores,
        random_scores=random_scores,
        guided_scores=guided_scores,
    )
    return derived[checkpoints[-1]]


def _run_alpha_plan(context: StageContext) -> ResultRecord:
    from minppi_mofs.min_ppi import estimate_minppi_exact
    from minppi_mofs.nested_estimation import (
        estimate_guided_checkpoints,
        estimate_random_checkpoints,
    )
    from minppi_mofs.shapg_integration import build_mi_correlation_guidance

    started_at = _timestamp()
    relpath = _alpha_plan_relpath(context)
    path = context.layout.root / relpath
    if path.exists():
        document = _read_alpha_plan(context)
        return _selection_record(
            context,
            started_at=started_at,
            relpath=relpath,
            metrics=_alpha_selection_metrics(
                document,
                {
                    str(item["alpha_token"]): float(
                        item["component_runtime_s"]
                    )
                    for item in document["entries"]
                },
            ),
        )
    dataset = _task9_authority(context, stage="alpha")
    X_train, y_train, X_val, y_val, data_row = (
        _load_prepared_train_validation(context)
    )
    features = tuple(X_train.columns)
    budget = max(context.protocol.estimator_budgets)
    checkpoint, request, schedules = _alpha_estimator_authority(
        context, data_row, features
    )
    if schedules.guided_schedule is not None:
        mi_scores, neighbors = build_mi_correlation_guidance(
            X_train, y_train, task=dataset.task, seed=context.job.seed
        )
    else:
        mi_scores, neighbors = None, None
    entries = []
    selection_times: dict[str, float] = {}
    group_logical_coalitions: set[tuple[str, ...]] = set()
    group_started = time.perf_counter()
    with _open_selection_cache(
        context, (X_train, y_train, X_val, y_val), data_row
    ) as cache:
        group_before = cache.stats
        for alpha in context.protocol.alpha_grid:
            alpha_started = time.perf_counter()
            alpha_before = cache.stats
            view = _AlphaCoalitionUtilityView(
                cache, utility=dataset.utility, alpha=alpha
            )
            exact_scores = None
            random_scores = None
            guided_scores = None
            if len(features) <= 20:
                exact_scores = estimate_minppi_exact(
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    alpha=alpha,
                    task=dataset.task,
                    utility=dataset.utility,
                    acc_cache=view,
                    seed=context.job.seed,
                )
            else:
                if len(features) > 75:
                    if schedules.random_schedule is None:
                        raise RuntimeError(
                            "alpha random schedule authority is missing"
                        )
                    random_scores = estimate_random_checkpoints(
                        schedules.random_schedule, view
                    )
                if schedules.guided_schedule is None:
                    raise RuntimeError(
                        "alpha guided schedule authority is missing"
                    )
                guided_scores = estimate_guided_checkpoints(
                    schedules.guided_schedule,
                    view,
                    mi_scores=mi_scores,
                    correlation_neighbors=neighbors,
                )
            scores = _derive_alpha_max_budget_scores(
                feature_count=len(features),
                budgets=request.budgets,
                exact_scores=exact_scores,
                random_scores=random_scores,
                guided_scores=guided_scores,
            )
            ordered_scores = _task9_scores(scores, features)
            removed = [
                feature
                for feature in features
                if float(scores[feature]) == -1.0
            ]
            subset = _task9_subset_row(features, removed)
            validation_utility = float(
                view.evaluate(set(subset["retained_features"]))
            )
            alpha_after = cache.stats
            physical_delta = (
                alpha_after.physical_model_fit_count
                - alpha_before.physical_model_fit_count
            )
            hit_delta = (
                alpha_after.cache_hit_count - alpha_before.cache_hit_count
            )
            coalition_delta = (
                alpha_after.coalition_count - alpha_before.coalition_count
            )
            logical_coalitions = view.logical_coalitions
            group_logical_coalitions.update(logical_coalitions)
            alpha_token = _alpha_token(alpha)
            component_runtime = time.perf_counter() - alpha_started
            selection_times[alpha_token] = component_runtime
            entries.append(
                {
                    "alpha": alpha,
                    "alpha_token": alpha_token,
                    "harmful_count": len(removed),
                    "compression": len(removed) / len(features),
                    "validation_utility": validation_utility,
                    "retained_features": subset["retained_features"],
                    "removed_features": subset["removed_features"],
                    "retained_hash": subset["retained_hash"],
                    "model_fit_count": physical_delta,
                    "coalition_count": coalition_delta,
                    "logical_evaluation_count": len(logical_coalitions),
                    "logical_unique_coalition_count": len(
                        set(logical_coalitions)
                    ),
                    "physical_model_fit_count": physical_delta,
                    "cache_hit_count": hit_delta,
                    "component_runtime_s": component_runtime,
                    "scores": ordered_scores,
                }
            )
        group_after = cache.stats
    group_runtime = time.perf_counter() - group_started
    group_physical_fits = (
        group_after.physical_model_fit_count
        - group_before.physical_model_fit_count
    )
    group_cache_hits = (
        group_after.cache_hit_count - group_before.cache_hit_count
    )
    group_coalition_delta = (
        group_after.coalition_count - group_before.coalition_count
    )
    document = {
        "schema_version": 2, "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id, "dataset_id": context.job.dataset_id,
        "seed": context.job.seed, "task": dataset.task, "utility": dataset.utility,
        "budget": budget, "features": list(features), "entries": entries,
        "group_request_digest": request.digest,
        "schedule_digest": checkpoint.schedule_digest,
        "initial_model_fit_count": 0,
        "initial_coalition_count": 0,
        "selection_model_fit_count": group_physical_fits,
        "selection_coalition_count": group_coalition_delta,
        "selection_logical_evaluation_count": sum(
            int(entry["logical_evaluation_count"]) for entry in entries
        ),
        "selection_logical_unique_coalition_count": len(
            group_logical_coalitions
        ),
        "selection_physical_model_fit_count": group_physical_fits,
        "selection_cache_hit_count": group_cache_hits,
        "selection_group_runtime_s": group_runtime,
    }
    _validate_alpha_plan(context, document)
    _publish_idempotent_json(path, document, context="alpha selection plan")
    return _selection_record(
        context,
        started_at=started_at,
        relpath=relpath,
        metrics=_alpha_selection_metrics(document, selection_times),
    )


def _alpha_result_relpath(context: StageContext, alpha: float) -> str:
    return f"alpha/{context.job.dataset_id}/s{context.job.seed}/{_alpha_token(alpha)}/results.json"


def _validate_alpha_result(context: StageContext, value: object) -> dict[str, object]:
    expected = {
        "schema_version", "protocol_hash", "job_id", "dataset_id", "seed", "alpha",
        "alpha_token", "harmful_count", "retained_features", "removed_features",
        "retained_hash", "compression", "validation_utility", "test_utility",
        "full_test_utility", "delta_utility", "preservation", "selection_time_s",
        "evaluation_time_s", "selection_model_fit_count",
        "selection_coalition_count", "evaluation_model_fit_count",
        "selection_logical_evaluation_count",
        "selection_logical_unique_coalition_count",
        "selection_physical_model_fit_count", "selection_cache_hit_count",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("alpha result artifact schema is invalid")
    alpha = context.job.parameters["alpha"]
    if (
        value["schema_version"] != 1 or value["protocol_hash"] != context.job.protocol_hash
        or value["job_id"] != context.job.job_id or value["dataset_id"] != context.job.dataset_id
        or value["seed"] != context.job.seed or value["alpha"] != alpha
        or value["alpha_token"] != _alpha_token(alpha)
    ):
        raise ValueError("alpha result artifact identity is invalid")
    plan = _read_alpha_plan(context)
    entry = next((item for item in plan["entries"] if item["alpha"] == alpha), None)
    if entry is None:
        raise ValueError("alpha result is outside the frozen plan")
    plan_record = _task9_plan_result(context, stage="alpha")
    expected_evaluation_fits = sum(
        bool(features)
        for features in {
            tuple(plan["features"]), tuple(entry["retained_features"])
        }
    )
    if (
        value["selection_time_s"]
        != plan_record.metrics[f"{entry['alpha_token']}_selection_time_s"]
        or value["selection_model_fit_count"] != entry["model_fit_count"]
        or value["selection_coalition_count"] != entry["coalition_count"]
        or value["selection_logical_evaluation_count"]
        != entry["logical_evaluation_count"]
        or value["selection_logical_unique_coalition_count"]
        != entry["logical_unique_coalition_count"]
        or value["selection_physical_model_fit_count"]
        != entry["physical_model_fit_count"]
        or value["selection_cache_hit_count"] != entry["cache_hit_count"]
        or value["evaluation_model_fit_count"] != expected_evaluation_fits
    ):
        raise ValueError("alpha result counters or selection time are not exact")
    for field in (
        "harmful_count", "retained_features", "removed_features", "retained_hash",
        "compression", "validation_utility",
    ):
        if value[field] != entry[field]:
            raise ValueError("alpha result did not reuse the exact frozen plan entry")
    numeric = (
        "test_utility", "full_test_utility", "delta_utility", "preservation",
        "selection_time_s", "evaluation_time_s",
    )
    if any(
        isinstance(value[field], bool) or not isinstance(value[field], (int, float))
        or not __import__("math").isfinite(float(value[field])) for field in numeric
    ):
        raise ValueError("alpha result metrics must be finite")
    if value["selection_time_s"] < 0 or value["evaluation_time_s"] < 0:
        raise ValueError("alpha result runtimes must be nonnegative")
    if value["delta_utility"] != value["test_utility"] - value["full_test_utility"]:
        raise ValueError("alpha result utility delta is invalid")
    expected_preservation = value["test_utility"] / value["full_test_utility"] if value["full_test_utility"] > 0 else 1.0
    if value["preservation"] != expected_preservation:
        raise ValueError("alpha result preservation is invalid")
    for field in (
        "selection_model_fit_count", "selection_coalition_count",
        "selection_logical_evaluation_count",
        "selection_logical_unique_coalition_count",
        "selection_physical_model_fit_count", "selection_cache_hit_count",
        "evaluation_model_fit_count",
    ):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
            raise ValueError("alpha result fit count is invalid")
    return value


def _run_alpha_evaluation(context: StageContext) -> ResultRecord:
    import numpy as np
    import pandas as pd

    started_at = _timestamp()
    alpha = float(context.job.parameters["alpha"])
    relpath = _alpha_result_relpath(context, alpha)
    plan_record = _task9_plan_result(context, stage="alpha")
    existing = _task9_result_existing(context, relpath, validator=_validate_alpha_result)
    if existing is not None:
        return _result_record_from_artifact(context, started_at=started_at, relpath=relpath, document=existing)
    plan = _read_alpha_plan(context)
    entry = next(item for item in plan["entries"] if item["alpha"] == alpha)
    X_train, y_train, X_val, y_val, row = _load_prepared_train_validation(context)
    X_test, y_test = _task9_load_test(context, row, reason="final_alpha_grid_evaluation")
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])
    dataset = _task9_authority(context, stage="alpha")
    evaluation_started = time.perf_counter()
    cache: dict[tuple[str, ...], float] = {}

    def evaluate(features: Sequence[str]) -> float:
        key = tuple(features)
        if key not in cache:
            cache[key] = evaluate_fixed_subset(
                X_train_val, y_train_val, X_test, y_test, key,
                task=dataset.task, utility=dataset.utility,
                model_params=context.protocol.model_params,
                backend=context.protocol.backend, seed=context.job.seed,
            )
        return cache[key]

    full = evaluate(tuple(plan["features"]))
    selected = evaluate(tuple(entry["retained_features"]))
    evaluation_time = time.perf_counter() - evaluation_started
    document = {
        "schema_version": 1, "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id, "dataset_id": context.job.dataset_id,
        "seed": context.job.seed, "alpha": alpha, "alpha_token": _alpha_token(alpha),
        "harmful_count": entry["harmful_count"],
        "retained_features": entry["retained_features"],
        "removed_features": entry["removed_features"],
        "retained_hash": entry["retained_hash"], "compression": entry["compression"],
        "validation_utility": entry["validation_utility"], "test_utility": selected,
        "full_test_utility": full, "delta_utility": selected - full,
        "preservation": selected / full if full > 0 else 1.0,
        "selection_time_s": float(
            plan_record.metrics[f"{entry['alpha_token']}_selection_time_s"]
        ),
        "evaluation_time_s": evaluation_time,
        "selection_model_fit_count": int(entry["model_fit_count"]),
        "selection_coalition_count": int(entry["coalition_count"]),
        "selection_logical_evaluation_count": int(
            entry["logical_evaluation_count"]
        ),
        "selection_logical_unique_coalition_count": int(
            entry["logical_unique_coalition_count"]
        ),
        "selection_physical_model_fit_count": int(
            entry["physical_model_fit_count"]
        ),
        "selection_cache_hit_count": int(entry["cache_hit_count"]),
        "evaluation_model_fit_count": sum(bool(key) for key in cache),
    }
    _validate_alpha_result(context, document)
    _publish_idempotent_json(context.layout.root / relpath, document, context="alpha result artifact")
    return _result_record_from_artifact(context, started_at=started_at, relpath=relpath, document=document)


def run_alpha_job(context: StageContext) -> ResultRecord:
    """Run one frozen alpha-grid selection plan or final evaluation job."""
    dataset = _task9_authority(context, stage="alpha")
    _validate_alpha_scope(context)
    expected = {"task", "utility", "alpha_grid"} if context.job.method == "selection_plan" else {"task", "utility", "alpha"}
    if set(context.job.parameters) != expected or context.job.parameters["task"] != dataset.task or context.job.parameters["utility"] != dataset.utility:
        raise ValueError("alpha job parameters conflict with dataset authority")
    if context.job.method == "selection_plan":
        if context.job.parameters["alpha_grid"] != list(context.protocol.alpha_grid):
            raise ValueError("alpha plan grid conflicts with protocol")
        return _run_alpha_plan(context)
    if context.job.method != "alpha_evaluation":
        raise ValueError("alpha method is not canonical")
    alpha = context.job.parameters["alpha"]
    if isinstance(alpha, bool) or alpha not in context.protocol.alpha_grid or sum(value == alpha for value in context.protocol.alpha_grid) != 1:
        raise ValueError("alpha evaluation value is not a unique frozen grid member")
    return _run_alpha_evaluation(context)


_NSGA2_VARIANTS = ("full", "minppi_prefilter")
_NSGA2_REF_POINT = (1.1, 1.1)


def _nsga2_authority(
    context: StageContext,
) -> tuple[object, tuple[JobSpec, ...], int, int, int]:
    dataset = _task9_authority(context, stage="nsga2")
    if tuple(context.protocol.nsga2.get("variants", ())) != _NSGA2_VARIANTS:
        raise ValueError("NSGA-II variants are not the canonical ordered pair")
    jobs = tuple(
        job
        for job in context.status.manifest
        if job.stage == "nsga2"
        and job.dataset_id == context.job.dataset_id
        and job.seed == context.job.seed
    )
    if tuple(job.method for job in jobs) != _NSGA2_VARIANTS:
        raise ValueError("NSGA-II manifest scope is incomplete or out of order")
    population = context.protocol.nsga2.get("population")
    evaluations = context.protocol.nsga2.get("evaluations")
    if (
        isinstance(population, bool)
        or not isinstance(population, int)
        or population < 1
        or isinstance(evaluations, bool)
        or not isinstance(evaluations, int)
        or evaluations < 1
    ):
        raise ValueError("NSGA-II population and evaluation budget must be positive integers")
    if evaluations % population:
        raise ValueError(
            "NSGA-II evaluation budget must be divisible by the population size"
        )
    max_gen = evaluations // population - 1
    if max_gen < 0:
        raise ValueError("NSGA-II budget cannot be smaller than its initial population")
    expected_parameters = {
        "task",
        "utility",
        "evaluations",
        "population",
        "ref_point",
    }
    for job in jobs:
        if (
            set(job.parameters) != expected_parameters
            or job.parameters["task"] != dataset.task
            or job.parameters["utility"] != dataset.utility
            or job.parameters["evaluations"] != evaluations
            or job.parameters["population"] != population
            or job.parameters["ref_point"] != list(_NSGA2_REF_POINT)
            or job.depends_on
        ):
            raise ValueError("NSGA-II job parameters conflict with frozen authority")
    if tuple(context.protocol.nsga2.get("ref_point", ())) != _NSGA2_REF_POINT:
        raise ValueError("NSGA-II reference point is not the frozen (1.1, 1.1)")
    return dataset, jobs, population, evaluations, max_gen


def _nsga2_job_context(context: StageContext, job: JobSpec) -> StageContext:
    return StageContext(
        context.protocol, context.layout, job, context.status, context.audit
    )


def _nsga2_removal_authority(
    context: StageContext,
) -> tuple[dict[str, object], ResultRecord, tuple[str, ...]]:
    minppi_jobs = tuple(
        job
        for job in context.status.manifest
        if job.stage == "removal"
        and job.dataset_id == context.job.dataset_id
        and job.seed == context.job.seed
        and job.method == "minppi"
    )
    if len(minppi_jobs) != 1:
        raise ValueError("NSGA-II requires the canonical MinPPI removal job")
    removal_context = _nsga2_job_context(context, minppi_jobs[0])
    _validate_removal_scope(removal_context)
    plan = _read_removal_plan(removal_context)
    plan_record = _task9_plan_result(removal_context, stage="removal")
    method_plan = next(
        item for item in plan["methods"] if item["method"] == "minppi"
    )
    retained = tuple(method_plan["trials"][0]["retained_features"])
    return plan, plan_record, retained


def _nsga2_front_relpath(context: StageContext, variant: str) -> str:
    return (
        f"nsga2/{context.job.dataset_id}/s{context.job.seed}/"
        f"{variant}/validation_front.json"
    )


def _nsga2_result_relpath(context: StageContext) -> str:
    return (
        f"nsga2/{context.job.dataset_id}/s{context.job.seed}/"
        f"{context.job.method}/results.json"
    )


def _nsga2_search_features(
    variant: str, original: tuple[str, ...], retained: tuple[str, ...]
) -> tuple[str, ...]:
    if variant == "full":
        return original
    if variant == "minppi_prefilter":
        return retained
    raise ValueError("unknown NSGA-II variant")


def _finite_nonnegative(value: object, *, name: str) -> float:
    import math

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative finite number")
    return float(value)


def _exact_nonnegative_int(value: object, expected: int, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value != expected
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _validate_nsga2_front_entries(
    entries: object,
    *,
    search_features: tuple[str, ...],
    denominator: int,
    objective_name: str,
) -> list[dict[str, object]]:
    import math

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"NSGA-II {objective_name} front must be a nonempty list")
    expected = {"mask", "selected_features", "objectives"}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected:
            raise ValueError(f"NSGA-II {objective_name} front entry schema is invalid")
        mask = entry["mask"]
        if (
            not isinstance(mask, list)
            or len(mask) != len(search_features)
            or any(isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1) for bit in mask)
        ):
            raise ValueError(f"NSGA-II {objective_name} mask is invalid")
        selected = [
            feature for feature, bit in zip(search_features, mask, strict=True) if bit
        ]
        if entry["selected_features"] != selected:
            raise ValueError(f"NSGA-II {objective_name} selected features conflict with mask")
        objectives = entry["objectives"]
        if (
            not isinstance(objectives, list)
            or len(objectives) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in objectives
            )
        ):
            raise ValueError(f"NSGA-II {objective_name} objectives are invalid")
        if objectives[0] != sum(mask) / denominator or not 0.0 <= objectives[1] <= 1.0:
            raise ValueError(f"NSGA-II {objective_name} objectives conflict with mask or utility")
    return entries


def _validate_nsga2_front_artifact(
    context: StageContext,
    value: object,
    *,
    variant: str,
    original_features: tuple[str, ...],
    retained_features: tuple[str, ...],
) -> dict[str, object]:
    import numpy as np

    dataset, jobs, population, evaluations, max_gen = _nsga2_authority(context)
    job = next(item for item in jobs if item.method == variant)
    expected = {
        "schema_version", "protocol_hash", "job_id", "dataset_id", "seed",
        "variant", "task", "utility", "population", "evaluations", "max_gen",
        "ref_point", "original_features", "search_features",
        "cardinality_denominator", "validation_front", "validation_hv",
        "hv_history", "converged_gen", "convergence_evaluation_count",
        "logical_evaluation_count", "unique_subset_count",
        "memoization_hit_count", "physical_model_fit_count",
        "cache_hit_count", "search_runtime_s",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("NSGA-II validation-front artifact schema is invalid")
    search_features = _nsga2_search_features(
        variant, original_features, retained_features
    )
    for field, expected_value in (
        ("schema_version", 1),
        ("seed", context.job.seed),
        ("population", population),
        ("evaluations", evaluations),
        ("max_gen", max_gen),
        ("cardinality_denominator", len(original_features)),
    ):
        _exact_nonnegative_int(
            value[field], expected_value, name=f"NSGA-II {field}"
        )
    if (
        value["protocol_hash"] != context.job.protocol_hash
        or value["job_id"] != job.job_id
        or value["dataset_id"] != context.job.dataset_id
        or value["variant"] != variant
        or value["task"] != dataset.task
        or value["utility"] != dataset.utility
        or value["ref_point"] != list(_NSGA2_REF_POINT)
        or value["original_features"] != list(original_features)
        or value["search_features"] != list(search_features)
    ):
        raise ValueError("NSGA-II validation-front artifact identity is invalid")
    entries = _validate_nsga2_front_entries(
        value["validation_front"],
        search_features=search_features,
        denominator=len(original_features),
        objective_name="validation",
    )
    points = np.asarray([entry["objectives"] for entry in entries], dtype=float)
    if set(_non_dominated_sort(points)[0].tolist()) != set(range(len(entries))):
        raise ValueError("NSGA-II validation front contains a dominated point")
    expected_hv = _compute_hv(points, np.asarray(_NSGA2_REF_POINT))
    validation_hv = _finite_nonnegative(
        value["validation_hv"], name="NSGA-II validation hypervolume"
    )
    if validation_hv != expected_hv:
        raise ValueError("NSGA-II validation hypervolume is invalid")
    history = value["hv_history"]
    if (
        not isinstance(history, list)
        or len(history) != max_gen
        or any(_finite_nonnegative(item, name="NSGA-II HV history value") != item for item in history)
    ):
        raise ValueError("NSGA-II HV history is invalid")
    converged_gen = value["converged_gen"]
    if (
        isinstance(converged_gen, bool)
        or not isinstance(converged_gen, int)
        or converged_gen != _converged_gen(history)
    ):
        raise ValueError("NSGA-II convergence generation is invalid")
    _exact_nonnegative_int(
        value["convergence_evaluation_count"],
        population * (converged_gen + 1),
        name="NSGA-II convergence evaluation count",
    )
    logical_count = _exact_nonnegative_int(
        value["logical_evaluation_count"],
        evaluations,
        name="NSGA-II logical evaluation count",
    )
    unique_count = value["unique_subset_count"]
    memoization_hits = value["memoization_hit_count"]
    physical_fits = value["physical_model_fit_count"]
    cache_hits = value["cache_hit_count"]
    for name, count in (
        ("unique subset count", unique_count),
        ("memoization hit count", memoization_hits),
        ("physical model-fit count", physical_fits),
        ("cache-hit count", cache_hits),
    ):
        _exact_nonnegative_int(count, count, name=f"NSGA-II {name}")
    if (
        logical_count != unique_count + memoization_hits
        or physical_fits > unique_count
        or cache_hits > unique_count
        or physical_fits + cache_hits > unique_count
    ):
        raise ValueError("NSGA-II validation-front accounting is invalid")
    _finite_nonnegative(value["search_runtime_s"], name="NSGA-II search runtime")
    return value


def _freeze_nsga2_front(
    context: StageContext,
    *,
    job: JobSpec,
    original_features: tuple[str, ...],
    retained_features: tuple[str, ...],
    X_train: object,
    y_train: object,
    X_val: object,
    y_val: object,
    data_row: Mapping[str, object],
) -> dict[str, object]:
    import numpy as np

    variant_context = _nsga2_job_context(context, job)
    relpath = _nsga2_front_relpath(variant_context, job.method)
    path = context.layout.root / relpath
    if path.exists():
        return _validate_nsga2_front_artifact(
            variant_context,
            read_canonical_json(path, context="NSGA-II validation-front artifact"),
            variant=job.method,
            original_features=original_features,
            retained_features=retained_features,
        )
    dataset, _, population, evaluations, max_gen = _nsga2_authority(variant_context)
    search_features = _nsga2_search_features(
        job.method, original_features, retained_features
    )
    with _open_selection_cache(
        variant_context,
        (X_train, y_train, X_val, y_val),
        data_row,
    ) as cache:
        cache_before = cache.stats
        result = nsga2_feature_selection(
            X_train.loc[:, list(search_features)],
            y_train,
            X_val.loc[:, list(search_features)],
            y_val,
            task=dataset.task,
            utility=dataset.utility,
            model_params=context.protocol.model_params,
            backend=context.protocol.backend,
            seed=context.job.seed,
            pop_size=population,
            max_gen=max_gen,
            ref_point=_NSGA2_REF_POINT,
            verbose=False,
            cardinality_denominator=len(original_features),
            utility_evaluator=lambda selected: cache.evaluate(
                set(selected), dataset.utility
            ),
        )
        cache_after = cache.stats
    physical_model_fit_count = (
        cache_after.physical_model_fit_count
        - cache_before.physical_model_fit_count
    )
    cache_hit_count = (
        cache_after.cache_hit_count - cache_before.cache_hit_count
    )
    entries = []
    for mask, objectives in result.pareto_front:
        bits = np.asarray(mask, dtype=int).tolist()
        selected = [
            feature
            for feature, bit in zip(search_features, bits, strict=True)
            if bit
        ]
        entries.append(
            {
                "mask": bits,
                "selected_features": selected,
                "objectives": [float(objectives[0]), float(objectives[1])],
            }
        )
    document = {
        "schema_version": 1,
        "protocol_hash": context.job.protocol_hash,
        "job_id": job.job_id,
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "variant": job.method,
        "task": dataset.task,
        "utility": dataset.utility,
        "population": population,
        "evaluations": evaluations,
        "max_gen": max_gen,
        "ref_point": list(_NSGA2_REF_POINT),
        "original_features": list(original_features),
        "search_features": list(search_features),
        "cardinality_denominator": len(original_features),
        "validation_front": entries,
        "validation_hv": float(result.hv),
        "hv_history": [float(value) for value in result.hv_history],
        "converged_gen": int(result.converged_gen),
        "convergence_evaluation_count": population * (result.converged_gen + 1),
        "logical_evaluation_count": int(result.logical_evaluation_count),
        "unique_subset_count": int(result.unique_subset_count),
        "memoization_hit_count": int(result.memoization_hit_count),
        "physical_model_fit_count": physical_model_fit_count,
        "cache_hit_count": cache_hit_count,
        "search_runtime_s": float(result.runtime_s),
    }
    _validate_nsga2_front_artifact(
        variant_context,
        document,
        variant=job.method,
        original_features=original_features,
        retained_features=retained_features,
    )
    _publish_idempotent_json(path, document, context="NSGA-II validation-front artifact")
    return document


def _ensure_nsga2_fronts(
    context: StageContext,
) -> tuple[
    dict[str, dict[str, object]],
    tuple[object, object, object, object, Mapping[str, object]],
    ResultRecord,
]:
    _, jobs, _, _, _ = _nsga2_authority(context)
    plan, removal_record, retained = _nsga2_removal_authority(context)
    original = tuple(plan["features"])
    if not original:
        raise ValueError("NSGA-II requires a nonempty frozen prepared feature set")
    X_train, y_train, X_val, y_val, row = _load_prepared_train_validation(context)
    if tuple(X_train.columns) != original or tuple(X_val.columns) != original:
        raise ValueError("NSGA-II prepared feature order conflicts with removal plan")
    documents: dict[str, dict[str, object]] = {}
    for job in jobs:
        documents[job.method] = _freeze_nsga2_front(
            context,
            job=job,
            original_features=original,
            retained_features=retained,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            data_row=row,
        )
    # Re-read and validate both published fronts before any audited test access.
    for job in jobs:
        variant_context = _nsga2_job_context(context, job)
        documents[job.method] = _validate_nsga2_front_artifact(
            variant_context,
            read_canonical_json(
                context.layout.root / _nsga2_front_relpath(variant_context, job.method),
                context="NSGA-II validation-front artifact",
            ),
            variant=job.method,
            original_features=original,
            retained_features=retained,
        )
    return documents, (X_train, y_train, X_val, y_val, row), removal_record


def _validate_nsga2_result(
    context: StageContext,
    value: object,
    *,
    front: Mapping[str, object],
    minppi_selection_time: float,
) -> dict[str, object]:
    import numpy as np

    dataset, _, population, evaluations, _ = _nsga2_authority(context)
    expected = {
        "schema_version", "protocol_hash", "job_id", "dataset_id", "seed",
        "variant", "task", "utility", "population", "evaluation_count",
        "logical_evaluation_count", "unique_subset_count",
        "memoization_hit_count", "physical_model_fit_count",
        "cache_hit_count",
        "ref_point", "original_dimension", "search_dimension",
        "dimensional_reduction", "validation_hv", "test_hv",
        "validation_front", "test_front", "convergence_evaluation_count",
        "final_front_evaluation_count", "final_front_unique_subset_count",
        "final_front_memoization_hit_count", "evaluation_model_fit_count",
        "evaluation_cache_hit_count",
        "minppi_selection_time_s", "search_runtime_s", "evaluation_runtime_s",
        "total_runtime_s",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("NSGA-II result artifact schema is invalid")
    original = tuple(front["original_features"])
    search = tuple(front["search_features"])
    expected_selection_time = (
        minppi_selection_time if context.job.method == "minppi_prefilter" else 0.0
    )
    expected_reduction = (len(original) - len(search)) / len(original)
    expected_integer_fields = {
        "schema_version": 1,
        "seed": context.job.seed,
        "population": population,
        "evaluation_count": evaluations,
        "logical_evaluation_count": evaluations,
        "unique_subset_count": int(front["unique_subset_count"]),
        "memoization_hit_count": int(front["memoization_hit_count"]),
        "physical_model_fit_count": int(front["physical_model_fit_count"]),
        "cache_hit_count": int(front["cache_hit_count"]),
        "original_dimension": len(original),
        "search_dimension": len(search),
        "convergence_evaluation_count": int(front["convergence_evaluation_count"]),
        "final_front_evaluation_count": len(front["validation_front"]),
        "final_front_unique_subset_count": len(
            {
                tuple(entry["selected_features"])
                for entry in front["validation_front"]
            }
        ),
        "final_front_memoization_hit_count": (
            len(front["validation_front"])
            - len(
                {
                    tuple(entry["selected_features"])
                    for entry in front["validation_front"]
                }
            )
        ),
    }
    for field, expected_value in expected_integer_fields.items():
        _exact_nonnegative_int(
            value[field], expected_value, name=f"NSGA-II result {field}"
        )
    for field in (
        "evaluation_model_fit_count",
        "evaluation_cache_hit_count",
    ):
        _exact_nonnegative_int(
            value[field], value[field], name=f"NSGA-II result {field}"
        )
    if (
        value["evaluation_model_fit_count"]
        > value["final_front_unique_subset_count"]
        or value["evaluation_cache_hit_count"]
        > value["final_front_unique_subset_count"]
        or value["evaluation_model_fit_count"]
        + value["evaluation_cache_hit_count"]
        > value["final_front_unique_subset_count"]
    ):
        raise ValueError("NSGA-II final-front accounting is invalid")
    for field in (
        "dimensional_reduction",
        "validation_hv",
        "test_hv",
        "minppi_selection_time_s",
        "search_runtime_s",
        "evaluation_runtime_s",
        "total_runtime_s",
    ):
        _finite_nonnegative(value[field], name=f"NSGA-II result {field}")
    if (
        value["protocol_hash"] != context.job.protocol_hash
        or value["job_id"] != context.job.job_id
        or value["dataset_id"] != context.job.dataset_id
        or value["variant"] != context.job.method
        or value["task"] != dataset.task
        or value["utility"] != dataset.utility
        or value["ref_point"] != list(_NSGA2_REF_POINT)
        or value["dimensional_reduction"] != expected_reduction
        or value["validation_hv"] != front["validation_hv"]
        or value["validation_front"] != front["validation_front"]
        or value["minppi_selection_time_s"] != expected_selection_time
        or value["search_runtime_s"] != front["search_runtime_s"]
    ):
        raise ValueError("NSGA-II result artifact conflicts with its frozen front")
    test_entries = _validate_nsga2_front_entries(
        value["test_front"],
        search_features=search,
        denominator=len(original),
        objective_name="test",
    )
    if len(test_entries) != len(front["validation_front"]):
        raise ValueError("NSGA-II test front is incomplete")
    for validation_entry, test_entry in zip(
        front["validation_front"], test_entries, strict=True
    ):
        if (
            test_entry["mask"] != validation_entry["mask"]
            or test_entry["selected_features"] != validation_entry["selected_features"]
            or test_entry["objectives"][0] != validation_entry["objectives"][0]
        ):
            raise ValueError("NSGA-II test front changed a frozen mask")
    expected_hv = _compute_hv(
        np.asarray([entry["objectives"] for entry in test_entries], dtype=float),
        np.asarray(_NSGA2_REF_POINT),
    )
    if value["test_hv"] != expected_hv:
        raise ValueError("NSGA-II held-out test hypervolume is invalid")
    evaluation_runtime = _finite_nonnegative(
        value["evaluation_runtime_s"], name="NSGA-II evaluation runtime"
    )
    total_runtime = _finite_nonnegative(
        value["total_runtime_s"], name="NSGA-II total runtime"
    )
    if total_runtime != (
        expected_selection_time + float(front["search_runtime_s"]) + evaluation_runtime
    ):
        raise ValueError("NSGA-II total runtime accounting is invalid")
    return value


def _nsga2_result_record(
    context: StageContext,
    *,
    started_at: str,
    relpath: str,
    document: Mapping[str, object],
) -> ResultRecord:
    metric_keys = {
        "population", "evaluation_count", "original_dimension", "search_dimension",
        "logical_evaluation_count", "unique_subset_count",
        "memoization_hit_count", "physical_model_fit_count",
        "cache_hit_count",
        "dimensional_reduction", "validation_hv", "test_hv",
        "convergence_evaluation_count", "final_front_evaluation_count",
        "final_front_unique_subset_count",
        "final_front_memoization_hit_count", "evaluation_model_fit_count",
        "evaluation_cache_hit_count", "minppi_selection_time_s",
        "search_runtime_s", "evaluation_runtime_s", "total_runtime_s",
    }
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics={key: value for key, value in document.items() if key in metric_keys},
        artifacts={"results": relpath},
        started_at=started_at,
        finished_at=_timestamp(),
    )


def run_nsga2_job(context: StageContext) -> ResultRecord:
    """Freeze both fair validation fronts, then evaluate this job's front once."""
    import numpy as np
    import pandas as pd

    started_at = _timestamp()
    documents, prepared, removal_record = _ensure_nsga2_fronts(context)
    front = documents[context.job.method]
    minppi_selection_time = float(
        removal_record.metrics["minppi_selection_time_s"]
    )
    relpath = _nsga2_result_relpath(context)
    path = context.layout.root / relpath
    if path.exists():
        existing = _validate_nsga2_result(
            context,
            read_canonical_json(path, context="NSGA-II result artifact"),
            front=front,
            minppi_selection_time=minppi_selection_time,
        )
        return _nsga2_result_record(
            context, started_at=started_at, relpath=relpath, document=existing
        )
    X_train, y_train, X_val, y_val, row = prepared
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])
    dataset, _, population, evaluations, _ = _nsga2_authority(context)
    evaluation_started = time.perf_counter()
    test_front = []
    utility_by_subset: dict[tuple[str, ...], float] = {}
    with _open_final_test_cache(
        context,
        (X_train_val, y_train_val),
        None,
        row,
        reason="final_nsga2_front_evaluation",
    ) as cache:
        cache_before = cache.stats
        for entry in front["validation_front"]:
            selected = tuple(entry["selected_features"])
            if selected not in utility_by_subset:
                utility_by_subset[selected] = cache.evaluate(
                    set(selected), dataset.utility
                )
            utility = utility_by_subset[selected]
            test_front.append(
                {
                    "mask": entry["mask"],
                    "selected_features": entry["selected_features"],
                    "objectives": [
                        entry["objectives"][0],
                        1.0 - float(np.clip(utility, 0.0, 1.0)),
                    ],
                }
            )
        cache_after = cache.stats
    evaluation_model_fit_count = (
        cache_after.physical_model_fit_count
        - cache_before.physical_model_fit_count
    )
    evaluation_cache_hit_count = (
        cache_after.cache_hit_count - cache_before.cache_hit_count
    )
    evaluation_runtime = time.perf_counter() - evaluation_started
    test_hv = _compute_hv(
        np.asarray([entry["objectives"] for entry in test_front], dtype=float),
        np.asarray(_NSGA2_REF_POINT),
    )
    original_dimension = len(front["original_features"])
    search_dimension = len(front["search_features"])
    selection_time = (
        minppi_selection_time if context.job.method == "minppi_prefilter" else 0.0
    )
    total_runtime = selection_time + float(front["search_runtime_s"]) + evaluation_runtime
    document = {
        "schema_version": 1,
        "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id,
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "variant": context.job.method,
        "task": dataset.task,
        "utility": dataset.utility,
        "population": population,
        "evaluation_count": evaluations,
        "logical_evaluation_count": front["logical_evaluation_count"],
        "unique_subset_count": front["unique_subset_count"],
        "memoization_hit_count": front["memoization_hit_count"],
        "physical_model_fit_count": front["physical_model_fit_count"],
        "cache_hit_count": front["cache_hit_count"],
        "ref_point": list(_NSGA2_REF_POINT),
        "original_dimension": original_dimension,
        "search_dimension": search_dimension,
        "dimensional_reduction": (
            original_dimension - search_dimension
        ) / original_dimension,
        "validation_hv": front["validation_hv"],
        "test_hv": float(test_hv),
        "validation_front": front["validation_front"],
        "test_front": test_front,
        "convergence_evaluation_count": front["convergence_evaluation_count"],
        "final_front_evaluation_count": len(front["validation_front"]),
        "final_front_unique_subset_count": len(utility_by_subset),
        "final_front_memoization_hit_count": (
            len(front["validation_front"]) - len(utility_by_subset)
        ),
        "evaluation_model_fit_count": evaluation_model_fit_count,
        "evaluation_cache_hit_count": evaluation_cache_hit_count,
        "minppi_selection_time_s": selection_time,
        "search_runtime_s": front["search_runtime_s"],
        "evaluation_runtime_s": evaluation_runtime,
        "total_runtime_s": total_runtime,
    }
    _validate_nsga2_result(
        context,
        document,
        front=front,
        minppi_selection_time=minppi_selection_time,
    )
    _publish_idempotent_json(path, document, context="NSGA-II result artifact")
    return _nsga2_result_record(
        context, started_at=started_at, relpath=relpath, document=document
    )


_SUMMARY_REQUIRED_STAGES = ("estimator", "removal", "alpha", "nsga2", "ablation")
_SUMMARY_FILES = {
    "raw": "raw_metrics.csv",
    "aggregate": "aggregate_metrics.csv",
    "paired": "paired_comparisons.csv",
    "front": "plot_sources.csv",
    "latex_macros": "latex_macros.tex",
    "latex_macros_provenance": "latex_macros_provenance.json",
    "bundle_manifest": "summary_bundle_manifest.json",
}
_FORMAL_SEEDS = (42, 123, 456, 789, 1024)


def _summary_job_context(context: StageContext, job: JobSpec) -> StageContext:
    return StageContext(
        context.protocol, context.layout, job, context.status, context.audit
    )


def _numeric_metrics(
    *,
    context: StageContext,
    sequence: int,
    method: str,
    parameter: str,
    trial: int,
    metrics: Mapping[str, object],
) -> list[dict[str, object]]:
    import math

    rows: list[dict[str, object]] = []
    for name in sorted(metrics):
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("summary metric must be finite")
        rows.append(
            {
                "run_kind": context.protocol.run_kind,
                "manifest_sequence": sequence,
                "stage": context.job.stage,
                "dataset_id": context.job.dataset_id,
                "seed": context.job.seed,
                "method": method,
                "trial": trial,
                "parameter": parameter,
                "metric": name,
                "value": numeric,
            }
        )
    return rows


def _validated_estimator_summary_metrics(
    metrics: Mapping[str, object],
    *,
    approximate_scores: Mapping[str, int],
    exact_scores: Mapping[str, int] | None,
    budget: int,
    model_fit_count: int,
    coalition_count: int,
    execution_mode: str = "legacy",
) -> dict[str, object]:
    agreement_keys = {
        "harmful_precision", "harmful_recall", "harmful_f1", "score_agreement"
    }
    validated = validate_estimator_reporting_schema(
        metrics,
        execution_mode=execution_mode,
        exact_scores_available=exact_scores is not None,
    )
    if (
        metrics["budget"] != budget
        or metrics["model_fit_count"] != model_fit_count
        or metrics["coalition_count"] != coalition_count
    ):
        raise ValueError("estimator ResultRecord metrics conflict with score artifact")
    runtime = metrics["runtime_s"]
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not __import__("math").isfinite(float(runtime))
        or runtime < 0
    ):
        raise ValueError("estimator ResultRecord runtime is invalid")
    published = dict(validated)
    if exact_scores is not None:
        recomputed = _agreement_metrics(approximate_scores, exact_scores)
        if agreement_keys <= set(metrics) and any(
            metrics[key] != recomputed[key] for key in agreement_keys
        ):
            raise ValueError("estimator ResultRecord agreement metrics are invalid")
        published.update(recomputed)
    return published


def _validated_utility_ablation_summary_metrics(
    metrics: Mapping[str, object],
    *,
    value: str,
    document: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        "axis", "value", "budget", "shared_group_id",
        "shared_group_model_fit_count", "shared_group_coalition_count",
    }
    if set(metrics) != fields:
        raise ValueError("utility ablation ResultRecord metric schema is invalid")
    if metrics["axis"] != "utility" or metrics["value"] != value or any(
        metrics[field] != document[field]
        for field in fields - {"axis", "value"}
    ):
        raise ValueError("utility ablation ResultRecord metrics conflict with artifact")
    return dict(metrics)


def _validated_single_ablation_summary_metrics(
    metrics: Mapping[str, object],
    *,
    axis: str,
    value: str,
    document: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        "axis", "value", "budget", "model_fit_count", "coalition_count",
        "logical_evaluation_count", "logical_unique_coalition_count",
        "physical_model_fit_count", "cache_hit_count",
    }
    if set(metrics) != fields:
        raise ValueError("single ablation ResultRecord metric schema is invalid")
    if metrics["axis"] != axis or metrics["value"] != value or any(
        metrics[field] != document[field]
        for field in fields - {"axis", "value"}
    ):
        raise ValueError("single ablation ResultRecord metrics conflict with artifact")
    return dict(metrics)


def _ablation_score_outcomes(
    document: Mapping[str, object],
    *,
    utility_value: str | None = None,
) -> dict[str, float | int]:
    scores = document.get("scores")
    if utility_value is not None:
        if not isinstance(scores, Mapping) or utility_value not in scores:
            raise ValueError("utility ablation selected score map is invalid")
        scores = scores[utility_value]
    if not isinstance(scores, list) or not scores:
        raise ValueError("ablation selected score map is invalid")
    values: list[int] = []
    for entry in scores:
        score = entry.get("score") if isinstance(entry, Mapping) else None
        if isinstance(score, bool) or not isinstance(score, int) or score not in {-1, 0, 1}:
            raise ValueError("ablation selected score map is invalid")
        values.append(score)
    harmful_count = values.count(-1)
    return {
        "harmful_count": harmful_count,
        "harmful_fraction": harmful_count / len(values),
        "neutral_count": values.count(0),
        "beneficial_count": values.count(1),
    }


def _artifact_bound_record_metrics(
    record: ResultRecord,
    document: Mapping[str, object],
    fields: set[str],
    *,
    context: str,
) -> dict[str, object]:
    if set(record.metrics) != fields:
        raise ValueError(f"{context} ResultRecord metric schema is invalid")
    if any(record.metrics[field] != document[field] for field in fields):
        raise ValueError(f"{context} ResultRecord metrics conflict with artifact")
    return dict(record.metrics)


def _validate_summary_ablation_artifact(
    context: StageContext,
    record: ResultRecord,
    expected_features: tuple[str, ...],
) -> dict[str, object]:
    """Match Task 8's single-ablation artifact contract without split access."""
    axis = str(context.job.parameters["axis"])
    value = str(context.job.parameters["value"])
    budget = max(context.protocol.estimator_budgets)
    if axis == "utility":
        group_id = f"utility:{context.job.dataset_id}:s{context.job.seed}:b{budget}"
        relpath = (
            f"ablation/{context.job.dataset_id}/s{context.job.seed}/"
            f"utility/paired/b{budget}/scores.json"
        )
        if record.artifacts != {"paired_scores": relpath}:
            raise ValueError("utility ablation artifact path is invalid")
        document = _read_utility_pair_artifact(
            context, context.layout.root / relpath, budget, group_id
        )
        return document
    relpath = (
        f"ablation/{context.job.dataset_id}/s{context.job.seed}/"
        f"{axis}/{value}/b{budget}/scores.json"
    )
    if record.artifacts != {"scores": relpath}:
        raise ValueError("ablation score artifact path is invalid")
    document = read_canonical_json(
        context.layout.root / relpath, context=f"{axis} ablation artifact"
    )
    expected_method = "mi_correlation_guided" if axis == "guidance" else value
    checkpoint, _request, _schedules = _ablation_estimator_authority(
        context,
        expected_method,
        expected_features,
    )
    scheduled = checkpoint.schedule_digest is not None
    expected_keys = {
        "protocol_hash", "job_id", "dataset_id", "seed", "axis", "value",
        "method", "budget", "schedule_seed", "scores", "model_fit_count",
        "coalition_count", "logical_evaluation_count",
        "logical_unique_coalition_count", "physical_model_fit_count",
        "cache_hit_count",
    }
    if scheduled:
        expected_keys.add("schedule_digest")
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise ValueError("ablation score artifact schema is invalid")
    identity = {
        "protocol_hash": context.job.protocol_hash,
        "job_id": context.job.job_id,
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "axis": axis,
        "value": value,
        "method": expected_method,
        "budget": budget,
        "schedule_seed": context.job.seed,
    }
    if any(document[field] != expected for field, expected in identity.items()):
        raise ValueError("ablation score artifact identity is invalid")
    scores = document["scores"]
    if not isinstance(scores, list) or len(scores) != len(expected_features):
        raise ValueError("ablation score artifact scores are invalid")
    features: list[str] = []
    for entry in scores:
        score = entry.get("score") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or set(entry) != {"feature", "score"}
            or not isinstance(entry["feature"], str)
            or isinstance(score, bool)
            or not isinstance(score, int)
            or score not in {-1, 0, 1}
        ):
            raise ValueError("ablation score artifact scores are invalid")
        features.append(entry["feature"])
    if tuple(features) != expected_features or len(set(features)) != len(features):
        raise ValueError("ablation score artifact feature order is invalid")
    counter_fields = (
        "model_fit_count", "coalition_count", "logical_evaluation_count",
        "logical_unique_coalition_count", "physical_model_fit_count",
        "cache_hit_count",
    )
    counters = {field: document[field] for field in counter_fields}
    if (
        any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in counters.values()
        )
        or counters["model_fit_count"] > counters["coalition_count"]
        or counters["logical_unique_coalition_count"]
        > counters["logical_evaluation_count"]
        or counters["physical_model_fit_count"]
        > counters["logical_unique_coalition_count"]
        or counters["cache_hit_count"]
        > counters["logical_evaluation_count"]
    ):
        raise ValueError("ablation score artifact counters are invalid")
    if any(
        record.metrics.get(field) != document[field]
        for field in {"budget", *counter_fields}
    ) or record.metrics.get("axis") != axis or record.metrics.get("value") != value:
        raise ValueError("ablation ResultRecord metrics conflict with score artifact")
    if scheduled:
        expected_digest = checkpoint.schedule_digest
        if document["schedule_digest"] != expected_digest:
            raise ValueError("ablation score artifact schedule digest is invalid")
    return document


def _upstream_skip_metadata(
    context: StageContext,
) -> tuple[list[dict[str, object]], set[str]]:
    skipped: list[dict[str, object]] = []
    omitted: set[str] = set()
    for job in context.status.manifest:
        if job.stage not in _SUMMARY_REQUIRED_STAGES:
            continue
        entry = context.status.entry(job.job_id)
        if entry["state"] != "skipped":
            continue
        reason = entry["skip_reason"]
        if not isinstance(reason, str) or not reason:
            raise ValueError("approved skipped job has no deterministic reason")
        skipped.append(
            {
                "job_id": job.job_id,
                "stage": job.stage,
                "dataset_id": job.dataset_id,
                "seed": job.seed,
                "method": job.method,
                "reason": reason,
            }
        )
        omitted.add(job.job_id)
    return skipped, omitted


def _validated_summary_rows(
    context: StageContext,
    *,
    omitted_job_ids: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Extract values only after strict canonical stage-artifact validation."""
    raw_rows: list[dict[str, object]] = []
    front_rows: list[dict[str, object]] = []
    grouped_estimator_entries: list[
        tuple[int, StageContext, dict[str, object]]
    ] = []
    omitted = set() if omitted_job_ids is None else set(omitted_job_ids)
    for sequence, job in enumerate(context.status.manifest):
        if job.stage not in _SUMMARY_REQUIRED_STAGES:
            continue
        if job.job_id in omitted:
            continue
        item = _summary_job_context(context, job)
        record = read_result_record(context.layout.root / job.output_relpath, job)
        parameter = ""
        if job.stage == "estimator":
            parameter = f"budget={job.parameters['budget']}"
            budget = int(job.parameters["budget"])
            expected_relpath = (
                f"estimator/{job.dataset_id}/s{job.seed}/"
                f"{job.method}/b{budget}/scores.json"
            )
            if record.artifacts != {"scores": expected_relpath}:
                raise ValueError("estimator score artifact path is invalid")
            features = tuple(
                json.loads(str(_task9_manifest_row(item)["selected_features_json"]))
            )
            approximate_scores = _read_score_artifact(
                context.layout.root / expected_relpath,
                expected_job=job,
                budget=budget,
                expected_features=features,
                expected_metrics=record.metrics,
            )
            exact_jobs = tuple(
                candidate for candidate in context.status.manifest
                if candidate.stage == "estimator"
                and candidate.dataset_id == job.dataset_id
                and candidate.seed == job.seed
                and candidate.method == "exact"
                and candidate.parameters.get("budget") == budget
            )
            if len(exact_jobs) > 1:
                raise ValueError("estimator manifest has duplicate exact authority")
            exact_scores = None
            if exact_jobs and context.status.entry(exact_jobs[0].job_id)["state"] == "success":
                exact_job = exact_jobs[0]
                exact_record = read_result_record(
                    context.layout.root / exact_job.output_relpath, exact_job
                )
                exact_relpath = (
                    f"estimator/{exact_job.dataset_id}/s{exact_job.seed}/"
                    f"exact/b{budget}/scores.json"
                )
                if exact_record.artifacts != {"scores": exact_relpath}:
                    raise ValueError("exact estimator score artifact path is invalid")
                exact_scores = _read_score_artifact(
                    context.layout.root / exact_relpath,
                    expected_job=exact_job,
                    budget=budget,
                    expected_features=features,
                    expected_metrics=exact_record.metrics,
                )
            validated_metrics = _validated_estimator_summary_metrics(
                record.metrics,
                approximate_scores=approximate_scores,
                exact_scores=exact_scores,
                budget=budget,
                model_fit_count=int(record.metrics["model_fit_count"]),
                coalition_count=int(record.metrics["coalition_count"]),
                execution_mode=context.protocol.execution.mode,
            )
            if context.protocol.execution.mode == "grouped_nested_v1":
                grouped_estimator_entries.append(
                    (sequence, item, validated_metrics)
                )
            projection_metrics = estimator_projection_reporting_metrics(
                validated_metrics,
                execution_mode=context.protocol.execution.mode,
            )
            raw_rows.extend(
                _numeric_metrics(
                    context=item, sequence=sequence, method=job.method,
                    parameter=parameter, trial=0, metrics=projection_metrics,
                )
            )
            continue
        if job.stage == "ablation":
            parameter = f"{job.parameters['axis']}={job.parameters['value']}"
            features = tuple(
                json.loads(str(_task9_manifest_row(item)["selected_features_json"]))
            )
            document = _validate_summary_ablation_artifact(item, record, features)
            validated_metrics = (
                _validated_utility_ablation_summary_metrics(
                    record.metrics,
                    value=str(job.parameters["value"]),
                    document=document,
                )
                if job.parameters["axis"] == "utility"
                else _validated_single_ablation_summary_metrics(
                    record.metrics,
                    axis=str(job.parameters["axis"]),
                    value=str(job.parameters["value"]),
                    document=document,
                )
            )
            validated_metrics.update(
                _ablation_score_outcomes(
                    document,
                    utility_value=(
                        str(job.parameters["value"])
                        if job.parameters["axis"] == "utility"
                        else None
                    ),
                )
            )
            raw_rows.extend(
                _numeric_metrics(
                    context=item, sequence=sequence, method=job.method,
                    parameter=parameter, trial=0, metrics=validated_metrics,
                )
            )
            continue
        if job.stage == "removal":
            if job.method == "selection_plan":
                plan = _read_removal_plan(item)
                if record.artifacts != {
                    "selection_plan": _removal_plan_relpath(item)
                }:
                    raise ValueError("removal plan result artifact path is invalid")
                if int(plan["k"]) < 0:
                    raise ValueError("validated removal plan k is invalid")
                continue
            relpath = _removal_result_relpath(item)
            if record.artifacts != {"results": relpath}:
                raise ValueError("removal result artifact path is invalid")
            document = _validate_removal_result(
                item,
                read_canonical_json(
                    context.layout.root / relpath,
                    context="removal result artifact",
                ),
            )
            _artifact_bound_record_metrics(
                record,
                document,
                {
                    "k", "selection_time_s", "evaluation_time_s",
                    "selection_model_fit_count", "selection_coalition_count",
                    "evaluation_model_fit_count",
                },
                context="removal",
            )
            for row in document["rows"]:
                raw_rows.extend(
                    _numeric_metrics(
                        context=item, sequence=sequence, method=job.method,
                        parameter="", trial=(0 if row["trial"] is None else int(row["trial"])),
                        metrics={
                            key: row[key]
                            for key in (
                                "k", "full_utility", "selected_utility",
                                "delta_utility", "compression", "selection_time_s",
                                "evaluation_time_s", "selection_model_fit_count",
                                "selection_coalition_count", "evaluation_model_fit_count",
                            )
                        },
                    )
                )
            continue
        if job.stage == "alpha":
            if job.method == "selection_plan":
                _read_alpha_plan(item)
                if record.artifacts != {"selection_plan": _alpha_plan_relpath(item)}:
                    raise ValueError("alpha plan result artifact path is invalid")
                continue
            alpha = float(job.parameters["alpha"])
            relpath = _alpha_result_relpath(item, alpha)
            if record.artifacts != {"results": relpath}:
                raise ValueError("alpha result artifact path is invalid")
            document = _validate_alpha_result(
                item,
                read_canonical_json(
                    context.layout.root / relpath, context="alpha result artifact"
                ),
            )
            _artifact_bound_record_metrics(
                record,
                document,
                {
                    "harmful_count", "selection_time_s", "evaluation_time_s",
                    "selection_model_fit_count", "selection_coalition_count",
                    "selection_logical_evaluation_count",
                    "selection_logical_unique_coalition_count",
                    "selection_physical_model_fit_count",
                    "selection_cache_hit_count",
                    "evaluation_model_fit_count",
                },
                context="alpha",
            )
            raw_rows.extend(
                _numeric_metrics(
                    context=item, sequence=sequence, method=job.method,
                    parameter=f"alpha={format(alpha, '.17g')}", trial=0,
                    metrics={
                        key: document[key]
                        for key in (
                            "harmful_count", "compression", "validation_utility",
                            "test_utility", "full_test_utility", "delta_utility",
                            "preservation", "selection_time_s", "evaluation_time_s",
                            "selection_model_fit_count", "selection_coalition_count",
                            "evaluation_model_fit_count",
                        )
                    },
                )
            )
            continue
        if job.stage == "nsga2":
            plan, removal_record, retained = _nsga2_removal_authority(item)
            original = tuple(plan["features"])
            front_relpath = _nsga2_front_relpath(item, job.method)
            front = _validate_nsga2_front_artifact(
                item,
                read_canonical_json(
                    context.layout.root / front_relpath,
                    context="NSGA-II validation-front artifact",
                ),
                variant=job.method,
                original_features=original,
                retained_features=retained,
            )
            result_relpath = _nsga2_result_relpath(item)
            if record.artifacts != {"results": result_relpath}:
                raise ValueError("NSGA-II result artifact path is invalid")
            document = _validate_nsga2_result(
                item,
                read_canonical_json(
                    context.layout.root / result_relpath,
                    context="NSGA-II result artifact",
                ),
                front=front,
                minppi_selection_time=float(
                    removal_record.metrics["minppi_selection_time_s"]
                ),
            )
            _artifact_bound_record_metrics(
                record,
                document,
                {
                    "population", "evaluation_count", "original_dimension",
                    "search_dimension", "dimensional_reduction", "validation_hv",
                    "test_hv", "convergence_evaluation_count",
                    "final_front_evaluation_count", "evaluation_model_fit_count",
                    "minppi_selection_time_s", "search_runtime_s",
                    "evaluation_runtime_s", "total_runtime_s",
                },
                context="NSGA-II",
            )
            raw_rows.extend(
                _numeric_metrics(
                    context=item, sequence=sequence, method=job.method,
                    parameter="", trial=0,
                    metrics={
                        key: document[key]
                        for key in (
                            "population", "evaluation_count", "original_dimension",
                            "search_dimension", "dimensional_reduction", "validation_hv",
                            "test_hv", "convergence_evaluation_count",
                            "final_front_evaluation_count", "evaluation_model_fit_count",
                            "minppi_selection_time_s", "search_runtime_s",
                            "evaluation_runtime_s", "total_runtime_s",
                        )
                    },
                )
            )
            for front_kind, entries, hypervolume in (
                ("validation", document["validation_front"], document["validation_hv"]),
                ("test", document["test_front"], document["test_hv"]),
            ):
                for point_index, entry in enumerate(entries):
                    front_rows.append(
                        {
                            "run_kind": context.protocol.run_kind,
                            "dataset_id": job.dataset_id,
                            "seed": job.seed,
                            "method": job.method,
                            "front_kind": front_kind,
                            "point_index": point_index,
                            "compression": 1.0 - float(entry["objectives"][0]),
                            "utility": 1.0 - float(entry["objectives"][1]),
                            "evaluation": 0,
                            "hypervolume": float(hypervolume),
                        }
                    )
            population = int(document["population"])
            for point_index, hypervolume in enumerate(front["hv_history"]):
                front_rows.append(
                    {
                        "run_kind": context.protocol.run_kind,
                        "dataset_id": job.dataset_id,
                        "seed": job.seed,
                        "method": job.method,
                        "front_kind": "history",
                        "point_index": point_index,
                        "compression": 0.0,
                        "utility": 0.0,
                        "evaluation": population * (point_index + 2),
                        "hypervolume": float(hypervolume),
                    }
                )
    if grouped_estimator_entries:
        accounting = deduplicate_estimator_group_metrics(
            [metrics for _, _, metrics in grouped_estimator_entries]
        )
        for group in accounting["groups"]:
            digest = str(group["group_request_digest"])
            sequence, item, _ = next(
                entry
                for entry in grouped_estimator_entries
                if entry[2]["group_request_digest"] == digest
            )
            raw_rows.extend(
                _numeric_metrics(
                    context=item,
                    sequence=sequence,
                    method="__deduplicated_group_physical__",
                    parameter=f"group_request_digest={digest}",
                    trial=0,
                    metrics={
                        field: group[field]
                        for field in (
                            "group_physical_model_fit_count",
                            "group_cache_hit_count",
                            "group_runtime_s",
                        )
                    },
                )
            )
    return raw_rows, front_rows


def _validate_frozen_bundle_bytes(
    destination: Path,
    payloads: Mapping[str, bytes],
    *,
    conflict_message: str,
) -> None:
    if not destination.is_dir() or destination.is_symlink():
        raise RuntimeError(conflict_message)
    entries = tuple(destination.iterdir())
    if (
        {path.name for path in entries} != set(payloads)
        or any(not path.is_file() or path.is_symlink() for path in entries)
        or any(
            (destination / name).read_bytes() != payload
            for name, payload in payloads.items()
        )
    ):
        raise RuntimeError(conflict_message)


def _publish_bundle(
    parent: Path,
    directory_name: str,
    payloads: Mapping[str, bytes],
) -> Path:
    """Install an immutable directory as one atomic publication unit."""
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / directory_name
    if destination.exists():
        _validate_frozen_bundle_bytes(
            destination,
            payloads,
            conflict_message=(
                "frozen publication bundle conflicts with recomputed content"
            ),
        )
        return destination
    staging = parent / f".{directory_name}.{uuid4().hex}.tmp"
    staging.mkdir()
    try:
        for name, payload in payloads.items():
            _atomic_write_bytes(staging / name, payload)
        try:
            os.replace(staging, destination)
        except FileExistsError:
            _validate_frozen_bundle_bytes(
                destination,
                payloads,
                conflict_message="concurrent publication bundle conflicts",
            )
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _summary_publication_payloads(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    macros: str,
    provenance: Mapping[str, object],
    run_kind: str,
    skipped_jobs: Sequence[Mapping[str, object]],
    publication_allowed: bool,
) -> tuple[dict[str, bytes], str]:
    columns = {
        "raw": RAW_COLUMNS,
        "aggregate": AGGREGATE_COLUMNS,
        "paired": PAIRED_COLUMNS,
        "front": FRONT_COLUMNS,
    }
    payloads = {
        _SUMMARY_FILES[key]: csv_bytes(tables[key], columns[key])
        for key in ("raw", "aggregate", "paired", "front")
    }
    payloads[_SUMMARY_FILES["latex_macros"]] = macros.encode("utf-8")
    payloads[_SUMMARY_FILES["latex_macros_provenance"]] = provenance_bytes(
        provenance
    )
    row_counts = {
        _SUMMARY_FILES["raw"]: len(tables["raw"]),
        _SUMMARY_FILES["aggregate"]: len(tables["aggregate"]),
        _SUMMARY_FILES["paired"]: len(tables["paired"]),
        _SUMMARY_FILES["front"]: len(tables["front"]),
        _SUMMARY_FILES["latex_macros"]: sum(
            bool(line) for line in macros.splitlines()
        ),
        _SUMMARY_FILES["latex_macros_provenance"]: len(
            provenance.get("macros", {})
        ),
    }
    file_columns = {
        _SUMMARY_FILES[key]: list(columns[key])
        for key in ("raw", "aggregate", "paired", "front")
    }
    manifest = {
        "schema_version": 1,
        "run_kind": run_kind,
        "publication_allowed": publication_allowed,
        "skipped_jobs": [dict(row) for row in skipped_jobs],
        "files": [
            {
                "filename": filename,
                "schema_version": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_length": len(payload),
                "row_count": row_counts[filename],
                "columns": file_columns.get(filename, []),
            }
            for filename, payload in payloads.items()
        ],
    }
    manifest_payload = provenance_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    payloads[_SUMMARY_FILES["bundle_manifest"]] = manifest_payload
    return payloads, manifest_hash


def _validate_summary_bundle(
    bundle: Path,
    *,
    expected_manifest_sha256: str,
    expected_run_kind: str,
) -> dict[str, object]:
    root = Path(bundle)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("summary bundle is not a regular directory")
    expected_names = set(_SUMMARY_FILES.values())
    entries = tuple(root.iterdir())
    if (
        {path.name for path in entries} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise ValueError("summary bundle entries are incomplete or unsafe")
    manifest_path = root / _SUMMARY_FILES["bundle_manifest"]
    manifest_payload = manifest_path.read_bytes()
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest_sha256:
        raise ValueError("summary bundle manifest hash is invalid")
    manifest = read_canonical_json(
        manifest_path, context="summary bundle manifest"
    )
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {
            "schema_version", "run_kind", "publication_allowed",
            "skipped_jobs", "files",
        }
        or manifest["schema_version"] != 1
        or manifest["run_kind"] != expected_run_kind
        or not isinstance(manifest["publication_allowed"], bool)
        or not isinstance(manifest["skipped_jobs"], list)
        or not isinstance(manifest["files"], list)
    ):
        raise ValueError("summary bundle manifest schema is invalid")
    skip_keys = {"job_id", "stage", "dataset_id", "seed", "method", "reason"}
    if any(
        not isinstance(entry, dict)
        or set(entry) != skip_keys
        or not isinstance(entry["job_id"], str)
        or not isinstance(entry["stage"], str)
        or not isinstance(entry["dataset_id"], str)
        or isinstance(entry["seed"], bool)
        or not isinstance(entry["seed"], int)
        or entry["seed"] < 0
        or not isinstance(entry["method"], str)
        or not isinstance(entry["reason"], str)
        or not entry["reason"]
        for entry in manifest["skipped_jobs"]
    ) or len({entry["job_id"] for entry in manifest["skipped_jobs"]}) != len(
        manifest["skipped_jobs"]
    ):
        raise ValueError("summary bundle skipped-job metadata is invalid")
    expected_data_names = [
        _SUMMARY_FILES[key]
        for key in (
            "raw", "aggregate", "paired", "front", "latex_macros",
            "latex_macros_provenance",
        )
    ]
    if [entry.get("filename") for entry in manifest["files"]] != expected_data_names:
        raise ValueError("summary bundle manifest file order is invalid")
    expected_columns = {
        _SUMMARY_FILES["raw"]: list(RAW_COLUMNS),
        _SUMMARY_FILES["aggregate"]: list(AGGREGATE_COLUMNS),
        _SUMMARY_FILES["paired"]: list(PAIRED_COLUMNS),
        _SUMMARY_FILES["front"]: list(FRONT_COLUMNS),
    }
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {
            "filename", "schema_version", "sha256", "byte_length",
            "row_count", "columns",
        } or entry["schema_version"] != 1:
            raise ValueError("summary bundle file metadata schema is invalid")
        filename = entry["filename"]
        path = root / filename
        payload = path.read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != entry["sha256"]
            or len(payload) != entry["byte_length"]
            or b"\r" in payload
        ):
            raise ValueError("summary bundle file hash, length, or canonical bytes are invalid")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("summary bundle file is not UTF-8") from exc
        if filename.endswith(".csv"):
            rows = list(csv.reader(io.StringIO(text, newline="")))
            if not rows or rows[0] != expected_columns[filename]:
                raise ValueError("summary bundle CSV schema is invalid")
            actual_rows = len(rows) - 1
        elif filename == _SUMMARY_FILES["latex_macros"]:
            actual_rows = sum(bool(line) for line in text.splitlines())
        else:
            document = read_canonical_json(path, context="macro provenance")
            actual_rows = len(document.get("macros", {})) if isinstance(document, dict) else -1
        if actual_rows != entry["row_count"] or entry["columns"] != expected_columns.get(filename, []):
            raise ValueError("summary bundle row count or columns are invalid")
    return manifest


_FIGURE_FILENAMES = (
    "fig1_overview.pdf", "fig2_cd_heatmap.pdf", "fig3_prefilter.pdf"
)
_FIGURE_MANIFEST = "figure_bundle_manifest.json"


def _figure_publication_payloads(
    generated: Sequence[Path],
) -> tuple[dict[str, bytes], str]:
    paths = tuple(Path(path) for path in generated)
    if tuple(path.name for path in paths) != _FIGURE_FILENAMES or any(
        not path.is_file() or path.is_symlink() for path in paths
    ):
        raise ValueError("figure generation must return the exact three PDF filenames")
    payloads = {path.name: path.read_bytes() for path in paths}
    if any(not payload.startswith(b"%PDF-") for payload in payloads.values()):
        raise ValueError("figure output is not a PDF")
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "filename": filename,
                "sha256": hashlib.sha256(payloads[filename]).hexdigest(),
                "byte_length": len(payloads[filename]),
            }
            for filename in _FIGURE_FILENAMES
        ],
    }
    manifest_payload = provenance_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    payloads[_FIGURE_MANIFEST] = manifest_payload
    return payloads, manifest_hash


def _validate_figure_bundle(
    bundle: Path, *, expected_manifest_sha256: str
) -> dict[str, object]:
    root = Path(bundle)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("figure bundle is not a regular directory")
    entries = tuple(root.iterdir())
    expected_names = {*_FIGURE_FILENAMES, _FIGURE_MANIFEST}
    if (
        {path.name for path in entries} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise ValueError("figure bundle does not contain the exact regular PDF set")
    manifest_path = root / _FIGURE_MANIFEST
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
        raise ValueError("figure bundle manifest hash is invalid")
    manifest = read_canonical_json(manifest_path, context="figure bundle manifest")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "files"}
        or manifest["schema_version"] != 1
        or not isinstance(manifest["files"], list)
        or [entry.get("filename") for entry in manifest["files"]]
        != list(_FIGURE_FILENAMES)
    ):
        raise ValueError("figure bundle manifest schema is invalid")
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {
            "filename", "sha256", "byte_length"
        }:
            raise ValueError("figure bundle file metadata is invalid")
        pdf = (root / entry["filename"]).read_bytes()
        if (
            not pdf.startswith(b"%PDF-")
            or hashlib.sha256(pdf).hexdigest() != entry["sha256"]
            or len(pdf) != entry["byte_length"]
        ):
            raise ValueError("figure PDF hash or length is invalid")
    return manifest


def run_summarize_job(context: StageContext) -> ResultRecord:
    """Validate all upstream jobs before extracting or computing any statistic."""
    started_at = _timestamp()
    manifest = context.status.manifest
    assert_stage_complete(context.layout, manifest, _SUMMARY_REQUIRED_STAGES)
    if context.protocol.run_kind == "formal" and context.protocol.seeds != _FORMAL_SEEDS:
        raise ValueError("formal summary requires seeds (42, 123, 456, 789, 1024)")
    if (
        context.job.stage != "summarize"
        or context.job not in manifest
        or context.job.dataset_id != "__run__"
        or context.job.seed != 0
        or context.job.method != "aggregate_results"
        or context.job.parameters.get("required_stages") != list(_SUMMARY_REQUIRED_STAGES)
    ):
        raise ValueError("summarize job conflicts with frozen run-level authority")
    skipped_jobs, omitted_job_ids = _upstream_skip_metadata(context)
    raw_rows, front_rows = _validated_summary_rows(
        context, omitted_job_ids=omitted_job_ids
    )
    tables = aggregate_canonical_records(
        raw_rows,
        front_rows,
        allow_approved_skip_subset=(
            context.protocol.run_kind == "formal" and bool(skipped_jobs)
        ),
    )
    publication_allowed = not (
        context.protocol.run_kind == "formal" and bool(skipped_jobs)
    )
    if publication_allowed:
        macros, provenance = generate_latex_macros(
            tables["aggregate"], tables["paired"], run_kind=context.protocol.run_kind
        )
    else:
        macros = ""
        provenance = {
            "schema_version": 1,
            "run_kind": context.protocol.run_kind,
            "publication_allowed": False,
            "skipped_jobs": skipped_jobs,
            "macros": {},
        }
    payloads, bundle_manifest_sha256 = _summary_publication_payloads(
        tables,
        macros=macros,
        provenance=provenance,
        run_kind=context.protocol.run_kind,
        skipped_jobs=skipped_jobs,
        publication_allowed=publication_allowed,
    )
    _publish_bundle(context.layout.tables, "summary", payloads)
    artifacts = {
        key: f"tables/summary/{filename}" for key, filename in _SUMMARY_FILES.items()
    }
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics={
            "raw_row_count": len(tables["raw"]),
            "aggregate_row_count": len(tables["aggregate"]),
            "paired_row_count": len(tables["paired"]),
            "front_row_count": len(tables["front"]),
            "run_kind": context.protocol.run_kind,
            "skipped_job_count": len(skipped_jobs),
            "publication_allowed": publication_allowed,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "bundle_manifest_byte_length": len(
                payloads[_SUMMARY_FILES["bundle_manifest"]]
            ),
        },
        artifacts=artifacts,
        started_at=started_at,
        finished_at=_timestamp(),
    )


def _validate_summarize_result_record(
    context: StageContext, record: ResultRecord
) -> dict[str, object]:
    expected_artifacts = {
        key: f"tables/summary/{filename}" for key, filename in _SUMMARY_FILES.items()
    }
    expected_metrics = {
        "raw_row_count", "aggregate_row_count", "paired_row_count",
        "front_row_count", "run_kind", "skipped_job_count",
        "publication_allowed", "bundle_manifest_sha256",
        "bundle_manifest_byte_length",
    }
    if record.artifacts != expected_artifacts or set(record.metrics) != expected_metrics:
        raise ValueError("summarize ResultRecord schema is invalid")
    digest = record.metrics["bundle_manifest_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or record.metrics["run_kind"] != context.protocol.run_kind
        or not isinstance(record.metrics["publication_allowed"], bool)
    ):
        raise ValueError("summarize ResultRecord identity metrics are invalid")
    bundle = context.layout.root / "tables/summary"
    manifest = _validate_summary_bundle(
        bundle,
        expected_manifest_sha256=digest,
        expected_run_kind=context.protocol.run_kind,
    )
    manifest_bytes = (bundle / _SUMMARY_FILES["bundle_manifest"]).read_bytes()
    if record.metrics["bundle_manifest_byte_length"] != len(manifest_bytes):
        raise ValueError("summarize bundle manifest length conflicts with result")
    row_counts = {
        entry["filename"]: entry["row_count"] for entry in manifest["files"]
    }
    expected_counts = {
        "raw_row_count": row_counts[_SUMMARY_FILES["raw"]],
        "aggregate_row_count": row_counts[_SUMMARY_FILES["aggregate"]],
        "paired_row_count": row_counts[_SUMMARY_FILES["paired"]],
        "front_row_count": row_counts[_SUMMARY_FILES["front"]],
        "skipped_job_count": len(manifest["skipped_jobs"]),
    }
    if any(record.metrics[key] != value for key, value in expected_counts.items()):
        raise ValueError("summarize ResultRecord counts conflict with bundle")
    if record.metrics["publication_allowed"] != manifest["publication_allowed"]:
        raise ValueError("summarize publication state conflicts with bundle")
    return manifest


def _validate_figures_result_record(
    context: StageContext, record: ResultRecord
) -> None:
    summary_jobs = tuple(
        job for job in context.status.manifest if job.stage == "summarize"
    )
    if len(summary_jobs) != 1:
        raise ValueError("figures validation requires one summarize job")
    summary_job = summary_jobs[0]
    summary_record = read_result_record(
        context.layout.root / summary_job.output_relpath, summary_job
    )
    _validate_summarize_result_record(
        _summary_job_context(context, summary_job), summary_record
    )
    current_summary_digest = summary_record.metrics["bundle_manifest_sha256"]
    suppressed = record.metrics.get("publication_suppressed")
    if suppressed is True:
        if set(record.metrics) != {
            "figure_count", "run_kind", "publication_suppressed",
            "summary_bundle_manifest_sha256",
        } or record.artifacts or record.metrics["figure_count"] != 0 or record.metrics[
            "summary_bundle_manifest_sha256"
        ] != current_summary_digest or record.metrics["run_kind"] != context.protocol.run_kind:
            raise ValueError("suppressed figures ResultRecord schema is invalid")
        return
    expected_metrics = {
        "figure_count", "run_kind", "publication_suppressed",
        "summary_bundle_manifest_sha256", "bundle_manifest_sha256",
        "bundle_manifest_byte_length",
    }
    expected_artifacts = {
        Path(filename).stem: f"figures/paper/{filename}"
        for filename in _FIGURE_FILENAMES
    }
    expected_artifacts["bundle_manifest"] = f"figures/paper/{_FIGURE_MANIFEST}"
    if (
        set(record.metrics) != expected_metrics
        or record.artifacts != expected_artifacts
        or record.metrics["figure_count"] != 3
        or record.metrics["run_kind"] != context.protocol.run_kind
        or record.metrics["publication_suppressed"] is not False
        or record.metrics["summary_bundle_manifest_sha256"] != current_summary_digest
    ):
        raise ValueError("figures ResultRecord schema is invalid")
    digest = record.metrics["bundle_manifest_sha256"]
    if not isinstance(digest, str):
        raise ValueError("figure bundle manifest hash is invalid")
    bundle = context.layout.figures / "paper"
    _validate_figure_bundle(bundle, expected_manifest_sha256=digest)
    if record.metrics["bundle_manifest_byte_length"] != len(
        (bundle / _FIGURE_MANIFEST).read_bytes()
    ):
        raise ValueError("figure bundle manifest length conflicts with result")


def _validate_task11_result_record(context: StageContext, record: ResultRecord) -> None:
    if context.job.stage == "summarize":
        _validate_summarize_result_record(context, record)
    elif context.job.stage == "figures":
        _validate_figures_result_record(context, record)


def run_figures_job(context: StageContext) -> ResultRecord:
    """Generate deterministic figures solely from explicit summary table paths."""
    started_at = _timestamp()
    manifest = context.status.manifest
    assert_stage_complete(context.layout, manifest, ("summarize",))
    if (
        context.job.stage != "figures"
        or context.job not in manifest
        or context.job.dataset_id != "__run__"
        or context.job.seed != 0
        or context.job.method != "generate_figures"
    ):
        raise ValueError("figures job conflicts with frozen run-level authority")
    summary_jobs = tuple(job for job in manifest if job.stage == "summarize")
    if len(summary_jobs) != 1:
        raise ValueError("figures requires exactly one canonical summarize job")
    summary = read_result_record(
        context.layout.root / summary_jobs[0].output_relpath, summary_jobs[0]
    )
    summary_context = _summary_job_context(context, summary_jobs[0])
    summary_manifest = _validate_summarize_result_record(summary_context, summary)
    summary_digest = str(summary.metrics["bundle_manifest_sha256"])
    if not summary_manifest["publication_allowed"]:
        return ResultRecord(
            job_id=context.job.job_id,
            protocol_hash=context.job.protocol_hash,
            stage=context.job.stage,
            dataset_id=context.job.dataset_id,
            seed=context.job.seed,
            method=context.job.method,
            metrics={
                "figure_count": 0,
                "run_kind": context.protocol.run_kind,
                "publication_suppressed": True,
                "summary_bundle_manifest_sha256": summary_digest,
            },
            artifacts={},
            started_at=started_at,
            finished_at=_timestamp(),
        )

    if __package__:
        from .make_paper_figures import generate_all_figures
    else:
        from make_paper_figures import generate_all_figures  # type: ignore[no-redef]

    staging = context.layout.figures / f".paper.{uuid4().hex}.tmp"
    staging.mkdir()
    try:
        generated = generate_all_figures(
            context.layout.root / summary.artifacts["raw"],
            context.layout.root / summary.artifacts["aggregate"],
            context.layout.root / summary.artifacts["paired"],
            context.layout.root / summary.artifacts["front"],
            staging,
            run_kind=context.protocol.run_kind,
        )
        payloads, bundle_manifest_sha256 = _figure_publication_payloads(generated)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _publish_bundle(context.layout.figures, "paper", payloads)
    artifacts = {
        Path(filename).stem: f"figures/paper/{filename}"
        for filename in _FIGURE_FILENAMES
    }
    artifacts["bundle_manifest"] = f"figures/paper/{_FIGURE_MANIFEST}"
    return ResultRecord(
        job_id=context.job.job_id,
        protocol_hash=context.job.protocol_hash,
        stage=context.job.stage,
        dataset_id=context.job.dataset_id,
        seed=context.job.seed,
        method=context.job.method,
        metrics={
            "figure_count": 3,
            "run_kind": context.protocol.run_kind,
            "publication_suppressed": False,
            "summary_bundle_manifest_sha256": summary_digest,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "bundle_manifest_byte_length": len(payloads[_FIGURE_MANIFEST]),
        },
        artifacts=artifacts,
        started_at=started_at,
        finished_at=_timestamp(),
    )


STAGE_EXECUTORS.update(
    {
        "validate": run_validate_job,
        "preprocess": run_preprocess_job,
        "estimator": run_estimator_job,
        "removal": run_removal_job,
        "alpha": run_alpha_job,
        "nsga2": run_nsga2_job,
        "ablation": run_ablation_job,
        "summarize": run_summarize_job,
        "figures": run_figures_job,
    }
)
