"""Synchronous, manifest-authoritative ICOMP 2026 stage execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence
from uuid import uuid4

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


STAGE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "validate": (),
    "preprocess": ("validate",),
    "estimator": ("preprocess",),
    "removal": ("preprocess",),
    "alpha": ("preprocess",),
    "nsga2": ("preprocess", "removal"),
    "ablation": ("preprocess",),
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
        decision = status.decide(job.job_id)
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


def run_estimator_job(context: StageContext) -> ResultRecord:
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
    )


def _read_score_artifact(
    path: Path,
    *,
    expected_job: JobSpec,
    budget: int,
    expected_features: tuple[str, ...],
) -> dict[str, int]:
    try:
        document = read_canonical_json(path, context="estimator score artifact")
    except ValueError as exc:
        raise ValueError(
            "estimator score artifact is missing, malformed, or noncanonical JSON"
        ) from exc
    expected_keys = {
        "protocol_hash", "job_id", "dataset_id", "seed", "method", "budget",
        "scores", "model_fit_count", "coalition_count",
    }
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
    from minppi_mofs.min_ppi import (
        algorithm_b_schedule_digest,
        estimate_minppi_mi_correlation_guided,
        make_algorithm_b_perturbation_schedule,
    )
    from minppi_mofs.shapg_integration import (
        build_mi_correlation_guidance,
        compute_mi_guidance_signal,
        get_correlation_neighbors,
    )

    started_at = _timestamp()
    value = context.job.parameters["value"]
    allowed = {"random", "mi", "correlation", "mi_correlation"}
    if value not in allowed:
        raise ValueError("guidance ablation value is not frozen")
    budget = max(context.protocol.estimator_budgets)
    X_train, y_train, X_val, y_val, _ = _load_prepared_train_validation(context)
    schedule = make_algorithm_b_perturbation_schedule(
        tuple(X_train.columns), K_adv=budget, seed=context.job.seed
    )
    schedule_digest = algorithm_b_schedule_digest(schedule)
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
    elif value == "mi_correlation":
        mi_scores, correlation_neighbors = build_mi_correlation_guidance(
            X_train, y_train, task=task, seed=context.job.seed
        )
    cache = _make_job_cache(context, X_train, X_val, y_train, y_val)
    scores = estimate_minppi_mi_correlation_guided(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        alpha=context.protocol.alpha_default,
        task=task,
        utility=str(context.job.parameters["utility"]),
        K_adv=budget,
        mi_scores=mi_scores,
        correlation_neighbors=correlation_neighbors,
        perturbation_schedule=schedule,
        acc_cache=cache,
        seed=context.job.seed,
    )
    return _publish_single_ablation(
        context,
        started_at=started_at,
        value=str(value),
        estimator_method="mi_correlation_guided",
        budget=budget,
        scores=scores,
        cache=cache,
        schedule_digest=schedule_digest,
    )


def _run_dispatcher_ablation(context: StageContext) -> ResultRecord:
    from minppi_mofs.min_ppi import (
        algorithm_b_schedule_digest,
        estimate_minppi_exact,
        estimate_minppi_hybrid,
        estimate_minppi_mi_correlation_guided,
        estimate_minppi_random_mc,
        make_algorithm_b_perturbation_schedule,
    )
    from minppi_mofs.shapg_integration import build_mi_correlation_guidance

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
    X_train, y_train, X_val, y_val, _ = _load_prepared_train_validation(context)
    task = str(context.job.parameters["task"])
    utility = str(context.job.parameters["utility"])
    cache = _make_job_cache(context, X_train, X_val, y_train, y_val)
    common = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "alpha": context.protocol.alpha_default,
        "task": task,
        "utility": utility,
        "acc_cache": cache,
        "seed": context.job.seed,
    }
    schedule_digest = None
    if value == "exact":
        scores = estimate_minppi_exact(**common)
    elif value == "random_mc":
        scores = estimate_minppi_random_mc(K=budget, **common)
    else:
        schedule = make_algorithm_b_perturbation_schedule(
            tuple(X_train.columns), K_adv=budget, seed=context.job.seed
        )
        schedule_digest = algorithm_b_schedule_digest(schedule)
        mi_scores, correlation_neighbors = build_mi_correlation_guidance(
            X_train, y_train, task=task, seed=context.job.seed
        )
        if value == "mi_correlation_guided":
            scores = estimate_minppi_mi_correlation_guided(
                K_adv=budget,
                mi_scores=mi_scores,
                correlation_neighbors=correlation_neighbors,
                perturbation_schedule=schedule,
                **common,
            )
        else:
            scores = estimate_minppi_hybrid(
                K=budget,
                K_adv=budget,
                mi_scores=mi_scores,
                correlation_neighbors=correlation_neighbors,
                perturbation_schedule=schedule,
                **common,
            )
    return _publish_single_ablation(
        context,
        started_at=started_at,
        value=str(value),
        estimator_method=str(value),
        budget=budget,
        scores=scores,
        cache=cache,
        schedule_digest=schedule_digest,
    )


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
    cache: object,
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
        "model_fit_count": cache.model_fit_count,
        "coalition_count": cache.coalition_count,
    }
    if schedule_digest is not None:
        document["schedule_digest"] = schedule_digest
    _publish_idempotent_json(
        context.layout.root / relpath,
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
            "model_fit_count": cache.model_fit_count,
            "coalition_count": cache.coalition_count,
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


STAGE_EXECUTORS.update(
    {
        "validate": run_validate_job,
        "preprocess": run_preprocess_job,
        "estimator": run_estimator_job,
        "ablation": run_ablation_job,
    }
)
