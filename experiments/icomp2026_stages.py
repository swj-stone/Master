"""Synchronous, manifest-authoritative ICOMP 2026 stage execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
        "registered_stages": sorted(required_executors),
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
    {"validate": run_validate_job, "preprocess": run_preprocess_job}
)
