"""Frozen, validated configuration for the ICOMP 2026 experiment rerun.

This module contains protocol mechanics that do not require model fitting or
dataset loading.  The first public boundary is :func:`load_resolved_protocol`,
which merges the global experiment configuration with per-dataset YAML files
and rejects ambiguous or misspelled settings before a run can start.
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
import platform
import re
import stat
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd
import yaml


_VALID_TASKS = frozenset({"binary", "multiclass", "regression"})
_CLASSIFICATION_UTILITIES = frozenset(
    {"roc_auc", "ovr_auc", "accuracy", "pr_auc"}
)
_REGRESSION_UTILITIES = frozenset({"r2", "rmse"})
_VALID_BACKENDS = frozenset({"cpu", "cuda"})
_VALID_ESTIMATORS = frozenset(
    {"exact", "random_mc", "mi_correlation_guided", "hybrid"}
)
_CANONICAL_REMOVAL_CONTROLS = (
    "full",
    "minppi",
    "shap_bottom",
    "minshap_bottom",
    "banzhaf_bottom",
    "random_matched",
)
_CANONICAL_NSGA2_VARIANTS = ("full", "minppi_prefilter")
_STAGE_ORDER = (
    "validate",
    "preprocess",
    "estimator",
    "removal",
    "alpha",
    "nsga2",
    "ablation",
    "summarize",
    "figures",
)
_RUN_LEVEL_DATASET_ID = "__run__"
_ALLOWED_TEST_ACCESS = frozenset(
    {
        ("removal", "final_removal_evaluation"),
        ("alpha", "final_alpha_grid_evaluation"),
        ("nsga2", "final_nsga2_front_evaluation"),
    }
)
_AUDIT_SCHEMA_VERSION = 1
_DATA_MANIFEST_FIELDS = (
    "protocol_hash", "job_id", "dataset_id", "seed", "task", "utility",
    "target", "raw_file_sha256", "raw_row_count", "raw_feature_count",
    "selected_feature_count", "selected_features_json", "train_relpath",
    "val_relpath", "test_relpath", "train_row_count", "val_row_count",
    "test_row_count", "train_index_sha256", "val_index_sha256",
    "test_index_sha256", "train_file_sha256", "val_file_sha256",
    "test_file_sha256", "train_label_summary_json", "val_label_summary_json",
    "test_label_summary_json", "transform_metadata_json",
)
_DATA_MANIFEST_INTEGER_FIELDS = frozenset(
    {
        "seed", "raw_row_count", "raw_feature_count", "selected_feature_count",
        "train_row_count", "val_row_count", "test_row_count",
    }
)
_DATA_MANIFEST_SHA_FIELDS = frozenset(
    {
        "raw_file_sha256", "train_index_sha256", "val_index_sha256",
        "test_index_sha256", "train_file_sha256", "val_file_sha256",
        "test_file_sha256",
    }
)

_GLOBAL_KEYS = frozenset(
    {
        "protocol_version",
        "datasets",
        "seeds",
        "backend",
        "model",
        "alpha",
        "estimator",
        "removal",
        "nsga2",
        "ablations",
        "smoke",
    }
)
_MODEL_KEYS = frozenset(
    {"n_estimators", "max_depth", "learning_rate", "n_jobs"}
)
_ALPHA_KEYS = frozenset({"default", "grid"})
_ESTIMATOR_KEYS = frozenset({"methods", "budgets", "exact_max_features"})
_REMOVAL_KEYS = frozenset({"controls", "random_repeats"})
_NSGA2_KEYS = frozenset(
    {"variants", "population", "evaluations", "ref_point"}
)
_ABLATION_KEYS = frozenset({"utility", "guidance", "dispatcher"})
_SMOKE_KEYS = frozenset(
    {
        "datasets",
        "seeds",
        "estimator_budgets",
        "random_repeats",
        "nsga2_population",
        "nsga2_evaluations",
    }
)
_DATASET_FILE_KEYS = frozenset({"dataset", "preprocessing", "split"})
_DATASET_KEYS = frozenset(
    {"id", "display_name", "raw_path", "target", "task", "utility"}
)
_PREPROCESSING_KEYS = frozenset({"max_features", "variance_threshold"})
_SPLIT_KEYS = frozenset({"train_frac", "val_frac", "test_frac"})


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset's fully resolved, explicit protocol settings."""

    dataset_id: str
    display_name: str
    raw_path: Path
    target: str
    task: str
    utility: str
    max_features: int | None
    variance_threshold: float
    train_frac: float
    val_frac: float
    test_frac: float


@dataclass(frozen=True)
class ResolvedProtocol:
    """Validated experiment settings after CLI and smoke overrides."""

    protocol_version: str
    run_kind: str
    datasets: tuple[DatasetSpec, ...]
    seeds: tuple[int, ...]
    backend: str
    model_params: dict[str, object]
    alpha_default: float
    alpha_grid: tuple[float, ...]
    estimator_methods: tuple[str, ...]
    estimator_budgets: tuple[int, ...]
    exact_max_features: int
    nsga2: dict[str, object]
    removal: dict[str, object]
    ablations: dict[str, object]
    smoke: dict[str, object]


@dataclass(frozen=True)
class JobSpec:
    """One deterministic unit of work in a frozen experiment manifest."""

    job_id: str
    protocol_hash: str
    stage: str
    dataset_id: str
    seed: int
    method: str
    parameters: dict[str, object]
    depends_on: tuple[str, ...]
    output_relpath: str


@dataclass(frozen=True)
class ResultRecord:
    """The complete, validated output of one manifest job."""

    job_id: str
    protocol_hash: str
    stage: str
    dataset_id: str
    seed: int
    method: str
    metrics: dict[str, float | int | str | bool | None]
    artifacts: dict[str, str]
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class RunLayout:
    """Canonical filesystem layout for one immutable experiment run."""

    root: Path
    configs_frozen: Path
    raw_metrics: Path
    tables: Path
    figures: Path
    audit_path: Path
    manifest_path: Path
    status_path: Path
    log_path: Path


@dataclass(frozen=True)
class ResumeDecision:
    """The deterministic action to take when resuming one manifest job."""

    action: str
    reason: str

    def __post_init__(self) -> None:
        if self.action not in {"execute", "skip"}:
            raise ValueError("resume decision action is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("resume decision reason is invalid")


@dataclass(frozen=True)
class CompletenessReport:
    """Validated, manifest-authoritative completion information."""

    required_stages: tuple[str, ...]
    complete_job_ids: tuple[str, ...]


class IncompleteRunError(RuntimeError):
    """Raised when one or more manifest jobs are not canonical and complete."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("incomplete run: " + "; ".join(self.issues))


class TestAccessViolation(PermissionError):
    """Raised when a stage attempts an unapproved test-partition read."""


@dataclass(frozen=True)
class PreparedSplitPaths:
    """Immutable CSV locations for one already-prepared dataset split."""

    dataset_id: str
    seed: int
    target: str
    train_csv: Path
    val_csv: Path
    test_csv: Path

    def __post_init__(self) -> None:
        _safe_identifier(self.dataset_id, "dataset_id")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.target, str) or not self.target or "\x00" in self.target:
            raise ValueError("target must be a non-empty CSV column name")
        parents: set[Path] = set()
        identities: dict[str, tuple[int, int] | None] = {}
        for field in ("train_csv", "val_csv", "test_csv"):
            raw_path = Path(getattr(self, field))
            if (
                not raw_path.name
                or raw_path.suffix.lower() != ".csv"
                or "\x00" in str(raw_path)
                or ".." in raw_path.parts
            ):
                raise ValueError(f"{field} is not a safe path")
            identity = _split_file_identity(raw_path, field, allow_missing=False)
            path = raw_path.resolve()
            parents.add(path.parent)
            object.__setattr__(self, field, path)
            identities[field] = identity
        if len(parents) != 1:
            raise ValueError("split CSV paths must share one prepared split directory")
        existing_identities = {value for value in identities.values() if value is not None}
        if (
            len({self.train_csv, self.val_csv, self.test_csv}) != 3
            or len(existing_identities)
            != sum(value is not None for value in identities.values())
        ):
            raise ValueError("train, validation, and test CSV paths must be distinct")
        object.__setattr__(self, "_identities", MappingProxyType(identities))


class AuditLog:
    """Append-only audit log; ``AuditLog(path)`` fails closed for test reads."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str | None = None,
    ) -> None:
        self._path = Path(path)
        if not self._path.name:
            raise ValueError("audit path is invalid")
        derived_run_id = self._path.parent.name
        self._run_id = _safe_identifier(derived_run_id, "audit run_id")
        if run_id is not None and _safe_identifier(run_id, "audit run_id") != self._run_id:
            raise ValueError("audit run_id must match the audit path parent")
        self._job_authority: Mapping[str, tuple[str, str, int]] | None = None
        self._manifest_hash: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    @classmethod
    def from_manifest(cls, path: Path, manifest: Sequence[JobSpec]) -> "AuditLog":
        """Bind test access to the complete trusted manifest authority."""
        jobs = _validated_status_manifest(manifest)
        audit = cls(path)
        authority = {
            job.job_id: (job.stage, job.dataset_id, job.seed) for job in jobs
        }
        object.__setattr__(audit, "_job_authority", MappingProxyType(authority))
        object.__setattr__(audit, "_manifest_hash", _status_manifest_hash(jobs))
        return audit

    def read_events(self) -> list[dict[str, object]]:
        """Return a deep copy of all persisted events without creating a file."""
        return deepcopy(self._read_document()["events"])

    def append_test_access(
        self,
        *,
        job_id: str,
        stage: str,
        reason: object,
        decision: str,
        dataset_id: str = "__audit__",
        seed: int = 0,
    ) -> dict[str, object]:
        """Persist one test-access decision before the CSV can be read."""
        _safe_identifier(job_id, "audit job_id")
        _safe_identifier(dataset_id, "audit dataset_id")
        _positive_or_zero_int(seed, "audit seed")
        _audit_text(stage, "audit stage")
        normalized_reason = _audit_reason(reason)
        if decision not in {"allowed", "denied"}:
            raise ValueError("audit decision is invalid")
        document = self._read_document()
        authority = None if self._job_authority is None else self._job_authority.get(job_id)
        decision = (
            "allowed"
            if (
                decision == "allowed"
                and (stage, normalized_reason) in _ALLOWED_TEST_ACCESS
                and authority == (stage, dataset_id, seed)
            )
            else "denied"
        )
        event: dict[str, object] = {
            "event_id": uuid4().hex,
            "event_type": "test_access",
            "run_id": self._run_id,
            "job_id": job_id,
            "stage": stage,
            "dataset_id": dataset_id,
            "seed": seed,
            "reason": normalized_reason,
            "decision": decision,
            "at": _utc_timestamp(),
        }
        document["events"].append(event)
        atomic_write_json(self._path, document)
        return deepcopy(event)

    def resolve_denied(
        self, denied_event_id: str, resolution: str
    ) -> dict[str, object]:
        """Append the explicit resolution of one currently unresolved denial."""
        _safe_identifier(denied_event_id, "denied audit event ID")
        if not isinstance(resolution, str) or not resolution.strip() or "\x00" in resolution:
            raise ValueError("resolution must be non-empty")
        document = self._read_document()
        events = document["events"]
        denied = next(
            (
                event
                for event in events
                if event["event_type"] == "test_access"
                and event["event_id"] == denied_event_id
                and event["decision"] == "denied"
                and event["run_id"] == self._run_id
            ),
            None,
        )
        if denied is None:
            raise ValueError("resolution must reference a real denied audit event")
        if any(
            event["event_type"] == "denied_resolution"
            and event["denied_event_id"] == denied_event_id
            for event in events
        ):
            raise ValueError("denied audit event is already resolved")
        event: dict[str, object] = {
            "event_id": uuid4().hex,
            "event_type": "denied_resolution",
            "run_id": self._run_id,
            "denied_event_id": denied_event_id,
            "resolution": resolution.strip(),
            "at": _utc_timestamp(),
        }
        events.append(event)
        atomic_write_json(self._path, document)
        return deepcopy(event)

    def integrity_issues(
        self,
        *,
        expected_run_id: str,
        allowed_job_ids: Sequence[str],
        expected_job_authority: Mapping[str, tuple[str, str, int]] | None = None,
    ) -> tuple[str, ...]:
        """Validate the on-disk trail against trusted run and manifest identity."""
        expected = _safe_identifier(expected_run_id, "expected audit run_id")
        allowlist = {_safe_identifier(job_id, "manifest job ID") for job_id in allowed_job_ids}
        authority = (
            None
            if expected_job_authority is None
            else {
                _safe_identifier(job_id, "manifest job ID"): (
                    _audit_text(details[0], "manifest job stage"),
                    _safe_identifier(details[1], "manifest dataset ID"),
                    _positive_or_zero_int(details[2], "manifest seed"),
                )
                for job_id, details in expected_job_authority.items()
            }
        )
        if authority is not None and set(authority) != allowlist:
            raise ValueError("audit manifest authority does not match the job allowlist")
        if not self._path.exists():
            return ()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return (f"audit malformed: {exc}",)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "events"}
            or not _exact_int(payload.get("schema_version"), _AUDIT_SCHEMA_VERSION)
            or not isinstance(payload.get("events"), list)
        ):
            return ("audit malformed: invalid document schema",)
        issues: list[str] = []
        valid: list[dict[str, object]] = []
        event_ids: set[str] = set()
        duplicate_event_ids: set[str] = set()
        for index, raw_event in enumerate(payload["events"]):
            try:
                event = _validated_audit_event(raw_event)
            except ValueError as exc:
                issues.append(f"audit event {index} malformed: {exc}")
                continue
            valid.append(event)
            event_id = event["event_id"]
            if event_id in event_ids:
                duplicate_event_ids.add(event_id)
                issues.append(f"audit event {index} has duplicate event ID")
            event_ids.add(event_id)
            if event["run_id"] != expected:
                issues.append(f"audit event {index} has foreign run ID")
            if event["event_type"] == "test_access":
                if event["job_id"] not in allowlist:
                    issues.append(f"audit event {index} has unknown job ID")
                elif authority is not None and (
                    event["stage"], event["dataset_id"], event["seed"]
                ) != authority[event["job_id"]]:
                    issues.append(
                        f"audit event {index} does not match manifest job authority"
                    )
                if (
                    event["decision"] == "allowed"
                    and (event["stage"], event["reason"]) not in _ALLOWED_TEST_ACCESS
                ):
                    issues.append(f"audit event {index} has forged allowed reason")
        denied_by_id = {
            event["event_id"]: event
            for event in valid
            if event["event_type"] == "test_access" and event["decision"] == "denied"
        }
        resolved_ids: set[str] = set()
        prior_denied: dict[str, dict[str, object]] = {}
        for index, event in enumerate(valid):
            if (
                event["event_type"] == "test_access"
                and event["decision"] == "denied"
            ):
                prior_denied[event["event_id"]] = event
                continue
            if event["event_type"] != "denied_resolution":
                continue
            denied_event_id = event["denied_event_id"]
            denied = prior_denied.get(denied_event_id)
            if denied is None:
                issues.append(
                    f"audit event {index} resolution references no prior denied event"
                )
            elif denied["run_id"] != event["run_id"]:
                issues.append(f"audit event {index} resolution has foreign denied event")
            elif denied_event_id in resolved_ids:
                issues.append(f"audit event {index} duplicate denied resolution")
            else:
                resolved_ids.add(denied_event_id)
        for event_id, event in denied_by_id.items():
            if (
                event["run_id"] == expected
                and (event_id not in resolved_ids or event_id in duplicate_event_ids)
            ):
                issues.append(f"audit denied event {event_id} is unresolved denied access")
        return tuple(issues)

    def _read_document(self) -> dict[str, object]:
        if not self._path.exists():
            return {"schema_version": _AUDIT_SCHEMA_VERSION, "events": []}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("audit log is malformed") from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "events"}
            or not _exact_int(payload.get("schema_version"), _AUDIT_SCHEMA_VERSION)
            or not isinstance(payload.get("events"), list)
        ):
            raise ValueError("audit log is malformed")
        events = [_validated_audit_event(event) for event in payload["events"]]
        return {"schema_version": _AUDIT_SCHEMA_VERSION, "events": events}


class AuditedSplitAccess:
    """The sole supported boundary for split CSV reads during stage execution."""

    def __init__(
        self,
        paths: PreparedSplitPaths,
        *,
        stage: str,
        job_id: str,
        audit: AuditLog,
    ) -> None:
        if not isinstance(paths, PreparedSplitPaths):
            raise TypeError("paths must be PreparedSplitPaths")
        _audit_text(stage, "stage")
        _safe_identifier(job_id, "job_id")
        if not isinstance(audit, AuditLog):
            raise TypeError("audit must be an AuditLog")
        self._paths = paths
        self._stage = stage
        self._job_id = job_id
        self._audit = audit

    def load_train_validation(self):
        """Load fresh train/validation feature matrices and target arrays only."""
        X_train, y_train = _load_split_csv(
            self._paths.train_csv, self._paths.target, self._paths._identities["train_csv"]
        )
        X_val, y_val = _load_split_csv(
            self._paths.val_csv, self._paths.target, self._paths._identities["val_csv"]
        )
        return X_train, y_train, X_val, y_val

    def load_test(self, *, reason: object):
        """Audit first, then permit only one frozen final-evaluation purpose."""
        normalized_reason = _audit_reason(reason)
        decision = (
            "allowed"
            if (self._stage, normalized_reason) in _ALLOWED_TEST_ACCESS
            else "denied"
        )
        event = self._audit.append_test_access(
            job_id=self._job_id,
            stage=self._stage,
            reason=normalized_reason,
            decision=decision,
            dataset_id=self._paths.dataset_id,
            seed=self._paths.seed,
        )
        if event["decision"] == "denied":
            raise TestAccessViolation(
                f"test access denied for stage={self._stage!r}, reason={event['reason']!r}"
            )
        return _load_split_csv(
            self._paths.test_csv, self._paths.target, self._paths._identities["test_csv"]
        )


def _audit_text(value: object, context: str) -> str:
    if type(value) is not str or "\x00" in value:
        raise ValueError(f"{context} is invalid")
    return value


def _audit_reason(value: object) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        return "invalid_reason"
    return value


def _validated_audit_event(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not isinstance(value.get("event_type"), str):
        raise ValueError("event schema is invalid")
    event_type = value["event_type"]
    if event_type == "test_access":
        expected = {
            "event_id",
            "event_type",
            "run_id",
            "job_id",
            "stage",
            "dataset_id",
            "seed",
            "reason",
            "decision",
            "at",
        }
        if set(value) != expected:
            raise ValueError("test access event schema is invalid")
        _safe_identifier(value["event_id"], "audit event ID")
        _safe_identifier(value["run_id"], "audit run_id")
        _safe_identifier(value["job_id"], "audit job ID")
        _audit_text(value["stage"], "audit stage")
        _safe_identifier(value["dataset_id"], "audit dataset ID")
        _positive_or_zero_int(value["seed"], "audit seed")
        _audit_text(value["reason"], "audit reason")
        if value["decision"] not in {"allowed", "denied"}:
            raise ValueError("test access decision is invalid")
        _utc_status_timestamp(value["at"], "audit timestamp")
    elif event_type == "denied_resolution":
        expected = {
            "event_id",
            "event_type",
            "run_id",
            "denied_event_id",
            "resolution",
            "at",
        }
        if set(value) != expected:
            raise ValueError("denied resolution event schema is invalid")
        _safe_identifier(value["event_id"], "audit event ID")
        _safe_identifier(value["run_id"], "audit run_id")
        _safe_identifier(value["denied_event_id"], "denied audit event ID")
        resolution = value["resolution"]
        if not isinstance(resolution, str) or not resolution.strip() or "\x00" in resolution:
            raise ValueError("denied resolution is invalid")
        _utc_status_timestamp(value["at"], "audit timestamp")
    else:
        raise ValueError("audit event type is invalid")
    return dict(value)


def _split_file_identity(
    path: Path, context: str, *, allow_missing: bool
) -> tuple[int, int] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return None
        raise ValueError(f"{context} split file is missing") from None
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if path.is_symlink() or (reparse and attributes & reparse) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{context} split path must be a regular non-symlink file")
    return metadata.st_dev, metadata.st_ino


def _load_split_csv(
    path: Path, target: str, expected_identity: tuple[int, int] | None
):
    before = _split_file_identity(path, "prepared", allow_missing=False)
    if expected_identity is None or before != expected_identity:
        raise ValueError("prepared split file identity changed")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"could not open split CSV {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != expected_identity:
            raise ValueError("prepared split file identity changed after open")
        with os.fdopen(descriptor, "r", encoding="utf-8", newline="") as handle:
            descriptor = -1
            frame = pd.read_csv(handle)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"could not load split CSV {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if target not in frame.columns:
        raise ValueError(f"split CSV {path} is missing target column {target!r}")
    features = frame.drop(columns=[target]).copy(deep=True)
    targets = frame[target].to_numpy(copy=True)
    return features, targets


def canonical_hash(value: object) -> str:
    """Return a stable SHA-256 hash for a JSON-compatible value."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolved_protocol_hash(protocol: ResolvedProtocol) -> str:
    """Return the canonical identity of one resolved protocol."""
    if not isinstance(protocol, ResolvedProtocol):
        raise TypeError("protocol must be a ResolvedProtocol")
    return _protocol_hash(protocol)


def create_new_run(
    output_root: Path,
    run_id: str | None,
    protocol: ResolvedProtocol,
) -> RunLayout:
    """Create and return a new run directory without overwriting anything."""
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{_protocol_hash(protocol)[:10]}"
    root = _validated_run_root(output_root, run_id)

    layout = _run_layout(root)
    try:
        layout.root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"run directory already exists: {layout.root}"
        ) from exc
    for directory in (
        layout.configs_frozen,
        layout.raw_metrics,
        layout.tables,
        layout.figures,
    ):
        directory.mkdir()
    return layout


def capture_source_fingerprint(
    project_root: Path,
    protocol: ResolvedProtocol,
) -> dict[str, object]:
    """Hash experiment Python sources and record informational Git state."""
    project_root = Path(project_root).resolve()
    files: dict[str, str] = {}
    for source_root_name in ("src", "experiments"):
        source_root = project_root / source_root_name
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.py")):
            if path.is_file():
                relative = path.relative_to(project_root).as_posix()
                files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    git_commit = _git_output(project_root, "rev-parse", "HEAD")
    git_status = _git_output(
        project_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    return {
        "protocol_hash": _protocol_hash(protocol),
        "files": files,
        "git_commit": git_commit,
        "git_dirty": None if git_status is None else bool(git_status),
    }


def freeze_run_metadata(
    layout: RunLayout,
    protocol: ResolvedProtocol,
    *,
    project_root: Path,
) -> None:
    """Atomically freeze the protocol, environment, and source identity."""
    if layout != _run_layout(layout.root):
        raise ValueError("run layout must use the canonical paths")
    if not layout.root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {layout.root}")

    dataset_ids = tuple(
        _safe_identifier(dataset.dataset_id, "dataset_id")
        for dataset in protocol.datasets
    )
    _ensure_unique(dataset_ids, "duplicate dataset IDs")
    protocol_payload = _protocol_payload(protocol)
    resolved_yaml = yaml.safe_dump(
        protocol_payload,
        sort_keys=True,
        allow_unicode=True,
    )
    text_artifacts = {
        layout.root / "resolved_protocol.yaml": resolved_yaml,
        layout.root / "environment.json": _json_text(
            _capture_environment(protocol.backend)
        ),
        layout.root / "source_fingerprint.json": _json_text(
            capture_source_fingerprint(project_root, protocol)
        ),
        layout.configs_frozen
        / "icomp2026_pipeline.resolved.yaml": resolved_yaml,
    }
    for dataset, dataset_id in zip(protocol.datasets, dataset_ids, strict=True):
        dataset_payload = _dataset_payload(dataset)
        dataset_path = layout.configs_frozen / f"{dataset_id}.yaml"
        _require_direct_child(
            layout.configs_frozen,
            dataset_path,
            "dataset_id",
        )
        text_artifacts[dataset_path] = yaml.safe_dump(
            dataset_payload,
            sort_keys=True,
            allow_unicode=True,
        )

    marker_path = layout.root / "metadata_complete.json"
    existing_artifacts = [
        path for path in (*text_artifacts, marker_path) if path.exists()
    ]
    if existing_artifacts or any(layout.configs_frozen.iterdir()):
        raise FileExistsError(
            f"run metadata is already frozen or partially written: {layout.root}"
        )

    for path, text_value in text_artifacts.items():
        _atomic_write_text(path, text_value)
    artifact_hashes = {
        path.relative_to(layout.root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in text_artifacts
    }
    _atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "protocol_hash": _protocol_hash(protocol),
            "artifacts": artifact_hashes,
        },
    )


def open_resumed_run(
    output_root: Path,
    run_id: str,
    protocol: ResolvedProtocol,
    *,
    project_root: Path,
) -> RunLayout:
    """Open a run only when its immutable protocol and sources still match."""
    root = _validated_run_root(output_root, run_id)
    layout = _run_layout(root)
    if not layout.root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {layout.root}")

    _validate_frozen_metadata(layout, protocol)
    protocol_path = layout.root / "resolved_protocol.yaml"
    fingerprint_path = layout.root / "source_fingerprint.json"

    frozen_protocol = _read_yaml(protocol_path)
    if canonical_hash(frozen_protocol) != _protocol_hash(protocol):
        raise RuntimeError("resolved protocol drift prevents resume")

    try:
        frozen_fingerprint = json.loads(
            fingerprint_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("source fingerprint metadata is invalid") from exc
    if not isinstance(frozen_fingerprint, dict):
        raise RuntimeError("source fingerprint metadata is invalid")
    current_fingerprint = capture_source_fingerprint(project_root, protocol)
    if (
        frozen_fingerprint.get("protocol_hash")
        != current_fingerprint["protocol_hash"]
        or frozen_fingerprint.get("files") != current_fingerprint["files"]
    ):
        raise RuntimeError("source fingerprint drift prevents resume")

    required_directories = (
        layout.configs_frozen,
        layout.raw_metrics,
        layout.tables,
        layout.figures,
    )
    if not all(path.is_dir() for path in required_directories):
        raise RuntimeError("run layout is incomplete and cannot be resumed")
    return layout


def _run_layout(root: Path) -> RunLayout:
    root = Path(root)
    return RunLayout(
        root=root,
        configs_frozen=root / "configs_frozen",
        raw_metrics=root / "raw_metrics",
        tables=root / "tables",
        figures=root / "figures",
        audit_path=root / "protocol_audit.json",
        manifest_path=root / "run_manifest.csv",
        status_path=root / "status.json",
        log_path=root / "output.log",
    )


def _validated_run_root(output_root: Path, run_id: object) -> Path:
    _validate_run_id(run_id)
    if not isinstance(run_id, str):
        raise TypeError("validated run_id must be a string")
    output_root = Path(output_root)
    resolved_output_root = output_root.resolve()
    resolved_candidate = (resolved_output_root / run_id).resolve()
    if resolved_candidate.parent != resolved_output_root:
        raise ValueError("run_id must identify a direct child of output_root")
    return output_root / run_id


def _validate_run_id(run_id: object) -> None:
    if (
        not isinstance(run_id, str)
        or run_id in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is None
    ):
        raise ValueError(
            "run_id must contain only ASCII letters, digits, '.', '_', or '-'"
        )


def _validate_frozen_metadata(
    layout: RunLayout,
    protocol: ResolvedProtocol,
) -> None:
    marker_path = layout.root / "metadata_complete.json"
    if not marker_path.is_file():
        raise RuntimeError("run metadata is incomplete and cannot be resumed")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("metadata completion marker is invalid") from exc
    if not isinstance(marker, dict) or marker.get("schema_version") != 1:
        raise RuntimeError("metadata completion marker is invalid")
    if marker.get("protocol_hash") != _protocol_hash(protocol):
        raise RuntimeError("resolved protocol drift prevents resume")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or not all(
        isinstance(relative, str) and isinstance(digest, str)
        for relative, digest in artifacts.items()
    ):
        raise RuntimeError("metadata completion marker is invalid")

    dataset_ids = tuple(
        _safe_identifier(dataset.dataset_id, "dataset_id")
        for dataset in protocol.datasets
    )
    expected_artifacts = {
        "resolved_protocol.yaml",
        "environment.json",
        "source_fingerprint.json",
        "configs_frozen/icomp2026_pipeline.resolved.yaml",
        *(f"configs_frozen/{dataset_id}.yaml" for dataset_id in dataset_ids),
    }
    if set(artifacts) != expected_artifacts:
        raise RuntimeError("frozen metadata artifact set is invalid")
    for relative, expected_digest in artifacts.items():
        path = layout.root / relative
        _require_within(layout.root, path, "metadata artifact")
        if not path.is_file():
            raise RuntimeError(f"frozen metadata artifact is missing: {relative}")
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise RuntimeError(f"frozen metadata artifact was changed: {relative}")


def _require_direct_child(parent: Path, candidate: Path, context: str) -> None:
    if candidate.resolve().parent != parent.resolve():
        raise ValueError(f"{context} must be a safe path component")


def _require_within(parent: Path, candidate: Path, context: str) -> None:
    try:
        candidate.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{context} escapes the run directory") from exc


def _protocol_payload(protocol: ResolvedProtocol) -> dict[str, object]:
    payload = _json_safe(asdict(protocol))
    if not isinstance(payload, dict):
        raise TypeError("resolved protocol must serialize to a mapping")
    return payload


def _protocol_hash(protocol: ResolvedProtocol) -> str:
    return canonical_hash(_protocol_payload(protocol))


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _dataset_payload(dataset: DatasetSpec) -> dict[str, object]:
    return {
        "dataset": {
            "id": dataset.dataset_id,
            "display_name": dataset.display_name,
            "raw_path": str(dataset.raw_path),
            "target": dataset.target,
            "task": dataset.task,
            "utility": dataset.utility,
        },
        "preprocessing": {
            "max_features": dataset.max_features,
            "variance_threshold": dataset.variance_threshold,
        },
        "split": {
            "train_frac": dataset.train_frac,
            "val_frac": dataset.val_frac,
            "test_frac": dataset.test_frac,
        },
    }


def _capture_environment(backend: str) -> dict[str, object]:
    try:
        import xgboost as xgb

        raw_build_info = xgb.build_info()
        build_info: object = _json_safe(raw_build_info)
        cuda_build_info = {
            str(key): _json_safe(value)
            for key, value in raw_build_info.items()
            if "cuda" in str(key).lower() or "nccl" in str(key).lower()
        }
    except (ImportError, AttributeError, RuntimeError) as exc:
        build_info = {"unavailable": f"{type(exc).__name__}: {exc}"}
        cuda_build_info = {"unavailable": True}
    local_time = datetime.now().astimezone()
    offset = local_time.strftime("%z")
    utc_offset = f"{offset[:3]}:{offset[3:]}" if offset else "unavailable"
    return {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit_learn": _package_version("scikit-learn"),
        "xgboost": _package_version("xgboost"),
        "xgboost_build_info": build_info,
        "cuda_build_info": cuda_build_info,
        "cuda_driver": _nvidia_driver_version(),
        "os": platform.platform(),
        "timezone": local_time.tzname() or "unavailable",
        "utc_offset": utc_offset,
        "backend": backend,
    }


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _git_output(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    versions = sorted(set(completed.stdout.split()))
    return ",".join(versions) if versions else "unavailable"


def build_job_manifest(
    protocol: ResolvedProtocol,
    stages: tuple[str, ...],
    *,
    data_manifest: tuple[dict[str, object], ...] | list[dict[str, object]]
    | None = None,
) -> tuple[JobSpec, ...]:
    """Expand selected protocol stages into a stable, immutable job sequence."""
    requested_stages = _manifest_stages(stages)
    selected_features = (
        _selected_feature_counts(protocol, data_manifest)
        if {"estimator", "ablation"}.intersection(requested_stages)
        else None
    )
    protocol_hash = _protocol_hash(protocol)
    jobs: list[JobSpec] = []
    for stage in requested_stages:
        if stage == "validate":
            jobs.append(_make_job(protocol_hash, stage, _RUN_LEVEL_DATASET_ID, 0, "environment_validation", {"backend": protocol.backend}))
        elif stage == "preprocess":
            for dataset in protocol.datasets:
                for seed in protocol.seeds:
                    jobs.append(
                        _make_job(
                            protocol_hash,
                            stage,
                            dataset.dataset_id,
                            seed,
                            "prepare_split",
                            _dataset_parameters(dataset),
                        )
                    )
        elif stage == "estimator":
            if selected_features is None:
                raise AssertionError("estimator data manifest was not loaded")
            for dataset in protocol.datasets:
                for seed in protocol.seeds:
                    count = selected_features[(dataset.dataset_id, seed)]
                    for method in protocol.estimator_methods:
                        if method == "exact" and count > protocol.exact_max_features:
                            continue
                        for budget in protocol.estimator_budgets:
                            jobs.append(
                                _make_job(
                                    protocol_hash,
                                    stage,
                                    dataset.dataset_id,
                                    seed,
                                    method,
                                    _dataset_parameters(dataset, budget=budget),
                                )
                            )
        elif stage == "removal":
            for dataset in protocol.datasets:
                for seed in protocol.seeds:
                    plan = _make_job(
                        protocol_hash,
                        stage,
                        dataset.dataset_id,
                        seed,
                        "selection_plan",
                        _dataset_parameters(
                            dataset,
                            random_repeats=protocol.removal["random_repeats"],
                        ),
                    )
                    jobs.append(plan)
                    for method in protocol.removal["controls"]:
                        jobs.append(
                            _make_job(
                                protocol_hash,
                                stage,
                                dataset.dataset_id,
                                seed,
                                str(method),
                                _dataset_parameters(dataset),
                                depends_on=(plan.job_id,),
                            )
                        )
        elif stage == "alpha":
            for dataset in protocol.datasets:
                for seed in protocol.seeds:
                    plan = _make_job(
                        protocol_hash,
                        stage,
                        dataset.dataset_id,
                        seed,
                        "selection_plan",
                        _dataset_parameters(
                            dataset, alpha_grid=list(protocol.alpha_grid)
                        ),
                    )
                    jobs.append(plan)
                    for alpha in protocol.alpha_grid:
                        jobs.append(
                            _make_job(
                                protocol_hash,
                                stage,
                                dataset.dataset_id,
                                seed,
                                "alpha_evaluation",
                                _dataset_parameters(dataset, alpha=alpha),
                                depends_on=(plan.job_id,),
                            )
                        )
        elif stage == "nsga2":
            for dataset in protocol.datasets:
                for seed in protocol.seeds:
                    for variant in protocol.nsga2["variants"]:
                        jobs.append(
                            _make_job(
                                protocol_hash,
                                stage,
                                dataset.dataset_id,
                                seed,
                                str(variant),
                                _dataset_parameters(
                                    dataset,
                                    evaluations=protocol.nsga2["evaluations"],
                                    population=protocol.nsga2["population"],
                                    ref_point=list(protocol.nsga2["ref_point"]),
                                ),
                            )
                        )
        elif stage == "ablation":
            if selected_features is None:
                raise AssertionError("ablation data manifest was not loaded")
            for dataset in protocol.datasets:
                for seed in protocol.seeds:
                    for axis in sorted(protocol.ablations):
                        if axis == "utility" and dataset.dataset_id != "isolet":
                            continue
                        for value in protocol.ablations[axis]:
                            if (
                                axis == "dispatcher"
                                and value == "exact"
                                and selected_features[(dataset.dataset_id, seed)]
                                > protocol.exact_max_features
                            ):
                                continue
                            jobs.append(
                                _make_job(
                                    protocol_hash,
                                    stage,
                                    dataset.dataset_id,
                                    seed,
                                    axis,
                                    _dataset_parameters(
                                        dataset, axis=axis, value=value
                                    ),
                                )
                            )
        elif stage == "summarize":
            jobs.append(_make_job(protocol_hash, stage, _RUN_LEVEL_DATASET_ID, 0, "aggregate_results", {"required_stages": _STAGE_ORDER[2:7]}))
        elif stage == "figures":
            jobs.append(_make_job(protocol_hash, stage, _RUN_LEVEL_DATASET_ID, 0, "generate_figures", {"required_stages": ("summarize",)}))
    _ensure_unique(tuple(job.job_id for job in jobs), "duplicate manifest job IDs")
    return tuple(jobs)


def atomic_write_json(path: Path, value: object) -> None:
    """Atomically replace *path* with finite, canonical JSON."""
    _atomic_write_json(Path(path), value)


class ManifestStore:
    """Atomically persist and append the ordered manifest authority."""

    _FIELDS = (
        "sequence",
        "job_id",
        "protocol_hash",
        "stage",
        "dataset_id",
        "seed",
        "method",
        "parameters_json",
        "depends_on_json",
        "output_relpath",
    )

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.name:
            raise ValueError("manifest path is invalid")

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self, manifest: Sequence[JobSpec]) -> tuple[JobSpec, ...]:
        """Create a new manifest without replacing an existing authority."""
        jobs = _validated_manifest_authority(manifest)
        if self._path.exists():
            raise FileExistsError(f"manifest already exists: {self._path}")
        _atomic_write_text(self._path, self._csv_text(jobs))
        return jobs

    def load(self) -> tuple[JobSpec, ...]:
        """Strictly reload the exact persisted order."""
        try:
            with self._path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                if tuple(reader.fieldnames or ()) != self._FIELDS:
                    raise ValueError("manifest CSV header is invalid")
                rows = list(reader)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise ValueError("manifest CSV is malformed") from exc
        jobs: list[JobSpec] = []
        try:
            for index, row in enumerate(rows):
                if None in row or set(row) != set(self._FIELDS):
                    raise ValueError(f"manifest row {index} has an invalid schema")
                if row["sequence"] != str(index):
                    raise ValueError("manifest row sequence does not preserve order")
                seed_text = row["seed"]
                if not re.fullmatch(r"0|[1-9][0-9]*", seed_text):
                    raise ValueError(f"manifest row {index} seed is invalid")
                parameters = json.loads(row["parameters_json"])
                dependencies = json.loads(row["depends_on_json"])
                if not isinstance(parameters, dict) or not isinstance(dependencies, list):
                    raise ValueError(f"manifest row {index} JSON fields are invalid")
                jobs.append(
                    JobSpec(
                        job_id=row["job_id"],
                        protocol_hash=row["protocol_hash"],
                        stage=row["stage"],
                        dataset_id=row["dataset_id"],
                        seed=int(seed_text),
                        method=row["method"],
                        parameters=parameters,
                        depends_on=tuple(dependencies),
                        output_relpath=row["output_relpath"],
                    )
                )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"manifest CSV is malformed: {exc}") from exc
        return _validated_manifest_authority(jobs)

    def append(self, jobs: Sequence[JobSpec]) -> tuple[JobSpec, ...]:
        """Atomically append contiguous previously unfrozen stage chunks."""
        existing = self.load()
        supplied = _validated_status_manifest(jobs)
        if any(job.protocol_hash != existing[0].protocol_hash for job in supplied):
            raise ValueError("manifest append has a foreign protocol hash")
        suffix = _validated_manifest_authority(supplied)
        suffix_stages = tuple(dict.fromkeys(job.stage for job in suffix))
        existing_stages = tuple(dict.fromkeys(job.stage for job in existing))
        expected_indices = tuple(
            range(
                _STAGE_ORDER.index(existing_stages[-1]) + 1,
                _STAGE_ORDER.index(existing_stages[-1]) + 1 + len(suffix_stages),
            )
        )
        actual_indices = tuple(_STAGE_ORDER.index(stage) for stage in suffix_stages)
        if actual_indices != expected_indices:
            raise ValueError(
                "manifest append must contain contiguous canonical stage chunks"
            )
        combined = _validated_manifest_authority(existing + suffix)
        _atomic_write_text(self._path, self._csv_text(combined))
        return combined

    def _csv_text(self, jobs: Sequence[JobSpec]) -> str:
        from io import StringIO

        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=self._FIELDS, lineterminator="\n")
        writer.writeheader()
        for sequence, job in enumerate(jobs):
            writer.writerow(
                {
                    "sequence": sequence,
                    "job_id": job.job_id,
                    "protocol_hash": job.protocol_hash,
                    "stage": job.stage,
                    "dataset_id": job.dataset_id,
                    "seed": job.seed,
                    "method": job.method,
                    "parameters_json": json.dumps(
                        job.parameters, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False,
                    ),
                    "depends_on_json": json.dumps(
                        list(job.depends_on), separators=(",", ":"), ensure_ascii=False,
                    ),
                    "output_relpath": job.output_relpath,
                }
            )
        return output.getvalue()


def read_data_manifest(
    path: Path, protocol: ResolvedProtocol
) -> tuple[dict[str, object], ...]:
    """Validate a complete rich manifest and return its job-expansion projection."""
    rows = _read_data_manifest_rows(Path(path), protocol)
    expected = _data_manifest_scope(protocol)
    if tuple((row["dataset_id"], row["seed"]) for row in rows) != expected:
        raise ValueError("data_manifest does not match the complete protocol scope")
    return tuple(
        {
            "dataset_id": row["dataset_id"],
            "seed": row["seed"],
            "selected_feature_count": row["selected_feature_count"],
        }
        for row in rows
    )


def upsert_data_manifest(
    path: Path, protocol: ResolvedProtocol, row: Mapping[str, object]
) -> None:
    """Atomically add one frozen preprocess row to a partial rich manifest."""
    path = Path(path)
    action = preflight_data_manifest(path, protocol, row)
    if action == "identical":
        return
    candidate = _validate_data_manifest_row(dict(row), protocol)
    existing = _read_data_manifest_rows(path, protocol) if path.exists() else ()
    key = (candidate["dataset_id"], candidate["seed"])
    by_key = {(item["dataset_id"], item["seed"]): item for item in existing}
    if key in by_key:
        if by_key[key] == candidate:
            return
        raise ValueError("data_manifest row conflicts with an already-frozen row")
    by_key[key] = candidate
    scope = _data_manifest_scope(protocol)
    ordered = tuple(by_key[item] for item in scope if item in by_key)
    _validate_data_manifest_raw_authority(ordered)
    _atomic_write_text(path, _data_manifest_csv_text(ordered))


def preflight_data_manifest(
    path: Path, protocol: ResolvedProtocol, row: Mapping[str, object]
) -> str:
    """Validate one candidate without mutating the partial manifest."""
    path = Path(path)
    candidate = _validate_data_manifest_row(dict(row), protocol)
    existing = _read_data_manifest_rows(path, protocol) if path.exists() else ()
    key = (candidate["dataset_id"], candidate["seed"])
    by_key = {(item["dataset_id"], item["seed"]): item for item in existing}
    if key in by_key:
        if by_key[key] == candidate:
            return "identical"
        raise ValueError("data_manifest row conflicts with an already-frozen row")
    combined = (*existing, candidate)
    _validate_data_manifest_raw_authority(combined)
    return "insert"


def _read_data_manifest_rows(
    path: Path, protocol: ResolvedProtocol
) -> tuple[dict[str, object], ...]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != _DATA_MANIFEST_FIELDS:
                raise ValueError("data_manifest CSV header is invalid")
            raw_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("data_manifest CSV is malformed") from exc
    rows: list[dict[str, object]] = []
    for index, row in enumerate(raw_rows):
        if None in row or set(row) != set(_DATA_MANIFEST_FIELDS):
            raise ValueError(f"data_manifest[{index}] has an invalid schema")
        parsed: dict[str, object] = dict(row)
        for field in _DATA_MANIFEST_INTEGER_FIELDS:
            text = row[field]
            if not re.fullmatch(r"0|[1-9][0-9]*", text):
                raise ValueError(f"data_manifest[{index}].{field} is invalid")
            parsed[field] = int(text)
        rows.append(_validate_data_manifest_row(parsed, protocol))
    keys = tuple((row["dataset_id"], row["seed"]) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("data_manifest contains duplicate dataset/seed rows")
    _validate_data_manifest_raw_authority(rows)
    scope = _data_manifest_scope(protocol)
    if keys != tuple(key for key in scope if key in set(keys)):
        raise ValueError("data_manifest rows are not in canonical protocol order")
    return tuple(rows)


def _validate_data_manifest_raw_authority(
    rows: Sequence[Mapping[str, object]],
) -> None:
    authority: dict[str, tuple[object, object, object]] = {}
    for row in rows:
        dataset_id = str(row["dataset_id"])
        identity = (
            row["raw_file_sha256"],
            row["raw_row_count"],
            row["raw_feature_count"],
        )
        frozen = authority.setdefault(dataset_id, identity)
        if frozen != identity:
            raise ValueError(
                "data_manifest raw authority differs across seeds for "
                f"dataset {dataset_id}"
            )


def _data_manifest_scope(protocol: ResolvedProtocol) -> tuple[tuple[str, int], ...]:
    return tuple(
        (dataset.dataset_id, seed)
        for dataset in protocol.datasets
        for seed in protocol.seeds
    )


def _validate_data_manifest_row(
    row: dict[str, object], protocol: ResolvedProtocol
) -> dict[str, object]:
    if set(row) != set(_DATA_MANIFEST_FIELDS):
        raise ValueError("data_manifest row has an invalid schema")
    dataset_id = row["dataset_id"]
    seed = row["seed"]
    if (dataset_id, seed) not in _data_manifest_scope(protocol):
        raise ValueError("data_manifest row is outside the protocol scope")
    dataset = next(item for item in protocol.datasets if item.dataset_id == dataset_id)
    preprocess_job = next(
        job
        for job in build_job_manifest(protocol, stages=("preprocess",))
        if job.dataset_id == dataset_id and job.seed == seed
    )
    expected_identity = {
        "protocol_hash": _protocol_hash(protocol),
        "job_id": preprocess_job.job_id,
        "task": dataset.task,
        "utility": dataset.utility,
        "target": dataset.target,
    }
    for field, expected in expected_identity.items():
        if row[field] != expected:
            raise ValueError(f"data_manifest {field} does not match protocol authority")
    for field in _DATA_MANIFEST_INTEGER_FIELDS:
        value = row[field]
        minimum = 0 if field == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"data_manifest {field} is invalid")
    if row["selected_feature_count"] > row["raw_feature_count"]:
        raise ValueError("data_manifest selected_feature_count exceeds raw features")
    if (
        row["train_row_count"] + row["val_row_count"] + row["test_row_count"]
        != row["raw_row_count"]
    ):
        raise ValueError("data_manifest partition row counts do not match raw_row_count")
    for field in _DATA_MANIFEST_SHA_FIELDS:
        if not isinstance(row[field], str) or not re.fullmatch(r"[0-9a-f]{64}", row[field]):
            raise ValueError(f"data_manifest {field} must be a lowercase SHA-256")
    prefix = f"preprocess/{dataset_id}/s{seed}"
    for partition, field in (
        ("train", "train_relpath"), ("val", "val_relpath"), ("test", "test_relpath")
    ):
        expected = f"{prefix}/{partition}.csv"
        if _safe_relative_path(row[field], f"data_manifest {field}") != expected:
            raise ValueError(f"data_manifest {field} is not the canonical split path")
    selected = _canonical_manifest_json(
        row["selected_features_json"], "selected_features_json"
    )
    if (
        not isinstance(selected, list)
        or len(selected) != row["selected_feature_count"]
        or not all(isinstance(item, str) and item for item in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("data_manifest selected_features_json is invalid")
    for field in (
        "train_label_summary_json", "val_label_summary_json",
        "test_label_summary_json", "transform_metadata_json",
    ):
        value = _canonical_manifest_json(row[field], field)
        if not isinstance(value, dict):
            raise ValueError(f"data_manifest {field} must encode a JSON object")
    return {field: row[field] for field in _DATA_MANIFEST_FIELDS}


def _canonical_manifest_json(value: object, field: str) -> object:
    if not isinstance(value, str):
        raise ValueError(f"data_manifest {field} must be canonical JSON text")
    try:
        parsed = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
        canonical = json.dumps(
            parsed, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"data_manifest {field} is not finite canonical JSON") from exc
    if value != canonical:
        raise ValueError(f"data_manifest {field} is not canonical JSON")
    return parsed


def _data_manifest_csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=_DATA_MANIFEST_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def read_result_record(path: Path, expected_job: JobSpec) -> ResultRecord:
    """Read one result record only if it exactly matches its manifest job."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("result record is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("result record must be a JSON object")
    expected_keys = {"job_id", "protocol_hash", "stage", "dataset_id", "seed", "method", "metrics", "artifacts", "started_at", "finished_at"}
    if set(payload) != expected_keys:
        raise ValueError("result record has an invalid schema")
    for field in ("job_id", "protocol_hash", "stage", "dataset_id", "seed", "method"):
        if payload[field] != getattr(expected_job, field):
            raise ValueError(f"result record {field} does not match manifest job")
    _positive_or_zero_int(payload["seed"], "result record seed")
    metrics = _result_metrics(payload["metrics"])
    artifacts = _result_artifacts(payload["artifacts"])
    started_at = _result_timestamp(payload["started_at"], "started_at")
    finished_at = _result_timestamp(payload["finished_at"], "finished_at")
    if finished_at < started_at:
        raise ValueError("result record finished_at precedes started_at")
    return ResultRecord(expected_job.job_id, expected_job.protocol_hash, expected_job.stage, expected_job.dataset_id, expected_job.seed, expected_job.method, metrics, artifacts, payload["started_at"], payload["finished_at"])


class StatusStore:
    """Durable, manifest-bound state transitions for one experiment run.

    Stage documents own individual job states while the root document contains
    only a rebuilt aggregate.  Both are atomically replaced after every legal
    transition; a later constructor validates their exact schemas before it
    permits a resumed controller to use them.
    """

    _SCHEMA_VERSION = 1
    _ENTRY_KEYS = frozenset(
        {
            "state",
            "timestamp",
            "attempt",
            "result_path",
            "exception_class",
            "message",
            "traceback_path",
            "recovery",
            "skip_reason",
        }
    )

    def __init__(
        self,
        layout: RunLayout,
        manifest: Sequence[JobSpec],
        approved_skips: Mapping[str, str] | None = None,
    ) -> None:
        if layout != _run_layout(layout.root):
            raise ValueError("run layout must use the canonical paths")
        self._layout = layout
        self._manifest = _validated_status_manifest(manifest)
        self._by_id = {job.job_id: job for job in self._manifest}
        self._stages = tuple(dict.fromkeys(job.stage for job in self._manifest))
        self._manifest_hash = _status_manifest_hash(self._manifest)
        supplied_skips = _validated_approved_skips(approved_skips, self._by_id)
        self._usable = True
        stage_exists = {stage: self._stage_path(stage).exists() for stage in self._stages}
        if not layout.status_path.exists() and not any(stage_exists.values()):
            self._approved_skips = MappingProxyType(dict(supplied_skips))
            self._entries = {
                job.job_id: self._new_entry(job) for job in self._manifest
            }
            self._persist_all()
        elif not layout.status_path.exists():
            if not all(stage_exists.values()):
                raise ValueError("partial status initialization has missing stage status")
            self._entries, persisted_skips = self._load_stage_entries(self._stages)
            if approved_skips is not None and supplied_skips != persisted_skips:
                raise ValueError("approved skips do not match persisted status")
            self._approved_skips = MappingProxyType(dict(persisted_skips))
            self._persist_root()
        else:
            root = _read_status_json(layout.status_path, "root status")
            if not isinstance(root, dict) or set(root) != {
                "schema_version", "manifest_hash", "approved_skips", "stages"
            }:
                raise ValueError("root status has an invalid schema")
            if not _exact_int(root["schema_version"], self._SCHEMA_VERSION):
                raise ValueError("root status schema version is invalid")
            if not isinstance(root["manifest_hash"], str) or not root["manifest_hash"]:
                raise ValueError("root status manifest hash is invalid")
            if root["manifest_hash"] == self._manifest_hash:
                if not all(stage_exists.values()):
                    raise ValueError("partial status initialization has missing stage status")
                persisted_skips = self._validate_root(root, self._manifest)
                if approved_skips is not None and supplied_skips != persisted_skips:
                    raise ValueError("approved skips do not match persisted status")
                self._approved_skips = MappingProxyType(dict(persisted_skips))
                self._entries, stage_skips = self._load_stage_entries(self._stages)
                if stage_skips != persisted_skips:
                    raise ValueError("root and stage approved skips do not match")
                if root["stages"] != self._aggregate():
                    self._persist_root()
            else:
                try:
                    durable_manifest = ManifestStore(layout.manifest_path).load()
                except ValueError as exc:
                    raise ValueError(
                        "status extension requires a valid durable manifest"
                    ) from exc
                if durable_manifest != self._manifest:
                    raise ValueError(
                        "status extension must equal the published durable manifest"
                    )
                self._extend_from_root(root, supplied_skips, approved_skips is not None)

    def entry(self, job_id: str) -> dict[str, object]:
        """Return an isolated copy of one current entry."""
        self._require_usable()
        job = self._job(job_id)
        return deepcopy(self._entries[job.job_id])

    @property
    def manifest(self) -> tuple[JobSpec, ...]:
        """Return an isolated ordered copy of the bound manifest authority."""
        self._require_usable()
        return deepcopy(self._manifest)

    @property
    def layout(self) -> RunLayout:
        """Return the canonical run layout bound to this store."""
        return self._layout

    @property
    def protocol_hash(self) -> str:
        """Return the single protocol identity bound by all manifest jobs."""
        hashes = {job.protocol_hash for job in self._manifest}
        if len(hashes) != 1:
            raise ValueError("status manifest contains foreign protocol hashes")
        return next(iter(hashes))

    def decide(self, job_id: str) -> ResumeDecision:
        """Return whether a job may be skipped, validating prior success."""
        self._require_usable()
        job = self._job(job_id)
        entry = self._entries[job.job_id]
        state = entry["state"]
        if state == "pending":
            return ResumeDecision("execute", "pending")
        if state == "running":
            return ResumeDecision("execute", "stale_running")
        if state == "failed":
            return ResumeDecision("execute", "failed")
        if state == "skipped":
            reason = entry["skip_reason"]
            if self._approved_skips.get(job.job_id) == reason:
                return ResumeDecision("skip", "approved_skip")
            raise RuntimeError("persisted skipped job is not approved")
        if state != "success":  # guarded by schema validation
            raise RuntimeError("persisted status state is invalid")
        try:
            read_result_record(self._result_path(job), job)
        except ValueError as exc:
            self._replace_entry(
                job,
                self._updated_entry(
                    entry,
                    state="failed",
                    exception_class=type(exc).__name__,
                    message=str(exc),
                    traceback_path=None,
                    recovery=None,
                    skip_reason=None,
                ),
            )
            return ResumeDecision("execute", "invalid_success")
        return ResumeDecision("skip", "valid_success")

    def start(self, job_id: str) -> dict[str, object]:
        """Start pending/failed work, or explicitly recover stale running work."""
        self._require_usable()
        job = self._job(job_id)
        entry = self._entries[job.job_id]
        state = entry["state"]
        if state not in {"pending", "failed", "running"}:
            raise RuntimeError(f"illegal status transition {state} -> running")
        recovery = None
        if state == "running":
            recovery = {
                "event": "interrupted_recovery",
                "at": _utc_timestamp(),
                "previous_attempt": entry["attempt"],
            }
        updated = self._updated_entry(
            entry,
            state="running",
            attempt=_status_attempt(entry["attempt"]) + 1,
            exception_class=None,
            message=None,
            traceback_path=None,
            recovery=recovery,
            skip_reason=None,
        )
        self._replace_entry(job, updated)
        return deepcopy(updated)

    mark_running = start

    def succeed(self, job_id: str) -> dict[str, object]:
        """Validate the canonical record before recording a successful job."""
        self._require_usable()
        job = self._job(job_id)
        entry = self._entries[job.job_id]
        if entry["state"] != "running":
            raise RuntimeError(
                f"illegal status transition {entry['state']} -> success"
            )
        # Validation is deliberately before any status write, so a bad result
        # cannot make either status file claim success.
        read_result_record(self._result_path(job), job)
        updated = self._updated_entry(
            entry,
            state="success",
            exception_class=None,
            message=None,
            traceback_path=None,
            recovery=None,
            skip_reason=None,
        )
        self._replace_entry(job, updated)
        return deepcopy(updated)

    mark_success = succeed

    def fail(
        self,
        job_id: str,
        exception: BaseException,
        *,
        traceback_path: str | None = None,
    ) -> dict[str, object]:
        """Record an execution failure without swallowing its provenance."""
        self._require_usable()
        if not isinstance(exception, BaseException):
            raise TypeError("status failure exception must be a BaseException")
        job = self._job(job_id)
        entry = self._entries[job.job_id]
        if entry["state"] != "running":
            raise RuntimeError(
                f"illegal status transition {entry['state']} -> failed"
            )
        _optional_status_relpath(traceback_path, "traceback_path")
        updated = self._updated_entry(
            entry,
            state="failed",
            exception_class=type(exception).__name__,
            message=str(exception),
            traceback_path=traceback_path,
            recovery=None,
            skip_reason=None,
        )
        self._replace_entry(job, updated)
        return deepcopy(updated)

    mark_failed = fail

    def skip(self, job_id: str, reason: str) -> dict[str, object]:
        """Record only an exactly pre-approved manifest exception."""
        self._require_usable()
        job = self._job(job_id)
        entry = self._entries[job.job_id]
        if entry["state"] != "pending":
            raise RuntimeError(
                f"illegal status transition {entry['state']} -> skipped"
            )
        if self._approved_skips.get(job.job_id) != reason:
            raise ValueError("job skip is not exactly approved")
        updated = self._updated_entry(
            entry,
            state="skipped",
            exception_class=None,
            message=None,
            traceback_path=None,
            recovery=None,
            skip_reason=reason,
        )
        self._replace_entry(job, updated)
        return deepcopy(updated)

    def _job(self, job_id: str) -> JobSpec:
        if not isinstance(job_id, str) or job_id not in self._by_id:
            raise ValueError("status job ID is not in the manifest")
        return self._by_id[job_id]

    def _require_usable(self) -> None:
        if not self._usable:
            raise RuntimeError("status store must be reopened after a root write failure")

    def _new_entry(self, job: JobSpec) -> dict[str, object]:
        return {
            "state": "pending",
            "timestamp": _utc_timestamp(),
            "attempt": 0,
            "result_path": job.output_relpath,
            "exception_class": None,
            "message": None,
            "traceback_path": None,
            "recovery": None,
            "skip_reason": None,
        }

    def _updated_entry(self, entry: Mapping[str, object], **updates: object) -> dict[str, object]:
        return {**entry, **updates, "timestamp": _utc_timestamp()}

    def _replace_entry(self, job: JobSpec, entry: dict[str, object]) -> None:
        _validate_status_entry(entry, job)
        previous = self._entries[job.job_id]
        self._entries[job.job_id] = entry
        try:
            self._persist_stage(job.stage)
        except Exception:
            self._entries[job.job_id] = previous
            raise
        try:
            self._persist_root()
        except Exception:
            # The stage replacement is already durable.  Do not let this live
            # object write a stale in-memory view over it; reopen to reconcile.
            self._usable = False
            raise

    def _persist_all(self) -> None:
        for stage in self._stages:
            self._persist_stage(stage)
        self._persist_root()

    def _persist_stage(self, stage: str) -> None:
        stage_manifest = self._stage_manifest(stage)
        jobs = {
            job.job_id: self._entries[job.job_id]
            for job in stage_manifest
        }
        atomic_write_json(
            self._stage_path(stage),
            {
                "schema_version": self._SCHEMA_VERSION,
                "manifest_hash": _status_manifest_hash(stage_manifest),
                "stage": stage,
                "approved_skips": {
                    job.job_id: self._approved_skips[job.job_id]
                    for job in stage_manifest if job.job_id in self._approved_skips
                },
                "jobs": jobs,
            },
        )

    def _persist_root(self) -> None:
        atomic_write_json(
            self._layout.status_path,
            {
                "schema_version": self._SCHEMA_VERSION,
                "manifest_hash": self._manifest_hash,
                "approved_skips": dict(self._approved_skips),
                "stages": self._aggregate(),
            },
        )

    def _validate_root(
        self, root: object, manifest: Sequence[JobSpec]
    ) -> dict[str, str]:
        jobs = tuple(manifest)
        by_id = {job.job_id: job for job in jobs}
        stages = tuple(dict.fromkeys(job.stage for job in jobs))
        if not isinstance(root, dict) or set(root) != {
            "schema_version", "manifest_hash", "approved_skips", "stages"
        }:
            raise ValueError("root status has an invalid schema")
        if not _exact_int(root["schema_version"], self._SCHEMA_VERSION):
            raise ValueError("root status schema version is invalid")
        if root["manifest_hash"] != _status_manifest_hash(jobs):
            raise ValueError("status manifest does not match the supplied manifest")
        persisted_skips = _validated_approved_skips(root["approved_skips"], by_id)
        if not isinstance(root["stages"], dict) or set(root["stages"]) != set(stages):
            raise ValueError("root status has foreign or missing stages")
        for stage, counts in root["stages"].items():
            _validate_stage_counts(counts, stage)
        return persisted_skips

    def _aggregate(self) -> dict[str, dict[str, int]]:
        return {
            stage: _stage_counts(
                self._entries[job.job_id]
                for job in self._manifest
                if job.stage == stage
            )
            for stage in self._stages
        }

    def _load_stage_entries(
        self, stages: Sequence[str]
    ) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
        entries: dict[str, dict[str, object]] = {}
        approvals: dict[str, str] = {}
        for stage in stages:
            payload = _read_status_json(self._stage_path(stage), f"{stage} status")
            stage_manifest = self._stage_manifest(stage)
            expected_ids = {job.job_id for job in stage_manifest}
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version", "manifest_hash", "stage", "approved_skips", "jobs"
            }:
                raise ValueError(f"{stage} status has an invalid schema")
            if (
                not _exact_int(payload["schema_version"], self._SCHEMA_VERSION)
                or payload["manifest_hash"] != _status_manifest_hash(stage_manifest)
                or payload["stage"] != stage
                or not isinstance(payload["approved_skips"], dict)
                or not isinstance(payload["jobs"], dict)
                or set(payload["jobs"]) != expected_ids
            ):
                raise ValueError(f"{stage} status is foreign or malformed")
            for job_id, entry in payload["jobs"].items():
                job = self._by_id[job_id]
                _validate_status_entry(entry, job)
                entries[job_id] = dict(entry)
            stage_approvals = _validated_approved_skips(
                payload["approved_skips"],
                {job.job_id: job for job in self._manifest if job.stage == stage},
            )
            approvals.update(stage_approvals)
            for job_id, entry in payload["jobs"].items():
                if entry["state"] == "skipped" and approvals.get(job_id) != entry["skip_reason"]:
                    raise ValueError("skipped stage status does not match approval")
        return entries, approvals

    def _extend_from_root(
        self,
        root: object,
        supplied_skips: Mapping[str, str],
        skips_were_supplied: bool,
    ) -> None:
        if not isinstance(root, dict) or not isinstance(root.get("manifest_hash"), str):
            raise ValueError("root status has an invalid schema")
        prefix: tuple[JobSpec, ...] | None = None
        prefix_stages: tuple[str, ...] = ()
        for index, stage in enumerate(self._stages[:-1], start=1):
            candidate_stages = self._stages[:index]
            candidate = tuple(
                job for job in self._manifest if job.stage in candidate_stages
            )
            if _status_manifest_hash(candidate) == root["manifest_hash"]:
                prefix = candidate
                prefix_stages = candidate_stages
                break
        if prefix is None:
            raise ValueError("status manifest is not an exact persisted manifest prefix")
        persisted_skips = self._validate_root(root, prefix)
        if skips_were_supplied and {
            job_id: supplied_skips[job_id]
            for job_id in persisted_skips
            if job_id in supplied_skips
        } != persisted_skips:
            raise ValueError("approved skips do not match persisted status")
        approvals = dict(persisted_skips)
        approvals.update(supplied_skips)
        self._approved_skips = MappingProxyType(approvals)
        self._entries, stage_skips = self._load_stage_entries(prefix_stages)
        if stage_skips != persisted_skips:
            raise ValueError("root and stage approved skips do not match")
        for stage in self._stages[len(prefix_stages):]:
            if self._stage_path(stage).exists():
                suffix_entries, suffix_skips = self._load_stage_entries((stage,))
                for entry in suffix_entries.values():
                    if entry["state"] != "pending" or entry["attempt"] != 0:
                        raise ValueError("appended stage status is not an unstarted suffix")
                expected_skips = {
                    job_id: reason
                    for job_id, reason in approvals.items()
                    if self._by_id[job_id].stage == stage
                }
                if suffix_skips != expected_skips:
                    raise ValueError("appended stage approved skips do not match")
                self._entries.update(suffix_entries)
            else:
                self._entries.update(
                    {
                        job.job_id: self._new_entry(job)
                        for job in self._stage_manifest(stage)
                    }
                )
                self._persist_stage(stage)
        self._persist_root()

    def _stage_manifest(self, stage: str) -> tuple[JobSpec, ...]:
        return tuple(job for job in self._manifest if job.stage == stage)

    def _stage_path(self, stage: str) -> Path:
        return self._layout.root / stage / "status.json"

    def _result_path(self, job: JobSpec) -> Path:
        return _canonical_result_path(self._layout, job)


def assert_stage_complete(
    layout: RunLayout,
    manifest: Sequence[JobSpec],
    required_stages: Sequence[str],
) -> CompletenessReport:
    """Return completion only when every required manifest job is canonical.

    Files that are not named by the manifest are intentionally ignored: neither
    their presence nor their count can make a run complete.
    """
    issues: list[str] = []
    required = _required_completion_stages(required_stages)
    jobs = tuple(manifest)
    ids = [job.job_id for job in jobs if isinstance(job, JobSpec)]
    duplicate_ids = tuple(sorted(job_id for job_id in set(ids) if ids.count(job_id) > 1))
    for job_id in duplicate_ids:
        issues.append(f"duplicate manifest job ID: {job_id}")
    for stage in required:
        if not any(isinstance(job, JobSpec) and job.stage == stage for job in jobs):
            issues.append(f"{stage}: no manifest jobs")
    full_manifest_hash: str | None = None
    try:
        full_manifest_hash = _status_manifest_hash(_validated_status_manifest(jobs))
    except (TypeError, ValueError) as exc:
        issues.append(f"full manifest is invalid: {exc}")
    unique_jobs = tuple(
        job for index, job in enumerate(jobs)
        if (
            isinstance(job, JobSpec)
            and job.stage in required
            and job.job_id not in ids[:index]
        )
    )
    entries, approvals, snapshot_issues = _read_completion_snapshot(
        layout, unique_jobs, expected_manifest_hash=full_manifest_hash
    )
    issues.extend(snapshot_issues)
    try:
        audit = AuditLog(layout.audit_path)
        issues.extend(
            audit.integrity_issues(
                expected_run_id=layout.root.name,
                allowed_job_ids=tuple(
                    job.job_id for job in jobs if isinstance(job, JobSpec)
                ),
                expected_job_authority={
                    job.job_id: (job.stage, job.dataset_id, job.seed)
                    for job in jobs
                    if isinstance(job, JobSpec)
                },
            )
        )
    except (TypeError, ValueError) as exc:
        issues.append(f"audit configuration is invalid: {exc}")

    complete_ids: list[str] = []
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, JobSpec):
            issues.append("manifest contains a non-JobSpec entry")
            continue
        if job.stage not in required or job.job_id in seen:
            continue
        seen.add(job.job_id)
        entry = entries.get(job.job_id)
        if entry is None:
            issues.append(f"{job.job_id}: missing or invalid status")
            continue
        state = entry["state"]
        if state == "skipped":
            if approvals.get(job.job_id) == entry["skip_reason"]:
                complete_ids.append(job.job_id)
            else:
                issues.append(f"{job.job_id}: unapproved skipped status")
            continue
        if state != "success":
            issues.append(f"{job.job_id}: status is {state}")
            continue
        try:
            read_result_record(_canonical_result_path(layout, job), job)
        except ValueError as exc:
            issues.append(f"{job.job_id}: invalid or foreign-protocol result: {exc}")
            continue
        complete_ids.append(job.job_id)
    if issues:
        raise IncompleteRunError(tuple(issues))
    return CompletenessReport(tuple(required), tuple(complete_ids))


def _validated_status_manifest(manifest: Sequence[JobSpec]) -> tuple[JobSpec, ...]:
    if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes)):
        raise TypeError("status manifest must be a sequence of JobSpec entries")
    jobs = tuple(manifest)
    if not jobs or not all(isinstance(job, JobSpec) for job in jobs):
        raise ValueError("status manifest must contain JobSpec entries")
    for job in jobs:
        _safe_builtin_identifier(job.job_id, "manifest job ID")
        _safe_builtin_identifier(job.dataset_id, "manifest dataset ID")
        if type(job.stage) is not str or job.stage not in _STAGE_ORDER:
            raise ValueError("manifest job has an unknown stage")
        _positive_or_zero_int(job.seed, "manifest seed")
        _safe_relative_path(job.output_relpath, "manifest result path")
    _ensure_unique(tuple(job.job_id for job in jobs), "duplicate manifest job IDs")
    return jobs


def _validated_manifest_authority(
    manifest: Sequence[JobSpec],
) -> tuple[JobSpec, ...]:
    jobs = _validated_status_manifest(manifest)
    protocol_hashes: set[str] = set()
    output_paths: list[str] = []
    prior_ids: set[str] = set()
    prior_stage_index = -1
    closed_stages: set[str] = set()
    active_stage: str | None = None
    for job in jobs:
        if type(job.protocol_hash) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", job.protocol_hash
        ):
            raise ValueError("manifest protocol hash is invalid")
        protocol_hashes.add(job.protocol_hash)
        if type(job.method) is not str or not job.method or "\x00" in job.method:
            raise ValueError("manifest method is invalid")
        if type(job.parameters) is not dict:
            raise ValueError("manifest parameters must be a mapping")
        try:
            canonical_parameters = json.loads(
                json.dumps(job.parameters, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest parameters are not finite JSON") from exc
        if canonical_parameters != job.parameters:
            raise ValueError("manifest parameters are not canonical JSON")
        if type(job.depends_on) is not tuple or not all(
            type(dependency) is str and dependency for dependency in job.depends_on
        ):
            raise ValueError("manifest dependencies are invalid")
        if len(set(job.depends_on)) != len(job.depends_on) or job.job_id in job.depends_on:
            raise ValueError("manifest dependencies contain duplicates or a cycle")
        if not set(job.depends_on).issubset(prior_ids):
            raise ValueError("manifest dependencies must reference prior jobs")
        identity = {
            "protocol_hash": job.protocol_hash,
            "stage": job.stage,
            "dataset_id": job.dataset_id,
            "seed": job.seed,
            "method": job.method,
            "parameters": job.parameters,
            "depends_on": job.depends_on,
        }
        if canonical_hash(identity) != job.job_id:
            raise ValueError("manifest job ID does not match its canonical identity")
        if job.output_relpath != f"raw_metrics/{job.stage}/{job.job_id}.json":
            raise ValueError("manifest result path is not canonical for its job")
        output_paths.append(job.output_relpath)
        stage_index = _STAGE_ORDER.index(job.stage)
        if job.stage != active_stage:
            if active_stage is not None:
                closed_stages.add(active_stage)
            if job.stage in closed_stages or stage_index <= prior_stage_index:
                raise ValueError("manifest contains repeated or out-of-order stage chunks")
            active_stage = job.stage
            prior_stage_index = stage_index
        prior_ids.add(job.job_id)
    if len(protocol_hashes) != 1:
        raise ValueError("manifest contains foreign protocol hashes")
    _ensure_unique(tuple(output_paths), "duplicate manifest result paths")
    return jobs


def _status_manifest_hash(manifest: Sequence[JobSpec]) -> str:
    return canonical_hash(
        [
            {
                "job_id": job.job_id,
                "protocol_hash": job.protocol_hash,
                "stage": job.stage,
                "dataset_id": job.dataset_id,
                "seed": job.seed,
                "method": job.method,
                "parameters": _json_safe(job.parameters),
                "depends_on": list(job.depends_on),
                "output_relpath": job.output_relpath,
            }
            for job in manifest
        ]
    )


def _validated_approved_skips(
    value: Mapping[str, str] | object,
    jobs: Mapping[str, JobSpec],
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("approved skips must be a mapping")
    approved: dict[str, str] = {}
    for job_id, reason in value.items():
        if (
            not isinstance(job_id, str)
            or job_id not in jobs
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("approved skips contain an unknown job or invalid reason")
        approved[job_id] = reason
    return approved


def _read_status_json(path: Path, context: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid JSON") from exc


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _status_attempt(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("status attempt must be a non-negative integer")
    return value


def _exact_int(value: object, expected: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


def _optional_status_text(value: object, context: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"status {context} is invalid")


def _safe_relative_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} is invalid")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or candidate.drive
        or candidate.root
        or "\\" in value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"{context} is invalid")
    return value


def _optional_status_relpath(value: object, context: str) -> None:
    if value is not None:
        _safe_relative_path(value, f"status {context}")


def _validate_status_entry(entry: object, job: JobSpec) -> None:
    if not isinstance(entry, dict) or set(entry) != StatusStore._ENTRY_KEYS:
        raise ValueError("status entry has an invalid schema")
    if entry["state"] not in {"pending", "running", "success", "failed", "skipped"}:
        raise ValueError("status entry state is invalid")
    attempt = _status_attempt(entry["attempt"])
    if entry["result_path"] != job.output_relpath:
        raise ValueError("status entry result path is foreign")
    _utc_status_timestamp(entry["timestamp"], "status timestamp")
    _optional_status_text(entry["exception_class"], "exception class")
    _optional_status_text(entry["message"], "message")
    _optional_status_relpath(entry["traceback_path"], "traceback path")
    recovery = entry["recovery"]
    if recovery is not None:
        if not isinstance(recovery, dict) or set(recovery) != {
            "event", "at", "previous_attempt"
        }:
            raise ValueError("status recovery metadata is invalid")
        if recovery["event"] != "interrupted_recovery":
            raise ValueError("status recovery event is invalid")
        _utc_status_timestamp(recovery["at"], "status recovery timestamp")
        previous_attempt = _status_attempt(recovery["previous_attempt"])
    else:
        previous_attempt = None
    skip_reason = entry["skip_reason"]
    _optional_status_text(skip_reason, "skip reason")
    state = entry["state"]
    if state == "skipped" and skip_reason is None:
        raise ValueError("skipped status requires a reason")
    if state != "skipped" and skip_reason is not None:
        raise ValueError("non-skipped status cannot have a skip reason")
    has_error = (
        entry["exception_class"] is not None
        or entry["message"] is not None
        or entry["traceback_path"] is not None
    )
    if state == "pending":
        if attempt != 0 or has_error or recovery is not None:
            raise ValueError("pending status has invalid semantic fields")
    elif state == "running":
        if attempt < 1 or has_error:
            raise ValueError("running status has invalid semantic fields")
        if recovery is not None and (previous_attempt is None or previous_attempt < 1 or previous_attempt != attempt - 1):
            raise ValueError("running status recovery does not match attempt")
    elif state == "success":
        if attempt < 1 or has_error or recovery is not None:
            raise ValueError("success status has invalid semantic fields")
    elif state == "failed":
        if (
            attempt < 1
            or not isinstance(entry["exception_class"], str)
            or not entry["exception_class"]
            or not isinstance(entry["message"], str)
            or recovery is not None
        ):
            raise ValueError("failed status has invalid semantic fields")
    elif state == "skipped":
        if attempt != 0 or has_error or recovery is not None:
            raise ValueError("skipped status has invalid semantic fields")


def _utc_status_timestamp(value: object, context: str) -> datetime:
    timestamp = _result_timestamp(value, context)
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"{context} must use UTC")
    return timestamp


def _validate_stage_counts(value: object, stage: str) -> None:
    expected = {"pending", "running", "success", "failed", "skipped"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"root status aggregate for {stage} has an invalid schema")
    for count in value.values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"root status aggregate for {stage} is invalid")


def _stage_counts(entries: Sequence[Mapping[str, object]] | object) -> dict[str, int]:
    counts = {state: 0 for state in ("pending", "running", "success", "failed", "skipped")}
    for entry in entries:  # type: ignore[union-attr]
        state = entry["state"]
        if state not in counts:
            raise ValueError("status entry state is invalid")
        counts[state] += 1
    return counts


def _required_completion_stages(required_stages: Sequence[str]) -> tuple[str, ...]:
    if (
        not isinstance(required_stages, Sequence)
        or isinstance(required_stages, (str, bytes))
        or not required_stages
    ):
        raise ValueError("required stages must be a non-empty sequence")
    requested = tuple(required_stages)
    if not all(isinstance(stage, str) for stage in requested):
        raise ValueError("required stages must be strings")
    _ensure_unique(requested, "duplicate required stages")
    unknown = set(requested) - set(_STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown required stages: {sorted(unknown)}")
    return tuple(stage for stage in _STAGE_ORDER if stage in requested)


def _canonical_result_path(layout: RunLayout, job: JobSpec) -> Path:
    _safe_relative_path(job.output_relpath, "manifest result path")
    candidate = layout.root / job.output_relpath
    try:
        _require_within(layout.root, candidate, "result path")
    except RuntimeError as exc:
        raise ValueError("result path escapes the run directory") from exc
    return candidate.resolve(strict=False)


def _read_completion_snapshot(
    layout: RunLayout,
    jobs: Sequence[JobSpec],
    *,
    expected_manifest_hash: str | None,
) -> tuple[dict[str, dict[str, object]], dict[str, str], list[str]]:
    """Read the same status schema as ``StatusStore`` without mutating it."""
    entries: dict[str, dict[str, object]] = {}
    approvals: dict[str, str] = {}
    issues: list[str] = []
    if not jobs:
        return entries, approvals, issues
    try:
        valid_jobs = _validated_status_manifest(jobs)
    except (TypeError, ValueError) as exc:
        return entries, approvals, [f"manifest status validation failed: {exc}"]
    by_id = {job.job_id: job for job in valid_jobs}
    stages = tuple(dict.fromkeys(job.stage for job in valid_jobs))
    manifest_hash: str | None = None
    root: object | None = None
    if not layout.status_path.exists():
        issues.append("root status is missing")
    else:
        try:
            root = _read_status_json(layout.status_path, "root status")
            if not isinstance(root, dict) or set(root) != {
                "schema_version", "manifest_hash", "approved_skips", "stages"
            }:
                raise ValueError("root status has an invalid schema")
            if not _exact_int(root["schema_version"], StatusStore._SCHEMA_VERSION):
                raise ValueError("root status schema version is invalid")
            if not isinstance(root["manifest_hash"], str) or not root["manifest_hash"]:
                raise ValueError("root status manifest is invalid")
            manifest_hash = root["manifest_hash"]
            if (
                expected_manifest_hash is not None
                and manifest_hash != expected_manifest_hash
            ):
                raise ValueError("root status manifest does not match full manifest")
            if not isinstance(root["stages"], dict) or not set(stages).issubset(root["stages"]):
                raise ValueError("root status has foreign or missing stages")
            for stage in stages:
                _validate_stage_counts(root["stages"][stage], stage)
            if not isinstance(root["approved_skips"], dict):
                raise ValueError("root status approved skips are invalid")
            approvals = _validated_approved_skips(
                {
                    job_id: reason
                    for job_id, reason in root["approved_skips"].items()
                    if job_id in by_id
                },
                by_id,
            )
        except ValueError as exc:
            issues.append(f"root status is invalid: {exc}")
            root = None
    stage_approvals: dict[str, str] = {}
    for stage in stages:
        path = layout.root / stage / "status.json"
        if not path.exists():
            issues.append(f"{stage} status is missing")
            continue
        try:
            payload = _read_status_json(path, f"{stage} status")
            stage_manifest = tuple(job for job in valid_jobs if job.stage == stage)
            expected_ids = {job.job_id for job in stage_manifest}
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version", "manifest_hash", "stage", "approved_skips", "jobs"
            }:
                raise ValueError("invalid schema")
            if (
                not _exact_int(payload["schema_version"], StatusStore._SCHEMA_VERSION)
                or not isinstance(payload["manifest_hash"], str)
                or payload["manifest_hash"] != _status_manifest_hash(stage_manifest)
                or payload["stage"] != stage
                or not isinstance(payload["jobs"], dict)
                or set(payload["jobs"]) != expected_ids
            ):
                raise ValueError("foreign or malformed")
            local_approvals = _validated_approved_skips(
                payload["approved_skips"],
                {job.job_id: job for job in valid_jobs if job.stage == stage},
            )
            for job_id, entry in payload["jobs"].items():
                try:
                    _validate_status_entry(entry, by_id[job_id])
                    if (
                        entry["state"] == "skipped"
                        and local_approvals.get(job_id) != entry["skip_reason"]
                    ):
                        raise ValueError("skipped entry is not exactly approved")
                    entries[job_id] = dict(entry)
                except ValueError as exc:
                    issues.append(f"{job_id}: status is invalid: {exc}")
            stage_approvals.update(local_approvals)
        except ValueError as exc:
            issues.append(f"{stage} status is invalid: {exc}")
    if root is not None:
        if approvals != stage_approvals:
            issues.append("root and stage approved skips do not match")
        aggregate = {
            stage: _stage_counts(
                entries[job.job_id]
                for job in valid_jobs
                if job.stage == stage and job.job_id in entries
            )
            for stage in stages
        }
        for stage, counts in aggregate.items():
            if root["stages"][stage] != counts:
                issues.append("root status aggregate does not match stage status")
                break
    return entries, stage_approvals, issues


def _manifest_stages(stages: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(stages, tuple) or not stages:
        raise ValueError("stages must be a non-empty tuple")
    if not all(isinstance(stage, str) for stage in stages):
        raise ValueError("stage names must be strings")
    if len(set(stages)) != len(stages):
        raise ValueError("duplicate stage requested")
    unknown = sorted(set(stages) - set(_STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown stage requested: {unknown}")
    return tuple(stage for stage in _STAGE_ORDER if stage in stages)


def _make_job(
    protocol_hash: str,
    stage: str,
    dataset_id: str,
    seed: int,
    method: str,
    parameters: dict[str, object],
    *,
    depends_on: tuple[str, ...] = (),
) -> JobSpec:
    canonical_parameters = _json_safe(parameters)
    if not isinstance(canonical_parameters, dict):
        raise TypeError("job parameters must serialize to a mapping")
    identity = {
        "protocol_hash": protocol_hash,
        "stage": stage,
        "dataset_id": dataset_id,
        "seed": seed,
        "method": method,
        "parameters": canonical_parameters,
        "depends_on": depends_on,
    }
    job_id = canonical_hash(identity)
    return JobSpec(
        job_id=job_id,
        protocol_hash=protocol_hash,
        stage=stage,
        dataset_id=dataset_id,
        seed=seed,
        method=method,
        parameters=canonical_parameters,
        depends_on=depends_on,
        output_relpath=f"raw_metrics/{stage}/{job_id}.json",
    )


def _dataset_parameters(
    dataset: DatasetSpec, **parameters: object
) -> dict[str, object]:
    return {"task": dataset.task, "utility": dataset.utility, **parameters}


def _selected_feature_counts(
    protocol: ResolvedProtocol,
    data_manifest: tuple[dict[str, object], ...] | list[dict[str, object]]
    | None,
) -> dict[tuple[str, int], int]:
    if data_manifest is None or not isinstance(data_manifest, (tuple, list)):
        raise ValueError("estimator stage requires a data_manifest")
    expected = {
        (dataset.dataset_id, seed)
        for dataset in protocol.datasets
        for seed in protocol.seeds
    }
    counts: dict[tuple[str, int], int] = {}
    for index, entry in enumerate(data_manifest):
        if not isinstance(entry, dict) or set(entry) != {
            "dataset_id",
            "seed",
            "selected_feature_count",
        }:
            raise ValueError(f"data_manifest[{index}] has an invalid schema")
        dataset_id = entry["dataset_id"]
        seed = entry["seed"]
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"data_manifest[{index}].dataset_id is invalid")
        _positive_or_zero_int(seed, f"data_manifest[{index}].seed")
        count = entry["selected_feature_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"data_manifest[{index}].selected_feature_count must be a positive integer"
            )
        key = (dataset_id, seed)
        if key in counts:
            raise ValueError("data_manifest contains duplicate dataset/seed rows")
        counts[key] = count
    if set(counts) != expected:
        raise ValueError("data_manifest does not match the selected dataset/seed scope")
    return counts


def _positive_or_zero_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _result_metrics(value: object) -> dict[str, float | int | str | bool | None]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and key for key in value
    ):
        raise ValueError("result record metrics must be a string-keyed mapping")
    metrics: dict[str, float | int | str | bool | None] = {}
    for key, metric in value.items():
        if metric is None or isinstance(metric, (str, bool)):
            metrics[key] = metric
        elif isinstance(metric, int):
            metrics[key] = metric
        elif isinstance(metric, float) and -float("inf") < metric < float("inf"):
            metrics[key] = metric
        else:
            raise ValueError(f"result record metric {key!r} is invalid")
    return metrics


def _result_artifacts(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and key and isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError("result record artifacts must be a string mapping")
    artifacts = dict(value)
    for artifact in artifacts.values():
        candidate = Path(artifact)
        if (
            not artifact
            or candidate.is_absolute()
            or candidate.drive
            or candidate.root
            or "\\" in artifact
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ValueError("result record artifact path is invalid")
    return artifacts


def _result_timestamp(value: object, context: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"result record {context} must be an ISO timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"result record {context} must be an ISO timestamp"
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"result record {context} must include a timezone")
    return timestamp


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, _json_text(value))


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_parent_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_parent_directory(parent: Path) -> None:
    try:
        descriptor = os.open(parent, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def load_resolved_protocol(
    config_dir: Path,
    *,
    seeds: tuple[int, ...] | None,
    backend: str | None,
    smoke: bool,
) -> ResolvedProtocol:
    """Load, merge, and strictly validate the ICOMP 2026 protocol."""
    config_dir = Path(config_dir)
    global_config = _read_yaml(config_dir / "icomp2026_pipeline.yaml")
    _expect_exact_keys(global_config, _GLOBAL_KEYS, "global protocol")

    protocol_version = _nonempty_string(
        global_config["protocol_version"], "protocol_version"
    )
    configured_ids = _dataset_id_tuple(global_config["datasets"], "datasets")
    _ensure_unique(configured_ids, "duplicate dataset IDs")

    smoke_config = _mapping(global_config["smoke"], "smoke")
    _expect_exact_keys(smoke_config, _SMOKE_KEYS, "smoke")
    smoke_ids = _dataset_id_tuple(
        smoke_config["datasets"], "smoke.datasets"
    )
    _ensure_unique(smoke_ids, "duplicate smoke dataset IDs")
    unknown_smoke = sorted(set(smoke_ids) - set(configured_ids))
    if unknown_smoke:
        raise ValueError(
            f"smoke.datasets contains unknown dataset IDs: {unknown_smoke}"
        )

    selected_ids = smoke_ids if smoke else configured_ids
    datasets = tuple(
        _load_dataset_spec(config_dir, dataset_id) for dataset_id in selected_ids
    )
    loaded_ids = tuple(dataset.dataset_id for dataset in datasets)
    _ensure_unique(loaded_ids, "duplicate dataset IDs in dataset files")
    if loaded_ids != selected_ids:
        raise ValueError(
            "dataset.id values must match the configured dataset filenames: "
            f"expected {selected_ids}, got {loaded_ids}"
        )

    configured_seeds = _positive_int_tuple(global_config["seeds"], "seeds")
    _ensure_unique(configured_seeds, "duplicate seeds")
    smoke_seeds = _positive_int_tuple(smoke_config["seeds"], "smoke.seeds")
    _ensure_unique(smoke_seeds, "duplicate smoke seeds")

    if smoke:
        selected_seeds = smoke_seeds
        if seeds is not None and tuple(seeds) != smoke_seeds:
            raise ValueError(
                "smoke mode seeds must match the frozen smoke.seeds setting "
                f"{smoke_seeds}, got {tuple(seeds)}"
            )
    else:
        selected_seeds = (
            _positive_int_tuple(seeds, "seeds override")
            if seeds is not None
            else configured_seeds
        )
        _ensure_unique(selected_seeds, "duplicate seeds")

    configured_backend = _backend(global_config["backend"], "backend")
    selected_backend = (
        _backend(backend, "backend override")
        if backend is not None
        else configured_backend
    )

    model = _mapping(global_config["model"], "model")
    _expect_exact_keys(model, _MODEL_KEYS, "model")
    model_params = _validate_model(model)

    alpha = _mapping(global_config["alpha"], "alpha")
    _expect_exact_keys(alpha, _ALPHA_KEYS, "alpha")
    alpha_default = _alpha(alpha["default"], "alpha.default")
    alpha_grid = tuple(
        _alpha(value, f"alpha.grid[{index}]")
        for index, value in enumerate(_sequence(alpha["grid"], "alpha.grid"))
    )
    if not alpha_grid:
        raise ValueError("alpha.grid must not be empty")
    _ensure_unique(alpha_grid, "duplicate alpha values")

    estimator = _mapping(global_config["estimator"], "estimator")
    _expect_exact_keys(estimator, _ESTIMATOR_KEYS, "estimator")
    estimator_methods = _string_tuple(
        estimator["methods"], "estimator.methods"
    )
    _ensure_unique(estimator_methods, "duplicate estimator methods")
    invalid_methods = sorted(set(estimator_methods) - _VALID_ESTIMATORS)
    if invalid_methods:
        raise ValueError(f"unknown estimator methods: {invalid_methods}")
    configured_budgets = _positive_int_tuple(
        estimator["budgets"], "estimator.budgets"
    )
    _ensure_unique(configured_budgets, "duplicate estimator budgets")
    exact_max_features = _positive_int(
        estimator["exact_max_features"], "estimator.exact_max_features"
    )
    smoke_budgets = _positive_int_tuple(
        smoke_config["estimator_budgets"], "smoke.estimator_budgets"
    )
    _ensure_unique(smoke_budgets, "duplicate smoke estimator budgets")
    estimator_budgets = smoke_budgets if smoke else configured_budgets

    removal = _mapping(global_config["removal"], "removal")
    _expect_exact_keys(removal, _REMOVAL_KEYS, "removal")
    controls = _string_tuple(removal["controls"], "removal.controls")
    if controls != _CANONICAL_REMOVAL_CONTROLS:
        raise ValueError(
            "removal.controls must use the canonical ordered control set"
        )
    random_repeats = _positive_int(
        smoke_config["random_repeats"] if smoke else removal["random_repeats"],
        "removal.random_repeats",
    )
    removal_resolved: dict[str, object] = {
        "controls": controls,
        "random_repeats": random_repeats,
    }

    nsga2 = _mapping(global_config["nsga2"], "nsga2")
    _expect_exact_keys(nsga2, _NSGA2_KEYS, "nsga2")
    variants = _string_tuple(nsga2["variants"], "nsga2.variants")
    if variants != _CANONICAL_NSGA2_VARIANTS:
        raise ValueError(
            "nsga2.variants must use the canonical ordered variant set"
        )
    population = _positive_int(
        smoke_config["nsga2_population"] if smoke else nsga2["population"],
        "nsga2.population",
    )
    evaluations = _positive_int(
        smoke_config["nsga2_evaluations"] if smoke else nsga2["evaluations"],
        "nsga2.evaluations",
    )
    ref_point_values = _sequence(nsga2["ref_point"], "nsga2.ref_point")
    if len(ref_point_values) != 2:
        raise ValueError("nsga2.ref_point must contain exactly two values")
    ref_point = tuple(
        _finite_number(value, f"nsga2.ref_point[{index}]")
        for index, value in enumerate(ref_point_values)
    )
    if ref_point != (1.1, 1.1):
        raise ValueError(
            "nsga2.ref_point must be the frozen reference point [1.1, 1.1], "
            f"got {list(ref_point)}"
        )
    nsga2_resolved: dict[str, object] = {
        "variants": variants,
        "population": population,
        "evaluations": evaluations,
        "ref_point": ref_point,
    }

    ablations = _mapping(global_config["ablations"], "ablations")
    _expect_exact_keys(ablations, _ABLATION_KEYS, "ablations")
    ablations_resolved = {
        key: _string_tuple(ablations[key], f"ablations.{key}")
        for key in sorted(_ABLATION_KEYS)
    }
    for key, values in ablations_resolved.items():
        if not values:
            raise ValueError(f"ablations.{key} must not be empty")
        _ensure_unique(values, f"duplicate ablations.{key} values")

    smoke_resolved: dict[str, object] = {
        "datasets": smoke_ids,
        "seeds": smoke_seeds,
        "estimator_budgets": smoke_budgets,
        "random_repeats": _positive_int(
            smoke_config["random_repeats"], "smoke.random_repeats"
        ),
        "nsga2_population": _positive_int(
            smoke_config["nsga2_population"], "smoke.nsga2_population"
        ),
        "nsga2_evaluations": _positive_int(
            smoke_config["nsga2_evaluations"], "smoke.nsga2_evaluations"
        ),
    }

    return ResolvedProtocol(
        protocol_version=protocol_version,
        run_kind="smoke" if smoke else "formal",
        datasets=datasets,
        seeds=selected_seeds,
        backend=selected_backend,
        model_params=model_params,
        alpha_default=alpha_default,
        alpha_grid=alpha_grid,
        estimator_methods=estimator_methods,
        estimator_budgets=estimator_budgets,
        exact_max_features=exact_max_features,
        nsga2=nsga2_resolved,
        removal=removal_resolved,
        ablations=ablations_resolved,
        smoke=smoke_resolved,
    )


def _load_dataset_spec(config_dir: Path, expected_id: str) -> DatasetSpec:
    path = config_dir / f"{expected_id}.yaml"
    _require_direct_child(config_dir, path, "dataset ID")
    config = _read_yaml(path)
    _expect_exact_keys(config, _DATASET_FILE_KEYS, f"dataset file {path.name}")

    dataset = _mapping(config["dataset"], f"{expected_id}.dataset")
    preprocessing = _mapping(
        config["preprocessing"], f"{expected_id}.preprocessing"
    )
    split = _mapping(config["split"], f"{expected_id}.split")
    _expect_exact_keys(dataset, _DATASET_KEYS, f"{expected_id}.dataset")
    _expect_exact_keys(
        preprocessing,
        _PREPROCESSING_KEYS,
        f"{expected_id}.preprocessing",
    )
    _expect_exact_keys(split, _SPLIT_KEYS, f"{expected_id}.split")

    dataset_id = _safe_identifier(
        dataset["id"], f"{expected_id}.dataset.id"
    )
    display_name = _nonempty_string(
        dataset["display_name"], f"{expected_id}.dataset.display_name"
    )
    target = _nonempty_string(
        dataset["target"], f"{expected_id}.dataset.target"
    )
    task = _nonempty_string(dataset["task"], f"{expected_id}.dataset.task")
    utility = _nonempty_string(
        dataset["utility"], f"{expected_id}.dataset.utility"
    )
    _validate_task_utility(task, utility, expected_id)

    raw_path_text = _nonempty_string(
        dataset["raw_path"], f"{expected_id}.dataset.raw_path"
    )
    raw_path = Path(raw_path_text)
    if not raw_path.is_absolute():
        raw_path = (config_dir.parent / raw_path).resolve()

    max_features_value = preprocessing["max_features"]
    max_features = (
        None
        if max_features_value is None
        else _positive_int(
            max_features_value,
            f"{expected_id}.preprocessing.max_features",
        )
    )
    variance_threshold = _finite_number(
        preprocessing["variance_threshold"],
        f"{expected_id}.preprocessing.variance_threshold",
    )
    if variance_threshold < 0.0:
        raise ValueError(
            f"{expected_id}.preprocessing.variance_threshold must be >= 0"
        )

    train_frac = _fraction(split["train_frac"], f"{expected_id}.split.train_frac")
    val_frac = _fraction(split["val_frac"], f"{expected_id}.split.val_frac")
    test_frac = _fraction(split["test_frac"], f"{expected_id}.split.test_frac")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"{expected_id}.split fractions must sum to 1.0, got {total}"
        )

    return DatasetSpec(
        dataset_id=dataset_id,
        display_name=display_name,
        raw_path=raw_path,
        target=target,
        task=task,
        utility=utility,
        max_features=max_features,
        variance_threshold=variance_threshold,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing configuration file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    return _mapping(value, str(path))


def _expect_exact_keys(
    value: dict[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {unknown}")
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return value


def _sequence(value: object, context: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a sequence")
    return value


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _safe_identifier(value: object, context: str) -> str:
    identifier = _nonempty_string(value, context)
    if identifier in {".", ".."} or re.fullmatch(
        r"[A-Za-z0-9._-]+", identifier
    ) is None:
        raise ValueError(f"{context} must be a safe dataset ID")
    return identifier


def _safe_builtin_identifier(value: object, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} must be a built-in string")
    return _safe_identifier(value, context)


def _dataset_id_tuple(value: object, context: str) -> tuple[str, ...]:
    sequence = _sequence(value, context)
    if not sequence:
        raise ValueError(f"{context} must not be empty")
    return tuple(
        _safe_identifier(item, f"{context}[{index}]")
        for index, item in enumerate(sequence)
    )


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    sequence = _sequence(value, context)
    if not sequence:
        raise ValueError(f"{context} must not be empty")
    return tuple(
        _nonempty_string(item, f"{context}[{index}]")
        for index, item in enumerate(sequence)
    )


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer, got {value!r}")
    return value


def _positive_int_tuple(value: object, context: str) -> tuple[int, ...]:
    sequence = _sequence(value, context)
    if not sequence:
        raise ValueError(f"{context} must not be empty")
    return tuple(
        _positive_int(item, f"{context}[{index}]")
        for index, item in enumerate(sequence)
    )


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric, got {value!r}")
    number = float(value)
    if not (-float("inf") < number < float("inf")):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return number


def _alpha(value: object, context: str) -> float:
    number = _finite_number(value, context)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{context} alpha must be in (0, 1], got {number}")
    return number


def _fraction(value: object, context: str) -> float:
    number = _finite_number(value, context)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{context} must be in (0, 1), got {number}")
    return number


def _backend(value: object, context: str) -> str:
    backend = _nonempty_string(value, context)
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{context} must be one of {sorted(_VALID_BACKENDS)}, got {backend!r}"
        )
    return backend


def _ensure_unique(values: tuple[object, ...], context: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(context)


def _validate_model(model: dict[str, Any]) -> dict[str, object]:
    return {
        "n_estimators": _positive_int(
            model["n_estimators"], "model.n_estimators"
        ),
        "max_depth": _positive_int(model["max_depth"], "model.max_depth"),
        "learning_rate": _positive_number(
            model["learning_rate"], "model.learning_rate"
        ),
        "n_jobs": _positive_int(model["n_jobs"], "model.n_jobs"),
    }


def _positive_number(value: object, context: str) -> float:
    number = _finite_number(value, context)
    if number <= 0.0:
        raise ValueError(f"{context} must be positive, got {number}")
    return number


def _validate_task_utility(task: str, utility: str, context: str) -> None:
    if task not in _VALID_TASKS:
        raise ValueError(
            f"{context}.task must be one of {sorted(_VALID_TASKS)}, got {task!r}"
        )
    valid_utilities = (
        _REGRESSION_UTILITIES
        if task == "regression"
        else _CLASSIFICATION_UTILITIES
    )
    if utility not in valid_utilities:
        raise ValueError(
            f"{context}.utility {utility!r} is invalid for task {task!r}; "
            f"choose from {sorted(valid_utilities)}"
        )
