"""Training/validation-only acceptance gates for grouped acceleration.

This command is deliberately independent of the final-test stage.  It reads
only frozen run metadata, estimator/removal validation evidence, grouped
bundles, and cache statistics.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import yaml

import experiments.icomp2026_estimator_bundle as estimator_bundle_module
from experiments.icomp2026_estimator_bundle import (
    EstimatorGroupRequest,
    bundle_from_document,
)
from experiments.icomp2026_progress import progress_from_document
from minppi_mofs.experiment_protocol import (
    ManifestStore,
    _DATA_MANIFEST_FIELDS,
    _read_completion_snapshot,
    _run_layout,
    _status_manifest_hash,
    atomic_write_json,
    canonical_hash,
    read_canonical_json,
    read_result_record,
)


_EVIDENCE_NAMES = frozenset({"abalone", "spambase", "madelon", "isolet"})
_ABALONE_SEEDS = (42, 123, 456, 789, 1024)
_ABALONE_METHODS = (
    "exact",
    "random_mc",
    "mi_correlation_guided",
    "hybrid",
)
_TERNARY_SCORES = frozenset({-1, 0, 1})
_PROTOCOL_KEYS = frozenset(
    {
        "ablations",
        "alpha_default",
        "alpha_grid",
        "backend",
        "datasets",
        "estimator_budgets",
        "estimator_methods",
        "exact_max_features",
        "execution",
        "model_params",
        "nsga2",
        "protocol_version",
        "removal",
        "run_kind",
        "seeds",
        "smoke",
    }
)
_TARGET_SCIENTIFIC_PROTOCOL_FIELDS = (
    "protocol_version",
    "backend",
    "model_params",
    "alpha_default",
    "alpha_grid",
    "estimator_methods",
    "estimator_budgets",
    "exact_max_features",
    "nsga2",
    "removal",
    "ablations",
)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float | str
    threshold: float | str
    detail: str


@dataclass(frozen=True)
class RunMetadata:
    root: Path
    protocol: Mapping[str, object]
    protocol_hash: str
    execution_mode: str
    backend: str
    model_params: Mapping[str, object]
    budgets: tuple[int, ...]
    datasets: Mapping[str, Mapping[str, object]]
    rows: Mapping[tuple[str, int], Mapping[str, str]]
    manifest: tuple[object, ...]
    status_entries: Mapping[str, Mapping[str, object]]
    environment: Mapping[str, object]


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def classification_gate(legacy: float, candidate: float) -> GateResult:
    legacy_value = _finite(legacy, "legacy classification utility")
    candidate_value = _finite(candidate, "candidate classification utility")
    delta = candidate_value - legacy_value
    return GateResult(
        "classification_utility",
        delta >= -0.005 - 1e-12,
        delta,
        "delta >= -0.005",
        f"candidate={candidate_value:.12g}, legacy={legacy_value:.12g}",
    )


def r2_gate(legacy: float, candidate: float) -> GateResult:
    legacy_value = _finite(legacy, "legacy r2")
    candidate_value = _finite(candidate, "candidate r2")
    delta = candidate_value - legacy_value
    return GateResult(
        "r2",
        delta >= -0.01 - 1e-12,
        delta,
        "delta >= -0.01",
        f"candidate={candidate_value:.12g}, legacy={legacy_value:.12g}",
    )


def compression_gate(
    legacy_compression: float,
    candidate_compression: float,
    legacy_utility: float,
    candidate_utility: float,
) -> GateResult:
    legacy_compression_value = _finite(
        legacy_compression, "legacy compression"
    )
    candidate_compression_value = _finite(
        candidate_compression, "candidate compression"
    )
    legacy_utility_value = _finite(legacy_utility, "legacy utility")
    candidate_utility_value = _finite(candidate_utility, "candidate utility")
    compression_delta = (
        candidate_compression_value - legacy_compression_value
    )
    utility_delta = candidate_utility_value - legacy_utility_value
    passed = (
        compression_delta >= -0.02 - 1e-12
        or utility_delta >= 0.005 - 1e-12
    )
    return GateResult(
        "compression",
        passed,
        compression_delta,
        "compression delta >= -0.02 or utility delta >= 0.005",
        f"utility_delta={utility_delta:.12g}",
    )


def speed_gate(
    dataset_id: str,
    legacy_runtime_s: float,
    candidate_runtime_s: float,
    cache_state: str,
) -> GateResult:
    if cache_state != "cold":
        raise ValueError("speed evidence must use a cold candidate cache")
    thresholds = {"spambase": 2.5, "madelon": 3.5}
    if dataset_id not in thresholds:
        raise ValueError("speed dataset is not authorized")
    legacy_runtime = _finite(legacy_runtime_s, "legacy runtime")
    candidate_runtime = _finite(candidate_runtime_s, "candidate runtime")
    if legacy_runtime <= 0 or candidate_runtime <= 0:
        raise ValueError("speed runtimes must be positive")
    factor = legacy_runtime / candidate_runtime
    return GateResult(
        f"{dataset_id}_speed",
        factor >= thresholds[dataset_id],
        factor,
        thresholds[dataset_id],
        (
            f"legacy_s={legacy_runtime:.12g}, "
            f"candidate_s={candidate_runtime:.12g}"
        ),
    )


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def prediction_digest_gate(
    accuracy_digest: str, ovr_auc_digest: str
) -> GateResult:
    accuracy = _sha256(accuracy_digest, "accuracy prediction digest")
    ovr_auc = _sha256(ovr_auc_digest, "OVR-AUC prediction digest")
    return GateResult(
        "isolet_prediction_digest",
        accuracy == ovr_auc,
        accuracy,
        ovr_auc,
        "accuracy and OVR-AUC must derive from one prediction array",
    )


def _score_mapping(
    value: object, *, seed: int, source: str
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{source} seed {seed} scores must be a nonempty mapping")
    result: list[tuple[str, int]] = []
    for feature, score in value.items():
        if (
            not isinstance(feature, str)
            or not feature
            or isinstance(score, bool)
            or not isinstance(score, int)
            or score not in _TERNARY_SCORES
        ):
            raise ValueError(f"{source} seed {seed} scores are invalid")
        result.append((feature, score))
    if len({feature for feature, _ in result}) != len(result):
        raise ValueError(f"{source} seed {seed} features are duplicated")
    return tuple(result)


def _five_seed_scores(
    value: object, source: str
) -> dict[int, tuple[tuple[str, int], ...]]:
    if not isinstance(value, Mapping) or set(value) != set(_ABALONE_SEEDS):
        raise ValueError(
            f"{source} Abalone scores require the exact canonical five seeds"
        )
    return {
        seed: _score_mapping(value[seed], seed=seed, source=source)
        for seed in _ABALONE_SEEDS
    }


def _agreement_and_harmful_recall(
    exact: tuple[tuple[str, int], ...],
    approximation: tuple[tuple[str, int], ...],
) -> tuple[float, float]:
    if tuple(feature for feature, _ in approximation) != tuple(
        feature for feature, _ in exact
    ):
        raise ValueError("Abalone score feature order differs from exact")
    approximation_scores = dict(approximation)
    agreement = math.fsum(
        approximation_scores[feature] == score for feature, score in exact
    ) / len(exact)
    harmful = tuple(
        feature for feature, score in exact if score == -1
    )
    if harmful:
        recall = math.fsum(
            approximation_scores[feature] == -1 for feature in harmful
        ) / len(harmful)
    else:
        recall = (
            1.0
            if all(score != -1 for score in approximation_scores.values())
            else 0.0
        )
    return agreement, recall


def abalone_exact_gate(
    exact_by_seed: Mapping[int, Mapping[str, int]],
    legacy_by_seed: Mapping[int, Mapping[str, int]],
    candidate_by_seed: Mapping[int, Mapping[str, int]],
) -> GateResult:
    exact = _five_seed_scores(exact_by_seed, "exact")
    legacy = _five_seed_scores(legacy_by_seed, "legacy")
    candidate = _five_seed_scores(candidate_by_seed, "candidate")
    legacy_metrics = [
        _agreement_and_harmful_recall(exact[seed], legacy[seed])
        for seed in _ABALONE_SEEDS
    ]
    candidate_metrics = [
        _agreement_and_harmful_recall(exact[seed], candidate[seed])
        for seed in _ABALONE_SEEDS
    ]
    legacy_agreement = math.fsum(item[0] for item in legacy_metrics) / 5
    legacy_recall = math.fsum(item[1] for item in legacy_metrics) / 5
    candidate_agreement = math.fsum(item[0] for item in candidate_metrics) / 5
    candidate_recall = math.fsum(item[1] for item in candidate_metrics) / 5
    passed = (
        candidate_agreement >= legacy_agreement - 1e-12
        and candidate_recall >= legacy_recall - 1e-12
    )
    return GateResult(
        "abalone_exact",
        passed,
        (
            f"agreement={candidate_agreement:.12g},"
            f"harmful_recall={candidate_recall:.12g}"
        ),
        (
            f"agreement>={legacy_agreement:.12g},"
            f"harmful_recall>={legacy_recall:.12g}"
        ),
        (
            f"legacy_agreement={legacy_agreement:.12g}, "
            f"legacy_harmful_recall={legacy_recall:.12g}, "
            f"candidate_agreement={candidate_agreement:.12g}, "
            f"candidate_harmful_recall={candidate_recall:.12g}"
        ),
    )


def parse_named_runs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError("run evidence must use name=absolute_path")
        name, raw_path = value.split("=", 1)
        if name not in _EVIDENCE_NAMES:
            raise ValueError("run evidence name is not authorized")
        if name in result:
            raise ValueError("run evidence name is duplicated")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError("run evidence path must be absolute")
        result[name] = path.resolve(strict=False)
    if set(result) != _EVIDENCE_NAMES:
        raise ValueError(
            "run evidence must contain the exact names "
            "abalone, spambase, madelon, and isolet"
        )
    return result


def reject_test_artifact_fields(value: object) -> None:
    """Reject held-out result fields/paths from acceptance evidence."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact field names must be strings")
            normalized = key.lower()
            if (
                normalized == "test"
                or normalized.startswith("test_")
                or normalized.endswith("_test")
                or "_test_" in normalized
            ):
                raise ValueError("held-out test result fields are forbidden")
            reject_test_artifact_fields(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            reject_test_artifact_fields(item)
    elif isinstance(value, str):
        segments = tuple(
            segment.lower()
            for segment in re.split(r"[\\/]", value)
            if segment
        )
        if any(
            segment == "test"
            or segment == "final_test"
            or segment.startswith("test_")
            or segment.endswith("_test")
            for segment in segments
        ):
            raise ValueError("held-out test artifact paths are forbidden")


def _read_resolved_protocol(root: Path) -> tuple[dict[str, object], str]:
    path = root / "resolved_protocol.yaml"
    if path.is_symlink():
        raise ValueError("resolved protocol must be a regular file")
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("resolved protocol is missing or malformed") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        not in {_PROTOCOL_KEYS, _PROTOCOL_KEYS - {"execution"}}
        or yaml.safe_dump(value, sort_keys=True, allow_unicode=True) != text
    ):
        raise ValueError("resolved protocol is not canonical")
    protocol_hash = canonical_hash(value)
    return value, protocol_hash


def _read_data_rows(
    root: Path,
    *,
    protocol: Mapping[str, object],
    protocol_hash: str,
) -> dict[tuple[str, int], dict[str, str]]:
    path = root / "data_manifest.csv"
    if path.is_symlink():
        raise ValueError("data manifest must be a regular file")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != _DATA_MANIFEST_FIELDS:
                raise ValueError("data manifest header is invalid")
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("data manifest is missing or malformed") from exc
    datasets_value = protocol["datasets"]
    if not isinstance(datasets_value, list):
        raise ValueError("resolved protocol datasets are invalid")
    datasets = {
        item.get("dataset_id"): item
        for item in datasets_value
        if isinstance(item, dict)
    }
    result: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        try:
            dataset_id = row["dataset_id"]
            seed = int(row["seed"])
            selected = json.loads(row["selected_features_json"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("data manifest row is malformed") from exc
        dataset = datasets.get(dataset_id)
        if (
            row["protocol_hash"] != protocol_hash
            or not isinstance(dataset, dict)
            or row["task"] != dataset.get("task")
            or row["utility"] != dataset.get("utility")
            or not isinstance(selected, list)
            or not selected
            or any(not isinstance(feature, str) or not feature for feature in selected)
            or len(selected) != len(set(selected))
            or int(row["selected_feature_count"]) != len(selected)
        ):
            raise ValueError("data manifest row conflicts with resolved protocol")
        for field in (
            "train_index_sha256",
            "val_index_sha256",
            "train_file_sha256",
            "val_file_sha256",
        ):
            _sha256(row[field], f"data manifest {field}")
        key = (dataset_id, seed)
        if key in result:
            raise ValueError("data manifest dataset/seed is duplicated")
        result[key] = row
    if not result:
        raise ValueError("data manifest must not be empty")
    return result


def _read_environment(root: Path, protocol_hash: str) -> dict[str, object]:
    environment = read_canonical_json(
        root / "environment.json", context="run environment"
    )
    fingerprint = read_canonical_json(
        root / "source_fingerprint.json", context="source fingerprint"
    )
    if (
        not isinstance(environment, dict)
        or environment.get("backend") != "cuda"
        or not isinstance(environment.get("xgboost_build_info"), dict)
        or environment["xgboost_build_info"].get("USE_CUDA") is not True
    ):
        raise ValueError("run environment is not resolved CUDA")
    if (
        not isinstance(fingerprint, dict)
        or set(fingerprint)
        != {"protocol_hash", "files", "git_commit", "git_dirty"}
        or fingerprint["protocol_hash"] != protocol_hash
        or not isinstance(fingerprint["files"], dict)
        or any(
            not isinstance(name, str)
            or _sha256(digest, "source fingerprint digest") != digest
            for name, digest in fingerprint["files"].items()
        )
    ):
        raise ValueError("source fingerprint is internally invalid")
    return environment


def read_run_metadata(root: Path) -> RunMetadata:
    """Strictly validate one persisted run without comparing current source."""
    run_root = Path(root).resolve(strict=False)
    if not run_root.is_dir() or run_root.is_symlink():
        raise ValueError("resolved protocol run root is missing or invalid")
    protocol, protocol_hash = _read_resolved_protocol(run_root)
    execution = protocol.get("execution")
    if execution is None:
        execution_mode = "legacy"
    elif (
        isinstance(execution, dict)
        and set(execution)
        == {
            "mode",
            "cache_schema",
            "checkpoint_every_fits",
            "progress_every_seconds",
        }
        and execution["mode"] in {"legacy", "grouped_nested_v1"}
        and execution["cache_schema"] == "coalition_utility_v1"
    ):
        execution_mode = str(execution["mode"])
    else:
        raise ValueError("resolved protocol execution authority is invalid")
    if protocol["backend"] != "cuda":
        raise ValueError("resolved protocol execution authority is invalid")
    model_params = protocol["model_params"]
    budgets_value = protocol["estimator_budgets"]
    datasets_value = protocol["datasets"]
    if (
        not isinstance(model_params, dict)
        or not isinstance(budgets_value, list)
        or not budgets_value
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in budgets_value
        )
        or len(set(budgets_value)) != len(budgets_value)
        or not isinstance(datasets_value, list)
    ):
        raise ValueError("resolved protocol estimator authority is invalid")
    datasets: dict[str, Mapping[str, object]] = {}
    for dataset in datasets_value:
        if (
            not isinstance(dataset, dict)
            or not isinstance(dataset.get("dataset_id"), str)
            or dataset["dataset_id"] in datasets
        ):
            raise ValueError("resolved protocol dataset authority is invalid")
        datasets[dataset["dataset_id"]] = dataset
    rows = _read_data_rows(
        run_root, protocol=protocol, protocol_hash=protocol_hash
    )
    environment = _read_environment(run_root, protocol_hash)
    try:
        manifest = ManifestStore(run_root / "run_manifest.csv").load()
    except ValueError as exc:
        raise ValueError("run manifest is invalid") from exc
    if any(job.protocol_hash != protocol_hash for job in manifest):
        raise ValueError("run manifest conflicts with resolved protocol")
    layout = _run_layout(run_root)
    entries, _approvals, issues = _read_completion_snapshot(
        layout,
        manifest,
        expected_manifest_hash=_status_manifest_hash(manifest),
    )
    if issues:
        raise ValueError(f"run status is invalid: {'; '.join(issues)}")
    return RunMetadata(
        root=run_root,
        protocol=protocol,
        protocol_hash=protocol_hash,
        execution_mode=execution_mode,
        backend="cuda",
        model_params=dict(model_params),
        budgets=tuple(budgets_value),
        datasets=datasets,
        rows=rows,
        manifest=manifest,
        status_entries=entries,
        environment=environment,
    )


def _safe_artifact_path(root: Path, relative: object, context: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{context} path is invalid")
    reject_test_artifact_fields(relative)
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context} path escapes the run") from exc
    if candidate.is_symlink():
        raise ValueError(f"{context} must be a regular file")
    return candidate


def _successful_job(
    metadata: RunMetadata,
    *,
    dataset_id: str,
    seed: int,
    method: str,
    budget: int,
):
    jobs = tuple(
        job
        for job in metadata.manifest
        if job.stage == "estimator"
        and job.dataset_id == dataset_id
        and job.seed == seed
        and job.method == method
        and job.parameters.get("budget") == budget
    )
    if len(jobs) != 1:
        raise ValueError("estimator evidence job is missing or duplicated")
    job = jobs[0]
    entry = metadata.status_entries.get(job.job_id)
    if not isinstance(entry, Mapping) or entry.get("state") != "success":
        raise ValueError("estimator evidence job is not successful")
    record = read_result_record(metadata.root / job.output_relpath, job)
    reject_test_artifact_fields(asdict(record))
    return job, entry, record


def _score_evidence(
    metadata: RunMetadata,
    *,
    dataset_id: str,
    seed: int,
    method: str,
    budget: int,
) -> dict[str, object]:
    from experiments.icomp2026_stages import _read_score_artifact

    row = metadata.rows.get((dataset_id, seed))
    if row is None:
        raise ValueError("score evidence has no data-manifest row")
    features = tuple(json.loads(row["selected_features_json"]))
    job, entry, record = _successful_job(
        metadata,
        dataset_id=dataset_id,
        seed=seed,
        method=method,
        budget=budget,
    )
    if set(record.artifacts) != {"scores"}:
        raise ValueError("estimator ResultRecord artifact schema is invalid")
    artifact_path = _safe_artifact_path(
        metadata.root, record.artifacts["scores"], "estimator score artifact"
    )
    scores = _read_score_artifact(
        artifact_path,
        expected_job=job,
        budget=budget,
        expected_features=features,
        expected_metrics=record.metrics,
        expected_execution_mode=metadata.execution_mode,
    )
    return {
        "scores": scores,
        "record": record,
        "status": entry,
        "features": features,
    }


def _abalone_method_evidence(
    metadata: RunMetadata,
    *,
    seed: int,
    budget: int,
) -> dict[str, dict[str, object]]:
    """Load every canonical Abalone estimator artifact for one seed."""
    return {
        method: _score_evidence(
            metadata,
            dataset_id="abalone",
            seed=seed,
            method=method,
            budget=budget,
        )
        for method in _ABALONE_METHODS
    }


def _scientific_protocol_projection(
    protocol: Mapping[str, object],
    *,
    dataset_id: str,
) -> dict[str, object]:
    if not isinstance(protocol, Mapping):
        raise TypeError("resolved protocol must be a mapping")
    datasets = protocol.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("resolved protocol dataset scope is invalid")
    target_datasets = [
        item
        for item in datasets
        if isinstance(item, Mapping)
        and item.get("dataset_id") == dataset_id
    ]
    if len(target_datasets) != 1:
        raise ValueError(
            "target dataset scientific protocol authority is missing"
        )
    missing = set(_TARGET_SCIENTIFIC_PROTOCOL_FIELDS) - set(protocol)
    if missing:
        raise ValueError(
            "target scientific protocol authority is incomplete"
        )
    return {
        **{
            key: protocol[key]
            for key in _TARGET_SCIENTIFIC_PROTOCOL_FIELDS
        },
        "dataset": dict(target_datasets[0]),
    }


def _validate_scientific_protocol_equivalence(
    legacy: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    dataset_id: str,
) -> None:
    """Require identical scientific authority across execution modes."""
    if _scientific_protocol_projection(
        legacy, dataset_id=dataset_id
    ) != _scientific_protocol_projection(
        candidate, dataset_id=dataset_id
    ):
        raise ValueError(
            "legacy/candidate scientific protocol authority differs"
        )


def _validate_cross_run_pair(
    legacy: RunMetadata,
    candidate: RunMetadata,
    *,
    dataset_id: str,
    seeds: Sequence[int],
) -> None:
    if (
        legacy.execution_mode != "legacy"
        or candidate.execution_mode != "grouped_nested_v1"
    ):
        raise ValueError("legacy/candidate execution modes are invalid")
    _validate_scientific_protocol_equivalence(
        legacy.protocol,
        candidate.protocol,
        dataset_id=dataset_id,
    )
    if (
        legacy.backend != candidate.backend
        or legacy.backend != "cuda"
        or dict(legacy.model_params) != dict(candidate.model_params)
        or legacy.budgets != candidate.budgets
    ):
        raise ValueError("legacy/candidate model, budget, or CUDA authority differs")
    legacy_dataset = legacy.datasets.get(dataset_id)
    candidate_dataset = candidate.datasets.get(dataset_id)
    if (
        legacy_dataset is None
        or candidate_dataset is None
        or legacy_dataset.get("task") != candidate_dataset.get("task")
        or legacy_dataset.get("utility") != candidate_dataset.get("utility")
    ):
        raise ValueError("legacy/candidate dataset or utility authority differs")
    for seed in seeds:
        legacy_row = legacy.rows.get((dataset_id, seed))
        candidate_row = candidate.rows.get((dataset_id, seed))
        if legacy_row is None or candidate_row is None:
            raise ValueError("legacy/candidate split evidence is missing")
        for field in (
            "dataset_id",
            "seed",
            "task",
            "utility",
            "selected_features_json",
            "train_index_sha256",
            "val_index_sha256",
            "train_file_sha256",
            "val_file_sha256",
        ):
            if legacy_row[field] != candidate_row[field]:
                raise ValueError(
                    f"legacy/candidate training-validation {field} differs"
                )


def _group_request_from_metadata(
    metadata: RunMetadata,
    *,
    dataset_id: str,
    seed: int,
) -> EstimatorGroupRequest:
    protocol = metadata.protocol
    execution = protocol.get("execution")
    dataset = metadata.datasets.get(dataset_id)
    row = metadata.rows.get((dataset_id, seed))
    methods_value = protocol.get("estimator_methods")
    exact_max_features = protocol.get("exact_max_features")
    if (
        metadata.execution_mode != "grouped_nested_v1"
        or not isinstance(execution, Mapping)
        or not isinstance(dataset, Mapping)
        or row is None
        or not isinstance(methods_value, list)
        or isinstance(exact_max_features, bool)
        or not isinstance(exact_max_features, int)
    ):
        raise ValueError(
            "candidate estimator request authority is incomplete"
        )
    try:
        features_value = json.loads(row["selected_features_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "candidate prepared feature authority is invalid"
        ) from exc
    if not isinstance(features_value, list):
        raise ValueError(
            "candidate prepared feature authority is invalid"
        )
    features = tuple(features_value)
    methods = tuple(
        method
        for method in methods_value
        if method != "exact" or len(features) <= exact_max_features
    )
    return EstimatorGroupRequest(
        protocol_hash=metadata.protocol_hash,
        dataset_id=dataset_id,
        seed=seed,
        task=dataset.get("task"),
        utility=dataset.get("utility"),
        ordered_features=features,
        split_hashes={
            "train": row["train_file_sha256"],
            "validation": row["val_file_sha256"],
        },
        model_params=metadata.model_params,
        backend=metadata.backend,
        methods=methods,
        budgets=metadata.budgets,
        exact_max_features=exact_max_features,
        alpha=protocol.get("alpha_default"),
        cache_schema=execution.get("cache_schema"),
        checkpoint_every_fits=execution.get(
            "checkpoint_every_fits"
        ),
        progress_every_seconds=execution.get(
            "progress_every_seconds"
        ),
    )


def _validate_cold_owner_job(
    metadata: RunMetadata,
    *,
    progress: object,
    request: EstimatorGroupRequest,
) -> Mapping[str, object]:
    """Validate the first grouped job as the sole cold-build authority."""
    owner_job_id = _sha256(
        getattr(progress, "build_owner_job_id", None),
        "build owner job id",
    )
    owners = tuple(
        job
        for job in metadata.manifest
        if job.job_id == owner_job_id
    )
    if len(owners) != 1:
        raise ValueError("candidate build owner job is missing")
    owner = owners[0]
    if (
        owner.stage != "estimator"
        or owner.dataset_id != request.dataset_id
        or owner.seed != request.seed
        or owner.method not in request.methods
        or owner.parameters.get("budget") not in request.budgets
    ):
        raise ValueError(
            "candidate build owner job is outside the estimator group"
        )
    status = metadata.status_entries.get(owner_job_id)
    if (
        not isinstance(status, Mapping)
        or status.get("state") != "success"
        or status.get("attempt") != 1
        or status.get("recovery") is not None
    ):
        raise ValueError(
            "candidate build owner job does not prove a cold first attempt"
        )
    return status


def _bundle_evidence(
    metadata: RunMetadata,
    *,
    dataset_id: str,
    seed: int,
    budget: int,
    hybrid: Mapping[str, object],
) -> dict[str, object]:
    directory = (
        metadata.root
        / "estimator"
        / "_bundles"
        / dataset_id
        / f"s{seed}"
    )
    bundle_document = read_canonical_json(
        directory / "bundle.json", context="estimator bundle"
    )
    progress_document = read_canonical_json(
        directory / "progress.json", context="estimator progress"
    )
    reject_test_artifact_fields(bundle_document)
    reject_test_artifact_fields(progress_document)
    request = _group_request_from_metadata(
        metadata, dataset_id=dataset_id, seed=seed
    )
    bundle = bundle_from_document(bundle_document, request)
    progress = progress_from_document(progress_document)
    schedules = estimator_bundle_module._schedule_authority(request)
    checkpoints = [
        item
        for item in bundle.checkpoints
        if item.method == "hybrid" and item.budget == budget
    ]
    if len(checkpoints) != 1:
        raise ValueError("maximum-budget hybrid bundle checkpoint is missing")
    checkpoint = checkpoints[0]
    expected_scores = tuple(hybrid["scores"].items())
    if checkpoint.scores != expected_scores:
        raise ValueError("bundle and score artifact hybrid scores differ")
    record = hybrid["record"]
    group_runtime = _finite(
        bundle.group_runtime_s, "bundle group runtime"
    )
    physical_fits = bundle.physical_model_fit_count
    cache_hits = bundle.cache_hit_count
    progress_event = progress.event
    _validate_cold_owner_job(
        metadata, progress=progress, request=request
    )
    if (
        group_runtime <= 0
        or record.metrics.get("group_runtime_s") != group_runtime
        or record.metrics.get("group_request_digest")
        != bundle.request_digest
        or physical_fits < 1
        or cache_hits < 0
        or progress.build_owner_run_id != metadata.root.name
        or progress.initial_fitted_entry_count != 0
        or progress_event.algorithm != "complete"
        or progress_event.request_digest != bundle.request_digest
        or progress_event.schedule_digest != schedules.master_digest
        or progress_event.dataset_id != dataset_id
        or progress_event.seed != seed
        or progress_event.feature_index != len(request.ordered_features)
        or progress_event.feature_count != len(request.ordered_features)
        or progress_event.budget != request.budgets[-1]
        or progress_event.max_budget != request.budgets[-1]
        or progress_event.physical_model_fit_count != physical_fits
        or progress_event.cache_hit_count != cache_hits
        or progress_event.elapsed_s != group_runtime
        or progress_event.eta_s != 0.0
        or progress.last_completed_trial_index
        > schedules.trial_upper_bound
    ):
        raise ValueError("candidate bundle is not valid cold-cache evidence")
    return {
        "cache_state": "cold",
        "group_runtime_s": group_runtime,
        "physical_model_fit_count": physical_fits,
        "cache_hit_count": cache_hits,
        "request_digest": bundle.request_digest,
    }


def _legacy_group_runtime(
    metadata: RunMetadata,
    *,
    dataset_id: str,
    seed: int,
    budget: int,
) -> float:
    runtimes: list[float] = []
    for method in ("random_mc", "mi_correlation_guided", "hybrid"):
        evidence = _score_evidence(
            metadata,
            dataset_id=dataset_id,
            seed=seed,
            method=method,
            budget=budget,
        )
        runtime = _finite(
            evidence["record"].metrics.get("runtime_s"),
            f"legacy {method} runtime",
        )
        if runtime <= 0:
            raise ValueError("legacy estimator runtime must be positive")
        runtimes.append(runtime)
    return math.fsum(runtimes)


def _load_train_validation(
    metadata: RunMetadata, *, dataset_id: str, seed: int
):
    import pandas as pd

    row = metadata.rows[(dataset_id, seed)]
    dataset = metadata.datasets[dataset_id]
    target = dataset.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("dataset target authority is invalid")
    frames = []
    for split in ("train", "val"):
        path = _safe_artifact_path(
            metadata.root,
            row[f"{split}_relpath"],
            f"{split} validation split",
        )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"{split} validation split is missing") from exc
        if hashlib.sha256(payload).hexdigest() != row[f"{split}_file_sha256"]:
            raise ValueError(f"{split} validation split hash differs")
        frame = pd.read_csv(path)
        if target not in frame:
            raise ValueError(f"{split} validation split target is missing")
        frames.append(
            (
                frame.drop(columns=[target]),
                frame[target].to_numpy(copy=True),
            )
        )
    expected_features = tuple(json.loads(row["selected_features_json"]))
    if any(tuple(X.columns) != expected_features for X, _ in frames):
        raise ValueError("training-validation feature order differs")
    return frames[0][0], frames[0][1], frames[1][0], frames[1][1]


def _derived_removal_evidence(
    metadata: RunMetadata,
    *,
    dataset_id: str,
    seed: int,
    hybrid: Mapping[str, object],
) -> dict[str, object]:
    from minppi_mofs.modeling import evaluate_fixed_subset

    features = tuple(hybrid["features"])
    scores = hybrid["scores"]
    retained = tuple(feature for feature in features if scores[feature] != -1)
    X_train, y_train, X_val, y_val = _load_train_validation(
        metadata, dataset_id=dataset_id, seed=seed
    )
    dataset = metadata.datasets[dataset_id]
    utility = evaluate_fixed_subset(
        X_train,
        y_train,
        X_val,
        y_val,
        retained,
        task=str(dataset["task"]),
        utility=str(dataset["utility"]),
        model_params=metadata.model_params,
        backend=metadata.backend,
        seed=seed,
    )
    return {
        "compression": (len(features) - len(retained)) / len(features),
        "validation_utility": float(utility),
        "retained_features": list(retained),
    }


def _isolet_digests(metadata: RunMetadata) -> dict[str, str]:
    dataset_id, seed = "isolet", 42
    budget = max(metadata.budgets)
    jobs = tuple(
        job
        for job in metadata.manifest
        if job.stage == "ablation"
        and job.dataset_id == dataset_id
        and job.seed == seed
        and job.method == "utility"
        and job.parameters.get("value") in {"accuracy", "ovr_auc"}
    )
    if len(jobs) != 2:
        raise ValueError("ISOLET paired utility jobs are missing")
    paths: set[str] = set()
    for job in jobs:
        entry = metadata.status_entries.get(job.job_id)
        if not isinstance(entry, Mapping) or entry.get("state") != "success":
            raise ValueError("ISOLET paired utility job is not successful")
        record = read_result_record(metadata.root / job.output_relpath, job)
        reject_test_artifact_fields(asdict(record))
        if set(record.artifacts) != {"paired_scores"}:
            raise ValueError("ISOLET paired ResultRecord artifact is invalid")
        paths.add(record.artifacts["paired_scores"])
    if len(paths) != 1:
        raise ValueError("ISOLET paired jobs do not share one artifact")
    document = read_canonical_json(
        _safe_artifact_path(
            metadata.root, paths.pop(), "ISOLET paired prediction artifact"
        ),
        context="ISOLET paired prediction artifact",
    )
    reject_test_artifact_fields(document)
    if (
        not isinstance(document, dict)
        or document.get("protocol_hash") != metadata.protocol_hash
        or document.get("dataset_id") != dataset_id
        or document.get("seed") != seed
        or document.get("axis") != "utility"
        or document.get("budget") != budget
        or document.get("secondary_model_fit_delta") != 0
        or document.get("secondary_coalition_delta") != 0
        or not isinstance(document.get("prediction_records"), list)
        or not isinstance(document.get("prediction_digests"), dict)
        or set(document["prediction_digests"]) != {"accuracy", "ovr_auc"}
    ):
        raise ValueError("ISOLET paired prediction artifact schema is invalid")
    expected = hashlib.sha256(
        json.dumps(
            document["prediction_records"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    digests = document["prediction_digests"]
    for metric in ("accuracy", "ovr_auc"):
        if _sha256(digests[metric], f"ISOLET {metric} digest") != expected:
            raise ValueError("ISOLET prediction digest conflicts with records")
    return dict(digests)


def build_acceptance_report(
    *,
    config_dir: Path,
    legacy_runs: Mapping[str, Path],
    candidate_runs: Mapping[str, Path],
) -> dict[str, object]:
    config_root = Path(config_dir)
    if (
        not config_root.is_dir()
        or not (config_root / "icomp2026_pipeline.yaml").is_file()
    ):
        raise ValueError("config directory is missing the frozen protocol source")
    if set(legacy_runs) != _EVIDENCE_NAMES or set(candidate_runs) != _EVIDENCE_NAMES:
        raise ValueError("acceptance evidence names are incomplete")
    legacy = {
        name: read_run_metadata(path) for name, path in legacy_runs.items()
    }
    candidate = {
        name: read_run_metadata(path) for name, path in candidate_runs.items()
    }
    _validate_cross_run_pair(
        legacy["abalone"],
        candidate["abalone"],
        dataset_id="abalone",
        seeds=_ABALONE_SEEDS,
    )
    for dataset_id in ("spambase", "madelon", "isolet"):
        _validate_cross_run_pair(
            legacy[dataset_id],
            candidate[dataset_id],
            dataset_id=dataset_id,
            seeds=(42,),
        )

    abalone_exact: dict[int, Mapping[str, int]] = {}
    abalone_legacy: dict[int, Mapping[str, int]] = {}
    abalone_candidate: dict[int, Mapping[str, int]] = {}
    abalone_budget = max(legacy["abalone"].budgets)
    for seed in _ABALONE_SEEDS:
        legacy_methods = _abalone_method_evidence(
            legacy["abalone"], seed=seed, budget=abalone_budget
        )
        candidate_methods = _abalone_method_evidence(
            candidate["abalone"], seed=seed, budget=abalone_budget
        )
        legacy_exact = legacy_methods["exact"]
        candidate_exact = candidate_methods["exact"]
        legacy_hybrid = legacy_methods["hybrid"]
        candidate_hybrid = candidate_methods["hybrid"]
        if legacy_exact["scores"] != candidate_exact["scores"]:
            raise ValueError("cross-mode Abalone exact scores differ")
        _bundle_evidence(
            candidate["abalone"],
            dataset_id="abalone",
            seed=seed,
            budget=abalone_budget,
            hybrid=candidate_hybrid,
        )
        abalone_exact[seed] = legacy_exact["scores"]
        abalone_legacy[seed] = legacy_hybrid["scores"]
        abalone_candidate[seed] = candidate_hybrid["scores"]

    gates = [
        abalone_exact_gate(
            abalone_exact, abalone_legacy, abalone_candidate
        )
    ]
    pilot_evidence: dict[str, object] = {}
    for dataset_id in ("spambase", "madelon"):
        legacy_metadata = legacy[dataset_id]
        candidate_metadata = candidate[dataset_id]
        budget = max(legacy_metadata.budgets)
        legacy_hybrid = _score_evidence(
            legacy_metadata,
            dataset_id=dataset_id,
            seed=42,
            method="hybrid",
            budget=budget,
        )
        candidate_hybrid = _score_evidence(
            candidate_metadata,
            dataset_id=dataset_id,
            seed=42,
            method="hybrid",
            budget=budget,
        )
        legacy_removal = _derived_removal_evidence(
            legacy_metadata,
            dataset_id=dataset_id,
            seed=42,
            hybrid=legacy_hybrid,
        )
        candidate_removal = _derived_removal_evidence(
            candidate_metadata,
            dataset_id=dataset_id,
            seed=42,
            hybrid=candidate_hybrid,
        )
        bundle = _bundle_evidence(
            candidate_metadata,
            dataset_id=dataset_id,
            seed=42,
            budget=budget,
            hybrid=candidate_hybrid,
        )
        legacy_runtime = _legacy_group_runtime(
            legacy_metadata,
            dataset_id=dataset_id,
            seed=42,
            budget=budget,
        )
        task = str(legacy_metadata.datasets[dataset_id]["task"])
        utility_gate = (
            r2_gate
            if task == "regression"
            else classification_gate
        )(
            legacy_removal["validation_utility"],
            candidate_removal["validation_utility"],
        )
        gates.extend(
            (
                GateResult(
                    f"{dataset_id}_{utility_gate.name}",
                    utility_gate.passed,
                    utility_gate.observed,
                    utility_gate.threshold,
                    utility_gate.detail,
                ),
                GateResult(
                    f"{dataset_id}_compression",
                    compression_gate(
                        legacy_removal["compression"],
                        candidate_removal["compression"],
                        legacy_removal["validation_utility"],
                        candidate_removal["validation_utility"],
                    ).passed,
                    candidate_removal["compression"]
                    - legacy_removal["compression"],
                    (
                        "compression delta >= -0.02 or "
                        "utility delta >= 0.005"
                    ),
                    (
                        "candidate and legacy retained subsets were derived "
                        "from maximum-budget hybrid scores"
                    ),
                ),
                speed_gate(
                    dataset_id,
                    legacy_runtime,
                    bundle["group_runtime_s"],
                    bundle["cache_state"],
                ),
            )
        )
        pilot_evidence[dataset_id] = {
            "budget": budget,
            "legacy_runtime_s": legacy_runtime,
            "candidate_group_runtime_s": bundle["group_runtime_s"],
            "legacy_compression": legacy_removal["compression"],
            "candidate_compression": candidate_removal["compression"],
            "legacy_validation_utility": legacy_removal[
                "validation_utility"
            ],
            "candidate_validation_utility": candidate_removal[
                "validation_utility"
            ],
            "candidate_physical_model_fit_count": bundle[
                "physical_model_fit_count"
            ],
            "candidate_cache_hit_count": bundle["cache_hit_count"],
            "cache_state": bundle["cache_state"],
        }
    legacy_digests = _isolet_digests(legacy["isolet"])
    candidate_digests = _isolet_digests(candidate["isolet"])
    for side, digests in (
        ("legacy", legacy_digests),
        ("candidate", candidate_digests),
    ):
        gate = prediction_digest_gate(
            digests["accuracy"], digests["ovr_auc"]
        )
        gates.append(
            GateResult(
                f"{side}_{gate.name}",
                gate.passed,
                gate.observed,
                gate.threshold,
                gate.detail,
            )
        )
    evidence = {
        "data_scope": "training_validation_only",
        "run_names": sorted(_EVIDENCE_NAMES),
        "cross_mode_protocol_hash_equality_required": False,
        "source_fingerprint_compared_to_current_tree": False,
        "pilot": pilot_evidence,
        "isolet": {
            "legacy_accuracy_digest": legacy_digests["accuracy"],
            "legacy_ovr_auc_digest": legacy_digests["ovr_auc"],
            "candidate_accuracy_digest": candidate_digests["accuracy"],
            "candidate_ovr_auc_digest": candidate_digests["ovr_auc"],
        },
    }
    return {
        "passed": all(gate.passed for gate in gates),
        "gates": gates,
        "evidence": evidence,
    }


def _report_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "passed",
        "gates",
        "evidence",
    }:
        raise ValueError("acceptance report has an invalid schema")
    gates_value = value["gates"]
    if not isinstance(gates_value, Sequence) or isinstance(
        gates_value, (str, bytes)
    ):
        raise ValueError("acceptance report gates must be a sequence")
    gates: list[dict[str, object]] = []
    for gate in gates_value:
        if isinstance(gate, GateResult):
            gates.append(asdict(gate))
        elif isinstance(gate, Mapping) and set(gate) == {
            "name",
            "passed",
            "observed",
            "threshold",
            "detail",
        }:
            gates.append(dict(gate))
        else:
            raise ValueError("acceptance report gate is invalid")
    passed = value["passed"]
    if (
        not isinstance(passed, bool)
        or passed != all(gate["passed"] is True for gate in gates)
        or not isinstance(value["evidence"], Mapping)
    ):
        raise ValueError("acceptance report result is inconsistent")
    return {
        "passed": passed,
        "gates": gates,
        "evidence": dict(value["evidence"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate grouped acceleration without held-out test access."
    )
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--legacy-run", action="append", required=True)
    parser.add_argument("--candidate-run", action="append", required=True)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        legacy_runs = parse_named_runs(args.legacy_run)
        candidate_runs = parse_named_runs(args.candidate_run)
        report = _report_payload(
            build_acceptance_report(
                config_dir=args.config_dir,
                legacy_runs=legacy_runs,
                candidate_runs=candidate_runs,
            )
        )
        atomic_write_json(args.report, report)
    except (OSError, TypeError, ValueError) as exc:
        print(f"malformed acceptance evidence: {exc}", file=sys.stderr)
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
