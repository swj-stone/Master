"""Frozen, validated configuration for the ICOMP 2026 experiment rerun.

This module contains protocol mechanics that do not require model fitting or
dataset loading.  The first public boundary is :func:`load_resolved_protocol`,
which merges the global experiment configuration with per-dataset YAML files
and rejects ambiguous or misspelled settings before a run can start.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

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


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, _json_text(value))


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    _ensure_unique(controls, "duplicate removal controls")
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
    _ensure_unique(variants, "duplicate nsga2 variants")
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
