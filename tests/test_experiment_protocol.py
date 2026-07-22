"""Tests for the frozen ICOMP 2026 experiment protocol."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import re

import pytest
import yaml

import minppi_mofs.experiment_protocol as protocol_module
from minppi_mofs.experiment_protocol import (
    IncompleteRunError,
    ManifestStore,
    ResultRecord,
    StatusStore,
    assert_stage_complete,
    atomic_write_json,
    build_job_manifest,
    canonical_hash,
    create_new_run,
    freeze_run_metadata,
    load_resolved_protocol,
    open_resumed_run,
    publish_json_exclusive,
    read_data_manifest,
    read_rich_data_manifest,
    read_result_record,
    resolved_protocol_hash,
    upsert_data_manifest,
)


GLOBAL_CONFIG = {
    "protocol_version": "1.0",
    "datasets": ["spambase", "abalone"],
    "seeds": [42, 123],
    "backend": "cpu",
    "model": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_jobs": 1,
    },
    "alpha": {"default": 0.95, "grid": [0.85, 0.90, 0.95, 0.99]},
    "estimator": {
        "methods": ["exact", "random_mc", "mi_correlation_guided", "hybrid"],
        "budgets": [10, 25],
        "exact_max_features": 16,
    },
    "removal": {
        "controls": [
            "full",
            "minppi",
            "shap_bottom",
            "minshap_bottom",
            "banzhaf_bottom",
            "random_matched",
        ],
        "random_repeats": 30,
    },
    "nsga2": {
        "variants": ["full", "minppi_prefilter"],
        "population": 40,
        "evaluations": 800,
        "ref_point": [1.1, 1.1],
    },
    "ablations": {
        "utility": ["accuracy", "ovr_auc"],
        "guidance": ["random", "mi", "correlation", "mi_correlation"],
        "dispatcher": ["exact", "random_mc", "mi_correlation_guided", "hybrid"],
    },
    "smoke": {
        "datasets": ["spambase", "abalone"],
        "seeds": [42],
        "estimator_budgets": [2],
        "random_repeats": 2,
        "nsga2_population": 4,
        "nsga2_evaluations": 8,
    },
}

DATASET_CONFIGS = {
    "spambase": {
        "dataset": {
            "id": "spambase",
            "display_name": "Spambase",
            "raw_path": "data/raw/spambase.csv",
            "target": "is_spam",
            "task": "binary",
            "utility": "roc_auc",
        },
        "preprocessing": {"max_features": 100, "variance_threshold": 0.01},
        "split": {"train_frac": 0.64, "val_frac": 0.16, "test_frac": 0.20},
    },
    "abalone": {
        "dataset": {
            "id": "abalone",
            "display_name": "Abalone",
            "raw_path": "data/raw/abalone.csv",
            "target": "Rings",
            "task": "regression",
            "utility": "r2",
        },
        "preprocessing": {"max_features": None, "variance_threshold": 0.0},
        "split": {"train_frac": 0.64, "val_frac": 0.16, "test_frac": 0.20},
    },
}


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_protocol(
    config_dir: Path,
    *,
    global_config: dict | None = None,
    dataset_configs: dict[str, dict] | None = None,
) -> Path:
    config_dir.mkdir()
    _write_yaml(
        config_dir / "icomp2026_pipeline.yaml",
        deepcopy(global_config if global_config is not None else GLOBAL_CONFIG),
    )
    configs = dataset_configs if dataset_configs is not None else DATASET_CONFIGS
    for dataset_id, config in configs.items():
        _write_yaml(config_dir / f"{dataset_id}.yaml", deepcopy(config))
    return config_dir


def test_load_resolved_protocol_merges_global_and_dataset_configs(
    tmp_path: Path,
) -> None:
    config_dir = _write_protocol(tmp_path / "configs")

    protocol = load_resolved_protocol(
        config_dir,
        seeds=(42,),
        backend="cpu",
        smoke=True,
    )

    assert protocol.protocol_version == "1.0"
    assert protocol.seeds == (42,)
    assert protocol.backend == "cpu"
    assert protocol.run_kind == "smoke"
    assert tuple(d.dataset_id for d in protocol.datasets) == (
        "spambase",
        "abalone",
    )
    assert protocol.datasets[0].task == "binary"
    assert protocol.datasets[0].utility == "roc_auc"
    assert protocol.datasets[1].task == "regression"
    assert protocol.datasets[1].utility == "r2"
    assert protocol.estimator_budgets == (2,)
    assert protocol.removal["random_repeats"] == 2
    assert protocol.nsga2["population"] == 4
    assert protocol.nsga2["evaluations"] == 8


def test_canonical_hash_is_independent_of_mapping_order() -> None:
    left = {"b": [2, 3], "a": {"y": 1, "x": 0}}
    right = {"a": {"x": 0, "y": 1}, "b": [2, 3]}

    assert canonical_hash(left) == canonical_hash(right)


def test_repository_protocol_configs_are_valid() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    formal = load_resolved_protocol(
        config_dir,
        seeds=None,
        backend=None,
        smoke=False,
    )
    smoke = load_resolved_protocol(
        config_dir,
        seeds=(42,),
        backend="cuda",
        smoke=True,
    )

    assert len(formal.datasets) == 7
    assert formal.seeds == (42, 123, 456, 789, 1024)
    assert tuple(dataset.dataset_id for dataset in smoke.datasets) == (
        "spambase",
        "abalone",
    )
    assert smoke.backend == "cuda"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda g, _d: g.update({"unknown": True}), "unknown"),
        (lambda g, _d: g["alpha"].update({"default": 1.2}), "alpha"),
        (lambda g, _d: g["estimator"].update({"budgets": [10, 0]}), "budget"),
        (lambda g, _d: g.update({"seeds": [42, 42]}), "seed"),
        (
            lambda g, _d: g["nsga2"].update({"ref_point": [1.0, 1.0]}),
            "ref_point",
        ),
        (
            lambda g, _d: g.update({"datasets": ["spambase", "spambase"]}),
            "duplicate",
        ),
        (
            lambda _g, d: d["spambase"]["dataset"].pop("target"),
            "target",
        ),
        (
            lambda _g, d: d["spambase"]["dataset"].update(
                {"task": "regression", "utility": "roc_auc"}
            ),
            "utility",
        ),
    ],
)
def test_invalid_protocol_is_rejected(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    global_config = deepcopy(GLOBAL_CONFIG)
    dataset_configs = deepcopy(DATASET_CONFIGS)
    mutator(global_config, dataset_configs)
    config_dir = _write_protocol(
        tmp_path / "configs",
        global_config=global_config,
        dataset_configs=dataset_configs,
    )

    with pytest.raises(ValueError, match=message):
        load_resolved_protocol(
            config_dir,
            seeds=None,
            backend=None,
            smoke=False,
        )


@pytest.fixture
def resolved_protocol(tmp_path: Path):
    config_dir = _write_protocol(tmp_path / "protocol_configs")
    return load_resolved_protocol(
        config_dir,
        seeds=(42,),
        backend="cpu",
        smoke=True,
    )


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


def _data_manifest_row(
    protocol, dataset_id: str, seed: int = 42
) -> dict[str, object]:
    dataset = next(item for item in protocol.datasets if item.dataset_id == dataset_id)
    job = next(
        item
        for item in build_job_manifest(protocol, stages=("preprocess",))
        if item.dataset_id == dataset_id and item.seed == seed
    )
    prefix = f"preprocess/{dataset_id}/s{seed}"
    if dataset.task == "regression":
        summaries = {
            "train": {"count": 6, "max": 5.0, "mean": 2.5, "min": 0.0, "std": 1.707825127659933},
            "val": {"count": 2, "max": 7.0, "mean": 6.5, "min": 6.0, "std": 0.5},
            "test": {"count": 2, "max": 9.0, "mean": 8.5, "min": 8.0, "std": 0.5},
        }
    else:
        summaries = {
            "train": {"classes": [{"count": 3, "label": 0}, {"count": 3, "label": 1}], "count": 6},
            "val": {"classes": [{"count": 1, "label": 0}, {"count": 1, "label": 1}], "count": 2},
            "test": {"classes": [{"count": 1, "label": 0}, {"count": 1, "label": 1}], "count": 2},
        }
    return {
        "protocol_hash": resolved_protocol_hash(protocol),
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "seed": seed,
        "task": dataset.task,
        "utility": dataset.utility,
        "target": dataset.target,
        "raw_file_sha256": "1" * 64,
        "raw_row_count": 10,
        "raw_feature_count": 4,
        "selected_feature_count": 2,
        "selected_features_json": '["f0","f1"]',
        "train_relpath": f"{prefix}/train.csv",
        "val_relpath": f"{prefix}/val.csv",
        "test_relpath": f"{prefix}/test.csv",
        "train_row_count": 6,
        "val_row_count": 2,
        "test_row_count": 2,
        "train_index_sha256": "2" * 64,
        "val_index_sha256": "3" * 64,
        "test_index_sha256": "4" * 64,
        "train_file_sha256": "5" * 64,
        "val_file_sha256": "6" * 64,
        "test_file_sha256": "7" * 64,
        "train_label_summary_json": json.dumps(summaries["train"], sort_keys=True, separators=(",", ":")),
        "val_label_summary_json": json.dumps(summaries["val"], sort_keys=True, separators=(",", ":")),
        "test_label_summary_json": json.dumps(summaries["test"], sort_keys=True, separators=(",", ":")),
        "transform_metadata_json": '{"scaler_mean":[0.0,1.0]}',
    }


def test_data_manifest_partial_upsert_is_canonical_idempotent_and_complete_read_projects(
    tmp_path: Path, resolved_protocol
) -> None:
    path = tmp_path / "data_manifest.csv"
    abalone = _data_manifest_row(resolved_protocol, "abalone")
    spambase = _data_manifest_row(resolved_protocol, "spambase")

    upsert_data_manifest(path, resolved_protocol, abalone)
    partial = path.read_bytes()
    upsert_data_manifest(path, resolved_protocol, abalone)
    assert path.read_bytes() == partial
    with pytest.raises(ValueError, match="complete|scope"):
        read_data_manifest(path, resolved_protocol)

    upsert_data_manifest(path, resolved_protocol, spambase)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == _DATA_MANIFEST_FIELDS
    assert [row["dataset_id"] for row in rows] == ["spambase", "abalone"]
    assert read_data_manifest(path, resolved_protocol) == (
        {"dataset_id": "spambase", "seed": 42, "selected_feature_count": 2},
        {"dataset_id": "abalone", "seed": 42, "selected_feature_count": 2},
    )

    rich = read_rich_data_manifest(path, resolved_protocol)
    assert rich == (spambase, abalone)
    rich[0]["dataset_id"] = "mutated"
    assert read_rich_data_manifest(path, resolved_protocol)[0]["dataset_id"] == "spambase"


def test_data_manifest_rejects_conflict_and_malformed_rich_fields_without_mutation(
    tmp_path: Path, resolved_protocol
) -> None:
    path = tmp_path / "data_manifest.csv"
    row = _data_manifest_row(resolved_protocol, "spambase")
    upsert_data_manifest(path, resolved_protocol, row)
    frozen = path.read_bytes()

    conflict = dict(row, raw_file_sha256="a" * 64)
    with pytest.raises(ValueError, match="conflict|frozen"):
        upsert_data_manifest(path, resolved_protocol, conflict)
    assert path.read_bytes() == frozen

    malformed = dict(_data_manifest_row(resolved_protocol, "abalone"))
    malformed["test_relpath"] = "../escape.csv"
    with pytest.raises(ValueError, match="test_relpath|path"):
        upsert_data_manifest(path, resolved_protocol, malformed)
    assert path.read_bytes() == frozen

    malformed = dict(_data_manifest_row(resolved_protocol, "abalone"))
    malformed["transform_metadata_json"] = '{"bad":NaN}'
    with pytest.raises(ValueError, match="JSON|finite|canonical"):
        upsert_data_manifest(path, resolved_protocol, malformed)
    assert path.read_bytes() == frozen


@pytest.mark.parametrize(
    ("dataset_id", "field", "summary"),
    [
        ("spambase", "train_label_summary_json", {"count": 6}),
        ("spambase", "train_label_summary_json", {"classes": [{"count": 6, "label": True}], "count": 6}),
        ("spambase", "train_label_summary_json", {"classes": [{"count": 5, "label": 0}], "count": 6}),
        ("abalone", "train_label_summary_json", {"count": 6, "max": 5.0, "mean": 2.5, "min": 0.0}),
        ("abalone", "train_label_summary_json", {"count": True, "max": 5.0, "mean": 2.5, "min": 0.0, "std": 1.0}),
        ("abalone", "train_label_summary_json", {"count": 6, "max": float("inf"), "mean": 2.5, "min": 0.0, "std": 1.0}),
    ],
)
def test_rich_manifest_rejects_invalid_task_specific_label_summary(
    tmp_path: Path, resolved_protocol, dataset_id: str, field: str, summary: object
) -> None:
    row = _data_manifest_row(resolved_protocol, dataset_id)
    row[field] = json.dumps(summary, sort_keys=True, separators=(",", ":"))

    with pytest.raises(ValueError, match="label_summary|summary|finite|canonical"):
        upsert_data_manifest(tmp_path / "data_manifest.csv", resolved_protocol, row)


def _multiclass_protocol(protocol):
    return replace(
        protocol,
        datasets=tuple(
            replace(dataset, task="multiclass", utility="ovr_auc")
            if dataset.dataset_id == "spambase"
            else dataset
            for dataset in protocol.datasets
        ),
    )


@pytest.mark.parametrize(
    ("field", "summary"),
    [
        (
            "test_label_summary_json",
            {"classes": [{"count": 2, "label": 2}], "count": 2},
        ),
        (
            "train_label_summary_json",
            {"classes": [{"count": 6, "label": 0}], "count": 6},
        ),
    ],
)
def test_binary_manifest_rejects_cross_partition_or_invalid_train_classes(
    tmp_path: Path,
    resolved_protocol,
    field: str,
    summary: dict[str, object],
) -> None:
    row = _data_manifest_row(resolved_protocol, "spambase")
    row[field] = json.dumps(summary, sort_keys=True, separators=(",", ":"))

    with pytest.raises(ValueError, match="binary|class|label|train|partition"):
        upsert_data_manifest(
            tmp_path / "data_manifest.csv", resolved_protocol, row
        )


@pytest.mark.parametrize("partition", ["val", "test"])
def test_multiclass_manifest_rejects_unseen_partition_class(
    tmp_path: Path, resolved_protocol, partition: str
) -> None:
    protocol = _multiclass_protocol(resolved_protocol)
    row = _data_manifest_row(protocol, "spambase")
    row[f"{partition}_label_summary_json"] = json.dumps(
        {"classes": [{"count": 2, "label": 2}], "count": 2},
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(ValueError, match="multiclass|class|label|train|partition"):
        upsert_data_manifest(tmp_path / "data_manifest.csv", protocol, row)


def test_multiclass_manifest_rejects_noncontiguous_train_classes(
    tmp_path: Path, resolved_protocol
) -> None:
    protocol = _multiclass_protocol(resolved_protocol)
    row = _data_manifest_row(protocol, "spambase")
    row["train_label_summary_json"] = json.dumps(
        {
            "classes": [
                {"count": 3, "label": 0},
                {"count": 3, "label": 2},
            ],
            "count": 6,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(ValueError, match="multiclass|class|label|train"):
        upsert_data_manifest(tmp_path / "data_manifest.csv", protocol, row)


def test_multiclass_manifest_allows_val_and_test_to_omit_train_classes(
    tmp_path: Path, resolved_protocol
) -> None:
    protocol = _multiclass_protocol(resolved_protocol)
    row = _data_manifest_row(protocol, "spambase")
    summaries = {
        "train": {
            "classes": [
                {"count": 2, "label": 0},
                {"count": 2, "label": 1},
                {"count": 2, "label": 2},
            ],
            "count": 6,
        },
        "val": {"classes": [{"count": 2, "label": 1}], "count": 2},
        "test": {
            "classes": [
                {"count": 1, "label": 0},
                {"count": 1, "label": 2},
            ],
            "count": 2,
        },
    }
    for partition, summary in summaries.items():
        row[f"{partition}_label_summary_json"] = json.dumps(
            summary, sort_keys=True, separators=(",", ":")
        )

    upsert_data_manifest(tmp_path / "data_manifest.csv", protocol, row)


def test_data_manifest_atomic_upsert_preserves_previous_file_on_replace_failure(
    tmp_path: Path, resolved_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data_manifest.csv"
    upsert_data_manifest(path, resolved_protocol, _data_manifest_row(resolved_protocol, "spambase"))
    frozen = path.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(protocol_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        upsert_data_manifest(path, resolved_protocol, _data_manifest_row(resolved_protocol, "abalone"))
    assert path.read_bytes() == frozen
    assert not tuple(tmp_path.glob(f".{path.name}.*.tmp"))


def test_data_manifest_partial_upsert_rejects_cross_seed_raw_authority_drift(
    tmp_path: Path, resolved_protocol
) -> None:
    protocol = replace(resolved_protocol, seeds=(42, 123))
    path = tmp_path / "data_manifest.csv"
    first = _data_manifest_row(protocol, "spambase", 42)
    second = dict(
        _data_manifest_row(protocol, "spambase", 123),
        raw_file_sha256="a" * 64,
        raw_row_count=11,
        train_row_count=7,
        train_label_summary_json=(
            '{"classes":[{"count":4,"label":0},{"count":3,"label":1}],"count":7}'
        ),
    )
    upsert_data_manifest(path, protocol, first)
    frozen = path.read_bytes()

    with pytest.raises(ValueError, match="raw|authority|seed"):
        upsert_data_manifest(path, protocol, second)

    assert path.read_bytes() == frozen


def test_data_manifest_complete_read_rejects_cross_seed_raw_authority_drift(
    tmp_path: Path, resolved_protocol
) -> None:
    protocol = replace(resolved_protocol, seeds=(42, 123))
    path = tmp_path / "data_manifest.csv"
    rows = []
    for dataset in protocol.datasets:
        for seed in protocol.seeds:
            row = _data_manifest_row(protocol, dataset.dataset_id, seed)
            if dataset.dataset_id == "spambase" and seed == 123:
                row = dict(row, raw_file_sha256="b" * 64)
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_DATA_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="raw|authority|seed"):
        read_data_manifest(path, protocol)


def test_create_new_run_builds_canonical_layout(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    layout = create_new_run(
        tmp_path / "results",
        "run-1",
        resolved_protocol,
    )

    assert layout.root == tmp_path / "results" / "run-1"
    assert layout.root.is_dir()
    assert layout.configs_frozen.is_dir()
    assert layout.raw_metrics.is_dir()
    assert layout.tables.is_dir()
    assert layout.figures.is_dir()
    assert layout.audit_path == layout.root / "protocol_audit.json"
    assert layout.manifest_path == layout.root / "run_manifest.csv"
    assert layout.status_path == layout.root / "status.json"
    assert layout.log_path == layout.root / "output.log"

    with pytest.raises(FileExistsError, match="already exists"):
        create_new_run(tmp_path / "results", "run-1", resolved_protocol)


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "has space", "../escape", "a/b", "a\\b"],
)
def test_create_new_run_rejects_unsafe_run_ids(
    tmp_path: Path,
    resolved_protocol,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="run_id"):
        create_new_run(tmp_path / "results", run_id, resolved_protocol)


@pytest.mark.parametrize("run_id", [".", ".."])
def test_resume_rejects_special_path_run_ids(
    tmp_path: Path,
    resolved_protocol,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="run_id"):
        open_resumed_run(
            tmp_path / "results",
            run_id,
            resolved_protocol,
            project_root=tmp_path,
        )


def test_create_new_run_generates_protocol_bound_id(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    layout = create_new_run(tmp_path / "results", None, resolved_protocol)

    assert re.fullmatch(
        r"\d{8}T\d{6}Z-[0-9a-f]{10}",
        layout.root.name,
    )


def _write_project_sources(project_root: Path) -> Path:
    source = project_root / "src" / "package" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    experiment = project_root / "experiments" / "runner.py"
    experiment.parent.mkdir()
    experiment.write_text("STAGE = 'validate'\n", encoding="utf-8")
    return source


def _refresh_marker_hash(layout, relative_path: str) -> None:
    artifact = layout.root / relative_path
    marker_path = layout.root / "metadata_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifacts"][relative_path] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def test_freeze_run_metadata_records_protocol_environment_and_sources(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)

    freeze_run_metadata(
        layout,
        resolved_protocol,
        project_root=project_root,
    )

    assert (layout.root / "resolved_protocol.yaml").is_file()
    environment = json.loads(
        (layout.root / "environment.json").read_text(encoding="utf-8")
    )
    fingerprint = json.loads(
        (layout.root / "source_fingerprint.json").read_text(encoding="utf-8")
    )
    assert environment["python"]
    assert environment["xgboost"]
    assert environment["backend"] == "cpu"
    assert {
        "numpy",
        "pandas",
        "scikit_learn",
        "xgboost_build_info",
        "cuda_build_info",
        "cuda_driver",
        "os",
        "timezone",
        "utc_offset",
    } <= set(environment)
    assert fingerprint["protocol_hash"]
    assert set(fingerprint["files"]) == {
        "experiments/runner.py",
        "src/package/module.py",
    }
    assert (layout.configs_frozen / "icomp2026_pipeline.resolved.yaml").is_file()
    assert (layout.configs_frozen / "spambase.yaml").is_file()
    assert (layout.configs_frozen / "abalone.yaml").is_file()
    assert (layout.root / "metadata_complete.json").is_file()


def test_freeze_run_metadata_refuses_to_replace_frozen_metadata(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    freeze_run_metadata(layout, resolved_protocol, project_root=project_root)

    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_run_metadata(layout, resolved_protocol, project_root=project_root)


def test_failed_metadata_freeze_never_publishes_completion_marker(
    tmp_path: Path,
    resolved_protocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    real_replace = protocol_module.os.replace

    def fail_on_environment(source, destination) -> None:
        if Path(destination).name == "environment.json":
            raise OSError("injected metadata write failure")
        real_replace(source, destination)

    monkeypatch.setattr(protocol_module.os, "replace", fail_on_environment)
    with pytest.raises(OSError, match="injected"):
        freeze_run_metadata(layout, resolved_protocol, project_root=project_root)

    assert not (layout.root / "metadata_complete.json").exists()
    with pytest.raises(RuntimeError, match="incomplete"):
        open_resumed_run(
            tmp_path / "results",
            "run-1",
            resolved_protocol,
            project_root=project_root,
        )


def test_freeze_run_metadata_rejects_unsafe_dataset_id(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    unsafe_dataset = replace(
        resolved_protocol.datasets[0],
        dataset_id="../escape",
    )
    unsafe_protocol = replace(
        resolved_protocol,
        datasets=(unsafe_dataset,),
    )
    layout = create_new_run(tmp_path / "results", "run-1", unsafe_protocol)

    with pytest.raises(ValueError, match="dataset_id"):
        freeze_run_metadata(
            layout,
            unsafe_protocol,
            project_root=project_root,
        )
    assert not (layout.root / "escape.yaml").exists()


def test_resume_accepts_unchanged_run(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    freeze_run_metadata(layout, resolved_protocol, project_root=project_root)

    resumed = open_resumed_run(
        tmp_path / "results",
        "run-1",
        resolved_protocol,
        project_root=project_root,
    )

    assert resumed == layout


def test_resume_ignores_informational_git_metadata_changes(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    freeze_run_metadata(layout, resolved_protocol, project_root=project_root)
    fingerprint_path = layout.root / "source_fingerprint.json"
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    fingerprint["git_commit"] = "new-commit-with-identical-sources"
    fingerprint["git_dirty"] = not bool(fingerprint["git_dirty"])
    fingerprint_path.write_text(
        json.dumps(fingerprint),
        encoding="utf-8",
    )
    _refresh_marker_hash(layout, "source_fingerprint.json")

    resumed = open_resumed_run(
        tmp_path / "results",
        "run-1",
        resolved_protocol,
        project_root=project_root,
    )

    assert resumed == layout


def test_resume_rejects_source_drift(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    project_root = tmp_path / "project"
    source = _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    freeze_run_metadata(layout, resolved_protocol, project_root=project_root)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fingerprint"):
        open_resumed_run(
            tmp_path / "results",
            "run-1",
            resolved_protocol,
            project_root=project_root,
        )


def test_resume_rejects_protocol_drift(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    freeze_run_metadata(layout, resolved_protocol, project_root=project_root)
    changed = replace(resolved_protocol, alpha_default=0.90)

    with pytest.raises(RuntimeError, match="protocol"):
        open_resumed_run(
            tmp_path / "results",
            "run-1",
            changed,
            project_root=project_root,
        )


@pytest.mark.parametrize(
    ("relative_path", "delete"),
    [
        ("environment.json", True),
        ("configs_frozen/spambase.yaml", False),
    ],
)
def test_resume_rejects_missing_or_tampered_frozen_metadata(
    tmp_path: Path,
    resolved_protocol,
    relative_path: str,
    delete: bool,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    freeze_run_metadata(layout, resolved_protocol, project_root=project_root)
    artifact = layout.root / relative_path
    if delete:
        artifact.unlink()
    else:
        artifact.write_text("tampered: true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata"):
        open_resumed_run(
            tmp_path / "results",
            "run-1",
            resolved_protocol,
            project_root=project_root,
        )


def test_resume_rejects_partially_frozen_metadata(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    (layout.root / "resolved_protocol.yaml").write_text(
        "protocol_version: incomplete\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        open_resumed_run(
            tmp_path / "results",
            "run-1",
            resolved_protocol,
            project_root=tmp_path,
        )


def test_resume_rejects_non_mapping_source_fingerprint(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    project_root = tmp_path / "project"
    _write_project_sources(project_root)
    layout = create_new_run(tmp_path / "results", "run-1", resolved_protocol)
    freeze_run_metadata(layout, resolved_protocol, project_root=project_root)
    fingerprint_path = layout.root / "source_fingerprint.json"
    fingerprint_path.write_text("[]\n", encoding="utf-8")
    _refresh_marker_hash(layout, "source_fingerprint.json")

    with pytest.raises(RuntimeError, match="source fingerprint metadata"):
        open_resumed_run(
            tmp_path / "results",
            "run-1",
            resolved_protocol,
            project_root=project_root,
        )


def test_resume_requires_existing_run(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        open_resumed_run(
            tmp_path / "results",
            "missing",
            resolved_protocol,
            project_root=tmp_path,
        )


def test_resume_requires_a_valid_run_id(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    with pytest.raises(ValueError, match="run_id"):
        open_resumed_run(
            tmp_path / "results",
            None,  # type: ignore[arg-type]
            resolved_protocol,
            project_root=tmp_path,
        )


@pytest.mark.parametrize("dataset_id", [".", "..", "../escape", "C:\\escape"])
def test_protocol_rejects_unsafe_dataset_ids(
    tmp_path: Path,
    dataset_id: str,
) -> None:
    global_config = deepcopy(GLOBAL_CONFIG)
    global_config["datasets"] = [dataset_id]
    config_dir = _write_protocol(
        tmp_path / "configs",
        global_config=global_config,
    )

    with pytest.raises(ValueError, match="safe dataset ID"):
        load_resolved_protocol(
            config_dir,
            seeds=None,
            backend=None,
            smoke=False,
        )


def test_protocol_rejects_unsafe_id_inside_dataset_file(tmp_path: Path) -> None:
    dataset_configs = deepcopy(DATASET_CONFIGS)
    dataset_configs["spambase"]["dataset"]["id"] = "../escape"
    config_dir = _write_protocol(
        tmp_path / "configs",
        dataset_configs=dataset_configs,
    )

    with pytest.raises(ValueError, match="safe dataset ID"):
        load_resolved_protocol(
            config_dir,
            seeds=(42,),
            backend="cpu",
            smoke=True,
        )


def test_job_manifest_is_deterministic_unique_and_uses_smoke_scope(
    resolved_protocol,
) -> None:
    first = build_job_manifest(
        resolved_protocol, stages=("preprocess", "removal", "alpha")
    )
    reordered = replace(
        resolved_protocol,
        removal={
            "random_repeats": 2,
            "controls": resolved_protocol.removal["controls"],
        },
    )
    second = build_job_manifest(
        reordered, stages=("preprocess", "removal", "alpha")
    )

    assert [job.job_id for job in first] == [job.job_id for job in second]
    assert len({job.job_id for job in first}) == len(first)
    assert {job.dataset_id for job in first} == {"spambase", "abalone"}
    assert {job.seed for job in first} == {42}


def test_job_manifest_expands_exact_stage_dimensions_and_dependencies(
    resolved_protocol,
) -> None:
    data_manifest = (
        {"dataset_id": "spambase", "seed": 42, "selected_feature_count": 10},
        {"dataset_id": "abalone", "seed": 42, "selected_feature_count": 20},
    )
    jobs = build_job_manifest(
        resolved_protocol,
        stages=(
            "validate",
            "preprocess",
            "estimator",
            "removal",
            "alpha",
            "nsga2",
            "ablation",
            "summarize",
            "figures",
        ),
        data_manifest=data_manifest,
    )
    by_stage = {
        stage: tuple(job for job in jobs if job.stage == stage)
        for stage in {job.stage for job in jobs}
    }

    assert len(by_stage["validate"]) == 1
    assert len(by_stage["preprocess"]) == 2
    assert len(by_stage["estimator"]) == 7
    assert len(by_stage["removal"]) == 14
    assert len(by_stage["alpha"]) == 10
    assert len(by_stage["nsga2"]) == 4
    assert len(by_stage["ablation"]) == 15
    assert len(by_stage["summarize"]) == len(by_stage["figures"]) == 1
    assert all(
        job.depends_on
        for job in by_stage["removal"]
        if job.method != "selection_plan"
    )
    assert all(
        job.depends_on
        for job in by_stage["alpha"]
        if job.method != "selection_plan"
    )
    assert all(
        job.output_relpath == f"raw_metrics/{job.stage}/{job.job_id}.json"
        for job in jobs
    )


@pytest.mark.parametrize(
    "stages",
    [("unknown",), ("preprocess", "preprocess")],
)
def test_job_manifest_rejects_unknown_or_duplicate_stages(
    resolved_protocol,
    stages: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="stage"):
        build_job_manifest(resolved_protocol, stages=stages)


def test_estimator_manifest_requires_complete_valid_data_manifest(
    resolved_protocol,
) -> None:
    with pytest.raises(ValueError, match="data_manifest"):
        build_job_manifest(resolved_protocol, stages=("estimator",))
    with pytest.raises(ValueError, match="selected_feature_count"):
        build_job_manifest(
            resolved_protocol,
            stages=("estimator",),
            data_manifest=(
                {"dataset_id": "spambase", "seed": 42, "selected_feature_count": True},
                {"dataset_id": "abalone", "seed": 42, "selected_feature_count": 1},
            ),
        )


def test_atomic_write_json_replaces_from_the_destination_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "nested" / "record.json"
    observed: list[tuple[Path, Path]] = []
    real_replace = protocol_module.os.replace

    def record_replace(source: Path, target: Path) -> None:
        observed.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(protocol_module.os, "replace", record_replace)
    atomic_write_json(destination, {"a": 1})

    assert observed and observed[0][0].parent == observed[0][1].parent
    assert json.loads(destination.read_text(encoding="utf-8")) == {"a": 1}


def test_read_result_record_is_canonical_and_strict(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    expected_job = build_job_manifest(resolved_protocol, stages=("preprocess",))[0]
    record = ResultRecord(
        job_id=expected_job.job_id,
        protocol_hash=expected_job.protocol_hash,
        stage=expected_job.stage,
        dataset_id=expected_job.dataset_id,
        seed=expected_job.seed,
        method=expected_job.method,
        metrics={"score": 1.0, "complete": True, "note": None},
        artifacts={"split": "preprocess/spambase/s42/train.csv"},
        started_at="2026-07-21T00:00:00+00:00",
        finished_at="2026-07-21T00:00:01+00:00",
    )
    path = tmp_path / "record.json"
    atomic_write_json(path, record.__dict__)

    assert read_result_record(path, expected_job) == record


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "wrong"),
        ("protocol_hash", "wrong"),
        ("stage", "wrong"),
        ("dataset_id", "wrong"),
        ("seed", 43),
        ("seed", True),
        ("method", "wrong"),
        ("metrics", {"score": float("nan")}),
        ("metrics", {"score": float("inf")}),
        ("metrics", {"score": -float("inf")}),
        ("metrics", []),
        ("metrics", {"": 1.0}),
        ("metrics", {"score": []}),
        ("artifacts", {"split": "../escape"}),
        ("artifacts", {"split": "/absolute"}),
        ("artifacts", {"split": "C:drive-relative"}),
        ("artifacts", {"split": "folder\\file"}),
        ("started_at", "not-a-timestamp"),
        ("started_at", "2026-07-21T00:00:00"),
    ],
)
def test_read_result_record_rejects_mismatched_or_invalid_schema(
    tmp_path: Path,
    resolved_protocol,
    field: str,
    value: object,
) -> None:
    expected_job = build_job_manifest(resolved_protocol, stages=("preprocess",))[0]
    payload = {
        "job_id": expected_job.job_id,
        "protocol_hash": expected_job.protocol_hash,
        "stage": expected_job.stage,
        "dataset_id": expected_job.dataset_id,
        "seed": expected_job.seed,
        "method": expected_job.method,
        "metrics": {"score": 1.0},
        "artifacts": {"split": "preprocess/spambase/s42/train.csv"},
        "started_at": "2026-07-21T00:00:00+00:00",
        "finished_at": "2026-07-21T00:00:01+00:00",
    }
    payload[field] = value
    path = tmp_path / "record.json"
    if field == "metrics":
        path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    else:
        atomic_write_json(path, payload)

    with pytest.raises(ValueError):
        read_result_record(path, expected_job)


@pytest.mark.parametrize("extra", [False, True])
def test_read_result_record_rejects_missing_or_extra_top_level_keys(
    tmp_path: Path,
    resolved_protocol,
    extra: bool,
) -> None:
    expected_job = build_job_manifest(resolved_protocol, stages=("preprocess",))[0]
    payload = {
        "job_id": expected_job.job_id,
        "protocol_hash": expected_job.protocol_hash,
        "stage": expected_job.stage,
        "dataset_id": expected_job.dataset_id,
        "seed": expected_job.seed,
        "method": expected_job.method,
        "metrics": {},
        "artifacts": {},
        "started_at": "2026-07-21T00:00:00+00:00",
        "finished_at": "2026-07-21T00:00:01+00:00",
    }
    if extra:
        payload["unexpected"] = True
    else:
        del payload["method"]
    path = tmp_path / "record.json"
    atomic_write_json(path, payload)

    with pytest.raises(ValueError, match="schema"):
        read_result_record(path, expected_job)


def test_read_result_record_rejects_reversed_timestamps(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    expected_job = build_job_manifest(resolved_protocol, stages=("preprocess",))[0]
    payload = {
        "job_id": expected_job.job_id,
        "protocol_hash": expected_job.protocol_hash,
        "stage": expected_job.stage,
        "dataset_id": expected_job.dataset_id,
        "seed": expected_job.seed,
        "method": expected_job.method,
        "metrics": {},
        "artifacts": {},
        "started_at": "2026-07-21T00:00:02+00:00",
        "finished_at": "2026-07-21T00:00:01+00:00",
    }
    path = tmp_path / "record.json"
    atomic_write_json(path, payload)

    with pytest.raises(ValueError, match="precedes"):
        read_result_record(path, expected_job)


def test_read_result_record_rejects_malformed_json_and_frozen_dependency_tuple(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    expected_job = build_job_manifest(resolved_protocol, stages=("preprocess",))[0]
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        read_result_record(malformed, expected_job)
    with pytest.raises(FrozenInstanceError):
        expected_job.depends_on += ("other",)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("removal", "controls", ["full", "minppi"]),
        (
            "removal",
            "controls",
            [
                "minppi",
                "full",
                "shap_bottom",
                "minshap_bottom",
                "banzhaf_bottom",
                "random_matched",
            ],
        ),
        (
            "removal",
            "controls",
            [
                "full",
                "minppi",
                "shap_bottom",
                "minshap_bottom",
                "banzhaf_bottom",
                "invalid_control",
            ],
        ),
        ("nsga2", "variants", ["full"]),
        ("nsga2", "variants", ["minppi_prefilter", "full"]),
    ],
)
def test_protocol_requires_canonical_removal_controls_and_nsga_variants(
    tmp_path: Path,
    section: str,
    key: str,
    value: list[str],
) -> None:
    global_config = deepcopy(GLOBAL_CONFIG)
    global_config[section][key] = value
    config_dir = _write_protocol(tmp_path / "configs", global_config=global_config)

    with pytest.raises(ValueError, match="canonical"):
        load_resolved_protocol(
            config_dir, seeds=(42,), backend="cpu", smoke=True
        )


def test_dataset_jobs_include_task_and_utility_metadata(resolved_protocol) -> None:
    rows = (
        {"dataset_id": "spambase", "seed": 42, "selected_feature_count": 10},
        {"dataset_id": "abalone", "seed": 42, "selected_feature_count": 20},
    )
    jobs = build_job_manifest(
        resolved_protocol,
        stages=("preprocess", "estimator", "removal", "alpha", "nsga2", "ablation"),
        data_manifest=rows,
    )
    expected = {
        dataset.dataset_id: (dataset.task, dataset.utility)
        for dataset in resolved_protocol.datasets
    }

    assert all(
        job.parameters["task"] == expected[job.dataset_id][0]
        and job.parameters["utility"] == expected[job.dataset_id][1]
        for job in jobs
    )
    assert {
        job.method for job in jobs if job.stage == "removal"
    } == {
        "selection_plan",
        "full",
        "minppi",
        "shap_bottom",
        "minshap_bottom",
        "banzhaf_bottom",
        "random_matched",
    }
    assert {
        job.method for job in jobs if job.stage == "nsga2"
    } == {"full", "minppi_prefilter"}
    assert all(
        job.parameters["ref_point"] == [1.1, 1.1]
        for job in jobs
        if job.stage == "nsga2"
    )


@pytest.mark.parametrize(
    "invalid_rows",
    [
        ({"dataset_id": "spambase", "seed": 42, "selected_feature_count": 0},),
        ({"dataset_id": "spambase", "seed": 42, "selected_feature_count": -1},),
        ({"dataset_id": "spambase", "seed": 42, "selected_feature_count": 2.0},),
        ({"dataset_id": "spambase", "seed": 42, "selected_feature_count": True},),
        ({"dataset_id": "spambase", "seed": 42},),
    ],
)
def test_data_manifest_rejects_invalid_feature_count_or_row(
    resolved_protocol,
    invalid_rows: tuple[dict[str, object], ...],
) -> None:
    with pytest.raises(ValueError, match="data_manifest|selected_feature_count"):
        build_job_manifest(
            resolved_protocol, stages=("estimator",), data_manifest=invalid_rows
        )


@pytest.mark.parametrize(
    "rows",
    [
        (
            {"dataset_id": "spambase", "seed": 42, "selected_feature_count": 1},
            {"dataset_id": "spambase", "seed": 42, "selected_feature_count": 2},
        ),
        ({"dataset_id": "spambase", "seed": 42, "selected_feature_count": 1},),
        (
            {"dataset_id": "spambase", "seed": 42, "selected_feature_count": 1},
            {"dataset_id": "outside", "seed": 42, "selected_feature_count": 1},
        ),
        (
            {"dataset_id": "spambase", "seed": 42, "selected_feature_count": 1},
            {"dataset_id": "abalone", "seed": 42, "selected_feature_count": 1},
            {"dataset_id": "outside", "seed": 42, "selected_feature_count": 1},
        ),
    ],
)
def test_data_manifest_rejects_duplicate_missing_or_out_of_scope_rows(
    resolved_protocol,
    rows: tuple[dict[str, object], ...],
) -> None:
    with pytest.raises(ValueError, match="data_manifest"):
        build_job_manifest(
            resolved_protocol, stages=("estimator",), data_manifest=rows
        )


def test_ablation_requires_data_manifest_and_utility_axis_is_isolet_only(
    tmp_path: Path,
    resolved_protocol,
) -> None:
    with pytest.raises(ValueError, match="data_manifest"):
        build_job_manifest(resolved_protocol, stages=("ablation",))

    global_config = deepcopy(GLOBAL_CONFIG)
    global_config["datasets"] = ["spambase", "isolet"]
    global_config["smoke"]["datasets"] = ["spambase", "isolet"]
    dataset_configs = deepcopy(DATASET_CONFIGS)
    dataset_configs["isolet"] = deepcopy(DATASET_CONFIGS["spambase"])
    dataset_configs["isolet"]["dataset"].update(
        {"id": "isolet", "display_name": "ISOLET", "task": "multiclass", "utility": "ovr_auc"}
    )
    formal = load_resolved_protocol(
        _write_protocol(
            tmp_path / "configs", global_config=global_config, dataset_configs=dataset_configs
        ),
        seeds=(42,),
        backend="cpu",
        smoke=False,
    )
    jobs = build_job_manifest(
        formal,
        stages=("ablation",),
        data_manifest=(
            {"dataset_id": "spambase", "seed": 42, "selected_feature_count": 20},
            {"dataset_id": "isolet", "seed": 42, "selected_feature_count": 10},
        ),
    )

    assert len(jobs) == 17
    assert {
        job.parameters["value"]
        for job in jobs
        if job.parameters["axis"] == "utility"
    } == {"accuracy", "ovr_auc"}
    assert {
        job.dataset_id for job in jobs if job.parameters["axis"] == "utility"
    } == {"isolet"}
    assert "exact" not in {
        job.parameters["value"]
        for job in jobs
        if job.dataset_id == "spambase" and job.parameters["axis"] == "dispatcher"
    }
    assert "exact" in {
        job.parameters["value"]
        for job in jobs
        if job.dataset_id == "isolet" and job.parameters["axis"] == "dispatcher"
    }


def test_atomic_write_json_flushes_and_syncs_before_replacing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    calls: list[int] = []
    real_fsync = protocol_module.os.fsync

    def record_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(protocol_module.os, "fsync", record_fsync)
    atomic_write_json(destination, {"ok": True})

    assert len(calls) >= 1


def test_atomic_write_json_cleans_temporary_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "record.json"
    monkeypatch.setattr(
        protocol_module.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(destination, {"ok": True})
    assert not tuple(tmp_path.glob(".record.json.*.tmp"))


def test_atomic_write_json_rejects_nonfinite_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "record.json", {"metric": float("inf")})


@pytest.mark.parametrize(
    ("existing", "candidate"),
    [
        (b'{\n  "value": true\n}\n', {"value": 1}),
        (b'{\n  "value": false\n}\n', {"value": 0}),
    ],
)
def test_publish_json_exclusive_identity_is_bool_int_type_sensitive(
    tmp_path: Path, existing: bytes, candidate: object
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_bytes(existing)

    with pytest.raises(RuntimeError, match="conflict"):
        publish_json_exclusive(destination, candidate, context="strict artifact")

    assert destination.read_bytes() == existing
    assert not tuple(tmp_path.glob(".artifact.json.*.tmp"))


@pytest.mark.parametrize(
    "existing",
    [
        b'{"value":0,"value":1}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'{"value":-Infinity}\n',
        b'{ "value": 1 }\n',
        b"not-json\n",
    ],
)
def test_publish_json_exclusive_rejects_malformed_or_noncanonical_existing_json(
    tmp_path: Path, existing: bytes
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_bytes(existing)

    with pytest.raises(RuntimeError, match="malformed"):
        publish_json_exclusive(
            destination, {"value": 1}, context="strict artifact"
        )

    assert destination.read_bytes() == existing
    assert not tuple(tmp_path.glob(".artifact.json.*.tmp"))


def test_publish_json_exclusive_reuses_only_canonical_identical_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    canonical = b'{\n  "value": 1\n}\n'
    destination.write_bytes(canonical)

    publish_json_exclusive(destination, {"value": 1}, context="strict artifact")

    assert destination.read_bytes() == canonical
    assert not tuple(tmp_path.glob(".artifact.json.*.tmp"))


def test_publish_json_exclusive_rejects_existing_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b'{\n  "value": 1\n}\n')
    destination = tmp_path / "artifact.json"
    destination.write_bytes(target.read_bytes())
    real_is_symlink = Path.is_symlink

    def report_destination_symlink(path: Path) -> bool:
        return path == destination or real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_destination_symlink)

    with pytest.raises(RuntimeError, match="regular file"):
        publish_json_exclusive(
            destination, {"value": 1}, context="strict artifact"
        )

    assert target.read_bytes() == b'{\n  "value": 1\n}\n'
    assert destination.read_bytes() == target.read_bytes()
    assert not tuple(tmp_path.glob(".artifact.json.*.tmp"))


def test_read_canonical_json_accepts_only_strict_canonical_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    atomic_write_json(destination, {"score": 1, "valid": True})

    assert protocol_module.read_canonical_json(
        destination, context="strict reader artifact"
    ) == {"score": 1, "valid": True}

    destination.write_text(
        '{"score":1,"score":1,"valid":true}\n',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="canonical|duplicate|malformed|JSON"):
        protocol_module.read_canonical_json(
            destination, context="strict reader artifact"
        )


def _status_fixture(tmp_path: Path, resolved_protocol):
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    return layout, jobs, StatusStore(layout, jobs)


def _write_canonical_result(layout, job) -> None:
    atomic_write_json(
        layout.root / job.output_relpath,
        {
            "job_id": job.job_id,
            "protocol_hash": job.protocol_hash,
            "stage": job.stage,
            "dataset_id": job.dataset_id,
            "seed": job.seed,
            "method": job.method,
            "metrics": {"score": 1.0},
            "artifacts": {},
            "started_at": "2026-07-21T00:00:00+00:00",
            "finished_at": "2026-07-21T00:00:01+00:00",
        },
    )


@pytest.mark.parametrize(
    ("state", "has_valid_output", "expected"),
    [
        ("pending", False, "execute"),
        ("running", False, "execute"),
        ("failed", False, "execute"),
        ("success", True, "skip"),
        ("success", False, "execute"),
    ],
)
def test_status_store_resume_decision_matrix(
    tmp_path: Path,
    resolved_protocol,
    state: str,
    has_valid_output: bool,
    expected: str,
) -> None:
    layout = create_new_run(tmp_path / "results", "symlink-complete", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(layout, jobs)
    job = jobs[0]
    if state != "pending":
        status.start(job.job_id)
        if state == "failed":
            status.fail(job.job_id, RuntimeError("expected"))
        elif state == "success":
            _write_canonical_result(layout, job)
            status.succeed(job.job_id)
            if not has_valid_output:
                (layout.root / job.output_relpath).unlink()

    decision = status.decide(job.job_id)

    assert decision.action == expected
    assert decision.reason in {"pending", "stale_running", "failed", "valid_success", "invalid_success"}
    if state == "success" and not has_valid_output:
        assert status.entry(job.job_id)["state"] == "failed"


def test_status_store_recovers_stale_attempt_and_rebuilds_aggregate(
    tmp_path: Path, resolved_protocol
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    job = jobs[0]
    assert status.start(job.job_id)["attempt"] == 1
    assert status.decide(job.job_id).reason == "stale_running"
    recovered = status.start(job.job_id)

    assert recovered["state"] == "running"
    assert recovered["attempt"] == 2
    assert recovered["recovery"]["event"] == "interrupted_recovery"
    root = json.loads(layout.status_path.read_text(encoding="utf-8"))
    assert root["stages"][job.stage]["running"] == 1


def test_status_store_success_validation_and_illegal_transition_are_atomic(
    tmp_path: Path, resolved_protocol
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    job = jobs[0]
    status.start(job.job_id)
    root_before = layout.status_path.read_bytes()
    stage_path = layout.root / job.stage / "status.json"
    stage_before = stage_path.read_bytes()
    with pytest.raises(ValueError, match="result"):
        status.succeed(job.job_id)
    assert layout.status_path.read_bytes() == root_before
    assert stage_path.read_bytes() == stage_before
    status.fail(job.job_id, RuntimeError("expected"))
    root_before = layout.status_path.read_bytes()
    stage_before = stage_path.read_bytes()
    with pytest.raises(RuntimeError, match="illegal"):
        status.succeed(job.job_id)
    assert layout.status_path.read_bytes() == root_before
    assert stage_path.read_bytes() == stage_before


def test_status_store_rejects_unapproved_skip_and_manifest_drift(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(layout, jobs)
    with pytest.raises(ValueError, match="approved"):
        status.skip(jobs[0].job_id, "not-approved")
    layout = create_new_run(tmp_path / "results", "approved-run", resolved_protocol)
    approved = StatusStore(layout, jobs, approved_skips={jobs[0].job_id: "waived"})
    approved.skip(jobs[0].job_id, "waived")
    approved.start(jobs[1].job_id)
    with pytest.raises(ValueError, match="manifest"):
        StatusStore(layout, jobs[1:])


def test_status_store_reconstruction_preserves_attempts_and_rejects_malformed_status(
    tmp_path: Path, resolved_protocol
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    status.start(jobs[0].job_id)
    resumed = StatusStore(layout, jobs)
    assert resumed.entry(jobs[0].job_id)["attempt"] == 1
    atomic_write_json(layout.status_path, {"schema_version": True})
    with pytest.raises(ValueError, match="schema"):
        StatusStore(layout, jobs)


def test_assert_stage_complete_aggregates_manifest_authoritative_defects(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(
        layout, jobs, approved_skips={jobs[1].job_id: "not applicable"}
    )
    first, second = jobs
    status.start(first.job_id)
    status.fail(first.job_id, RuntimeError("expected"))
    with pytest.raises(IncompleteRunError) as raised:
        assert_stage_complete(layout, (first, second, second), ("preprocess",))

    message = str(raised.value)
    assert first.job_id in message
    assert second.job_id in message
    assert "duplicate manifest job ID" in message


def test_assert_stage_complete_accepts_only_valid_success_or_exact_approved_skip(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(
        layout, jobs, approved_skips={jobs[1].job_id: "not applicable"}
    )
    first, second = jobs
    _write_canonical_result(layout, first)
    status.start(first.job_id)
    status.succeed(first.job_id)
    status.skip(second.job_id, "not applicable")

    report = assert_stage_complete(layout, jobs, ("preprocess",))

    assert report.complete_job_ids == tuple(job.job_id for job in jobs)


def test_assert_stage_complete_rejects_foreign_result_and_ignores_extra_files(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(
        layout, jobs, approved_skips={jobs[1].job_id: "not applicable"}
    )
    _write_canonical_result(layout, jobs[0])
    status.start(jobs[0].job_id)
    status.succeed(jobs[0].job_id)
    status.skip(jobs[1].job_id, "not applicable")
    atomic_write_json(layout.root / "raw_metrics" / "unrelated.json", {"noise": True})
    payload = json.loads((layout.root / jobs[0].output_relpath).read_text(encoding="utf-8"))
    payload["protocol_hash"] = "foreign"
    atomic_write_json(layout.root / jobs[0].output_relpath, payload)

    with pytest.raises(IncompleteRunError, match="foreign-protocol"):
        assert_stage_complete(layout, jobs, ("preprocess",))


def test_completeness_is_read_only_and_reports_missing_required_stage(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))

    with pytest.raises(IncompleteRunError, match="estimator: no manifest jobs"):
        assert_stage_complete(layout, jobs, ("preprocess", "estimator"))

    assert not layout.status_path.exists()
    assert not (layout.root / "preprocess" / "status.json").exists()


def test_status_store_recovers_missing_root_from_authoritative_stage_docs(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(
        layout, jobs, approved_skips={jobs[1].job_id: "not applicable"}
    )
    status.start(jobs[0].job_id)
    status.skip(jobs[1].job_id, "not applicable")
    layout.status_path.unlink()

    resumed = StatusStore(layout, jobs)

    assert resumed.entry(jobs[0].job_id)["attempt"] == 1
    assert resumed.decide(jobs[1].job_id).reason == "approved_skip"
    assert json.loads(layout.status_path.read_text(encoding="utf-8"))["approved_skips"] == {
        jobs[1].job_id: "not applicable"
    }


def test_status_store_reconciles_stale_root_but_rejects_partial_stage_set(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess", "validate"))
    status = StatusStore(layout, jobs)
    status.start(next(job.job_id for job in jobs if job.stage == "preprocess"))
    root = json.loads(layout.status_path.read_text(encoding="utf-8"))
    root["stages"]["preprocess"]["running"] = 0
    root["stages"]["preprocess"]["pending"] += 1
    atomic_write_json(layout.status_path, root)
    assert StatusStore(layout, jobs).entry(
        next(job.job_id for job in jobs if job.stage == "preprocess")
    )["state"] == "running"
    (layout.root / "validate" / "status.json").unlink()
    with pytest.raises(ValueError, match="partial"):
        StatusStore(layout, jobs)


def test_root_write_failure_poisoned_store_recovers_without_losing_stage_progress(
    tmp_path: Path, resolved_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    real_write = protocol_module.atomic_write_json

    def fail_root(path: Path, value: object) -> None:
        if Path(path) == layout.status_path:
            raise OSError("root write failed")
        real_write(path, value)

    monkeypatch.setattr(protocol_module, "atomic_write_json", fail_root)
    with pytest.raises(OSError, match="root write"):
        status.start(jobs[0].job_id)
    with pytest.raises(RuntimeError, match="reopen"):
        status.decide(jobs[0].job_id)
    monkeypatch.setattr(protocol_module, "atomic_write_json", real_write)
    resumed = StatusStore(layout, jobs)
    assert resumed.entry(jobs[0].job_id)["state"] == "running"
    assert resumed.entry(jobs[0].job_id)["attempt"] == 1


@pytest.mark.parametrize(
    ("state", "changes"),
    [
        ("pending", {"attempt": 1}),
        ("running", {"attempt": 0}),
        ("success", {"recovery": {"event": "interrupted_recovery", "at": "2026-07-21T00:00:00+00:00", "previous_attempt": 0}}),
        ("failed", {"exception_class": ""}),
        ("skipped", {"attempt": 1}),
    ],
)
def test_status_store_rejects_state_semantic_violations(
    tmp_path: Path, resolved_protocol, state: str, changes: dict[str, object]
) -> None:
    if state == "skipped":
        layout = create_new_run(tmp_path / "results", "skipped-run", resolved_protocol)
        jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
        status = StatusStore(
            layout, jobs, approved_skips={jobs[0].job_id: "waived"}
        )
    else:
        layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    job = jobs[0]
    if state in {"running", "failed", "success"}:
        status.start(job.job_id)
    if state == "failed":
        status.fail(job.job_id, RuntimeError("expected"))
    elif state == "success":
        _write_canonical_result(layout, job)
        status.succeed(job.job_id)
    elif state == "skipped":
        status.skip(job.job_id, "waived")
    entry = status.entry(job.job_id)
    entry.update(changes)
    payload = json.loads((layout.root / job.stage / "status.json").read_text(encoding="utf-8"))
    payload["jobs"][job.job_id] = entry
    atomic_write_json(layout.root / job.stage / "status.json", payload)
    with pytest.raises(ValueError):
        StatusStore(layout, jobs)


def test_status_store_rejects_bool_root_counts_and_non_utc_status_timestamp(
    tmp_path: Path, resolved_protocol
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    root = json.loads(layout.status_path.read_text(encoding="utf-8"))
    root["stages"][jobs[0].stage]["pending"] = True
    atomic_write_json(layout.status_path, root)
    with pytest.raises(ValueError, match="aggregate"):
        StatusStore(layout, jobs)
    root["stages"][jobs[0].stage]["pending"] = len(jobs)
    atomic_write_json(layout.status_path, root)
    stage = json.loads((layout.root / jobs[0].stage / "status.json").read_text(encoding="utf-8"))
    stage["jobs"][jobs[0].job_id]["timestamp"] = "2026-07-21T00:00:00+03:00"
    atomic_write_json(layout.root / jobs[0].stage / "status.json", stage)
    with pytest.raises(ValueError, match="UTC"):
        StatusStore(layout, jobs)


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_completeness_ignores_unrequested_stage_status_damage(
    tmp_path: Path, resolved_protocol, damage: str
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("validate", "preprocess"))
    status = StatusStore(layout, jobs)
    requested = tuple(job for job in jobs if job.stage == "preprocess")
    for job in requested:
        _write_canonical_result(layout, job)
        status.start(job.job_id)
        status.succeed(job.job_id)
    foreign_stage = layout.root / "validate" / "status.json"
    if damage == "missing":
        foreign_stage.unlink()
    else:
        atomic_write_json(foreign_stage, {"broken": True})

    assert assert_stage_complete(layout, jobs, ("preprocess",)).complete_job_ids == tuple(
        job.job_id for job in requested
    )


def test_completeness_scopes_stage_files_but_keeps_full_manifest_identity(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("validate", "preprocess"))
    status = StatusStore(layout, jobs)
    for job in jobs:
        if job.stage == "preprocess":
            _write_canonical_result(layout, job)
            status.start(job.job_id)
            status.succeed(job.job_id)
    root = json.loads(layout.status_path.read_text(encoding="utf-8"))
    stage_path = layout.root / "preprocess" / "status.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    root["manifest_hash"] = stage["manifest_hash"] = "forged-manifest-hash"
    atomic_write_json(layout.status_path, root)
    atomic_write_json(stage_path, stage)
    (layout.root / "validate" / "status.json").unlink()

    with pytest.raises(IncompleteRunError, match="manifest"):
        assert_stage_complete(layout, jobs, ("preprocess",))


def test_completeness_reports_running_schema_invalid_and_unapproved_skip(
    tmp_path: Path, resolved_protocol
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    status.start(jobs[0].job_id)
    stage_path = layout.root / jobs[0].stage / "status.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    skipped = stage["jobs"][jobs[1].job_id]
    skipped.update({"state": "skipped", "skip_reason": "unapproved"})
    stage["jobs"][jobs[1].job_id] = skipped
    atomic_write_json(stage_path, stage)

    with pytest.raises(IncompleteRunError) as raised:
        assert_stage_complete(layout, jobs, ("preprocess",))

    message = str(raised.value)
    assert "status is running" in message
    assert "skipped" in message


@pytest.mark.parametrize(
    ("state", "action"),
    [
        ("pending", "succeed"), ("pending", "fail"),
        ("running", "skip"),
        ("failed", "succeed"), ("failed", "fail"), ("failed", "skip"),
        ("success", "start"), ("success", "succeed"), ("success", "fail"), ("success", "skip"),
        ("skipped", "start"), ("skipped", "succeed"), ("skipped", "fail"), ("skipped", "skip"),
    ],
)
def test_full_illegal_transition_matrix_preserves_status_bytes(
    tmp_path: Path, resolved_protocol, state: str, action: str
) -> None:
    layout = create_new_run(tmp_path / "results", "status-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(layout, jobs, approved_skips={jobs[0].job_id: "waived"})
    job = jobs[0]
    if state in {"running", "success", "failed"}:
        status.start(job.job_id)
    if state == "success":
        _write_canonical_result(layout, job)
        status.succeed(job.job_id)
    elif state == "failed":
        status.fail(job.job_id, RuntimeError("expected"))
    elif state == "skipped":
        status.skip(job.job_id, "waived")
    root_before = layout.status_path.read_bytes()
    stage_before = (layout.root / job.stage / "status.json").read_bytes()
    illegal = {
        "start": lambda: status.start(job.job_id),
        "succeed": lambda: status.succeed(job.job_id),
        "fail": lambda: status.fail(job.job_id, RuntimeError("expected")),
        "skip": lambda: status.skip(job.job_id, "waived"),
    }[action]

    with pytest.raises((RuntimeError, ValueError)):
        illegal()
    assert layout.status_path.read_bytes() == root_before
    assert (layout.root / job.stage / "status.json").read_bytes() == stage_before


def test_status_store_rejects_result_symlink_escape(
    tmp_path: Path, resolved_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    job = jobs[0]
    status.start(job.job_id)
    _write_canonical_result(layout, job)
    real_within = protocol_module._require_within

    def escaped(parent: Path, candidate: Path, context: str) -> None:
        if context == "result path":
            raise RuntimeError("result path escapes the run directory")
        real_within(parent, candidate, context)

    monkeypatch.setattr(protocol_module, "_require_within", escaped)

    with pytest.raises(ValueError, match="result path"):
        status.succeed(job.job_id)


def test_symlink_escape_resume_fails_success_and_completeness_aggregates(
    tmp_path: Path, resolved_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = create_new_run(tmp_path / "results", "symlink-run", resolved_protocol)
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(layout, jobs)
    job = jobs[0]
    _write_canonical_result(layout, job)
    status.start(job.job_id)
    status.succeed(job.job_id)
    real_within = protocol_module._require_within

    def escaped(parent: Path, candidate: Path, context: str) -> None:
        if context == "result path":
            raise RuntimeError("result path escapes the run directory")
        real_within(parent, candidate, context)

    monkeypatch.setattr(protocol_module, "_require_within", escaped)
    assert status.decide(job.job_id).action == "execute"
    assert status.entry(job.job_id)["state"] == "failed"

    monkeypatch.setattr(protocol_module, "_require_within", real_within)
    layout = create_new_run(
        tmp_path / "results", "symlink-complete", resolved_protocol
    )
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    status = StatusStore(layout, jobs)
    job = jobs[0]
    _write_canonical_result(layout, job)
    status.start(job.job_id)
    status.succeed(job.job_id)
    monkeypatch.setattr(protocol_module, "_require_within", escaped)
    with pytest.raises(IncompleteRunError, match="result path"):
        assert_stage_complete(layout, jobs, ("preprocess",))


def test_completeness_rejects_schema_invalid_success_result(
    tmp_path: Path, resolved_protocol
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    job = jobs[0]
    _write_canonical_result(layout, job)
    status.start(job.job_id)
    status.succeed(job.job_id)
    path = layout.root / job.output_relpath
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["method"]
    atomic_write_json(path, payload)

    with pytest.raises(IncompleteRunError, match="invalid or foreign-protocol result"):
        assert_stage_complete(layout, jobs, ("preprocess",))


def test_failed_retry_after_reconstruction_increments_attempt_and_keeps_empty_message(
    tmp_path: Path, resolved_protocol
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    status.start(jobs[0].job_id)
    status.fail(jobs[0].job_id, RuntimeError())
    resumed = StatusStore(layout, jobs)

    assert resumed.entry(jobs[0].job_id)["message"] == ""
    assert resumed.start(jobs[0].job_id)["attempt"] == 2


def test_entry_returns_a_deep_copy_and_completeness_aggregates_simultaneous_defects(
    tmp_path: Path, resolved_protocol
) -> None:
    layout, jobs, status = _status_fixture(tmp_path, resolved_protocol)
    status.start(jobs[0].job_id)
    recovered = status.start(jobs[0].job_id)
    recovered["recovery"]["event"] = "tampered"
    assert status.entry(jobs[0].job_id)["recovery"]["event"] == "interrupted_recovery"
    status.fail(jobs[0].job_id, RuntimeError("expected"))

    with pytest.raises(IncompleteRunError) as raised:
        assert_stage_complete(layout, jobs, ("preprocess",))

    assert jobs[0].job_id in str(raised.value)
    assert jobs[1].job_id in str(raised.value)


def test_manifest_store_persists_and_reloads_exact_ordered_authority(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "manifest-run", resolved_protocol)
    validate = build_job_manifest(resolved_protocol, stages=("validate",))
    preprocess = build_job_manifest(resolved_protocol, stages=("preprocess",))
    store = ManifestStore(layout.manifest_path)

    assert store.initialize(validate) == validate
    assert store.append(preprocess) == validate + preprocess
    assert ManifestStore(layout.manifest_path).load() == validate + preprocess
    assert layout.manifest_path.read_text(encoding="utf-8").splitlines()[0] == (
        "sequence,job_id,protocol_hash,stage,dataset_id,seed,method,parameters_json,"
        "depends_on_json,output_relpath"
    )


def test_manifest_store_rejects_replacement_reordering_and_repeated_stage_chunk(
    tmp_path: Path, resolved_protocol
) -> None:
    path = tmp_path / "run_manifest.csv"
    validate = build_job_manifest(resolved_protocol, stages=("validate",))
    preprocess = build_job_manifest(resolved_protocol, stages=("preprocess",))
    store = ManifestStore(path)
    store.initialize(validate)

    with pytest.raises(FileExistsError):
        store.initialize(validate)
    with pytest.raises(ValueError, match="order|stage|append"):
        store.append(preprocess + build_job_manifest(resolved_protocol, stages=("summarize",)))
    store.append(preprocess)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="stage|append"):
        store.append(preprocess)
    assert path.read_bytes() == before


def test_manifest_store_rejects_foreign_hash_duplicate_ids_and_malformed_csv(
    tmp_path: Path, resolved_protocol
) -> None:
    path = tmp_path / "run_manifest.csv"
    validate = build_job_manifest(resolved_protocol, stages=("validate",))
    preprocess = build_job_manifest(resolved_protocol, stages=("preprocess",))
    store = ManifestStore(path)
    store.initialize(validate)
    with pytest.raises(ValueError, match="protocol"):
        store.append((replace(preprocess[0], protocol_hash="f" * 64),))
    with pytest.raises(ValueError, match="duplicate"):
        ManifestStore(tmp_path / "duplicate.csv").initialize((validate[0], validate[0]))

    path.write_text(
        "job_id,protocol_hash,stage,dataset_id,seed,method,parameters_json,"
        "depends_on_json,output_relpath\n"
        f"{validate[0].job_id},{validate[0].protocol_hash},validate,__run__,0,method,"
        '"[]","[]",../escape.json\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest"):
        store.load()


def test_manifest_store_rejects_edited_identity_and_reordered_persisted_rows(
    tmp_path: Path, resolved_protocol
) -> None:
    jobs = build_job_manifest(resolved_protocol, stages=("preprocess",))
    identity_path = tmp_path / "edited.csv"
    ManifestStore(identity_path).initialize(jobs)
    identity_path.write_text(
        identity_path.read_text(encoding="utf-8").replace(
            ",prepare_split,", ",changed_method,", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical|identity|job ID"):
        ManifestStore(identity_path).load()

    order_path = tmp_path / "reordered.csv"
    ManifestStore(order_path).initialize(jobs)
    lines = order_path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    order_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sequence|order"):
        ManifestStore(order_path).load()


def test_status_store_extends_exact_prefix_without_rebinding_old_stage_document(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "extend-run", resolved_protocol)
    validate = build_job_manifest(resolved_protocol, stages=("validate",))
    preprocess = build_job_manifest(resolved_protocol, stages=("preprocess",))
    manifest = validate + preprocess
    manifests = ManifestStore(layout.manifest_path)
    manifests.initialize(validate)
    status = StatusStore(layout, validate)
    status.start(validate[0].job_id)
    old_stage_path = layout.root / "validate" / "status.json"
    old_stage_bytes = old_stage_path.read_bytes()

    manifests.append(preprocess)
    extended = StatusStore(layout, manifest)

    assert extended.entry(validate[0].job_id)["attempt"] == 1
    assert extended.entry(preprocess[0].job_id)["state"] == "pending"
    assert old_stage_path.read_bytes() == old_stage_bytes
    root = json.loads(layout.status_path.read_text(encoding="utf-8"))
    validate_doc = json.loads(old_stage_path.read_text(encoding="utf-8"))
    assert root["manifest_hash"] != validate_doc["manifest_hash"]


def test_status_store_recovers_manifest_before_status_extension_and_partial_suffix(
    tmp_path: Path, resolved_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = create_new_run(tmp_path / "results", "recover-extension", resolved_protocol)
    validate = build_job_manifest(resolved_protocol, stages=("validate",))
    preprocess = build_job_manifest(resolved_protocol, stages=("preprocess",))
    manifests = ManifestStore(layout.manifest_path)
    manifests.initialize(validate)
    StatusStore(layout, validate)
    manifests.append(preprocess)
    real_write = protocol_module.atomic_write_json

    def fail_root(path: Path, value: object) -> None:
        if Path(path) == layout.status_path:
            raise OSError("interrupted root extension")
        real_write(path, value)

    monkeypatch.setattr(protocol_module, "atomic_write_json", fail_root)
    with pytest.raises(OSError, match="interrupted"):
        StatusStore(layout, validate + preprocess)
    monkeypatch.setattr(protocol_module, "atomic_write_json", real_write)

    recovered = StatusStore(layout, validate + preprocess)
    assert recovered.entry(validate[0].job_id)["attempt"] == 0
    assert recovered.entry(preprocess[0].job_id)["state"] == "pending"


def test_status_store_rejects_nonprefix_extension_and_changed_frozen_stage(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "bad-extension", resolved_protocol)
    validate = build_job_manifest(resolved_protocol, stages=("validate",))
    preprocess = build_job_manifest(resolved_protocol, stages=("preprocess",))
    manifests = ManifestStore(layout.manifest_path)
    manifests.initialize(validate)
    StatusStore(layout, validate)

    with pytest.raises(ValueError, match="prefix|manifest"):
        StatusStore(layout, preprocess)
    manifests.append(preprocess)
    changed = (replace(validate[0], method="changed"),) + preprocess
    with pytest.raises(ValueError, match="prefix|manifest"):
        StatusStore(layout, changed)


def test_status_store_rejects_unpublished_extension_suffix(
    tmp_path: Path, resolved_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "unpublished-extension", resolved_protocol)
    validate = build_job_manifest(resolved_protocol, stages=("validate",))
    preprocess = build_job_manifest(resolved_protocol, stages=("preprocess",))
    ManifestStore(layout.manifest_path).initialize(validate)
    StatusStore(layout, validate)

    with pytest.raises(ValueError, match="durable|published|manifest"):
        StatusStore(layout, validate + preprocess)

    assert ManifestStore(layout.manifest_path).load() == validate
    assert not (layout.root / "preprocess" / "status.json").exists()


def test_read_data_manifest_validates_rich_rows_and_returns_strict_projection(
    tmp_path: Path, resolved_protocol
) -> None:
    path = tmp_path / "data_manifest.csv"
    for dataset in resolved_protocol.datasets:
        upsert_data_manifest(
            path,
            resolved_protocol,
            _data_manifest_row(resolved_protocol, dataset.dataset_id),
        )

    rows = read_data_manifest(path, resolved_protocol)

    assert build_job_manifest(
        resolved_protocol, stages=("estimator",), data_manifest=rows
    )
    path.write_text("dataset_id,seed,selected_feature_count\nspambase,42,7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data_manifest"):
        read_data_manifest(path, resolved_protocol)
