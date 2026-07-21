"""Tests for the frozen ICOMP 2026 experiment protocol."""

from __future__ import annotations

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
    ResultRecord,
    atomic_write_json,
    build_job_manifest,
    canonical_hash,
    create_new_run,
    freeze_run_metadata,
    load_resolved_protocol,
    open_resumed_run,
    read_result_record,
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
