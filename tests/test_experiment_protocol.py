"""Tests for the frozen ICOMP 2026 experiment protocol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import pytest
import yaml

import minppi_mofs.experiment_protocol as protocol_module
from minppi_mofs.experiment_protocol import (
    canonical_hash,
    create_new_run,
    freeze_run_metadata,
    load_resolved_protocol,
    open_resumed_run,
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
        "controls": ["full", "minppi", "random_matched"],
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
