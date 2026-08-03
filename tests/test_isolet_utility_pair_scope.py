from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from minppi_mofs.experiment_protocol import (
    DatasetSpec,
    ExecutionSpec,
    ManifestStore,
    ResultRecord,
    ResolvedProtocol,
    StatusStore,
    atomic_write_json,
    build_job_manifest,
    create_new_run,
    load_resolved_protocol,
    resolved_protocol_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
def _protocol(tmp_path: Path, *, execution_mode: str = "legacy") -> ResolvedProtocol:
    dataset = DatasetSpec(
        dataset_id="isolet",
        display_name="ISOLET",
        raw_path=tmp_path / "isolet.csv",
        raw_sha256="0" * 64,
        raw_byte_count=1,
        target="letter",
        task="multiclass",
        utility="ovr_auc",
        max_features=100,
        variance_threshold=0.0,
        train_frac=0.64,
        val_frac=0.16,
        test_frac=0.20,
    )
    return ResolvedProtocol(
        protocol_version="1.0",
        run_kind="formal",
        datasets=(dataset,),
        seeds=(42,),
        backend="cuda",
        execution=ExecutionSpec(
            mode=execution_mode,
            cache_schema="coalition_utility_v1",
            checkpoint_every_fits=50,
            progress_every_seconds=30,
        ),
        model_params={
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_jobs": 1,
        },
        alpha_default=0.95,
        alpha_grid=(0.85, 0.90, 0.95, 0.99),
        estimator_methods=(
            "exact",
            "random_mc",
            "mi_correlation_guided",
            "hybrid",
        ),
        estimator_budgets=(10, 25, 50, 100),
        exact_max_features=16,
        nsga2={
            "variants": ["full", "minppi_prefilter"],
            "population": 40,
            "evaluations": 800,
            "ref_point": [1.1, 1.1],
        },
        removal={
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
        ablations={
            "utility": ["accuracy", "ovr_auc"],
            "guidance": ["random", "mi", "correlation", "mi_correlation"],
            "dispatcher": [
                "exact",
                "random_mc",
                "mi_correlation_guided",
                "hybrid",
            ],
        },
        smoke={
            "datasets": ["isolet"],
            "seeds": [42],
            "estimator_budgets": [2],
            "random_repeats": 2,
            "nsga2_population": 4,
            "nsga2_evaluations": 8,
        },
    )


def _write_formal_scoped_config(
    tmp_path: Path,
    *,
    dataset_id: str,
) -> Path:
    """Derive a one-dataset formal config from the repository authority."""
    config_dir = tmp_path / f"formal-{dataset_id}-s42"
    config_dir.mkdir()
    shutil.copy2(PROJECT_ROOT / "configs" / "icomp2026_pipeline.yaml", config_dir)
    shutil.copy2(PROJECT_ROOT / "configs" / f"{dataset_id}.yaml", config_dir)

    pipeline_path = config_dir / "icomp2026_pipeline.yaml"
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    pipeline["datasets"] = [dataset_id]
    pipeline["seeds"] = [42]
    pipeline["backend"] = "cuda"
    pipeline["estimator"]["budgets_by_dataset"] = {
        dataset_id: pipeline["estimator"]["budgets_by_dataset"][dataset_id]
    }
    pipeline["stage_scopes"] = {
        stage: {"datasets": [dataset_id], "seeds": [42]}
        for stage in pipeline["stage_scopes"]
    }
    pipeline["smoke"]["datasets"] = [dataset_id]
    pipeline["smoke"]["seeds"] = [42]
    pipeline_path.write_text(
        yaml.safe_dump(pipeline, sort_keys=False),
        encoding="utf-8",
    )
    return config_dir


def _selected_counts() -> tuple[dict[str, object], ...]:
    return (
        {
            "dataset_id": "isolet",
            "seed": 42,
            "selected_feature_count": 100,
        },
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result(job, *, artifacts: dict[str, str] | None = None) -> ResultRecord:
    stamp = _timestamp()
    return ResultRecord(
        job_id=job.job_id,
        protocol_hash=job.protocol_hash,
        stage=job.stage,
        dataset_id=job.dataset_id,
        seed=job.seed,
        method=job.method,
        metrics={"test_fixture": True},
        artifacts={} if artifacts is None else artifacts,
        started_at=stamp,
        finished_at=stamp,
    )


def _complete(status: StatusStore, layout, job) -> None:
    status.start(job.job_id)
    atomic_write_json(layout.root / job.output_relpath, asdict(_result(job)))
    status.succeed(job.job_id)


def test_scoped_manifest_uses_only_canonical_isolet_utility_pair(
    tmp_path: Path,
) -> None:
    spec = importlib.util.find_spec("experiments.run_isolet_utility_pair")
    assert spec is not None, "dedicated ISOLET utility-pair module is missing"
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    protocol = _protocol(tmp_path)

    manifest = module.build_isolet_utility_pair_manifest(protocol)

    canonical_ablation = build_job_manifest(
        protocol,
        stages=("ablation",),
        data_manifest=_selected_counts(),
    )
    expected_pair = tuple(
        job
        for job in canonical_ablation
        if job.dataset_id == "isolet"
        and job.seed == 42
        and job.method == "utility"
    )
    assert tuple(job.stage for job in manifest) == (
        "validate",
        "preprocess",
        "ablation",
        "ablation",
    )
    assert manifest[-2:] == expected_pair
    assert {
        job.parameters["value"] for job in manifest[-2:]
    } == {"accuracy", "ovr_auc"}


@pytest.mark.parametrize(
    "change, match",
    [
        ({"run_kind": "smoke"}, "formal"),
        ({"backend": "cpu"}, "backend=cuda"),
        ({"seeds": (123,)}, "seed 42"),
        (
            {
                "execution": ExecutionSpec(
                    mode="future_mode",
                    cache_schema="coalition_utility_v1",
                    checkpoint_every_fits=50,
                    progress_every_seconds=30,
                )
            },
            "execution mode",
        ),
        (
            {
                "ablations": {
                    "utility": ["accuracy", "accuracy"],
                    "guidance": [],
                    "dispatcher": [],
                }
            },
            "accuracy and ovr_auc",
        ),
    ],
)
def test_scoped_manifest_rejects_noncanonical_protocol_scope(
    tmp_path: Path,
    change: dict[str, object],
    match: str,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    protocol = replace(_protocol(tmp_path), **change)

    with pytest.raises(ValueError, match=match):
        module.build_isolet_utility_pair_manifest(protocol)


def test_scoped_manifest_rejects_non_isolet_dataset(tmp_path: Path) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    protocol = _protocol(tmp_path)
    foreign = replace(protocol.datasets[0], dataset_id="spambase", task="binary")

    with pytest.raises(ValueError, match="canonical ISOLET"):
        module.build_isolet_utility_pair_manifest(
            replace(protocol, datasets=(foreign,))
        )


def test_scoped_manifest_accepts_legacy_and_grouped_configs(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")

    legacy = module.build_isolet_utility_pair_manifest(
        _protocol(tmp_path, execution_mode="legacy")
    )
    grouped = module.build_isolet_utility_pair_manifest(
        _protocol(tmp_path, execution_mode="grouped_nested_v1")
    )

    for manifest in (legacy, grouped):
        assert [
            (
                job.stage,
                job.dataset_id,
                job.seed,
                job.method,
                job.parameters,
            )
            for job in manifest[-2:]
        ] == [
            (
                "ablation",
                "isolet",
                42,
                "utility",
                {
                    "task": "multiclass",
                    "utility": "ovr_auc",
                    "axis": "utility",
                    "value": "accuracy",
                },
            ),
            (
                "ablation",
                "isolet",
                42,
                "utility",
                {
                    "task": "multiclass",
                    "utility": "ovr_auc",
                    "axis": "utility",
                    "value": "ovr_auc",
                },
            ),
        ]


def test_produced_data_manifest_must_reconfirm_exact_scoped_authority(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    protocol = _protocol(tmp_path)
    manifest = module.build_isolet_utility_pair_manifest(protocol)

    pair = module.validate_isolet_utility_pair_manifest(
        protocol,
        manifest,
        _selected_counts(),
    )

    assert pair == manifest[-2:]


@pytest.mark.parametrize("mutation", ["duplicate", "foreign_job_id"])
def test_produced_data_manifest_rejects_noncanonical_scoped_jobs(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    protocol = _protocol(tmp_path)
    manifest = list(module.build_isolet_utility_pair_manifest(protocol))
    if mutation == "duplicate":
        manifest[-1] = manifest[-2]
    else:
        manifest[-1] = replace(manifest[-1], job_id="0" * 64)

    with pytest.raises(ValueError, match="scoped manifest"):
        module.validate_isolet_utility_pair_manifest(
            protocol,
            tuple(manifest),
            _selected_counts(),
        )


def test_pair_dispatch_requires_preprocess_but_not_estimator(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    protocol = _protocol(tmp_path)
    manifest = module.build_isolet_utility_pair_manifest(protocol)
    layout = create_new_run(tmp_path / "results", "pair-dependency", protocol)
    ManifestStore(layout.manifest_path).initialize(manifest)
    status = StatusStore(layout, manifest)
    _complete(status, layout, manifest[0])

    blocked = module.execute_isolet_utility_pair_jobs(
        protocol,
        layout,
        manifest,
        _selected_counts(),
        executor=lambda context: _result(context.job),
        result_validator=lambda context, record: None,
    )

    assert blocked.exit_code == 1
    assert blocked.dependency_issues
    _complete(StatusStore(layout, manifest), layout, manifest[1])
    artifact_relpath = "ablation/isolet/s42/utility/paired/b200/scores.json"

    def executor(context):
        artifact = layout.root / artifact_relpath
        atomic_write_json(artifact, {"valid": True})
        return _result(context.job, artifacts={"paired_scores": artifact_relpath})

    def validator(context, record):
        assert record.artifacts == {"paired_scores": artifact_relpath}
        assert (layout.root / artifact_relpath).read_text(encoding="utf-8")

    completed = module.execute_isolet_utility_pair_jobs(
        protocol,
        layout,
        manifest,
        _selected_counts(),
        executor=executor,
        result_validator=validator,
    )

    assert completed.succeeded_job_ids == tuple(
        job.job_id for job in manifest[-2:]
    )
    assert completed.failed_job_ids == ()
    reopened = StatusStore(layout, manifest)
    assert all(
        reopened.entry(job.job_id)["state"] == "success"
        for job in manifest[-2:]
    )
    assert not any(job.stage == "estimator" for job in manifest)
    assert not (layout.root / "estimator" / "status.json").exists()


def test_pair_dispatch_resume_fails_closed_on_tampered_shared_artifact(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    protocol = _protocol(tmp_path)
    manifest = module.build_isolet_utility_pair_manifest(protocol)
    layout = create_new_run(tmp_path / "results", "pair-tamper", protocol)
    ManifestStore(layout.manifest_path).initialize(manifest)
    status = StatusStore(layout, manifest)
    for job in manifest[:2]:
        _complete(status, layout, job)
    artifact_relpath = "ablation/isolet/s42/utility/paired/b200/scores.json"
    artifact = layout.root / artifact_relpath

    def executor(context):
        if artifact.exists():
            if not json.loads(artifact.read_text(encoding="utf-8"))["valid"]:
                raise ValueError("tampered paired artifact")
        else:
            atomic_write_json(artifact, {"valid": True})
        return _result(context.job, artifacts={"paired_scores": artifact_relpath})

    def validator(context, record):
        if not json.loads(artifact.read_text(encoding="utf-8"))["valid"]:
            raise ValueError("tampered paired artifact")

    first = module.execute_isolet_utility_pair_jobs(
        protocol,
        layout,
        manifest,
        _selected_counts(),
        executor=executor,
        result_validator=validator,
    )
    assert first.exit_code == 0
    atomic_write_json(artifact, {"valid": False})

    resumed = module.execute_isolet_utility_pair_jobs(
        protocol,
        layout,
        manifest,
        _selected_counts(),
        executor=executor,
        result_validator=validator,
    )

    assert resumed.exit_code == 1
    assert resumed.failed_job_ids


def test_dedicated_runner_persists_four_jobs_without_estimator(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    stage_module = importlib.import_module("experiments.icomp2026_stages")
    protocol = _protocol(tmp_path)
    layout = create_new_run(tmp_path / "results", "pair-runner", protocol)
    artifact_relpath = "ablation/isolet/s42/utility/paired/b200/scores.json"
    calls: list[tuple[str, str]] = []

    def executor(context):
        calls.append((context.job.stage, context.job.method))
        artifacts = {}
        if context.job.stage == "ablation":
            atomic_write_json(
                layout.root / artifact_relpath,
                {"valid": True},
            )
            artifacts = {"paired_scores": artifact_relpath}
        return _result(context.job, artifacts=artifacts)

    runner = module.IsoletUtilityPairRunner(
        protocol,
        layout,
        executors={
            "validate": executor,
            "preprocess": executor,
            "ablation": executor,
        },
        data_manifest_reader=lambda *args: _selected_counts(),
        pair_result_validator=lambda context, record: None,
    )

    assert runner.run() == 0
    manifest = ManifestStore(layout.manifest_path).load()
    assert manifest == module.build_isolet_utility_pair_manifest(protocol)
    assert calls == [
        ("validate", "environment_validation"),
        ("preprocess", "prepare_split"),
        ("ablation", "utility"),
        ("ablation", "utility"),
    ]
    status = StatusStore(layout, manifest)
    assert all(status.entry(job.job_id)["state"] == "success" for job in manifest)
    assert stage_module.STAGE_DEPENDENCIES["ablation"] == (
        "preprocess",
        "estimator",
    )
    assert not (layout.root / "estimator" / "status.json").exists()


def test_cli_forces_formal_cuda_seed42_and_freezes_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    protocol = _protocol(tmp_path)
    layout = create_new_run(tmp_path / "created", "cli-pair", protocol)
    observed: dict[str, object] = {}

    def load(config_dir, **kwargs):
        observed["load"] = (config_dir, kwargs)
        return protocol

    def create(output_root, run_id, supplied):
        observed["create"] = (output_root, run_id, supplied)
        return layout

    def freeze(supplied_layout, supplied_protocol, *, project_root):
        observed["freeze"] = (
            supplied_layout,
            supplied_protocol,
            project_root,
        )

    class Runner:
        def __init__(self, supplied_protocol, supplied_layout):
            observed["runner"] = (supplied_protocol, supplied_layout)

        def run(self):
            return 0

    monkeypatch.setattr(module, "load_resolved_protocol", load)
    monkeypatch.setattr(module, "create_new_run", create)
    monkeypatch.setattr(module, "freeze_run_metadata", freeze)
    monkeypatch.setattr(module, "IsoletUtilityPairRunner", Runner)

    exit_code = module.main(
        [
            "--config-dir",
            str(tmp_path / "config"),
            "--output-root",
            str(tmp_path / "results"),
            "--run-id",
            "cli-pair",
        ]
    )

    assert exit_code == 0
    _, load_kwargs = observed["load"]
    assert load_kwargs == {
        "seeds": (42,),
        "backend": "cuda",
        "smoke": False,
    }
    assert observed["create"][1] == "cli-pair"
    assert observed["freeze"][0] == layout


def test_cli_help_exits_without_starting_isolated_worker() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/run_isolet_utility_pair.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert "--config-dir" in completed.stdout
    assert "isolation controller error" not in completed.stderr


def test_real_repository_config_projects_to_canonical_isolet_scope() -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    loaded = load_resolved_protocol(
        PROJECT_ROOT / "configs",
        seeds=(42,),
        backend="cuda",
        smoke=False,
    )

    projected = module.project_isolet_utility_pair_protocol(loaded)
    manifest = module.build_isolet_utility_pair_manifest(projected)

    assert len(loaded.datasets) == 6
    assert tuple(dataset.dataset_id for dataset in projected.datasets) == (
        "isolet",
    )
    assert projected.seeds == (42,)
    assert projected.backend == "cuda"
    assert projected.run_kind == "formal"
    assert replace(
        projected,
        datasets=loaded.datasets,
        seeds=loaded.seeds,
        estimator_budgets_by_dataset=(
            loaded.estimator_budgets_by_dataset
        ),
    ) == loaded
    assert len(manifest) == 4
    assert {job.protocol_hash for job in manifest} == {
        resolved_protocol_hash(projected)
    }


def test_real_task9_scoped_config_projection_is_identity(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    config_dir = _write_formal_scoped_config(tmp_path, dataset_id="isolet")
    loaded = load_resolved_protocol(
        config_dir,
        seeds=(42,),
        backend="cuda",
        smoke=False,
    )

    projected = module.project_isolet_utility_pair_protocol(loaded)

    assert projected == loaded
    assert module.build_isolet_utility_pair_manifest(projected) == (
        module.build_isolet_utility_pair_manifest(loaded)
    )


def test_real_config_without_isolet_fails_before_run_directory_creation(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("experiments.run_isolet_utility_pair")
    output_root = tmp_path / "must-not-exist"
    config_dir = _write_formal_scoped_config(tmp_path, dataset_id="abalone")

    exit_code = module.main(
        [
            "--config-dir",
            str(config_dir),
            "--output-root",
            str(output_root),
            "--run-id",
            "invalid-no-isolet",
        ]
    )

    assert exit_code == 2
    assert not output_root.exists()
