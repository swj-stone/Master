from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import experiments.monitor_icomp2026_campaign as campaign_monitor
from experiments.summarize_multifidelity_campaign import (
    RuntimeTier,
    ShardRecord,
    analyze_removal_ranks,
    build_campaign_index,
    build_campaign_tables,
    forecast_runtime,
    merge_shard_summary_rows,
    read_shard_record,
    validate_campaign_shards,
)
from minppi_mofs.experiment_protocol import atomic_write_json
from experiments.run_icomp2026_campaign import (
    build_pipeline_command,
    campaign_execution_stages,
    default_seed_shards,
)
from experiments.monitor_icomp2026_campaign import summarize_status_document
from experiments.forecast_icomp2026_48h import (
    estimate_dataset_seconds,
    finalize_48h_forecast,
)


def _shard(name: str, seeds: tuple[int, ...]) -> ShardRecord:
    return ShardRecord(
        root=Path(name),
        protocol_hash=hashlib.sha256(name.encode("utf-8")).hexdigest(),
        seeds=seeds,
        model_policy_digest="a" * 64,
        evidence_digest="b" * 64,
        source_files_digest="c" * 64,
        raw_authority_digest="d" * 64,
        scientific_environment_digest="e" * 64,
        complete=True,
    )


def test_campaign_accepts_exact_nonoverlapping_five_seed_union() -> None:
    shards = (
        _shard("screen", (42,)),
        _shard("confirm", (123, 456, 789, 1024)),
    )

    validated = validate_campaign_shards(shards)

    assert validated == shards


def test_campaign_rejects_overlap_and_incomplete_seed_union() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_campaign_shards(
            (_shard("screen", (42, 123)), _shard("confirm", (123, 456, 789)))
        )
    with pytest.raises(ValueError, match="seed union"):
        validate_campaign_shards((_shard("screen", (42,)),))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_policy_digest", "d" * 64, "model policy"),
        ("evidence_digest", "e" * 64, "evidence"),
        ("source_files_digest", "f" * 64, "source"),
        ("raw_authority_digest", "1" * 64, "raw authority"),
        (
            "scientific_environment_digest",
            "2" * 64,
            "scientific environment",
        ),
        ("complete", False, "complete"),
    ],
)
def test_campaign_rejects_incompatible_or_incomplete_shards(
    field: str, value: object, message: str
) -> None:
    screen = _shard("screen", (42,))
    confirm = replace(_shard("confirm", (123, 456, 789, 1024)), **{field: value})

    with pytest.raises(ValueError, match=message):
        validate_campaign_shards((screen, confirm))


def _write_provenance_shard(
    root: Path, seeds: tuple[int, ...]
) -> tuple[Path, str, str]:
    root.mkdir()
    raw_sha256 = "9" * 64
    protocol = {
        "run_kind": "formal",
        "seeds": list(seeds),
        "datasets": [
            {
                "dataset_id": "toy",
                "raw_sha256": raw_sha256,
                "raw_byte_count": 3,
            }
        ],
        "model_policy_digest": "a" * 64,
        "registered_science": "same-across-shards",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(
            protocol,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resolved_protocol_path = root / "resolved_protocol.yaml"
    resolved_protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=True), encoding="utf-8"
    )
    atomic_write_json(
        root / "source_fingerprint.json",
        {"protocol_hash": protocol_hash, "files": {"src/model.py": "f" * 64}},
    )
    raw_authority = {
        "protocol_hash": protocol_hash,
        "datasets": {"toy": {"byte_count": 3, "sha256": raw_sha256}},
    }
    atomic_write_json(root / "validate" / "raw_authority.json", raw_authority)
    environment_digest = (
        "f1be56844e172b850f7788ba55b5eee3827995a2b24ba88c858f6551ceb07c31"
    )
    atomic_write_json(
        root / "environment.json",
        {
            "python": "3.13.9",
            "numpy": "2.4.6",
            "pandas": "3.0.3",
            "scikit_learn": "1.9.0",
            "xgboost": "3.3.0",
            "xgboost_build_info": {
                "CUDA_VERSION": [12, 9],
                "USE_CUDA": True,
                "libxgboost": "C:/local/venv/xgboost.dll",
            },
            "cuda_build_info": {"CUDA_VERSION": [12, 9], "USE_CUDA": True},
            "cuda_driver": "591.91",
            "os": "Windows-11",
            "timezone": "Local Zone",
            "utc_offset": "+08:00",
            "backend": "cuda",
            "scientific_environment_sha256": environment_digest,
        },
    )
    frozen_configs = root / "configs_frozen"
    frozen_configs.mkdir()
    pipeline_path = frozen_configs / "icomp2026_pipeline.resolved.yaml"
    pipeline_path.write_bytes(resolved_protocol_path.read_bytes())
    dataset_path = frozen_configs / "toy.yaml"
    dataset_path.write_text(
        yaml.safe_dump(protocol["datasets"][0], sort_keys=True),
        encoding="utf-8",
    )
    metadata_artifacts = (
        resolved_protocol_path,
        root / "environment.json",
        root / "source_fingerprint.json",
        pipeline_path,
        dataset_path,
    )
    atomic_write_json(
        root / "metadata_complete.json",
        {
            "schema_version": 1,
            "protocol_hash": protocol_hash,
            "scientific_environment_sha256": environment_digest,
            "artifacts": {
                path.relative_to(root).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in metadata_artifacts
            },
        },
    )
    for seed in seeds:
        bundle = root / "estimator" / "_bundles" / "toy" / f"s{seed}"
        bundle.mkdir(parents=True)
        (bundle / "bundle.json").write_text("{}\n", encoding="utf-8")
    return root, (
        "bf22bc9268e5252ee20d0304f738916e0768b531a680b7471633e32ae86f7ee6"
    ), environment_digest


def test_shard_record_derives_registered_raw_and_environment_identity(
    tmp_path: Path,
) -> None:
    root, raw_digest, environment_digest = _write_provenance_shard(
        tmp_path / "shard", (42,)
    )

    record = read_shard_record(root)

    assert record.raw_authority_digest == raw_digest
    assert record.scientific_environment_digest == environment_digest


def test_shard_record_requires_completed_frozen_metadata(
    tmp_path: Path,
) -> None:
    root, _, _ = _write_provenance_shard(tmp_path / "shard", (42,))
    (root / "metadata_complete.json").unlink()

    with pytest.raises(ValueError, match="metadata completion marker"):
        read_shard_record(root)


def test_shard_record_rejects_tampered_frozen_metadata_artifact(
    tmp_path: Path,
) -> None:
    root, _, _ = _write_provenance_shard(tmp_path / "shard", (42,))
    (root / "configs_frozen" / "toy.yaml").write_text(
        "tampered: true\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="frozen metadata artifact"):
        read_shard_record(root)


def test_shard_record_binds_protocol_hash_to_resolved_protocol_content(
    tmp_path: Path,
) -> None:
    root, _, _ = _write_provenance_shard(tmp_path / "shard", (42,))
    protocol_path = root / "resolved_protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["registered_science"] = "changed-after-freeze"
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=True), encoding="utf-8"
    )
    marker_path = root / "metadata_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifacts"]["resolved_protocol.yaml"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    atomic_write_json(marker_path, marker)

    with pytest.raises(ValueError, match="protocol hash|protocol identity"):
        read_shard_record(root)


def test_campaign_index_publishes_shared_raw_and_environment_identity(
    tmp_path: Path,
) -> None:
    first, raw_digest, environment_digest = _write_provenance_shard(
        tmp_path / "first", (42, 123)
    )
    second, _, _ = _write_provenance_shard(
        tmp_path / "second", (456, 789, 1024)
    )

    index = build_campaign_index((first, second))

    assert index["raw_authority_digest"] == raw_digest
    assert index["scientific_environment_digest"] == environment_digest


def test_shard_record_rejects_raw_authority_outside_frozen_protocol(
    tmp_path: Path,
) -> None:
    root, _, _ = _write_provenance_shard(tmp_path / "shard", (42,))
    atomic_write_json(
        root / "validate" / "raw_authority.json",
        {
            "protocol_hash": hashlib.sha256(root.name.encode("utf-8")).hexdigest(),
            "datasets": {"toy": {"byte_count": 4, "sha256": "8" * 64}},
        },
    )

    with pytest.raises(ValueError, match="raw authority"):
        read_shard_record(root)


def test_shard_record_rejects_environment_content_outside_frozen_digest(
    tmp_path: Path,
) -> None:
    root, _, _ = _write_provenance_shard(tmp_path / "shard", (42,))
    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["numpy"] = "9.9.9"
    atomic_write_json(environment_path, environment)
    marker_path = root / "metadata_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifacts"]["environment.json"] = hashlib.sha256(
        environment_path.read_bytes()
    ).hexdigest()
    atomic_write_json(marker_path, marker)

    with pytest.raises(ValueError, match="scientific environment digest"):
        read_shard_record(root)


def test_runtime_forecast_uses_guard_band_and_enforces_p0_cap() -> None:
    forecast = forecast_runtime(
        (
            RuntimeTier("P0", 100, 2.0),
            RuntimeTier("P1", 10, 1.0),
            RuntimeTier("P2", 5, 4.0),
        )
    )

    assert forecast["P0"]["estimated_s"] == 260.0
    assert forecast["P1"]["estimated_s"] == 13.0
    assert forecast["P2"]["estimated_s"] == 26.0

    with pytest.raises(ValueError, match="48 hours"):
        forecast_runtime((RuntimeTier("P0", 100_000, 4.0),))


def test_default_campaign_uses_three_exact_nonoverlapping_seed_shards() -> None:
    assert default_seed_shards() == (
        (42, 123),
        (456, 789),
        (1024,),
    )


def test_campaign_pipeline_command_is_explicit_and_resumable() -> None:
    command = build_pipeline_command(
        python=Path("venv/python.exe"),
        pipeline_script=Path("experiments/run_icomp2026_pipeline.py"),
        stage="estimator",
        config_dir=Path("configs"),
        output_root=Path("results/formal"),
        model_policy=Path("results/calibration/model_policy.json"),
        run_id="formal-s42-123",
        seeds=(42, 123),
        resume=True,
    )

    assert command == (
        "venv/python.exe",
        "experiments/run_icomp2026_pipeline.py",
        "--stage",
        "estimator",
        "--config-dir",
        "configs",
        "--output-root",
        "results/formal",
        "--backend",
        "cuda",
        "--model-policy",
        "results/calibration/model_policy.json",
        "--run-id",
        "formal-s42-123",
        "--seeds",
        "42",
        "123",
        "--resume",
    )


def _formal_raw_row(seed: int, sequence: int) -> dict[str, object]:
    return {
        "run_kind": "formal",
        "manifest_sequence": sequence,
        "stage": "removal",
        "dataset_id": "spambase",
        "seed": seed,
        "method": "minppi",
        "trial": 0,
        "parameter": "",
        "metric": "delta_utility",
        "value": 0.001 * seed,
    }


def test_campaign_merges_shard_rows_into_one_exact_five_seed_table() -> None:
    shards = (
        (
            (42, 123),
            [_formal_raw_row(42, 7), _formal_raw_row(123, 8)],
            [],
        ),
        (
            (456, 789),
            [_formal_raw_row(456, 3), _formal_raw_row(789, 4)],
            [],
        ),
        ((1024,), [_formal_raw_row(1024, 12)], []),
    )

    raw, front = merge_shard_summary_rows(shards)

    assert [row["seed"] for row in raw] == [42, 123, 456, 789, 1024]
    assert [row["manifest_sequence"] for row in raw] == [0, 1, 2, 3, 4]
    assert front == []


def test_campaign_builds_one_five_seed_aggregate_from_all_shards() -> None:
    shards = (
        ((42, 123), [_formal_raw_row(42, 0), _formal_raw_row(123, 1)], []),
        ((456, 789), [_formal_raw_row(456, 0), _formal_raw_row(789, 1)], []),
        ((1024,), [_formal_raw_row(1024, 0)], []),
    )

    tables = build_campaign_tables(shards)

    row = tables["aggregate"][0]
    assert row["stage"] == "removal"
    assert row["method"] == "minppi"
    assert row["metric"] == "delta_utility"
    assert row["n"] == 5


def test_removal_ranking_uses_datasets_not_seed_replicates_as_blocks() -> None:
    methods = (
        "full",
        "minppi",
        "shap_bottom",
        "minshap_bottom",
        "banzhaf_bottom",
        "random_matched",
    )
    rows = []
    sequence = 0
    for dataset_index in range(7):
        dataset = f"d{dataset_index}"
        for seed in (42, 123, 456, 789, 1024):
            for method_index, method in enumerate(methods):
                repeats = 3 if method == "random_matched" else 1
                for trial in range(repeats):
                    rows.append(
                        {
                            "run_kind": "formal",
                            "manifest_sequence": sequence,
                            "stage": "removal",
                            "dataset_id": dataset,
                            "seed": seed,
                            "method": method,
                            "trial": trial,
                            "parameter": "",
                            "metric": "selected_utility",
                            "value": 1.0 - 0.1 * method_index,
                        }
                    )
                sequence += 1

    report = analyze_removal_ranks(rows)

    assert report["block_unit"] == "dataset_mean_over_five_seeds"
    assert report["dataset_count"] == 7
    assert report["seed_count_per_dataset"] == 5
    assert report["mean_ranks"]["full"] == 1.0
    assert report["mean_ranks"]["random_matched"] == 6.0
    assert 0.0 <= report["friedman_p_value"] <= 1.0
    assert report["nemenyi_critical_difference_0_05"] > 0.0
    assert len(report["minppi_pairwise_wilcoxon_holm"]) == 5


def test_campaign_merge_rejects_duplicate_seed_authority() -> None:
    row = _formal_raw_row(42, 0)
    with pytest.raises(ValueError, match="overlap"):
        merge_shard_summary_rows(
            (
                ((42,), [row], []),
                ((42, 123, 456, 789, 1024), [row], []),
            )
        )


def test_campaign_summary_runs_once_after_all_shard_stages() -> None:
    shard_stages, summarize = campaign_execution_stages("summarize")

    assert shard_stages == (
        "validate",
        "preprocess",
        "estimator",
        "removal",
        "alpha",
        "nsga2",
        "ablation",
    )
    assert summarize is True
    assert campaign_execution_stages("estimator") == (
        ("validate", "preprocess", "estimator"),
        False,
    )


def test_campaign_monitor_sums_stage_states_without_losing_failures() -> None:
    summary = summarize_status_document(
        {
            "schema_version": 1,
            "approved_skips": {},
            "manifest_hash": "a" * 64,
            "stages": {
                "preprocess": {
                    "pending": 0,
                    "running": 0,
                    "success": 7,
                    "failed": 0,
                    "skipped": 0,
                },
                "estimator": {
                    "pending": 12,
                    "running": 4,
                    "success": 20,
                    "failed": 1,
                    "skipped": 0,
                },
            },
        }
    )

    assert (
        summary["estimator"]
        == "20/37 success, 4 running, 1 failed, 12 pending"
    )


@pytest.mark.parametrize(
    "watch_args",
    [
        ("--watch", "--interval-seconds", "300"),
        ("--watch",),
    ],
)
def test_campaign_monitor_watch_refreshes_snapshots_without_real_sleep(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    watch_args: tuple[str, ...],
) -> None:
    output_root = tmp_path / "results"
    campaign_id = "campaign"
    (output_root / campaign_id).mkdir(parents=True)
    (output_root / campaign_id / "campaign_plan.json").write_text(
        '{"run_ids": ["run-one"]}', encoding="utf-8"
    )
    (output_root / "run-one").mkdir()
    (output_root / "run-one" / "status.json").write_text(
        '{"stages": {}}', encoding="utf-8"
    )
    delays: list[float] = []

    def interrupt_after_two_intervals(seconds: float) -> None:
        delays.append(seconds)
        if len(delays) == 2:
            raise KeyboardInterrupt

    exit_code = campaign_monitor.main(
        (
            "--output-root",
            str(output_root),
            "--campaign-id",
            campaign_id,
            *watch_args,
        ),
        sleep=interrupt_after_two_intervals,
    )

    assert exit_code == 130
    assert delays == [300.0, 300.0]
    assert capsys.readouterr().out.count("run-one:") == 2


def test_campaign_monitor_watch_preserves_latest_failed_snapshot_exit_code(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    campaign_id = "campaign"
    (output_root / campaign_id).mkdir(parents=True)
    (output_root / campaign_id / "campaign_plan.json").write_text(
        '{"run_ids": ["run-one"]}', encoding="utf-8"
    )
    (output_root / "run-one").mkdir()
    (output_root / "run-one" / "status.json").write_text(
        """{
            "stages": {
                "estimator": {
                    "pending": 0,
                    "running": 0,
                    "success": 0,
                    "failed": 1,
                    "skipped": 0
                }
            }
        }""",
        encoding="utf-8",
    )

    def interrupt_after_snapshot(_seconds: float) -> None:
        raise KeyboardInterrupt

    exit_code = campaign_monitor.main(
        (
            "--output-root",
            str(output_root),
            "--campaign-id",
            campaign_id,
            "--watch",
            "--interval-seconds",
            "300",
        ),
        sleep=interrupt_after_snapshot,
    )

    assert exit_code == 1


@pytest.mark.parametrize(
    "interval",
    (
        "0",
        "-1",
        "1.5",
        "nan",
        "inf",
        "86401",
        "1" + "0" * 100,
        "1" + "0" * 400,
    ),
)
def test_campaign_monitor_rejects_invalid_or_unsafe_watch_intervals(
    interval: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        campaign_monitor.main(
            (
                "--campaign-id",
                "campaign",
                "--watch",
                "--interval-seconds",
                interval,
            )
        )

    assert error.value.code == 2
    assert "positive finite integer" in capsys.readouterr().err


def test_48h_forecast_scales_nested_budget_but_not_exact_dispatch() -> None:
    assert estimate_dataset_seconds(
        observed_seconds=100.0,
        observed_budget=5,
        target_budget=25,
        seed_count=5,
        exact=False,
    ) == 2500.0
    assert estimate_dataset_seconds(
        observed_seconds=100.0,
        observed_budget=5,
        target_budget=25,
        seed_count=5,
        exact=True,
    ) == 500.0


def test_48h_forecast_applies_measured_concurrency_and_guard_band() -> None:
    forecast = finalize_48h_forecast(
        sequential_estimator_seconds=25 * 3600.0,
        minimum_measured_speedup=1.25,
        support_stage_p90_hours=13.0,
        guard_band=1.30,
    )

    assert forecast["estimator_p50_wall_hours"] == 20.0
    assert forecast["estimator_p90_wall_hours"] == 26.0
    assert forecast["p0_total_p90_hours"] == 39.0
    assert forecast["passes_48h_gate"] is True
