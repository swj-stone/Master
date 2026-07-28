from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from experiments.summarize_multifidelity_campaign import (
    RuntimeTier,
    ShardRecord,
    analyze_removal_ranks,
    build_campaign_tables,
    forecast_runtime,
    merge_shard_summary_rows,
    validate_campaign_shards,
)
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
