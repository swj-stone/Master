"""Focused deterministic reporting and figure tests for ICOMP 2026."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import experiments.make_paper_figures as figure_module
import experiments.summarize_icomp2026 as reporting_module
from experiments.make_paper_figures import generate_all_figures
from experiments.summarize_icomp2026 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    aggregate_canonical_records,
    generate_latex_macros,
    holm_adjust,
    paired_sign_flip,
    summarize_values,
)

FORMAL_SEEDS = (42, 123, 456, 789, 1024)


def test_grouped_estimator_reporting_schema_preserves_comparison_and_accounting_metrics() -> (
    None
):
    metrics = {
        "budget": 25,
        "model_fit_count": 100,
        "coalition_count": 101,
        "logical_evaluation_count": 2500,
        "logical_unique_coalition_count": 100,
        "physical_model_fit_count": 40,
        "cache_hit_count": 60,
        "runtime_s": 12.5,
        "group_request_digest": "a" * 64,
        "group_physical_model_fit_count": 400,
        "group_cache_hit_count": 600,
        "group_runtime_s": 60.0,
    }

    validated = reporting_module.validate_estimator_reporting_schema(
        metrics,
        execution_mode="grouped_nested_v1",
        exact_scores_available=False,
    )

    assert validated == metrics


def test_grouped_reporting_requires_group_fields_without_changing_legacy_schema() -> (
    None
):
    legacy = {
        "budget": 25,
        "model_fit_count": 100,
        "coalition_count": 101,
        "runtime_s": 12.5,
    }
    assert (
        reporting_module.validate_estimator_reporting_schema(
            legacy,
            execution_mode="legacy",
            exact_scores_available=False,
        )
        == legacy
    )
    with pytest.raises(ValueError, match="schema"):
        reporting_module.validate_estimator_reporting_schema(
            legacy,
            execution_mode="grouped_nested_v1",
            exact_scores_available=False,
        )


def test_grouped_reporting_deduplicates_physical_totals_by_request_digest() -> None:
    first_group = {
        "group_request_digest": "a" * 64,
        "group_physical_model_fit_count": 400,
        "group_cache_hit_count": 600,
        "group_runtime_s": 60.0,
    }
    second_group = {
        "group_request_digest": "b" * 64,
        "group_physical_model_fit_count": 50,
        "group_cache_hit_count": 75,
        "group_runtime_s": 10.0,
    }

    accounting = reporting_module.deduplicate_estimator_group_metrics(
        [first_group, dict(first_group), second_group]
    )

    assert accounting == {
        "groups": [first_group, second_group],
        "totals": {
            "deduplicated_group_count": 2,
            "deduplicated_group_physical_model_fit_count": 450,
            "deduplicated_group_cache_hit_count": 675,
            "deduplicated_group_runtime_s": 70.0,
        },
    }


def test_grouped_projection_reporting_excludes_complete_group_values() -> None:
    metrics = {
        "budget": 25,
        "model_fit_count": 100,
        "coalition_count": 101,
        "logical_evaluation_count": 2500,
        "logical_unique_coalition_count": 100,
        "physical_model_fit_count": 40,
        "cache_hit_count": 60,
        "runtime_s": 12.5,
        "group_request_digest": "a" * 64,
        "group_physical_model_fit_count": 400,
        "group_cache_hit_count": 600,
        "group_runtime_s": 60.0,
    }

    projected = reporting_module.estimator_projection_reporting_metrics(
        metrics,
        execution_mode="grouped_nested_v1",
    )

    assert projected == {
        "budget": 25,
        "model_fit_count": 100,
        "coalition_count": 101,
        "logical_evaluation_count": 2500,
        "logical_unique_coalition_count": 100,
        "physical_model_fit_count": 40,
        "cache_hit_count": 60,
        "runtime_s": 12.5,
    }


def test_grouped_reporting_rejects_conflicting_duplicate_group_values() -> None:
    group = {
        "group_request_digest": "a" * 64,
        "group_physical_model_fit_count": 400,
        "group_cache_hit_count": 600,
        "group_runtime_s": 60.0,
    }
    with pytest.raises(ValueError, match="conflicting.*request digest"):
        reporting_module.deduplicate_estimator_group_metrics(
            [group, {**group, "group_runtime_s": 61.0}]
        )


def test_five_seed_statistics_are_exact_deterministic_and_low_power() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    summary = summarize_values(values)

    assert BOOTSTRAP_SEED == 42
    assert BOOTSTRAP_RESAMPLES == 10_000
    assert summary == {
        "n": 5,
        "mean": 3.0,
        "sample_std": pytest.approx(2.5**0.5),
        "median": 3.0,
        "ci95_low": 1.8,
        "ci95_high": 4.2,
    }


def test_paired_sign_flip_requires_exact_seed_alignment_and_holm_is_local() -> None:
    left = {42: 0.10, 123: 0.20, 456: 0.30, 789: 0.40, 1024: 0.50}
    right = {42: 0.20, 123: 0.40, 456: 0.60, 789: 0.80, 1024: 1.00}

    result = paired_sign_flip(left, right)

    assert result["n"] == 5
    assert result["mean_difference"] == pytest.approx(0.3)
    assert result["p_value"] == pytest.approx(0.0625)
    assert result["paired_rank_biserial"] == 1.0
    assert result["test"] == "exact_paired_sign_flip"
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(ValueError, match="seed alignment"):
        paired_sign_flip(left, {**right, 2048: 1.2})


def test_paired_rank_biserial_uses_signed_ranks_not_sign_counts() -> None:
    result = paired_sign_flip(
        {42: 0.0, 123: 0.0, 456: 0.0},
        {42: 1.0, 123: 2.0, 456: -3.0},
    )
    assert result["paired_rank_biserial"] == 0.0


def test_pooled_nsga_comparison_still_uses_five_seed_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = []
    sequence = 0
    for dataset in (f"dataset_{index}" for index in range(7)):
        for seed in (42, 123, 456, 789, 1024):
            for method, value in (("full", 0.5), ("minppi_prefilter", 0.6)):
                raw.append(
                    {
                        "run_kind": "formal",
                        "manifest_sequence": sequence,
                        "stage": "nsga2",
                        "dataset_id": dataset,
                        "seed": seed,
                        "method": method,
                        "trial": 0,
                        "parameter": "",
                        "metric": "test_hv",
                        "value": value,
                    }
                )
                sequence += 1
    real = reporting_module.paired_sign_flip
    observed_sizes = []

    def bounded(left, right):
        observed_sizes.append((len(left), len(right)))
        return real(left, right)

    monkeypatch.setattr(reporting_module, "paired_sign_flip", bounded)
    tables = reporting_module.aggregate_canonical_records(raw, [])

    pooled = next(row for row in tables["paired"] if row["dataset_id"] == "__all__")
    assert pooled["n"] == 5
    assert max(max(pair) for pair in observed_sizes) == 5


def test_formal_aggregation_requires_exact_canonical_five_seed_scope() -> None:
    raw = [
        {
            "run_kind": "formal",
            "manifest_sequence": index,
            "stage": "removal",
            "dataset_id": "toy",
            "seed": seed,
            "method": "minppi",
            "trial": 0,
            "parameter": "",
            "metric": "delta_utility",
            "value": 0.01 * index,
        }
        for index, seed in enumerate(FORMAL_SEEDS[:-1])
    ]
    with pytest.raises(ValueError, match="formal.*seed"):
        aggregate_canonical_records(raw, [])
    tables = aggregate_canonical_records(
        raw,
        [],
        allow_approved_skip_subset=True,
    )
    assert {row["seed"] for row in tables["raw"]} == set(FORMAL_SEEDS[:-1])
    assert tables["paired"] == []
    with pytest.raises(ValueError, match="nonempty.*seed subset"):
        aggregate_canonical_records(
            [],
            [],
            allow_approved_skip_subset=True,
        )


def test_incomplete_paired_scope_is_omitted_and_holm_families_are_endpoint_local() -> (
    None
):
    raw = []
    sequence = 0
    for dataset in ("a", "b"):
        for seed in FORMAL_SEEDS:
            for metric in ("validation_hv", "test_hv"):
                for method, value in (("full", 0.5), ("minppi_prefilter", 0.6)):
                    if (
                        dataset == "b"
                        and metric == "validation_hv"
                        and method == "minppi_prefilter"
                        and seed == 1024
                    ):
                        continue
                    raw.append(
                        {
                            "run_kind": "formal",
                            "manifest_sequence": sequence,
                            "stage": "nsga2",
                            "dataset_id": dataset,
                            "seed": seed,
                            "method": method,
                            "trial": 0,
                            "parameter": "",
                            "metric": metric,
                            "value": value,
                        }
                    )
                    sequence += 1
    tables = aggregate_canonical_records(raw, [])

    validation = [row for row in tables["paired"] if row["metric"] == "validation_hv"]
    test_hv = [row for row in tables["paired"] if row["metric"] == "test_hv"]
    assert [row["dataset_id"] for row in validation] == ["a"]
    assert [row["dataset_id"] for row in test_hv] == ["a", "b", "__all__"]
    assert {row["holm_family_id"] for row in validation} == {"nsga2:validation_hv"}
    assert {row["holm_family_size"] for row in validation} == {1}
    assert {row["holm_family_id"] for row in test_hv} == {"nsga2:test_hv"}
    assert {row["holm_family_size"] for row in test_hv} == {3}
    assert all(row["holm_p_value"] == pytest.approx(0.1875) for row in test_hv)


def test_raw_trials_are_preserved_and_aggregates_use_seed_units() -> None:
    raw = []
    for seed, value in zip((42, 123, 456, 789, 1024), (1, 2, 3, 4, 5), strict=True):
        for trial in (0, 1):
            raw.append(
                {
                    "run_kind": "formal",
                    "manifest_sequence": seed,
                    "stage": "removal",
                    "dataset_id": "toy",
                    "seed": seed,
                    "method": "random_matched",
                    "trial": trial,
                    "parameter": "",
                    "metric": "delta_utility",
                    "value": float(value + trial),
                }
            )

    tables = aggregate_canonical_records(raw, [])

    assert list(tables) == ["raw", "aggregate", "paired", "front"]
    assert [row["value"] for row in tables["raw"]] == [
        float(value + trial)
        for seed, value in zip((42, 123, 456, 789, 1024), (1, 2, 3, 4, 5), strict=True)
        for trial in (0, 1)
    ]
    row = tables["aggregate"][0]
    assert row["n"] == 5
    assert row["mean"] == 3.5
    assert row["sample_std"] == pytest.approx(2.5**0.5)
    assert row["analysis_label"] == "descriptive_low_power"


def test_latex_macros_have_complete_provenance_and_smoke_never_invents_credit() -> None:
    aggregate = [
        {
            "run_kind": "formal",
            "stage": "removal",
            "dataset_id": "credit_card_fraud",
            "method": "minppi",
            "parameter": "",
            "metric": "compression",
            "n": 5,
            "mean": 0.25,
            "sample_std": 0.01,
            "median": 0.25,
            "ci95_low": 0.24,
            "ci95_high": 0.26,
            "analysis_label": "descriptive_low_power",
        },
        {
            "run_kind": "formal",
            "stage": "removal",
            "dataset_id": "credit_card_fraud",
            "method": "minppi",
            "parameter": "",
            "metric": "delta_utility",
            "n": 5,
            "mean": -0.0123,
            "sample_std": 0.002,
            "median": -0.012,
            "ci95_low": -0.014,
            "ci95_high": -0.010,
            "analysis_label": "descriptive_low_power",
        },
    ]
    paired = [
        {
            "run_kind": "formal",
            "stage": "nsga2",
            "dataset_id": "__all__",
            "metric": "test_hv",
            "baseline_method": "full",
            "comparison_method": "minppi_prefilter",
            "n": 5,
            "mean_difference": 0.0345,
            "median_difference": 0.03,
            "paired_rank_biserial": 0.6,
            "p_value": 0.01,
            "holm_p_value": 0.02,
            "holm_family_id": "nsga2:test_hv",
            "holm_family_size": 8,
            "test": "exact_paired_sign_flip",
            "analysis_label": "descriptive_low_power",
        }
    ]

    macros, provenance = generate_latex_macros(aggregate, paired, run_kind="formal")

    assert "\\newcommand{\\CreditCompression}{25.0\\%}" in macros
    assert "\\newcommand{\\CreditDeltaUtility}{-0.012}" in macros
    assert "\\newcommand{\\NsgaMeanHvChange}{+0.035}" in macros
    assert set(provenance["macros"]) == {
        "CreditCompression",
        "CreditDeltaUtility",
        "NsgaMeanHvChange",
    }
    for entry in provenance["macros"].values():
        assert set(entry) == {
            "source_table",
            "row_selector",
            "source_column",
            "raw_numeric_value",
            "formatted_value",
        }
    smoke_macros, smoke_provenance = generate_latex_macros(
        [{**aggregate[0], "run_kind": "smoke", "dataset_id": "spambase"}],
        [],
        run_kind="smoke",
    )
    assert "Credit" not in smoke_macros
    assert not set(smoke_provenance["macros"]).intersection(
        {"CreditCompression", "CreditDeltaUtility"}
    )

    invalid, invalid_provenance = generate_latex_macros(
        aggregate,
        [{**paired[0], "n": 4}],
        run_kind="formal",
    )
    assert "NsgaMeanHvChange" not in invalid
    assert "NsgaMeanHvChange" not in invalid_provenance["macros"]


def _write_figure_tables(root: Path) -> tuple[Path, Path, Path, Path]:
    raw = root / "raw_metrics.csv"
    aggregate = root / "aggregate_metrics.csv"
    paired = root / "paired_comparisons.csv"
    front = root / "plot_sources.csv"
    pd.DataFrame(
        [
            {
                "run_kind": "smoke",
                "manifest_sequence": 0,
                "stage": "estimator",
                "dataset_id": "spambase",
                "seed": 42,
                "method": "hybrid",
                "trial": 0,
                "parameter": "budget=2",
                "metric": "score_agreement",
                "value": 0.8,
            },
            {
                "run_kind": "smoke",
                "manifest_sequence": 1,
                "stage": "alpha",
                "dataset_id": "spambase",
                "seed": 42,
                "method": "alpha_evaluation",
                "trial": 0,
                "parameter": "alpha=0.95",
                "metric": "test_utility",
                "value": 0.9,
            },
        ]
    ).to_csv(raw, index=False, lineterminator="\n")
    aggregate_rows = (
        [
            {
                "run_kind": "smoke",
                "stage": "estimator",
                "dataset_id": "spambase",
                "method": "hybrid",
                "parameter": "budget=2",
                "metric": metric,
                "n": 1,
                "mean": value,
                "sample_std": 0.0,
                "median": value,
                "ci95_low": value,
                "ci95_high": value,
                "analysis_label": "descriptive_smoke",
            }
            for metric, value in (("harmful_f1", 0.75), ("score_agreement", 0.8))
        ]
        + [
            {
                "run_kind": "smoke",
                "stage": "ablation",
                "dataset_id": "spambase",
                "method": axis,
                "parameter": f"{axis}={value}",
                "metric": "harmful_fraction",
                "n": 1,
                "mean": harmful_fraction,
                "sample_std": 0.0,
                "median": harmful_fraction,
                "ci95_low": harmful_fraction,
                "ci95_high": harmful_fraction,
                "analysis_label": "descriptive_smoke",
            }
            for axis, value, harmful_fraction in (
                ("guidance", "random", 0.50),
                ("guidance", "mi", 0.45),
                ("guidance", "correlation", 0.40),
                ("guidance", "mi_correlation", 0.35),
                ("dispatcher", "random_mc", 0.55),
                ("dispatcher", "mi_correlation_guided", 0.38),
                ("dispatcher", "hybrid", 0.30),
            )
        ]
        + [
            {
                "run_kind": "smoke",
                "stage": "alpha",
                "dataset_id": "spambase",
                "method": "alpha_evaluation",
                "parameter": "alpha=0.95",
                "metric": metric,
                "n": 1,
                "mean": value,
                "sample_std": 0.0,
                "median": value,
                "ci95_low": value,
                "ci95_high": value,
                "analysis_label": "descriptive_smoke",
            }
            for metric, value in (("compression", 0.25), ("test_utility", 0.9))
        ]
        + [
            {
                "run_kind": "smoke",
                "stage": "removal",
                "dataset_id": "spambase",
                "method": method,
                "parameter": "",
                "metric": "delta_utility",
                "n": 1,
                "mean": value,
                "sample_std": 0.0,
                "median": value,
                "ci95_low": value,
                "ci95_high": value,
                "analysis_label": "descriptive_smoke",
            }
            for method, value in (
                ("minppi", -0.01),
                ("shap_bottom", -0.02),
                ("minshap_bottom", -0.03),
                ("banzhaf_bottom", -0.04),
                ("random_matched", -0.05),
            )
        ]
        + [
            {
                "run_kind": "smoke",
                "stage": "nsga2",
                "dataset_id": "spambase",
                "method": method,
                "parameter": "",
                "metric": metric,
                "n": 1,
                "mean": value,
                "sample_std": 0.0,
                "median": value,
                "ci95_low": value,
                "ci95_high": value,
                "analysis_label": "descriptive_smoke",
            }
            for method, metric, value in (
                ("full", "test_hv", 0.6),
                ("minppi_prefilter", "test_hv", 0.7),
                ("full", "convergence_evaluation_count", 8.0),
                ("minppi_prefilter", "convergence_evaluation_count", 4.0),
            )
        ]
    )
    pd.DataFrame(aggregate_rows).to_csv(aggregate, index=False, lineterminator="\n")
    pd.DataFrame(
        columns=[
            "run_kind",
            "stage",
            "dataset_id",
            "metric",
            "baseline_method",
            "comparison_method",
            "n",
            "mean_difference",
            "median_difference",
            "paired_rank_biserial",
            "p_value",
            "holm_p_value",
            "holm_family_id",
            "holm_family_size",
            "test",
            "analysis_label",
        ]
    ).to_csv(paired, index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {
                "run_kind": "smoke",
                "dataset_id": "spambase",
                "seed": 42,
                "method": "full",
                "front_kind": "test",
                "point_index": 0,
                "compression": 0.2,
                "utility": 0.8,
                "evaluation": 0,
                "hypervolume": 0.6,
            },
            {
                "run_kind": "smoke",
                "dataset_id": "spambase",
                "seed": 42,
                "method": "minppi_prefilter",
                "front_kind": "test",
                "point_index": 0,
                "compression": 0.4,
                "utility": 0.85,
                "evaluation": 0,
                "hypervolume": 0.7,
            },
        ]
    ).to_csv(front, index=False, lineterminator="\n")
    return raw, aggregate, paired, front


def test_figures_are_read_only_explicit_smoke_labeled_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    tables = _write_figure_tables(tmp_path)
    source_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in tables]
    first = generate_all_figures(*tables, tmp_path / "figures-a", run_kind="smoke")
    second = generate_all_figures(*tables, tmp_path / "figures-b", run_kind="smoke")

    assert [path.name for path in first] == [
        "fig1_overview.pdf",
        "fig2_cd_heatmap.pdf",
        "fig3_prefilter.pdf",
    ]
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in first] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in second
    ]
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in tables
    ] == source_hashes
    assert all(path.parent == tmp_path / "figures-a" for path in first)


def test_figure_panels_ignore_unrelated_metrics_instead_of_mixing_units(
    tmp_path: Path,
) -> None:
    tables = _write_figure_tables(tmp_path)
    baseline = generate_all_figures(*tables, tmp_path / "baseline", run_kind="smoke")
    aggregate = pd.read_csv(tables[1], keep_default_na=False)
    template = aggregate.iloc[0].to_dict()
    aggregate.loc[len(aggregate)] = {
        **template,
        "metric": "runtime_s",
        "mean": 9999.0,
        "median": 9999.0,
        "ci95_low": 9999.0,
        "ci95_high": 9999.0,
    }
    ablation_template = aggregate[aggregate["stage"] == "ablation"].iloc[0].to_dict()
    for metric, value in (("model_fit_count", 12345.0), ("runtime_s", 9876.0)):
        aggregate.loc[len(aggregate)] = {
            **ablation_template,
            "metric": metric,
            "mean": value,
            "median": value,
            "ci95_low": value,
            "ci95_high": value,
        }
    aggregate.to_csv(tables[1], index=False, lineterminator="\n")
    with_extra = generate_all_figures(
        *tables, tmp_path / "with-extra", run_kind="smoke"
    )
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in baseline] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in with_extra
    ]


def _overview_panels() -> dict[str, pd.DataFrame]:
    alpha = pd.DataFrame(
        [
            {
                "dataset_id": "spambase",
                "method": "alpha_evaluation",
                "parameter": f"alpha={alpha_value}",
                "compression": compression,
                "test_utility": utility,
                "alpha": alpha_value,
            }
            for alpha_value, compression, utility in (
                (0.85, 0.20, 0.88),
                (0.95, 0.30, 0.90),
            )
        ]
    )
    front = pd.DataFrame(
        [
            {
                "dataset_id": "spambase",
                "seed": 42,
                "method": method,
                "front_kind": "test",
                "point_index": point_index,
                "compression": compression,
                "utility": utility,
            }
            for method, points in (
                ("full", ((0.10, 0.82), (0.25, 0.88))),
                ("minppi_prefilter", ((0.20, 0.84), (0.40, 0.91))),
            )
            for point_index, (compression, utility) in enumerate(points)
        ]
    )
    return {
        "estimator": pd.DataFrame(columns=("method", "budget", "metric", "mean")),
        "ablation": pd.DataFrame(columns=("dataset_id", "axis", "value_name", "mean")),
        "alpha": alpha,
        "front": front,
    }


def _formal_overview_panels() -> dict[str, pd.DataFrame]:
    panels = _overview_panels()
    alpha_rows = [
        {
            "dataset_id": dataset,
            "method": "alpha_evaluation",
            "parameter": f"alpha={alpha_value:.17g}",
            "compression": compression,
            "test_utility": utility,
            "alpha": alpha_value,
        }
        for dataset, base in (("madelon", 0.20), ("credit_card_fraud", 0.30))
        for alpha_value, offset in (
            (0.85, 0.01),
            (0.90, 0.02),
            (0.95, 0.03),
            (0.99, 0.04),
        )
        for compression, utility in ((base + offset, 0.90 + offset),)
    ]
    panels["alpha"] = pd.DataFrame(alpha_rows)
    panels["alpha_aggregate_rows"] = pd.DataFrame(
        [
            {
                "dataset_id": row["dataset_id"],
                "method": row["method"],
                "parameter": row["parameter"],
                "metric": metric,
                "n": 1,
                "mean": row[metric],
            }
            for row in alpha_rows
            for metric in ("compression", "test_utility")
        ]
    )
    panels["raw"] = pd.DataFrame(
        [
            {
                "run_kind": "formal",
                "manifest_sequence": index,
                "stage": "alpha",
                "dataset_id": row["dataset_id"],
                "seed": 42,
                "method": row["method"],
                "trial": 0,
                "parameter": row["parameter"],
                "metric": metric,
                "value": row[metric],
            }
            for index, row in enumerate(alpha_rows)
            for metric in ("compression", "test_utility")
        ]
    )
    panels["front"] = pd.DataFrame(
        [
            {
                "dataset_id": dataset,
                "seed": seed,
                "method": method,
                "front_kind": "test",
                "point_index": 0,
                "compression": compression,
                "utility": utility,
            }
            for dataset in ("madelon", "credit_card_fraud", "spambase")
            for seed in (42, 123)
            for method in ("full", "minppi_prefilter")
            for compression, utility in (((0.20, 0.80) if seed == 42 else (8.0, 8.0)),)
        ]
    )
    return panels


def test_article_fig3_overview_uses_two_rows_with_dedicated_legends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_figure(fig: object, destination: Path) -> Path:
        captured["figure"] = fig
        captured["destination"] = destination
        return destination

    monkeypatch.setattr(figure_module, "_save", capture_figure)
    figure_module._render_fig1(_overview_panels(), tmp_path, run_kind="smoke")
    fig = captured["figure"]
    try:
        assert len(fig.axes) == 6
        top_axes = fig.axes[:3]
        assert [axis.get_title() for axis in top_axes] == [
            "(a) MinSS classification (Example 2)",
            "(b) Conditional detection vs budget K",
            "(c) Alpha points and held-out Pareto fronts",
        ]
        assert [patch.get_height() for patch in top_axes[0].patches] == [
            0,
            -1,
            0,
            0,
            0,
        ]
        assert [
            tuple(point) for point in top_axes[0].collections[0].get_offsets()
        ] == pytest.approx([(0, 0), (2, 0), (3, 0), (4, 0)])
        assert all(axis.get_legend() is None for axis in top_axes)
        assert all(not axis.axison for axis in fig.axes[3:])
        assert all(axis.get_legend() is not None for axis in fig.axes[3:])
        assert "outside smoke scope" in top_axes[2].texts[0].get_text()
        assert captured["destination"] == tmp_path / "fig1_overview.pdf"
    finally:
        figure_module.plt.close(fig)


def test_article_fig3_detection_curve_uses_exact_example2_probability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_figure(fig: object, destination: Path) -> Path:
        captured["figure"] = fig
        return destination

    monkeypatch.setattr(figure_module, "_save", capture_figure)
    figure_module._render_fig1(_overview_panels(), tmp_path, run_kind="smoke")
    fig = captured["figure"]
    try:
        curves = {
            line.get_label(): line
            for line in fig.axes[1].lines
            if not line.get_label().startswith("_")
        }
        random_mc = curves["Algorithm A"]
        guided = curves["Algorithm B"]
        assert random_mc.get_xdata() == pytest.approx([1, 2, 3, 5, 10, 20, 50, 100])
        assert random_mc.get_ydata() == pytest.approx(
            [1.0 - (29.0 / 30.0) ** k for k in random_mc.get_xdata()]
        )
        assert guided.get_ydata() == pytest.approx(
            [1.0 - (1.0 - 0.129) ** k for k in guided.get_xdata()]
        )
        assert all(
            "38" not in text.get_text()
            for text in fig.findobj(match=figure_module.plt.Text)
        )
    finally:
        figure_module.plt.close(fig)


def test_article_fig3_formal_panel_selects_seed42_named_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panels = _formal_overview_panels()
    captured: dict[str, object] = {}

    def capture_figure(fig: object, destination: Path) -> Path:
        captured["figure"] = fig
        return destination

    monkeypatch.setattr(figure_module, "_save", capture_figure)
    figure_module._render_fig1(panels, tmp_path, run_kind="formal")
    fig = captured["figure"]
    try:
        front_offsets = [
            tuple(collection.get_offsets()[0]) for collection in fig.axes[2].collections
        ]
        assert len(front_offsets) == 4
        assert front_offsets == pytest.approx([(0.20, 0.80)] * 4)
        legend_text = [text.get_text() for text in fig.axes[5].get_legend().get_texts()]
        assert len(legend_text) == 6
        assert any("Madelon" in text for text in legend_text)
        assert any("Credit-Card-Fraud" in text for text in legend_text)
        assert all("Spambase" not in text for text in legend_text)
    finally:
        figure_module.plt.close(fig)


def test_article_fig3_formal_fails_without_complete_seed42_named_fronts(
    tmp_path: Path,
) -> None:
    panels = _formal_overview_panels()
    panels["front"] = panels["front"][
        panels["front"]["dataset_id"] != "credit_card_fraud"
    ]

    with pytest.raises(ValueError, match="seed-42.*Madelon.*Credit-Card-Fraud"):
        figure_module._render_fig1(panels, tmp_path, run_kind="formal")


@pytest.mark.parametrize("case", ("missing", "duplicate", "wrong_seed"))
def test_article_fig3_formal_alpha_authority_fails_closed(
    tmp_path: Path, case: str
) -> None:
    panels = _formal_overview_panels()
    target = panels["raw"].index == panels["raw"].index[0]
    if case == "missing":
        panels["raw"] = panels["raw"][~target]
    elif case == "duplicate":
        panels["raw"] = pd.concat(
            [panels["raw"], panels["raw"][target]], ignore_index=True
        )
    else:
        panels["raw"].loc[target, "seed"] = 123

    with pytest.raises(ValueError, match="formal overview|seed 42"):
        figure_module._render_fig1(panels, tmp_path, run_kind="formal")


@pytest.mark.parametrize("foreign_dataset", (False, True))
def test_formal_figures_require_exact_seven_dataset_authority(
    tmp_path: Path, foreign_dataset: bool
) -> None:
    tables = _write_figure_tables(tmp_path)
    for path in tables:
        frame = pd.read_csv(path, keep_default_na=False)
        if "run_kind" in frame:
            frame["run_kind"] = "formal"
        if "n" in frame:
            frame["n"] = 5
        if foreign_dataset and "dataset_id" in frame:
            frame["dataset_id"] = frame["dataset_id"].replace(
                {"spambase": "foreign_dataset"}
            )
        frame.to_csv(path, index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="formal NSGA-II.*seven|canonical dataset"):
        generate_all_figures(
            *tables, tmp_path / "invalid-formal-datasets", run_kind="formal"
        )
    assert not (tmp_path / "invalid-formal-datasets").exists()


def test_figures_reject_foreign_nsga_parameter(
    tmp_path: Path,
) -> None:
    tables = _write_figure_tables(tmp_path)
    aggregate = pd.read_csv(tables[1], keep_default_na=False)
    aggregate.loc[aggregate["stage"] == "nsga2", "parameter"] = "foreign=1"
    aggregate.to_csv(tables[1], index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="NSGA-II.*parameter"):
        generate_all_figures(
            *tables, tmp_path / "foreign-nsga-parameter", run_kind="smoke"
        )
    assert not (tmp_path / "foreign-nsga-parameter").exists()


def test_prefilter_asset_retains_two_registered_endpoint_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nsga = pd.DataFrame(
        [
            {
                "dataset_id": "spambase",
                "method": method,
                "metric": metric,
                "mean": value,
                "sample_std": 0.1,
            }
            for method, metric, value in (
                ("full", "test_hv", 0.6),
                ("minppi_prefilter", "test_hv", 0.7),
                ("full", "convergence_evaluation_count", 8.0),
                ("minppi_prefilter", "convergence_evaluation_count", 4.0),
            )
        ]
    )
    captured: dict[str, object] = {}

    def capture_figure(fig: object, destination: Path) -> Path:
        captured["figure"] = fig
        return destination

    monkeypatch.setattr(figure_module, "_save", capture_figure)
    figure_module._render_fig3({"nsga": nsga}, tmp_path, run_kind="smoke")
    fig = captured["figure"]
    try:
        assert len(fig.axes) == 2
        assert [axis.get_title() for axis in fig.axes] == [
            "(a) Held-out hypervolume",
            "(b) Convergence evaluations",
        ]
        for axis in fig.axes:
            assert [label.get_text() for label in axis.get_xticklabels()] == [
                "spambase"
            ]
            assert [text.get_text() for text in axis.get_legend().get_texts()] == [
                "full",
                "minppi_prefilter",
            ]
    finally:
        figure_module.plt.close(fig)


@pytest.mark.parametrize("case", ("missing", "duplicate"))
def test_formal_figures_fail_closed_on_incomplete_ablation_selectors(
    tmp_path: Path, case: str
) -> None:
    tables = _write_figure_tables(tmp_path)
    for path in tables:
        frame = pd.read_csv(path, keep_default_na=False)
        if "run_kind" in frame:
            frame["run_kind"] = "formal"
        if "n" in frame:
            frame["n"] = 5
        frame.to_csv(path, index=False, lineterminator="\n")
    aggregate = pd.read_csv(tables[1], keep_default_na=False)
    selected = (
        (aggregate["stage"] == "ablation")
        & (aggregate["parameter"] == "guidance=random")
        & (aggregate["metric"] == "harmful_fraction")
    )
    if case == "missing":
        aggregate = aggregate[~selected]
    else:
        duplicate = aggregate[selected].iloc[0].copy()
        duplicate["mean"] = 0.51
        duplicate["median"] = 0.51
        aggregate.loc[len(aggregate)] = duplicate
    aggregate.to_csv(tables[1], index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="ablation.*selector|missing.*ablation"):
        generate_all_figures(*tables, tmp_path / case, run_kind="formal")
    assert not (tmp_path / case).exists()


def test_formal_figures_fail_closed_on_missing_required_removal_selector(
    tmp_path: Path,
) -> None:
    tables = _write_figure_tables(tmp_path)
    for path in tables:
        frame = pd.read_csv(path, keep_default_na=False)
        if "run_kind" in frame:
            frame["run_kind"] = "formal"
        if "n" in frame:
            frame["n"] = 5
        frame.to_csv(path, index=False, lineterminator="\n")
    aggregate = pd.read_csv(tables[1], keep_default_na=False)
    aggregate = aggregate[
        ~((aggregate["stage"] == "removal") & (aggregate["method"] == "banzhaf_bottom"))
    ]
    aggregate.to_csv(tables[1], index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="removal.*matrix|missing.*removal"):
        generate_all_figures(*tables, tmp_path / "formal", run_kind="formal")
    assert not (tmp_path / "formal").exists()


def test_macro_provenance_is_canonical_json_serializable() -> None:
    _, provenance = generate_latex_macros([], [], run_kind="smoke")
    assert json.loads(json.dumps(provenance, allow_nan=False)) == provenance
