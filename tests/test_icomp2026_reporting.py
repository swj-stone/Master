"""Focused deterministic reporting and figure tests for ICOMP 2026."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

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
                        "run_kind": "formal", "manifest_sequence": sequence,
                        "stage": "nsga2", "dataset_id": dataset, "seed": seed,
                        "method": method, "trial": 0, "parameter": "",
                        "metric": "test_hv", "value": value,
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
            "run_kind": "formal", "manifest_sequence": index,
            "stage": "removal", "dataset_id": "toy", "seed": seed,
            "method": "minppi", "trial": 0, "parameter": "",
            "metric": "delta_utility", "value": 0.01 * index,
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


def test_incomplete_paired_scope_is_omitted_and_holm_families_are_endpoint_local() -> None:
    raw = []
    sequence = 0
    for dataset in ("a", "b"):
        for seed in FORMAL_SEEDS:
            for metric in ("validation_hv", "test_hv"):
                for method, value in (("full", 0.5), ("minppi_prefilter", 0.6)):
                    if dataset == "b" and metric == "validation_hv" and method == "minppi_prefilter" and seed == 1024:
                        continue
                    raw.append(
                        {
                            "run_kind": "formal", "manifest_sequence": sequence,
                            "stage": "nsga2", "dataset_id": dataset, "seed": seed,
                            "method": method, "trial": 0, "parameter": "",
                            "metric": metric, "value": value,
                        }
                    )
                    sequence += 1
    tables = aggregate_canonical_records(raw, [])

    validation = [
        row for row in tables["paired"] if row["metric"] == "validation_hv"
    ]
    test_hv = [row for row in tables["paired"] if row["metric"] == "test_hv"]
    assert [row["dataset_id"] for row in validation] == ["a"]
    assert [row["dataset_id"] for row in test_hv] == ["a", "b", "__all__"]
    assert {row["holm_family_id"] for row in validation} == {
        "nsga2:validation_hv"
    }
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
            "run_kind": "formal", "stage": "removal",
            "dataset_id": "credit_card_fraud", "method": "minppi",
            "parameter": "", "metric": "compression", "n": 5,
            "mean": 0.25, "sample_std": 0.01, "median": 0.25,
            "ci95_low": 0.24, "ci95_high": 0.26,
            "analysis_label": "descriptive_low_power",
        },
        {
            "run_kind": "formal", "stage": "removal",
            "dataset_id": "credit_card_fraud", "method": "minppi",
            "parameter": "", "metric": "delta_utility", "n": 5,
            "mean": -0.0123, "sample_std": 0.002, "median": -0.012,
            "ci95_low": -0.014, "ci95_high": -0.010,
            "analysis_label": "descriptive_low_power",
        },
    ]
    paired = [
        {
            "run_kind": "formal", "stage": "nsga2", "dataset_id": "__all__",
            "metric": "test_hv", "baseline_method": "full",
            "comparison_method": "minppi_prefilter", "n": 5,
            "mean_difference": 0.0345, "median_difference": 0.03,
            "paired_rank_biserial": 0.6, "p_value": 0.01,
            "holm_p_value": 0.02, "holm_family_id": "nsga2:test_hv",
            "holm_family_size": 8, "test": "exact_paired_sign_flip",
            "analysis_label": "descriptive_low_power",
        }
    ]

    macros, provenance = generate_latex_macros(aggregate, paired, run_kind="formal")

    assert "\\newcommand{\\CreditCompression}{25.0\\%}" in macros
    assert "\\newcommand{\\CreditDeltaUtility}{-0.012}" in macros
    assert "\\newcommand{\\NsgaMeanHvChange}{+0.035}" in macros
    assert set(provenance["macros"]) == {
        "CreditCompression", "CreditDeltaUtility", "NsgaMeanHvChange"
    }
    for entry in provenance["macros"].values():
        assert set(entry) == {
            "source_table", "row_selector", "source_column",
            "raw_numeric_value", "formatted_value",
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
                "run_kind": "smoke", "manifest_sequence": 0, "stage": "estimator",
                "dataset_id": "spambase", "seed": 42, "method": "hybrid",
                "trial": 0, "parameter": "budget=2", "metric": "score_agreement",
                "value": 0.8,
            },
            {
                "run_kind": "smoke", "manifest_sequence": 1, "stage": "alpha",
                "dataset_id": "spambase", "seed": 42, "method": "alpha_evaluation",
                "trial": 0, "parameter": "alpha=0.95", "metric": "test_utility",
                "value": 0.9,
            },
        ]
    ).to_csv(raw, index=False, lineterminator="\n")
    aggregate_rows = [
            {
                "run_kind": "smoke", "stage": "estimator", "dataset_id": "spambase",
                "method": "hybrid", "parameter": "budget=2",
                "metric": metric, "n": 1, "mean": value, "sample_std": 0.0,
                "median": value, "ci95_low": value, "ci95_high": value,
                "analysis_label": "descriptive_smoke",
            }
            for metric, value in (("harmful_f1", 0.75), ("score_agreement", 0.8))
        ] + [
            {
                "run_kind": "smoke", "stage": "ablation",
                "dataset_id": "spambase", "method": axis,
                "parameter": f"{axis}={value}", "metric": "harmful_fraction",
                "n": 1, "mean": harmful_fraction, "sample_std": 0.0,
                "median": harmful_fraction, "ci95_low": harmful_fraction,
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
        ] + [
            {
                "run_kind": "smoke", "stage": "alpha", "dataset_id": "spambase",
                "method": "alpha_evaluation", "parameter": "alpha=0.95",
                "metric": metric, "n": 1, "mean": value, "sample_std": 0.0,
                "median": value, "ci95_low": value, "ci95_high": value,
                "analysis_label": "descriptive_smoke",
            }
            for metric, value in (("compression", 0.25), ("test_utility", 0.9))
        ] + [
            {
                "run_kind": "smoke", "stage": "removal", "dataset_id": "spambase",
                "method": method, "parameter": "", "metric": "delta_utility", "n": 1,
                "mean": value, "sample_std": 0.0, "median": value,
                "ci95_low": value, "ci95_high": value,
                "analysis_label": "descriptive_smoke",
            }
            for method, value in (
                ("minppi", -0.01), ("shap_bottom", -0.02),
                ("minshap_bottom", -0.03), ("banzhaf_bottom", -0.04),
                ("random_matched", -0.05),
            )
        ] + [
            {
                "run_kind": "smoke", "stage": "nsga2", "dataset_id": "spambase",
                "method": method, "parameter": "", "metric": metric, "n": 1,
                "mean": value, "sample_std": 0.0, "median": value,
                "ci95_low": value, "ci95_high": value,
                "analysis_label": "descriptive_smoke",
            }
            for method, metric, value in (
                ("full", "test_hv", 0.6),
                ("minppi_prefilter", "test_hv", 0.7),
                ("full", "convergence_evaluation_count", 8.0),
                ("minppi_prefilter", "convergence_evaluation_count", 4.0),
            )
        ]
    pd.DataFrame(aggregate_rows).to_csv(aggregate, index=False, lineterminator="\n")
    pd.DataFrame(
        columns=[
            "run_kind", "stage", "dataset_id", "metric", "baseline_method",
            "comparison_method", "n", "mean_difference", "median_difference",
            "paired_rank_biserial", "p_value", "holm_p_value", "holm_family_id",
            "holm_family_size", "test",
            "analysis_label",
        ]
    ).to_csv(paired, index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {
                "run_kind": "smoke", "dataset_id": "spambase", "seed": 42,
                "method": "full", "front_kind": "test", "point_index": 0,
                "compression": 0.2, "utility": 0.8,
                "evaluation": 0, "hypervolume": 0.6,
            },
            {
                "run_kind": "smoke", "dataset_id": "spambase", "seed": 42,
                "method": "minppi_prefilter", "front_kind": "test", "point_index": 0,
                "compression": 0.4, "utility": 0.85,
                "evaluation": 0, "hypervolume": 0.7,
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
        "fig1_overview.pdf", "fig2_cd_heatmap.pdf", "fig3_prefilter.pdf"
    ]
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in first] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in second
    ]
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in tables] == source_hashes
    assert all(path.parent == tmp_path / "figures-a" for path in first)


def test_figure_panels_ignore_unrelated_metrics_instead_of_mixing_units(
    tmp_path: Path,
) -> None:
    tables = _write_figure_tables(tmp_path)
    baseline = generate_all_figures(
        *tables, tmp_path / "baseline", run_kind="smoke"
    )
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
        ~(
            (aggregate["stage"] == "removal")
            & (aggregate["method"] == "banzhaf_bottom")
        )
    ]
    aggregate.to_csv(tables[1], index=False, lineterminator="\n")

    with pytest.raises(ValueError, match="removal.*matrix|missing.*removal"):
        generate_all_figures(*tables, tmp_path / "formal", run_kind="formal")
    assert not (tmp_path / "formal").exists()


def test_macro_provenance_is_canonical_json_serializable() -> None:
    _, provenance = generate_latex_macros([], [], run_kind="smoke")
    assert json.loads(json.dumps(provenance, allow_nan=False)) == provenance
