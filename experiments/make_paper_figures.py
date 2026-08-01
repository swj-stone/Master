"""Read-only deterministic paper figures from Task 11 canonical tables."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PDF_METADATA = {
    "Creator": "minppi-mofs Task 11",
    "Producer": "matplotlib",
    "CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
}
_RC = {
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.compression": 6,
}
_EXPECTED_RAW = (
    "run_kind", "manifest_sequence", "stage", "dataset_id", "seed", "method",
    "trial", "parameter", "metric", "value",
)
_EXPECTED_AGGREGATE = (
    "run_kind", "stage", "dataset_id", "method", "parameter", "metric", "n",
    "mean", "sample_std", "median", "ci95_low", "ci95_high", "analysis_label",
)
_EXPECTED_PAIRED = (
    "run_kind", "stage", "dataset_id", "metric", "baseline_method",
    "comparison_method", "n", "mean_difference", "median_difference",
    "paired_rank_biserial", "p_value", "holm_p_value", "holm_family_id",
    "holm_family_size", "test", "analysis_label",
)
_EXPECTED_FRONT = (
    "run_kind", "dataset_id", "seed", "method", "front_kind", "point_index",
    "compression", "utility", "evaluation", "hypervolume",
)
_REMOVAL_METHODS = (
    "minppi", "shap_bottom", "minshap_bottom", "banzhaf_bottom", "random_matched"
)
_NSGA_METHODS = ("full", "minppi_prefilter")
_REQUIRED_ABLATION_SELECTORS = {
    ("guidance", "random"),
    ("guidance", "mi"),
    ("guidance", "correlation"),
    ("guidance", "mi_correlation"),
    ("dispatcher", "random_mc"),
    ("dispatcher", "mi_correlation_guided"),
    ("dispatcher", "hybrid"),
}
_OPTIONAL_ABLATION_SELECTORS = {
    ("dispatcher", "exact"),
    ("utility", "accuracy"),
    ("utility", "ovr_auc"),
}


def _read_table(path: Path, expected: tuple[str, ...], run_kind: str) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"canonical figure table is not a regular file: {source}")
    try:
        payload = source.read_bytes()
        if b"\r" in payload:
            raise ValueError("canonical figure table must use LF line endings")
        payload.decode("utf-8")
        frame = pd.read_csv(source, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"canonical figure table is malformed: {source.name}") from exc
    if tuple(frame.columns) != expected:
        raise ValueError(f"canonical figure table schema is invalid: {source.name}")
    if not frame.empty and set(frame["run_kind"].astype(str)) != {run_kind}:
        raise ValueError("figure table contains a foreign run kind")
    if frame.duplicated().any():
        raise ValueError("canonical figure table contains duplicate rows")
    numeric = {
        "manifest_sequence", "seed", "trial", "value", "n", "mean",
        "sample_std", "median", "ci95_low", "ci95_high", "mean_difference",
        "median_difference", "paired_rank_biserial", "p_value", "holm_p_value",
        "holm_family_size", "point_index", "compression", "utility",
        "evaluation", "hypervolume",
    }.intersection(frame.columns)
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"canonical figure table {column} is not finite numeric")
    integer_columns = {
        "manifest_sequence", "seed", "trial", "n", "holm_family_size",
        "point_index", "evaluation",
    }.intersection(frame.columns)
    for column in integer_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"canonical figure table {column} is not nonnegative integer")
    for column in set(frame.columns) - numeric:
        if frame[column].isna().any():
            raise ValueError(f"canonical figure table {column} contains missing text")
    return frame


def _unique(frame: pd.DataFrame, keys: list[str], context: str) -> None:
    if frame.duplicated(keys, keep=False).any():
        raise ValueError(f"{context} contains duplicate selector rows")


def _prepare_panels(
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    front: pd.DataFrame,
    *,
    run_kind: str,
    require_front: bool = True,
) -> dict[str, pd.DataFrame]:
    expected_n = 5 if run_kind == "formal" else 1
    # Only enforce seed-unit check on stages that are always multi-seed.
    # alpha and ablation are seed-42-only by design; estimator exact may be
    # unavailable on some dataset/seed combinations.
    _MULTI_SEED_ONLY = frozenset({"removal", "nsga2"})
    selected_n = aggregate[
        aggregate["stage"].isin(_MULTI_SEED_ONLY)
    ]
    if not selected_n.empty and not (selected_n["n"] == expected_n).all():
        raise ValueError("figure aggregate rows do not use the required seed units")

    estimator = aggregate[
        (aggregate["stage"] == "estimator")
        & aggregate["metric"].isin(("harmful_f1", "score_agreement"))
    ].copy()
    _unique(estimator, ["dataset_id", "method", "parameter", "metric"], "estimator agreement")
    if not estimator.empty:
        estimator["budget"] = estimator["parameter"].map(
            lambda value: int(re.fullmatch(r"budget=([0-9]+)", str(value)).group(1))
            if re.fullmatch(r"budget=([0-9]+)", str(value)) else -1
        )
        if (estimator["budget"] < 1).any():
            raise ValueError("estimator agreement parameter is invalid")
        counts = estimator.groupby(
            ["dataset_id", "method", "parameter"], sort=True
        )["metric"].nunique()
        if (counts != 2).any():
            if run_kind == "formal":
                raise ValueError("missing formal estimator agreement rows")
            estimator = estimator.iloc[0:0].copy()
    elif run_kind == "formal":
        raise ValueError("missing formal estimator agreement rows")

    ablation = aggregate[
        (aggregate["stage"] == "ablation")
        & (aggregate["metric"] == "harmful_fraction")
    ].copy()
    _unique(
        ablation,
        ["dataset_id", "method", "parameter", "metric"],
        "ablation selector",
    )
    if ablation.empty:
        raise ValueError("missing ablation harmful_fraction selector rows")
    parsed = ablation["parameter"].map(
        lambda parameter: re.fullmatch(r"([^=]+)=([^=]+)", str(parameter))
    )
    if parsed.isna().any():
        raise ValueError("ablation selector parameter is invalid")
    ablation["axis"] = [match.group(1) for match in parsed]
    ablation["value_name"] = [match.group(2) for match in parsed]
    if not (ablation["method"] == ablation["axis"]).all():
        raise ValueError("ablation selector method conflicts with axis")
    allowed = _REQUIRED_ABLATION_SELECTORS | _OPTIONAL_ABLATION_SELECTORS
    for dataset, rows in ablation.groupby("dataset_id", sort=True):
        actual = set(zip(rows["axis"], rows["value_name"], strict=False))
        if not actual <= allowed:
            raise ValueError(f"ablation selector is not canonical for {dataset}")
        if run_kind == "formal" and not _REQUIRED_ABLATION_SELECTORS <= actual:
            raise ValueError(f"missing formal ablation selector rows for {dataset}")
        utility = {value for axis, value in actual if axis == "utility"}
        if utility and utility != {"accuracy", "ovr_auc"}:
            raise ValueError(f"missing paired utility ablation selector rows for {dataset}")

    alpha = aggregate[
        (aggregate["stage"] == "alpha")
        & (aggregate["method"] == "alpha_evaluation")
        & aggregate["metric"].isin(("compression", "test_utility"))
    ].copy()
    _unique(alpha, ["dataset_id", "method", "parameter", "metric"], "alpha operating point")
    compression = alpha[alpha["metric"] == "compression"][
        ["dataset_id", "method", "parameter", "mean"]
    ].rename(columns={"mean": "compression"})
    utility = alpha[alpha["metric"] == "test_utility"][
        ["dataset_id", "method", "parameter", "mean"]
    ].rename(columns={"mean": "test_utility"})
    alpha_points = compression.merge(
        utility,
        on=["dataset_id", "method", "parameter"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if alpha_points.empty or (alpha_points["_merge"] != "both").any():
        raise ValueError("missing paired alpha compression/test_utility rows")
    alpha_points = alpha_points.drop(columns="_merge")
    alpha_points["alpha"] = alpha_points["parameter"].map(
        lambda value: float(re.fullmatch(r"alpha=(.+)", str(value)).group(1))
        if re.fullmatch(r"alpha=(.+)", str(value)) else np.nan
    )
    if not np.isfinite(alpha_points["alpha"]).all():
        raise ValueError("alpha operating point parameter is invalid")

    removal = aggregate[
        (aggregate["stage"] == "removal")
        & (aggregate["metric"] == "delta_utility")
        & aggregate["method"].isin(_REMOVAL_METHODS)
    ].copy()
    _unique(removal, ["dataset_id", "method"], "removal delta_utility matrix")
    datasets = sorted(removal["dataset_id"].astype(str).unique())
    expected_pairs = {(dataset, method) for dataset in datasets for method in _REMOVAL_METHODS}
    actual_pairs = set(zip(removal["dataset_id"], removal["method"], strict=False))
    if not datasets or actual_pairs != expected_pairs:
        raise ValueError("missing removal delta_utility matrix rows")
    matrix = removal.pivot(index="dataset_id", columns="method", values="mean").loc[
        datasets, list(_REMOVAL_METHODS)
    ]
    ranks = matrix.rank(axis=1, ascending=False, method="average")

    nsga = aggregate[
        (aggregate["stage"] == "nsga2")
        & aggregate["method"].isin(_NSGA_METHODS)
        & aggregate["metric"].isin(("test_hv", "convergence_evaluation_count"))
    ].copy()
    _unique(nsga, ["dataset_id", "method", "metric"], "NSGA-II aggregate")
    nsga_datasets = sorted(nsga["dataset_id"].astype(str).unique())
    expected_nsga = {
        (dataset, method, metric)
        for dataset in nsga_datasets
        for method in _NSGA_METHODS
        for metric in ("test_hv", "convergence_evaluation_count")
    }
    actual_nsga = set(zip(nsga["dataset_id"], nsga["method"], nsga["metric"], strict=False))
    if not nsga_datasets or actual_nsga != expected_nsga:
        raise ValueError("missing NSGA-II HV/convergence selector rows")

    test_front = front[
        (front["front_kind"] == "test") & front["method"].isin(_NSGA_METHODS)
    ].copy()
    _unique(
        test_front,
        ["dataset_id", "seed", "method", "front_kind", "point_index"],
        "held-out Pareto front",
    )
    if require_front:
        front_pairs = set(
            zip(test_front["dataset_id"], test_front["method"], strict=False)
        )
        expected_front_pairs = {
            (dataset, method) for dataset in nsga_datasets for method in _NSGA_METHODS
        }
        if front_pairs != expected_front_pairs:
            raise ValueError("missing held-out Pareto front rows")
    if not paired.empty:
        if not paired["holm_family_id"].str.startswith("nsga2:").all():
            raise ValueError("paired comparison family is invalid")
        if not (paired["n"] == expected_n).all():
            raise ValueError("paired comparison seed count is invalid")
    return {
        "estimator": estimator,
        "ablation": ablation,
        "alpha": alpha_points,
        "removal": matrix,
        "ranks": ranks,
        "nsga": nsga,
        "front": test_front,
    }


def _smoke_label(fig: plt.Figure, run_kind: str) -> None:
    if run_kind == "smoke":
        fig.text(
            0.5, 0.988, "SMOKE - NOT FOR PAPER", ha="center", va="top",
            color="#b00020", fontsize=12, fontweight="bold",
        )


def _save(fig: plt.Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, format="pdf", bbox_inches="tight", metadata=_PDF_METADATA)
    plt.close(fig)
    return destination


def _render_fig1(panels: dict[str, pd.DataFrame], output: Path, run_kind: str) -> Path:
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        estimator = panels["estimator"]
        if estimator.empty:
            axes[0].text(0.5, 0.5, "Insufficient estimator agreement rows", ha="center")
        else:
            grouped = estimator.groupby(["method", "budget", "metric"], sort=True)["mean"].mean()
            for (method, metric), series in grouped.groupby(level=(0, 2), sort=True):
                values = series.droplevel((0, 2)).sort_index()
                display_method = "exact reference" if method == "exact" else method
                axes[0].plot(
                    values.index, values.values, marker="o",
                    linestyle="-" if metric == "harmful_f1" else "--",
                    label=f"{display_method} {metric}",
                )
            axes[0].legend(fontsize=5)
        axes[0].set_title("(a) Estimator agreement by budget")
        axes[0].set_xlabel("Budget")
        axes[0].set_ylabel("Agreement score")

        ablation = panels["ablation"].sort_values(
            ["dataset_id", "axis", "value_name"]
        )
        labels = [
            f"{row.dataset_id}\n{row.axis}={row.value_name}"
            for row in ablation.itertuples()
        ]
        axes[1].bar(range(len(ablation)), ablation["mean"], color="#4c78a8")
        axes[1].set_xticks(range(len(ablation)), labels, rotation=45, ha="right")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("(b) Ablation harmful fraction")
        axes[1].set_ylabel("Harmful-feature fraction")

        markers = {"full": "o", "minppi_prefilter": "s"}
        for (dataset, method), group in panels["front"].groupby(
            ["dataset_id", "method"], sort=True
        ):
            axes[2].scatter(
                group["compression"], group["utility"], s=14, alpha=0.65,
                marker=markers[method], label=f"{dataset} {method}",
            )
        for dataset, group in panels["alpha"].groupby("dataset_id", sort=True):
            ordered = group.sort_values("alpha")
            axes[2].plot(
                ordered["compression"], ordered["test_utility"], marker="X",
                linestyle="--", linewidth=0.8, label=f"{dataset} alpha",
            )
        axes[2].legend(fontsize=5)
        axes[2].set_title("(c) Alpha points and held-out Pareto fronts")
        axes[2].set_xlabel("Compression")
        axes[2].set_ylabel("Test utility")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        _smoke_label(fig, run_kind)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, output / "fig1_overview.pdf")


def _render_fig2(panels: dict[str, pd.DataFrame], output: Path, run_kind: str) -> Path:
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        mean_ranks = panels["ranks"].mean(axis=0).sort_values()
        axes[0].barh(range(len(mean_ranks)), mean_ranks.values, color="#72b7b2")
        axes[0].set_yticks(range(len(mean_ranks)), mean_ranks.index)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Mean descending rank (1 = best)")
        axes[0].set_title("(a) Descriptive delta-utility ranks")
        axes[0].grid(True, axis="x", alpha=0.25)

        matrix = panels["removal"]
        image = axes[1].imshow(matrix.to_numpy(), cmap="RdYlGn", aspect="auto")
        axes[1].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        axes[1].set_yticks(range(len(matrix.index)), matrix.index)
        axes[1].set_title("(b) Removal test delta_utility")
        for row_index in range(len(matrix.index)):
            for column_index in range(len(matrix.columns)):
                axes[1].text(
                    column_index, row_index,
                    f"{matrix.iloc[row_index, column_index]:+.3f}",
                    ha="center", va="center", fontsize=5,
                )
        fig.colorbar(image, ax=axes[1], shrink=0.8, label="delta_utility")
        _smoke_label(fig, run_kind)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, output / "fig2_cd_heatmap.pdf")


def _render_fig3(panels: dict[str, pd.DataFrame], output: Path, run_kind: str) -> Path:
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        nsga = panels["nsga"]
        datasets = sorted(nsga["dataset_id"].astype(str).unique())
        x = np.arange(len(datasets), dtype=float)
        width = 0.36
        for axis, metric, title in (
            (axes[0], "test_hv", "(a) Held-out hypervolume"),
            (axes[1], "convergence_evaluation_count", "(b) Convergence evaluations"),
        ):
            for offset, method, color in (
                (-width / 2, "full", "#999999"),
                (width / 2, "minppi_prefilter", "#54a24b"),
            ):
                rows = nsga[(nsga["metric"] == metric) & (nsga["method"] == method)].set_index("dataset_id").loc[datasets]
                errors = rows["sample_std"].to_numpy() if run_kind == "formal" else None
                axis.bar(
                    x + offset, rows["mean"].to_numpy(), width, yerr=errors,
                    capsize=2 if errors is not None else 0, label=method, color=color,
                )
            axis.set_xticks(x, datasets, rotation=35, ha="right")
            axis.set_title(title)
            axis.grid(True, axis="y", alpha=0.25)
            axis.legend(fontsize=6)
        _smoke_label(fig, run_kind)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, output / "fig3_prefilter.pdf")


def make_fig1_overview(
    aggregate_table_path: Path, front_table_path: Path, output_directory: Path, *, run_kind: str
) -> Path:
    aggregate = _read_table(aggregate_table_path, _EXPECTED_AGGREGATE, run_kind)
    paired = pd.DataFrame(columns=_EXPECTED_PAIRED)
    front = _read_table(front_table_path, _EXPECTED_FRONT, run_kind)
    return _render_fig1(
        _prepare_panels(aggregate, paired, front, run_kind=run_kind),
        Path(output_directory), run_kind,
    )


def make_fig2_cd_heatmap(
    aggregate_table_path: Path, output_directory: Path, *, run_kind: str
) -> Path:
    aggregate = _read_table(aggregate_table_path, _EXPECTED_AGGREGATE, run_kind)
    empty_front = pd.DataFrame(columns=_EXPECTED_FRONT)
    panels = _prepare_panels(
        aggregate,
        pd.DataFrame(columns=_EXPECTED_PAIRED),
        empty_front,
        run_kind=run_kind,
        require_front=False,
    )
    return _render_fig2(panels, Path(output_directory), run_kind)


def make_fig3_prefilter(
    aggregate_table_path: Path,
    comparison_table_path: Path,
    output_directory: Path,
    *,
    run_kind: str,
) -> Path:
    aggregate = _read_table(aggregate_table_path, _EXPECTED_AGGREGATE, run_kind)
    paired = _read_table(comparison_table_path, _EXPECTED_PAIRED, run_kind)
    empty_front = pd.DataFrame(columns=_EXPECTED_FRONT)
    panels = _prepare_panels(
        aggregate, paired, empty_front, run_kind=run_kind, require_front=False
    )
    return _render_fig3(panels, Path(output_directory), run_kind)


def generate_all_figures(
    raw_table_path: Path,
    aggregate_table_path: Path,
    comparison_table_path: Path,
    front_table_path: Path,
    output_directory: Path,
    *,
    run_kind: str,
) -> tuple[Path, Path, Path]:
    """Preflight all semantics, then generate the three exact figure roles."""
    if run_kind not in {"formal", "smoke"}:
        raise ValueError("run_kind must be formal or smoke")
    _read_table(raw_table_path, _EXPECTED_RAW, run_kind)
    aggregate = _read_table(aggregate_table_path, _EXPECTED_AGGREGATE, run_kind)
    paired = _read_table(comparison_table_path, _EXPECTED_PAIRED, run_kind)
    front = _read_table(front_table_path, _EXPECTED_FRONT, run_kind)
    panels = _prepare_panels(aggregate, paired, front, run_kind=run_kind)
    output = Path(output_directory)
    return (
        _render_fig1(panels, output, run_kind),
        _render_fig2(panels, output, run_kind),
        _render_fig3(panels, output, run_kind),
    )


def _parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-table", required=True, type=Path)
    parser.add_argument("--aggregate-table", required=True, type=Path)
    parser.add_argument("--comparison-table", required=True, type=Path)
    parser.add_argument("--front-table", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--run-kind", required=True, choices=("formal", "smoke"))
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    generate_all_figures(
        arguments.raw_table,
        arguments.aggregate_table,
        arguments.comparison_table,
        arguments.front_table,
        arguments.output_directory,
        run_kind=arguments.run_kind,
    )
