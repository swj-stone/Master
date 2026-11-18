"""Read-only deterministic paper figures from canonical campaign tables."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


_PDF_METADATA = {
    "Creator": "minppi-mofs ICOMP 2026",
    "Producer": "matplotlib",
    "CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
}
_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.compression": 6,
}
_EXPECTED_RAW = (
    "run_kind",
    "manifest_sequence",
    "stage",
    "dataset_id",
    "seed",
    "method",
    "trial",
    "parameter",
    "metric",
    "value",
)
_EXPECTED_AGGREGATE = (
    "run_kind",
    "stage",
    "dataset_id",
    "method",
    "parameter",
    "metric",
    "n",
    "mean",
    "sample_std",
    "median",
    "ci95_low",
    "ci95_high",
    "analysis_label",
)
_EXPECTED_PAIRED = (
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
)
_EXPECTED_FRONT = (
    "run_kind",
    "dataset_id",
    "seed",
    "method",
    "front_kind",
    "point_index",
    "compression",
    "utility",
    "evaluation",
    "hypervolume",
)
_REMOVAL_METHODS = (
    "minppi",
    "shap_bottom",
    "minshap_bottom",
    "banzhaf_bottom",
    "random_matched",
)
_RANK_METHODS = ("full", *_REMOVAL_METHODS)
_NSGA_METHODS = ("full", "minppi_prefilter")
_NSGA_METRICS = (
    "test_hv",
    "convergence_evaluation_count",
)
_CANONICAL_DATASETS = (
    "abalone",
    "credit_card_fraud",
    "gisette",
    "isolet",
    "madelon",
    "spambase",
)
_DISPLAY_DATASETS = {
    "abalone": "Abalone",
    "credit_card_fraud": "Credit-Card-Fraud",
    "crimedata": "Communities & Crime",
    "gisette": "Gisette",
    "isolet": "ISOLET",
    "madelon": "Madelon",
    "spambase": "Spambase",
}
_DISPLAY_METHODS = {
    "full": "Full",
    "minppi": "MinPPI",
    "minppi_prefilter": "MinPPI-prefiltered",
    "shap_bottom": "SHAP-bottom",
    "minshap_bottom": "MinShap-bottom",
    "banzhaf_bottom": "Banzhaf-bottom",
    "random_matched": "Random-matched",
}
_OVERVIEW_FORMAL_DATASETS = ("madelon", "credit_card_fraud")
_OVERVIEW_ALPHAS = (0.85, 0.90, 0.95, 0.99)
_EXAMPLE2_WINNING = frozenset(
    {
        frozenset({0, 2}),
        frozenset({0, 3}),
        frozenset({0, 4}),
        frozenset({0, 2, 3}),
        frozenset({0, 2, 4}),
        frozenset({0, 3, 4}),
        frozenset({0, 2, 3, 4}),
        frozenset({0, 1, 3}),
        frozenset({0, 1, 4}),
        frozenset({0, 1, 3, 4}),
        frozenset({0, 1, 2, 3}),
        frozenset({0, 1, 2, 4}),
        frozenset({0, 1, 2, 3, 4}),
    }
)
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
        "manifest_sequence",
        "seed",
        "trial",
        "value",
        "n",
        "mean",
        "sample_std",
        "median",
        "ci95_low",
        "ci95_high",
        "mean_difference",
        "median_difference",
        "paired_rank_biserial",
        "p_value",
        "holm_p_value",
        "holm_family_size",
        "point_index",
        "compression",
        "utility",
        "evaluation",
        "hypervolume",
    }.intersection(frame.columns)
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"canonical figure table {column} is not finite numeric")
    integer_columns = {
        "manifest_sequence",
        "seed",
        "trial",
        "n",
        "holm_family_size",
        "point_index",
        "evaluation",
    }.intersection(frame.columns)
    for column in integer_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(
                f"canonical figure table {column} is not nonnegative integer"
            )
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
    raw: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    expected_n = 5 if run_kind == "formal" else 1
    # Only enforce seed-unit check on stages that are always multi-seed.
    # alpha and ablation are seed-42-only by design; estimator exact may be
    # unavailable on some dataset/seed combinations.
    _MULTI_SEED_ONLY = frozenset({"removal", "nsga2"})
    selected_n = aggregate[aggregate["stage"].isin(_MULTI_SEED_ONLY)]
    if not selected_n.empty and not (selected_n["n"] == expected_n).all():
        raise ValueError("figure aggregate rows do not use the required seed units")

    estimator = aggregate[
        (aggregate["stage"] == "estimator")
        & aggregate["metric"].isin(("harmful_f1", "score_agreement"))
    ].copy()
    _unique(
        estimator,
        ["dataset_id", "method", "parameter", "metric"],
        "estimator agreement",
    )
    if not estimator.empty:
        estimator["budget"] = estimator["parameter"].map(
            lambda value: (
                int(re.fullmatch(r"budget=([0-9]+)", str(value)).group(1))
                if re.fullmatch(r"budget=([0-9]+)", str(value))
                else -1
            )
        )
        if (estimator["budget"] < 1).any():
            raise ValueError("estimator agreement parameter is invalid")
        counts = estimator.groupby(["dataset_id", "method", "parameter"], sort=True)[
            "metric"
        ].nunique()
        if (counts != 2).any():
            if run_kind == "formal":
                raise ValueError("missing formal estimator agreement rows")
            estimator = estimator.iloc[0:0].copy()
    elif run_kind == "formal":
        raise ValueError("missing formal estimator agreement rows")

    ablation = aggregate[
        (aggregate["stage"] == "ablation") & (aggregate["metric"] == "harmful_fraction")
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
            raise ValueError(
                f"missing paired utility ablation selector rows for {dataset}"
            )

    alpha = aggregate[
        (aggregate["stage"] == "alpha")
        & (aggregate["method"] == "alpha_evaluation")
        & aggregate["metric"].isin(("compression", "test_utility"))
    ].copy()
    _unique(
        alpha, ["dataset_id", "method", "parameter", "metric"], "alpha operating point"
    )
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
        lambda value: (
            float(re.fullmatch(r"alpha=(.+)", str(value)).group(1))
            if re.fullmatch(r"alpha=(.+)", str(value))
            else np.nan
        )
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
    expected_pairs = {
        (dataset, method) for dataset in datasets for method in _REMOVAL_METHODS
    }
    actual_pairs = set(zip(removal["dataset_id"], removal["method"], strict=False))
    if not datasets or actual_pairs != expected_pairs:
        raise ValueError("missing removal delta_utility matrix rows")
    matrix = removal.pivot(index="dataset_id", columns="method", values="mean").loc[
        datasets, list(_REMOVAL_METHODS)
    ]
    rank_rows = aggregate[
        (aggregate["stage"] == "removal")
        & (aggregate["metric"] == "delta_utility")
        & aggregate["method"].isin(_RANK_METHODS)
    ].copy()
    _unique(rank_rows, ["dataset_id", "method"], "removal rank matrix")
    expected_rank_pairs = {
        (dataset, method) for dataset in datasets for method in _RANK_METHODS
    }
    actual_rank_pairs = set(
        zip(rank_rows["dataset_id"], rank_rows["method"], strict=False)
    )
    if actual_rank_pairs != expected_rank_pairs:
        raise ValueError("missing six-method removal rank rows")
    ranks = (
        rank_rows.pivot(index="dataset_id", columns="method", values="mean")
        .loc[datasets, list(_RANK_METHODS)]
        .rank(axis=1, ascending=False, method="average")
        .loc[:, list(_REMOVAL_METHODS)]
    )

    nsga = aggregate[
        (aggregate["stage"] == "nsga2")
        & aggregate["method"].isin(_NSGA_METHODS)
        & aggregate["metric"].isin(_NSGA_METRICS)
    ].copy()
    if not (nsga["parameter"] == "").all():
        raise ValueError("NSGA-II aggregate parameter must be empty")
    _unique(nsga, ["dataset_id", "method", "metric"], "NSGA-II aggregate")
    nsga_datasets = sorted(nsga["dataset_id"].astype(str).unique())
    if run_kind == "formal" and tuple(nsga_datasets) != _CANONICAL_DATASETS:
        raise ValueError(
            "formal NSGA-II figures require the exact six canonical datasets"
        )
    expected_nsga = {
        (dataset, method, metric)
        for dataset in nsga_datasets
        for method in _NSGA_METHODS
        for metric in _NSGA_METRICS
    }
    actual_nsga = set(
        zip(nsga["dataset_id"], nsga["method"], nsga["metric"], strict=False)
    )
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
    front_history = front[
        (front["front_kind"] == "history") & front["method"].isin(_NSGA_METHODS)
    ].copy()
    _unique(
        front_history,
        ["dataset_id", "seed", "method", "point_index"],
        "history front",
    )
    if not paired.empty:
        if not paired["holm_family_id"].str.startswith("nsga2:").all():
            raise ValueError("paired comparison family is invalid")
        if not (paired["n"] == expected_n).all():
            raise ValueError("paired comparison seed count is invalid")
    return {
        "raw": (raw.copy() if raw is not None else pd.DataFrame(columns=_EXPECTED_RAW)),
        "estimator": estimator,
        "ablation": ablation,
        "alpha": alpha_points,
        "alpha_aggregate_rows": alpha,
        "removal": matrix,
        "ranks": ranks,
        "nsga": nsga,
        "front": test_front,
        "front_history": front_history,
    }


def _smoke_label(fig: plt.Figure, run_kind: str) -> None:
    if run_kind == "smoke":
        fig.text(
            0.5,
            0.988,
            "SMOKE - NOT FOR PAPER",
            ha="center",
            va="top",
            color="#b00020",
            fontsize=12,
            fontweight="bold",
        )


def _save(fig: plt.Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, format="pdf", bbox_inches="tight", metadata=_PDF_METADATA)
    plt.close(fig)
    return destination


def _example2_minss() -> np.ndarray:
    """Return exhaustive MinSS values for the five-feature constructed game."""
    values: list[int] = []
    players = tuple(range(5))
    for feature in players:
        other = tuple(player for player in players if player != feature)
        marginals = [
            int((frozenset(coalition) | {feature}) in _EXAMPLE2_WINNING)
            - int(frozenset(coalition) in _EXAMPLE2_WINNING)
            for size in range(len(other) + 1)
            for coalition in combinations(other, size)
        ]
        values.append(min(marginals))
    return np.asarray(values, dtype=int)


def _example2_detection_curves() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, float, float
]:
    """Return exact conditional curves for the two registered search kernels."""
    target = 1
    other = tuple(player for player in range(5) if player != target)
    negative_probability = sum(
        sum(
            int(
                int((frozenset(coalition) | {target}) in _EXAMPLE2_WINNING)
                - int(frozenset(coalition) in _EXAMPLE2_WINNING)
                == -1
            )
            for coalition in combinations(other, size)
        )
        / len(tuple(combinations(other, size)))
        for size in range(len(other) + 1)
    ) / (len(other) + 1)
    guided_seed = frozenset({0, 2})
    guided_probability = 0.0
    for keep_zero in (False, True):
        for keep_two in (False, True):
            coalition = {
                feature for feature, keep in ((0, keep_zero), (2, keep_two)) if keep
            }
            removal_probability = (0.7 if keep_zero else 0.3) * (
                0.7 if keep_two else 0.3
            )
            for add_count in range(len(other) + 1):
                for add_order in permutations(other):
                    outside = [
                        feature for feature in add_order if feature not in coalition
                    ]
                    perturbed = coalition | set(outside[:add_count])
                    if frozenset(perturbed) == guided_seed:
                        guided_probability += (
                            removal_probability
                            / (len(other) + 1)
                            / len(tuple(permutations(other)))
                        )
    budgets = np.asarray((1, 2, 3, 5, 10, 20, 50, 100), dtype=int)
    random_mc = 1.0 - (1.0 - negative_probability) ** budgets
    guided = 1.0 - (1.0 - guided_probability) ** budgets
    return (
        budgets,
        random_mc,
        guided,
        float(negative_probability),
        float(guided_probability),
    )


def _canonical_overview_alpha(value: object) -> float:
    match = re.fullmatch(r"alpha=(.+)", str(value))
    if match is None:
        raise ValueError("formal overview alpha parameter is invalid")
    try:
        parsed = float(match.group(1))
    except ValueError as exc:
        raise ValueError("formal overview alpha parameter is invalid") from exc
    matches = [
        candidate
        for candidate in _OVERVIEW_ALPHAS
        if np.isclose(parsed, candidate, rtol=0.0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError("formal overview alpha grid is not canonical")
    return matches[0]


def _formal_overview_alpha(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = panels["raw"]
    raw_alpha = raw[
        (raw["stage"] == "alpha")
        & raw["dataset_id"].isin(_OVERVIEW_FORMAL_DATASETS)
        & (raw["method"] == "alpha_evaluation")
        & raw["metric"].isin(("compression", "test_utility"))
    ].copy()
    if (
        raw_alpha.empty
        or not (raw_alpha["seed"] == 42).all()
        or not (raw_alpha["trial"] == 0).all()
    ):
        raise ValueError(
            "formal overview alpha authority must use only seed 42 and trial 0"
        )
    raw_alpha["alpha"] = raw_alpha["parameter"].map(_canonical_overview_alpha)
    _unique(
        raw_alpha,
        ["dataset_id", "seed", "method", "alpha", "metric"],
        "formal overview raw alpha",
    )
    expected = {
        (dataset, alpha, metric)
        for dataset in _OVERVIEW_FORMAL_DATASETS
        for alpha in _OVERVIEW_ALPHAS
        for metric in ("compression", "test_utility")
    }
    actual = set(
        zip(
            raw_alpha["dataset_id"],
            raw_alpha["alpha"],
            raw_alpha["metric"],
            strict=False,
        )
    )
    if actual != expected:
        raise ValueError(
            "formal overview requires the complete seed-42 alpha grid for "
            "Madelon and Credit-Card-Fraud"
        )
    points = (
        raw_alpha.pivot(index=["dataset_id", "alpha"], columns="metric", values="value")
        .reset_index()
        .rename_axis(columns=None)
    )
    aggregate_rows = panels["alpha_aggregate_rows"]
    aggregate_rows = aggregate_rows[
        aggregate_rows["dataset_id"].isin(_OVERVIEW_FORMAL_DATASETS)
    ].copy()
    if aggregate_rows.empty or not (aggregate_rows["n"] == 1).all():
        raise ValueError("formal overview aggregate alpha rows must have n=1")
    aggregate_rows["alpha"] = aggregate_rows["parameter"].map(_canonical_overview_alpha)
    aggregate_actual = set(
        zip(
            aggregate_rows["dataset_id"],
            aggregate_rows["alpha"],
            aggregate_rows["metric"],
            strict=False,
        )
    )
    if aggregate_actual != expected:
        raise ValueError("formal overview aggregate alpha grid is incomplete")
    aggregate_points = (
        aggregate_rows.pivot(
            index=["dataset_id", "alpha"], columns="metric", values="mean"
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    comparison = points.merge(
        aggregate_points,
        on=["dataset_id", "alpha"],
        suffixes=("_raw", "_aggregate"),
        validate="one_to_one",
    )
    for metric in ("compression", "test_utility"):
        if not np.allclose(
            comparison[f"{metric}_raw"],
            comparison[f"{metric}_aggregate"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("formal overview raw and aggregate alpha values disagree")
    return points


def _overview_panel_data(
    panels: dict[str, pd.DataFrame], run_kind: str
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Select the publication overview's named datasets and seed-42 fronts."""
    front = panels["front"]
    seed42 = front[front["seed"] == 42].copy()
    if run_kind == "formal":
        datasets = _OVERVIEW_FORMAL_DATASETS
        alpha = _formal_overview_alpha(panels)
        selected_front = seed42[seed42["dataset_id"].isin(datasets)].copy()
        front_pairs = set(
            zip(selected_front["dataset_id"], selected_front["method"], strict=False)
        )
        expected_pairs = {
            (dataset, method) for dataset in datasets for method in _NSGA_METHODS
        }
        if front_pairs != expected_pairs:
            raise ValueError(
                "seed-42 overview fronts must include both registered methods for "
                "Madelon and Credit-Card-Fraud"
            )
        for _, rows in selected_front.groupby(["dataset_id", "method"], sort=True):
            indices = sorted(rows["point_index"].astype(int))
            if indices != list(range(len(indices))):
                raise ValueError(
                    "seed-42 overview front point indices must be contiguous from zero"
                )
        bounded = (
            selected_front["compression"].between(0.0, 1.0).all()
            and selected_front["utility"].between(0.0, 1.0).all()
            and alpha["compression"].between(0.0, 1.0).all()
            and alpha["test_utility"].between(0.0, 1.0).all()
        )
        if not bounded:
            raise ValueError("formal overview panel values must lie in [0, 1]")
    else:
        datasets = ()
        alpha = pd.DataFrame(
            columns=("dataset_id", "alpha", "compression", "test_utility")
        )
        selected_front = seed42.iloc[0:0].copy()
    return datasets, alpha, selected_front


def _render_fig1(panels: dict[str, pd.DataFrame], output: Path, run_kind: str) -> Path:
    with plt.rc_context(_RC):
        datasets, alpha, front = _overview_panel_data(panels, run_kind)
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(7.5, 4.7),
            gridspec_kw={"height_ratios": (3.2, 1.25)},
        )

        minss = _example2_minss()
        feature_positions = np.arange(len(minss))
        feature_colors = ["#d62728" if value == -1 else "#9e9e9e" for value in minss]
        axes[0, 0].bar(
            feature_positions,
            minss,
            color=feature_colors,
            edgecolor="#333333",
            linewidth=0.5,
        )
        neutral_positions = feature_positions[minss == 0]
        axes[0, 0].scatter(
            neutral_positions,
            np.zeros_like(neutral_positions),
            s=24,
            marker="o",
            color="#9e9e9e",
            edgecolors="#333333",
            linewidths=0.5,
            zorder=3,
        )
        axes[0, 0].set_xticks(feature_positions, [f"f{i}" for i in feature_positions])
        axes[0, 0].set_yticks((-1, 0, 1))
        axes[0, 0].set_ylim(-1.25, 1.20)
        axes[0, 0].axhline(0.0, color="#555555", linewidth=0.7)
        axes[0, 0].set_ylabel("MinSS")
        axes[0, 0].set_title("(a) MinSS classification\n(constructed game)")

        budgets, random_mc, guided, negative_probability, guided_probability = (
            _example2_detection_curves()
        )
        line_a = axes[0, 1].plot(
            budgets,
            random_mc,
            "o-",
            color="#d62728",
            linewidth=1.3,
            markersize=4,
            label="Algorithm A",
        )[0]
        line_b = axes[0, 1].plot(
            budgets,
            guided,
            "s-",
            color="#2ca02c",
            linewidth=1.3,
            markersize=4,
            label="Algorithm B",
        )[0]
        axes[0, 1].set_xscale("log")
        displayed_budgets = (1, 2, 5, 10, 20, 100)
        axes[0, 1].set_xticks(
            displayed_budgets, [str(value) for value in displayed_budgets]
        )
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].set_xlabel("Budget K")
        axes[0, 1].set_ylabel("Harmful-feature detection rate")
        axes[0, 1].set_title("(b) Conditional detection\nvs budget K")

        method_style = {
            "full": ("#999999", "full"),
            "minppi_prefilter": ("#4c78a8", "filtered"),
        }
        dataset_markers = ("o", "s")
        legend_handles: list[object] = []
        legend_labels: list[str] = []
        code_index = 0
        for dataset_index, dataset in enumerate(datasets):
            marker = dataset_markers[dataset_index]
            for method in _NSGA_METHODS:
                color, method_name = method_style[method]
                rows = front[
                    (front["dataset_id"] == dataset) & (front["method"] == method)
                ].sort_values("point_index")
                artist = axes[0, 2].scatter(
                    rows["compression"],
                    rows["utility"],
                    s=22,
                    alpha=0.75,
                    marker=marker,
                    color=color,
                    edgecolors="white",
                    linewidths=0.35,
                )
                code = chr(ord("A") + code_index)
                legend_handles.append(artist)
                legend_labels.append(f"{code}  {method_name}")
                code_index += 1
            alpha_rows = alpha[alpha["dataset_id"] == dataset].sort_values("alpha")
            alpha_line = axes[0, 2].plot(
                alpha_rows["compression"],
                alpha_rows["test_utility"],
                marker="X" if dataset_index == 0 else "P",
                linestyle="--",
                linewidth=1.0,
                markersize=5,
                color="#f28e2b",
            )[0]
            code = chr(ord("A") + code_index)
            legend_handles.append(alpha_line)
            legend_labels.append(f"{code}  alpha")
            code_index += 1
        if not datasets:
            axes[0, 2].text(
                0.5,
                0.5,
                "Madelon and Credit-Card-Fraud\nare outside smoke scope",
                transform=axes[0, 2].transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
            legend_handles.append(Patch(facecolor="none", edgecolor="none"))
            legend_labels.append("Formal seed-42 panel generated downstream")
        axes[0, 2].set_title(
            "(c) Held-out evaluations of\nvalidation-selected fronts"
        )
        axes[0, 2].set_xlabel("Compression")
        axes[0, 2].set_ylabel("Held-out utility")

        for axis in axes[0]:
            axis.grid(True, alpha=0.25)
        for axis in axes[1]:
            axis.axis("off")
        axes[1, 0].legend(
            handles=(
                Patch(facecolor="#d62728", edgecolor="#333333"),
                Patch(facecolor="#9e9e9e", edgecolor="#333333"),
            ),
            labels=("Harmful: MinSS = -1", "Neutral: MinSS = 0"),
            loc="upper center",
            frameon=False,
            fontsize=8,
        )
        axes[1, 1].legend(
            handles=(line_a, line_b),
            labels=(
                f"Algorithm A: random MC\np- = {negative_probability:.4f}",
                "Algorithm B: conditional oracle guidance\n"
                f"rho = {guided_probability:.3f}",
            ),
            loc="upper center",
            frameon=False,
            fontsize=8,
        )
        if datasets:
            dataset_codebook = "A-C: Madelon     D-F: Credit-Card-Fraud"
            legend_order = (0, 3, 1, 4, 2, 5)
            legend_handles = [legend_handles[index] for index in legend_order]
            legend_labels = [legend_labels[index] for index in legend_order]
        else:
            dataset_codebook = "Formal targets: Madelon and Credit-Card-Fraud"
        axes[1, 2].text(
            0.5,
            1.0,
            dataset_codebook,
            transform=axes[1, 2].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        axes[1, 2].legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.78),
            ncol=3 if datasets else 1,
            frameon=False,
            fontsize=8,
            columnspacing=0.8,
            handletextpad=0.5,
        )
        _smoke_label(fig, run_kind)
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.8)
        return _save(fig, output / "fig1_overview.pdf")


def _render_fig2(panels: dict[str, pd.DataFrame], output: Path, run_kind: str) -> Path:
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.5, 4.0),
            gridspec_kw={"width_ratios": (0.85, 1.15)},
        )
        mean_ranks = panels["ranks"].mean(axis=0).sort_values()
        axes[0].barh(range(len(mean_ranks)), mean_ranks.values, color="#72b7b2")
        axes[0].set_yticks(
            range(len(mean_ranks)),
            [_DISPLAY_METHODS.get(str(method), str(method)) for method in mean_ranks.index],
        )
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Mean descending rank (1 = best)")
        axes[0].set_title("(a) Descriptive delta-utility ranks")
        axes[0].grid(True, axis="x", alpha=0.25)

        matrix = panels["removal"]
        image = axes[1].imshow(matrix.to_numpy(), cmap="RdYlGn", aspect="auto")
        axes[1].set_xticks(
            range(len(matrix.columns)),
            [
                _DISPLAY_METHODS.get(str(method), str(method))
                for method in matrix.columns
            ],
            rotation=35,
            ha="right",
        )
        axes[1].set_yticks(
            range(len(matrix.index)),
            [
                _DISPLAY_DATASETS.get(str(dataset), str(dataset))
                for dataset in matrix.index
            ],
        )
        axes[1].set_title("(b) Removal held-out ΔU")
        for row_index in range(len(matrix.index)):
            for column_index in range(len(matrix.columns)):
                value = matrix.iloc[row_index, column_index]
                label = f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")
                axes[1].text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axes[1], shrink=0.8, label="Held-out ΔU")
        _smoke_label(fig, run_kind)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, output / "fig2_cd_heatmap.pdf")


def _render_fig3(panels: dict[str, pd.DataFrame], output: Path, run_kind: str) -> Path:
    with plt.rc_context(_RC):
        history = panels["front_history"]
        datasets = sorted(history["dataset_id"].astype(str).unique())
        if not datasets:
            raise ValueError("prefilter figure requires hypervolume history rows")
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(7.5, 5.0),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"hspace": 0.1, "wspace": 0.1},
        )
        for axis, dataset in zip(axes.ravel(), datasets):
            for method, color, label in (
                ("full", "#4c78a8", "Full"),
                ("minppi_prefilter", "#54a24b", "MinPPI-prefiltered"),
            ):
                rows = history[
                    (history["dataset_id"] == dataset) & (history["method"] == method)
                ]
                agg = rows.groupby("evaluation")["hypervolume"].agg(["mean", "std"])
                axis.plot(
                    agg.index,
                    agg["mean"],
                    color=color,
                    linewidth=1.6,
                    label=label,
                )
                if run_kind == "formal":
                    axis.fill_between(
                        agg.index,
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        color=color,
                        alpha=0.15,
                    )
            axis.set_title(_DISPLAY_DATASETS.get(dataset, dataset))
            axis.set_box_aspect(1)
            axis.grid(True, alpha=0.25)
        for axis in axes.ravel()[len(datasets):]:
            axis.axis("off")
        axes[1, 0].set_xlabel("Evaluation count")
        fig.supylabel("Cumulative-best hypervolume")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            fontsize=8,
            frameon=False,
        )
        _smoke_label(fig, run_kind)
        return _save(fig, output / "fig3_prefilter.pdf")


def make_fig1_overview(
    aggregate_table_path: Path,
    front_table_path: Path,
    output_directory: Path,
    *,
    run_kind: str,
    raw_table_path: Path | None = None,
) -> Path:
    """Generate the overview; formal output additionally requires raw seed authority."""
    aggregate = _read_table(aggregate_table_path, _EXPECTED_AGGREGATE, run_kind)
    paired = pd.DataFrame(columns=_EXPECTED_PAIRED)
    front = _read_table(front_table_path, _EXPECTED_FRONT, run_kind)
    raw = (
        _read_table(raw_table_path, _EXPECTED_RAW, run_kind)
        if raw_table_path is not None
        else None
    )
    return _render_fig1(
        _prepare_panels(aggregate, paired, front, run_kind=run_kind, raw=raw),
        Path(output_directory),
        run_kind,
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
    front_table_path: Path,
    output_directory: Path,
    *,
    run_kind: str,
) -> Path:
    aggregate = _read_table(aggregate_table_path, _EXPECTED_AGGREGATE, run_kind)
    paired = _read_table(comparison_table_path, _EXPECTED_PAIRED, run_kind)
    front = _read_table(front_table_path, _EXPECTED_FRONT, run_kind)
    panels = _prepare_panels(
        aggregate, paired, front, run_kind=run_kind, require_front=False
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
    raw = _read_table(raw_table_path, _EXPECTED_RAW, run_kind)
    aggregate = _read_table(aggregate_table_path, _EXPECTED_AGGREGATE, run_kind)
    paired = _read_table(comparison_table_path, _EXPECTED_PAIRED, run_kind)
    front = _read_table(front_table_path, _EXPECTED_FRONT, run_kind)
    panels = _prepare_panels(aggregate, paired, front, run_kind=run_kind, raw=raw)
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
