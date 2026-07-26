"""Pure, deterministic statistics and publication tables for ICOMP 2026.

This module performs no filesystem access at import time.  Stage completeness,
canonical artifact validation, and publication are owned by ``icomp2026_stages``.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
FORMAL_SEEDS = (42, 123, 456, 789, 1024)
ESTIMATOR_BASE_METRIC_FIELDS = frozenset(
    {"budget", "model_fit_count", "coalition_count", "runtime_s"}
)
ESTIMATOR_AGREEMENT_METRIC_FIELDS = frozenset(
    {"harmful_precision", "harmful_recall", "harmful_f1", "score_agreement"}
)
GROUPED_ESTIMATOR_METRIC_FIELDS = frozenset(
    {
        "logical_evaluation_count",
        "logical_unique_coalition_count",
        "physical_model_fit_count",
        "cache_hit_count",
        "group_request_digest",
        "group_physical_model_fit_count",
        "group_cache_hit_count",
        "group_runtime_s",
    }
)
ESTIMATOR_GROUP_METRIC_FIELDS = (
    "group_request_digest",
    "group_physical_model_fit_count",
    "group_cache_hit_count",
    "group_runtime_s",
)

RAW_COLUMNS = (
    "run_kind", "manifest_sequence", "stage", "dataset_id", "seed", "method",
    "trial", "parameter", "metric", "value",
)
AGGREGATE_COLUMNS = (
    "run_kind", "stage", "dataset_id", "method", "parameter", "metric", "n",
    "mean", "sample_std", "median", "ci95_low", "ci95_high", "analysis_label",
)
PAIRED_COLUMNS = (
    "run_kind", "stage", "dataset_id", "metric", "baseline_method",
    "comparison_method", "n", "mean_difference", "median_difference",
    "paired_rank_biserial", "p_value", "holm_p_value", "holm_family_id",
    "holm_family_size", "test", "analysis_label",
)
FRONT_COLUMNS = (
    "run_kind", "dataset_id", "seed", "method", "front_kind", "point_index",
    "compression", "utility", "evaluation", "hypervolume",
)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def validate_estimator_reporting_schema(
    metrics: Mapping[str, object],
    *,
    execution_mode: str,
    exact_scores_available: bool,
) -> dict[str, object]:
    """Validate the mode-aware estimator metric fields consumed by reporting."""
    if execution_mode not in {"legacy", "grouped_nested_v1"}:
        raise ValueError("estimator reporting execution mode is invalid")
    if type(exact_scores_available) is not bool:
        raise TypeError("exact_scores_available must be a bool")
    required = set(ESTIMATOR_BASE_METRIC_FIELDS)
    if execution_mode == "grouped_nested_v1":
        required.update(GROUPED_ESTIMATOR_METRIC_FIELDS)
    allowed_schemas = {frozenset(required)}
    if execution_mode == "grouped_nested_v1":
        allowed_schemas.add(frozenset(required | {"schedule_digest"}))
    if exact_scores_available:
        allowed_schemas.add(frozenset(required | ESTIMATOR_AGREEMENT_METRIC_FIELDS))
        if execution_mode == "grouped_nested_v1":
            allowed_schemas.add(
                frozenset(
                    required
                    | ESTIMATOR_AGREEMENT_METRIC_FIELDS
                    | {"schedule_digest"}
                )
            )
    if frozenset(metrics) not in allowed_schemas:
        raise ValueError("estimator ResultRecord metric schema is invalid")
    return dict(metrics)


def deduplicate_estimator_group_metrics(
    projected_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return unique grouped physical accounting and fail on digest conflicts."""
    if isinstance(projected_metrics, (str, bytes)):
        raise TypeError("projected estimator metrics must be a sequence")
    groups_by_digest: dict[str, dict[str, object]] = {}
    for metrics in projected_metrics:
        if not isinstance(metrics, Mapping) or any(
            field not in metrics for field in ESTIMATOR_GROUP_METRIC_FIELDS
        ):
            raise ValueError("grouped estimator accounting schema is invalid")
        digest = metrics["group_request_digest"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("grouped estimator request digest is invalid")
        counts: dict[str, int] = {}
        for field in (
            "group_physical_model_fit_count",
            "group_cache_hit_count",
        ):
            value = metrics[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"grouped estimator {field} is invalid")
            counts[field] = value
        runtime = _finite_number(metrics["group_runtime_s"], "group runtime")
        if runtime < 0:
            raise ValueError("grouped estimator group_runtime_s is invalid")
        group = {
            "group_request_digest": digest,
            **counts,
            "group_runtime_s": runtime,
        }
        previous = groups_by_digest.get(digest)
        if previous is not None and previous != group:
            raise ValueError(
                "conflicting grouped estimator accounting for request digest"
            )
        groups_by_digest.setdefault(digest, group)
    groups = list(groups_by_digest.values())
    return {
        "groups": groups,
        "totals": {
            "deduplicated_group_count": len(groups),
            "deduplicated_group_physical_model_fit_count": sum(
                int(group["group_physical_model_fit_count"]) for group in groups
            ),
            "deduplicated_group_cache_hit_count": sum(
                int(group["group_cache_hit_count"]) for group in groups
            ),
            "deduplicated_group_runtime_s": sum(
                float(group["group_runtime_s"]) for group in groups
            ),
        },
    }


def estimator_projection_reporting_metrics(
    metrics: Mapping[str, object],
    *,
    execution_mode: str,
) -> dict[str, object]:
    """Keep per-projection metrics separate from complete grouped accounting."""
    agreement_fields = ESTIMATOR_AGREEMENT_METRIC_FIELDS & frozenset(metrics)
    validated = validate_estimator_reporting_schema(
        metrics,
        execution_mode=execution_mode,
        exact_scores_available=bool(agreement_fields),
    )
    if execution_mode == "legacy":
        return validated
    return {
        field: value
        for field, value in validated.items()
        if field not in ESTIMATOR_GROUP_METRIC_FIELDS
    }


def summarize_values(values: Sequence[float]) -> dict[str, int | float]:
    """Return deterministic descriptive statistics for one set of seed units."""
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("summary values must be a nonempty sequence")
    array = np.asarray(
        [_finite_number(value, "summary value") for value in values], dtype=float
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = array[
        rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    ].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975), method="linear")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def paired_sign_flip(
    baseline_by_seed: Mapping[object, float],
    comparison_by_seed: Mapping[object, float],
) -> dict[str, int | float | str]:
    """Run an exact, deterministic paired sign-flip permutation test."""
    baseline_keys = tuple(baseline_by_seed)
    if set(baseline_keys) != set(comparison_by_seed):
        raise ValueError("paired comparison requires exact seed alignment")
    if not baseline_keys:
        raise ValueError("paired comparison requires at least one aligned seed")
    if len(baseline_keys) > 20:
        raise ValueError("exact paired sign-flip is limited to 20 paired units")
    keys = tuple(sorted(baseline_keys, key=lambda value: str(value)))
    differences = np.asarray(
        [
            _finite_number(comparison_by_seed[key], "comparison value")
            - _finite_number(baseline_by_seed[key], "baseline value")
            for key in keys
        ],
        dtype=float,
    )
    observed = abs(float(differences.mean()))
    extreme = 0
    total = 1 << len(differences)
    for mask in range(total):
        signs = np.fromiter(
            (1.0 if mask & (1 << index) else -1.0 for index in range(len(differences))),
            dtype=float,
            count=len(differences),
        )
        if abs(float((differences * signs).mean())) >= observed - 1e-15:
            extreme += 1
    nonzero = differences[differences != 0.0]
    if len(nonzero) == 0:
        rank_biserial = 0.0
    else:
        absolute = np.abs(nonzero)
        order = np.argsort(absolute, kind="stable")
        ranks = np.empty(len(nonzero), dtype=float)
        position = 0
        while position < len(nonzero):
            end = position + 1
            while end < len(nonzero) and absolute[order[end]] == absolute[order[position]]:
                end += 1
            average_rank = (position + 1 + end) / 2.0
            ranks[order[position:end]] = average_rank
            position = end
        total_rank = float(ranks.sum())
        rank_biserial = float(
            (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / total_rank
        )
    return {
        "n": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "paired_rank_biserial": rank_biserial,
        "p_value": float(extreme / total),
        "test": "exact_paired_sign_flip",
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply Holm's family-wise correction without a statistics dependency."""
    checked = [_finite_number(value, "p-value") for value in p_values]
    if any(value < 0.0 or value > 1.0 for value in checked):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(len(checked)), key=lambda index: (checked[index], index))
    adjusted = [0.0] * len(checked)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(checked) - rank) * checked[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _validated_row(row: Mapping[str, object], columns: Sequence[str], context: str) -> dict[str, object]:
    if not isinstance(row, Mapping) or set(row) != set(columns):
        raise ValueError(f"{context} row has an invalid schema")
    return {column: row[column] for column in columns}


def aggregate_canonical_records(
    raw_rows: Sequence[Mapping[str, object]],
    front_rows: Sequence[Mapping[str, object]],
    *,
    allow_approved_skip_subset: bool = False,
) -> dict[str, list[dict[str, object]]]:
    """Build ordered raw, aggregate, paired, and front tables from validated rows."""
    if type(allow_approved_skip_subset) is not bool:
        raise TypeError("allow_approved_skip_subset must be a bool")
    raw = [_validated_row(row, RAW_COLUMNS, "raw") for row in raw_rows]
    front = [_validated_row(row, FRONT_COLUMNS, "front") for row in front_rows]
    if allow_approved_skip_subset and not raw:
        raise ValueError(
            "formal approved-skip summary requires a nonempty canonical seed subset"
        )
    if raw:
        run_kinds = {row["run_kind"] for row in raw}
        if run_kinds not in ({"formal"}, {"smoke"}):
            raise ValueError("raw rows must belong to exactly one run kind")
        if run_kinds == {"formal"}:
            observed_seeds = {int(row["seed"]) for row in raw}
            if allow_approved_skip_subset:
                if not observed_seeds or not observed_seeds <= set(FORMAL_SEEDS):
                    raise ValueError(
                        "formal approved-skip summary requires a nonempty canonical seed subset"
                    )
            elif observed_seeds != set(FORMAL_SEEDS):
                raise ValueError("formal summary requires the canonical five-seed scope")
    for row in raw:
        if row["run_kind"] not in {"formal", "smoke"}:
            raise ValueError("raw row run_kind is invalid")
        for field in ("stage", "dataset_id", "method", "metric"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"raw row {field} is invalid")
        if not isinstance(row["parameter"], str):
            raise ValueError("raw row parameter is invalid")
        _finite_number(row["value"], "raw value")
        for field in ("manifest_sequence", "seed", "trial"):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0:
                raise ValueError(f"raw row {field} is invalid")

    raw.sort(
        key=lambda row: (
            int(row["manifest_sequence"]), str(row["metric"]), int(row["trial"])
        )
    )
    per_seed: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in raw:
        key = (
            row["run_kind"], row["stage"], row["dataset_id"], row["method"],
            row["parameter"], row["metric"], row["seed"],
        )
        per_seed[key].append(float(row["value"]))
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for key, values in per_seed.items():
        groups[key[:-1]].append(float(np.mean(values)))
    aggregate: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        run_kind, stage, dataset_id, method, parameter, metric = key
        summary = summarize_values(groups[key])
        aggregate.append(
            {
                "run_kind": run_kind,
                "stage": stage,
                "dataset_id": dataset_id,
                "method": method,
                "parameter": parameter,
                "metric": metric,
                **summary,
                "analysis_label": (
                    "descriptive_smoke" if run_kind == "smoke" else "descriptive_low_power"
                ),
            }
        )

    paired: list[dict[str, object]] = []
    nsga_metrics = ("validation_hv", "test_hv", "convergence_evaluation_count")
    nsga = [row for row in raw if row["stage"] == "nsga2" and row["metric"] in nsga_metrics]
    dataset_ids = sorted({str(row["dataset_id"]) for row in nsga})
    scopes: list[str] = [*dataset_ids]
    if dataset_ids:
        scopes.append("__all__")
    for metric in nsga_metrics:
        family: list[dict[str, object]] = []
        for dataset_id in scopes:
            selected = [
                row for row in nsga
                if row["metric"] == metric
                and (dataset_id == "__all__" or row["dataset_id"] == dataset_id)
            ]
            if dataset_id == "__all__":
                by_method_seed: dict[str, dict[int, list[tuple[str, float]]]] = {
                    "full": defaultdict(list),
                    "minppi_prefilter": defaultdict(list),
                }
                for row in selected:
                    method = str(row["method"])
                    if method in by_method_seed:
                        by_method_seed[method][int(row["seed"])].append(
                            (str(row["dataset_id"]), float(row["value"]))
                        )
                expected_datasets = set(dataset_ids)
                pooled_complete = True
                for method, seed_rows in by_method_seed.items():
                    for seed, entries in seed_rows.items():
                        names = [name for name, _ in entries]
                        if len(names) != len(set(names)) or set(names) != expected_datasets:
                            pooled_complete = False
                baseline = {
                    seed: float(np.mean([value for _, value in entries]))
                    for seed, entries in by_method_seed["full"].items()
                } if pooled_complete else {}
                comparison = {
                    seed: float(np.mean([value for _, value in entries]))
                    for seed, entries in by_method_seed["minppi_prefilter"].items()
                } if pooled_complete else {}
            else:
                baseline_rows = [row for row in selected if row["method"] == "full"]
                comparison_rows = [
                    row for row in selected if row["method"] == "minppi_prefilter"
                ]
                if len({int(row["seed"]) for row in baseline_rows}) != len(baseline_rows):
                    raise ValueError("paired baseline contains duplicate seeds")
                if len({int(row["seed"]) for row in comparison_rows}) != len(comparison_rows):
                    raise ValueError("paired comparison contains duplicate seeds")
                baseline = {int(row["seed"]): float(row["value"]) for row in baseline_rows}
                comparison = {
                    int(row["seed"]): float(row["value"]) for row in comparison_rows
                }
            if not baseline and not comparison:
                continue
            required_keys = set(FORMAL_SEEDS) if selected[0]["run_kind"] == "formal" else set(baseline)
            if set(baseline) != set(comparison) or set(baseline) != required_keys:
                continue
            result = paired_sign_flip(baseline, comparison)
            run_kind = str(selected[0]["run_kind"])
            family.append(
                {
                    "run_kind": run_kind,
                    "stage": "nsga2",
                    "dataset_id": dataset_id,
                    "metric": metric,
                    "baseline_method": "full",
                    "comparison_method": "minppi_prefilter",
                    **result,
                    "holm_p_value": 0.0,
                    "holm_family_id": f"nsga2:{metric}",
                    "holm_family_size": 0,
                    "analysis_label": (
                        "descriptive_smoke" if run_kind == "smoke" else "descriptive_low_power"
                    ),
                }
            )
        adjusted = holm_adjust([float(row["p_value"]) for row in family])
        for row, value in zip(family, adjusted, strict=True):
            row["holm_p_value"] = value
            row["holm_family_size"] = len(family)
        paired.extend(family)

    for row in front:
        if row["run_kind"] not in {"formal", "smoke"}:
            raise ValueError("front row run_kind is invalid")
        for field in ("dataset_id", "method", "front_kind"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"front row {field} is invalid")
        for field in ("seed", "point_index", "evaluation"):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0:
                raise ValueError(f"front row {field} is invalid")
        for field in ("compression", "utility", "hypervolume"):
            _finite_number(row[field], f"front row {field}")
    front.sort(
        key=lambda row: (
            str(row["dataset_id"]), int(row["seed"]), str(row["method"]),
            str(row["front_kind"]), int(row["point_index"]), int(row["evaluation"]),
        )
    )
    return {"raw": raw, "aggregate": aggregate, "paired": paired, "front": front}


def _selector(row: Mapping[str, object], columns: Sequence[str]) -> dict[str, object]:
    return {column: row[column] for column in columns}


def generate_latex_macros(
    aggregate_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
    *,
    run_kind: str,
) -> tuple[str, dict[str, object]]:
    """Generate only macros supported by complete rows from this run kind."""
    if run_kind not in {"formal", "smoke"}:
        raise ValueError("run_kind must be formal or smoke")
    macros: dict[str, dict[str, object]] = {}

    def add_aggregate(name: str, metric: str, formatter) -> None:
        matches = [
            row for row in aggregate_rows
            if row.get("run_kind") == run_kind
            and row.get("stage") == "removal"
            and row.get("dataset_id") == "credit_card_fraud"
            and row.get("method") == "minppi"
            and row.get("parameter") == ""
            and row.get("metric") == metric
            and row.get("n") == (5 if run_kind == "formal" else 1)
        ]
        if len(matches) > 1:
            raise ValueError(f"macro {name} selector is not unique")
        if not matches:
            return
        row = matches[0]
        value = _finite_number(row["mean"], f"macro {name}")
        formatted = formatter(value)
        macros[name] = {
            "source_table": "aggregate_metrics.csv",
            "row_selector": _selector(
                row, ("run_kind", "stage", "dataset_id", "method", "parameter", "metric")
            ),
            "source_column": "mean",
            "raw_numeric_value": value,
            "formatted_value": formatted,
        }

    add_aggregate("CreditCompression", "compression", lambda value: f"{value * 100:.1f}\\%")
    add_aggregate("CreditDeltaUtility", "delta_utility", lambda value: f"{value:+.3f}")
    matches = [
        row for row in paired_rows
        if row.get("run_kind") == run_kind
        and row.get("stage") == "nsga2"
        and row.get("dataset_id") == "__all__"
        and row.get("metric") == "test_hv"
        and row.get("baseline_method") == "full"
        and row.get("comparison_method") == "minppi_prefilter"
        and row.get("n") == (5 if run_kind == "formal" else 1)
    ]
    if len(matches) > 1:
        raise ValueError("macro NsgaMeanHvChange selector is not unique")
    if matches:
        row = matches[0]
        value = _finite_number(row["mean_difference"], "macro NsgaMeanHvChange")
        formatted = f"{value:+.3f}"
        macros["NsgaMeanHvChange"] = {
            "source_table": "paired_comparisons.csv",
            "row_selector": _selector(
                row,
                (
                    "run_kind", "stage", "dataset_id", "metric",
                    "baseline_method", "comparison_method",
                ),
            ),
            "source_column": "mean_difference",
            "raw_numeric_value": value,
            "formatted_value": formatted,
        }
    text = "".join(
        f"\\newcommand{{\\{name}}}{{{entry['formatted_value']}}}\n"
        for name, entry in macros.items()
    )
    return text, {"schema_version": 1, "run_kind": run_kind, "macros": macros}


def csv_bytes(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> bytes:
    """Serialize validated rows as stable UTF-8 CSV with LF line endings."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        checked = _validated_row(row, columns, "CSV")
        serialized = {
            key: (format(value, ".17g") if isinstance(value, float) else value)
            for key, value in checked.items()
        }
        writer.writerow(serialized)
    return output.getvalue().encode("utf-8")


def provenance_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
