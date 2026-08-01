"""Build the measured P0/P1 runtime ledger for the ICOMP 2026 campaign."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

from minppi_mofs.experiment_protocol import (
    atomic_write_json,
    estimator_budgets_for,
    load_resolved_protocol,
    read_canonical_json,
)


_SUPPORT_STAGE_P90_HOURS = {
    "validate_and_preprocess": 1.0,
    "safety_removal_and_controls": 2.0,
    "paired_nsga2": 3.0,
    "ranking_and_statistics": 4.0,
    "alpha_and_representative_ablation": 2.0,
    "summaries_figures_and_paper_exports": 1.0,
}


def estimate_dataset_seconds(
    *,
    observed_seconds: float,
    observed_budget: int,
    target_budget: int,
    seed_count: int,
    exact: bool,
) -> float:
    """Project one measured seed to a frozen nested target and seed count."""
    for name, value in (
        ("observed_seconds", observed_seconds),
        ("observed_budget", observed_budget),
        ("target_budget", target_budget),
        ("seed_count", seed_count),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    scale = 1.0 if exact else target_budget / observed_budget
    return float(observed_seconds) * scale * int(seed_count)


def finalize_48h_forecast(
    *,
    sequential_estimator_seconds: float,
    minimum_measured_speedup: float,
    support_stage_p90_hours: float,
    guard_band: float,
) -> dict[str, float | bool]:
    """Convert sequential work into a conservative three-shard wall forecast."""
    values = (
        sequential_estimator_seconds,
        minimum_measured_speedup,
        support_stage_p90_hours,
        guard_band,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        for value in values
    ):
        raise ValueError("48h forecast inputs must be positive and finite")
    p50 = (
        float(sequential_estimator_seconds)
        / float(minimum_measured_speedup)
        / 3600.0
    )
    p90 = p50 * float(guard_band)
    total = p90 + float(support_stage_p90_hours)
    return {
        "sequential_estimator_hours": (
            float(sequential_estimator_seconds) / 3600.0
        ),
        "minimum_measured_concurrency_speedup": float(
            minimum_measured_speedup
        ),
        "guard_band": float(guard_band),
        "estimator_p50_wall_hours": p50,
        "estimator_p90_wall_hours": p90,
        "support_stage_p90_hours": float(support_stage_p90_hours),
        "p0_total_p90_hours": total,
        "remaining_margin_hours": 48.0 - total,
        "passes_48h_gate": total <= 48.0,
    }


def _bundle_measurement(path: Path) -> tuple[float, int, int, bool]:
    value = read_canonical_json(path, context="measured estimator bundle")
    if not isinstance(value, dict) or not isinstance(
        value.get("checkpoints"), list
    ):
        raise ValueError(f"estimator bundle is invalid: {path}")
    checkpoints = value["checkpoints"]
    budgets = {
        item.get("budget")
        for item in checkpoints
        if isinstance(item, dict)
    }
    exact = any(
        isinstance(item, dict) and item.get("method") == "exact"
        for item in checkpoints
    )
    runtime = value.get("group_runtime_s")
    fits = value.get("physical_model_fit_count")
    if (
        not budgets
        or any(isinstance(item, bool) or not isinstance(item, int) for item in budgets)
        or isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or runtime <= 0
        or isinstance(fits, bool)
        or not isinstance(fits, int)
        or fits < 0
    ):
        raise ValueError(f"estimator bundle measurement is invalid: {path}")
    return float(runtime), max(budgets), fits, exact


def _markdown(report: Mapping[str, object]) -> str:
    forecast = report["forecast"]
    if not isinstance(forecast, Mapping):
        raise ValueError("forecast report is invalid")
    lines = [
        "# ICOMP 2026 48-hour measured runtime ledger",
        "",
        "> Validation-only forecast. No sealed test partition was read.",
        "",
        "| Dataset | Observed K | Formal K | Exact | Measured s | "
        "Projected 5-seed sequential h |",
        "|---|---:|---:|:---:|---:|---:|",
    ]
    datasets = report["datasets"]
    if not isinstance(datasets, Mapping):
        raise ValueError("forecast dataset rows are invalid")
    for dataset_id, raw in datasets.items():
        if not isinstance(raw, Mapping):
            raise ValueError("forecast dataset row is invalid")
        lines.append(
            f"| {dataset_id} | {raw['observed_budget']} | "
            f"{raw['target_budget']} | {'yes' if raw['exact'] else 'no'} | "
            f"{float(raw['observed_seconds']):.2f} | "
            f"{float(raw['projected_sequential_hours']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## P0 wall-time gate",
            "",
            f"- Sequential estimator work: "
            f"{float(forecast['sequential_estimator_hours']):.2f} h.",
            f"- Conservative measured concurrency speedup: "
            f"{float(forecast['minimum_measured_concurrency_speedup']):.2f}x.",
            f"- Estimator P50/P90 wall: "
            f"{float(forecast['estimator_p50_wall_hours']):.2f} / "
            f"{float(forecast['estimator_p90_wall_hours']):.2f} h.",
            f"- Non-estimator P90 reserve: "
            f"{float(forecast['support_stage_p90_hours']):.2f} h.",
            f"- P0 total P90: "
            f"{float(forecast['p0_total_p90_hours']):.2f} h; "
            f"margin {float(forecast['remaining_margin_hours']):.2f} h.",
            f"- 48-hour gate: "
            f"{'PASS' if forecast['passes_48h_gate'] else 'FAIL'}.",
            "",
            "P1 ISOLET K25 seed-42 sensitivity is deliberately outside the P0 "
            "critical path and runs only if the remaining margin permits.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast the frozen formal campaign from measured bundles.",
        allow_abbrev=False,
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--model-policy", type=Path, required=True)
    parser.add_argument("--decision-run", type=Path, required=True)
    parser.add_argument("--credit-probe-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_resolved_protocol(
        args.config_dir,
        seeds=None,
        backend="cuda",
        smoke=False,
        model_policy_path=args.model_policy,
    )
    dataset_rows: dict[str, object] = {}
    sequential_seconds = 0.0
    for dataset in protocol.datasets:
        source_root = (
            args.credit_probe_run
            if dataset.dataset_id == "credit_card_fraud"
            else args.decision_run
        )
        path = (
            source_root
            / "estimator"
            / "_bundles"
            / dataset.dataset_id
            / "s42"
            / "bundle.json"
        )
        runtime, observed_budget, fits, exact = _bundle_measurement(path)
        target_budget = max(
            estimator_budgets_for(protocol, dataset.dataset_id)
        )
        projected = estimate_dataset_seconds(
            observed_seconds=runtime,
            observed_budget=observed_budget,
            target_budget=target_budget,
            seed_count=len(protocol.seeds),
            exact=exact,
        )
        sequential_seconds += projected
        dataset_rows[dataset.dataset_id] = {
            "source_bundle": str(path.resolve()),
            "observed_seconds": runtime,
            "observed_budget": observed_budget,
            "observed_physical_fits": fits,
            "target_budget": target_budget,
            "seed_count": len(protocol.seeds),
            "exact": exact,
            "projected_sequential_hours": projected / 3600.0,
        }
    support_hours = sum(_SUPPORT_STAGE_P90_HOURS.values())
    forecast = finalize_48h_forecast(
        sequential_estimator_seconds=sequential_seconds,
        minimum_measured_speedup=1.25,
        support_stage_p90_hours=support_hours,
        guard_band=1.30,
    )
    isolet = dataset_rows["isolet"]
    if not isinstance(isolet, Mapping):
        raise ValueError("ISOLET forecast row is invalid")
    p1_isolet_k25_hours = (
        float(isolet["observed_seconds"])
        * (25 / int(isolet["observed_budget"]))
        / 3600.0
    )
    report = {
        "schema_version": 1,
        "test_partition_read": False,
        "datasets": dataset_rows,
        "support_stage_p90_hours": _SUPPORT_STAGE_P90_HOURS,
        "forecast": forecast,
        "p1_isolet_k25_seed42_hours": p1_isolet_k25_hours,
        "interpretation": (
            "P0 passes only if source/configuration remains frozen and the "
            "measured >=1.25x three-shard speedup is retained."
        ),
    }
    atomic_write_json(args.output, report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(
        f"[forecast] P0_P90={forecast['p0_total_p90_hours']:.2f}h "
        f"gate={'PASS' if forecast['passes_48h_gate'] else 'FAIL'}",
        flush=True,
    )
    return 0 if forecast["passes_48h_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
