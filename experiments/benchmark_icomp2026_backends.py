"""Validation-only throughput benchmark for the frozen ICOMP 2026 model.

The benchmark intentionally loads only the prepared training and validation
partitions from an existing run.  It compares end-to-end canonical model fits
on an identical deterministic coalition workload and records both throughput
and validation-utility drift.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import yaml

if __package__:
    from .calibrate_icomp2026_models import (
        MarginalPair,
        calibration_metrics,
        load_prepared_train_validation,
        make_marginal_panel,
        panel_coalitions,
    )
else:
    from calibrate_icomp2026_models import (
        MarginalPair,
        calibration_metrics,
        load_prepared_train_validation,
        make_marginal_panel,
        panel_coalitions,
    )
from minppi_mofs.modeling import fit_predict
from minppi_mofs.utility import compute_utility


@dataclass(frozen=True)
class BenchmarkCandidate:
    backend: str
    n_jobs: int
    workers: int

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError("benchmark backend must be cpu or cuda")
        for field in ("n_jobs", "workers"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"benchmark {field} must be a positive integer")

    @property
    def label(self) -> str:
        return f"{self.backend}-n{self.n_jobs}-w{self.workers}"


def make_benchmark_workload(
    features: Sequence[str],
    *,
    seed: int,
    fit_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Return repeated nested 25/50/75/100% coalitions in canonical order."""
    ordered = tuple(features)
    if (
        not ordered
        or len(set(ordered)) != len(ordered)
        or not all(isinstance(feature, str) and feature for feature in ordered)
    ):
        raise ValueError("benchmark features must be non-empty unique strings")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("benchmark seed must be a non-negative integer")
    if isinstance(fit_count, bool) or not isinstance(fit_count, int) or fit_count < 4:
        raise ValueError("benchmark fit_count must be at least four")
    rng = np.random.default_rng(seed)
    permutation = tuple(str(value) for value in rng.permutation(ordered))
    nested: list[tuple[str, ...]] = []
    for fraction in (0.25, 0.50, 0.75, 1.00):
        count = max(1, min(len(ordered), int(round(fraction * len(ordered)))))
        selected = frozenset(permutation[:count])
        nested.append(tuple(feature for feature in ordered if feature in selected))
    return tuple(nested[index % len(nested)] for index in range(fit_count))


def make_structural_panel(
    features: Sequence[str],
    *,
    target_feature_count: int,
    seed: int,
) -> tuple[MarginalPair, ...]:
    """Select a predeclared, evenly stratified subset of the full panel."""
    ordered = tuple(features)
    if (
        isinstance(target_feature_count, bool)
        or not isinstance(target_feature_count, int)
        or target_feature_count < 1
        or target_feature_count > len(ordered)
    ):
        raise ValueError(
            "target_feature_count must be within the feature count"
        )
    full_panel = make_marginal_panel(ordered, seed=seed)
    if target_feature_count == 1:
        target_indices = (0,)
    else:
        target_indices = tuple(
            int(index * (len(ordered) - 1) / (target_feature_count - 1))
            for index in range(target_feature_count)
        )
    targets = {ordered[index] for index in target_indices}
    return tuple(row for row in full_panel if row.feature in targets)


def choose_fastest_equivalent(
    reports: Sequence[Mapping[str, object]],
    *,
    reference: BenchmarkCandidate,
    utility_tolerance: float,
) -> str:
    """Choose the highest-throughput candidate inside the declared drift gate."""
    if (
        isinstance(utility_tolerance, bool)
        or not isinstance(utility_tolerance, (int, float))
        or not math.isfinite(float(utility_tolerance))
        or utility_tolerance < 0
    ):
        raise ValueError("utility_tolerance must be finite and non-negative")
    eligible: list[Mapping[str, object]] = []
    labels: set[str] = set()
    for report in reports:
        label = report.get("candidate")
        throughput = report.get("fits_per_second")
        drift = report.get("max_abs_utility_delta_to_reference")
        if (
            not isinstance(label, str)
            or not label
            or label in labels
            or isinstance(throughput, bool)
            or not isinstance(throughput, (int, float))
            or not math.isfinite(float(throughput))
            or throughput <= 0
            or isinstance(drift, bool)
            or not isinstance(drift, (int, float))
            or not math.isfinite(float(drift))
            or drift < 0
        ):
            raise ValueError("benchmark candidate report is invalid")
        labels.add(label)
        if float(drift) <= float(utility_tolerance):
            eligible.append(report)
    if reference.label not in labels:
        raise ValueError("benchmark reference report is missing")
    if not eligible:
        raise ValueError("no benchmark candidate satisfies the utility gate")
    winner = max(eligible, key=lambda item: float(item["fits_per_second"]))
    return str(winner["candidate"])


def _parse_candidate(value: str) -> BenchmarkCandidate:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "candidate must have BACKEND:N_JOBS:WORKERS form"
        )
    try:
        return BenchmarkCandidate(parts[0], int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _read_manifest_row(
    run_root: Path,
    *,
    dataset_id: str,
    seed: int,
) -> dict[str, object]:
    path = run_root / "data_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row.get("dataset_id") == dataset_id
        and row.get("seed") == str(seed)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one manifest row for {dataset_id} seed={seed}"
        )
    row: dict[str, object] = dict(matches[0])
    for field in ("train_row_count", "val_row_count"):
        try:
            row[field] = int(str(row[field]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"manifest {field} is invalid") from exc
    return row


def _resolved_protocol(run_root: Path) -> dict[str, object]:
    path = run_root / "resolved_protocol.yaml"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("cannot read the frozen resolved protocol") from exc
    if not isinstance(value, dict):
        raise ValueError("frozen resolved protocol is invalid")
    return value


def _dataset_authority(
    protocol: Mapping[str, object],
    dataset_id: str,
) -> tuple[str, str, dict[str, object]]:
    datasets = protocol.get("datasets")
    policies = protocol.get("model_params_by_dataset")
    if not isinstance(datasets, list) or not isinstance(policies, dict):
        raise ValueError("resolved protocol dataset/model authority is invalid")
    matches = [
        item
        for item in datasets
        if isinstance(item, dict) and item.get("dataset_id") == dataset_id
    ]
    params = policies.get(dataset_id)
    if len(matches) != 1 or not isinstance(params, dict):
        raise ValueError(f"resolved protocol is missing dataset {dataset_id}")
    task = matches[0].get("task")
    utility = matches[0].get("utility")
    if task not in {"binary", "multiclass", "regression"}:
        raise ValueError("resolved dataset task is invalid")
    if not isinstance(utility, str) or not utility:
        raise ValueError("resolved dataset utility is invalid")
    return str(task), utility, dict(params)


def _evaluate_fit(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    coalition: tuple[str, ...],
    *,
    task: str,
    utility: str,
    model_params: Mapping[str, object],
    candidate: BenchmarkCandidate,
    seed: int,
) -> tuple[float, float]:
    started = time.perf_counter()
    predictions = fit_predict(
        X_train.loc[:, list(coalition)],
        y_train,
        X_val.loc[:, list(coalition)],
        task=task,
        model_params={**dict(model_params), "n_jobs": candidate.n_jobs},
        backend=candidate.backend,
        seed=seed,
    )
    value = float(compute_utility(y_val, predictions, metric=utility))
    elapsed = time.perf_counter() - started
    return value, elapsed


def _benchmark_candidate(
    *,
    candidate: BenchmarkCandidate,
    workload: tuple[tuple[str, ...], ...],
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    task: str,
    utility: str,
    model_params: Mapping[str, object],
    seed: int,
) -> tuple[list[float], list[float], float]:
    # Warm the selected backend before measuring.  This is deliberately not
    # included in the throughput denominator.
    _evaluate_fit(
        X_train,
        y_train,
        X_val,
        y_val,
        workload[0],
        task=task,
        utility=utility,
        model_params=model_params,
        candidate=candidate,
        seed=seed,
    )
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=candidate.workers) as pool:
        futures = [
            pool.submit(
                _evaluate_fit,
                X_train,
                y_train,
                X_val,
                y_val,
                coalition,
                task=task,
                utility=utility,
                model_params=model_params,
                candidate=candidate,
                seed=seed,
            )
            for coalition in workload
        ]
        evaluated = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    return (
        [item[0] for item in evaluated],
        [item[1] for item in evaluated],
        wall_s,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark real validation-only ICOMP 2026 model fits.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-count", type=int, default=8)
    parser.add_argument(
        "--marginal-feature-count",
        type=int,
        help=(
            "replace the throughput workload with an evenly stratified "
            "two-scale marginal panel over this many target features"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        dest="candidates",
        required=True,
        help="repeat BACKEND:N_JOBS:WORKERS; first candidate is the reference",
    )
    parser.add_argument("--utility-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len({candidate.label for candidate in args.candidates}) != len(
        args.candidates
    ):
        raise ValueError("benchmark candidates must be unique")
    protocol = _resolved_protocol(args.run_root)
    report: dict[str, object] = {
        "schema_version": 1,
        "test_partition_read": False,
        "run_root": str(args.run_root.resolve()),
        "seed": args.seed,
        "fit_count_per_candidate": args.fit_count,
        "marginal_feature_count": args.marginal_feature_count,
        "alpha": args.alpha,
        "utility_tolerance": args.utility_tolerance,
        "reference_candidate": args.candidates[0].label,
        "datasets": {},
    }
    dataset_reports: dict[str, object] = {}
    for dataset_id in args.datasets:
        row = _read_manifest_row(
            args.run_root,
            dataset_id=dataset_id,
            seed=args.seed,
        )
        X_train, y_train, X_val, y_val = load_prepared_train_validation(
            args.run_root, row
        )
        task, utility, params = _dataset_authority(protocol, dataset_id)
        features = tuple(str(column) for column in X_train.columns)
        structural_panel = (
            None
            if args.marginal_feature_count is None
            else make_structural_panel(
                features,
                target_feature_count=args.marginal_feature_count,
                seed=args.seed,
            )
        )
        workload = (
            make_benchmark_workload(
                features,
                seed=args.seed,
                fit_count=args.fit_count,
            )
            if structural_panel is None
            else panel_coalitions(features, structural_panel)
        )
        reference_utilities: list[float] | None = None
        candidate_reports: list[dict[str, object]] = []
        for candidate in args.candidates:
            print(
                f"[benchmark] dataset={dataset_id} candidate={candidate.label} "
                f"fits={len(workload)}",
                flush=True,
            )
            utilities, task_times, wall_s = _benchmark_candidate(
                candidate=candidate,
                workload=workload,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                task=task,
                utility=utility,
                model_params=params,
                seed=args.seed,
            )
            if reference_utilities is None:
                reference_utilities = utilities
            drift = max(
                abs(value - expected)
                for value, expected in zip(
                    utilities, reference_utilities, strict=True
                )
            )
            ordered_times = sorted(task_times)
            p90_index = min(
                len(ordered_times) - 1,
                max(0, math.ceil(0.90 * len(ordered_times)) - 1),
            )
            candidate_report: dict[str, object] = {
                "candidate": candidate.label,
                "backend": candidate.backend,
                "n_jobs": candidate.n_jobs,
                "workers": candidate.workers,
                "fit_count": len(workload),
                "wall_s": wall_s,
                "fits_per_second": len(workload) / wall_s,
                "median_task_s": statistics.median(task_times),
                "p90_task_s": ordered_times[p90_index],
                "max_abs_utility_delta_to_reference": drift,
                "validation_utility_equivalent": (
                    drift <= args.utility_tolerance
                ),
                "utilities": utilities,
            }
            if structural_panel is not None:
                reference_map = dict(
                    zip(workload, reference_utilities, strict=True)
                )
                candidate_map = dict(zip(workload, utilities, strict=True))
                candidate_report["structural_fidelity"] = calibration_metrics(
                    reference_map,
                    candidate_map,
                    panel=structural_panel,
                    full=features,
                    alpha=args.alpha,
                )
            candidate_reports.append(candidate_report)
        winner = choose_fastest_equivalent(
            candidate_reports,
            reference=args.candidates[0],
            utility_tolerance=args.utility_tolerance,
        )
        reference_rate = float(candidate_reports[0]["fits_per_second"])
        for candidate_report in candidate_reports:
            candidate_report["throughput_speedup_vs_reference"] = (
                float(candidate_report["fits_per_second"]) / reference_rate
            )
        dataset_reports[dataset_id] = {
            "task": task,
            "utility": utility,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "feature_count": X_train.shape[1],
            "workload_feature_counts": [
                len(coalition) for coalition in workload
            ],
            "structural_panel_target_features": (
                []
                if structural_panel is None
                else list(dict.fromkeys(row.feature for row in structural_panel))
            ),
            "fastest_validation_utility_equivalent_candidate": winner,
            "candidates": candidate_reports,
        }
    report["datasets"] = dataset_reports
    _atomic_json(args.output, report)
    print(f"[benchmark] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
