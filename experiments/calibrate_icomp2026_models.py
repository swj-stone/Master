"""Calibrate a frozen per-dataset XGBoost round policy on validation data only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from minppi_mofs.experiment_protocol import (
    DatasetSpec,
    load_resolved_protocol,
    read_rich_data_manifest,
)
from minppi_mofs.modeling import (
    _empty_subset_predictions,
    evaluate_fixed_subset,
    fit_predict_checkpoints,
)
from minppi_mofs.multifidelity import (
    CalibrationRules,
    CandidateCalibration,
    DatasetModelPolicy,
    ModelPolicy,
    select_candidate_round,
    write_model_policy,
)
from minppi_mofs.utility import compute_utility


_CANDIDATE_ROUNDS = (10, 20, 40, 100)
_CALIBRATION_SEED = 42


@dataclass(frozen=True)
class MarginalPair:
    """One validation-only marginal comparison for a named feature."""

    feature: str
    fraction: float
    without: tuple[str, ...]
    with_feature: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str) or not self.feature:
            raise ValueError("feature must be a non-empty string")
        if self.fraction not in (0.25, 0.75):
            raise ValueError("fraction must be 0.25 or 0.75")
        if (
            self.feature in self.without
            or self.feature not in self.with_feature
            or len(self.with_feature) != len(self.without) + 1
        ):
            raise ValueError("marginal pair coalitions are inconsistent")


def _ordered_subset(
    features: tuple[str, ...], selected: set[str]
) -> tuple[str, ...]:
    return tuple(feature for feature in features if feature in selected)


def make_marginal_panel(
    features: Sequence[str], *, seed: int = _CALIBRATION_SEED
) -> tuple[MarginalPair, ...]:
    """Create two deterministic 25%/75% marginal pairs for every feature."""
    ordered = tuple(features)
    if (
        not ordered
        or len(set(ordered)) != len(ordered)
        or not all(isinstance(feature, str) and feature for feature in ordered)
    ):
        raise ValueError("features must contain unique non-empty strings")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rows: list[MarginalPair] = []
    for feature in ordered:
        remaining = tuple(item for item in ordered if item != feature)
        for fraction in (0.25, 0.75):
            sample_size = int(round(fraction * len(remaining)))
            material = (
                f"{seed}\0{feature}\0{fraction:.2f}\0"
                + "\0".join(ordered)
            ).encode("utf-8")
            local_seed = int.from_bytes(
                hashlib.sha256(material).digest()[:8], "big"
            )
            sampled = set(
                random.Random(local_seed).sample(remaining, sample_size)
            )
            without = _ordered_subset(ordered, sampled)
            with_feature = _ordered_subset(ordered, sampled | {feature})
            rows.append(
                MarginalPair(
                    feature=feature,
                    fraction=fraction,
                    without=without,
                    with_feature=with_feature,
                )
            )
    return tuple(rows)


def panel_coalitions(
    features: Sequence[str], panel: Sequence[MarginalPair]
) -> tuple[tuple[str, ...], ...]:
    """Return the stable, de-duplicated coalition set used by every candidate."""
    full = tuple(features)
    if not full or len(set(full)) != len(full):
        raise ValueError("features must be non-empty and unique")
    allowed = set(full)
    result: list[tuple[str, ...]] = [full]
    seen = {full}
    for row in panel:
        if not isinstance(row, MarginalPair):
            raise TypeError("panel must contain MarginalPair values")
        for coalition in (row.without, row.with_feature):
            if not set(coalition).issubset(allowed):
                raise ValueError("panel coalition is outside feature scope")
            if coalition not in seen:
                seen.add(coalition)
                result.append(coalition)
    return tuple(result)


def _spearman(reference: Sequence[float], candidate: Sequence[float]) -> float:
    left = pd.Series(reference, dtype=float).rank(method="average")
    right = pd.Series(candidate, dtype=float).rank(method="average")
    correlation = float(left.corr(right))
    if math.isfinite(correlation):
        return round(max(-1.0, min(1.0, correlation)), 12)
    return 1.0 if np.allclose(reference, candidate) else 0.0


def _sign(value: float, *, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _harmful_labels(
    values: Mapping[tuple[str, ...], float],
    panel: Sequence[MarginalPair],
    *,
    threshold: float,
) -> tuple[bool, ...]:
    return tuple(
        values[row.without] >= threshold
        and values[row.with_feature] < threshold
        for row in panel
    )


def calibration_metrics(
    reference: Mapping[tuple[str, ...], float],
    candidate: Mapping[tuple[str, ...], float],
    *,
    panel: Sequence[MarginalPair],
    full: tuple[str, ...],
    alpha: float,
) -> dict[str, float]:
    """Compare one candidate utility landscape with the 100-round reference."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    keys = tuple(reference)
    if not keys or set(candidate) != set(keys) or full not in reference:
        raise ValueError("candidate and reference utility scopes must match")
    if any(
        not math.isfinite(float(values[key]))
        for values in (reference, candidate)
        for key in keys
    ):
        raise ValueError("calibration utilities must be finite")
    reference_full = float(reference[full])
    candidate_full = float(candidate[full])
    reference_winning = tuple(
        float(reference[key]) >= alpha * reference_full for key in keys
    )
    candidate_winning = tuple(
        float(candidate[key]) >= alpha * candidate_full for key in keys
    )
    reference_signs = tuple(
        _sign(float(reference[row.with_feature]) - float(reference[row.without]))
        for row in panel
    )
    candidate_signs = tuple(
        _sign(float(candidate[row.with_feature]) - float(candidate[row.without]))
        for row in panel
    )
    reference_harmful = _harmful_labels(
        reference, panel, threshold=alpha * reference_full
    )
    candidate_harmful = _harmful_labels(
        candidate, panel, threshold=alpha * candidate_full
    )
    harmful_total = sum(reference_harmful)
    nonharmful_total = len(reference_harmful) - harmful_total
    true_harmful = sum(
        expected and actual
        for expected, actual in zip(reference_harmful, candidate_harmful)
    )
    false_harmful = sum(
        not expected and actual
        for expected, actual in zip(reference_harmful, candidate_harmful)
    )
    return {
        "full_utility_loss": round(
            max(0.0, reference_full - candidate_full), 12
        ),
        "spearman": _spearman(
            [float(reference[key]) for key in keys],
            [float(candidate[key]) for key in keys],
        ),
        "winning_agreement": sum(
            left == right
            for left, right in zip(reference_winning, candidate_winning)
        )
        / len(keys),
        "marginal_sign_agreement": sum(
            left == right
            for left, right in zip(reference_signs, candidate_signs)
        )
        / len(reference_signs),
        "harmful_recall": (
            true_harmful / harmful_total if harmful_total else 1.0
        ),
        "false_harmful_rate": (
            false_harmful / nonharmful_total if nonharmful_total else 0.0
        ),
    }


def _safe_partition_path(run_root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("prepared partition path is invalid")
    candidate = (run_root / relative).resolve()
    root = run_root.resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("prepared partition path escapes the run root")
    return candidate


def load_prepared_train_validation(
    run_root: Path, row: Mapping[str, object]
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load and hash-check train/validation only; the test path is never read."""
    root = Path(run_root)
    try:
        selected_value = json.loads(str(row["selected_features_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("selected feature metadata is invalid") from exc
    if (
        not isinstance(selected_value, list)
        or not selected_value
        or len(set(selected_value)) != len(selected_value)
        or not all(isinstance(item, str) and item for item in selected_value)
    ):
        raise ValueError("selected feature metadata is invalid")
    selected = tuple(selected_value)
    target_name = row.get("target")
    task = row.get("task")
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("prepared target name is invalid")
    if task not in {"binary", "multiclass", "regression"}:
        raise ValueError("prepared task is invalid")
    loaded: list[tuple[pd.DataFrame, np.ndarray]] = []
    for partition in ("train", "val"):
        path = _safe_partition_path(root, row.get(f"{partition}_relpath"))
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"prepared {partition} CSV is missing") from exc
        expected_digest = row.get(f"{partition}_file_sha256")
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise ValueError(f"prepared {partition} CSV SHA-256 drift")
        try:
            frame = pd.read_csv(BytesIO(payload))
        except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"prepared {partition} CSV is malformed") from exc
        if list(frame.columns) != [*selected, target_name]:
            raise ValueError(f"prepared {partition} columns are invalid")
        if len(frame) != row.get(f"{partition}_row_count"):
            raise ValueError(f"prepared {partition} row count drift")
        try:
            numeric = frame.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prepared {partition} data must be numeric") from exc
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"prepared {partition} data must be finite")
        X = numeric.loc[:, list(selected)].astype(float)
        y = numeric[target_name].to_numpy()
        if task == "regression":
            y = y.astype(float)
        else:
            if not np.equal(y, np.floor(y)).all():
                raise ValueError("classification targets must be integers")
            y = y.astype(int)
        loaded.append((X, y))
    return loaded[0][0], loaded[0][1], loaded[1][0], loaded[1][1]


def _evaluate_candidate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    coalitions: Sequence[tuple[str, ...]],
    dataset: DatasetSpec,
    base_model_params: Mapping[str, object],
    rounds: int,
    backend: str,
    seed: int,
    evaluator: Callable[..., float] = evaluate_fixed_subset,
) -> tuple[dict[tuple[str, ...], float], float, int]:
    params = {**dict(base_model_params), "n_estimators": rounds}
    values: dict[tuple[str, ...], float] = {}
    physical_fits = 0
    started = time.perf_counter()
    for ordinal, coalition in enumerate(coalitions, start=1):
        values[coalition] = float(
            evaluator(
                X_train,
                y_train,
                X_val,
                y_val,
                coalition,
                task=dataset.task,
                utility=dataset.utility,
                model_params=params,
                backend=backend,
                seed=seed,
            )
        )
        physical_fits += int(bool(coalition))
        if ordinal == len(coalitions) or ordinal % 50 == 0:
            print(
                f"[calibrate] dataset={dataset.dataset_id} "
                f"rounds={rounds} coalitions={ordinal}/{len(coalitions)}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return values, elapsed, physical_fits


def _evaluate_checkpoints(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    coalitions: Sequence[tuple[str, ...]],
    dataset: DatasetSpec,
    base_model_params: Mapping[str, object],
    rounds: Sequence[int],
    backend: str,
    seed: int,
    checkpoint_predictor: Callable[..., Mapping[int, np.ndarray]] = (
        fit_predict_checkpoints
    ),
) -> tuple[dict[int, dict[tuple[str, ...], float]], float, int]:
    """Fit each coalition once and evaluate every boosting prefix."""
    checkpoints = tuple(rounds)
    if (
        not checkpoints
        or tuple(sorted(checkpoints)) != checkpoints
        or len(set(checkpoints)) != len(checkpoints)
    ):
        raise ValueError("rounds must be unique and increasing")
    params = {
        **dict(base_model_params),
        "n_estimators": checkpoints[-1],
    }
    values = {candidate: {} for candidate in checkpoints}
    physical_fits = 0
    started = time.perf_counter()
    for ordinal, coalition in enumerate(coalitions, start=1):
        if coalition:
            predictions = checkpoint_predictor(
                X_train.loc[:, list(coalition)],
                y_train,
                X_val.loc[:, list(coalition)],
                task=dataset.task,
                model_params=params,
                checkpoints=checkpoints,
                backend=backend,
                seed=seed,
            )
            physical_fits += 1
        else:
            constant = _empty_subset_predictions(
                y_train, task=dataset.task, row_count=len(X_val)
            )
            predictions = {
                candidate: constant for candidate in checkpoints
            }
        if tuple(predictions) != checkpoints:
            raise ValueError("checkpoint predictor returned the wrong rounds")
        for candidate in checkpoints:
            value = float(
                compute_utility(
                    y_val,
                    np.asarray(predictions[candidate]),
                    metric=dataset.utility,
                )
            )
            if not math.isfinite(value):
                raise ValueError("checkpoint utility must be finite")
            values[candidate][coalition] = value
        if ordinal == len(coalitions) or ordinal % 50 == 0:
            print(
                f"[calibrate] dataset={dataset.dataset_id} "
                f"shared-prefix coalitions={ordinal}/{len(coalitions)}",
                flush=True,
            )
    return values, time.perf_counter() - started, physical_fits


def calibrate_dataset(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    dataset: DatasetSpec,
    base_model_params: Mapping[str, object],
    candidate_rounds: Sequence[int],
    rules: CalibrationRules,
    backend: str,
    seed: int,
    alpha: float,
    checkpoint_predictor: Callable[..., Mapping[int, np.ndarray]] = (
        fit_predict_checkpoints
    ),
) -> DatasetModelPolicy:
    """Evaluate one shared panel and select the smallest passing round count."""
    rounds = tuple(candidate_rounds)
    features = tuple(str(column) for column in X_train.columns)
    panel = make_marginal_panel(features, seed=seed)
    coalitions = panel_coalitions(features, panel)
    if 100 not in rounds:
        raise ValueError("candidate_rounds must contain the 100-round reference")
    evaluated, shared_elapsed, physical_fits = _evaluate_checkpoints(
        X_train,
        y_train,
        X_val,
        y_val,
        coalitions=coalitions,
        dataset=dataset,
        base_model_params=base_model_params,
        rounds=rounds,
        backend=backend,
        seed=seed,
        checkpoint_predictor=checkpoint_predictor,
    )
    reference = evaluated[100]
    rows: list[CandidateCalibration] = []
    for candidate in rounds:
        values = evaluated[candidate]
        metrics = calibration_metrics(
            reference,
            values,
            panel=panel,
            full=features,
            alpha=alpha,
        )
        rows.append(
            CandidateCalibration(
                n_estimators=candidate,
                full_utility=values[features],
                seconds_per_fit=(
                    shared_elapsed
                    / max(physical_fits, 1)
                    * candidate
                    / rounds[-1]
                ),
                physical_fits=max(physical_fits, 1),
                **metrics,
            )
        )
    calibrations = tuple(rows)
    return DatasetModelPolicy(
        selected_n_estimators=select_candidate_round(calibrations, rules),
        reference_n_estimators=100,
        calibrations=calibrations,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary_markdown(policy: ModelPolicy) -> str:
    lines = [
        "# ICOMP 2026 validation-only model calibration",
        "",
        "> This policy was calibrated on train/validation partitions only. "
        "The test partitions remained sealed.",
        "",
        "| Dataset | Selected rounds | Reference s/fit | Selected s/fit | "
        "Estimated speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset_id, item in policy.datasets.items():
        by_round = {row.n_estimators: row for row in item.calibrations}
        selected = by_round[item.selected_n_estimators]
        reference = by_round[item.reference_n_estimators]
        speedup = (
            reference.seconds_per_fit / selected.seconds_per_fit
            if selected.seconds_per_fit > 0
            else 0.0
        )
        lines.append(
            f"| {dataset_id} | {item.selected_n_estimators} | "
            f"{reference.seconds_per_fit:.6f} | "
            f"{selected.seconds_per_fit:.6f} | {speedup:.2f}x |"
        )
        lines.extend(
            [
                "",
                f"## {dataset_id}",
                "",
                "| Rounds | Full utility | Loss | Spearman | Winning | "
                "Marginal sign | Harmful recall | False harmful | Pass |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in item.calibrations:
            lines.append(
                f"| {row.n_estimators} | {row.full_utility:.6f} | "
                f"{row.full_utility_loss:.6f} | {row.spearman:.4f} | "
                f"{row.winning_agreement:.4f} | "
                f"{row.marginal_sign_agreement:.4f} | "
                f"{row.harmful_recall:.4f} | "
                f"{row.false_harmful_rate:.4f} | "
                f"{'yes' if row.meets(policy.rules) else 'no'} |"
            )
        if item.selected_n_estimators == 100:
            lines.extend(
                [
                    "",
                    "Fallback: no lower-round candidate met every frozen "
                    "fidelity rule; 100 rounds remain authoritative.",
                ]
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate per-dataset XGBoost rounds without test access.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=_CALIBRATION_SEED)
    parser.add_argument(
        "--candidate-rounds",
        type=int,
        nargs="+",
        default=list(_CANDIDATE_ROUNDS),
    )
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rounds = tuple(args.candidate_rounds)
    if (
        rounds != tuple(sorted(set(rounds)))
        or not rounds
        or rounds[-1] != 100
        or any(value < 1 for value in rounds)
    ):
        raise ValueError(
            "candidate rounds must be unique, increasing, positive, and end at 100"
        )
    protocol = load_resolved_protocol(
        args.config_dir,
        seeds=(args.seed,),
        backend=args.backend,
        smoke=False,
    )
    rows = read_rich_data_manifest(
        args.run_root / "data_manifest.csv", protocol
    )
    rules = CalibrationRules()
    policies: dict[str, DatasetModelPolicy] = {}
    for dataset in protocol.datasets:
        row = next(
            item
            for item in rows
            if item["dataset_id"] == dataset.dataset_id
            and item["seed"] == args.seed
        )
        print(
            f"[calibrate] start dataset={dataset.dataset_id} "
            f"features={row['selected_feature_count']}",
            flush=True,
        )
        X_train, y_train, X_val, y_val = load_prepared_train_validation(
            args.run_root, row
        )
        policies[dataset.dataset_id] = calibrate_dataset(
            X_train,
            y_train,
            X_val,
            y_val,
            dataset=dataset,
            base_model_params=protocol.model_params,
            candidate_rounds=rounds,
            rules=rules,
            backend=args.backend,
            seed=args.seed,
            alpha=args.alpha,
        )
        print(
            f"[calibrate] selected dataset={dataset.dataset_id} "
            f"rounds={policies[dataset.dataset_id].selected_n_estimators}",
            flush=True,
        )
    policy = ModelPolicy(
        schema_version=1,
        calibration_seed=args.seed,
        candidate_rounds=rounds,
        rules=rules,
        datasets=policies,
    )
    output = (
        args.output
        if args.output is not None
        else args.run_root / "calibration" / "model_policy.json"
    )
    write_model_policy(output, policy)
    summary = output.with_name("calibration_summary.md")
    _atomic_write_text(summary, _summary_markdown(policy))
    print(f"[calibrate] policy={output}", flush=True)
    print(f"[calibrate] summary={summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
