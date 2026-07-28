"""Build a provisional, validation-only report from a seed-42 screen run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
from typing import Mapping, Sequence
from uuid import uuid4

import yaml

from minppi_mofs.experiment_protocol import (
    atomic_write_json,
    estimator_budgets_for,
    load_resolved_protocol,
    read_canonical_json,
    read_rich_data_manifest,
)
from minppi_mofs.modeling import evaluate_fixed_subset
from minppi_mofs.multifidelity import model_params_for

from experiments.calibrate_icomp2026_models import (
    load_prepared_train_validation,
)


def read_frozen_run_seeds(run_root: Path) -> tuple[int, ...]:
    """Read the complete seed authority needed to validate a shard manifest."""
    path = Path(run_root) / "resolved_protocol.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("screen run protocol is missing or malformed") from exc
    seeds = document.get("seeds") if isinstance(document, dict) else None
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 1
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("screen run seed authority is invalid")
    return tuple(seeds)


@dataclass(frozen=True)
class CoalitionValue:
    coalition: tuple[str, ...]
    utility: float

    def __post_init__(self) -> None:
        if (
            len(set(self.coalition)) != len(self.coalition)
            or not all(isinstance(item, str) and item for item in self.coalition)
            or not math.isfinite(float(self.utility))
        ):
            raise ValueError("coalition value is invalid")
        object.__setattr__(self, "utility", float(self.utility))


@dataclass(frozen=True)
class HarmfulWitness:
    feature: str
    without: tuple[str, ...]
    with_feature: tuple[str, ...]
    without_utility: float
    with_utility: float
    utility_drop: float
    threshold: float


@dataclass(frozen=True)
class ClaimEvidence:
    complete: bool
    harmful_count: int
    score_stability: float
    witness_count: int
    removal_utility_delta: float
    advantage_over_random: float


def classify_conclusion(evidence: ClaimEvidence) -> str:
    """Map fixed directional criteria to one of five provisional grades."""
    if not isinstance(evidence, ClaimEvidence):
        raise TypeError("evidence must be ClaimEvidence")
    if not evidence.complete:
        return "insufficient_evidence"
    if (
        evidence.removal_utility_delta < -0.01
        or evidence.advantage_over_random < -0.01
    ):
        return "counterexample"
    if evidence.harmful_count == 0:
        return "neutral"
    required_witnesses = max(1, math.ceil(evidence.harmful_count / 2))
    if (
        evidence.score_stability >= 0.80
        and evidence.witness_count >= required_witnesses
        and evidence.removal_utility_delta >= -0.005
        and evidence.advantage_over_random >= -0.005
    ):
        return "strong_support"
    if (
        evidence.score_stability >= 0.50
        or evidence.witness_count > 0
    ):
        return "partial_support"
    return "counterexample"


def find_harmful_witness(
    feature: str,
    records: Sequence[CoalitionValue],
    *,
    full: tuple[str, ...],
    alpha: float,
) -> HarmfulWitness | None:
    """Return the largest cached threshold-crossing drop for one feature."""
    if not isinstance(feature, str) or not feature:
        raise ValueError("feature must be a non-empty string")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    values = {record.coalition: record.utility for record in records}
    if len(values) != len(records) or full not in values:
        raise ValueError("coalition records are duplicated or missing full utility")
    threshold = alpha * values[full]
    candidates: list[HarmfulWitness] = []
    full_order = tuple(full)
    for without, without_utility in values.items():
        if feature in without:
            continue
        with_set = set(without) | {feature}
        with_feature = tuple(item for item in full_order if item in with_set)
        if with_feature not in values:
            continue
        with_utility = values[with_feature]
        if without_utility >= threshold and with_utility < threshold:
            candidates.append(
                HarmfulWitness(
                    feature=feature,
                    without=without,
                    with_feature=with_feature,
                    without_utility=without_utility,
                    with_utility=with_utility,
                    utility_drop=round(
                        without_utility - with_utility, 12
                    ),
                    threshold=round(threshold, 12),
                )
            )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.utility_drop,
            item.without_utility,
            item.without,
        ),
    )


def _read_cache_values(path: Path, utility: str) -> tuple[CoalitionValue, ...]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"selection cache is missing or unsafe: {path}")
    try:
        connection = sqlite3.connect(
            f"file:{path.resolve().as_posix()}?mode=ro", uri=True
        )
        rows = connection.execute(
            "SELECT coalition_key, payload_json, payload_sha256 "
            "FROM coalition_utility ORDER BY coalition_key"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError("selection cache is malformed") from exc
    finally:
        if "connection" in locals():
            connection.close()
    records: list[CoalitionValue] = []
    for coalition_key, payload_json, expected_sha in rows:
        if (
            not isinstance(coalition_key, str)
            or not isinstance(payload_json, str)
            or hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            != expected_sha
        ):
            raise ValueError("selection cache row checksum is invalid")
        try:
            key = json.loads(coalition_key)
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("selection cache row JSON is invalid") from exc
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        utilities = payload.get("utilities") if isinstance(payload, dict) else None
        if (
            canonical != payload_json
            or not isinstance(key, list)
            or payload.get("coalition") != key
            or not isinstance(utilities, dict)
            or utility not in utilities
        ):
            raise ValueError("selection cache row payload is invalid")
        records.append(
            CoalitionValue(tuple(key), float(utilities[utility]))
        )
    return tuple(records)


def _checkpoint(
    bundle: Mapping[str, object], method: str, budget: int
) -> Mapping[str, object]:
    checkpoints = bundle.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("estimator bundle checkpoints are invalid")
    matches = [
        item
        for item in checkpoints
        if isinstance(item, dict)
        and item.get("method") == method
        and item.get("budget") == budget
    ]
    if len(matches) != 1:
        raise ValueError(
            f"estimator bundle lacks one {method} K={budget} checkpoint"
        )
    return matches[0]


def _scores(checkpoint: Mapping[str, object]) -> dict[str, int]:
    values = checkpoint.get("scores")
    if not isinstance(values, list):
        raise ValueError("checkpoint scores are invalid")
    result: dict[str, int] = {}
    for item in values:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or item[1] not in {-1, 0, 1}
            or isinstance(item[1], bool)
            or item[0] in result
        ):
            raise ValueError("checkpoint scores are invalid")
        result[item[0]] = item[1]
    if not result:
        raise ValueError("checkpoint scores are empty")
    return result


def select_screen_budget_pair(budgets: Sequence[int]) -> tuple[int, int]:
    """Return the adjacent high-budget checkpoints used for screen stability."""
    values = tuple(budgets)
    if (
        len(values) < 2
        or tuple(sorted(values)) != values
        or len(set(values)) != len(values)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        )
    ):
        raise ValueError("screen requires at least two increasing budgets")
    return values[-2], values[-1]


def _validation_removal(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    dataset,
    scores: Mapping[str, int],
    model_params: Mapping[str, object],
    backend: str,
    seed: int,
    random_repeats: int = 5,
) -> dict[str, object]:
    features = tuple(str(item) for item in X_train.columns)
    harmful = tuple(feature for feature in features if scores[feature] == -1)
    retained = tuple(feature for feature in features if feature not in harmful)

    def evaluate(selected: tuple[str, ...]) -> float:
        return evaluate_fixed_subset(
            X_train,
            y_train,
            X_val,
            y_val,
            selected,
            task=dataset.task,
            utility=dataset.utility,
            model_params=model_params,
            backend=backend,
            seed=seed,
        )

    full_utility = evaluate(features)
    retained_utility = evaluate(retained)
    random_utilities: list[float] = []
    for repeat in range(random_repeats):
        material = (
            f"{dataset.dataset_id}\0{seed}\0{repeat}\0{len(retained)}"
        ).encode("utf-8")
        local_seed = int.from_bytes(
            hashlib.sha256(material).digest()[:8], "big"
        )
        selected = set(
            random.Random(local_seed).sample(features, len(retained))
        )
        ordered = tuple(feature for feature in features if feature in selected)
        random_utilities.append(evaluate(ordered))
    random_mean = float(sum(random_utilities) / len(random_utilities))
    return {
        "full_utility": full_utility,
        "retained_utility": retained_utility,
        "utility_delta": retained_utility - full_utility,
        "random_matched_utilities": random_utilities,
        "random_matched_mean": random_mean,
        "advantage_over_random": retained_utility - random_mean,
        "removed_features": list(harmful),
        "retained_features": list(retained),
        "compression": len(harmful) / len(features),
    }


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


def _screen_dataset(
    run_root: Path,
    protocol,
    dataset,
    row: Mapping[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    bundle_path = (
        run_root
        / "estimator"
        / "_bundles"
        / dataset.dataset_id
        / f"s{seed}"
        / "bundle.json"
    )
    if not bundle_path.is_file():
        return {
            "dataset_id": dataset.dataset_id,
            "complete": False,
            "conclusion_grade": "insufficient_evidence",
            "issue": "completed estimator bundle is missing",
        }
    bundle = read_canonical_json(
        bundle_path, context="completed estimator bundle"
    )
    if not isinstance(bundle, dict):
        raise ValueError("completed estimator bundle is invalid")
    lower_budget, upper_budget = select_screen_budget_pair(
        estimator_budgets_for(protocol, dataset.dataset_id)
    )
    lower_scores = _scores(
        _checkpoint(bundle, "hybrid", lower_budget)
    )
    upper_checkpoint = _checkpoint(bundle, "hybrid", upper_budget)
    upper_scores = _scores(upper_checkpoint)
    if tuple(lower_scores) != tuple(upper_scores):
        raise ValueError("screen checkpoint feature orders differ")
    stability = sum(
        lower_scores[key] == upper_scores[key] for key in upper_scores
    ) / len(upper_scores)
    harmful = tuple(
        key for key, value in upper_scores.items() if value == -1
    )
    cache_path = (
        run_root
        / "cache"
        / "selection_validation"
        / dataset.dataset_id
        / f"s{seed}"
        / "coalitions.sqlite3"
    )
    cached = _read_cache_values(cache_path, dataset.utility)
    full = tuple(upper_scores)
    witnesses = tuple(
        witness
        for feature in harmful
        if (
            witness := find_harmful_witness(
                feature,
                cached,
                full=full,
                alpha=protocol.alpha_default,
            )
        )
        is not None
    )
    X_train, y_train, X_val, y_val = load_prepared_train_validation(
        run_root, row
    )
    removal = _validation_removal(
        X_train,
        y_train,
        X_val,
        y_val,
        dataset=dataset,
        scores=upper_scores,
        model_params=model_params_for(protocol, dataset.dataset_id),
        backend=protocol.backend,
        seed=seed,
    )
    evidence = ClaimEvidence(
        complete=True,
        harmful_count=len(harmful),
        score_stability=stability,
        witness_count=len(witnesses),
        removal_utility_delta=float(removal["utility_delta"]),
        advantage_over_random=float(removal["advantage_over_random"]),
    )
    return {
        "dataset_id": dataset.dataset_id,
        "complete": True,
        "provisional": True,
        "validation_only": True,
        "task": dataset.task,
        "utility": dataset.utility,
        "train_rows": row["train_row_count"],
        "validation_rows": row["val_row_count"],
        "selected_feature_count": row["selected_feature_count"],
        "model_n_estimators": model_params_for(
            protocol, dataset.dataset_id
        )["n_estimators"],
        "group_physical_model_fit_count": bundle[
            "physical_model_fit_count"
        ],
        "group_runtime_s": bundle["group_runtime_s"],
        "stability_budget_pair": [lower_budget, upper_budget],
        "adjacent_budget_score_stability": stability,
        "harmful_count": len(harmful),
        "harmful_features": list(harmful),
        "witness_count": len(witnesses),
        "witness_coverage": (
            len(witnesses) / len(harmful) if harmful else 1.0
        ),
        "witnesses": [asdict(item) for item in witnesses],
        "validation_removal": removal,
        "conclusion_grade": classify_conclusion(evidence),
    }


def _report_markdown(report: Mapping[str, object]) -> str:
    cards = report["datasets"]
    lines = [
        "# ICOMP 2026 seed-42 screen report",
        "",
        "> Provisional and validation-only. No test partition is used by this "
        "report, and these grades must not be presented as final paper results.",
        "",
        "| Dataset | Grade | Rounds | Harmful | Witnesses | Adjacent K | "
        "Removal Δ | vs random |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for card in cards:
        if not card["complete"]:
            lines.append(
                f"| {card['dataset_id']} | insufficient_evidence | — | — | "
                "— | — | — | — |"
            )
            continue
        removal = card["validation_removal"]
        lines.append(
            f"| {card['dataset_id']} | {card['conclusion_grade']} | "
            f"{card['model_n_estimators']} | {card['harmful_count']} | "
            f"{card['witness_count']} | "
            f"{card['adjacent_budget_score_stability']:.3f} | "
            f"{removal['utility_delta']:+.5f} | "
            f"{removal['advantage_over_random']:+.5f} |"
        )
    lines.extend(["", "## Claim matrix", ""])
    counts = report["grade_counts"]
    for grade in (
        "strong_support",
        "partial_support",
        "neutral",
        "counterexample",
        "insufficient_evidence",
    ):
        lines.append(f"- `{grade}`: {counts.get(grade, 0)} dataset(s)")
    forecast = report["runtime_forecast"]
    lines.extend(
        [
            "",
            "## Four-seed confirmation forecast",
            "",
            f"- Estimated physical fits: "
            f"{forecast['estimated_physical_fits']}",
            f"- Guarded runtime: {forecast['estimated_gpu_hours']:.2f} GPU h",
            f"- Hard cap exceeded: {forecast['hard_cap_exceeded']}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpret a completed seed-42 estimator screen.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--model-policy", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--estimator-budgets",
        type=int,
        nargs="+",
        help="budget prefix used by the screen run",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_seeds = read_frozen_run_seeds(args.run_root)
    if args.seed not in run_seeds:
        raise ValueError("requested screen seed is outside the run shard")
    protocol = load_resolved_protocol(
        args.config_dir,
        seeds=run_seeds,
        backend=args.backend,
        smoke=False,
        model_policy_path=args.model_policy,
        estimator_budgets_override=(
            None
            if args.estimator_budgets is None
            else tuple(args.estimator_budgets)
        ),
    )
    rows = read_rich_data_manifest(
        args.run_root / "data_manifest.csv", protocol
    )
    cards = [
        _screen_dataset(
            args.run_root,
            protocol,
            dataset,
            next(
                row
                for row in rows
                if row["dataset_id"] == dataset.dataset_id
                and row["seed"] == args.seed
            ),
            seed=args.seed,
        )
        for dataset in protocol.datasets
    ]
    grades: dict[str, int] = {}
    for card in cards:
        grade = str(card["conclusion_grade"])
        grades[grade] = grades.get(grade, 0) + 1
    estimated_fits = sum(
        int(card["group_physical_model_fit_count"]) * 4
        for card in cards
        if card["complete"]
    )
    estimated_s = sum(
        int(card["group_physical_model_fit_count"])
        * 4
        * (
            float(card["group_runtime_s"])
            / max(int(card["group_physical_model_fit_count"]), 1)
        )
        * 1.30
        for card in cards
        if card["complete"]
    )
    runtime_forecast = {
        "remaining_seed_count": 4,
        "guard_band": 1.30,
        "estimated_physical_fits": estimated_fits,
        "estimated_s": estimated_s,
        "estimated_gpu_hours": estimated_s / 3600.0,
        "soft_cap_gpu_hours": 72,
        "hard_cap_gpu_hours": 96,
        "hard_cap_exceeded": estimated_s > 96 * 3600,
    }
    report = {
        "schema_version": 1,
        "provisional": True,
        "validation_only": True,
        "seed": args.seed,
        "model_policy_digest": protocol.model_policy_digest,
        "datasets": cards,
        "grade_counts": grades,
        "runtime_forecast": runtime_forecast,
    }
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.run_root / "screening"
    )
    atomic_write_json(output_dir / "screen_report.json", report)
    _atomic_write_text(
        output_dir / "screen_report.md", _report_markdown(report)
    )
    for card in cards:
        atomic_write_json(
            output_dir / "claim_cards" / f"{card['dataset_id']}.json",
            card,
        )
    print(f"[screen] report={output_dir / 'screen_report.md'}", flush=True)
    return 2 if runtime_forecast["hard_cap_exceeded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
