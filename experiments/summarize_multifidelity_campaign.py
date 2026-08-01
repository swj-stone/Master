"""Validate seed-sharded ICOMP 2026 campaign provenance without copying data."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence
from uuid import uuid4

import yaml

from minppi_mofs.experiment_protocol import (
    AuditLog,
    ManifestStore,
    StatusStore,
    assert_stage_complete,
    atomic_write_json,
    build_job_manifest,
    canonical_hash,
    load_resolved_protocol,
    open_resumed_run,
    read_canonical_json,
    read_data_manifest,
    scientific_environment_digest,
)


_EXPECTED_SEEDS = frozenset({42, 123, 456, 789, 1024})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HARD_P0_SECONDS = 48 * 60 * 60


@dataclass(frozen=True)
class ShardRecord:
    """Immutable provenance summary for one seed shard."""

    root: Path
    protocol_hash: str
    seeds: tuple[int, ...]
    model_policy_digest: str
    evidence_digest: str
    source_files_digest: str
    raw_authority_digest: str
    scientific_environment_digest: str
    complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("root must be a Path")
        for field in (
            "protocol_hash",
            "model_policy_digest",
            "evidence_digest",
            "source_files_digest",
            "raw_authority_digest",
            "scientific_environment_digest",
        ):
            if not isinstance(getattr(self, field), str) or not _SHA256.fullmatch(
                getattr(self, field)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.seeds
            )
        ):
            raise ValueError("seeds must be unique non-negative integers")
        if type(self.complete) is not bool:
            raise TypeError("complete must be a bool")


@dataclass(frozen=True)
class RuntimeTier:
    """One measured-fit runtime contribution assigned to a priority tier."""

    priority: str
    estimated_physical_fits: int
    measured_seconds_per_fit: float

    def __post_init__(self) -> None:
        if self.priority not in {"P0", "P1", "P2"}:
            raise ValueError("priority must be P0, P1, or P2")
        if (
            isinstance(self.estimated_physical_fits, bool)
            or not isinstance(self.estimated_physical_fits, int)
            or self.estimated_physical_fits < 0
        ):
            raise ValueError("estimated_physical_fits must be non-negative")
        if (
            isinstance(self.measured_seconds_per_fit, bool)
            or not isinstance(self.measured_seconds_per_fit, (int, float))
            or not math.isfinite(float(self.measured_seconds_per_fit))
            or self.measured_seconds_per_fit < 0
        ):
            raise ValueError("measured_seconds_per_fit must be non-negative")


def forecast_runtime(
    tiers: Sequence[RuntimeTier],
) -> dict[str, dict[str, float | int]]:
    """Apply the frozen 30% guard band and reject a P0 forecast above 48 h."""
    totals = {
        priority: {
            "estimated_physical_fits": 0,
            "estimated_s": 0.0,
            "estimated_gpu_hours": 0.0,
        }
        for priority in ("P0", "P1", "P2")
    }
    for tier in tiers:
        if not isinstance(tier, RuntimeTier):
            raise TypeError("tiers must contain RuntimeTier values")
        estimated_s = (
            tier.estimated_physical_fits
            * float(tier.measured_seconds_per_fit)
            * 1.30
        )
        bucket = totals[tier.priority]
        bucket["estimated_physical_fits"] += tier.estimated_physical_fits
        bucket["estimated_s"] += estimated_s
        bucket["estimated_gpu_hours"] = bucket["estimated_s"] / 3600.0
    if totals["P0"]["estimated_s"] > _HARD_P0_SECONDS:
        raise ValueError("P0 runtime forecast exceeds the 48 hours hard cap")
    return totals


def validate_campaign_shards(
    shards: Sequence[ShardRecord],
    *,
    expected_seeds: frozenset[int] = _EXPECTED_SEEDS,
) -> tuple[ShardRecord, ...]:
    """Reject overlap or any scientific/source mismatch across seed shards."""
    records = tuple(shards)
    if not records or not all(isinstance(item, ShardRecord) for item in records):
        raise ValueError("campaign shards must be non-empty ShardRecord values")
    observed: set[int] = set()
    for record in records:
        overlap = observed.intersection(record.seeds)
        if overlap:
            raise ValueError(f"campaign shard seed overlap: {sorted(overlap)}")
        observed.update(record.seeds)
    if observed != set(expected_seeds):
        raise ValueError(
            "campaign seed union is incomplete or unexpected: "
            f"{sorted(observed)}"
        )
    if any(not item.complete for item in records):
        raise ValueError("every campaign shard must be complete")
    if len({item.model_policy_digest for item in records}) != 1:
        raise ValueError("campaign shards use different model policy digests")
    if len({item.evidence_digest for item in records}) != 1:
        raise ValueError("campaign shards use different scientific evidence")
    if len({item.source_files_digest for item in records}) != 1:
        raise ValueError("campaign shards use different source fingerprints")
    if len({item.raw_authority_digest for item in records}) != 1:
        raise ValueError("campaign shards use different raw authority digests")
    if len({item.scientific_environment_digest for item in records}) != 1:
        raise ValueError(
            "campaign shards use different scientific environment digests"
        )
    return records


def merge_shard_summary_rows(
    shards: Sequence[
        tuple[
            Sequence[int],
            Sequence[Mapping[str, object]],
            Sequence[Mapping[str, object]],
        ]
    ],
    *,
    expected_seeds: frozenset[int] = _EXPECTED_SEEDS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Merge validated shard rows under one deterministic sequence authority."""
    if isinstance(shards, (str, bytes)) or not shards:
        raise ValueError("summary shards must be a nonempty sequence")
    observed: set[int] = set()
    normalized = []
    for seeds_value, raw_rows, front_rows in shards:
        seeds = tuple(seeds_value)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in seeds
            )
        ):
            raise ValueError("summary shard seeds are invalid")
        overlap = observed.intersection(seeds)
        if overlap:
            raise ValueError(f"summary shard seed overlap: {sorted(overlap)}")
        observed.update(seeds)
        raw = [dict(row) for row in raw_rows]
        front = [dict(row) for row in front_rows]
        if any(row.get("seed") not in seeds for row in (*raw, *front)):
            raise ValueError("summary row is outside its shard seed authority")
        normalized.append((seeds, raw, front))
    if observed != set(expected_seeds):
        raise ValueError(
            "summary shard seed union is incomplete or unexpected: "
            f"{sorted(observed)}"
        )
    combined_raw: list[dict[str, object]] = []
    combined_front: list[dict[str, object]] = []
    next_sequence = 0
    for seeds, raw, front in sorted(
        normalized, key=lambda item: (min(item[0]), item[0])
    ):
        del seeds
        local_sequences = sorted(
            {
                row.get("manifest_sequence")
                for row in raw
                if isinstance(row.get("manifest_sequence"), int)
                and not isinstance(row.get("manifest_sequence"), bool)
            }
        )
        if len(local_sequences) != len(
            {row.get("manifest_sequence") for row in raw}
        ):
            raise ValueError("summary raw manifest sequence is invalid")
        sequence_map = {
            sequence: next_sequence + index
            for index, sequence in enumerate(local_sequences)
        }
        next_sequence += len(local_sequences)
        for row in raw:
            row["manifest_sequence"] = sequence_map[
                row["manifest_sequence"]
            ]
        combined_raw.extend(
            sorted(
                raw,
                key=lambda row: (
                    int(row["manifest_sequence"]),
                    str(row.get("metric", "")),
                    int(row.get("trial", 0)),
                ),
            )
        )
        combined_front.extend(front)
    return combined_raw, combined_front


def build_campaign_tables(
    shards: Sequence[
        tuple[
            Sequence[int],
            Sequence[Mapping[str, object]],
            Sequence[Mapping[str, object]],
        ]
    ],
    *,
    allow_approved_skip_subset: bool = False,
) -> dict[str, list[dict[str, object]]]:
    """Aggregate one canonical five-seed table set from validated shard rows."""
    from experiments.summarize_icomp2026 import aggregate_canonical_records

    raw_rows, front_rows = merge_shard_summary_rows(shards)
    return aggregate_canonical_records(
        raw_rows,
        front_rows,
        allow_approved_skip_subset=allow_approved_skip_subset,
    )


def analyze_removal_ranks(
    raw_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Rank matched removal methods using datasets as independent blocks."""
    import numpy as np
    from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

    from experiments.summarize_icomp2026 import holm_adjust

    methods = (
        "full",
        "minppi",
        "shap_bottom",
        "minshap_bottom",
        "banzhaf_bottom",
        "random_matched",
    )
    selected = [
        row
        for row in raw_rows
        if row.get("run_kind") == "formal"
        and row.get("stage") == "removal"
        and row.get("method") in methods
        and row.get("metric") == "selected_utility"
        and row.get("parameter") == ""
    ]
    if not selected:
        raise ValueError("removal ranking has no selected-utility rows")
    by_dataset: dict[
        str, dict[str, dict[int, list[float]]]
    ] = {}
    for row in selected:
        dataset = row.get("dataset_id")
        method = row.get("method")
        seed = row.get("seed")
        value = row.get("value")
        if (
            not isinstance(dataset, str)
            or method not in methods
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("removal ranking row is invalid")
        by_dataset.setdefault(
            dataset, {name: {} for name in methods}
        )[str(method)].setdefault(seed, []).append(float(value))
    if len(by_dataset) < 2:
        raise ValueError("removal ranking requires at least two datasets")
    dataset_values: dict[str, dict[str, float]] = {}
    for dataset, method_rows in by_dataset.items():
        if set(method_rows) != set(methods):
            raise ValueError("removal ranking method scope is incomplete")
        dataset_values[dataset] = {}
        for method in methods:
            seed_rows = method_rows[method]
            if set(seed_rows) != set(_EXPECTED_SEEDS):
                raise ValueError(
                    "removal ranking requires five seeds per dataset"
                )
            dataset_values[dataset][method] = float(
                np.mean(
                    [
                        float(np.mean(seed_rows[seed]))
                        for seed in sorted(seed_rows)
                    ]
                )
            )
    datasets = sorted(dataset_values)
    ranks = {
        method: [] for method in methods
    }
    for dataset in datasets:
        values = np.asarray(
            [dataset_values[dataset][method] for method in methods],
            dtype=float,
        )
        row_ranks = rankdata(-values, method="average")
        for method, rank in zip(methods, row_ranks, strict=True):
            ranks[method].append(float(rank))
    samples = [
        np.asarray(
            [dataset_values[dataset][method] for dataset in datasets],
            dtype=float,
        )
        for method in methods
    ]
    if all(np.array_equal(samples[0], sample) for sample in samples[1:]):
        friedman_statistic, friedman_p = 0.0, 1.0
    else:
        result = friedmanchisquare(*samples)
        friedman_statistic = float(result.statistic)
        friedman_p = float(result.pvalue)
    method_count = len(methods)
    dataset_count = len(datasets)
    q_alpha = float(
        studentized_range.ppf(0.95, method_count, np.inf)
        / np.sqrt(2.0)
    )
    critical_difference = q_alpha * math.sqrt(
        method_count * (method_count + 1) / (6.0 * dataset_count)
    )
    minppi = samples[methods.index("minppi")]
    pairwise: list[dict[str, object]] = []
    for method, sample in zip(methods, samples, strict=True):
        if method == "minppi":
            continue
        differences = minppi - sample
        if np.all(differences == 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        pairwise.append(
            {
                "baseline_method": method,
                "comparison_method": "minppi",
                "dataset_count": dataset_count,
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "holm_p_value": 0.0,
            }
        )
    adjusted = holm_adjust(
        [float(row["p_value"]) for row in pairwise]
    )
    for row, value in zip(pairwise, adjusted, strict=True):
        row["holm_p_value"] = value
    return {
        "schema_version": 1,
        "block_unit": "dataset_mean_over_five_seeds",
        "dataset_count": dataset_count,
        "seed_count_per_dataset": len(_EXPECTED_SEEDS),
        "datasets": datasets,
        "methods": list(methods),
        "mean_utilities": {
            method: float(np.mean(samples[index]))
            for index, method in enumerate(methods)
        },
        "mean_ranks": {
            method: float(np.mean(ranks[method])) for method in methods
        },
        "friedman_statistic": friedman_statistic,
        "friedman_p_value": friedman_p,
        "nemenyi_critical_difference_0_05": critical_difference,
        "minppi_pairwise_wilcoxon_holm": pairwise,
    }


def _ranking_markdown(report: Mapping[str, object]) -> str:
    ranks = report["mean_ranks"]
    if not isinstance(ranks, Mapping):
        raise ValueError("removal ranking report is invalid")
    lines = [
        "# ICOMP 2026 matched-removal ranking",
        "",
        "Statistical block: one mean per dataset over the five frozen seeds. "
        "Random-matched trials are averaged within seed before dataset means.",
        "",
        "| Method | Mean rank (lower is better) |",
        "|---|---:|",
    ]
    for method, value in sorted(
        ranks.items(), key=lambda item: (float(item[1]), str(item[0]))
    ):
        lines.append(f"| {method} | {float(value):.4f} |")
    lines.extend(
        [
            "",
            f"Friedman: statistic={float(report['friedman_statistic']):.6g}, "
            f"p={float(report['friedman_p_value']):.6g}.",
            f"Nemenyi CD (alpha=0.05): "
            f"{float(report['nemenyi_critical_difference_0_05']):.6g}.",
            "",
            "Pairwise MinPPI comparisons use two-sided Wilcoxon signed-rank "
            "tests across datasets with Holm correction.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_shard_summary_rows(
    record: ShardRecord,
    *,
    config_dir: Path,
    model_policy_path: Path,
    project_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Revalidate one shard's complete upstream artifacts without test reads."""
    from experiments.icomp2026_stages import (
        StageContext,
        _SUMMARY_REQUIRED_STAGES,
        _upstream_skip_metadata,
        _validated_summary_rows,
    )

    protocol = load_resolved_protocol(
        config_dir,
        seeds=record.seeds,
        backend="cuda",
        smoke=False,
        model_policy_path=model_policy_path,
    )
    layout = open_resumed_run(
        record.root.parent,
        record.root.name,
        protocol,
        project_root=project_root,
    )
    manifest = ManifestStore(layout.manifest_path).load()
    data_rows = read_data_manifest(layout.root / "data_manifest.csv", protocol)
    for stage in _SUMMARY_REQUIRED_STAGES:
        expected = build_job_manifest(
            protocol,
            stages=(stage,),
            data_manifest=data_rows,
        )
        actual = tuple(job for job in manifest if job.stage == stage)
        if tuple(job.job_id for job in actual) != tuple(
            job.job_id for job in expected
        ):
            raise ValueError(
                f"campaign shard {record.root.name} {stage} manifest differs"
            )
        if expected:
            assert_stage_complete(layout, manifest, (stage,))
    if not manifest:
        raise ValueError("campaign shard manifest is empty")
    status = StatusStore(layout, manifest)
    audit = AuditLog.from_manifest(layout.audit_path, manifest)
    context = StageContext(
        protocol, layout, manifest[0], status, audit
    )
    skipped, omitted = _upstream_skip_metadata(context)
    raw, front = _validated_summary_rows(
        context, omitted_job_ids=omitted
    )
    return raw, front, skipped


def build_campaign_summary(
    roots: Sequence[Path],
    *,
    campaign_root: Path,
    config_dir: Path,
    model_policy_path: Path,
    project_root: Path,
) -> dict[str, object]:
    """Publish one audited five-seed summary and figure bundle."""
    from experiments.icomp2026_stages import (
        _figure_publication_payloads,
        _publish_bundle,
        _summary_publication_payloads,
    )
    from experiments.make_paper_figures import generate_all_figures
    from experiments.summarize_icomp2026 import generate_latex_macros

    records = validate_campaign_shards(
        tuple(read_shard_record(Path(root)) for root in roots)
    )
    shard_rows = []
    skipped_jobs: list[dict[str, object]] = []
    for record in records:
        raw, front, skipped = _read_shard_summary_rows(
            record,
            config_dir=config_dir,
            model_policy_path=model_policy_path,
            project_root=project_root,
        )
        shard_rows.append((record.seeds, raw, front))
        skipped_jobs.extend(skipped)
    publication_allowed = not skipped_jobs
    tables = build_campaign_tables(
        shard_rows,
        allow_approved_skip_subset=not publication_allowed,
    )
    ranking = analyze_removal_ranks(tables["raw"])
    if publication_allowed:
        macros, provenance = generate_latex_macros(
            tables["aggregate"],
            tables["paired"],
            run_kind="formal",
        )
    else:
        macros = ""
        provenance = {
            "schema_version": 1,
            "run_kind": "formal",
            "publication_allowed": False,
            "skipped_jobs": skipped_jobs,
            "macros": {},
        }
    payloads, table_digest = _summary_publication_payloads(
        tables,
        macros=macros,
        provenance=provenance,
        run_kind="formal",
        skipped_jobs=skipped_jobs,
        publication_allowed=publication_allowed,
    )
    destination = Path(campaign_root)
    _publish_bundle(destination / "tables", "summary", payloads)
    ranking_json = (
        json.dumps(
            ranking,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    ranking_payloads = {
        "removal_ranking.json": ranking_json,
        "removal_ranking.md": _ranking_markdown(ranking).encode("utf-8"),
    }
    _publish_bundle(destination / "tables", "ranking", ranking_payloads)
    figure_digest: str | None = None
    if publication_allowed:
        staging = (
            destination / "figures" / f".paper.{uuid4().hex}.tmp"
        )
        staging.mkdir(parents=True)
        try:
            summary = destination / "tables" / "summary"
            generated = generate_all_figures(
                summary / "raw_metrics.csv",
                summary / "aggregate_metrics.csv",
                summary / "paired_comparisons.csv",
                summary / "plot_sources.csv",
                staging,
                run_kind="formal",
            )
            figure_payloads, figure_digest = (
                _figure_publication_payloads(generated)
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        _publish_bundle(
            destination / "figures", "paper", figure_payloads
        )
    report = {
        "schema_version": 1,
        "publication_allowed": publication_allowed,
        "test_partition_read_by_summary": False,
        "seeds": sorted(_EXPECTED_SEEDS),
        "shards": [str(record.root) for record in records],
        "raw_authority_digest": records[0].raw_authority_digest,
        "scientific_environment_digest": (
            records[0].scientific_environment_digest
        ),
        "skipped_jobs": skipped_jobs,
        "row_counts": {
            name: len(tables[name])
            for name in ("raw", "aggregate", "paired", "front")
        },
        "summary_bundle_manifest_sha256": table_digest,
        "removal_ranking_sha256": hashlib.sha256(
            ranking_json
        ).hexdigest(),
        "figure_bundle_manifest_sha256": figure_digest,
    }
    atomic_write_json(destination / "campaign_summary.json", report)
    return report


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"frozen protocol is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"frozen protocol is malformed: {path}")
    return value


def read_shard_record(root: Path) -> ShardRecord:
    """Read one shard's frozen metadata and estimator completion state."""
    shard_root = Path(root)
    marker = read_canonical_json(
        shard_root / "metadata_complete.json",
        context="metadata completion marker",
    )
    if not isinstance(marker, dict) or marker.get("schema_version") != 1:
        raise ValueError("metadata completion marker is invalid")
    protocol = _read_yaml_mapping(shard_root / "resolved_protocol.yaml")
    fingerprint = read_canonical_json(
        shard_root / "source_fingerprint.json",
        context="source fingerprint",
    )
    if not isinstance(fingerprint, dict):
        raise ValueError("source fingerprint is invalid")
    seeds_value = protocol.get("seeds")
    datasets_value = protocol.get("datasets")
    policy_digest = protocol.get("model_policy_digest")
    if (
        not isinstance(seeds_value, list)
        or not isinstance(datasets_value, list)
        or not isinstance(policy_digest, str)
        or not _SHA256.fullmatch(policy_digest)
    ):
        raise ValueError("frozen shard protocol lacks campaign identity")
    dataset_ids: list[str] = []
    expected_raw_authority: dict[str, dict[str, object]] = {}
    for dataset in datasets_value:
        if not isinstance(dataset, dict):
            raise ValueError("frozen shard dataset scope is invalid")
        dataset_id = dataset.get("dataset_id")
        raw_sha256 = dataset.get("raw_sha256")
        raw_byte_count = dataset.get("raw_byte_count")
        if (
            not isinstance(dataset_id, str)
            or not isinstance(raw_sha256, str)
            or _SHA256.fullmatch(raw_sha256) is None
            or isinstance(raw_byte_count, bool)
            or not isinstance(raw_byte_count, int)
            or raw_byte_count <= 0
            or dataset_id in expected_raw_authority
        ):
            raise ValueError("frozen shard dataset scope is invalid")
        dataset_ids.append(dataset_id)
        expected_raw_authority[dataset_id] = {
            "byte_count": raw_byte_count,
            "sha256": raw_sha256,
        }
    seeds = tuple(seeds_value)
    protocol_hash = fingerprint.get("protocol_hash")
    source_files = fingerprint.get("files")
    if (
        not isinstance(protocol_hash, str)
        or not _SHA256.fullmatch(protocol_hash)
        or not isinstance(source_files, dict)
    ):
        raise ValueError("source fingerprint is invalid")
    if canonical_hash(protocol) != protocol_hash:
        raise ValueError("resolved protocol content does not match its protocol hash")
    if marker.get("protocol_hash") != protocol_hash:
        raise ValueError("metadata completion marker protocol hash is invalid")
    artifact_hashes = marker.get("artifacts")
    expected_artifacts = {
        "resolved_protocol.yaml",
        "environment.json",
        "source_fingerprint.json",
        "configs_frozen/icomp2026_pipeline.resolved.yaml",
        *(f"configs_frozen/{dataset_id}.yaml" for dataset_id in dataset_ids),
    }
    if (
        not isinstance(artifact_hashes, dict)
        or set(artifact_hashes) != expected_artifacts
        or any(
            not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
            for digest in artifact_hashes.values()
        )
    ):
        raise ValueError("metadata completion marker artifact set is invalid")
    resolved_shard_root = shard_root.resolve()
    for relative, expected_digest in artifact_hashes.items():
        artifact_path = shard_root / relative
        try:
            artifact_path.resolve().relative_to(resolved_shard_root)
        except ValueError as exc:
            raise ValueError("frozen metadata artifact escapes the shard") from exc
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError(f"frozen metadata artifact is invalid: {relative}")
        actual_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(f"frozen metadata artifact was changed: {relative}")
    raw_authority = read_canonical_json(
        shard_root / "validate" / "raw_authority.json",
        context="raw authority",
    )
    if (
        not isinstance(raw_authority, dict)
        or set(raw_authority) != {"protocol_hash", "datasets"}
        or raw_authority.get("protocol_hash") != protocol_hash
        or raw_authority.get("datasets") != expected_raw_authority
    ):
        raise ValueError("raw authority does not match the frozen protocol")
    raw_authority_digest = _canonical_digest(expected_raw_authority)
    environment = read_canonical_json(
        shard_root / "environment.json",
        context="scientific environment",
    )
    if not isinstance(environment, dict):
        raise ValueError("scientific environment is invalid")
    frozen_environment_digest = environment.get(
        "scientific_environment_sha256"
    )
    try:
        calculated_environment_digest = scientific_environment_digest(
            environment
        )
    except ValueError as exc:
        raise ValueError("scientific environment is invalid") from exc
    if (
        not isinstance(frozen_environment_digest, str)
        or _SHA256.fullmatch(frozen_environment_digest) is None
        or calculated_environment_digest != frozen_environment_digest
        or marker.get("scientific_environment_sha256")
        != frozen_environment_digest
    ):
        raise ValueError("scientific environment digest is invalid")
    evidence = {
        key: value
        for key, value in protocol.items()
        if key not in {"seeds", "run_kind"}
    }
    complete = all(
        (
            shard_root
            / "estimator"
            / "_bundles"
            / dataset_id
            / f"s{seed}"
            / "bundle.json"
        ).is_file()
        for dataset_id in dataset_ids
        for seed in seeds
    )
    return ShardRecord(
        root=shard_root.resolve(),
        protocol_hash=protocol_hash,
        seeds=seeds,
        model_policy_digest=policy_digest,
        evidence_digest=_canonical_digest(evidence),
        source_files_digest=_canonical_digest(source_files),
        raw_authority_digest=raw_authority_digest,
        scientific_environment_digest=frozen_environment_digest,
        complete=complete,
    )


def build_campaign_index(
    roots: Sequence[Path],
) -> dict[str, object]:
    """Build a five-seed index referencing, but never copying, shard artifacts."""
    records = validate_campaign_shards(
        tuple(read_shard_record(Path(root)) for root in roots)
    )
    return {
        "schema_version": 1,
        "seeds": sorted(_EXPECTED_SEEDS),
        "model_policy_digest": records[0].model_policy_digest,
        "evidence_digest": records[0].evidence_digest,
        "source_files_digest": records[0].source_files_digest,
        "raw_authority_digest": records[0].raw_authority_digest,
        "scientific_environment_digest": (
            records[0].scientific_environment_digest
        ),
        "shards": [
            {
                **asdict(record),
                "root": str(record.root),
                "seeds": list(record.seeds),
            }
            for record in records
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and index compatible ICOMP 2026 seed shards.",
        allow_abbrev=False,
    )
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/icomp2026_multifidelity/campaign_index.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = build_campaign_index(args.shard)
    atomic_write_json(args.output, index)
    print(f"[campaign] index={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
