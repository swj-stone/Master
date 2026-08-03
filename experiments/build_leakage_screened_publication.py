"""Build a six-dataset paper view after a registered integrity exclusion.

The completed seven-dataset campaign is retained unchanged as the historical
experimental record.  This command validates that frozen summary, reproduces
the Communities & Crime target-construction leakage, excludes that dataset,
re-aggregates all statistics from canonical rows, and publishes a hash-bound
paper table and figure view.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from minppi_mofs.experiment_protocol import atomic_write_json, read_canonical_json

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from experiments.icomp2026_stages import (
        _figure_publication_payloads,
        _publish_bundle,
        _summary_publication_payloads,
        _validate_figure_bundle,
        _validate_summary_bundle,
    )
    from experiments.make_paper_figures import generate_all_figures
    from experiments.summarize_icomp2026 import (
        aggregate_canonical_records,
        generate_latex_macros,
    )
    from experiments.summarize_multifidelity_campaign import analyze_removal_ranks
except ModuleNotFoundError:  # Direct execution from the experiments directory.
    from icomp2026_stages import (  # type: ignore[no-redef]
        _figure_publication_payloads,
        _publish_bundle,
        _summary_publication_payloads,
        _validate_figure_bundle,
        _validate_summary_bundle,
    )
    from make_paper_figures import generate_all_figures  # type: ignore[no-redef]
    from summarize_icomp2026 import (  # type: ignore[no-redef]
        aggregate_canonical_records,
        generate_latex_macros,
    )
    from summarize_multifidelity_campaign import (  # type: ignore[no-redef]
        analyze_removal_ranks,
    )


RETAINED_DATASETS = (
    "spambase",
    "madelon",
    "isolet",
    "gisette",
    "credit_card_fraud",
    "abalone",
)
EXCLUDED_DATASET_ID = "crimedata"
_SOURCE_DATASETS = frozenset((*RETAINED_DATASETS, EXCLUDED_DATASET_ID))
_TARGET = "nonViolPerPop"
_COMPONENT_FEATURES = (
    "burglPerPop",
    "larcPerPop",
    "autoTheftPerPop",
    "arsonsPerPop",
)
_OUTCOME_ROLE_FEATURES = (
    "murders",
    "murdPerPop",
    "rapes",
    "rapesPerPop",
    "robberies",
    "robbbPerPop",
    "assaults",
    "assaultPerPop",
    "ViolentCrimesPerPop",
    "burglaries",
    "burglPerPop",
    "larcenies",
    "larcPerPop",
    "autoTheft",
    "autoTheftPerPop",
    "arsons",
    "arsonsPerPop",
)
_GLOBALLY_SUPERVISED_FEATURES = ("state_encoded",)
_SUMMARY_FILENAMES = {
    "raw": "raw_metrics.csv",
    "aggregate": "aggregate_metrics.csv",
    "paired": "paired_comparisons.csv",
    "front": "plot_sources.csv",
}


def audit_crimedata_target_construction(
    frame: pd.DataFrame,
    *,
    tolerance: float = 0.1,
    minimum_fraction: float = 0.95,
) -> dict[str, object]:
    """Reproduce the documented reason for excluding Communities & Crime."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("crimedata audit requires a nonempty data frame")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or tolerance < 0.0
    ):
        raise ValueError("tolerance must be a finite nonnegative number")
    if (
        isinstance(minimum_fraction, bool)
        or not isinstance(minimum_fraction, (int, float))
        or not math.isfinite(float(minimum_fraction))
        or not 0.0 < minimum_fraction <= 1.0
    ):
        raise ValueError("minimum_fraction must lie in (0, 1]")
    required = (
        _TARGET,
        *_OUTCOME_ROLE_FEATURES,
        *_GLOBALLY_SUPERVISED_FEATURES,
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"crimedata audit is missing columns: {missing}")
    numeric = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("crimedata audit columns must be finite numeric")
    target = numeric[_TARGET].to_numpy(dtype=float)
    constructed = numeric.loc[:, _COMPONENT_FEATURES].sum(axis=1).to_numpy(dtype=float)
    absolute_error = np.abs(target - constructed)
    matched = int(np.count_nonzero(absolute_error <= float(tolerance)))
    fraction = float(matched / len(frame))
    if fraction < float(minimum_fraction):
        raise ValueError("registered target-construction leakage was not reproduced")
    target_mean = float(np.mean(target))
    centered_sum_squares = float(np.sum(np.square(target - target_mean)))
    if centered_sum_squares <= 0.0:
        raise ValueError("crimedata audit target must have nonzero variance")
    correlation = float(np.corrcoef(target, constructed)[0, 1])
    reconstruction_r_squared = float(
        1.0 - np.sum(np.square(target - constructed)) / centered_sum_squares
    )
    if not math.isfinite(correlation) or not math.isfinite(reconstruction_r_squared):
        raise ValueError("crimedata reconstruction diagnostics must be finite")
    return {
        "dataset_id": EXCLUDED_DATASET_ID,
        "reason_code": "outcome_component_leakage",
        "post_hoc": True,
        "target": _TARGET,
        "component_features": list(_COMPONENT_FEATURES),
        "outcome_role_features": list(_OUTCOME_ROLE_FEATURES),
        "globally_supervised_features": list(_GLOBALLY_SUPERVISED_FEATURES),
        "row_count": int(len(frame)),
        "matched_row_count": matched,
        "match_fraction": fraction,
        "absolute_tolerance": float(tolerance),
        "median_absolute_error": float(np.median(absolute_error)),
        "construction_correlation": correlation,
        "construction_r_squared": reconstruction_r_squared,
        "target_global_mean": target_mean,
        "target_global_mean_match_count": int(
            np.count_nonzero(np.isclose(target, target_mean, rtol=0.0, atol=1e-12))
        ),
    }


def _copied_rows(
    rows: Sequence[Mapping[str, object]], *, context: str
) -> list[dict[str, object]]:
    if isinstance(rows, (str, bytes)) or not rows:
        raise ValueError(f"{context} rows must be a nonempty sequence")
    copied: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{context} row must be a mapping")
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"{context} row dataset_id is invalid")
        copied.append(dict(row))
    return copied


def filter_registered_exclusion(
    raw_rows: Sequence[Mapping[str, object]],
    front_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return copies restricted to the exact leakage-screened scope."""
    raw = _copied_rows(raw_rows, context="raw")
    front = _copied_rows(front_rows, context="front")
    if {str(row["dataset_id"]) for row in raw} != _SOURCE_DATASETS or {
        str(row["dataset_id"]) for row in front
    } != _SOURCE_DATASETS:
        raise ValueError("publication filtering requires the exact registered seven-dataset source scope")
    retained_raw = [
        row for row in raw if row["dataset_id"] != EXCLUDED_DATASET_ID
    ]
    retained_front = [
        row for row in front if row["dataset_id"] != EXCLUDED_DATASET_ID
    ]
    expected = set(RETAINED_DATASETS)
    if {str(row["dataset_id"]) for row in retained_raw} != expected or {
        str(row["dataset_id"]) for row in retained_front
    } != expected:
        raise ValueError("leakage-screened publication scope is incomplete")
    return retained_raw, retained_front


def _read_rows(path: Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"publication source is not a regular file: {source}")
    frame = pd.read_csv(source, keep_default_na=False)
    return frame.to_dict(orient="records")


def build_publication_view(
    campaign_root: Path,
    *,
    project_root: Path,
) -> dict[str, object]:
    """Publish filtered tables and figures without mutating canonical results."""
    campaign = Path(campaign_root).resolve()
    project = Path(project_root).resolve()
    if not campaign.is_dir() or campaign.is_symlink():
        raise ValueError("campaign root must be a regular directory")
    summary_path = campaign / "campaign_summary.json"
    summary = read_canonical_json(summary_path, context="campaign summary")
    if not isinstance(summary, dict) or summary.get("publication_allowed") is not True:
        raise ValueError("campaign summary does not allow publication")
    source_digest = summary.get("summary_bundle_manifest_sha256")
    if not isinstance(source_digest, str):
        raise ValueError("campaign summary has no source summary digest")
    source_bundle = campaign / "tables" / "summary"
    _validate_summary_bundle(
        source_bundle,
        expected_manifest_sha256=source_digest,
        expected_run_kind="formal",
    )

    crimedata_path = project / "data" / "raw" / "crimedata.csv"
    if not crimedata_path.is_file() or crimedata_path.is_symlink():
        raise ValueError("registered crimedata snapshot is unavailable")
    leakage_audit = audit_crimedata_target_construction(
        pd.read_csv(crimedata_path), tolerance=0.1, minimum_fraction=0.95
    )

    raw_rows = _read_rows(source_bundle / _SUMMARY_FILENAMES["raw"])
    front_rows = _read_rows(source_bundle / _SUMMARY_FILENAMES["front"])
    retained_raw, retained_front = filter_registered_exclusion(raw_rows, front_rows)
    tables = aggregate_canonical_records(retained_raw, retained_front)
    ranking = analyze_removal_ranks(tables["raw"])
    macros, provenance = generate_latex_macros(
        tables["aggregate"], tables["paired"], run_kind="formal"
    )
    provenance = {
        **provenance,
        "publication_scope": {
            "source_summary_bundle_manifest_sha256": source_digest,
            "retained_datasets": list(RETAINED_DATASETS),
            "excluded_datasets": [leakage_audit],
            "removal_ranking": ranking,
        },
    }
    table_payloads, publication_digest = _summary_publication_payloads(
        tables,
        macros=macros,
        provenance=provenance,
        run_kind="formal",
        skipped_jobs=(),
        publication_allowed=True,
    )
    publication_bundle = _publish_bundle(
        campaign / "tables", "publication", table_payloads
    )
    _validate_summary_bundle(
        publication_bundle,
        expected_manifest_sha256=publication_digest,
        expected_run_kind="formal",
    )

    staging = campaign / "figures" / f".paper-publication-{uuid4().hex}.tmp"
    staging.mkdir(parents=True)
    try:
        generated = generate_all_figures(
            publication_bundle / _SUMMARY_FILENAMES["raw"],
            publication_bundle / _SUMMARY_FILENAMES["aggregate"],
            publication_bundle / _SUMMARY_FILENAMES["paired"],
            publication_bundle / _SUMMARY_FILENAMES["front"],
            staging,
            run_kind="formal",
        )
        figure_payloads, figure_digest = _figure_publication_payloads(generated)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    figure_bundle = _publish_bundle(
        campaign / "figures", "paper", figure_payloads
    )
    _validate_figure_bundle(
        figure_bundle, expected_manifest_sha256=figure_digest
    )

    summary["figure_bundle_manifest_sha256"] = figure_digest
    summary["publication_view"] = {
        "schema_version": 1,
        "source_summary_bundle_manifest_sha256": source_digest,
        "publication_summary_bundle_manifest_sha256": publication_digest,
        "retained_datasets": list(RETAINED_DATASETS),
        "excluded_datasets": [leakage_audit],
        "row_counts": {
            name: len(tables[name])
            for name in ("raw", "aggregate", "paired", "front")
        },
    }
    atomic_write_json(summary_path, summary)
    return {
        "publication_summary_bundle_manifest_sha256": publication_digest,
        "figure_bundle_manifest_sha256": figure_digest,
        "retained_datasets": list(RETAINED_DATASETS),
        "excluded_datasets": [leakage_audit],
        "row_counts": summary["publication_view"]["row_counts"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_publication_view(
        args.campaign_root, project_root=args.project_root
    )
    print(f"retained datasets: {len(report['retained_datasets'])}")
    print(
        "publication summary manifest: "
        f"{report['publication_summary_bundle_manifest_sha256']}"
    )
    print(f"figure manifest: {report['figure_bundle_manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
