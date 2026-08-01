"""Print compact all-stage or two-group ICOMP 2026 campaign snapshots."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence


_STATES = ("pending", "running", "success", "failed", "skipped")
_STAGE_GROUPS = {
    "all": None,
    "initial": frozenset({"validate", "preprocess", "estimator"}),
    "downstream": frozenset({"removal", "alpha", "nsga2", "ablation"}),
}
MAX_INTERVAL_SECONDS = 86400


def _positive_finite_integer(value: str) -> int:
    try:
        interval = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a positive finite integer"
        ) from exc
    if not 1 <= interval <= MAX_INTERVAL_SECONDS:
        raise argparse.ArgumentTypeError("must be a positive finite integer")
    return interval


def summarize_status_document(
    value: Mapping[str, object],
) -> dict[str, str]:
    """Format validated stage counters from one root status document."""
    stages = value.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("status document stages are invalid")
    result: dict[str, str] = {}
    for stage, raw_counts in stages.items():
        if not isinstance(stage, str) or not isinstance(raw_counts, Mapping):
            raise ValueError("status stage counters are invalid")
        counts: dict[str, int] = {}
        for state in _STATES:
            raw = raw_counts.get(state)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ValueError("status stage counters are invalid")
            counts[state] = raw
        total = sum(counts.values())
        text = f"{counts['success']}/{total} success"
        if counts["running"]:
            text += f", {counts['running']} running"
        if counts["failed"]:
            text += f", {counts['failed']} failed"
        if counts["pending"]:
            text += f", {counts['pending']} pending"
        if counts["skipped"]:
            text += f", {counts['skipped']} skipped"
        result[stage] = text
    return result


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}") from exc


def _latest_progress(run_root: Path) -> dict[str, object] | None:
    paths = tuple(
        (run_root / "estimator" / "_bundles").glob("*/*/progress.json")
    )
    if not paths:
        return None
    latest = max(paths, key=lambda path: path.stat().st_mtime_ns)
    value = _read_json(latest)
    if not isinstance(value, dict):
        raise ValueError(f"progress document is invalid: {latest}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show current ICOMP 2026 campaign progress.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/icomp2026_48h"),
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--group",
        choices=tuple(_STAGE_GROUPS),
        default="all",
        help=(
            "stage view: initial=validate/preprocess/estimator; "
            "downstream=held-out-test stages plus publication status"
        ),
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--interval-seconds", type=_positive_finite_integer, default=300
    )
    return parser


def _formal_figure_bundle_ready(campaign_root: Path) -> bool:
    project_root = Path(__file__).resolve().parents[1]
    validator = project_root / "conference-paper" / "sync_figures.py"
    source = campaign_root / "figures" / "paper"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(validator),
                "--source",
                str(source),
                "--check",
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _print_campaign_level_status(campaign_root: Path, group: str) -> bool:
    if group == "initial":
        state = (
            "available"
            if (campaign_root / "campaign_index.json").is_file()
            else "not available"
        )
        print(f"campaign-index: {state}")
        screening_state = (
            "available"
            if (campaign_root / "screening" / "screen_report.json").is_file()
            else "not available"
        )
        print(f"screening-report: {screening_state}")
        return False
    if group != "downstream":
        return False
    summary_path = campaign_root / "campaign_summary.json"
    if not summary_path.is_file():
        print("campaign-summary: not available")
        return False
    value = _read_json(summary_path)
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or type(value.get("publication_allowed")) is not bool
    ):
        raise ValueError(f"campaign summary is invalid: {summary_path}")
    publication_allowed = value["publication_allowed"]
    figures_ready = (
        publication_allowed and _formal_figure_bundle_ready(campaign_root)
    )
    print(
        "campaign-summary: "
        f"publication_allowed={str(publication_allowed).lower()}, "
        f"figures={'ready' if figures_ready else 'not ready'}"
    )
    return not publication_allowed or not figures_ready


def _snapshot_campaign(
    output_root: Path, campaign_id: str, *, group: str = "all"
) -> int:
    if group not in _STAGE_GROUPS:
        raise ValueError(f"unsupported monitor group: {group}")
    campaign_root = output_root / campaign_id
    plan = _read_json(campaign_root / "campaign_plan.json")
    if not isinstance(plan, dict) or not isinstance(plan.get("run_ids"), list):
        raise ValueError("campaign plan is invalid")
    any_failure = False
    for raw_run_id in plan["run_ids"]:
        if not isinstance(raw_run_id, str) or not raw_run_id:
            raise ValueError("campaign run ID is invalid")
        run_root = output_root / raw_run_id
        status_path = run_root / "status.json"
        if not status_path.is_file():
            print(f"{raw_run_id}: not started")
            continue
        status = _read_json(status_path)
        if not isinstance(status, dict):
            raise ValueError(f"status document is invalid: {status_path}")
        summaries = summarize_status_document(status)
        print(f"{raw_run_id}:")
        for stage, summary in summaries.items():
            selected_stages = _STAGE_GROUPS[group]
            if selected_stages is not None and stage not in selected_stages:
                continue
            print(f"  {stage}: {summary}")
            any_failure = any_failure or " failed" in summary
        progress = (
            _latest_progress(run_root)
            if group in {"all", "initial"}
            else None
        )
        if progress is not None:
            eta = progress.get("eta_s")
            eta_text = (
                "?"
                if not isinstance(eta, (int, float))
                else f"{float(eta) / 3600.0:.2f}h"
            )
            print(
                "  latest: "
                f"{progress.get('dataset_id')} seed={progress.get('seed')} "
                f"{progress.get('algorithm')} "
                f"feature={progress.get('feature_index')}/"
                f"{progress.get('feature_count')} ETA={eta_text}"
            )
    any_failure = _print_campaign_level_status(campaign_root, group) or any_failure
    return 1 if any_failure else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    sleep: Callable[[int], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    if not args.watch:
        exit_code = _snapshot_campaign(
            args.output_root, args.campaign_id, group=args.group
        )
        sys.stdout.flush()
        return exit_code
    snapshot_exit_code = 0
    try:
        while True:
            snapshot_exit_code = _snapshot_campaign(
                args.output_root, args.campaign_id, group=args.group
            )
            sys.stdout.flush()
            sleep(args.interval_seconds)
    except KeyboardInterrupt:
        return snapshot_exit_code or 130


if __name__ == "__main__":
    raise SystemExit(main())
