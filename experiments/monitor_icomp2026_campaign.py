"""Print a compact snapshot of all three ICOMP 2026 campaign shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


_STATES = ("pending", "running", "success", "failed", "skipped")


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign_root = args.output_root / args.campaign_id
    plan = _read_json(campaign_root / "campaign_plan.json")
    if not isinstance(plan, dict) or not isinstance(plan.get("run_ids"), list):
        raise ValueError("campaign plan is invalid")
    any_failure = False
    for raw_run_id in plan["run_ids"]:
        if not isinstance(raw_run_id, str) or not raw_run_id:
            raise ValueError("campaign run ID is invalid")
        run_root = args.output_root / raw_run_id
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
            print(f"  {stage}: {summary}")
            any_failure = any_failure or " failed" in summary
        progress = _latest_progress(run_root)
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
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
