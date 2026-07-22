"""Authoritative command-line controller for the ICOMP 2026 pipeline."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


_CLI_STAGES = (
    "validate",
    "preprocess",
    "estimator",
    "removal",
    "alpha",
    "nsga2",
    "ablation",
    "summarize",
    "figures",
)


def _standard_library_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run manifest-authoritative ICOMP 2026 stages sequentially.",
        allow_abbrev=False,
    )
    parser.add_argument("--stage", required=True, choices=("all",) + _CLI_STAGES)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/icomp2026_rerun")
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-jobs", type=int)
    return parser


if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv[1:]):
    _standard_library_parser().parse_args()


from minppi_mofs.experiment_protocol import (  # noqa: E402
    AuditLog,
    IncompleteRunError,
    ManifestStore,
    StatusStore,
    assert_stage_complete,
    build_job_manifest,
    create_new_run,
    freeze_run_metadata,
    load_resolved_protocol,
    open_resumed_run,
    read_data_manifest,
)

if __package__:
    from . import icomp2026_stages as stage_module
else:
    import icomp2026_stages as stage_module  # type: ignore[no-redef]

STAGE_DEPENDENCIES = stage_module.STAGE_DEPENDENCIES
StageContext = stage_module.StageContext
StageExecutor = stage_module.StageExecutor
StageRunSummary = stage_module.StageRunSummary
execute_stage = stage_module.execute_stage


STAGE_ORDER = tuple(STAGE_DEPENDENCIES)


class PipelineRunner:
    """Incrementally freeze and synchronously dispatch requested stages."""

    def __init__(
        self,
        protocol,
        layout,
        *,
        executors: Mapping[str, StageExecutor] | None = None,
        stop_after_jobs: int | None = None,
        allow_manifest_initialize: bool = False,
    ) -> None:
        if type(allow_manifest_initialize) is not bool:
            raise TypeError("allow_manifest_initialize must be a bool")
        if stop_after_jobs is not None and (
            type(stop_after_jobs) is not int or stop_after_jobs < 1
        ):
            raise ValueError("stop_after_jobs must be a positive integer")
        self.protocol = protocol
        self.layout = layout
        self.executors = {} if executors is None else dict(executors)
        self.stop_after_jobs = stop_after_jobs
        self._allow_manifest_initialize = allow_manifest_initialize
        if stop_after_jobs is not None and protocol.run_kind != "smoke":
            raise ValueError("stop_after_jobs is allowed only for a smoke protocol")
        self._successful_writes = 0
        self.manifests = ManifestStore(layout.manifest_path)

    def run(self, requested_stage: str) -> int:
        if requested_stage != "all" and requested_stage not in STAGE_ORDER:
            raise ValueError(f"unknown stage: {requested_stage}")
        stages = STAGE_ORDER if requested_stage == "all" else (requested_stage,)
        exit_code = 0
        for stage in stages:
            outcome = self.execute(stage)
            if outcome is None:
                continue
            code = outcome if isinstance(outcome, int) else outcome.exit_code
            if code == 75:
                return 75
            if code != 0:
                exit_code = 1
        return exit_code

    def execute(self, stage: str) -> StageRunSummary:
        manifest, issue = self._manifest_for_stage(stage)
        if issue is not None:
            return StageRunSummary(stage=stage, dependency_issues=(issue,))
        status = StatusStore(self.layout, manifest)
        audit = AuditLog.from_manifest(self.layout.audit_path, manifest)
        contexts = tuple(
            StageContext(self.protocol, self.layout, job, status, audit)
            for job in manifest
        )
        remaining = None
        if self.stop_after_jobs is not None:
            remaining = self.stop_after_jobs - self._successful_writes
            if remaining < 1:
                return StageRunSummary(stage=stage, stopped_early=True)
        summary = execute_stage(
            stage,
            context=contexts,
            executors=self.executors,
            stop_after_jobs=remaining,
        )
        self._successful_writes += len(summary.succeeded_job_ids)
        return summary

    def _manifest_for_stage(self, stage: str):
        if self.manifests.path.exists():
            manifest = self.manifests.load()
        else:
            if not self._allow_manifest_initialize:
                raise RuntimeError(
                    "durable manifest is missing and this runner has no new-run "
                    "initialization authority"
                )
            fixed_status_paths = (
                self.layout.status_path,
                *(
                    self.layout.root / declared_stage / "status.json"
                    for declared_stage in STAGE_ORDER
                ),
            )
            if any(path.exists() for path in fixed_status_paths):
                raise RuntimeError(
                    "manifest initialization is forbidden after run status exists"
                )
            manifest = ()
        existing_stages = {job.stage for job in manifest}
        if stage in existing_stages:
            return manifest, None
        requested_index = STAGE_ORDER.index(stage)
        missing_stages = tuple(
            candidate
            for candidate in STAGE_ORDER[: requested_index + 1]
            if candidate not in existing_stages
        )
        data_rows = None
        if any(
            STAGE_ORDER.index(candidate) > STAGE_ORDER.index("preprocess")
            for candidate in missing_stages
        ):
            try:
                data_rows = read_data_manifest(
                    self.layout.root / "data_manifest.csv", self.protocol
                )
            except ValueError as exc:
                return manifest, str(exc)
        planned_jobs = []
        for candidate in missing_stages:
            dependencies = STAGE_DEPENDENCIES[candidate]
            if dependencies:
                if not manifest:
                    return (), f"{candidate}: prerequisite manifest is missing"
                try:
                    assert_stage_complete(self.layout, manifest, dependencies)
                except IncompleteRunError as exc:
                    return manifest, str(exc)
            planned_jobs.extend(
                build_job_manifest(
                    self.protocol,
                    stages=(candidate,),
                    data_manifest=data_rows,
                )
            )
        if manifest:
            manifest = self.manifests.append(tuple(planned_jobs))
        else:
            manifest = self.manifests.initialize(tuple(planned_jobs))
            self._allow_manifest_initialize = False
        return manifest, None


def build_parser() -> argparse.ArgumentParser:
    return _standard_library_parser()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.run_id is None:
        parser.error("--resume requires --run-id")
    if args.stop_after_jobs is not None and args.stop_after_jobs < 1:
        parser.error("--stop-after-jobs must be a positive integer")
    if args.stop_after_jobs is not None and not args.smoke:
        parser.error("--stop-after-jobs is allowed only with --smoke")

    project_root = Path(__file__).resolve().parents[1]
    protocol = load_resolved_protocol(
        args.config_dir,
        seeds=None if args.seeds is None else tuple(args.seeds),
        backend=args.backend,
        smoke=args.smoke,
    )
    if args.resume:
        layout = open_resumed_run(
            args.output_root,
            args.run_id,
            protocol,
            project_root=project_root,
        )
    else:
        layout = create_new_run(args.output_root, args.run_id, protocol)
        freeze_run_metadata(layout, protocol, project_root=project_root)
    _append_log(layout.log_path, f"start stage={args.stage} resume={args.resume}")
    try:
        result = PipelineRunner(
            protocol,
            layout,
            executors=stage_module.STAGE_EXECUTORS,
            stop_after_jobs=args.stop_after_jobs,
            allow_manifest_initialize=not args.resume,
        ).run(args.stage)
    except (OSError, RuntimeError, ValueError) as exc:
        _append_log(layout.log_path, f"error {type(exc).__name__}: {exc}")
        return 2
    _append_log(layout.log_path, f"finish stage={args.stage} exit_code={result}")
    return result


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{timestamp} {message}\n")


if __name__ == "__main__":
    raise SystemExit(main())
