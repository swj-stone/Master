"""Synchronous, manifest-authoritative ICOMP 2026 stage execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from minppi_mofs.experiment_protocol import (
    AuditLog,
    IncompleteRunError,
    JobSpec,
    ResolvedProtocol,
    ResultRecord,
    RunLayout,
    StatusStore,
    assert_stage_complete,
    atomic_write_json,
    resolved_protocol_hash,
)


STAGE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "validate": (),
    "preprocess": ("validate",),
    "estimator": ("preprocess",),
    "removal": ("preprocess",),
    "alpha": ("preprocess",),
    "nsga2": ("preprocess", "removal"),
    "ablation": ("preprocess",),
    "summarize": ("estimator", "removal", "alpha", "nsga2", "ablation"),
    "figures": ("summarize",),
}


@dataclass(frozen=True)
class StageContext:
    """Immutable authority and services for exactly one manifest job."""

    protocol: ResolvedProtocol
    layout: RunLayout
    job: JobSpec
    status: StatusStore
    audit: AuditLog


StageExecutor = Callable[[StageContext], ResultRecord]

# Tasks 7--11 register computational handlers here. Keeping this module-level
# boundary lightweight preserves import-safe CLI help in Task 6.
STAGE_EXECUTORS: dict[str, StageExecutor] = {}


@dataclass(frozen=True)
class StageRunSummary:
    """Outcome of one synchronous stage dispatch."""

    stage: str
    succeeded_job_ids: tuple[str, ...] = ()
    skipped_job_ids: tuple[str, ...] = ()
    failed_job_ids: tuple[str, ...] = ()
    blocked_job_ids: tuple[str, ...] = ()
    dependency_issues: tuple[str, ...] = ()
    stopped_early: bool = False

    @property
    def exit_code(self) -> int:
        if self.stopped_early:
            return 75
        if self.failed_job_ids or self.blocked_job_ids or self.dependency_issues:
            return 1
        return 0


def execute_stage(
    stage: str,
    *,
    context: Sequence[StageContext],
    executors: Mapping[str, StageExecutor],
    stop_after_jobs: int | None = None,
) -> StageRunSummary:
    """Execute one stage sequentially in the supplied manifest order."""
    if stage not in STAGE_DEPENDENCIES:
        raise ValueError(f"unknown stage: {stage}")
    if not isinstance(context, Sequence) or isinstance(context, (str, bytes)):
        raise TypeError("context must be an ordered sequence of StageContext values")
    contexts = tuple(context)
    if not contexts or not all(isinstance(item, StageContext) for item in contexts):
        raise ValueError("context must contain StageContext values")
    if isinstance(stop_after_jobs, bool) or (
        stop_after_jobs is not None
        and (not isinstance(stop_after_jobs, int) or stop_after_jobs < 1)
    ):
        raise ValueError("stop_after_jobs must be a positive integer")

    protocol = contexts[0].protocol
    layout = contexts[0].layout
    status = contexts[0].status
    manifest = tuple(item.job for item in contexts)
    if len({job.job_id for job in manifest}) != len(manifest):
        raise ValueError("context contains duplicate manifest jobs")
    if any(
        item.protocol != protocol or item.layout != layout or item.status is not status
        for item in contexts
    ):
        raise ValueError("all contexts must share one protocol, layout, and status store")
    if layout != status.layout:
        raise ValueError("context layout does not match StatusStore run authority")
    authority = status.manifest
    if manifest != authority:
        raise ValueError("context jobs do not match StatusStore manifest authority")
    if (
        resolved_protocol_hash(protocol) != status.protocol_hash
        or any(job.protocol_hash != status.protocol_hash for job in authority)
    ):
        raise ValueError("context protocol does not match StatusStore authority")
    if stop_after_jobs is not None and protocol.run_kind != "smoke":
        raise ValueError("stop_after_jobs is allowed only for a smoke protocol")
    manifest = authority
    stage_jobs = tuple(job for job in manifest if job.stage == stage)
    if not stage_jobs:
        raise ValueError(f"manifest contains no jobs for stage {stage}")

    # Rebind every dispatched context to the current complete trusted manifest.
    audit = AuditLog.from_manifest(layout.audit_path, manifest)
    dependencies = STAGE_DEPENDENCIES[stage]
    if dependencies:
        try:
            assert_stage_complete(layout, manifest, dependencies)
        except IncompleteRunError as exc:
            return StageRunSummary(stage=stage, dependency_issues=exc.issues)

    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    blocked: list[str] = []
    executor = executors.get(stage)
    for job in stage_jobs:
        if any(
            _dependency_state(status, dependency) != "success"
            for dependency in job.depends_on
        ):
            blocked.append(job.job_id)
            continue
        decision = status.decide(job.job_id)
        if decision.action == "skip":
            skipped.append(job.job_id)
            continue
        try:
            status.start(job.job_id)
        except Exception as exc:
            reopened = StatusStore(layout, manifest)
            if reopened.entry(job.job_id)["state"] != "running":
                raise exc
            status = reopened
        item = StageContext(protocol, layout, job, status, audit)
        try:
            if executor is None:
                raise RuntimeError(f"no executor registered for stage {stage}")
            result = executor(item)
            if not isinstance(result, ResultRecord):
                raise TypeError("stage executor must return ResultRecord")
            atomic_write_json(layout.root / job.output_relpath, asdict(result))
        except Exception as exc:
            status = _record_failure(status, layout, manifest, job.job_id, exc)
            failed.append(job.job_id)
            continue
        try:
            status.succeed(job.job_id)
        except Exception as exc:
            reopened = StatusStore(layout, manifest)
            state = reopened.entry(job.job_id)["state"]
            if state == "success":
                status = reopened
            elif state == "running":
                status = _record_failure(
                    reopened, layout, manifest, job.job_id, exc
                )
                failed.append(job.job_id)
                continue
            else:
                raise exc
        succeeded.append(job.job_id)
        if stop_after_jobs is not None and len(succeeded) == stop_after_jobs:
            return StageRunSummary(
                stage=stage,
                succeeded_job_ids=tuple(succeeded),
                skipped_job_ids=tuple(skipped),
                failed_job_ids=tuple(failed),
                blocked_job_ids=tuple(blocked),
                stopped_early=True,
            )
    return StageRunSummary(
        stage=stage,
        succeeded_job_ids=tuple(succeeded),
        skipped_job_ids=tuple(skipped),
        failed_job_ids=tuple(failed),
        blocked_job_ids=tuple(blocked),
    )


def _dependency_state(status: StatusStore, job_id: str) -> object:
    try:
        return status.entry(job_id)["state"]
    except ValueError:
        return None


def _record_failure(
    status: StatusStore,
    layout: RunLayout,
    manifest: Sequence[JobSpec],
    job_id: str,
    exception: Exception,
) -> StatusStore:
    try:
        status.fail(job_id, exception)
        return status
    except Exception as persistence_error:
        reopened = StatusStore(layout, manifest)
        if reopened.entry(job_id)["state"] == "failed":
            return reopened
        raise persistence_error from exception
