"""Run the canonical ISOLET seed-42 paired utility ablation only."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable


def _standard_library_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the canonical ISOLET seed-42 paired utility ablation."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/icomp2026_rerun"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv[1:]):
    _standard_library_parser().parse_args()


if __package__:
    from .native_process_guard import (
        CONTROLLER_ERROR,
        WorkerContract,
        run_isolated_cli,
        worker_contract,
        write_worker_receipt,
    )
else:
    from native_process_guard import (  # type: ignore[no-redef]
        CONTROLLER_ERROR,
        WorkerContract,
        run_isolated_cli,
        worker_contract,
        write_worker_receipt,
    )


_SCRIPT_ARGUMENTS: tuple[str, ...] = ()
_SCRIPT_WORKER_CONTRACT: WorkerContract | None = None
if __name__ == "__main__":
    _SCRIPT_ARGUMENTS = tuple(sys.argv[1:])
    try:
        _SCRIPT_WORKER_CONTRACT = worker_contract(os.environ)
    except ValueError as exc:
        print(f"utility-pair worker contract error: {exc}", file=sys.stderr)
        raise SystemExit(CONTROLLER_ERROR) from None
    if _SCRIPT_WORKER_CONTRACT is None:
        raise SystemExit(run_isolated_cli(Path(__file__), _SCRIPT_ARGUMENTS))


from minppi_mofs.experiment_protocol import (  # noqa: E402
    AuditLog,
    IncompleteRunError,
    JobSpec,
    ManifestStore,
    ResultRecord,
    ResolvedProtocol,
    RunLayout,
    StatusStore,
    assert_stage_complete,
    atomic_write_json,
    build_job_manifest,
    create_new_run,
    estimator_budgets_for,
    freeze_run_metadata,
    load_resolved_protocol,
    open_resumed_run,
    read_data_manifest,
    read_result_record,
)

if __package__:
    from . import icomp2026_stages as stage_module
    from .icomp2026_progress import append_run_log
else:
    import icomp2026_stages as stage_module  # type: ignore[no-redef]
    from icomp2026_progress import append_run_log  # type: ignore[no-redef]


_PAIR_VALUES = frozenset({"accuracy", "ovr_auc"})
_SUPPORTED_EXECUTION_MODES = frozenset({"legacy", "grouped_nested_v1"})


def project_isolet_utility_pair_protocol(
    protocol: ResolvedProtocol,
) -> ResolvedProtocol:
    """Project a formal loaded protocol onto canonical ISOLET seed-42 scope."""
    if not isinstance(protocol, ResolvedProtocol):
        raise TypeError("protocol must be a ResolvedProtocol")
    isolet = tuple(
        dataset
        for dataset in protocol.datasets
        if dataset.dataset_id == "isolet"
    )
    if len(isolet) != 1:
        raise ValueError(
            "ISOLET utility-pair projection requires exactly one ISOLET dataset"
        )
    projected = replace(
        protocol,
        datasets=isolet,
        seeds=(42,),
        estimator_budgets_by_dataset={
            "isolet": estimator_budgets_for(protocol, "isolet")
        },
    )
    _validate_scope_protocol(projected)
    return projected


def _validate_scope_protocol(protocol: ResolvedProtocol) -> None:
    if not isinstance(protocol, ResolvedProtocol):
        raise TypeError("protocol must be a ResolvedProtocol")
    if protocol.run_kind != "formal":
        raise ValueError("ISOLET utility-pair scope requires a formal protocol")
    if protocol.backend != "cuda":
        raise ValueError("ISOLET utility-pair scope requires backend=cuda")
    if protocol.execution.mode not in _SUPPORTED_EXECUTION_MODES:
        raise ValueError("ISOLET utility-pair scope execution mode is unsupported")
    if len(protocol.datasets) != 1:
        raise ValueError("ISOLET utility-pair scope requires exactly one dataset")
    dataset = protocol.datasets[0]
    if (
        dataset.dataset_id != "isolet"
        or dataset.task != "multiclass"
        or dataset.utility != "ovr_auc"
    ):
        raise ValueError(
            "ISOLET utility-pair scope requires the canonical ISOLET dataset"
        )
    if protocol.seeds != (42,):
        raise ValueError("ISOLET utility-pair scope requires exactly seed 42")
    values = protocol.ablations.get("utility")
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 2
        or frozenset(values) != _PAIR_VALUES
    ):
        raise ValueError(
            "ISOLET utility-pair scope requires accuracy and ovr_auc exactly once"
        )
    if (
        isinstance(dataset.max_features, bool)
        or not isinstance(dataset.max_features, int)
        or dataset.max_features < 1
    ):
        raise ValueError(
            "ISOLET utility-pair scope requires a positive max_features authority"
        )


def _provisional_selected_counts(
    protocol: ResolvedProtocol,
) -> tuple[dict[str, object], ...]:
    dataset = protocol.datasets[0]
    return (
        {
            "dataset_id": dataset.dataset_id,
            "seed": 42,
            "selected_feature_count": dataset.max_features,
        },
    )


def _canonical_pair(
    protocol: ResolvedProtocol,
    data_manifest: Sequence[dict[str, object]],
) -> tuple[JobSpec, JobSpec]:
    canonical = build_job_manifest(
        protocol,
        stages=("ablation",),
        data_manifest=tuple(data_manifest),
    )
    pair = tuple(
        job
        for job in canonical
        if (
            job.dataset_id == "isolet"
            and job.seed == 42
            and job.method == "utility"
            and job.parameters.get("axis") == "utility"
            and job.parameters.get("value") in _PAIR_VALUES
        )
    )
    if (
        len(pair) != 2
        or len({job.job_id for job in pair}) != 2
        or {job.parameters["value"] for job in pair} != _PAIR_VALUES
    ):
        raise ValueError("canonical ISOLET utility-pair jobs are missing or duplicated")
    return pair


def build_isolet_utility_pair_manifest(
    protocol: ResolvedProtocol,
) -> tuple[JobSpec, ...]:
    """Build the complete four-job authority before any status is created."""
    _validate_scope_protocol(protocol)
    validate_jobs = build_job_manifest(protocol, stages=("validate",))
    preprocess_jobs = build_job_manifest(protocol, stages=("preprocess",))
    pair = _canonical_pair(protocol, _provisional_selected_counts(protocol))
    manifest = validate_jobs + preprocess_jobs + pair
    if (
        len(manifest) != 4
        or len({job.job_id for job in manifest}) != len(manifest)
    ):
        raise ValueError("ISOLET utility-pair scoped manifest is not exactly four jobs")
    return manifest


def validate_isolet_utility_pair_manifest(
    protocol: ResolvedProtocol,
    manifest: Sequence[JobSpec],
    data_manifest: Sequence[dict[str, object]],
) -> tuple[JobSpec, JobSpec]:
    """Reconfirm the durable four-job authority from produced preprocessing data."""
    _validate_scope_protocol(protocol)
    if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes)):
        raise TypeError("scoped manifest must be a sequence of JobSpec entries")
    canonical_pair = _canonical_pair(protocol, data_manifest)
    expected = (
        build_job_manifest(protocol, stages=("validate",))
        + build_job_manifest(protocol, stages=("preprocess",))
        + canonical_pair
    )
    if tuple(manifest) != expected:
        raise ValueError(
            "scoped manifest differs from canonical validate, preprocess, "
            "and ISOLET utility-pair jobs"
        )
    return canonical_pair


PairExecutor = Callable[[stage_module.StageContext], ResultRecord]
PairResultValidator = Callable[[stage_module.StageContext, ResultRecord], None]
DataManifestReader = Callable[
    [Path, ResolvedProtocol],
    Sequence[dict[str, object]],
]


def _validate_utility_pair_result(
    context: stage_module.StageContext,
    record: ResultRecord,
) -> None:
    budget = max(
        estimator_budgets_for(context.protocol, context.job.dataset_id)
    )
    group_id = f"utility:isolet:s42:b{budget}"
    relpath = f"ablation/isolet/s42/utility/paired/b{budget}/scores.json"
    expected_metric_keys = {
        "axis",
        "value",
        "budget",
        "shared_group_id",
        "shared_group_model_fit_count",
        "shared_group_coalition_count",
    }
    if record.artifacts != {"paired_scores": relpath}:
        raise ValueError("utility-pair ResultRecord artifact authority is invalid")
    if (
        set(record.metrics) != expected_metric_keys
        or record.metrics["axis"] != "utility"
        or record.metrics["value"] != context.job.parameters["value"]
        or record.metrics["budget"] != budget
        or record.metrics["shared_group_id"] != group_id
    ):
        raise ValueError("utility-pair ResultRecord metrics are invalid")
    document = stage_module._read_utility_pair_artifact(
        context,
        context.layout.root / relpath,
        budget,
        group_id,
    )
    if (
        record.metrics["shared_group_model_fit_count"]
        != document["shared_group_model_fit_count"]
        or record.metrics["shared_group_coalition_count"]
        != document["shared_group_coalition_count"]
    ):
        raise ValueError(
            "utility-pair ResultRecord costs conflict with paired artifact"
        )


def execute_isolet_utility_pair_jobs(
    protocol: ResolvedProtocol,
    layout: RunLayout,
    manifest: Sequence[JobSpec],
    data_manifest: Sequence[dict[str, object]],
    *,
    executor: PairExecutor,
    result_validator: PairResultValidator,
) -> stage_module.StageRunSummary:
    """Dispatch only the canonical pair after validate and preprocess succeed."""
    pair = validate_isolet_utility_pair_manifest(
        protocol,
        manifest,
        data_manifest,
    )
    authority = tuple(manifest)
    try:
        assert_stage_complete(
            layout,
            authority,
            ("validate", "preprocess"),
        )
    except IncompleteRunError as exc:
        return stage_module.StageRunSummary(
            stage="ablation",
            dependency_issues=exc.issues,
        )
    status = StatusStore(layout, authority)
    audit = AuditLog.from_manifest(layout.audit_path, authority)
    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for job in pair:
        context = stage_module.StageContext(
            protocol,
            layout,
            job,
            status,
            audit,
        )
        decision = status.decide(
            job.job_id,
            result_validator=lambda record, bound=context: result_validator(
                bound, record
            ),
        )
        if decision.action == "skip":
            skipped.append(job.job_id)
            continue
        status.start(job.job_id)
        try:
            record = executor(context)
            if not isinstance(record, ResultRecord):
                raise TypeError("utility-pair executor must return ResultRecord")
            result_path = layout.root / job.output_relpath
            atomic_write_json(result_path, asdict(record))
            canonical_record = read_result_record(result_path, job)
            result_validator(context, canonical_record)
        except Exception as exc:
            status.fail(job.job_id, exc)
            failed.append(job.job_id)
            continue
        status.succeed(job.job_id)
        succeeded.append(job.job_id)
    return stage_module.StageRunSummary(
        stage="ablation",
        succeeded_job_ids=tuple(succeeded),
        skipped_job_ids=tuple(skipped),
        failed_job_ids=tuple(failed),
    )


class IsoletUtilityPairRunner:
    """Create or resume the dedicated four-job scoped authority."""

    def __init__(
        self,
        protocol: ResolvedProtocol,
        layout: RunLayout,
        *,
        executors: Mapping[str, PairExecutor] | None = None,
        data_manifest_reader: DataManifestReader = read_data_manifest,
        pair_result_validator: PairResultValidator = _validate_utility_pair_result,
    ) -> None:
        _validate_scope_protocol(protocol)
        self.protocol = protocol
        self.layout = layout
        self.executors = (
            dict(stage_module.STAGE_EXECUTORS)
            if executors is None
            else dict(executors)
        )
        missing = {"validate", "preprocess", "ablation"} - set(self.executors)
        if missing:
            raise ValueError(
                f"ISOLET utility-pair executors are missing: {sorted(missing)}"
            )
        self.data_manifest_reader = data_manifest_reader
        self.pair_result_validator = pair_result_validator

    def run(self) -> int:
        expected_manifest = build_isolet_utility_pair_manifest(self.protocol)
        manifests = ManifestStore(self.layout.manifest_path)
        if manifests.path.exists():
            manifest = manifests.load()
            if manifest != expected_manifest:
                raise ValueError(
                    "durable scoped manifest differs from canonical authority"
                )
        else:
            manifest = manifests.initialize(expected_manifest)
        for stage in ("validate", "preprocess"):
            summary = self._execute_standard_stage(stage, manifest)
            if summary.exit_code != 0:
                return 1
        data_manifest = self.data_manifest_reader(
            self.layout.root / "data_manifest.csv",
            self.protocol,
        )
        pair_summary = execute_isolet_utility_pair_jobs(
            self.protocol,
            self.layout,
            manifest,
            data_manifest,
            executor=self.executors["ablation"],
            result_validator=self.pair_result_validator,
        )
        return pair_summary.exit_code

    def _execute_standard_stage(
        self,
        stage: str,
        manifest: tuple[JobSpec, ...],
    ) -> stage_module.StageRunSummary:
        status = StatusStore(self.layout, manifest)
        audit = AuditLog.from_manifest(self.layout.audit_path, manifest)
        contexts = tuple(
            stage_module.StageContext(
                self.protocol,
                self.layout,
                job,
                status,
                audit,
            )
            for job in manifest
        )
        return stage_module.execute_stage(
            stage,
            context=contexts,
            executors=self.executors,
        )


def build_parser() -> argparse.ArgumentParser:
    return _standard_library_parser()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.run_id is None:
        raise ValueError("--resume requires --run-id")
    project_root = Path(__file__).resolve().parents[1]
    try:
        loaded_protocol = load_resolved_protocol(
            args.config_dir,
            seeds=(42,),
            backend="cuda",
            smoke=False,
        )
        protocol = project_isolet_utility_pair_protocol(loaded_protocol)
    except (OSError, TypeError, ValueError):
        return 2
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
    append_run_log(
        layout.log_path,
        f"start scope=isolet_utility_pair resume={args.resume}",
    )
    try:
        result = IsoletUtilityPairRunner(protocol, layout).run()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        append_run_log(
            layout.log_path,
            f"error {type(exc).__name__}: {exc}",
        )
        return 2
    append_run_log(
        layout.log_path,
        f"finish scope=isolet_utility_pair exit_code={result}",
    )
    return result


def _run_worker(contract: WorkerContract, arguments: Sequence[str]) -> int:
    result = main(arguments)
    try:
        write_worker_receipt(contract, result)
    except (OSError, TypeError, ValueError) as exc:
        print(f"utility-pair worker receipt error: {exc}", file=sys.stderr)
        return CONTROLLER_ERROR
    return result


if __name__ == "__main__":
    if _SCRIPT_WORKER_CONTRACT is None:
        raise SystemExit(CONTROLLER_ERROR)
    raise SystemExit(_run_worker(_SCRIPT_WORKER_CONTRACT, _SCRIPT_ARGUMENTS))
