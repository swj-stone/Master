"""Behavioral tests for audited ICOMP 2026 split access."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import minppi_mofs.experiment_protocol as protocol_module
import experiments.icomp2026_stages as stage_module
from minppi_mofs.experiment_protocol import (
    AuditLog,
    AuditedSplitAccess,
    IncompleteRunError,
    JobSpec,
    ManifestStore,
    PreparedSplitPaths,
    RunLayout,
    ResultRecord,
    StatusStore,
    TestAccessViolation as _TestAccessViolation,
    assert_stage_complete,
    atomic_write_json,
    build_job_manifest,
    create_new_run,
    load_resolved_protocol,
)
from experiments.icomp2026_stages import StageContext, execute_stage
from experiments.run_icomp2026_pipeline import (
    STAGE_ORDER,
    PipelineRunner,
    build_parser,
    main,
)


@pytest.fixture
def prepared_paths(tmp_path: Path) -> PreparedSplitPaths:
    frames = {
        "train": pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3, 4], "target": [0, 1]}),
        "val": pd.DataFrame({"feature_a": [5.0], "feature_b": [6], "target": [1]}),
        "test": pd.DataFrame({"feature_a": [7.0, 8.0], "feature_b": [9, 10], "target": [0, 1]}),
    }
    for name, frame in frames.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    return PreparedSplitPaths(
        dataset_id="toy_data",
        seed=42,
        target="target",
        train_csv=tmp_path / "train.csv",
        val_csv=tmp_path / "val.csv",
        test_csv=tmp_path / "test.csv",
    )


def _authorized_audit(
    path: Path, paths: PreparedSplitPaths, *, job_id: str, stage: str
) -> AuditLog:
    return AuditLog.from_manifest(
        path,
        (
            JobSpec(
                job_id, "hash", stage, paths.dataset_id, paths.seed, "method", {}, (),
                f"raw_metrics/{job_id}.json",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("stage", "reason"),
    [
        ("removal", "final_removal_evaluation"),
        ("alpha", "final_alpha_grid_evaluation"),
        ("nsga2", "final_nsga2_front_evaluation"),
    ],
)
def test_frozen_test_access_pairs_return_test_arrays_and_append_allowed_event(
    tmp_path: Path, prepared_paths: PreparedSplitPaths, stage: str, reason: str
) -> None:
    audit = _authorized_audit(
        tmp_path / "run-allowed" / "protocol_audit.json",
        prepared_paths,
        job_id=f"job-{stage}",
        stage=stage,
    )
    access = AuditedSplitAccess(
        prepared_paths, stage=stage, job_id=f"job-{stage}", audit=audit
    )

    X_test, y_test = access.load_test(reason=reason)

    assert list(X_test) == ["feature_a", "feature_b"]
    assert y_test.tolist() == [0, 1]
    event = audit.read_events()[-1]
    assert event["run_id"] == "run-allowed"
    assert event["decision"] == "allowed"
    assert event["stage"] == stage
    assert event["reason"] == (reason or "invalid_reason")
    assert event["dataset_id"] == prepared_paths.dataset_id
    assert event["seed"] == prepared_paths.seed


@pytest.mark.parametrize(
    ("stage", "reason"),
    [
        ("estimator", "selection"),
        ("removal", "final_removal_evaluation"),
        ("removal", "wrong_reason"),
        ("unknown_stage", "final_removal_evaluation"),
        ("alpha", ""),
    ],
)
def test_all_nonwhitelisted_test_requests_are_denied_and_recorded(
    tmp_path: Path, prepared_paths: PreparedSplitPaths, stage: str, reason: str
) -> None:
    audit = AuditLog(tmp_path / "run-denied" / "protocol_audit.json")
    access = AuditedSplitAccess(
        prepared_paths, stage=stage, job_id="job-estimator", audit=audit
    )

    with pytest.raises(_TestAccessViolation):
        access.load_test(reason=reason)

    event = audit.read_events()[-1]
    assert event["decision"] == "denied"
    assert event["stage"] == stage
    assert event["reason"] == (reason or "invalid_reason")


def test_train_validation_never_reads_or_returns_test_partition(
    tmp_path: Path, prepared_paths: PreparedSplitPaths
) -> None:
    audit = AuditLog(tmp_path / "run-train" / "protocol_audit.json")
    access = AuditedSplitAccess(
        prepared_paths, stage="estimator", job_id="job-estimator", audit=audit
    )
    X_train, y_train, X_val, y_val = access.load_train_validation()

    assert list(X_train) == list(X_val) == ["feature_a", "feature_b"]
    assert y_train.tolist() == [0, 1]
    assert y_val.tolist() == [1]
    assert not audit.path.exists()


def test_split_access_validates_target_and_returns_fresh_objects(
    tmp_path: Path, prepared_paths: PreparedSplitPaths
) -> None:
    access = AuditedSplitAccess(
        prepared_paths,
        stage="removal",
        job_id="job-removal",
        audit=_authorized_audit(
            tmp_path / "run-copy" / "protocol_audit.json",
            prepared_paths,
            job_id="job-removal",
            stage="removal",
        ),
    )
    first, _, _, _ = access.load_train_validation()
    first.loc[0, "feature_a"] = 999.0
    second, _, _, _ = access.load_train_validation()
    assert second.loc[0, "feature_a"] == 1.0

    broken = PreparedSplitPaths(
        dataset_id="toy_data", seed=42, target="missing", train_csv=prepared_paths.train_csv,
        val_csv=prepared_paths.val_csv, test_csv=prepared_paths.test_csv,
    )
    with pytest.raises(ValueError, match="target"):
        AuditedSplitAccess(
            broken, stage="estimator", job_id="job-estimator", audit=AuditLog(tmp_path / "audit.json")
        ).load_train_validation()


def test_prepared_paths_are_frozen_and_reject_unsafe_identifiers(
    tmp_path: Path, prepared_paths: PreparedSplitPaths
) -> None:
    with pytest.raises(FrozenInstanceError):
        prepared_paths.target = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="dataset"):
        PreparedSplitPaths("../escape", 42, "target", tmp_path / "a", tmp_path / "b", tmp_path / "c")
    with pytest.raises(ValueError, match="seed"):
        PreparedSplitPaths("toy", True, "target", tmp_path / "a", tmp_path / "b", tmp_path / "c")
    with pytest.raises(ValueError, match="distinct"):
        PreparedSplitPaths(
            "toy", 42, "target", prepared_paths.train_csv, prepared_paths.val_csv,
            prepared_paths.train_csv,
        )
    with pytest.raises(ValueError, match="missing"):
        PreparedSplitPaths(
            "toy", 42, "target", tmp_path / "missing.csv", prepared_paths.val_csv,
            prepared_paths.test_csv,
        )


def test_audit_append_is_sequential_and_recovers_from_atomic_replace_failure(
    tmp_path: Path, prepared_paths: PreparedSplitPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _authorized_audit(
        tmp_path / "run-atomic" / "protocol_audit.json",
        prepared_paths,
        job_id="job-removal",
        stage="removal",
    )
    access = AuditedSplitAccess(
        prepared_paths, stage="removal", job_id="job-removal", audit=audit
    )
    access.load_test(reason="final_removal_evaluation")
    access.load_test(reason="final_removal_evaluation")
    first, second = audit.read_events()
    assert first["event_id"] != second["event_id"]
    assert first["at"] <= second["at"]

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(protocol_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        access.load_test(reason="final_removal_evaluation")
    assert len(audit.read_events()) == 2
    assert not list(audit.path.parent.glob(".protocol_audit.json.*.tmp"))


def test_audit_integrity_rejects_malformed_foreign_unknown_and_forged_allowed(
    tmp_path: Path
) -> None:
    audit = AuditLog(tmp_path / "trusted-run" / "protocol_audit.json")
    audit.append_test_access(
        job_id="known-job", stage="removal", reason="final_removal_evaluation", decision="allowed"
    )
    payload = json.loads(audit.path.read_text(encoding="utf-8"))
    valid = payload["events"][0]
    forged = dict(valid, event_id="forged", reason="selection", decision="allowed")
    foreign = dict(valid, event_id="foreign", run_id="other-run")
    unknown = dict(valid, event_id="unknown", job_id="unknown-job")
    payload["events"].extend([forged, foreign, unknown, {"event_id": "malformed"}])
    atomic_write_json(audit.path, payload)

    issues = audit.integrity_issues(expected_run_id="trusted-run", allowed_job_ids={"known-job"})

    assert any("forged allowed" in issue for issue in issues)
    assert any("foreign run" in issue for issue in issues)
    assert any("unknown job" in issue for issue in issues)
    assert any("malformed" in issue for issue in issues)


def test_denied_event_requires_explicit_audited_resolution_lifecycle(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "run-resolve" / "protocol_audit.json")
    denied = audit.append_test_access(
        job_id="job-estimator", stage="estimator", reason="selection", decision="denied"
    )
    assert any("unresolved denied" in issue for issue in audit.integrity_issues(
        expected_run_id="run-resolve", allowed_job_ids={"job-estimator"}
    ))
    with pytest.raises(ValueError, match="denied"):
        audit.resolve_denied("not-a-denied-event", "reviewed")
    with pytest.raises(ValueError, match="resolution"):
        audit.resolve_denied(denied["event_id"], "")

    resolved = audit.resolve_denied(denied["event_id"], "reviewed and approved exception")

    assert resolved["event_type"] == "denied_resolution"
    assert resolved["resolution"].startswith("reviewed")
    assert not audit.integrity_issues(
        expected_run_id="run-resolve", allowed_job_ids={"job-estimator"}
    )


def test_audit_resolution_cannot_reference_a_future_denied_event(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "run-order" / "protocol_audit.json")
    denied = audit.append_test_access(
        job_id="job-estimator", stage="estimator", reason="selection", decision="denied"
    )
    resolution = audit.resolve_denied(denied["event_id"], "reviewed")
    atomic_write_json(
        audit.path,
        {"schema_version": 1, "events": [resolution, denied]},
    )

    issues = audit.integrity_issues(
        expected_run_id="run-order", allowed_job_ids={"job-estimator"}
    )

    assert any("prior denied" in issue for issue in issues)


def test_audit_rejects_duplicate_event_ids_so_each_denial_needs_its_own_resolution(
    tmp_path: Path,
) -> None:
    audit = AuditLog(tmp_path / "run-duplicate" / "protocol_audit.json")
    denied = audit.append_test_access(
        job_id="job-estimator", stage="estimator", reason="selection", decision="denied"
    )
    audit.resolve_denied(denied["event_id"], "reviewed")
    payload = json.loads(audit.path.read_text(encoding="utf-8"))
    payload["events"].append(dict(denied))
    atomic_write_json(audit.path, payload)

    issues = audit.integrity_issues(
        expected_run_id="run-duplicate", allowed_job_ids={"job-estimator"}
    )

    assert any("duplicate event ID" in issue for issue in issues)
    assert any("unresolved denied" in issue for issue in issues)


def test_assert_stage_complete_aggregates_audit_defects_without_creating_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "audit-run"
    root.mkdir()
    layout = RunLayout(
        root=root,
        configs_frozen=root / "configs_frozen",
        raw_metrics=root / "raw_metrics",
        tables=root / "tables",
        figures=root / "figures",
        audit_path=root / "protocol_audit.json",
        manifest_path=root / "run_manifest.csv",
        status_path=root / "status.json",
        log_path=root / "output.log",
    )
    jobs = (
        JobSpec(
            job_id="preprocess-job",
            protocol_hash="protocol-hash",
            stage="preprocess",
            dataset_id="toy_data",
            seed=42,
            method="preprocess",
            parameters={},
            depends_on=(),
            output_relpath="preprocess/toy.json",
        ),
    )
    audit = AuditLog(layout.audit_path)
    denied = audit.append_test_access(
        job_id=jobs[0].job_id, stage="estimator", reason="selection", decision="denied"
    )
    payload = json.loads(audit.path.read_text(encoding="utf-8"))
    payload["events"].append(dict(denied, event_id="bad-allowed", decision="allowed"))
    atomic_write_json(audit.path, payload)

    with pytest.raises(IncompleteRunError) as raised:
        assert_stage_complete(layout, jobs, ("preprocess",))

    message = str(raised.value)
    assert "unresolved denied" in message
    assert "forged allowed" in message
    layout.audit_path.unlink()
    with pytest.raises(IncompleteRunError):
        assert_stage_complete(layout, jobs, ("preprocess",))
    assert not layout.audit_path.exists()


def test_assert_stage_complete_rejects_audit_stage_that_does_not_match_job(
    tmp_path: Path,
) -> None:
    root = tmp_path / "audit-stage-run"
    root.mkdir()
    layout = RunLayout(
        root=root,
        configs_frozen=root / "configs_frozen",
        raw_metrics=root / "raw_metrics",
        tables=root / "tables",
        figures=root / "figures",
        audit_path=root / "protocol_audit.json",
        manifest_path=root / "run_manifest.csv",
        status_path=root / "status.json",
        log_path=root / "output.log",
    )
    job = JobSpec(
        job_id="preprocess-job",
        protocol_hash="protocol-hash",
        stage="preprocess",
        dataset_id="toy_data",
        seed=42,
        method="preprocess",
        parameters={},
        depends_on=(),
        output_relpath="preprocess/toy.json",
    )
    AuditLog(layout.audit_path).append_test_access(
        job_id=job.job_id,
        stage="removal",
        reason="final_removal_evaluation",
        decision="allowed",
    )

    with pytest.raises(IncompleteRunError, match="does not match manifest job authority"):
        assert_stage_complete(layout, (job,), ("preprocess",))


def test_split_paths_reject_hardlinked_partitions_and_directory_csv(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"feature": [1], "target": [0]}).to_csv(train, index=False)
    pd.DataFrame({"feature": [2], "target": [1]}).to_csv(val, index=False)
    os.link(train, test)
    with pytest.raises(ValueError, match="distinct"):
        PreparedSplitPaths("toy", 42, "target", train, val, test)
    test.unlink()
    test.mkdir()
    with pytest.raises(ValueError, match="regular"):
        PreparedSplitPaths("toy", 42, "target", train, val, test)


def test_split_paths_reject_symlink_partition_when_supported(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"feature": [1], "target": [0]}).to_csv(train, index=False)
    pd.DataFrame({"feature": [2], "target": [1]}).to_csv(val, index=False)
    try:
        os.symlink(train, test)
    except OSError:
        pytest.skip("symlink creation is unavailable in this Windows environment")
    with pytest.raises(ValueError, match="regular"):
        PreparedSplitPaths("toy", 42, "target", train, val, test)


def test_train_validation_rejects_partition_replaced_by_test_hardlink(
    prepared_paths: PreparedSplitPaths, tmp_path: Path
) -> None:
    access = AuditedSplitAccess(
        prepared_paths, stage="estimator", job_id="job", audit=AuditLog(tmp_path / "run" / "audit.json")
    )
    prepared_paths.train_csv.unlink()
    os.link(prepared_paths.test_csv, prepared_paths.train_csv)
    with pytest.raises(ValueError, match="identity"):
        access.load_train_validation()


@pytest.mark.parametrize("reason", [["selection"], "bad\x00reason"])
def test_invalid_test_reason_is_recorded_then_denied(
    tmp_path: Path, prepared_paths: PreparedSplitPaths, reason: object
) -> None:
    access = AuditedSplitAccess(
        prepared_paths,
        stage="removal",
        job_id="job-removal",
        audit=_authorized_audit(
            tmp_path / "run-invalid-reason" / "audit.json",
            prepared_paths,
            job_id="job-removal",
            stage="removal",
        ),
    )
    with pytest.raises(_TestAccessViolation):
        access.load_test(reason=reason)  # type: ignore[arg-type]
    event = access._audit.read_events()[-1]
    assert event["decision"] == "denied"
    assert event["reason"] == "invalid_reason"


def test_reason_with_broken_repr_is_audited_then_denied(
    tmp_path: Path, prepared_paths: PreparedSplitPaths
) -> None:
    class BrokenRepr:
        def __repr__(self) -> str:
            raise RuntimeError("repr must not run")

    audit = _authorized_audit(
        tmp_path / "run-broken-repr" / "audit.json", prepared_paths,
        job_id="job-removal", stage="removal",
    )
    with pytest.raises(_TestAccessViolation):
        AuditedSplitAccess(
            prepared_paths, stage="removal", job_id="job-removal", audit=audit
        ).load_test(reason=BrokenRepr())
    assert audit.read_events()[-1]["reason"] == "invalid_reason"


@pytest.mark.parametrize(
    "reason",
    [
        type("BadStrip", (str,), {"strip": lambda self: (_ for _ in ()).throw(RuntimeError("strip"))})("x"),
        type("BadHash", (str,), {"__hash__": lambda self: (_ for _ in ()).throw(RuntimeError("hash"))})("x"),
        type("BadRepr", (str,), {"__repr__": lambda self: (_ for _ in ()).throw(RuntimeError("repr"))})("x"),
    ],
)
def test_str_subclass_reason_is_audited_then_denied_without_methods(
    tmp_path: Path, prepared_paths: PreparedSplitPaths, reason: str
) -> None:
    audit = _authorized_audit(
        tmp_path / "run-str-subclass" / "audit.json", prepared_paths,
        job_id="job-removal", stage="removal",
    )
    with pytest.raises(_TestAccessViolation):
        AuditedSplitAccess(
            prepared_paths, stage="removal", job_id="job-removal", audit=audit
        ).load_test(reason=reason)
    assert audit.read_events()[-1]["reason"] == "invalid_reason"


def test_from_manifest_rejects_invalid_manifest_authority(tmp_path: Path) -> None:
    audit_path = tmp_path / "run-invalid-manifest" / "audit.json"
    with pytest.raises(ValueError, match="status manifest"):
        AuditLog.from_manifest(audit_path, ())
    with pytest.raises(ValueError, match="duplicate"):
        AuditLog.from_manifest(
            audit_path,
            (
                JobSpec("job", "hash", "removal", "toy", 42, "method", {}, (), "x.json"),
                JobSpec("job", "hash", "removal", "toy", 42, "method", {}, (), "y.json"),
            ),
        )
    with pytest.raises(ValueError, match="status manifest"):
        AuditLog.from_manifest(audit_path, (object(),))  # type: ignore[arg-type]
    for field, value in (
        ("seed", True),
        ("job_id", type("BadJob", (str,), {"__hash__": lambda self: 1})("job")),
        ("dataset_id", type("BadDataset", (str,), {})("toy")),
        ("stage", type("BadStage", (str,), {})("removal")),
    ):
        job = JobSpec("job", "hash", "removal", "toy", 1, "method", {}, (), "x.json")
        object.__setattr__(job, field, value)
        with pytest.raises(ValueError, match="manifest"):
            AuditLog.from_manifest(audit_path, (job,))


def test_access_rejects_str_subclass_stage_before_test_access(
    tmp_path: Path, prepared_paths: PreparedSplitPaths
) -> None:
    class BadStage(str):
        def __hash__(self) -> int:
            raise RuntimeError("hash must not run")

    with pytest.raises(ValueError, match="stage"):
        AuditedSplitAccess(
            prepared_paths,
            stage=BadStage("removal"),
            job_id="job",
            audit=AuditLog(tmp_path / "run-bad-stage" / "audit.json"),
        )


@pytest.mark.parametrize(
    ("stage", "job_id", "authority"),
    [
        ("alpha", "job-removal", ("removal", "toy_data", 42)),
        ("removal", "job-removal", ("removal", "other_data", 42)),
        ("removal", "job-removal", ("removal", "toy_data", 99)),
        ("removal", "unknown-job", ("removal", "toy_data", 42)),
    ],
)
def test_runtime_authority_mismatch_is_audited_before_test_read(
    tmp_path: Path,
    prepared_paths: PreparedSplitPaths,
    stage: str,
    job_id: str,
    authority: tuple[str, str, int],
) -> None:
    audit = AuditLog.from_manifest(
        tmp_path / "run-authority" / "audit.json",
        (JobSpec("job-removal", "hash", authority[0], authority[1], authority[2], "method", {}, (), "x.json"),),
    )
    access = AuditedSplitAccess(prepared_paths, stage=stage, job_id=job_id, audit=audit)
    prepared_paths.test_csv.unlink()
    prepared_paths.test_csv.mkdir()
    with pytest.raises(_TestAccessViolation):
        access.load_test(reason="final_removal_evaluation")
    assert audit.read_events()[-1]["decision"] == "denied"


def test_allowed_event_survives_test_csv_failure(
    tmp_path: Path, prepared_paths: PreparedSplitPaths
) -> None:
    audit = _authorized_audit(
        tmp_path / "run-read-failure" / "audit.json",
        prepared_paths,
        job_id="job-removal",
        stage="removal",
    )
    access = AuditedSplitAccess(
        prepared_paths, stage="removal", job_id="job-removal", audit=audit
    )
    prepared_paths.test_csv.unlink()
    prepared_paths.test_csv.mkdir()
    with pytest.raises(ValueError):
        access.load_test(reason="final_removal_evaluation")
    assert audit.read_events()[-1]["decision"] == "allowed"


def test_invalid_utf8_audit_is_aggregated_by_completeness(tmp_path: Path) -> None:
    root = tmp_path / "bad-audit-run"
    root.mkdir()
    layout = RunLayout(
        root=root, configs_frozen=root / "configs", raw_metrics=root / "metrics",
        tables=root / "tables", figures=root / "figures", audit_path=root / "audit.json",
        manifest_path=root / "manifest.csv", status_path=root / "status.json", log_path=root / "log.txt",
    )
    job = JobSpec("job", "hash", "preprocess", "toy", 42, "prepare", {}, (), "preprocess/x.json")
    layout.audit_path.write_bytes(b"\xff\xfe")
    with pytest.raises(IncompleteRunError, match="audit malformed"):
        assert_stage_complete(layout, (job,), ("preprocess",))


def test_audit_schema_rejects_bool_version_and_non_utc_event_time(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "run-schema" / "audit.json")
    event = audit.append_test_access(
        job_id="job", stage="estimator", reason="selection", decision="denied"
    )
    atomic_write_json(audit.path, {"schema_version": True, "events": [event]})
    assert any("malformed" in issue for issue in audit.integrity_issues(
        expected_run_id="run-schema", allowed_job_ids={"job"}
    ))
    atomic_write_json(
        audit.path,
        {"schema_version": 1, "events": [dict(event, seed=True)]},
    )
    assert any("malformed" in issue for issue in audit.integrity_issues(
        expected_run_id="run-schema", allowed_job_ids={"job"}
    ))
    atomic_write_json(
        audit.path,
        {"schema_version": 1, "events": [dict(event, at="2026-07-21T12:00:00+03:00")]},
    )
    assert any("malformed" in issue for issue in audit.integrity_issues(
        expected_run_id="run-schema", allowed_job_ids={"job"}
    ))


def test_audit_return_values_are_deep_copies_and_from_manifest_binds_jobs(
    tmp_path: Path,
) -> None:
    job = JobSpec("job", "hash", "removal", "toy", 42, "method", {}, (), "x.json")
    audit = AuditLog.from_manifest(tmp_path / "run-copy-events" / "audit.json", (job,))
    event = audit.append_test_access(
        job_id="job", stage="removal", dataset_id="toy", seed=42,
        reason="final_removal_evaluation", decision="allowed",
    )
    event["reason"] = "mutated"
    assert audit.read_events()[0]["reason"] == "final_removal_evaluation"
    returned = audit.read_events()
    returned[0]["decision"] = "denied"
    assert audit.read_events()[0]["decision"] == "allowed"


def test_integrity_rejects_foreign_and_duplicate_resolution_events(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "run-resolution-integrity" / "audit.json")
    denied = audit.append_test_access(
        job_id="job", stage="estimator", reason="selection", decision="denied"
    )
    resolution = audit.resolve_denied(denied["event_id"], "reviewed")
    payload = json.loads(audit.path.read_text(encoding="utf-8"))
    payload["events"].extend(
        [
            dict(resolution, event_id="foreign-resolution", run_id="other-run"),
            dict(resolution, event_id="duplicate-resolution"),
        ]
    )
    atomic_write_json(audit.path, payload)
    issues = audit.integrity_issues(
        expected_run_id="run-resolution-integrity", allowed_job_ids={"job"}
    )
    assert any("foreign run" in issue for issue in issues)
    assert any("foreign denied" in issue for issue in issues)
    assert any("duplicate denied resolution" in issue for issue in issues)


@pytest.fixture
def smoke_protocol():
    return load_resolved_protocol(
        Path(__file__).parents[1] / "configs",
        seeds=None,
        backend=None,
        smoke=True,
    )


def _result(job: JobSpec) -> ResultRecord:
    now = datetime.now(timezone.utc).isoformat()
    return ResultRecord(
        job_id=job.job_id,
        protocol_hash=job.protocol_hash,
        stage=job.stage,
        dataset_id=job.dataset_id,
        seed=job.seed,
        method=job.method,
        metrics={"score": 1.0},
        artifacts={},
        started_at=now,
        finished_at=now,
    )


def _contexts(protocol, layout, manifest, status):
    audit = AuditLog.from_manifest(layout.audit_path, manifest)
    return tuple(StageContext(protocol, layout, job, status, audit) for job in manifest)


def _complete_jobs(layout, status, jobs) -> None:
    for job in jobs:
        atomic_write_json(layout.root / job.output_relpath, asdict(_result(job)))
        status.start(job.job_id)
        status.succeed(job.job_id)


def _write_data_manifest(layout, protocol) -> None:
    rows = ["dataset_id,seed,selected_feature_count"]
    rows.extend(
        f"{dataset.dataset_id},{seed},7"
        for dataset in protocol.datasets
        for seed in protocol.seeds
    )
    (layout.root / "data_manifest.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def test_stage_context_is_frozen_single_job_value(tmp_path: Path, smoke_protocol) -> None:
    layout = create_new_run(tmp_path / "results", "context-run", smoke_protocol)
    manifest = build_job_manifest(smoke_protocol, stages=("validate",))
    status = StatusStore(layout, manifest)
    context = _contexts(smoke_protocol, layout, manifest, status)[0]

    with pytest.raises(FrozenInstanceError):
        context.job = manifest[0]  # type: ignore[misc]


def test_execute_stage_continues_independent_jobs_after_failure(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "independent-run", smoke_protocol)
    validate = build_job_manifest(smoke_protocol, stages=("validate",))
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    manifest = validate + preprocess
    status = StatusStore(layout, manifest)
    _complete_jobs(layout, status, validate)
    seen: list[str] = []

    def executor(context: StageContext) -> ResultRecord:
        seen.append(context.job.job_id)
        if context.job == preprocess[0]:
            raise RuntimeError("expected failure")
        return _result(context.job)

    summary = execute_stage(
        "preprocess",
        context=_contexts(smoke_protocol, layout, manifest, status),
        executors={"preprocess": executor},
    )

    assert tuple(seen) == tuple(job.job_id for job in preprocess)
    assert summary.exit_code == 1
    assert summary.failed_job_ids == (preprocess[0].job_id,)
    assert status.entry(preprocess[0].job_id)["state"] == "failed"
    assert all(status.entry(job.job_id)["state"] == "success" for job in preprocess[1:])


def test_execute_stage_skips_valid_success_on_resume(tmp_path: Path, smoke_protocol) -> None:
    layout = create_new_run(tmp_path / "results", "resume-stage", smoke_protocol)
    validate = build_job_manifest(smoke_protocol, stages=("validate",))
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    manifest = validate + preprocess
    status = StatusStore(layout, manifest)
    _complete_jobs(layout, status, validate + preprocess[:1])
    seen: list[str] = []

    summary = execute_stage(
        "preprocess",
        context=_contexts(smoke_protocol, layout, manifest, status),
        executors={"preprocess": lambda context: seen.append(context.job.job_id) or _result(context.job)},
    )

    assert preprocess[0].job_id not in seen
    assert summary.skipped_job_ids == (preprocess[0].job_id,)
    assert summary.exit_code == 0


def test_execute_stage_enforces_stage_and_job_dependencies(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "dependencies-run", smoke_protocol)
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    removal = build_job_manifest(smoke_protocol, stages=("removal",))
    manifest = preprocess + removal
    status = StatusStore(layout, manifest)
    contexts = _contexts(smoke_protocol, layout, manifest, status)

    blocked = execute_stage(
        "removal", context=contexts, executors={"removal": lambda context: _result(context.job)}
    )
    assert blocked.exit_code == 1
    assert blocked.dependency_issues
    assert all(status.entry(job.job_id)["state"] == "pending" for job in removal)

    _complete_jobs(layout, status, preprocess)

    def executor(context: StageContext) -> ResultRecord:
        if context.job == removal[0]:
            raise RuntimeError("plan failed")
        return _result(context.job)

    summary = execute_stage("removal", context=contexts, executors={"removal": executor})
    first_dataset_dependents = tuple(
        job for job in removal if removal[0].job_id in job.depends_on
    )
    assert summary.exit_code == 1
    assert all(status.entry(job.job_id)["state"] == "pending" for job in first_dataset_dependents)
    assert any(status.entry(job.job_id)["state"] == "success" for job in removal if job not in first_dataset_dependents and job != removal[0])


def test_missing_executor_is_failure_and_stop_limit_counts_successful_writes(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "missing-executor", smoke_protocol)
    validate = build_job_manifest(smoke_protocol, stages=("validate",))
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    manifest = validate + preprocess
    status = StatusStore(layout, manifest)
    _complete_jobs(layout, status, validate)
    contexts = _contexts(smoke_protocol, layout, manifest, status)
    missing = execute_stage("preprocess", context=contexts, executors={})
    assert missing.exit_code == 1
    assert all(status.entry(job.job_id)["state"] == "failed" for job in preprocess)

    retry = execute_stage(
        "preprocess",
        context=contexts,
        executors={"preprocess": lambda context: _result(context.job)},
        stop_after_jobs=1,
    )
    assert retry.exit_code == 75
    assert len(retry.succeeded_job_ids) == 1
    assert sum((layout.root / job.output_relpath).exists() for job in preprocess) == 1


def test_execute_stage_rejects_context_job_changed_from_status_authority(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "forged-context", smoke_protocol)
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    removal = build_job_manifest(smoke_protocol, stages=("removal",))
    manifest = preprocess + removal
    status = StatusStore(layout, manifest)
    _complete_jobs(layout, status, preprocess)
    contexts = list(_contexts(smoke_protocol, layout, manifest, status))
    dependent_index = next(
        index for index, item in enumerate(contexts) if item.job.depends_on
    )
    contexts[dependent_index] = StageContext(
        smoke_protocol,
        layout,
        replace(contexts[dependent_index].job, depends_on=()),
        status,
        contexts[dependent_index].audit,
    )

    with pytest.raises(ValueError, match="authority|manifest"):
        execute_stage(
            "removal",
            context=tuple(contexts),
            executors={"removal": lambda context: _result(context.job)},
        )


@pytest.mark.parametrize("foreign", ["protocol", "layout"])
def test_execute_stage_rejects_foreign_context_service_authority(
    tmp_path: Path, smoke_protocol, foreign: str
) -> None:
    layout = create_new_run(tmp_path / "results", "authority-run", smoke_protocol)
    manifest = build_job_manifest(smoke_protocol, stages=("validate",))
    status = StatusStore(layout, manifest)
    protocol = (
        replace(smoke_protocol, backend="cuda")
        if foreign == "protocol"
        else smoke_protocol
    )
    context_layout = (
        create_new_run(tmp_path / "results", "foreign-run", smoke_protocol)
        if foreign == "layout"
        else layout
    )
    audit = AuditLog.from_manifest(context_layout.audit_path, manifest)
    contexts = tuple(
        StageContext(protocol, context_layout, job, status, audit) for job in manifest
    )

    with pytest.raises(ValueError, match="protocol|layout|authority"):
        execute_stage(
            "validate",
            context=contexts,
            executors={"validate": lambda context: _result(context.job)},
        )
    assert status.entry(manifest[0].job_id)["state"] == "pending"


def test_execute_stage_reopens_after_root_write_failure_and_continues(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = create_new_run(tmp_path / "results", "root-recovery", smoke_protocol)
    validate = build_job_manifest(smoke_protocol, stages=("validate",))
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    manifest = validate + preprocess
    status = StatusStore(layout, manifest)
    _complete_jobs(layout, status, validate)
    real_write = protocol_module.atomic_write_json
    root_writes = 0

    def fail_one_root_write(path: Path, value: object) -> None:
        nonlocal root_writes
        if Path(path) == layout.status_path:
            root_writes += 1
            if root_writes == 2:
                raise OSError("root write failed once")
        real_write(path, value)

    monkeypatch.setattr(protocol_module, "atomic_write_json", fail_one_root_write)
    summary = execute_stage(
        "preprocess",
        context=_contexts(smoke_protocol, layout, manifest, status),
        executors={"preprocess": lambda context: _result(context.job)},
    )

    assert summary.exit_code == 0
    reopened = StatusStore(layout, manifest)
    assert all(reopened.entry(job.job_id)["state"] == "success" for job in preprocess)


def test_cli_declares_exact_stages_and_resume_requires_run_id() -> None:
    parser = build_parser()
    stage_action = next(action for action in parser._actions if action.dest == "stage")
    assert tuple(stage_action.choices) == ("all",) + STAGE_ORDER

    with pytest.raises(SystemExit) as raised:
        main(["--stage", "validate", "--resume"])
    assert raised.value.code == 2


def test_cli_help_works_as_a_direct_script_without_model_imports() -> None:
    script = Path(__file__).parents[1] / "experiments" / "run_icomp2026_pipeline.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--stage" in completed.stdout
    assert "xgboost" not in completed.stderr.lower()


def test_pipeline_all_dispatches_exact_dependency_order(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = object.__new__(PipelineRunner)
    seen: list[str] = []
    monkeypatch.setattr(runner, "execute", lambda stage: seen.append(stage))

    assert runner.run("all") == 0
    assert tuple(seen) == STAGE_ORDER


def test_selected_alpha_preflights_and_appends_missing_canonical_chunks(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "alpha-first", smoke_protocol)
    validate = build_job_manifest(smoke_protocol, stages=("validate",))
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    manifests = ManifestStore(layout.manifest_path)
    manifests.initialize(validate)
    manifests.append(preprocess)
    status = StatusStore(layout, validate + preprocess)
    _complete_jobs(layout, status, validate + preprocess)
    _write_data_manifest(layout, smoke_protocol)

    runner = PipelineRunner(
        smoke_protocol,
        layout,
        executors={"alpha": lambda context: _result(context.job)},
    )
    assert runner.run("alpha") == 0

    extended = manifests.load()
    assert tuple(dict.fromkeys(job.stage for job in extended)) == (
        "validate", "preprocess", "estimator", "removal", "alpha"
    )
    reopened = StatusStore(layout, extended)
    assert all(
        reopened.entry(job.job_id)["state"] == "pending"
        for job in extended
        if job.stage in {"estimator", "removal"}
    )


def test_stage_preflight_rejection_does_not_partially_append_manifest(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "no-partial", smoke_protocol)
    validate = build_job_manifest(smoke_protocol, stages=("validate",))
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    manifests = ManifestStore(layout.manifest_path)
    manifests.initialize(validate)
    manifests.append(preprocess)
    status = StatusStore(layout, validate + preprocess)
    _complete_jobs(layout, status, validate + preprocess)
    _write_data_manifest(layout, smoke_protocol)
    before = layout.manifest_path.read_bytes()

    assert PipelineRunner(smoke_protocol, layout).run("nsga2") == 1

    assert layout.manifest_path.read_bytes() == before
    assert not (layout.root / "estimator" / "status.json").exists()
    assert not (layout.root / "removal" / "status.json").exists()


def test_cli_consumes_extensible_stage_executor_registry(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stage_module,
        "STAGE_EXECUTORS",
        {"validate": lambda context: _result(context.job)},
        raising=False,
    )
    output_root = tmp_path / "results"

    assert main(
        [
            "--stage", "validate",
            "--config-dir", str(Path(__file__).parents[1] / "configs"),
            "--output-root", str(output_root),
            "--run-id", "registry-run",
            "--smoke",
        ]
    ) == 0


def test_public_stop_limit_rejects_formal_engine_and_runner(
    tmp_path: Path, smoke_protocol
) -> None:
    formal = replace(smoke_protocol, run_kind="formal")
    layout = create_new_run(tmp_path / "results", "formal-engine", formal)
    manifest = build_job_manifest(formal, stages=("validate",))
    status = StatusStore(layout, manifest)
    contexts = _contexts(formal, layout, manifest, status)

    with pytest.raises(ValueError, match="smoke"):
        execute_stage(
            "validate",
            context=contexts,
            executors={"validate": lambda context: _result(context.job)},
            stop_after_jobs=1,
        )
    with pytest.raises(ValueError, match="smoke"):
        PipelineRunner(formal, layout, stop_after_jobs=1)


@pytest.mark.parametrize(
    "invalid_limit",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
    ],
)
def test_pipeline_runner_rejects_nonpositive_or_bool_stop_before_mutation(
    tmp_path: Path, smoke_protocol, invalid_limit: object
) -> None:
    layout = create_new_run(tmp_path / "results", "invalid-stop", smoke_protocol)

    with pytest.raises(ValueError, match="positive integer"):
        PipelineRunner(
            smoke_protocol,
            layout,
            stop_after_jobs=invalid_limit,  # type: ignore[arg-type]
        )

    assert not layout.manifest_path.exists()
    assert not layout.status_path.exists()


def test_runner_without_new_run_authority_rejects_missing_manifest_without_recreation(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "missing-resume-manifest", smoke_protocol)
    validate = build_job_manifest(smoke_protocol, stages=("validate",))
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    manifest = validate + preprocess
    ManifestStore(layout.manifest_path).initialize(manifest)
    StatusStore(layout, manifest)
    root_status_before = layout.status_path.read_bytes()
    stage_status_before = {
        stage: (layout.root / stage / "status.json").read_bytes()
        for stage in ("validate", "preprocess")
    }
    layout.manifest_path.unlink()

    runner = PipelineRunner(smoke_protocol, layout)
    with pytest.raises(RuntimeError, match="manifest.*missing|missing.*manifest"):
        runner.run("validate")

    assert not layout.manifest_path.exists()
    assert layout.status_path.read_bytes() == root_status_before
    assert {
        stage: (layout.root / stage / "status.json").read_bytes()
        for stage in ("validate", "preprocess")
    } == stage_status_before


def test_explicit_manifest_initialization_rejects_existing_status_before_write(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "status-before-manifest", smoke_protocol)
    validate = build_job_manifest(smoke_protocol, stages=("validate",))
    preprocess = build_job_manifest(smoke_protocol, stages=("preprocess",))
    StatusStore(layout, validate + preprocess)
    root_before = layout.status_path.read_bytes()
    stages_before = {
        stage: (layout.root / stage / "status.json").read_bytes()
        for stage in ("validate", "preprocess")
    }

    runner = PipelineRunner(
        smoke_protocol, layout, allow_manifest_initialize=True
    )
    with pytest.raises(RuntimeError, match="status|initialization"):
        runner.run("validate")

    assert not layout.manifest_path.exists()
    assert layout.status_path.read_bytes() == root_before
    assert {
        stage: (layout.root / stage / "status.json").read_bytes()
        for stage in ("validate", "preprocess")
    } == stages_before


def test_manifest_initialization_authority_is_single_use_after_publication(
    tmp_path: Path, smoke_protocol
) -> None:
    layout = create_new_run(tmp_path / "results", "single-use-authority", smoke_protocol)
    runner = PipelineRunner(
        smoke_protocol, layout, allow_manifest_initialize=True
    )
    assert runner.run("validate") == 1
    assert layout.manifest_path.exists()
    layout.manifest_path.unlink()
    layout.status_path.unlink()
    (layout.root / "validate" / "status.json").unlink()

    with pytest.raises(RuntimeError, match="initialization authority|manifest.*missing"):
        runner.run("validate")

    assert not layout.manifest_path.exists()


def test_cli_rejects_stop_limit_in_formal_mode_before_creating_run(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--stage", "validate",
                "--config-dir", str(Path(__file__).parents[1] / "configs"),
                "--output-root", str(output_root),
                "--run-id", "formal-stop",
                "--stop-after-jobs", "1",
            ]
        )
    assert raised.value.code == 2
    assert not output_root.exists()


def test_cli_lifecycle_is_append_logged_and_never_replaces_process_streams_or_cwd(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    arguments = [
        "--stage", "validate",
        "--config-dir", str(Path(__file__).parents[1] / "configs"),
        "--output-root", str(output_root),
        "--run-id", "cli-run",
        "--smoke",
    ]
    cwd = Path.cwd()
    stdout = sys.stdout
    stderr = sys.stderr

    assert main(arguments) == 1

    run_root = output_root / "cli-run"
    first_log = (run_root / "output.log").read_bytes()
    assert Path.cwd() == cwd
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert main(arguments + ["--resume"]) == 1
    assert (run_root / "output.log").read_bytes().startswith(first_log)
    assert len((run_root / "output.log").read_bytes()) > len(first_log)


def test_all_with_missing_validate_executor_does_not_freeze_dependent_stage_jobs(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    assert main(
        [
            "--stage", "all",
            "--config-dir", str(Path(__file__).parents[1] / "configs"),
            "--output-root", str(output_root),
            "--run-id", "blocked-all",
            "--smoke",
        ]
    ) == 1
    manifest = ManifestStore(output_root / "blocked-all" / "run_manifest.csv").load()
    assert {job.stage for job in manifest} == {"validate"}
