"""Behavioral tests for audited ICOMP 2026 split access."""

from __future__ import annotations

import ast
import json
import hashlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

import minppi_mofs.experiment_protocol as protocol_module
import minppi_mofs.data_protocol as data_protocol_module
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
    freeze_run_metadata,
    load_resolved_protocol,
    upsert_data_manifest,
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
    jobs = {
        (job.dataset_id, job.seed): job
        for job in build_job_manifest(protocol, stages=("preprocess",))
    }
    for dataset in protocol.datasets:
        for seed in protocol.seeds:
            job = jobs[(dataset.dataset_id, seed)]
            prefix = f"preprocess/{dataset.dataset_id}/s{seed}"
            upsert_data_manifest(
                layout.root / "data_manifest.csv",
                protocol,
                {
                    "protocol_hash": job.protocol_hash,
                    "job_id": job.job_id,
                    "dataset_id": dataset.dataset_id,
                    "seed": seed,
                    "task": dataset.task,
                    "utility": dataset.utility,
                    "target": dataset.target,
                    "raw_file_sha256": "1" * 64,
                    "raw_row_count": 10,
                    "raw_feature_count": 8,
                    "selected_feature_count": 7,
                    "selected_features_json": json.dumps(
                        [f"f{index}" for index in range(7)], separators=(",", ":")
                    ),
                    "train_relpath": f"{prefix}/train.csv",
                    "val_relpath": f"{prefix}/val.csv",
                    "test_relpath": f"{prefix}/test.csv",
                    "train_row_count": 6,
                    "val_row_count": 2,
                    "test_row_count": 2,
                    "train_index_sha256": "2" * 64,
                    "val_index_sha256": "3" * 64,
                    "test_index_sha256": "4" * 64,
                    "train_file_sha256": "5" * 64,
                    "val_file_sha256": "6" * 64,
                    "test_file_sha256": "7" * 64,
                    "train_label_summary_json": '{"count":6}',
                    "val_label_summary_json": '{"count":2}',
                    "test_label_summary_json": '{"count":2}',
                    "transform_metadata_json": '{"scaler_mean":[0.0]}',
                },
            )


def _synthetic_stage_protocol(tmp_path: Path, smoke_protocol):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rng = np.random.default_rng(2026)
    count = 80
    spambase = pd.DataFrame(
        {
            "signal": np.tile([0.0, 1.0], count // 2) + rng.normal(0, 0.05, count),
            "noise": rng.normal(10, 2, count),
            "constant": np.full(count, 7.0),
            "is_spam": np.tile(["ham", "spam"], count // 2),
        }
    )
    abalone = pd.DataFrame(
        {
            "length": rng.normal(2, 0.4, count),
            "weight": rng.normal(5, 1.2, count),
            "constant": np.full(count, 4.0),
            "Rings": [str(4.5 + index / 20) for index in range(count)],
        }
    )
    raw_paths = {
        "spambase": raw_dir / "spambase.csv",
        "abalone": raw_dir / "abalone.csv",
    }
    spambase.to_csv(raw_paths["spambase"], index=False)
    abalone.to_csv(raw_paths["abalone"], index=False)
    datasets = tuple(
        replace(
            dataset,
            raw_path=raw_paths[dataset.dataset_id],
            max_features=2,
            variance_threshold=0.0,
        )
        for dataset in smoke_protocol.datasets
    )
    return replace(smoke_protocol, datasets=datasets), raw_paths


def test_real_validate_stage_writes_structured_checks_and_requires_raw_cache(
    tmp_path: Path, smoke_protocol
) -> None:
    protocol, raw_paths = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    layout = create_new_run(tmp_path / "results", "validate-real", protocol)
    freeze_run_metadata(layout, protocol, project_root=Path(__file__).parents[1])
    manifest = build_job_manifest(protocol, stages=("validate",))
    status = StatusStore(layout, manifest)
    context = _contexts(protocol, layout, manifest, status)[0]

    result = stage_module.run_validate_job(context)

    checks_path = layout.root / result.artifacts["checks"]
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    assert result.metrics["checks_passed"] is True
    assert checks["backend"] == "cpu"
    assert checks["raw_datasets"] == ["spambase", "abalone"]
    assert checks["registered_stages"] == ["preprocess", "validate"]
    assert all(path.exists() for path in raw_paths.values())

    raw_paths["spambase"].unlink()
    with pytest.raises(FileNotFoundError, match="spambase|raw"):
        stage_module.run_validate_job(context)


def test_real_validate_stage_rejects_forged_task_utility_pair(
    tmp_path: Path, smoke_protocol
) -> None:
    protocol, _ = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    forged = replace(
        protocol,
        datasets=(replace(protocol.datasets[0], utility="r2"), *protocol.datasets[1:]),
    )
    layout = create_new_run(tmp_path / "results", "validate-forged", forged)
    freeze_run_metadata(layout, forged, project_root=Path(__file__).parents[1])
    manifest = build_job_manifest(forged, stages=("validate",))
    status = StatusStore(layout, manifest)

    with pytest.raises(ValueError, match="task|utility|compatible"):
        stage_module.run_validate_job(_contexts(forged, layout, manifest, status)[0])


def _validation_context(tmp_path: Path, smoke_protocol, run_id: str, *, backend="cpu"):
    protocol, _ = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    protocol = replace(protocol, backend=backend)
    layout = create_new_run(tmp_path / "results", run_id, protocol)
    freeze_run_metadata(layout, protocol, project_root=Path(__file__).parents[1])
    manifest = build_job_manifest(protocol, stages=("validate",))
    status = StatusStore(layout, manifest)
    return layout, _contexts(protocol, layout, manifest, status)[0]


@pytest.mark.parametrize(
    ("package", "value"),
    [
        ("python", "unavailable"),
        ("numpy", " UnAvAiLaBlE "),
        ("pandas", "UNAVAILABLE"),
        ("scikit_learn", " unavailable "),
        ("xgboost", "Unavailable"),
    ],
)
def test_validate_rejects_unavailable_package_versions(
    tmp_path: Path, smoke_protocol, package: str, value: str
) -> None:
    layout, context = _validation_context(tmp_path, smoke_protocol, f"bad-{package}")
    path = layout.root / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment[package] = value
    atomic_write_json(path, environment)

    with pytest.raises(ValueError, match="package|version|unavailable"):
        stage_module.run_validate_job(context)


@pytest.mark.parametrize(
    "build_info",
    [{}, {"unavailable": "ImportError"}, {"details": {"Unavailable": True}}],
)
def test_validate_rejects_empty_or_unavailable_xgboost_build_info(
    tmp_path: Path, smoke_protocol, build_info: dict[str, object]
) -> None:
    layout, context = _validation_context(tmp_path, smoke_protocol, "bad-build")
    path = layout.root / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["xgboost_build_info"] = build_info
    atomic_write_json(path, environment)

    with pytest.raises(ValueError, match="build|unavailable"):
        stage_module.run_validate_job(context)


def test_validate_cuda_requires_frozen_xgboost_cuda_capability_even_with_visible_gpu(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, context = _validation_context(
        tmp_path, smoke_protocol, "cuda-cpu-build", backend="cuda"
    )
    path = layout.root / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment.update(
        {
            "python": "3.12.1",
            "numpy": "2.0.0",
            "pandas": "2.2.0",
            "scikit_learn": "1.5.0",
            "xgboost": "3.0.0",
            "xgboost_build_info": {"USE_CUDA": False},
            "cuda_build_info": {},
            "backend": "cuda",
        }
    )
    atomic_write_json(path, environment)
    monkeypatch.setattr(stage_module, "_cuda_visibility", lambda backend: "GPU 0")

    with pytest.raises(ValueError, match="CUDA|build|capability"):
        stage_module.run_validate_job(context)


def test_validate_freezes_raw_authority_and_preprocess_rejects_later_raw_drift(
    tmp_path: Path, smoke_protocol
) -> None:
    protocol, raw_paths = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    layout = create_new_run(tmp_path / "results", "raw-drift", protocol)
    freeze_run_metadata(layout, protocol, project_root=Path(__file__).parents[1])
    manifest = (
        build_job_manifest(protocol, stages=("validate",))
        + build_job_manifest(protocol, stages=("preprocess",))
    )
    status = StatusStore(layout, manifest)
    contexts = _contexts(protocol, layout, manifest, status)
    validate_context = next(item for item in contexts if item.job.stage == "validate")
    preprocess_context = next(
        item
        for item in contexts
        if item.job.stage == "preprocess" and item.job.dataset_id == "spambase"
    )

    validation = stage_module.run_validate_job(validate_context)
    checks_path = layout.root / validation.artifacts["checks"]
    authority_path = layout.root / validation.artifacts["raw_authority"]
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    original = raw_paths["spambase"].read_bytes()
    assert checks["raw_authority"]["spambase"] == {
        "byte_count": len(original),
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    assert authority == {
        "protocol_hash": protocol_module.resolved_protocol_hash(protocol),
        "datasets": checks["raw_authority"],
    }
    authority_before = authority_path.read_bytes()
    checks_before = checks_path.read_bytes()
    raw_paths["spambase"].write_bytes(original + b"\n")

    with pytest.raises(RuntimeError, match="raw|drift|authority"):
        stage_module.run_validate_job(validate_context)

    assert authority_path.read_bytes() == authority_before
    assert checks_path.read_bytes() == checks_before
    with pytest.raises(RuntimeError, match="raw|drift|authority"):
        stage_module.run_preprocess_job(preprocess_context)

    assert not (layout.root / "data_manifest.csv").exists()
    assert not (layout.root / "preprocess" / "spambase" / "s42").exists()


def test_preprocess_uses_independent_raw_authority_not_mutable_checks(
    tmp_path: Path, smoke_protocol
) -> None:
    protocol, raw_paths = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    layout = create_new_run(tmp_path / "results", "forged-checks-authority", protocol)
    freeze_run_metadata(layout, protocol, project_root=Path(__file__).parents[1])
    manifest = (
        build_job_manifest(protocol, stages=("validate",))
        + build_job_manifest(protocol, stages=("preprocess",))
    )
    status = StatusStore(layout, manifest)
    contexts = _contexts(protocol, layout, manifest, status)
    validate_context = next(item for item in contexts if item.job.stage == "validate")
    preprocess_context = next(
        item
        for item in contexts
        if item.job.stage == "preprocess" and item.job.dataset_id == "spambase"
    )
    validation = stage_module.run_validate_job(validate_context)

    raw_paths["spambase"].write_bytes(raw_paths["spambase"].read_bytes() + b"\n")
    changed = raw_paths["spambase"].read_bytes()
    checks_path = layout.root / validation.artifacts["checks"]
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    checks["raw_authority"]["spambase"] = {
        "byte_count": len(changed),
        "sha256": hashlib.sha256(changed).hexdigest(),
    }
    atomic_write_json(checks_path, checks)

    with pytest.raises(RuntimeError, match="raw|drift|authority"):
        stage_module.run_preprocess_job(preprocess_context)

    assert not (layout.root / "data_manifest.csv").exists()
    assert not (layout.root / "preprocess" / "spambase" / "s42").exists()


def test_validate_retry_reuses_authority_after_checks_write_failure(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, context = _validation_context(
        tmp_path, smoke_protocol, "authority-before-checks"
    )
    authority_path = layout.root / "validate" / "raw_authority.json"
    checks_path = layout.root / "validate" / "validation_checks.json"
    real_write = stage_module.atomic_write_json

    def fail_checks(path: Path, value: object) -> None:
        if Path(path) == checks_path:
            raise OSError("injected checks write failure")
        real_write(path, value)

    monkeypatch.setattr(stage_module, "atomic_write_json", fail_checks)
    with pytest.raises(OSError, match="checks write failure"):
        stage_module.run_validate_job(context)

    assert authority_path.is_file()
    authority_before = authority_path.read_bytes()
    assert not checks_path.exists()

    monkeypatch.setattr(stage_module, "atomic_write_json", real_write)
    result = stage_module.run_validate_job(context)

    assert authority_path.read_bytes() == authority_before
    assert result.artifacts == {
        "checks": "validate/validation_checks.json",
        "raw_authority": "validate/raw_authority.json",
    }
    assert checks_path.is_file()


@pytest.mark.parametrize("dataset_id", ["spambase", "abalone"])
def test_real_preprocess_stage_writes_audited_split_without_reopening_test_csv(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: str,
) -> None:
    protocol, _ = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    layout = create_new_run(tmp_path / "results", f"preprocess-{dataset_id}", protocol)
    freeze_run_metadata(layout, protocol, project_root=Path(__file__).parents[1])
    jobs = (
        build_job_manifest(protocol, stages=("validate",))
        + build_job_manifest(protocol, stages=("preprocess",))
    )
    status = StatusStore(layout, jobs)
    contexts = _contexts(protocol, layout, jobs, status)
    stage_module.run_validate_job(
        next(item for item in contexts if item.job.stage == "validate")
    )
    context = next(
        item for item in contexts
        if item.job.dataset_id == dataset_id
    )
    test_path = layout.root / f"preprocess/{dataset_id}/s42/test.csv"
    original_open = Path.open

    def guard_test_reopen(path: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if Path(path) == test_path and "r" in mode:
            raise AssertionError("preprocess executor reopened its produced test CSV")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_test_reopen)
    result = stage_module.run_preprocess_job(context)
    monkeypatch.undo()

    train = pd.read_csv(layout.root / result.artifacts["train"])
    validation = pd.read_csv(layout.root / result.artifacts["val"])
    test = pd.read_csv(test_path)
    metadata = json.loads(
        (layout.root / result.artifacts["split_metadata"]).read_text(encoding="utf-8")
    )
    target = next(item.target for item in protocol.datasets if item.dataset_id == dataset_id)
    feature_columns = [column for column in train if column != target]
    assert feature_columns == metadata["selected_features"]
    assert "constant" not in feature_columns
    assert np.allclose(train[feature_columns].mean(axis=0), 0.0, atol=1e-12)
    assert len(train) + len(validation) + len(test) == 80
    index_sets = [set(metadata[f"{name}_indices"]) for name in ("train", "val", "test")]
    assert not index_sets[0] & index_sets[1]
    assert not index_sets[0] & index_sets[2]
    assert not index_sets[1] & index_sets[2]
    assert set.union(*index_sets) == set(range(80))
    with (layout.root / "data_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rich_row = next(
            row for row in __import__("csv").DictReader(handle)
            if row["dataset_id"] == dataset_id
        )
    assert int(rich_row["selected_feature_count"]) == len(feature_columns)
    for name in ("train", "val", "test"):
        payload = (layout.root / result.artifacts[name]).read_bytes()
        assert rich_row[f"{name}_file_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result.metrics["selected_feature_count"] == len(feature_columns)


def test_real_stage_registry_is_exact_and_direct_cli_help_remains_safe() -> None:
    assert stage_module.STAGE_EXECUTORS == {
        "validate": stage_module.run_validate_job,
        "preprocess": stage_module.run_preprocess_job,
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "experiments" / "run_icomp2026_pipeline.py"),
            "--help",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--stage" in completed.stdout
    assert "traceback" not in completed.stderr.lower()


def test_direct_cli_help_uses_only_standard_library_before_exit() -> None:
    script = Path(__file__).parents[1] / "experiments" / "run_icomp2026_pipeline.py"
    probe = f"""
import builtins
import runpy
import sys

script = {str(script)!r}
real_import = builtins.__import__
attempts = []

def guarded_import(name, *args, **kwargs):
    root = name.split('.')[0]
    if root in {{'xgboost', 'pandas', 'sklearn'}}:
        attempts.append(name)
        raise ImportError('focused-review-blocked-' + root)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = [script, '--help']
try:
    runpy.run_path(script, run_name='__main__')
except SystemExit as exc:
    loaded = any(name.split('.')[0] in {{'xgboost', 'pandas', 'sklearn'}} for name in sys.modules)
    print('HEAVY_ATTEMPTS=' + ','.join(attempts))
    print('HEAVY_LOADED=' + str(loaded))
    raise
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--stage" in completed.stdout
    assert "HEAVY_ATTEMPTS=\n" in completed.stdout
    assert "HEAVY_LOADED=False" in completed.stdout


def _publication_context(tmp_path: Path, smoke_protocol, run_id: str):
    protocol, _ = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    layout = create_new_run(tmp_path / "results", run_id, protocol)
    freeze_run_metadata(layout, protocol, project_root=Path(__file__).parents[1])
    manifest = (
        build_job_manifest(protocol, stages=("validate",))
        + build_job_manifest(protocol, stages=("preprocess",))
    )
    status = StatusStore(layout, manifest)
    contexts = _contexts(protocol, layout, manifest, status)
    stage_module.run_validate_job(
        next(item for item in contexts if item.job.stage == "validate")
    )
    preprocess = next(
        item
        for item in contexts
        if item.job.stage == "preprocess" and item.job.dataset_id == "spambase"
    )
    return layout, preprocess


def _tree_bytes(path: Path) -> dict[str, bytes]:
    if not path.exists():
        return {}
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_preprocess_manifest_conflict_does_not_change_published_directory(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, context = _publication_context(tmp_path, smoke_protocol, "conflict-before-write")
    stage_module.run_preprocess_job(context)
    final = layout.root / "preprocess" / "spambase" / "s42"
    before_directory = _tree_bytes(final)
    before_manifest = (layout.root / "data_manifest.csv").read_bytes()
    real_prepare = data_protocol_module.prepare_split

    def changed_prepare(*args, **kwargs):
        split = real_prepare(*args, **kwargs)
        changed = split.X_train.copy(deep=True)
        changed.iloc[0, 0] += 1.0
        return replace(split, X_train=changed)

    monkeypatch.setattr(data_protocol_module, "prepare_split", changed_prepare)

    with pytest.raises(ValueError, match="conflict|frozen"):
        stage_module.run_preprocess_job(context)

    assert _tree_bytes(final) == before_directory
    assert (layout.root / "data_manifest.csv").read_bytes() == before_manifest


@pytest.mark.parametrize("failure", ["val.csv", "test.csv", "split_metadata.json"])
def test_preprocess_staging_failure_leaves_no_formal_directory_or_manifest(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    layout, context = _publication_context(tmp_path, smoke_protocol, f"stage-fail-{failure[:3]}")
    real_bytes = stage_module._atomic_write_bytes
    real_json = stage_module.atomic_write_json

    def fail_bytes(path: Path, payload: bytes) -> None:
        if Path(path).name == failure:
            raise OSError(f"injected {failure}")
        real_bytes(path, payload)

    def fail_json(path: Path, value: object) -> None:
        if Path(path).name == failure:
            raise OSError(f"injected {failure}")
        real_json(path, value)

    monkeypatch.setattr(stage_module, "_atomic_write_bytes", fail_bytes)
    monkeypatch.setattr(stage_module, "atomic_write_json", fail_json)

    with pytest.raises(OSError, match="injected"):
        stage_module.run_preprocess_job(context)

    assert not (layout.root / "preprocess" / "spambase" / "s42").exists()
    assert not (layout.root / "data_manifest.csv").exists()


def test_preprocess_manifest_replace_failure_rolls_back_formal_snapshot(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, context = _publication_context(tmp_path, smoke_protocol, "manifest-fail")
    real_replace = protocol_module.os.replace

    def fail_manifest_replace(source: Path, destination: Path) -> None:
        if Path(destination) == layout.root / "data_manifest.csv":
            raise OSError("injected manifest replace")
        real_replace(source, destination)

    monkeypatch.setattr(protocol_module.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="manifest replace"):
        stage_module.run_preprocess_job(context)

    assert not (layout.root / "preprocess" / "spambase" / "s42").exists()
    assert not (layout.root / "data_manifest.csv").exists()


def test_preprocess_identical_retry_does_not_read_test_or_republish(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, context = _publication_context(tmp_path, smoke_protocol, "retry-idempotent")
    first = stage_module.run_preprocess_job(context)
    final = layout.root / "preprocess" / "spambase" / "s42"
    test_path = final / "test.csv"
    before = _tree_bytes(final)
    original_read_bytes = Path.read_bytes
    original_open = Path.open

    def forbid_write(*args, **kwargs):
        raise AssertionError("identical retry attempted to republish")

    def guard_read_bytes(path: Path) -> bytes:
        if Path(path) == test_path:
            raise AssertionError("identical retry read the formal test CSV")
        return original_read_bytes(path)

    def guard_open(path: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if Path(path) == test_path and "r" in mode:
            raise AssertionError("identical retry opened the formal test CSV")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(stage_module, "_atomic_write_bytes", forbid_write)
    monkeypatch.setattr(stage_module, "atomic_write_json", forbid_write)
    monkeypatch.setattr(Path, "read_bytes", guard_read_bytes)
    monkeypatch.setattr(Path, "open", guard_open)
    second = stage_module.run_preprocess_job(context)
    monkeypatch.undo()

    assert second.metrics == first.metrics
    assert _tree_bytes(final) == before


@pytest.mark.parametrize(
    "artifact", ["train.csv", "val.csv", "split_metadata.json"]
)
def test_preprocess_identical_manifest_rejects_tampered_non_test_artifact(
    tmp_path: Path, smoke_protocol, artifact: str
) -> None:
    layout, context = _publication_context(
        tmp_path, smoke_protocol, f"retry-tampered-{artifact[:3]}"
    )
    stage_module.run_preprocess_job(context)
    final = layout.root / "preprocess" / "spambase" / "s42"
    artifact_path = final / artifact
    payload = artifact_path.read_bytes()
    artifact_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    before = _tree_bytes(final)

    with pytest.raises(RuntimeError, match="snapshot|artifact|manifest"):
        stage_module.run_preprocess_job(context)

    assert _tree_bytes(final) == before


@pytest.mark.parametrize("orphan", ["final", "staging"])
def test_preprocess_without_manifest_fails_closed_on_orphan_snapshot(
    tmp_path: Path, smoke_protocol, orphan: str
) -> None:
    layout, context = _publication_context(tmp_path, smoke_protocol, f"orphan-{orphan}")
    parent = layout.root / "preprocess" / "spambase"
    path = parent / ("s42" if orphan == "final" else ".s42.staging")
    path.mkdir(parents=True)
    (path / "sentinel.bin").write_bytes(b"orphan")
    before = _tree_bytes(path)

    with pytest.raises(RuntimeError, match="orphan|staging|manifest|snapshot"):
        stage_module.run_preprocess_job(context)

    assert _tree_bytes(path) == before
    assert not (layout.root / "data_manifest.csv").exists()


def test_legacy_preprocessors_are_import_safe_raw_cache_only_scripts() -> None:
    directory = Path(__file__).parents[1] / "experiments" / "preprocess_data"
    expected = {
        "preprocess_abalone.py",
        "preprocess_Credit-Card-Fraud.py",
        "preprocess_crimedata.py",
        "preprocess_Gisette.py",
        "preprocess_ISOLET.py",
        "preprocess_Madelon.py",
        "preprocess_Spambase.py",
    }
    paths = tuple(sorted(directory.glob("preprocess_*.py")))
    assert {path.name for path in paths} == expected

    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        assert "prepare_split" not in source, path.name
        assert "processed_v2" not in source, path.name
        assert "SEEDS" not in source, path.name
        assert not any(
            isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)
        ), path.name
        csv_writes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_csv"
        ]
        assert csv_writes, path.name
        assert all(
            call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "raw_path"
            for call in csv_writes
        ), path.name
        assert any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(
                isinstance(comparator, ast.Constant)
                and comparator.value == "__main__"
                for comparator in node.test.comparators
            )
            for node in ast.walk(tree)
        ), path.name


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


def test_cli_does_not_abbreviate_help_option() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["--he"])

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

    assert main(arguments) == 0

    run_root = output_root / "cli-run"
    first_log = (run_root / "output.log").read_bytes()
    assert Path.cwd() == cwd
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert main(arguments + ["--resume"]) == 0
    assert (run_root / "output.log").read_bytes().startswith(first_log)
    assert len((run_root / "output.log").read_bytes()) > len(first_log)


def test_all_stops_before_stages_depending_on_unregistered_future_executors(
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
    stages = {job.stage for job in manifest}
    assert {"validate", "preprocess", "estimator", "removal", "alpha"} <= stages
    assert "nsga2" not in stages
    assert all(
        (output_root / "blocked-all" / job.output_relpath).is_file()
        for job in manifest
        if job.stage in {"validate", "preprocess"}
    )
    assert all(
        not (output_root / "blocked-all" / job.output_relpath).exists()
        for job in manifest
        if job.stage in {"estimator", "removal", "alpha"}
    )
