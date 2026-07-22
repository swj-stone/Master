"""Behavioral tests for audited ICOMP 2026 split access."""

from __future__ import annotations

import ast
import json
import hashlib
import os
import subprocess
import sys
import warnings
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

import minppi_mofs.experiment_protocol as protocol_module
import minppi_mofs.data_protocol as data_protocol_module
import minppi_mofs.min_ppi as min_ppi_module
import minppi_mofs.shapg_integration as guidance_module
import minppi_mofs.voting_game as voting_game_module
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


def test_audited_test_access_validates_manifest_sha_during_its_single_read(
    tmp_path: Path, prepared_paths: PreparedSplitPaths
) -> None:
    audit = _authorized_audit(
        tmp_path / "run-sha" / "protocol_audit.json",
        prepared_paths,
        job_id="job-removal-sha",
        stage="removal",
    )
    access = AuditedSplitAccess(
        prepared_paths, stage="removal", job_id="job-removal-sha", audit=audit
    )
    digest = hashlib.sha256(prepared_paths.test_csv.read_bytes()).hexdigest()
    X_test, y_test = access.load_test(
        reason="final_removal_evaluation", expected_sha256=digest
    )
    assert len(X_test) == len(y_test) == 2

    with pytest.raises(ValueError, match="SHA-256"):
        access.load_test(
            reason="final_removal_evaluation", expected_sha256="0" * 64
        )


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
            if dataset.task == "regression":
                label_summaries = {
                    "train": {"count": 6, "max": 3.0, "mean": 1.5, "min": 0.0, "std": 1.0},
                    "val": {"count": 2, "max": 2.0, "mean": 1.5, "min": 1.0, "std": 0.5},
                    "test": {"count": 2, "max": 2.0, "mean": 1.5, "min": 1.0, "std": 0.5},
                }
            else:
                label_summaries = {
                    "train": {
                        "classes": [
                            {"count": 3, "label": 0},
                            {"count": 3, "label": 1},
                        ],
                        "count": 6,
                    },
                    "val": {
                        "classes": [
                            {"count": 1, "label": 0},
                            {"count": 1, "label": 1},
                        ],
                        "count": 2,
                    },
                    "test": {
                        "classes": [
                            {"count": 1, "label": 0},
                            {"count": 1, "label": 1},
                        ],
                        "count": 2,
                    },
                }
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
                    "train_label_summary_json": json.dumps(
                        label_summaries["train"], sort_keys=True, separators=(",", ":")
                    ),
                    "val_label_summary_json": json.dumps(
                        label_summaries["val"], sort_keys=True, separators=(",", ":")
                    ),
                    "test_label_summary_json": json.dumps(
                        label_summaries["test"], sort_keys=True, separators=(",", ":")
                    ),
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
    assert checks["registered_stages"] == [
        "ablation", "alpha", "estimator", "preprocess", "removal", "validate"
    ]
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
        "estimator": stage_module.run_estimator_job,
        "removal": stage_module.run_removal_job,
        "alpha": stage_module.run_alpha_job,
        "ablation": stage_module.run_ablation_job,
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


def _prepared_estimator_context(tmp_path: Path, smoke_protocol, run_id: str):
    protocol, _ = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    protocol = replace(protocol, datasets=(protocol.datasets[0],))
    layout = create_new_run(tmp_path / "results", run_id, protocol)
    freeze_run_metadata(layout, protocol, project_root=Path(__file__).parents[1])
    manifest = (
        build_job_manifest(protocol, stages=("validate",))
        + build_job_manifest(protocol, stages=("preprocess",))
        + build_job_manifest(
            protocol,
            stages=("estimator",),
            data_manifest=(
                {
                    "dataset_id": "spambase",
                    "seed": 42,
                    "selected_feature_count": 2,
                },
            ),
        )
    )
    status = StatusStore(layout, manifest)
    contexts = _contexts(protocol, layout, manifest, status)
    original_executors = dict(stage_module.STAGE_EXECUTORS)
    stage_module.STAGE_EXECUTORS["estimator"] = lambda item: _result(item.job)
    try:
        stage_module.run_validate_job(
            next(item for item in contexts if item.job.stage == "validate")
        )
    finally:
        stage_module.STAGE_EXECUTORS.clear()
        stage_module.STAGE_EXECUTORS.update(original_executors)
    stage_module.run_preprocess_job(
        next(item for item in contexts if item.job.stage == "preprocess")
    )
    return layout, next(item for item in contexts if item.job.stage == "estimator")


def test_prepared_data_loader_verifies_rich_manifest_without_touching_test(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, context = _prepared_estimator_context(tmp_path, smoke_protocol, "prepared")
    test_path = layout.root / "preprocess" / "spambase" / "s42" / "test.csv"
    original_open = Path.open
    original_read_bytes = Path.read_bytes
    original_stat = Path.stat

    def guard_open(path: Path, *args, **kwargs):
        if Path(path) == test_path:
            raise AssertionError("estimator opened test.csv")
        return original_open(path, *args, **kwargs)

    def guard_read_bytes(path: Path) -> bytes:
        if Path(path) == test_path:
            raise AssertionError("estimator read test.csv")
        return original_read_bytes(path)

    def guard_stat(path: Path, *args, **kwargs):
        if Path(path) == test_path:
            raise AssertionError("estimator statted test.csv")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_open)
    monkeypatch.setattr(Path, "read_bytes", guard_read_bytes)
    monkeypatch.setattr(Path, "stat", guard_stat)
    X_train, y_train, X_val, y_val, row = stage_module._load_prepared_train_validation(
        context
    )

    assert list(X_train) == list(X_val) == json.loads(row["selected_features_json"])
    assert len(X_train) == row["train_row_count"]
    assert len(X_val) == row["val_row_count"]
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)


def test_prepared_data_loader_fails_closed_on_hash_or_schema_drift(
    tmp_path: Path, smoke_protocol
) -> None:
    layout, context = _prepared_estimator_context(tmp_path, smoke_protocol, "drift")
    train_path = layout.root / "preprocess" / "spambase" / "s42" / "train.csv"
    original = train_path.read_bytes()
    train_path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="SHA-256|hash|drift"):
        stage_module._load_prepared_train_validation(context)

    train_path.write_bytes(original)
    frame = pd.read_csv(train_path)
    frame = frame[[frame.columns[1], frame.columns[0], frame.columns[-1]]]
    frame.to_csv(train_path, index=False, lineterminator="\n")
    manifest_path = layout.root / "data_manifest.csv"
    text = manifest_path.read_text(encoding="utf-8")
    old_hash = hashlib.sha256(original).hexdigest()
    new_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    manifest_path.write_text(text.replace(old_hash, new_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="feature|column|order"):
        stage_module._load_prepared_train_validation(context)


def _prepared_loader_case(
    tmp_path: Path,
    context: StageContext,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task: str,
    train_labels: list[float | int],
    val_labels: list[float | int],
    summary_override: tuple[str, dict[str, object]] | None = None,
):
    prefix = Path("preprocess") / context.job.dataset_id / f"s{context.job.seed}"
    directory = context.layout.root / prefix
    directory.mkdir(parents=True, exist_ok=True)
    columns = ["f0", "f1"]
    train = pd.DataFrame(
        {"f0": np.arange(len(train_labels), dtype=float), "f1": 1.0, "target": train_labels}
    )
    val = pd.DataFrame(
        {"f0": np.arange(len(val_labels), dtype=float), "f1": 2.0, "target": val_labels}
    )
    payloads = {
        "train": train.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        "val": val.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    }
    for partition, payload in payloads.items():
        (directory / f"{partition}.csv").write_bytes(payload)
    summaries = {
        "train": stage_module._label_summary(np.asarray(train_labels), task),
        "val": stage_module._label_summary(np.asarray(val_labels), task),
    }
    if summary_override is not None:
        summaries[summary_override[0]] = summary_override[1]
    row = {
        "dataset_id": context.job.dataset_id,
        "seed": context.job.seed,
        "task": task,
        "target": "target",
        "selected_features_json": json.dumps(columns, separators=(",", ":")),
        "train_relpath": (prefix / "train.csv").as_posix(),
        "val_relpath": (prefix / "val.csv").as_posix(),
        "train_file_sha256": hashlib.sha256(payloads["train"]).hexdigest(),
        "val_file_sha256": hashlib.sha256(payloads["val"]).hexdigest(),
        "train_row_count": len(train),
        "val_row_count": len(val),
        "train_label_summary_json": json.dumps(summaries["train"], sort_keys=True, separators=(",", ":")),
        "val_label_summary_json": json.dumps(summaries["val"], sort_keys=True, separators=(",", ":")),
    }
    monkeypatch.setattr(stage_module, "read_rich_data_manifest", lambda *args: (row,))
    return directory / "test.csv"


@pytest.mark.parametrize(
    ("task", "train_labels", "val_labels", "message"),
    [
        ("binary", [0, 0, 0, 0], [0, 0], "binary|class"),
        ("binary", [2, 3, 2, 3], [2, 3], "binary|0.*1|class"),
        ("multiclass", [1, 2, 1, 2], [1, 2], "multiclass|zero|contiguous"),
        ("multiclass", [0, 2, 0, 2], [0, 2], "multiclass|contiguous"),
        ("multiclass", [0, 1, 0, 1], [2, 2], "validation|unseen|subset"),
    ],
)
def test_prepared_loader_rejects_invalid_classification_label_shape_before_model_work(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    train_labels: list[int],
    val_labels: list[int],
    message: str,
) -> None:
    _, _, _, contexts, _ = _tiny_estimator_setup(tmp_path, smoke_protocol)
    context = contexts[0]
    _prepared_loader_case(
        tmp_path,
        context,
        monkeypatch,
        task=task,
        train_labels=train_labels,
        val_labels=val_labels,
    )

    with pytest.raises(ValueError, match=message):
        stage_module._load_prepared_train_validation(context)


def test_prepared_loader_recomputes_summary_with_regression_roundtrip_tolerance_and_no_test_access(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, contexts, _ = _tiny_estimator_setup(tmp_path, smoke_protocol)
    context = contexts[0]
    test_path = _prepared_loader_case(
        tmp_path,
        context,
        monkeypatch,
        task="regression",
        train_labels=[0.1, 0.2, 0.3, 0.4],
        val_labels=[0.15, 0.35],
    )
    original_open = Path.open
    original_read_bytes = Path.read_bytes
    original_stat = Path.stat

    def guard_open(path: Path, *args, **kwargs):
        if Path(path) == test_path:
            raise AssertionError("prepared loader opened test.csv")
        return original_open(path, *args, **kwargs)

    def guard_read_bytes(path: Path) -> bytes:
        if Path(path) == test_path:
            raise AssertionError("prepared loader read test.csv")
        return original_read_bytes(path)

    def guard_stat(path: Path, *args, **kwargs):
        if Path(path) == test_path:
            raise AssertionError("prepared loader statted test.csv")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_open)
    monkeypatch.setattr(Path, "read_bytes", guard_read_bytes)
    monkeypatch.setattr(Path, "stat", guard_stat)
    stage_module._load_prepared_train_validation(context)


def test_prepared_loader_rejects_train_or_validation_summary_mismatch(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, contexts, _ = _tiny_estimator_setup(tmp_path, smoke_protocol)
    context = contexts[0]
    _prepared_loader_case(
        tmp_path,
        context,
        monkeypatch,
        task="binary",
        train_labels=[0, 1, 0, 1],
        val_labels=[0, 1],
        summary_override=(
            "val", {"classes": [{"count": 2, "label": 0}], "count": 2}
        ),
    )

    with pytest.raises(ValueError, match="summary|manifest"):
        stage_module._load_prepared_train_validation(context)


def _tiny_estimator_setup(tmp_path: Path, smoke_protocol, *, exact_max_features=20):
    dataset = replace(smoke_protocol.datasets[0], max_features=5)
    protocol = replace(
        smoke_protocol,
        datasets=(dataset,),
        exact_max_features=exact_max_features,
        model_params={
            "n_estimators": 1,
            "max_depth": 1,
            "learning_rate": 0.3,
            "n_jobs": 1,
        },
    )
    projection = (
        {"dataset_id": dataset.dataset_id, "seed": 42, "selected_feature_count": 5},
    )
    manifest = build_job_manifest(
        protocol, stages=("estimator",), data_manifest=projection
    )
    layout = create_new_run(tmp_path / "results", f"est-{exact_max_features}", protocol)
    status = StatusStore(layout, manifest)
    contexts = _contexts(protocol, layout, manifest, status)
    rng = np.random.default_rng(108)
    X = pd.DataFrame(rng.normal(size=(36, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + 0.4 * X["f1"] > 0).astype(int).to_numpy()
    prepared = (X.iloc[:24].copy(), y[:24], X.iloc[24:].copy(), y[24:])
    row = {"selected_features_json": json.dumps(list(X.columns))}
    return protocol, layout, status, contexts, (*prepared, row)


def test_tiny_estimator_jobs_write_costs_and_exact_agreement(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, layout, status, contexts, prepared = _tiny_estimator_setup(
        tmp_path, smoke_protocol
    )
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    by_method = {item.job.method: item for item in contexts}
    records: dict[str, ResultRecord] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for method in ("exact", "random_mc", "mi_correlation_guided", "hybrid"):
            context = by_method[method]
            record = stage_module.run_estimator_job(context)
            records[method] = record
            atomic_write_json(layout.root / context.job.output_relpath, asdict(record))
            status.start(context.job.job_id)
            status.succeed(context.job.job_id)

    assert not any("deprecated" in str(item.message).lower() for item in caught)
    agreement = {
        "harmful_precision", "harmful_recall", "harmful_f1", "score_agreement"
    }
    for method, record in records.items():
        assert agreement <= set(record.metrics)
        assert all(0.0 <= float(record.metrics[key]) <= 1.0 for key in agreement)
        assert record.metrics["budget"] == 2
        assert int(record.metrics["model_fit_count"]) >= 1
        assert int(record.metrics["coalition_count"]) >= 1
        assert float(record.metrics["runtime_s"]) >= 0.0
        path = layout.root / record.artifacts["scores"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert path == (
            layout.root
            / f"estimator/spambase/s42/{method}/b2/scores.json"
        )
        assert payload["method"] == method
        assert payload["budget"] == 2
        assert [item["feature"] for item in payload["scores"]] == [f"f{i}" for i in range(5)]
        assert all(item["score"] in {-1, 0, 1} for item in payload["scores"])
        assert payload["model_fit_count"] == record.metrics["model_fit_count"]
        assert payload["coalition_count"] == record.metrics["coalition_count"]
        if method == "exact":
            accepted = stage_module._read_score_artifact(
                path,
                expected_job=by_method[method].job,
                budget=2,
                expected_features=tuple(f"f{index}" for index in range(5)),
            )
            assert tuple(accepted) == tuple(f"f{index}" for index in range(5))
            assert payload["coalition_count"] == payload["model_fit_count"] + 1
    assert all(records["exact"].metrics[key] == 1.0 for key in agreement)
    assert all(records["hybrid"].metrics[key] == 1.0 for key in agreement)


def test_estimator_omits_agreement_when_manifest_has_no_exact_job(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, contexts, prepared = _tiny_estimator_setup(
        tmp_path, smoke_protocol, exact_max_features=4
    )
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    context = next(item for item in contexts if item.job.method == "random_mc")

    record = stage_module.run_estimator_job(context)

    assert not {
        "harmful_precision", "harmful_recall", "harmful_f1", "score_agreement"
    }.intersection(record.metrics)


def _tiny_ablation_setup(tmp_path: Path, smoke_protocol):
    dataset = replace(
        smoke_protocol.datasets[0],
        dataset_id="isolet",
        display_name="Tiny ISOLET",
        task="multiclass",
        utility="ovr_auc",
        max_features=4,
    )
    protocol = replace(
        smoke_protocol,
        datasets=(dataset,),
        model_params={
            "n_estimators": 1,
            "max_depth": 1,
            "learning_rate": 0.3,
            "n_jobs": 1,
        },
    )
    manifest = build_job_manifest(
        protocol,
        stages=("ablation",),
        data_manifest=(
            {"dataset_id": "isolet", "seed": 42, "selected_feature_count": 4},
        ),
    )
    layout = create_new_run(tmp_path / "results", "utility-pair", protocol)
    status = StatusStore(layout, manifest)
    contexts = _contexts(protocol, layout, manifest, status)
    rng = np.random.default_rng(56)
    X = pd.DataFrame(rng.normal(size=(45, 4)), columns=[f"f{i}" for i in range(4)])
    y = np.tile(np.arange(3), 15)
    prepared = (X.iloc[:30].copy(), y[:30], X.iloc[30:].copy(), y[30:])
    row = {"selected_features_json": json.dumps(list(X.columns))}
    return layout, contexts, (*prepared, row)


def test_utility_ablation_publishes_and_reuses_one_shared_cache_pair(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contexts, prepared = _tiny_ablation_setup(tmp_path, smoke_protocol)
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    monkeypatch.setattr(
        stage_module,
        "read_rich_data_manifest",
        lambda *args: (
            {
                "dataset_id": "isolet",
                "seed": 42,
                "selected_features_json": '["f0","f1","f2","f3"]',
            },
        ),
    )
    constructed: list[voting_game_module.AccFullCache] = []
    real_cache = voting_game_module.AccFullCache

    class CountingCache(real_cache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructed.append(self)

    monkeypatch.setattr(voting_game_module, "AccFullCache", CountingCache)
    utility_contexts = {
        item.job.parameters["value"]: item
        for item in contexts
        if item.job.method == "utility"
    }

    accuracy = stage_module.run_ablation_job(utility_contexts["accuracy"])
    paired_path = layout.root / accuracy.artifacts["paired_scores"]
    before = paired_path.read_bytes()
    auc = stage_module.run_ablation_job(utility_contexts["ovr_auc"])

    assert len(constructed) == 1
    assert paired_path == (
        layout.root / "ablation/isolet/s42/utility/paired/b2/scores.json"
    )
    assert paired_path.read_bytes() == before
    assert accuracy.metrics["shared_group_id"] == auc.metrics["shared_group_id"]
    assert accuracy.metrics["shared_group_model_fit_count"] == auc.metrics[
        "shared_group_model_fit_count"
    ]
    assert accuracy.metrics["shared_group_coalition_count"] == auc.metrics[
        "shared_group_coalition_count"
    ]
    payload = json.loads(before)
    assert set(payload["scores"]) == {"accuracy", "ovr_auc"}
    assert payload["secondary_model_fit_delta"] == 0
    assert payload["secondary_coalition_delta"] == 0
    assert payload["shared_group_model_fit_count"] == accuracy.metrics[
        "shared_group_model_fit_count"
    ]
    assert payload["shared_group_coalition_count"] == accuracy.metrics[
        "shared_group_coalition_count"
    ]


def test_utility_ablation_reuse_rejects_manifest_feature_order_conflict(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contexts, prepared = _tiny_ablation_setup(tmp_path, smoke_protocol)
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    monkeypatch.setattr(
        stage_module,
        "read_rich_data_manifest",
        lambda *args: (
            {
                "dataset_id": "isolet",
                "seed": 42,
                "selected_features_json": '["f0","f1","f2","f3"]',
            },
        ),
    )
    utility_contexts = {
        item.job.parameters["value"]: item
        for item in contexts
        if item.job.method == "utility"
    }
    first = stage_module.run_ablation_job(utility_contexts["accuracy"])
    path = layout.root / first.artifacts["paired_scores"]
    document = json.loads(path.read_text(encoding="utf-8"))
    for metric in ("accuracy", "ovr_auc"):
        document["scores"][metric][0]["feature"] = "forged"
    atomic_write_json(path, document)

    with pytest.raises(ValueError, match="feature|manifest|identity"):
        stage_module.run_ablation_job(utility_contexts["ovr_auc"])


def test_guidance_ablation_maps_components_with_equal_budget_seed_and_schedule(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contexts, prepared = _tiny_ablation_setup(tmp_path, smoke_protocol)
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    mi = {f"f{i}": float(4 - i) for i in range(4)}
    neighbors = {f"f{i}": [f"f{(i + 1) % 4}"] for i in range(4)}
    monkeypatch.setattr(
        guidance_module, "compute_mi_guidance_signal", lambda *args, **kwargs: mi
    )
    monkeypatch.setattr(
        guidance_module, "get_correlation_neighbors", lambda *args, **kwargs: neighbors
    )
    monkeypatch.setattr(
        guidance_module,
        "build_mi_correlation_guidance",
        lambda *args, **kwargs: (mi, neighbors),
    )
    calls: list[dict[str, object]] = []

    def fake_algorithm_b(*args, **kwargs):
        calls.append(kwargs)
        value = (
            1
            if kwargs["mi_scores"] is None and kwargs["correlation_neighbors"] is None
            else 0
            if kwargs["correlation_neighbors"] is None
            else -1
        )
        return {feature: value for feature in kwargs["X_train"].columns}

    monkeypatch.setattr(
        min_ppi_module, "estimate_minppi_mi_correlation_guided", fake_algorithm_b
    )
    guidance_contexts = [item for item in contexts if item.job.method == "guidance"]
    records = [stage_module.run_ablation_job(item) for item in guidance_contexts]

    expected = {
        "random": (None, None),
        "mi": (mi, None),
        "correlation": (None, neighbors),
        "mi_correlation": (mi, neighbors),
    }
    by_value = {item.job.parameters["value"]: item for item in guidance_contexts}
    schedule_digests: set[str] = set()
    schedules: list[object] = []
    for call, record in zip(calls, records, strict=True):
        value = record.metrics["value"]
        assert (call["mi_scores"], call["correlation_neighbors"]) == expected[value]
        assert call["K_adv"] == record.metrics["budget"] == 2
        assert call["seed"] == by_value[value].job.seed == 42
        schedules.append(call["perturbation_schedule"])
        payload = json.loads(
            (layout.root / record.artifacts["scores"]).read_text(encoding="utf-8")
        )
        assert payload["value"] == value
        assert payload["budget"] == 2
        assert payload["schedule_seed"] == 42
        schedule_digests.add(payload["schedule_digest"])
        assert payload["method"] == "mi_correlation_guided"
        assert payload["model_fit_count"] >= 1
        assert payload["coalition_count"] >= 1
    assert len(schedule_digests) == 1
    assert all(schedule == schedules[0] for schedule in schedules)


def test_estimator_artifact_is_identical_across_python_hash_seeds(tmp_path: Path) -> None:
    script = r'''
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import experiments.icomp2026_stages as stages
from experiments.icomp2026_stages import StageContext
from minppi_mofs.experiment_protocol import (
    AuditLog, StatusStore, build_job_manifest, create_new_run, load_resolved_protocol,
)

root = Path(sys.argv[1])
protocol = load_resolved_protocol(
    Path("configs"), seeds=(42,), backend="cpu", smoke=True
)
dataset = replace(protocol.datasets[0], max_features=8)
protocol = replace(
    protocol,
    datasets=(dataset,),
    estimator_methods=("mi_correlation_guided",),
    estimator_budgets=(3,),
    model_params={"n_estimators": 1, "max_depth": 1, "learning_rate": 0.3, "n_jobs": 1},
)
manifest = build_job_manifest(
    protocol,
    stages=("estimator",),
    data_manifest=({"dataset_id": dataset.dataset_id, "seed": 42, "selected_feature_count": 8},),
)
layout = create_new_run(root, "hash-stable", protocol)
status = StatusStore(layout, manifest)
context = StageContext(
    protocol, layout, manifest[0], status,
    AuditLog.from_manifest(layout.audit_path, manifest),
)
values = np.arange(32 * 8, dtype=float).reshape(32, 8)
X = pd.DataFrame(values, columns=[f"f{i}" for i in range(8)])
y = np.tile([0, 1], 16)
row = {"selected_features_json": json.dumps(list(X.columns))}
stages._load_prepared_train_validation = lambda context: (
    X.iloc[:24].copy(), y[:24], X.iloc[24:].copy(), y[24:], row
)
record = stages.run_estimator_job(context)
document = json.loads((layout.root / record.artifacts["scores"]).read_text(encoding="utf-8"))
print(json.dumps({
    "scores": document["scores"],
    "model_fit_count": document["model_fit_count"],
    "coalition_count": document["coalition_count"],
    "schedule_digest": document["schedule_digest"],
}, sort_keys=True))
'''
    outputs = []
    for hash_seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / f"h{hash_seed}")],
            cwd=Path(__file__).parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    assert outputs[0] == outputs[1]


def test_dispatcher_ablation_maps_direct_estimators_and_exact_is_manifest_gated(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, contexts, prepared = _tiny_ablation_setup(tmp_path, smoke_protocol)
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def fake(name: str, score: int):
        def run(*args, **kwargs):
            calls.append((name, kwargs))
            return {feature: score for feature in kwargs["X_train"].columns}
        return run

    monkeypatch.setattr(min_ppi_module, "estimate_minppi_exact", fake("exact", 1))
    monkeypatch.setattr(min_ppi_module, "estimate_minppi_random_mc", fake("random_mc", 0))
    monkeypatch.setattr(
        min_ppi_module,
        "estimate_minppi_mi_correlation_guided",
        fake("mi_correlation_guided", -1),
    )
    monkeypatch.setattr(min_ppi_module, "estimate_minppi_hybrid", fake("hybrid", 0))
    monkeypatch.setattr(
        guidance_module,
        "build_mi_correlation_guidance",
        lambda *args, **kwargs: ({"f0": 1.0}, {"f0": ["f1"]}),
    )
    dispatcher_contexts = [item for item in contexts if item.job.method == "dispatcher"]
    records = [stage_module.run_ablation_job(item) for item in dispatcher_contexts]

    assert [name for name, _ in calls] == [
        item.job.parameters["value"] for item in dispatcher_contexts
    ]
    for (name, call), record in zip(calls, records, strict=True):
        assert record.metrics["value"] == name
        assert record.metrics["budget"] == 2
        if name == "random_mc":
            assert call["K"] == 2
        elif name == "mi_correlation_guided":
            assert call["K_adv"] == 2
        elif name == "hybrid":
            assert call["K"] == call["K_adv"] == 2
        payload = json.loads(
            (layout.root / record.artifacts["scores"]).read_text(encoding="utf-8")
        )
        assert payload["value"] == name

    exact_context = next(
        item
        for item in dispatcher_contexts
        if item.job.parameters["value"] == "exact"
    )
    protocol = replace(exact_context.protocol, exact_max_features=3)
    manifest = build_job_manifest(
        protocol,
        stages=("ablation",),
        data_manifest=(
            {"dataset_id": "isolet", "seed": 42, "selected_feature_count": 4},
        ),
    )
    assert not any(
        job.method == "dispatcher" and job.parameters["value"] == "exact"
        for job in manifest
    )
    gated_layout = create_new_run(tmp_path / "results", "exact-gated", protocol)
    gated_status = StatusStore(gated_layout, manifest)
    forged = StageContext(
        protocol,
        gated_layout,
        replace(exact_context.job, protocol_hash=manifest[0].protocol_hash),
        gated_status,
        AuditLog.from_manifest(gated_layout.audit_path, manifest),
    )
    with pytest.raises(ValueError, match="manifest|exact"):
        stage_module.run_ablation_job(forged)


def test_estimator_artifact_retry_is_identical_and_conflict_is_rejected(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, layout, _, contexts, prepared = _tiny_estimator_setup(
        tmp_path, smoke_protocol, exact_max_features=4
    )
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    context = next(item for item in contexts if item.job.method == "random_mc")
    first = stage_module.run_estimator_job(context)
    path = layout.root / first.artifacts["scores"]
    frozen = path.read_bytes()

    second = stage_module.run_estimator_job(context)
    assert second.artifacts == first.artifacts
    assert path.read_bytes() == frozen

    conflict = json.loads(frozen)
    conflict["scores"][0]["score"] = -1 if conflict["scores"][0]["score"] != -1 else 1
    atomic_write_json(path, conflict)
    conflicting_bytes = path.read_bytes()
    with pytest.raises(RuntimeError, match="conflict|frozen|identical"):
        stage_module.run_estimator_job(context)
    assert path.read_bytes() == conflicting_bytes


def _run_artifact_publish_processes(
    destination: Path, documents: tuple[dict[str, object], dict[str, object]]
) -> list[subprocess.CompletedProcess[str]]:
    script = """
import json
import sys
from pathlib import Path
import experiments.icomp2026_stages as stages

destination = Path(sys.argv[1]).resolve()
document = json.loads(sys.argv[2])
original_exists = Path.exists

def hide_destination(self):
    if self.resolve() == destination:
        return False
    return original_exists(self)

Path.exists = hide_destination
try:
    stages._publish_idempotent_json(
        destination, document, context="process-race artifact"
    )
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(4)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(destination), json.dumps(document)],
            cwd=Path(__file__).parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for document in documents
    ]
    return [
        subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        for process in processes
        for stdout, stderr in [process.communicate(timeout=30)]
    ]


def test_artifact_publication_process_race_reuses_identical_content(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "race" / "scores.json"
    document = {"scores": [1, 0, -1], "seed": 42}

    completed = _run_artifact_publish_processes(
        destination, (document, document.copy())
    )

    assert [item.returncode for item in completed] == [0, 0]
    assert json.loads(destination.read_text(encoding="utf-8")) == document
    assert not list(destination.parent.glob(".*.tmp"))


def test_artifact_publication_process_race_rejects_conflicting_content(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "race" / "scores.json"
    documents = ({"winner": "first"}, {"winner": "second"})

    completed = _run_artifact_publish_processes(destination, documents)

    assert sorted(item.returncode for item in completed) == [0, 4]
    conflict = next(item for item in completed if item.returncode == 4)
    assert "RuntimeError" in conflict.stderr
    assert json.loads(destination.read_text(encoding="utf-8")) in documents
    assert not list(destination.parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("model_fit_count", "coalition_count"),
    [
        (True, 2),
        (-1, 0),
        (0, 0),
        (1, 1),
        (2, 1),
        (1, 3),
    ],
)
def test_exact_score_artifact_rejects_invalid_cost_types_values_and_relationships(
    tmp_path: Path,
    smoke_protocol,
    model_fit_count: object,
    coalition_count: object,
) -> None:
    _, layout, _, contexts, _ = _tiny_estimator_setup(tmp_path, smoke_protocol)
    exact = next(item.job for item in contexts if item.job.method == "exact")
    path = layout.root / "exact-cost.json"
    atomic_write_json(
        path,
        {
            "protocol_hash": exact.protocol_hash,
            "job_id": exact.job_id,
            "dataset_id": exact.dataset_id,
            "seed": exact.seed,
            "method": exact.method,
            "budget": exact.parameters["budget"],
            "scores": [
                {"feature": f"f{index}", "score": 0} for index in range(5)
            ],
            "model_fit_count": model_fit_count,
            "coalition_count": coalition_count,
        },
    )

    with pytest.raises(ValueError, match="cost|fit|coalition|counter"):
        stage_module._read_score_artifact(
            path,
            expected_job=exact,
            budget=int(exact.parameters["budget"]),
            expected_features=tuple(f"f{index}" for index in range(5)),
        )


@pytest.mark.parametrize("score", [True, False])
def test_exact_score_artifact_rejects_boolean_scores(
    tmp_path: Path, smoke_protocol, score: bool
) -> None:
    _, layout, _, contexts, _ = _tiny_estimator_setup(tmp_path, smoke_protocol)
    exact = next(item.job for item in contexts if item.job.method == "exact")
    path = layout.root / "exact-bool-score.json"
    atomic_write_json(
        path,
        {
            "protocol_hash": exact.protocol_hash,
            "job_id": exact.job_id,
            "dataset_id": exact.dataset_id,
            "seed": exact.seed,
            "method": exact.method,
            "budget": exact.parameters["budget"],
            "scores": [
                {"feature": f"f{index}", "score": score if index == 0 else 0}
                for index in range(5)
            ],
            "model_fit_count": 1,
            "coalition_count": 2,
        },
    )

    with pytest.raises(ValueError, match="score|integer|boolean"):
        stage_module._read_score_artifact(
            path,
            expected_job=exact,
            budget=2,
            expected_features=tuple(f"f{index}" for index in range(5)),
        )


def test_exact_score_artifact_rejects_duplicate_feature_before_overwrite(
    tmp_path: Path, smoke_protocol
) -> None:
    _, layout, _, contexts, _ = _tiny_estimator_setup(tmp_path, smoke_protocol)
    exact = next(item.job for item in contexts if item.job.method == "exact")
    path = layout.root / "exact-duplicate-feature.json"
    scores = [
        {"feature": "f0", "score": -1},
        {"feature": "f0", "score": 1},
        *[
            {"feature": f"f{index}", "score": 0}
            for index in range(1, 5)
        ],
    ]
    atomic_write_json(
        path,
        {
            "protocol_hash": exact.protocol_hash,
            "job_id": exact.job_id,
            "dataset_id": exact.dataset_id,
            "seed": exact.seed,
            "method": exact.method,
            "budget": exact.parameters["budget"],
            "scores": scores,
            "model_fit_count": 1,
            "coalition_count": 2,
        },
    )

    with pytest.raises(ValueError, match="duplicate|feature|score"):
        stage_module._read_score_artifact(
            path,
            expected_job=exact,
            budget=2,
            expected_features=tuple(f"f{index}" for index in range(5)),
        )


@pytest.mark.parametrize("mutation", ["duplicate_key", "noncanonical", "nonfinite"])
def test_exact_score_reader_rejects_noncanonical_or_malformed_json(
    tmp_path: Path, smoke_protocol, mutation: str
) -> None:
    _, layout, _, contexts, _ = _tiny_estimator_setup(tmp_path, smoke_protocol)
    exact = next(item.job for item in contexts if item.job.method == "exact")
    path = layout.root / "exact-strict-json.json"
    document = {
        "protocol_hash": exact.protocol_hash,
        "job_id": exact.job_id,
        "dataset_id": exact.dataset_id,
        "seed": exact.seed,
        "method": exact.method,
        "budget": exact.parameters["budget"],
        "scores": [
            {"feature": f"f{index}", "score": 0} for index in range(5)
        ],
        "model_fit_count": 1,
        "coalition_count": 2,
    }
    atomic_write_json(path, document)
    canonical = path.read_text(encoding="utf-8")
    if mutation == "duplicate_key":
        path.write_text(
            canonical.replace('  "seed": 42\n', '  "seed": 42,\n  "seed": 42\n'),
            encoding="utf-8",
            newline="\n",
        )
    elif mutation == "noncanonical":
        path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    else:
        path.write_text(
            canonical.replace('  "seed": 42\n', '  "seed": NaN\n'),
            encoding="utf-8",
            newline="\n",
        )

    with pytest.raises(ValueError, match="malformed|canonical|JSON"):
        stage_module._read_score_artifact(
            path,
            expected_job=exact,
            budget=2,
            expected_features=tuple(f"f{index}" for index in range(5)),
        )


def _published_utility_pair_for_reader_test(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
):
    layout, contexts, prepared = _tiny_ablation_setup(tmp_path, smoke_protocol)
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    monkeypatch.setattr(
        stage_module,
        "read_rich_data_manifest",
        lambda *args: (
            {
                "dataset_id": "isolet",
                "seed": 42,
                "selected_features_json": '["f0","f1","f2","f3"]',
            },
        ),
    )
    utility_contexts = {
        item.job.parameters["value"]: item
        for item in contexts
        if item.job.method == "utility"
    }
    record = stage_module.run_ablation_job(utility_contexts["accuracy"])
    return layout.root / record.artifacts["paired_scores"], utility_contexts


@pytest.mark.parametrize("score", [True, False])
def test_utility_paired_reader_rejects_boolean_scores_before_refit(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    score: bool,
) -> None:
    path, contexts = _published_utility_pair_for_reader_test(
        tmp_path, smoke_protocol, monkeypatch
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scores"]["accuracy"][0]["score"] = score
    atomic_write_json(path, document)

    class RejectCacheConstruction:
        def __init__(self, *args, **kwargs):
            raise AssertionError("paired artifact validation attempted to refit")

    monkeypatch.setattr(voting_game_module, "AccFullCache", RejectCacheConstruction)
    with pytest.raises(ValueError, match="score|integer|boolean"):
        stage_module.run_ablation_job(contexts["ovr_auc"])


@pytest.mark.parametrize("mutation", ["duplicate_key", "noncanonical"])
def test_utility_paired_reader_rejects_noncanonical_json_before_refit(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    path, contexts = _published_utility_pair_for_reader_test(
        tmp_path, smoke_protocol, monkeypatch
    )
    canonical = path.read_text(encoding="utf-8")
    if mutation == "duplicate_key":
        path.write_text(
            canonical.replace('  "seed": 42,\n', '  "seed": 42,\n  "seed": 42,\n'),
            encoding="utf-8",
            newline="\n",
        )
    else:
        document = json.loads(canonical)
        path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    class RejectCacheConstruction:
        def __init__(self, *args, **kwargs):
            raise AssertionError("paired artifact validation attempted to refit")

    monkeypatch.setattr(voting_game_module, "AccFullCache", RejectCacheConstruction)
    with pytest.raises(ValueError, match="malformed|canonical|JSON"):
        stage_module.run_ablation_job(contexts["ovr_auc"])


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("plan_step_count", lambda value: value + 1),
        ("shared_group_model_fit_count", lambda value: value - 1),
        ("shared_group_coalition_count", lambda value: value + 1),
        ("secondary_model_fit_delta", lambda value: value + 1),
        ("secondary_coalition_delta", lambda value: value + 1),
    ],
)
def test_utility_paired_reuse_rejects_each_derived_plan_or_cost_field_before_refit(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutate,
) -> None:
    layout, contexts, prepared = _tiny_ablation_setup(tmp_path, smoke_protocol)
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    rich_row = {
        "dataset_id": "isolet",
        "seed": 42,
        "selected_features_json": '["f0","f1","f2","f3"]',
    }
    monkeypatch.setattr(
        stage_module, "read_rich_data_manifest", lambda *args: (rich_row,)
    )
    utility_contexts = {
        item.job.parameters["value"]: item
        for item in contexts
        if item.job.method == "utility"
    }
    first = stage_module.run_ablation_job(utility_contexts["accuracy"])
    path = layout.root / first.artifacts["paired_scores"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document[field] = mutate(document[field])
    atomic_write_json(path, document)

    class RejectCacheConstruction:
        def __init__(self, *args, **kwargs):
            raise AssertionError("paired artifact validation attempted to refit")

    monkeypatch.setattr(
        voting_game_module, "AccFullCache", RejectCacheConstruction
    )
    with pytest.raises(ValueError, match="plan|cost|cache|fit|coalition"):
        stage_module.run_ablation_job(utility_contexts["ovr_auc"])


def test_estimator_artifact_link_failure_leaves_no_partial_publication(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, layout, _, contexts, prepared = _tiny_estimator_setup(
        tmp_path, smoke_protocol, exact_max_features=4
    )
    monkeypatch.setattr(
        stage_module, "_load_prepared_train_validation", lambda context: prepared
    )
    context = next(item for item in contexts if item.job.method == "random_mc")
    destination = layout.root / "estimator/spambase/s42/random_mc/b2/scores.json"
    real_link = protocol_module.os.link

    def fail_score_link(source: Path, target: Path) -> None:
        if Path(target) == destination:
            raise OSError("injected score link")
        real_link(source, target)

    monkeypatch.setattr(protocol_module.os, "link", fail_score_link)
    with pytest.raises(OSError, match="score link"):
        stage_module.run_estimator_job(context)

    assert not destination.exists()
    assert not list(destination.parent.glob(".*.tmp"))


def test_task9_registry_contains_only_completed_stage_handlers() -> None:
    assert stage_module.STAGE_EXECUTORS == {
        "validate": stage_module.run_validate_job,
        "preprocess": stage_module.run_preprocess_job,
        "estimator": stage_module.run_estimator_job,
        "removal": stage_module.run_removal_job,
        "alpha": stage_module.run_alpha_job,
        "ablation": stage_module.run_ablation_job,
    }


class _Task9FakeCache:
    def __init__(self, features: tuple[str, ...]) -> None:
        self._coalitions = {tuple(sorted(features))}
        self._model_fit_count = 1

    @property
    def model_fit_count(self) -> int:
        return self._model_fit_count

    @property
    def coalition_count(self) -> int:
        return len(self._coalitions)

    def evaluate(self, features: set[str]) -> float:
        key = tuple(sorted(features))
        if key not in self._coalitions:
            self._coalitions.add(key)
            self._model_fit_count += bool(key)
        return 0.7 + 0.01 * len(features)


def _task9_contexts(tmp_path: Path, protocol, stage: str):
    layout = create_new_run(tmp_path / "results", f"task9-{stage}", protocol)
    manifest = build_job_manifest(protocol, stages=(stage,))
    status = StatusStore(layout, manifest)
    contexts = _contexts(protocol, layout, manifest, status)
    return layout, manifest, contexts


def _patch_task9_selection(
    monkeypatch: pytest.MonkeyPatch, *, harmful_by_alpha: bool = False
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, dict[str, int]]:
    X_train = pd.DataFrame(
        {name: np.arange(12, dtype=float) + index for index, name in enumerate(("f0", "f1", "f2", "f3"))}
    )
    y_train = np.array([0, 1] * 6)
    X_val = X_train.iloc[:4].reset_index(drop=True)
    y_val = y_train[:4]
    calls = {"loader": 0, "guidance": 0, "schedule": 0, "hybrid": 0}

    def loader(context):
        calls["loader"] += 1
        return X_train, y_train, X_val, y_val, {"test_must_not_be_touched": True}

    def guidance(*args, **kwargs):
        calls["guidance"] += 1
        return ({name: float(i) for i, name in enumerate(X_train)}, {name: [] for name in X_train})

    def schedule(*args, **kwargs):
        calls["schedule"] += 1
        return object()

    def hybrid(*args, **kwargs):
        calls["hybrid"] += 1
        if harmful_by_alpha and kwargs["alpha"] < 0.9:
            removed = {"f0"}
        elif harmful_by_alpha:
            removed = {"f1", "f3"}
        else:
            removed = {"f1", "f3"}
        return {name: (-1 if name in removed else 0) for name in X_train}

    monkeypatch.setattr(stage_module, "_load_prepared_train_validation", loader)
    monkeypatch.setattr(
        stage_module,
        "_task9_manifest_row",
        lambda context: {
            "dataset_id": context.job.dataset_id,
            "seed": context.job.seed,
            "task": context.job.parameters["task"],
            "selected_features_json": json.dumps(list(X_train.columns)),
        },
        raising=False,
    )
    monkeypatch.setattr(
        stage_module,
        "_make_job_cache",
        lambda context, X_train, *args, **kwargs: _Task9FakeCache(
            tuple(X_train.columns)
        ),
    )
    monkeypatch.setattr(guidance_module, "build_mi_correlation_guidance", guidance)
    monkeypatch.setattr(min_ppi_module, "make_algorithm_b_perturbation_schedule", schedule)
    monkeypatch.setattr(min_ppi_module, "estimate_minppi_hybrid", hybrid)
    return X_train, y_train, X_val, y_val, calls


@pytest.mark.parametrize("harmful", [True, False])
def test_removal_plan_freezes_complete_matched_scope_without_test_access(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    harmful: bool,
) -> None:
    layout, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "removal")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    X_train, _, _, _, calls = _patch_task9_selection(monkeypatch)
    from minppi_mofs.baselines import banzhaf_fs, min_shap, shap_top_k

    if not harmful:
        monkeypatch.setattr(
            min_ppi_module,
            "estimate_minppi_hybrid",
            lambda *args, **kwargs: {name: 0 for name in X_train},
        )
    monkeypatch.setattr(
        shap_top_k,
        "compute_shap_topk_importance",
        lambda *args, diagnostics, **kwargs: diagnostics.update(model_fit_count=1, coalition_count=1)
        or {"f0": 4.0, "f1": 1.0, "f2": 3.0, "f3": 2.0},
    )
    monkeypatch.setattr(
        min_shap,
        "compute_minshap_importance",
        lambda *args, diagnostics, **kwargs: diagnostics.update(model_fit_count=2, coalition_count=3)
        or {"f0": 1.0, "f1": 4.0, "f2": 2.0, "f3": 3.0},
    )
    monkeypatch.setattr(
        banzhaf_fs,
        "compute_banzhaf_importance",
        lambda *args, diagnostics, **kwargs: diagnostics.update(model_fit_count=3, coalition_count=4)
        or {"f0": 2.0, "f1": 3.0, "f2": 4.0, "f3": 1.0},
    )

    record = stage_module.run_removal_job(plan_context)
    document = json.loads((layout.root / record.artifacts["selection_plan"]).read_text(encoding="utf-8"))
    expected_k = 2 if harmful else 0
    assert document["k"] == expected_k
    assert [item["method"] for item in document["methods"]] == list(stage_module._REMOVAL_METHODS)
    for method in document["methods"]:
        expected_trials = smoke_protocol.removal["random_repeats"] if method["method"] == "random_matched" else 1
        assert len(method["trials"]) == expected_trials
        for trial in method["trials"]:
            assert len(trial["removed_features"]) == (0 if method["method"] == "full" else expected_k)
    assert calls["loader"] == 1
    assert calls["guidance"] == 1
    assert calls["schedule"] == 1
    methods_by_name = {item["method"]: item for item in document["methods"]}
    for method in stage_module._REMOVAL_METHODS:
        assert record.metrics[f"{method}_selection_model_fit_count"] == methods_by_name[method]["model_fit_count"]
        assert record.metrics[f"{method}_selection_coalition_count"] == methods_by_name[method]["coalition_count"]
        assert record.metrics[f"{method}_selection_time_s"] >= 0.0

    monkeypatch.setattr(
        stage_module,
        "_load_prepared_train_validation",
        lambda context: (_ for _ in ()).throw(AssertionError("retry refit selection")),
    )
    retry = stage_module.run_removal_job(plan_context)
    assert retry.metrics["selection_model_fit_count"] == record.metrics["selection_model_fit_count"]

    plan_path = layout.root / record.artifacts["selection_plan"]
    malformed = json.loads(plan_path.read_text(encoding="utf-8"))
    malformed["methods"][0]["trials"][0]["retained_hash"] = "0" * 64
    atomic_write_json(plan_path, malformed)
    with pytest.raises(ValueError, match="hash|plan|selection"):
        stage_module.run_removal_job(plan_context)


def test_alpha_plan_freezes_complete_grid_with_one_shared_cache_and_setup(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "alpha")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    _, _, _, _, calls = _patch_task9_selection(monkeypatch, harmful_by_alpha=True)

    record = stage_module.run_alpha_job(plan_context)
    document = json.loads((layout.root / record.artifacts["selection_plan"]).read_text(encoding="utf-8"))
    assert [item["alpha"] for item in document["entries"]] == list(smoke_protocol.alpha_grid)
    assert len({item["alpha_token"] for item in document["entries"]}) == len(smoke_protocol.alpha_grid)
    assert calls == {
        "loader": 1,
        "guidance": 1,
        "schedule": 1,
        "hybrid": len(smoke_protocol.alpha_grid),
    }
    assert not any("recommend" in key or "argmax" in key for key in document)
    assert document["initial_model_fit_count"] == 1
    assert document["initial_coalition_count"] == 1
    assert document["selection_model_fit_count"] == 1 + sum(
        entry["model_fit_count"] for entry in document["entries"]
    )
    assert document["selection_coalition_count"] == 1 + sum(
        entry["coalition_count"] for entry in document["entries"]
    )
    for entry in document["entries"]:
        assert [item["feature"] for item in entry["scores"]] == document["features"]
        assert entry["model_fit_count"] <= entry["coalition_count"]
        assert record.metrics[f"{entry['alpha_token']}_selection_time_s"] >= 0.0

    monkeypatch.setattr(
        stage_module,
        "_load_prepared_train_validation",
        lambda context: (_ for _ in ()).throw(AssertionError("retry refit selection")),
    )
    retry = stage_module.run_alpha_job(plan_context)
    assert retry.metrics["selection_model_fit_count"] == record.metrics["selection_model_fit_count"]

    plan_path = layout.root / record.artifacts["selection_plan"]
    malformed = json.loads(plan_path.read_text(encoding="utf-8"))
    malformed["entries"].pop()
    atomic_write_json(plan_path, malformed)
    with pytest.raises(ValueError, match="grid|plan|selection"):
        stage_module.run_alpha_job(plan_context)


def _mark_task9_plan_success(context: StageContext, record: ResultRecord) -> None:
    atomic_write_json(context.layout.root / context.job.output_relpath, asdict(record))
    context.status.start(context.job.job_id)
    context.status.succeed(context.job.job_id)


def _task9_evaluation_loader(features: tuple[str, ...]):
    X_train = pd.DataFrame(
        {feature: np.arange(8, dtype=float) + index for index, feature in enumerate(features)}
    )
    y_train = np.array([0, 1] * 4)
    X_val = X_train.iloc[:4].reset_index(drop=True)
    y_val = y_train[:4]
    row = {
        "selected_features_json": json.dumps(list(features)),
        "task": "binary",
    }
    return X_train, y_train, X_val, y_val, row


def test_random_removal_evaluation_reads_test_once_emits_raw_trials_and_reuses_retry(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "removal")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    _patch_task9_selection(monkeypatch)
    from minppi_mofs.baselines import banzhaf_fs, min_shap, shap_top_k

    scores = {"f0": 4.0, "f1": 1.0, "f2": 3.0, "f3": 2.0}
    for module, name in (
        (shap_top_k, "compute_shap_topk_importance"),
        (min_shap, "compute_minshap_importance"),
        (banzhaf_fs, "compute_banzhaf_importance"),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args, diagnostics, **kwargs: diagnostics.update(model_fit_count=1, coalition_count=1)
            or scores,
        )
    plan_record = stage_module.run_removal_job(plan_context)
    _mark_task9_plan_success(plan_context, plan_record)
    random_context = next(item for item in contexts if item.job.method == "random_matched")
    prepared = _task9_evaluation_loader(("f0", "f1", "f2", "f3"))
    monkeypatch.setattr(stage_module, "_load_prepared_train_validation", lambda context: prepared)
    test_reads: list[str] = []
    monkeypatch.setattr(
        stage_module,
        "_task9_load_test",
        lambda context, row, *, reason: test_reads.append(reason)
        or (prepared[0].iloc[:4].reset_index(drop=True), prepared[1][:4]),
    )
    evaluated: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        stage_module,
        "evaluate_fixed_subset",
        lambda *args, **kwargs: evaluated.append(tuple(args[4])) or (0.5 + 0.1 * len(args[4])),
    )

    record = stage_module.run_removal_job(random_context)
    document = json.loads((layout.root / record.artifacts["results"]).read_text(encoding="utf-8"))
    assert test_reads == ["final_removal_evaluation"]
    assert len(document["rows"]) == smoke_protocol.removal["random_repeats"]
    assert all(row["k"] == 2 and len(row["removed_features"]) == 2 for row in document["rows"])
    assert not any(token in document for token in ("mean", "median", "best", "recommendation"))
    assert len(evaluated) == len(set(evaluated))
    planned_method = next(
        item for item in json.loads(
            (layout.root / plan_record.artifacts["selection_plan"]).read_text(encoding="utf-8")
        )["methods"]
        if item["method"] == "random_matched"
    )
    assert document["selection_model_fit_count"] == planned_method["model_fit_count"]
    assert document["selection_coalition_count"] == planned_method["coalition_count"]
    assert document["selection_time_s"] == plan_record.metrics["random_matched_selection_time_s"]

    monkeypatch.setattr(
        stage_module,
        "_task9_load_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retry read test")),
    )
    retry = stage_module.run_removal_job(random_context)
    assert retry.metrics == record.metrics

    result_path = layout.root / record.artifacts["results"]
    original = json.loads(result_path.read_text(encoding="utf-8"))

    def mutate_subset(payload):
        payload["rows"][0]["removed_features"] = ["f0"]

    def mutate_top_evaluation_count(payload):
        payload["evaluation_model_fit_count"] += 1

    def mutate_row_evaluation_count(payload):
        payload["rows"][0]["evaluation_model_fit_count"] += 1

    def mutate_row_runtime(payload):
        payload["rows"][0]["evaluation_time_s"] += 1.0

    def mutate_shared_full_utility(payload):
        payload["rows"][-1]["full_utility"] += 0.1
        payload["rows"][-1]["delta_utility"] -= 0.1

    def mutate_coherent_selection_count(payload):
        payload["selection_model_fit_count"] += 1
        for result_row in payload["rows"]:
            result_row["selection_model_fit_count"] += 1

    def mutate_coherent_selection_runtime(payload):
        payload["selection_time_s"] += 1.0
        for result_row in payload["rows"]:
            result_row["selection_time_s"] += 1.0

    for mutation in (
        mutate_subset,
        mutate_top_evaluation_count,
        mutate_row_evaluation_count,
        mutate_row_runtime,
        mutate_shared_full_utility,
        mutate_coherent_selection_count,
        mutate_coherent_selection_runtime,
    ):
        payload = json.loads(json.dumps(original))
        mutation(payload)
        atomic_write_json(result_path, payload)
        with pytest.raises(ValueError, match="subset|reuse|row|removal|counter|time"):
            stage_module.run_removal_job(random_context)
    atomic_write_json(result_path, original)


@pytest.mark.parametrize("method", stage_module._REMOVAL_METHODS)
@pytest.mark.parametrize("harmful", [False, True])
def test_every_removal_method_preserves_exact_job_totals_for_k0_and_k2(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    harmful: bool,
) -> None:
    layout, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "removal")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    X_train, _, _, _, _ = _patch_task9_selection(monkeypatch)
    from minppi_mofs.baselines import banzhaf_fs, min_shap, shap_top_k

    if not harmful:
        monkeypatch.setattr(
            min_ppi_module,
            "estimate_minppi_hybrid",
            lambda *args, **kwargs: {name: 0 for name in X_train},
        )
    baseline_scores = {"f0": 4.0, "f1": 1.0, "f2": 3.0, "f3": 2.0}
    for module, name in (
        (shap_top_k, "compute_shap_topk_importance"),
        (min_shap, "compute_minshap_importance"),
        (banzhaf_fs, "compute_banzhaf_importance"),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args, diagnostics, **kwargs: diagnostics.update(
                model_fit_count=2, coalition_count=3
            )
            or baseline_scores,
        )
    plan_record = stage_module.run_removal_job(plan_context)
    _mark_task9_plan_success(plan_context, plan_record)
    eval_context = next(item for item in contexts if item.job.method == method)
    prepared = _task9_evaluation_loader(("f0", "f1", "f2", "f3"))
    monkeypatch.setattr(stage_module, "_load_prepared_train_validation", lambda context: prepared)
    test_reasons: list[str] = []
    monkeypatch.setattr(
        stage_module,
        "_task9_load_test",
        lambda context, row, *, reason: test_reasons.append(reason)
        or (prepared[0].iloc[:4].reset_index(drop=True), prepared[1][:4]),
    )
    evaluated: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        stage_module,
        "evaluate_fixed_subset",
        lambda *args, **kwargs: evaluated.append(tuple(args[4]))
        or (0.5 + 0.05 * len(args[4])),
    )

    record = stage_module.run_removal_job(eval_context)
    document = json.loads(
        (layout.root / record.artifacts["results"]).read_text(encoding="utf-8")
    )
    plan = json.loads(
        (layout.root / plan_record.artifacts["selection_plan"]).read_text(
            encoding="utf-8"
        )
    )
    method_plan = next(item for item in plan["methods"] if item["method"] == method)
    expected_subsets = {
        tuple(plan["features"]),
        *(tuple(item["retained_features"]) for item in method_plan["trials"]),
    }
    expected_fits = sum(bool(features) for features in expected_subsets)
    assert test_reasons == ["final_removal_evaluation"]
    assert document["k"] == (2 if harmful else 0)
    assert document["evaluation_model_fit_count"] == expected_fits
    assert len(evaluated) == expected_fits
    assert document["selection_model_fit_count"] == method_plan["model_fit_count"]
    assert document["selection_coalition_count"] == method_plan["coalition_count"]
    if method in {"full", "random_matched"}:
        assert document["selection_model_fit_count"] == 0
        assert document["selection_coalition_count"] == 0
    for result_row in document["rows"]:
        for field in (
            "selection_time_s",
            "evaluation_time_s",
            "selection_model_fit_count",
            "selection_coalition_count",
            "evaluation_model_fit_count",
        ):
            assert result_row[field] == document[field]
        assert result_row["full_utility"] == document["rows"][0]["full_utility"]


def test_removal_stage_uses_real_audited_test_accessor_exactly_once(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, _ = _synthetic_stage_protocol(tmp_path, smoke_protocol)
    protocol = replace(protocol, datasets=(protocol.datasets[0],))
    layout = create_new_run(tmp_path / "results", "task9-real-audit", protocol)
    freeze_run_metadata(layout, protocol, project_root=Path(__file__).parents[1])
    manifest = (
        build_job_manifest(protocol, stages=("validate",))
        + build_job_manifest(protocol, stages=("preprocess",))
        + build_job_manifest(
            protocol,
            stages=("removal",),
            data_manifest=(
                {
                    "dataset_id": "spambase",
                    "seed": 42,
                    "selected_feature_count": 2,
                },
            ),
        )
    )
    status = StatusStore(layout, manifest)
    contexts = _contexts(protocol, layout, manifest, status)
    stage_module.run_validate_job(
        next(item for item in contexts if item.job.stage == "validate")
    )
    stage_module.run_preprocess_job(
        next(item for item in contexts if item.job.stage == "preprocess")
    )
    plan_context = next(
        item
        for item in contexts
        if item.job.stage == "removal" and item.job.method == "selection_plan"
    )
    from minppi_mofs.baselines import banzhaf_fs, min_shap, shap_top_k

    monkeypatch.setattr(
        stage_module,
        "_make_job_cache",
        lambda context, X_train, *args, **kwargs: _Task9FakeCache(
            tuple(X_train.columns)
        ),
    )
    monkeypatch.setattr(
        guidance_module,
        "build_mi_correlation_guidance",
        lambda X, *args, **kwargs: (
            {feature: 0.0 for feature in X.columns},
            {feature: [] for feature in X.columns},
        ),
    )
    monkeypatch.setattr(
        min_ppi_module,
        "make_algorithm_b_perturbation_schedule",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        min_ppi_module,
        "estimate_minppi_hybrid",
        lambda X_train, **kwargs: {feature: 0 for feature in X_train.columns},
    )
    for module, name in (
        (shap_top_k, "compute_shap_topk_importance"),
        (min_shap, "compute_minshap_importance"),
        (banzhaf_fs, "compute_banzhaf_importance"),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda X, *args, diagnostics, **kwargs: diagnostics.update(
                model_fit_count=1, coalition_count=1
            )
            or {feature: float(index) for index, feature in enumerate(X.columns)},
        )
    plan_record = stage_module.run_removal_job(plan_context)
    _mark_task9_plan_success(plan_context, plan_record)
    eval_context = next(
        item
        for item in contexts
        if item.job.stage == "removal" and item.job.method == "full"
    )
    monkeypatch.setattr(
        stage_module,
        "evaluate_fixed_subset",
        lambda *args, **kwargs: 0.75,
    )

    stage_module.run_removal_job(eval_context)

    events = eval_context.audit.read_events()
    assert len(events) == 1
    assert events[0]["decision"] == "allowed"
    assert events[0]["stage"] == "removal"
    assert events[0]["job_id"] == eval_context.job.job_id
    assert events[0]["reason"] == "final_removal_evaluation"


def test_alpha_evaluation_reads_test_once_and_reuses_exact_frozen_entry(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "alpha")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    _patch_task9_selection(monkeypatch, harmful_by_alpha=True)
    plan_record = stage_module.run_alpha_job(plan_context)
    _mark_task9_plan_success(plan_context, plan_record)
    eval_context = next(item for item in contexts if item.job.method == "alpha_evaluation")
    prepared = _task9_evaluation_loader(("f0", "f1", "f2", "f3"))
    monkeypatch.setattr(stage_module, "_load_prepared_train_validation", lambda context: prepared)
    test_reads: list[str] = []
    monkeypatch.setattr(
        stage_module,
        "_task9_load_test",
        lambda context, row, *, reason: test_reads.append(reason)
        or (prepared[0].iloc[:4].reset_index(drop=True), prepared[1][:4]),
    )
    evaluated: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        stage_module,
        "evaluate_fixed_subset",
        lambda *args, **kwargs: evaluated.append(tuple(args[4])) or (0.9 if len(args[4]) == 4 else 0.72),
    )

    record = stage_module.run_alpha_job(eval_context)
    document = json.loads((layout.root / record.artifacts["results"]).read_text(encoding="utf-8"))
    plan = json.loads((layout.root / plan_record.artifacts["selection_plan"]).read_text(encoding="utf-8"))
    frozen = next(item for item in plan["entries"] if item["alpha"] == eval_context.job.parameters["alpha"])
    assert test_reads == ["final_alpha_grid_evaluation"]
    assert document["retained_features"] == frozen["retained_features"]
    assert document["removed_features"] == frozen["removed_features"]
    assert document["retained_hash"] == frozen["retained_hash"]
    assert document["preservation"] == document["test_utility"] / document["full_test_utility"]
    assert len(evaluated) == len(set(evaluated))
    assert not any("argmax" in key or "recommend" in key for key in document)
    assert document["selection_time_s"] == plan_record.metrics[
        f"{frozen['alpha_token']}_selection_time_s"
    ]
    assert document["selection_model_fit_count"] == frozen["model_fit_count"]
    assert document["selection_coalition_count"] == frozen["coalition_count"]

    monkeypatch.setattr(
        stage_module,
        "_task9_load_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retry read test")),
    )
    retry = stage_module.run_alpha_job(eval_context)
    assert retry.metrics == record.metrics

    result_path = layout.root / record.artifacts["results"]
    original = json.loads(result_path.read_text(encoding="utf-8"))

    def mutate_selection_time(payload):
        payload["selection_time_s"] += 1.0

    def mutate_selection_fits(payload):
        payload["selection_model_fit_count"] += 1

    def mutate_selection_coalitions(payload):
        payload["selection_coalition_count"] += 1

    def mutate_evaluation_fits(payload):
        payload["evaluation_model_fit_count"] += 1

    def mutate_negative_evaluation_time(payload):
        payload["evaluation_time_s"] = -1.0

    for mutation in (
        mutate_selection_time,
        mutate_selection_fits,
        mutate_selection_coalitions,
        mutate_evaluation_fits,
        mutate_negative_evaluation_time,
    ):
        payload = dict(original)
        mutation(payload)
        atomic_write_json(result_path, payload)
        with pytest.raises(ValueError, match="alpha|counter|selection|time"):
            stage_module.run_alpha_job(eval_context)
    atomic_write_json(result_path, original)


def test_alpha_evaluation_rejects_unsuccessful_plan_before_test_access_or_fit(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "alpha")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    _patch_task9_selection(monkeypatch, harmful_by_alpha=True)
    stage_module.run_alpha_job(plan_context)
    eval_context = next(item for item in contexts if item.job.method == "alpha_evaluation")
    monkeypatch.setattr(
        stage_module,
        "_task9_load_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("test accessed before plan success")),
    )
    monkeypatch.setattr(
        stage_module,
        "evaluate_fixed_subset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model fit before plan success")),
    )
    with pytest.raises(ValueError, match="dependency|successful"):
        stage_module.run_alpha_job(eval_context)


def test_removal_plan_rejects_coherent_feature_universe_conflict_before_refit(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "removal")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    _patch_task9_selection(monkeypatch)
    from minppi_mofs.baselines import banzhaf_fs, min_shap, shap_top_k

    baseline = {"f0": 0.0, "f1": 1.0, "f2": 2.0, "f3": 3.0}
    for module, name in (
        (shap_top_k, "compute_shap_topk_importance"),
        (min_shap, "compute_minshap_importance"),
        (banzhaf_fs, "compute_banzhaf_importance"),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args, diagnostics, **kwargs: diagnostics.update(
                model_fit_count=1, coalition_count=1
            )
            or baseline,
        )
    record = stage_module.run_removal_job(plan_context)
    path = layout.root / record.artifacts["selection_plan"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["features"][0] = "g0"
    for score_group in document["scores"]:
        score_group["values"][0]["feature"] = "g0"
    for method in document["methods"]:
        for row in method["trials"]:
            row["retained_features"] = [
                "g0" if feature == "f0" else feature
                for feature in row["retained_features"]
            ]
            row["removed_features"] = [
                "g0" if feature == "f0" else feature
                for feature in row["removed_features"]
            ]
            row["retained_hash"] = stage_module._task9_feature_hash(
                row["retained_features"]
            )
    atomic_write_json(path, document)
    monkeypatch.setattr(
        stage_module,
        "_load_prepared_train_validation",
        lambda context: (_ for _ in ()).throw(AssertionError("conflict refit")),
    )
    with pytest.raises(ValueError, match="manifest|feature|authority"):
        stage_module.run_removal_job(plan_context)


def test_alpha_plan_rejects_coherent_harmful_set_conflict_with_frozen_scores(
    tmp_path: Path, smoke_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "alpha")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    _patch_task9_selection(monkeypatch, harmful_by_alpha=True)
    record = stage_module.run_alpha_job(plan_context)
    path = layout.root / record.artifacts["selection_plan"]
    document = json.loads(path.read_text(encoding="utf-8"))
    entry = document["entries"][0]
    replacement = next(
        feature
        for feature in document["features"]
        if feature not in entry["removed_features"]
    )
    entry["removed_features"] = [replacement]
    entry["retained_features"] = [
        feature for feature in document["features"] if feature != replacement
    ]
    entry["retained_hash"] = stage_module._task9_feature_hash(
        entry["retained_features"]
    )
    entry["harmful_count"] = 1
    entry["compression"] = 1 / len(document["features"])
    atomic_write_json(path, document)
    monkeypatch.setattr(
        stage_module,
        "_load_prepared_train_validation",
        lambda context: (_ for _ in ()).throw(AssertionError("conflict refit")),
    )

    with pytest.raises(ValueError, match="harmful|score|plan"):
        stage_module.run_alpha_job(plan_context)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_path", "missing_metric", "string_runtime", "negative_count", "counter_drift"],
)
def test_alpha_plan_result_record_fails_closed_before_test_read_or_fit(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "alpha")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    _patch_task9_selection(monkeypatch, harmful_by_alpha=True)
    record = stage_module.run_alpha_job(plan_context)
    _mark_task9_plan_success(plan_context, record)
    path = plan_context.layout.root / plan_context.job.output_relpath
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "wrong_path":
        payload["artifacts"] = {"selection_plan": "alpha/foreign/selection_plan.json"}
    elif mutation == "missing_metric":
        payload["metrics"].pop(next(iter(payload["metrics"])))
    elif mutation == "string_runtime":
        payload["metrics"]["selection_time_s"] = "fast"
    elif mutation == "negative_count":
        payload["metrics"]["selection_model_fit_count"] = -1
    else:
        payload["metrics"]["selection_model_fit_count"] += 1
    atomic_write_json(path, payload)
    eval_context = next(item for item in contexts if item.job.method == "alpha_evaluation")
    monkeypatch.setattr(
        stage_module,
        "_task9_load_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bad plan read test")),
    )
    monkeypatch.setattr(
        stage_module,
        "evaluate_fixed_subset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bad plan fit")),
    )
    with pytest.raises(ValueError, match="plan|selection|metric|artifact|counter"):
        stage_module.run_alpha_job(eval_context)


@pytest.mark.parametrize("tampered_counter", ["model_fit_count", "coalition_count"])
def test_alpha_plan_rejects_coordinated_entry_aggregate_and_record_counter_tampering(
    tmp_path: Path,
    smoke_protocol,
    monkeypatch: pytest.MonkeyPatch,
    tampered_counter: str,
) -> None:
    layout, _, contexts = _task9_contexts(tmp_path, smoke_protocol, "alpha")
    plan_context = next(item for item in contexts if item.job.method == "selection_plan")
    _patch_task9_selection(monkeypatch, harmful_by_alpha=True)
    record = stage_module.run_alpha_job(plan_context)
    _mark_task9_plan_success(plan_context, record)
    plan_path = layout.root / record.artifacts["selection_plan"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in plan["entries"]
        if item["model_fit_count"] == item["coalition_count"] == 1
    )
    delta = 1 if tampered_counter == "model_fit_count" else -1
    entry[tampered_counter] += delta
    aggregate = f"selection_{tampered_counter}"
    plan[aggregate] += delta
    atomic_write_json(plan_path, plan)
    record_path = layout.root / plan_context.job.output_relpath
    result_record = json.loads(record_path.read_text(encoding="utf-8"))
    result_record["metrics"][aggregate] += delta
    atomic_write_json(record_path, result_record)
    eval_context = next(item for item in contexts if item.job.method == "alpha_evaluation")
    monkeypatch.setattr(
        stage_module,
        "_task9_load_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("coordinated counter tamper read test")
        ),
    )
    monkeypatch.setattr(
        stage_module,
        "evaluate_fixed_subset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("coordinated counter tamper fit")
        ),
    )

    with pytest.raises(ValueError, match="alpha|counter|cost|plan"):
        stage_module.run_alpha_job(eval_context)


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
    runner.executors = {stage: lambda context: _result(context.job) for stage in STAGE_ORDER}
    seen: list[str] = []
    monkeypatch.setattr(runner, "execute", lambda stage: seen.append(stage))

    assert runner.run("all") == 0
    assert tuple(seen) == STAGE_ORDER


def test_pipeline_all_advances_through_alpha_then_stops_at_unregistered_nsga2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(PipelineRunner)
    runner.executors = dict(stage_module.STAGE_EXECUTORS)
    seen: list[str] = []

    def execute(stage: str):
        seen.append(stage)
        if stage not in runner.executors:
            return stage_module.StageRunSummary(
                stage=stage, dependency_issues=("executor is not registered",)
            )
        return stage_module.StageRunSummary(stage=stage)

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.run("all") == 1
    assert tuple(seen) == ("validate", "preprocess", "estimator", "removal", "alpha")


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


def test_all_runs_through_alpha_then_stops_before_unregistered_future_executors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(PipelineRunner)
    runner.executors = dict(stage_module.STAGE_EXECUTORS)
    seen: list[str] = []
    monkeypatch.setattr(
        runner,
        "execute",
        lambda stage: seen.append(stage) or stage_module.StageRunSummary(stage=stage),
    )

    assert runner.run("all") == 1
    assert tuple(seen) == ("validate", "preprocess", "estimator", "removal", "alpha")
    assert "nsga2" not in seen
