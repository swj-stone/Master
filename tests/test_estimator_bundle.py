from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import experiments.icomp2026_stages as stage_module
import experiments.icomp2026_estimator_bundle as bundle_module
import minppi_mofs.min_ppi as min_ppi_module
import minppi_mofs.shapg_integration as guidance_module
import minppi_mofs.voting_game as voting_game_module
from experiments.icomp2026_estimator_bundle import (
    ESTIMATOR_BUNDLE_SCHEMA_VERSION,
    EstimatorBundle,
    EstimatorCheckpoint,
    EstimatorGroupRequest,
    build_estimator_bundle,
    bundle_from_document,
    bundle_relpath,
    bundle_to_document,
    ensure_estimator_bundle,
    group_request_from_context,
    progress_relpath,
    validate_estimator_bundle,
)
from experiments.icomp2026_stages import StageContext
from experiments.icomp2026_progress import (
    EstimatorProgressEvent,
    estimate_eta_s,
    format_progress,
    progress_from_document,
    progress_to_document,
    remaining_physical_fit_upper_bound,
)
from minppi_mofs.coalition_cache import (
    CACHE_SCHEMA_VERSION,
    CoalitionCacheIdentity,
    CoalitionCacheStats,
    CoalitionUtilityCache,
    CoalitionUtilityView,
    SQLiteCoalitionStore,
    hash_numeric_array,
    hash_numeric_frame,
)
from minppi_mofs.experiment_protocol import (
    DatasetSpec,
    ExecutionSpec,
    ResolvedProtocol,
    RunLayout,
    atomic_write_json,
    build_job_manifest,
    read_canonical_json,
    read_result_record,
    resolved_protocol_hash,
)


def _layout(root: Path) -> RunLayout:
    return RunLayout(
        root=root,
        configs_frozen=root / "configs_frozen",
        raw_metrics=root / "raw_metrics",
        tables=root / "tables",
        figures=root / "figures",
        cache_root=root / "cache",
        bundle_root=root / "estimator" / "_bundles",
        audit_path=root / "protocol_audit.json",
        manifest_path=root / "manifest.csv",
        status_path=root / "status.json",
        log_path=root / "run.log",
    )


@pytest.fixture
def stage_context(tmp_path: Path) -> StageContext:
    dataset = DatasetSpec(
        dataset_id="spambase",
        display_name="Spambase",
        raw_path=Path("data/raw/spambase.csv"),
        target="spam",
        task="binary",
        utility="roc_auc",
        max_features=None,
        variance_threshold=0.0,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
    )
    protocol = ResolvedProtocol(
        protocol_version="1.0",
        run_kind="formal",
        datasets=(dataset,),
        seeds=(42,),
        backend="cuda",
        execution=ExecutionSpec(
            mode="grouped_nested_v1",
            cache_schema="coalition_utility_v1",
            checkpoint_every_fits=50,
            progress_every_seconds=30,
        ),
        model_params={
            "learning_rate": 0.1,
            "max_depth": 4,
            "n_estimators": 100,
            "n_jobs": 1,
            "policy": {"subsample": [1.0, 0.75]},
        },
        alpha_default=0.95,
        alpha_grid=(0.85, 0.9, 0.95, 0.99),
        estimator_methods=(
            "exact",
            "random_mc",
            "mi_correlation_guided",
            "hybrid",
        ),
        estimator_budgets=(10, 25, 50, 100),
        exact_max_features=16,
        nsga2={},
        removal={},
        ablations={},
        smoke={},
    )
    manifest = build_job_manifest(
        protocol,
        stages=("estimator",),
        data_manifest=[
            {
                "dataset_id": "spambase",
                "seed": 42,
                "selected_feature_count": 2,
            }
        ],
    )
    job = next(
        item
        for item in manifest
        if item.method == "random_mc" and item.parameters["budget"] == 10
    )
    return StageContext(
        protocol=protocol,
        layout=_layout(tmp_path / "run"),
        job=job,
        status=SimpleNamespace(manifest=manifest),
        audit=SimpleNamespace(),
    )


@pytest.fixture
def prepared_columns() -> tuple[str, ...]:
    return ("word_freq_make", "word_freq_address")


@pytest.fixture
def data_row(
    stage_context: StageContext, prepared_columns: tuple[str, ...]
) -> dict[str, object]:
    preprocess_job = build_job_manifest(
        stage_context.protocol, stages=("preprocess",)
    )[0]
    train_labels = {
        "classes": [
            {"count": 3, "label": 0},
            {"count": 3, "label": 1},
        ],
        "count": 6,
    }
    held_out_labels = {
        "classes": [
            {"count": 1, "label": 0},
            {"count": 1, "label": 1},
        ],
        "count": 2,
    }
    return {
        "protocol_hash": stage_context.job.protocol_hash,
        "job_id": preprocess_job.job_id,
        "dataset_id": "spambase",
        "seed": 42,
        "task": "binary",
        "utility": "roc_auc",
        "target": "spam",
        "raw_file_sha256": "0" * 64,
        "raw_row_count": 10,
        "raw_feature_count": 3,
        "selected_feature_count": 2,
        "selected_features_json": json.dumps(
            list(prepared_columns),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "train_relpath": "preprocess/spambase/s42/train.csv",
        "val_relpath": "preprocess/spambase/s42/val.csv",
        "test_relpath": "preprocess/spambase/s42/test.csv",
        "train_row_count": 6,
        "val_row_count": 2,
        "test_row_count": 2,
        "train_index_sha256": "1" * 64,
        "val_index_sha256": "2" * 64,
        "test_index_sha256": "3" * 64,
        "train_file_sha256": "a" * 64,
        "val_file_sha256": "b" * 64,
        "test_file_sha256": "4" * 64,
        "train_label_summary_json": json.dumps(
            train_labels, sort_keys=True, separators=(",", ":")
        ),
        "val_label_summary_json": json.dumps(
            held_out_labels, sort_keys=True, separators=(",", ":")
        ),
        "test_label_summary_json": json.dumps(
            held_out_labels, sort_keys=True, separators=(",", ":")
        ),
        "transform_metadata_json": "{}",
    }


@pytest.fixture
def group_request(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
) -> EstimatorGroupRequest:
    return group_request_from_context(stage_context, data_row, prepared_columns)


def _checkpoint(
    request: EstimatorGroupRequest, method: str, budget: int
) -> EstimatorCheckpoint:
    is_exact = method == "exact"
    logical_evaluations = 4 if is_exact else budget * 2
    logical_unique = 3 if is_exact else budget + 1
    physical_fits = (
        2
        if is_exact
        else (0 if method == "hybrid" else max(1, budget // 5))
    )
    cache_hits = 1 if is_exact else (0 if method == "hybrid" else budget)
    schedules = bundle_module._schedule_authority(request)
    schedule_digest = {
        "exact": None,
        "random_mc": schedules.random_digest,
        "mi_correlation_guided": schedules.guided_digest,
        "hybrid": bundle_module._hybrid_schedule_digest(
            feature_count=len(request.ordered_features),
            random_digest=schedules.random_digest,
            guided_digest=schedules.guided_digest,
        ),
    }[method]
    return EstimatorCheckpoint(
        method=method,
        budget=budget,
        scores=tuple(
            (feature, 1 if index == 0 else -1)
            for index, feature in enumerate(request.ordered_features)
        ),
        logical_evaluation_count=logical_evaluations,
        logical_unique_coalition_count=logical_unique,
        physical_model_fit_count=physical_fits,
        cache_hit_count=cache_hits,
        schedule_digest=schedule_digest,
        component_runtime_s=budget / 100.0,
    )


@pytest.fixture
def bundle(group_request: EstimatorGroupRequest) -> EstimatorBundle:
    checkpoints = tuple(
        _checkpoint(group_request, method, budget)
        for method in group_request.methods
        for budget in group_request.budgets
    )
    return EstimatorBundle(
        schema_version=ESTIMATOR_BUNDLE_SCHEMA_VERSION,
        request_digest=group_request.digest,
        checkpoints=checkpoints,
        physical_model_fit_count=42,
        cache_hit_count=201,
        group_runtime_s=4.0,
    )


def test_group_request_identity_includes_every_frozen_input(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
) -> None:
    request = group_request_from_context(
        stage_context, data_row, tuple(prepared_columns)
    )

    assert request.dataset_id == stage_context.job.dataset_id
    assert request.seed == stage_context.job.seed
    assert request.budgets == (10, 25, 50, 100)
    assert request.backend == "cuda"
    assert request.split_hashes == {
        "train": data_row["train_file_sha256"],
        "validation": data_row["val_file_sha256"],
    }
    assert bundle_relpath(request) == (
        "estimator/_bundles/spambase/s42/bundle.json"
    )
    assert progress_relpath(request) == (
        "estimator/_bundles/spambase/s42/progress.json"
    )


def test_all_grouped_estimator_jobs_project_two_completed_bundles(
    tmp_path: Path,
    stage_context: StageContext,
    data_row: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases: list[
        tuple[
            tuple[StageContext, ...],
            tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, dict[str, object]],
            EstimatorGroupRequest,
            EstimatorBundle,
        ]
    ] = []
    for case_name, feature_count in (("exact", 2), ("gated", 17)):
        features = tuple(f"feature_{index}" for index in range(feature_count))
        row = {
            **data_row,
            "raw_feature_count": feature_count + 1,
            "selected_feature_count": feature_count,
            "selected_features_json": json.dumps(
                list(features), sort_keys=True, separators=(",", ":")
            ),
        }
        manifest = build_job_manifest(
            stage_context.protocol,
            stages=("estimator",),
            data_manifest=(
                {
                    "dataset_id": stage_context.job.dataset_id,
                    "seed": stage_context.job.seed,
                    "selected_feature_count": feature_count,
                },
            ),
        )
        status = SimpleNamespace(
            manifest=manifest,
            entry=lambda _job_id: {"state": "pending"},
        )
        layout = _layout(tmp_path / case_name)
        contexts = tuple(
            StageContext(
                protocol=stage_context.protocol,
                layout=layout,
                job=job,
                status=status,
                audit=SimpleNamespace(),
            )
            for job in manifest
        )
        X_train = pd.DataFrame(
            np.zeros((6, feature_count), dtype=float),
            columns=features,
        )
        X_val = pd.DataFrame(
            np.ones((2, feature_count), dtype=float),
            columns=features,
        )
        frames = (
            X_train,
            np.array([0, 1, 0, 1, 0, 1]),
            X_val,
            np.array([0, 1]),
            row,
        )
        request = group_request_from_context(contexts[0], row, features)
        checkpoints = tuple(
            _checkpoint(request, method, budget)
            for method in request.methods
            for budget in request.budgets
        )
        final_checkpoints = tuple(
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.budget == request.budgets[-1]
        )
        bundle = EstimatorBundle(
            schema_version=ESTIMATOR_BUNDLE_SCHEMA_VERSION,
            request_digest=request.digest,
            checkpoints=checkpoints,
            physical_model_fit_count=sum(
                checkpoint.physical_model_fit_count
                for checkpoint in final_checkpoints
            ),
            cache_hit_count=sum(
                checkpoint.cache_hit_count
                for checkpoint in final_checkpoints
            ),
            group_runtime_s=1.0
            + sum(
                checkpoint.component_runtime_s
                for checkpoint in final_checkpoints
            ),
        )
        cases.append((contexts, frames, request, bundle))

    frames_by_root = {
        contexts[0].layout.root: frames
        for contexts, frames, _request, _bundle in cases
    }
    bundles_by_digest = {
        request.digest: bundle
        for _contexts, _frames, request, bundle in cases
    }
    monkeypatch.setattr(
        stage_module,
        "_load_prepared_train_validation",
        lambda context: frames_by_root[context.layout.root],
    )
    monkeypatch.setattr(
        stage_module,
        "ensure_estimator_bundle",
        lambda request, *_args, **_kwargs: bundles_by_digest[request.digest],
    )
    monkeypatch.setattr(
        stage_module,
        "_run_legacy_estimator_job",
        lambda _context: pytest.fail("legacy estimator used"),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("grouped projection invoked direct estimator/model work")

    monkeypatch.setattr(stage_module, "fit_predict", forbidden)
    monkeypatch.setattr(stage_module, "evaluate_fixed_subset", forbidden)
    monkeypatch.setattr(voting_game_module, "AccFullCache", forbidden)
    monkeypatch.setattr(
        guidance_module, "build_mi_correlation_guidance", forbidden
    )
    for name in (
        "estimate_minppi_exact",
        "estimate_minppi_random_mc",
        "estimate_minppi_mi_correlation_guided",
        "estimate_minppi_hybrid",
    ):
        monkeypatch.setattr(min_ppi_module, name, forbidden)

    projected_paths: set[str] = set()
    projected_count = 0
    for contexts, frames, request, bundle in cases:
        expected_pairs = {
            (method, budget)
            for method in request.methods
            for budget in request.budgets
        }
        assert {
            (context.job.method, int(context.job.parameters["budget"]))
            for context in contexts
        } == expected_pairs
        if len(request.ordered_features) > request.exact_max_features:
            assert "exact" not in request.methods
            assert all(context.job.method != "exact" for context in contexts)

        for context in contexts:
            record = stage_module.run_estimator_job(context)
            budget = int(context.job.parameters["budget"])
            expected_relpath = (
                f"estimator/{context.job.dataset_id}/s{context.job.seed}/"
                f"{context.job.method}/b{budget}/scores.json"
            )
            assert record.artifacts == {"scores": expected_relpath}
            projected_paths.add(
                f"{context.layout.root.name}/{expected_relpath}"
            )
            projected_count += 1
            checkpoint = next(
                item
                for item in bundle.checkpoints
                if item.method == context.job.method
                and item.budget == budget
            )
            assert record.metrics["runtime_s"] == checkpoint.component_runtime_s
            assert (
                record.metrics["group_runtime_s"] == bundle.group_runtime_s
            )
            if context.job.method == "hybrid":
                assert record.metrics["physical_model_fit_count"] == 0
            stage_module._read_score_artifact(
                context.layout.root / expected_relpath,
                expected_job=context.job,
                budget=budget,
                expected_features=tuple(frames[0].columns),
                expected_metrics=record.metrics,
            )
            atomic_write_json(
                context.layout.root / context.job.output_relpath,
                asdict(record),
            )
            assert (
                read_result_record(
                    context.layout.root / context.job.output_relpath,
                    context.job,
                )
                == record
            )

    assert projected_count == 28
    assert len(projected_paths) == 28
    assert len(bundles_by_digest) == 2


def test_group_request_deeply_owns_identity_and_is_mapping_order_independent(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
) -> None:
    nested = [1.0, {"depth": 4}]
    params = {"z": nested, "a": True}
    request = EstimatorGroupRequest(
        protocol_hash=stage_context.job.protocol_hash,
        dataset_id="spambase",
        seed=42,
        task="binary",
        utility="roc_auc",
        ordered_features=prepared_columns,
        split_hashes={"validation": "b" * 64, "train": "a" * 64},
        model_params=params,
        backend="cuda",
        methods=("random_mc",),
        budgets=(10,),
        exact_max_features=16,
        alpha=0.95,
        cache_schema="coalition_utility_v1",
        checkpoint_every_fits=50,
        progress_every_seconds=30,
    )
    equivalent = replace(
        request,
        split_hashes={"train": "a" * 64, "validation": "b" * 64},
        model_params={"a": True, "z": [1.0, {"depth": 4}]},
    )
    digest = request.digest

    nested[1]["depth"] = 7
    params["new"] = "caller mutation"

    assert request.digest == digest == equivalent.digest
    with pytest.raises(TypeError):
        request.model_params["new"] = "mutation"  # type: ignore[index]
    with pytest.raises(TypeError):
        request.model_params["z"][1]["depth"] = 9  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.seed = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_hash", "f" * 64),
        ("dataset_id", "madelon"),
        ("seed", 43),
        ("utility", "accuracy"),
        ("ordered_features", ("word_freq_make", "capital_run_length")),
        ("split_hashes", {"train": "a" * 64, "validation": "f" * 64}),
        ("model_params", {"n_estimators": 99}),
        ("backend", "cpu"),
        ("methods", ("random_mc", "hybrid")),
        ("budgets", (10, 25, 51, 100)),
        ("exact_max_features", 15),
        ("alpha", 0.9),
        ("checkpoint_every_fits", 51),
        ("progress_every_seconds", 31),
    ],
)
def test_every_group_request_field_changes_digest(
    group_request: EstimatorGroupRequest, field: str, value: object
) -> None:
    # Removing the field from the canonical digest would make this fail.
    assert replace(group_request, **{field: value}).digest != group_request.digest


def test_task_and_fixed_cache_schema_participate_in_request_digest(
    group_request: EstimatorGroupRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accuracy_request = replace(group_request, utility="accuracy")
    assert (
        replace(accuracy_request, task="multiclass").digest
        != accuracy_request.digest
    )

    import experiments.icomp2026_estimator_bundle as bundle_module

    monkeypatch.setattr(
        bundle_module, "CACHE_SCHEMA_VERSION", "coalition_utility_v2"
    )
    assert (
        replace(group_request, cache_schema="coalition_utility_v2").digest
        != group_request.digest
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("methods", ("random_mc", "random_mc"), "methods"),
        ("methods", ("not-a-method",), "methods"),
        ("budgets", (10, 10), "budgets"),
        ("budgets", (25, 10), "budgets"),
        ("task", "ranking", "task"),
        ("utility", "rmse", "utility"),
        ("backend", "auto", "backend"),
        ("seed", True, "seed"),
        ("exact_max_features", 0, "exact_max_features"),
        ("alpha", float("nan"), "alpha"),
        ("cache_schema", "coalition_utility_v2", "cache_schema"),
        ("checkpoint_every_fits", 0, "checkpoint_every_fits"),
        ("progress_every_seconds", True, "progress_every_seconds"),
        ("ordered_features", ("f0", "f0"), "features"),
        ("split_hashes", {"train": "a" * 64}, "split_hashes"),
        (
            "model_params",
            {"learning_rate": {"bad": float("inf")}},
            "model",
        ),
    ],
)
def test_group_request_rejects_invalid_authority(
    group_request: EstimatorGroupRequest,
    field: str,
    value: object,
    message: str,
) -> None:
    # Weak constructor validation would accept this malformed frozen identity.
    with pytest.raises((TypeError, ValueError), match=message):
        replace(group_request, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_id", "madelon", "dataset"),
        ("seed", 43, "seed"),
        ("task", "multiclass", "task"),
        ("utility", "accuracy", "utility"),
        ("protocol_hash", "f" * 64, "protocol"),
    ],
)
def test_group_request_rejects_manifest_row_conflicts(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
    field: str,
    value: object,
    message: str,
) -> None:
    conflict = dict(data_row)
    conflict[field] = value

    with pytest.raises(ValueError, match=message):
        group_request_from_context(stage_context, conflict, prepared_columns)


def test_group_request_rejects_missing_manifest_hash(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
) -> None:
    missing = dict(data_row)
    del missing["val_file_sha256"]

    with pytest.raises(ValueError, match="hash"):
        group_request_from_context(stage_context, missing, prepared_columns)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "extra", "conflicting"),
)
def test_group_request_rejects_incomplete_or_conflicting_group_manifest(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
    mutation: str,
) -> None:
    manifest = list(stage_context.status.manifest)
    if mutation == "missing":
        manifest.pop()
    elif mutation == "duplicate":
        manifest.append(manifest[0])
    elif mutation == "extra":
        manifest.append(
            replace(
                manifest[0],
                job_id="f" * 64,
                parameters={
                    **manifest[0].parameters,
                    "budget": 999,
                },
            )
        )
    else:
        manifest[0] = replace(
            manifest[0],
            parameters={
                **manifest[0].parameters,
                "utility": "accuracy",
            },
        )
    context = replace(
        stage_context,
        status=SimpleNamespace(manifest=tuple(manifest)),
    )

    # Trusting only the current job would silently broaden this bad group.
    with pytest.raises(ValueError, match="manifest.*authority|authority.*manifest"):
        group_request_from_context(context, data_row, prepared_columns)


@pytest.mark.parametrize("mutation", ("extra", "missing", "noncanonical"))
def test_group_request_requires_authoritative_full_rich_manifest_row(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
    mutation: str,
) -> None:
    malformed = dict(data_row)
    if mutation == "extra":
        malformed["unexpected"] = True
    elif mutation == "missing":
        del malformed["test_file_sha256"]
    else:
        malformed["selected_features_json"] = json.dumps(
            list(prepared_columns)
        )

    with pytest.raises(ValueError, match="data_manifest|manifest"):
        group_request_from_context(
            stage_context, malformed, prepared_columns
        )


def test_group_request_rejects_job_dataset_seed_and_execution_conflicts(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
) -> None:
    bad_job = replace(stage_context.job, dataset_id="madelon", seed=43)
    bad_context = replace(stage_context, job=bad_job)
    with pytest.raises(ValueError, match="dataset|seed"):
        group_request_from_context(bad_context, data_row, prepared_columns)

    legacy_protocol = replace(
        stage_context.protocol,
        execution=replace(
            stage_context.protocol.execution,
            mode="legacy",
        ),
    )
    legacy_hash = resolved_protocol_hash(legacy_protocol)
    legacy_job = replace(stage_context.job, protocol_hash=legacy_hash)
    legacy_context = replace(
        stage_context,
        protocol=legacy_protocol,
        job=legacy_job,
        status=SimpleNamespace(manifest=(legacy_job,)),
    )
    legacy_row = {**data_row, "protocol_hash": legacy_hash}
    with pytest.raises(ValueError, match="execution"):
        group_request_from_context(
            legacy_context, legacy_row, prepared_columns
        )


def test_group_request_methods_must_follow_frozen_canonical_order(
    group_request: EstimatorGroupRequest,
) -> None:
    with pytest.raises(ValueError, match="method.*order"):
        replace(
            group_request,
            methods=(
                "random_mc",
                "exact",
                "mi_correlation_guided",
                "hybrid",
            ),
        )

    with pytest.raises(ValueError, match="exact.*authorized|method"):
        replace(
            group_request,
            ordered_features=("f0", "f1", "f2"),
            exact_max_features=2,
        )


def test_group_request_omits_and_rejects_ineligible_exact_manifest_authority(
    stage_context: StageContext,
    data_row: dict[str, object],
    prepared_columns: tuple[str, ...],
) -> None:
    protocol = replace(stage_context.protocol, exact_max_features=1)
    manifest = build_job_manifest(
        protocol,
        stages=("estimator",),
        data_manifest=[
            {
                "dataset_id": "spambase",
                "seed": 42,
                "selected_feature_count": 2,
            }
        ],
    )
    job = next(
        item
        for item in manifest
        if item.method == "random_mc" and item.parameters["budget"] == 10
    )
    context = replace(
        stage_context,
        protocol=protocol,
        job=job,
        status=SimpleNamespace(manifest=manifest),
    )
    row = {
        **data_row,
        "protocol_hash": resolved_protocol_hash(protocol),
        "job_id": build_job_manifest(
            protocol, stages=("preprocess",)
        )[0].job_id,
    }

    request = group_request_from_context(context, row, prepared_columns)

    assert request.methods == (
        "random_mc",
        "mi_correlation_guided",
        "hybrid",
    )

    extra_exact = replace(
        manifest[0],
        job_id="f" * 64,
        method="exact",
    )
    conflicting = replace(
        context,
        status=SimpleNamespace(manifest=(extra_exact, *manifest)),
    )
    with pytest.raises(ValueError, match="manifest.*authority"):
        group_request_from_context(
            conflicting, row, prepared_columns
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ordered_features", ("valid", "\ud800")),
        ("model_params", {"nested": {"value": "\ud800"}}),
        ("model_params", {"nested": {"\ud800": "value"}}),
    ],
)
def test_group_request_rejects_non_utf8_digest_strings(
    group_request: EstimatorGroupRequest,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        replace(group_request, **{field: value})


@pytest.mark.parametrize(
    "dataset_id",
    ("../escape", r"nested\escape", "/absolute", ".", "..", "C:drive"),
)
def test_bundle_paths_reject_unsafe_dataset_identifiers(
    group_request: EstimatorGroupRequest, dataset_id: str
) -> None:
    with pytest.raises(ValueError, match="dataset"):
        replace(group_request, dataset_id=dataset_id)


@pytest.mark.parametrize(
    "dataset_id",
    (
        "CON",
        "con.txt",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "com9.data",
        "LPT1",
        "lpt9.txt",
        "trailing.",
        "trailing ",
    ),
)
def test_bundle_paths_reject_windows_unsafe_dataset_identifiers(
    group_request: EstimatorGroupRequest, dataset_id: str
) -> None:
    with pytest.raises(ValueError, match="dataset"):
        replace(group_request, dataset_id=dataset_id)


def test_bundle_paths_accept_protocol_valid_dotted_dataset_identifier(
    group_request: EstimatorGroupRequest,
) -> None:
    dotted = replace(group_request, dataset_id="spam.base-v1")

    assert bundle_relpath(dotted) == (
        "estimator/_bundles/spam.base-v1/s42/bundle.json"
    )


def test_bundle_canonical_round_trip_uses_protocol_helpers(
    tmp_path: Path,
    group_request: EstimatorGroupRequest,
    bundle: EstimatorBundle,
) -> None:
    path = tmp_path / "bundle.json"

    atomic_write_json(path, bundle_to_document(bundle, group_request))
    document = read_canonical_json(path, context="estimator bundle")
    restored = bundle_from_document(document, group_request)

    assert restored == bundle
    assert bundle_to_document(restored, group_request) == document


def test_bundle_rejects_unknown_and_missing_fields(
    group_request: EstimatorGroupRequest, bundle: EstimatorBundle
) -> None:
    document = bundle_to_document(bundle, group_request)
    missing = dict(document)
    del missing["group_runtime_s"]
    unknown = {**document, "surprise": True}

    with pytest.raises(ValueError, match="schema"):
        bundle_from_document(missing, group_request)
    with pytest.raises(ValueError, match="schema"):
        bundle_from_document(unknown, group_request)


def test_bundle_rejects_request_digest_conflict(
    group_request: EstimatorGroupRequest, bundle: EstimatorBundle
) -> None:
    document = bundle_to_document(bundle, group_request)
    document["request_digest"] = "f" * 64

    with pytest.raises(ValueError, match="request digest"):
        bundle_from_document(document, group_request)


@pytest.mark.parametrize("mutation", ("duplicate", "missing", "unauthorized", "order"))
def test_bundle_rejects_noncanonical_or_unauthorized_checkpoint_pairs(
    group_request: EstimatorGroupRequest,
    bundle: EstimatorBundle,
    mutation: str,
) -> None:
    document = bundle_to_document(bundle, group_request)
    checkpoints = list(document["checkpoints"])
    if mutation == "duplicate":
        checkpoints[1] = dict(checkpoints[0])
    elif mutation == "missing":
        checkpoints.pop()
    elif mutation == "unauthorized":
        checkpoints[0] = {**checkpoints[0], "budget": 999}
    else:
        checkpoints[0], checkpoints[1] = checkpoints[1], checkpoints[0]
    document["checkpoints"] = checkpoints

    with pytest.raises(ValueError, match="method/budget|checkpoint order"):
        bundle_from_document(document, group_request)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("feature", "score feature"),
        ("feature_order", "score feature order"),
        ("score", "score value"),
    ],
)
def test_bundle_rejects_score_feature_order_and_values(
    group_request: EstimatorGroupRequest,
    bundle: EstimatorBundle,
    mutation: str,
    message: str,
) -> None:
    document = bundle_to_document(bundle, group_request)
    checkpoints = list(document["checkpoints"])
    first = dict(checkpoints[0])
    scores = [list(item) for item in first["scores"]]
    if mutation == "feature":
        scores[0][0] = "foreign_feature"
    elif mutation == "feature_order":
        scores[0], scores[1] = scores[1], scores[0]
    else:
        scores[0][1] = 2
    first["scores"] = scores
    checkpoints[0] = first
    document["checkpoints"] = checkpoints

    with pytest.raises(ValueError, match=message):
        bundle_from_document(document, group_request)


def test_bundle_rejects_schedule_digest_conflict(
    group_request: EstimatorGroupRequest, bundle: EstimatorBundle
) -> None:
    document = bundle_to_document(bundle, group_request)
    checkpoints = list(document["checkpoints"])
    guided_index = next(
        index
        for index, checkpoint in enumerate(checkpoints)
        if checkpoint["method"] == "mi_correlation_guided"
    )
    checkpoints[guided_index] = {
        **checkpoints[guided_index],
        "schedule_digest": "f" * 64,
    }
    document["checkpoints"] = checkpoints

    with pytest.raises(ValueError, match="schedule digest"):
        bundle_from_document(document, group_request)


def test_bundle_rejects_uniform_schedule_digest_tampering(
    group_request: EstimatorGroupRequest, bundle: EstimatorBundle
) -> None:
    document = bundle_to_document(bundle, group_request)
    document["checkpoints"] = [
        (
            {**checkpoint, "schedule_digest": "f" * 64}
            if checkpoint["method"] == "mi_correlation_guided"
            else checkpoint
        )
        for checkpoint in document["checkpoints"]
    ]

    with pytest.raises(ValueError, match="schedule.*authority"):
        bundle_from_document(document, group_request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("logical_evaluation_count", -1, "counter"),
        ("physical_model_fit_count", True, "counter"),
        ("component_runtime_s", float("inf"), "runtime"),
    ],
)
def test_bundle_rejects_invalid_checkpoint_counters_and_times(
    group_request: EstimatorGroupRequest,
    bundle: EstimatorBundle,
    field: str,
    value: object,
    message: str,
) -> None:
    document = bundle_to_document(bundle, group_request)
    checkpoints = list(document["checkpoints"])
    checkpoints[0] = {**checkpoints[0], field: value}
    document["checkpoints"] = checkpoints

    with pytest.raises(ValueError, match=message):
        bundle_from_document(document, group_request)


def test_checkpoint_rejects_more_unique_coalitions_than_evaluations(
    group_request: EstimatorGroupRequest,
) -> None:
    with pytest.raises(ValueError, match="logical.*counter"):
        replace(
            _checkpoint(group_request, "exact", 10),
            logical_evaluation_count=4,
            logical_unique_coalition_count=5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical_model_fit_count", 39),
        ("cache_hit_count", 199),
        ("group_runtime_s", 2.9),
    ],
)
def test_bundle_rejects_totals_inconsistent_with_final_components(
    group_request: EstimatorGroupRequest,
    bundle: EstimatorBundle,
    field: str,
    value: object,
) -> None:
    document = bundle_to_document(bundle, group_request)
    document[field] = value

    with pytest.raises(ValueError, match="total|runtime"):
        bundle_from_document(document, group_request)


def test_bundle_rejects_physical_hybrid_fits(
    group_request: EstimatorGroupRequest,
    bundle: EstimatorBundle,
) -> None:
    document = bundle_to_document(bundle, group_request)
    checkpoints = []
    for checkpoint in document["checkpoints"]:
        if checkpoint["method"] == "hybrid":
            checkpoints.append(
                {**checkpoint, "physical_model_fit_count": 1}
            )
        else:
            checkpoints.append(checkpoint)
    document["checkpoints"] = checkpoints

    with pytest.raises(ValueError, match="hybrid.*physical"):
        bundle_from_document(document, group_request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("physical_model_fit_count", -1, "counter"),
        ("cache_hit_count", True, "counter"),
        ("group_runtime_s", float("nan"), "runtime"),
    ],
)
def test_bundle_rejects_invalid_group_counters_and_times(
    group_request: EstimatorGroupRequest,
    bundle: EstimatorBundle,
    field: str,
    value: object,
    message: str,
) -> None:
    document = bundle_to_document(bundle, group_request)
    document[field] = value

    with pytest.raises(ValueError, match=message):
        bundle_from_document(document, group_request)


def test_validate_bundle_rejects_request_conflict(
    group_request: EstimatorGroupRequest, bundle: EstimatorBundle
) -> None:
    conflicting_request = replace(group_request, alpha=0.9)

    with pytest.raises(ValueError, match="request digest"):
        validate_estimator_bundle(bundle, conflicting_request)


def test_checkpoint_and_bundle_are_frozen(
    bundle: EstimatorBundle,
) -> None:
    with pytest.raises(FrozenInstanceError):
        bundle.group_runtime_s = 4.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bundle.checkpoints[0].budget = 99  # type: ignore[misc]


def test_bundle_documents_do_not_depend_on_dataclass_asdict(
    group_request: EstimatorGroupRequest, bundle: EstimatorBundle
) -> None:
    # Mapping proxies deliberately make generic deep-copy serialization unsafe.
    with pytest.raises(TypeError):
        asdict(group_request)
    assert bundle_to_document(bundle, group_request)["request_digest"] == (
        group_request.digest
    )


def _builder_request(feature_count: int) -> EstimatorGroupRequest:
    features = tuple(f"f{index}" for index in range(feature_count))
    methods = (
        ("exact", "random_mc", "mi_correlation_guided", "hybrid")
        if feature_count <= 16
        else ("random_mc", "mi_correlation_guided", "hybrid")
    )
    return EstimatorGroupRequest(
        protocol_hash="a" * 64,
        dataset_id=f"toy-{feature_count}",
        seed=7,
        task="binary",
        utility="accuracy",
        ordered_features=features,
        split_hashes={"train": "b" * 64, "validation": "c" * 64},
        model_params={"n_estimators": 2, "n_jobs": 1},
        backend="cpu",
        methods=methods,
        budgets=(1, 2),
        exact_max_features=16,
        alpha=0.95,
        cache_schema=CACHE_SCHEMA_VERSION,
        checkpoint_every_fits=2,
        progress_every_seconds=30,
    )


class _FakeSharedCache:
    def __init__(self, identity: object) -> None:
        self.identity = identity
        self.feature_order = identity.ordered_features
        self._known: set[tuple[str, ...]] = set()
        self._physical_fits = 0
        self._cache_hits = 0
        self.evaluation_call_count = 0

    @property
    def stats(self) -> CoalitionCacheStats:
        return CoalitionCacheStats(
            physical_model_fit_count=self._physical_fits,
            cache_hit_count=self._cache_hits,
            coalition_count=len(self._known),
        )

    def evaluate(self, coalition: object, utility: str) -> float:
        self.evaluation_call_count += 1
        key = tuple(
            feature for feature in self.feature_order if feature in coalition
        )
        if key in self._known:
            self._cache_hits += 1
        else:
            self._known.add(key)
            self._physical_fits += int(bool(key))
        return 1.0


def _install_group_builder_fakes(
    monkeypatch: pytest.MonkeyPatch,
    request: EstimatorGroupRequest,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    import experiments.icomp2026_estimator_bundle as bundle_module

    calls = SimpleNamespace(
        random_build=0,
        guided_build=0,
        exact=0,
        random=0,
        guided=0,
        hybrid=0,
        order=[],
    )
    identity = SimpleNamespace(
        schema_version=CACHE_SCHEMA_VERSION,
        protocol_hash=request.protocol_hash,
        dataset_id=request.dataset_id,
        seed=request.seed,
        namespace="selection_validation",
        ordered_features=request.ordered_features,
        task=request.task,
        class_labels=(0, 1),
        model_params=request.model_params,
        backend=request.backend,
        model_seed=request.seed,
        train_features_sha256=hash_numeric_frame(
            pd.DataFrame(columns=request.ordered_features)
        ),
        train_target_sha256=hash_numeric_array(np.array([], dtype=int)),
        eval_features_sha256=hash_numeric_frame(
            pd.DataFrame(columns=request.ordered_features)
        ),
        eval_target_sha256=hash_numeric_array(np.array([], dtype=int)),
    )
    store = SimpleNamespace(identity=identity)

    def make_cache(
        identity: object,
        X_train: object,
        y_train: object,
        X_val: object,
        y_val: object,
        store: SimpleNamespace,
    ) -> _FakeSharedCache:
        cache = _FakeSharedCache(identity)
        store.cache = cache
        return cache

    monkeypatch.setattr(bundle_module, "CoalitionUtilityCache", make_cache)

    real_random_builder = bundle_module.make_nested_random_schedule
    real_guided_builder = bundle_module.make_nested_guided_schedule

    def build_random(*args: object, **kwargs: object) -> object:
        calls.random_build += 1
        return real_random_builder(*args, **kwargs)

    def build_guided(*args: object, **kwargs: object) -> object:
        calls.guided_build += 1
        return real_guided_builder(*args, **kwargs)

    def exact(*args: object, **kwargs: object) -> dict[str, int]:
        calls.exact += 1
        calls.order.append("exact")
        view = kwargs["acc_cache"]
        for feature in request.ordered_features:
            view.eval_marginal(set(), feature)
        return {feature: 1 for feature in request.ordered_features}

    def random(schedule: object, view: object, **kwargs: object):
        calls.random += 1
        calls.order.append("random_mc")
        scores = {}
        for budget in request.budgets:
            for feature in request.ordered_features:
                view.eval_marginal(set(), feature)
            scores[budget] = {
                feature: 0 for feature in request.ordered_features
            }
        return scores

    def guided(schedule: object, view: object, **kwargs: object):
        calls.guided += 1
        calls.order.append("mi_correlation_guided")
        scores = {}
        anchor = request.ordered_features[0]
        for budget in request.budgets:
            for feature in request.ordered_features:
                coalition = set() if feature == anchor else {anchor}
                view.eval_marginal(coalition, feature)
            scores[budget] = {
                feature: (-1 if index % 2 else 1)
                for index, feature in enumerate(request.ordered_features)
            }
        return scores

    real_hybrid = bundle_module.derive_hybrid_checkpoints

    def hybrid(**kwargs: object):
        calls.hybrid += 1
        calls.order.append("hybrid")
        return real_hybrid(**kwargs)

    monkeypatch.setattr(bundle_module, "make_nested_random_schedule", build_random)
    monkeypatch.setattr(bundle_module, "make_nested_guided_schedule", build_guided)
    monkeypatch.setattr(bundle_module, "estimate_minppi_exact", exact)
    monkeypatch.setattr(bundle_module, "estimate_random_checkpoints", random)
    monkeypatch.setattr(bundle_module, "estimate_guided_checkpoints", guided)
    monkeypatch.setattr(bundle_module, "derive_hybrid_checkpoints", hybrid)
    monkeypatch.setattr(
        bundle_module,
        "build_mi_correlation_guidance",
        lambda *args, **kwargs: (
            {feature: 1.0 for feature in request.ordered_features},
            {feature: [] for feature in request.ordered_features},
        ),
    )
    return calls, store


def _method_score_map(
    bundle: EstimatorBundle, method: str
) -> dict[int, dict[str, int]]:
    return {
        checkpoint.budget: dict(checkpoint.scores)
        for checkpoint in bundle.checkpoints
        if checkpoint.method == method
    }


@pytest.mark.parametrize(
    ("feature_count", "expected_pair_count", "expected_exact_calls"),
    ((8, 8, 1), (55, 6, 0), (100, 6, 0)),
)
def test_group_build_computes_components_once_and_emits_all_pairs(
    monkeypatch: pytest.MonkeyPatch,
    feature_count: int,
    expected_pair_count: int,
    expected_exact_calls: int,
) -> None:
    request = _builder_request(feature_count)
    calls, store = _install_group_builder_fakes(monkeypatch, request)
    X = pd.DataFrame(columns=request.ordered_features)
    y = np.array([], dtype=int)

    result = build_estimator_bundle(
        request,
        X,
        y,
        X,
        y,
        store=store,
        progress_sink=lambda event: None,
    )

    expected_pairs = tuple(
        (method, budget)
        for method in request.methods
        for budget in request.budgets
    )
    assert tuple(
        (checkpoint.method, checkpoint.budget)
        for checkpoint in result.checkpoints
    ) == expected_pairs
    assert len(result.checkpoints) == expected_pair_count
    assert calls.random_build == calls.guided_build == 1
    assert calls.random == calls.guided == calls.hybrid == 1
    assert calls.exact == expected_exact_calls
    assert calls.order == (
        (["exact"] if expected_exact_calls else [])
        + ["random_mc", "mi_correlation_guided", "hybrid"]
    )

    random_scores = _method_score_map(result, "random_mc")
    guided_scores = _method_score_map(result, "mi_correlation_guided")
    hybrid_scores = _method_score_map(result, "hybrid")
    if feature_count <= 20:
        assert hybrid_scores == _method_score_map(result, "exact")
    elif feature_count <= 75:
        assert hybrid_scores == guided_scores
    else:
        assert hybrid_scores == {
            budget: {
                feature: min(
                    random_scores[budget][feature],
                    guided_scores[budget][feature],
                )
                for feature in request.ordered_features
            }
            for budget in request.budgets
        }
    assert validate_estimator_bundle(result, request) is result


@pytest.mark.parametrize(
    ("feature_count", "expected_exact_calls"),
    ((16, 1), (17, 1), (20, 1), (21, 0)),
)
def test_group_build_exact_boundary_keeps_internal_exact_private(
    monkeypatch: pytest.MonkeyPatch,
    feature_count: int,
    expected_exact_calls: int,
) -> None:
    request = _builder_request(feature_count)
    calls, store = _install_group_builder_fakes(monkeypatch, request)
    X = pd.DataFrame(columns=request.ordered_features)
    y = np.array([], dtype=int)

    result = build_estimator_bundle(
        request,
        X,
        y,
        X,
        y,
        store=store,
        progress_sink=lambda event: None,
    )

    assert calls.exact == expected_exact_calls
    assert any(
        checkpoint.method == "exact" for checkpoint in result.checkpoints
    ) is (feature_count <= 16)
    if feature_count <= 20:
        exact_scores = {feature: 1 for feature in request.ordered_features}
        assert _method_score_map(result, "hybrid") == {
            budget: exact_scores for budget in request.budgets
        }


def test_group_build_method_logical_sets_are_independent_on_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _builder_request(55)
    calls, store = _install_group_builder_fakes(monkeypatch, request)
    X = pd.DataFrame(columns=request.ordered_features)
    y = np.array([], dtype=int)

    result = build_estimator_bundle(
        request,
        X,
        y,
        X,
        y,
        store=store,
        progress_sink=lambda event: None,
    )
    final_random = [
        item for item in result.checkpoints if item.method == "random_mc"
    ][-1]
    final_guided = [
        item
        for item in result.checkpoints
        if item.method == "mi_correlation_guided"
    ][-1]

    assert calls.order == ["random_mc", "mi_correlation_guided", "hybrid"]
    assert final_random.logical_unique_coalition_count > 0
    assert final_guided.logical_unique_coalition_count > 0
    assert final_guided.physical_model_fit_count < (
        final_guided.logical_unique_coalition_count
    )
    assert result.physical_model_fit_count < sum(
        item.logical_unique_coalition_count for item in result.checkpoints
    )


def test_group_build_prefix_accounting_matches_traced_calls_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_count = 55
    request = _builder_request(feature_count)
    _, store = _install_group_builder_fakes(monkeypatch, request)
    X = pd.DataFrame(columns=request.ordered_features)
    y = np.array([], dtype=int)

    result = build_estimator_bundle(
        request,
        X,
        y,
        X,
        y,
        store=store,
        progress_sink=lambda event: None,
    )
    random = [
        item for item in result.checkpoints if item.method == "random_mc"
    ]
    guided = [
        item
        for item in result.checkpoints
        if item.method == "mi_correlation_guided"
    ]

    assert [item.logical_evaluation_count for item in random] == [
        1 + 2 * feature_count,
        1 + 4 * feature_count,
    ]
    assert [item.logical_unique_coalition_count for item in random] == [
        feature_count + 2,
        feature_count + 2,
    ]
    assert [item.physical_model_fit_count for item in random] == [
        feature_count + 1,
        feature_count + 1,
    ]
    assert [item.logical_evaluation_count for item in guided] == [
        1 + 2 * feature_count,
        1 + 4 * feature_count,
    ]
    assert [item.logical_unique_coalition_count for item in guided] == [
        feature_count + 2,
        feature_count + 2,
    ]
    assert [item.physical_model_fit_count for item in guided] == [
        feature_count - 1,
        feature_count - 1,
    ]
    assert store.cache.evaluation_call_count == 2 + 8 * feature_count
    assert result.physical_model_fit_count == 2 * feature_count


def test_group_build_feature_major_trace_does_not_fake_global_k_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module

    request = replace(_builder_request(3), methods=("random_mc",))
    _, store = _install_group_builder_fakes(monkeypatch, request)

    def feature_major_random(
        schedule: object, view: object, **kwargs: object
    ) -> dict[int, dict[str, int]]:
        scores = {budget: {} for budget in request.budgets}
        trials = {
            "f0": (set(), {"f1"}),
            "f1": ({"f0"}, set()),
            "f2": (set(), set()),
        }
        for feature in request.ordered_features:
            for budget, coalition in zip(
                request.budgets, trials[feature], strict=True
            ):
                view.eval_marginal(coalition, feature)
                scores[budget][feature] = 0
        return scores

    monkeypatch.setattr(
        bundle_module, "estimate_random_checkpoints", feature_major_random
    )
    X = pd.DataFrame(columns=request.ordered_features)
    y = np.array([], dtype=int)

    result = build_estimator_bundle(
        request,
        X,
        y,
        X,
        y,
        store=store,
        progress_sink=lambda event: None,
    )
    first, second = result.checkpoints

    # f0/K=2 creates {f0,f1} before f1/K=1 hits it. The K=1 physical
    # accounting must include that observed hit, but not misattribute the
    # excluded higher-budget fit to the lower logical prefix.
    assert (
        first.logical_evaluation_count,
        first.logical_unique_coalition_count,
        first.physical_model_fit_count,
        first.cache_hit_count,
    ) == (7, 5, 3, 3)
    assert (
        second.logical_evaluation_count,
        second.logical_unique_coalition_count,
        second.physical_model_fit_count,
        second.cache_hit_count,
    ) == (13, 6, 5, 7)
    assert store.cache.evaluation_call_count == 13


def test_group_build_real_multiclass_cache_shares_prediction_digests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import minppi_mofs.coalition_cache as cache_module

    X_train = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    X_val = pd.DataFrame(
        {"a": [0.5, 2.5, 4.5], "b": [4.5, 2.5, 0.5]}
    )
    y_train = np.array([0, 1, 2, 0, 1, 2])
    y_val = np.array([0, 1, 2])
    request = EstimatorGroupRequest(
        protocol_hash="d" * 64,
        dataset_id="multiclass-toy",
        seed=3,
        task="multiclass",
        utility="accuracy",
        ordered_features=("a", "b"),
        split_hashes={"train": "e" * 64, "validation": "f" * 64},
        model_params={"n_estimators": 2, "n_jobs": 1},
        backend="cpu",
        methods=("exact", "random_mc", "mi_correlation_guided", "hybrid"),
        budgets=(1, 2),
        exact_max_features=16,
        alpha=0.95,
        cache_schema=CACHE_SCHEMA_VERSION,
        checkpoint_every_fits=1,
        progress_every_seconds=30,
    )
    identity = CoalitionCacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        protocol_hash=request.protocol_hash,
        dataset_id=request.dataset_id,
        seed=request.seed,
        namespace="selection_validation",
        train_features_sha256=hash_numeric_frame(X_train),
        train_target_sha256=hash_numeric_array(y_train),
        eval_features_sha256=hash_numeric_frame(X_val),
        eval_target_sha256=hash_numeric_array(y_val),
        ordered_features=request.ordered_features,
        task=request.task,
        class_labels=(0, 1, 2),
        model_params=request.model_params,
        backend=request.backend,
        model_seed=request.seed,
        library_versions={"numpy": "test", "xgboost": "test"},
    )
    fit_calls: list[tuple[str, ...]] = []
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    )

    def fake_fit_predict(
        X_fit: pd.DataFrame,
        y_fit: np.ndarray,
        X_eval: pd.DataFrame,
        **kwargs: object,
    ) -> np.ndarray:
        fit_calls.append(tuple(X_fit.columns))
        return probabilities

    monkeypatch.setattr(cache_module, "fit_predict", fake_fit_predict)
    with SQLiteCoalitionStore(
        tmp_path / "bundle.sqlite3",
        identity,
        commit_every_fits=1,
    ) as store:
        result = build_estimator_bundle(
            request,
            X_train,
            y_train,
            X_val,
            y_val,
            store=store,
            progress_sink=lambda event: None,
        )
        shared_cache = CoalitionUtilityCache(
            identity, X_train, y_train, X_val, y_val, store
        )
        accuracy_view = CoalitionUtilityView(
            shared_cache, utility="accuracy", alpha=request.alpha
        )
        auc_view = CoalitionUtilityView(
            shared_cache, utility="ovr_auc", alpha=request.alpha
        )
        accuracy_view.evaluate({"a"})
        accuracy_record = shared_cache.record({"a"})
        auc_view.evaluate({"a"})
        auc_record = shared_cache.record({"a"})

    assert accuracy_record.prediction_sha256 == auc_record.prediction_sha256
    assert set(accuracy_record.utilities) == {"accuracy", "ovr_auc"}
    assert len(fit_calls) == result.physical_model_fit_count
    assert result.physical_model_fit_count < sum(
        item.logical_unique_coalition_count for item in result.checkpoints
    )
    assert validate_estimator_bundle(result, request) is result


def test_interrupted_ensure_resumes_committed_cache_without_cold_refit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import minppi_mofs.coalition_cache as cache_module

    request = replace(
        _builder_request(8),
        methods=("exact",),
        checkpoint_every_fits=10,
    )
    X_train = pd.DataFrame(
        np.arange(96, dtype=float).reshape(12, 8),
        columns=request.ordered_features,
    )
    X_val = pd.DataFrame(
        np.arange(32, dtype=float).reshape(4, 8),
        columns=request.ordered_features,
    )
    y_train = np.array([0, 1] * 6)
    y_val = np.array([0, 1, 0, 1])
    frames = (X_train, y_train, X_val, y_val)
    physical_fits: list[tuple[str, ...]] = []

    def fake_fit_predict(
        X_fit: pd.DataFrame,
        y_fit: np.ndarray,
        X_eval: pd.DataFrame,
        **kwargs: object,
    ) -> np.ndarray:
        del y_fit, kwargs
        physical_fits.append(tuple(X_fit.columns))
        return np.tile(np.array([[0.5, 0.5]]), (len(X_eval), 1))

    monkeypatch.setattr(cache_module, "fit_predict", fake_fit_predict)

    class InjectedStop(RuntimeError):
        pass

    def stop_after_51(count: int) -> None:
        if count >= 51:
            raise InjectedStop

    interrupted_root = tmp_path / "interrupted"
    with pytest.raises(InjectedStop):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=interrupted_root,
            progress_sink=lambda event: None,
            build_owner_job_id="a" * 64,
            fit_hook=stop_after_51,
        )

    progress_path = interrupted_root / progress_relpath(request)
    bundle_path = interrupted_root / bundle_relpath(request)
    cache_path = (
        interrupted_root
        / "cache"
        / "selection_validation"
        / request.dataset_id
        / f"s{request.seed}"
        / "coalitions.sqlite3"
    )
    assert progress_path.exists()
    assert not bundle_path.exists()
    interrupted_progress = progress_from_document(
        read_canonical_json(
            progress_path, context="interrupted estimator progress"
        )
    )
    assert interrupted_progress.build_owner_run_id == interrupted_root.name
    assert interrupted_progress.build_owner_job_id == "a" * 64
    assert interrupted_progress.initial_fitted_entry_count == 0
    with sqlite3.connect(cache_path) as connection:
        first_entry_count = connection.execute(
            "SELECT COUNT(*) FROM coalition_utility"
        ).fetchone()[0]
    assert first_entry_count >= 51
    assert len(physical_fits) == 51

    before_resume = len(physical_fits)
    resumed = ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=interrupted_root,
        progress_sink=lambda event: None,
        build_owner_job_id="a" * 64,
        fit_hook=None,
    )
    resumed_physical_fits = len(physical_fits) - before_resume
    assert bundle_path.exists()
    with sqlite3.connect(cache_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM coalition_utility"
        ).fetchone()[0] >= first_entry_count

    cold_root = tmp_path / "cold"
    before_cold = len(physical_fits)
    cold = ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=cold_root,
        progress_sink=lambda event: None,
        build_owner_job_id="b" * 64,
        fit_hook=None,
    )
    cold_physical_fits = len(physical_fits) - before_cold

    final_progress = progress_from_document(
        read_canonical_json(
            progress_path, context="resumed estimator progress"
        )
    )
    assert resumed.physical_model_fit_count == (
        51 + resumed_physical_fits
    )
    assert resumed.physical_model_fit_count == cold_physical_fits
    assert cold.physical_model_fit_count == cold_physical_fits
    assert resumed_physical_fits < cold_physical_fits
    assert (
        resumed.cache_hit_count
        > interrupted_progress.event.cache_hit_count
    )
    assert resumed.cache_hit_count > cold.cache_hit_count
    assert (
        resumed.group_runtime_s
        >= interrupted_progress.event.elapsed_s
    )
    assert final_progress.event.algorithm == "complete"
    assert (
        final_progress.build_owner_run_id
        == interrupted_progress.build_owner_run_id
    )
    assert (
        final_progress.build_owner_job_id
        == interrupted_progress.build_owner_job_id
    )
    assert (
        final_progress.initial_fitted_entry_count
        == interrupted_progress.initial_fitted_entry_count
        == 0
    )
    assert (
        final_progress.event.physical_model_fit_count
        == resumed.physical_model_fit_count
    )
    assert (
        final_progress.event.cache_hit_count
        == resumed.cache_hit_count
    )
    assert final_progress.event.elapsed_s == resumed.group_runtime_s


def _valid_ensure_frames(
    request: EstimatorGroupRequest,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    feature_count = len(request.ordered_features)
    X_train = pd.DataFrame(
        np.arange(6 * feature_count, dtype=float).reshape(
            6, feature_count
        ),
        columns=request.ordered_features,
    )
    X_val = pd.DataFrame(
        np.arange(2 * feature_count, dtype=float).reshape(
            2, feature_count
        ),
        columns=request.ordered_features,
    )
    return (
        X_train,
        np.array([0, 1, 0, 1, 0, 1]),
        X_val,
        np.array([0, 1]),
    )


def _progress_event(**changes: object) -> EstimatorProgressEvent:
    values: dict[str, object] = {
        "request_digest": "a" * 64,
        "schedule_digest": "b" * 64,
        "dataset_id": "madelon",
        "seed": 42,
        "algorithm": "random",
        "feature_index": 37,
        "feature_count": 100,
        "budget": 100,
        "max_budget": 100,
        "physical_model_fit_count": 8421,
        "cache_hit_count": 3105,
        "elapsed_s": 21 * 60.0,
        "eta_s": 34 * 60.0,
    }
    values.update(changes)
    return EstimatorProgressEvent(**values)


def test_progress_payload_round_trip_is_exact_and_redacted() -> None:
    event = _progress_event()

    document = progress_to_document(
        event,
        last_completed_trial_index=7200,
        build_owner_run_id="icomp2026-cold-20260726",
        build_owner_job_id="c" * 64,
        initial_fitted_entry_count=0,
    )

    assert set(document) == {
        "build_owner_run_id",
        "build_owner_job_id",
        "initial_fitted_entry_count",
        "request_digest",
        "schedule_digest",
        "dataset_id",
        "seed",
        "algorithm",
        "feature_index",
        "feature_count",
        "budget",
        "max_budget",
        "physical_model_fit_count",
        "cache_hit_count",
        "elapsed_s",
        "eta_s",
        "last_completed_trial_index",
    }
    restored = progress_from_document(document)
    assert restored.event == event
    assert restored.build_owner_run_id == "icomp2026-cold-20260726"
    assert restored.build_owner_job_id == "c" * 64
    assert restored.initial_fitted_entry_count == 0
    line = format_progress(event)
    assert line == (
        "[madelon seed=42 random] feature 37/100, K=100/100 "
        "fits=8421 cache_hits=3105 elapsed=21m ETA=34m"
    )
    assert "label" not in line
    assert "target" not in line
    assert "0.123456789" not in line


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("build_owner_run_id", "", "owner"),
        ("build_owner_run_id", "../foreign-run", "owner"),
        ("build_owner_job_id", "", "owner job"),
        ("build_owner_job_id", "not-a-job-id", "owner job"),
        ("initial_fitted_entry_count", -1, "initial fitted"),
        ("initial_fitted_entry_count", True, "initial fitted"),
    ],
)
def test_progress_rejects_invalid_build_provenance(
    field: str, value: object, message: str
) -> None:
    document = progress_to_document(
        _progress_event(),
        last_completed_trial_index=0,
        build_owner_run_id="icomp2026-cold-20260726",
        build_owner_job_id="c" * 64,
        initial_fitted_entry_count=0,
    )
    document[field] = value

    with pytest.raises(ValueError, match=message):
        progress_from_document(document)


@pytest.mark.parametrize(
    ("timestamps", "remaining", "expected"),
    [
        ((), 10, None),
        ((4.0,), 10, None),
        ((4.0, 4.0), 10, None),
        ((4.0, 6.0), 10, 20.0),
        ((1.0, 2.0, 5.0), 8, 16.0),
        ((4.0, 6.0), 0, 0.0),
    ],
)
def test_eta_uses_exact_recent_physical_fit_throughput(
    timestamps: tuple[float, ...],
    remaining: int,
    expected: float | None,
) -> None:
    assert estimate_eta_s(timestamps, remaining) == expected


def test_eta_converts_remaining_marginals_to_physical_fit_upper_bound() -> None:
    assert remaining_physical_fit_upper_bound(0) == 0
    assert remaining_physical_fit_upper_bound(1) == 2
    assert remaining_physical_fit_upper_bound(375) == 750


@pytest.mark.parametrize(
    "timestamps",
    [
        (2.0, 1.0),
        (-1.0, 2.0),
        (1.0, float("inf")),
        (1.0, float("nan")),
    ],
)
def test_eta_rejects_nonmonotone_or_nonfinite_times(
    timestamps: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        estimate_eta_s(timestamps, 10)


def test_completed_bundle_reuse_validates_identical_frames_without_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import minppi_mofs.coalition_cache as cache_module

    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    bundle = ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
        build_owner_job_id="a" * 64,
    )

    def forbid_fit(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("completed reuse performed a physical fit")

    monkeypatch.setattr(cache_module, "fit_predict", forbid_fit)
    reopened = ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
        build_owner_job_id="b" * 64,
        fit_hook=lambda count: pytest.fail(
            f"completed reuse crossed fit boundary {count}"
        ),
    )

    assert reopened == bundle
    assert reopened is not bundle
    progress = progress_from_document(
        read_canonical_json(
            tmp_path / progress_relpath(request),
            context="completed estimator progress",
        )
    )
    assert progress.build_owner_job_id == "a" * 64


def test_completed_bundle_reuse_fails_closed_when_progress_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    fits = _install_constant_binary_fit(monkeypatch)
    ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )
    (tmp_path / progress_relpath(request)).unlink()
    before_reuse = len(fits)

    with pytest.raises(
        ValueError, match="progress authority|recovery state is incomplete"
    ):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )

    assert len(fits) == before_reuse


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("build_owner_run_id", "foreign-run"),
        ("initial_fitted_entry_count", 1),
    ],
)
def test_completed_bundle_reuse_rejects_tampered_build_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )
    progress_path = tmp_path / progress_relpath(request)
    document = read_canonical_json(
        progress_path, context="estimator progress"
    )
    document[field] = replacement
    atomic_write_json(progress_path, document)

    with pytest.raises(ValueError, match="build provenance"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


def test_completed_bundle_reuse_rejects_changed_finite_frame_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )
    changed_train = frames[0].copy()
    changed_train.iloc[0, 0] += 0.25

    with pytest.raises(ValueError, match="identity|metadata"):
        ensure_estimator_bundle(
            request=request,
            frames=(changed_train, *frames[1:]),
            run_root=tmp_path,
            progress_sink=lambda event: None,
            fit_hook=lambda count: pytest.fail(
                f"changed frames crossed fit boundary {count}"
            ),
        )


@pytest.mark.parametrize("conflict", ["missing", "metadata"])
def test_completed_bundle_reuse_requires_matching_persisted_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    conflict: str,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )
    cache_path = (
        tmp_path
        / "cache"
        / "selection_validation"
        / request.dataset_id
        / f"s{request.seed}"
        / "coalitions.sqlite3"
    )
    if conflict == "missing":
        cache_path.unlink()
    else:
        with sqlite3.connect(cache_path) as connection:
            connection.execute(
                "UPDATE metadata SET value = ? "
                "WHERE key = 'identity_sha256'",
                ("f" * 64,),
            )

    with pytest.raises(ValueError, match="cache|metadata|identity"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
            fit_hook=lambda count: pytest.fail(
                f"cache conflict crossed fit boundary {count}"
            ),
        )


def test_completed_bundle_reuse_rejects_bundle_total_conflicting_with_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    fits = _install_constant_binary_fit(monkeypatch)
    ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )
    bundle_path = tmp_path / bundle_relpath(request)
    document = read_canonical_json(
        bundle_path, context="completed estimator bundle"
    )
    assert isinstance(document, dict)
    document["physical_model_fit_count"] += 1
    atomic_write_json(bundle_path, document)
    before_reuse = len(fits)

    with pytest.raises(ValueError, match="bundle.*cache totals"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )

    assert len(fits) == before_reuse


def test_conflicting_canonical_bundle_fails_closed(
    group_request: EstimatorGroupRequest,
    bundle: EstimatorBundle,
    tmp_path: Path,
) -> None:
    document = bundle_to_document(bundle, group_request)
    document["request_digest"] = "f" * 64
    atomic_write_json(
        tmp_path / bundle_relpath(group_request), document
    )

    with pytest.raises(ValueError, match="request digest"):
        ensure_estimator_bundle(
            request=group_request,
            frames=_valid_ensure_frames(group_request),
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


def test_malformed_progress_fails_before_cache_replay(
    group_request: EstimatorGroupRequest,
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / progress_relpath(group_request)
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="progress.*malformed"):
        ensure_estimator_bundle(
            request=group_request,
            frames=_valid_ensure_frames(group_request),
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


def _install_constant_binary_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, ...]]:
    import minppi_mofs.coalition_cache as cache_module

    physical_fits: list[tuple[str, ...]] = []

    def fake_fit_predict(
        X_fit: pd.DataFrame,
        y_fit: np.ndarray,
        X_eval: pd.DataFrame,
        **kwargs: object,
    ) -> np.ndarray:
        del y_fit, kwargs
        physical_fits.append(tuple(X_fit.columns))
        return np.tile(np.array([[0.5, 0.5]]), (len(X_eval), 1))

    monkeypatch.setattr(cache_module, "fit_predict", fake_fit_predict)
    return physical_fits


def _interrupt_after_first_fit(
    *,
    request: EstimatorGroupRequest,
    frames: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    run_root: Path,
) -> None:
    class InjectedStop(RuntimeError):
        pass

    def stop_after_first(count: int) -> None:
        if count >= 1:
            raise InjectedStop

    with pytest.raises(InjectedStop):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=run_root,
            progress_sink=lambda event: None,
            fit_hook=stop_after_first,
        )


def test_active_resume_rejects_different_owner_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    _interrupt_after_first_fit(
        request=request, frames=frames, run_root=tmp_path
    )

    with pytest.raises(ValueError, match="active.*owner job"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
            build_owner_job_id="d" * 64,
        )


@pytest.mark.parametrize(
    "digest_field", ["request_digest", "schedule_digest"]
)
def test_resume_rejects_request_or_schedule_digest_conflict_before_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    digest_field: str,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    fits = _install_constant_binary_fit(monkeypatch)
    _interrupt_after_first_fit(
        request=request, frames=frames, run_root=tmp_path
    )
    progress_path = tmp_path / progress_relpath(request)
    document = read_canonical_json(
        progress_path, context="test progress"
    )
    assert isinstance(document, dict)
    document[digest_field] = "f" * 64
    atomic_write_json(progress_path, document)
    before = len(fits)

    with pytest.raises(ValueError, match="request or schedule digest"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )

    assert len(fits) == before


def test_resume_rejects_progress_physical_total_conflicting_with_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    fits = _install_constant_binary_fit(monkeypatch)
    _interrupt_after_first_fit(
        request=request, frames=frames, run_root=tmp_path
    )
    progress_path = tmp_path / progress_relpath(request)
    document = read_canonical_json(
        progress_path, context="test progress"
    )
    assert isinstance(document, dict)
    document["physical_model_fit_count"] += 1
    atomic_write_json(progress_path, document)
    before_resume = len(fits)

    with pytest.raises(ValueError, match="progress.*cache totals"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )

    assert len(fits) == before_resume


def test_corrupt_sqlite_entry_fails_closed_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    _interrupt_after_first_fit(
        request=request, frames=frames, run_root=tmp_path
    )
    cache_path = (
        tmp_path
        / "cache"
        / "selection_validation"
        / request.dataset_id
        / f"s{request.seed}"
        / "coalitions.sqlite3"
    )
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            "UPDATE coalition_utility SET payload_sha256 = ?",
            ("0" * 64,),
        )

    with pytest.raises(ValueError, match="cache"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


def test_unreferenced_corrupt_sqlite_entry_also_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    _interrupt_after_first_fit(
        request=request, frames=frames, run_root=tmp_path
    )
    cache_path = (
        tmp_path
        / "cache"
        / "selection_validation"
        / request.dataset_id
        / f"s{request.seed}"
        / "coalitions.sqlite3"
    )
    payload = "{}"
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            "INSERT INTO coalition_utility"
            "(coalition_key, payload_json, payload_sha256) "
            "VALUES (?, ?, ?)",
            (
                '["outside-authority"]',
                payload,
                __import__("hashlib").sha256(
                    payload.encode("utf-8")
                ).hexdigest(),
            ),
        )

    with pytest.raises(ValueError, match="cache"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


def test_bundle_publication_is_reopened_and_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module

    request = replace(
        _builder_request(2),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    real_atomic_write = bundle_module.atomic_write_json

    def tamper_bundle(path: Path, value: object) -> None:
        if path.name == "bundle.json":
            assert isinstance(value, dict)
            value = {**value, "request_digest": "f" * 64}
        real_atomic_write(path, value)

    monkeypatch.setattr(
        bundle_module, "atomic_write_json", tamper_bundle
    )

    with pytest.raises(ValueError, match="request digest"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


def test_completed_bundle_repairs_post_publication_progress_once_without_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module
    import minppi_mofs.coalition_cache as cache_module

    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    fits = _install_constant_binary_fit(monkeypatch)
    bundle_path = tmp_path / bundle_relpath(request)
    progress_path = tmp_path / progress_relpath(request)
    real_atomic_write = bundle_module.atomic_write_json

    class InjectedStop(RuntimeError):
        pass

    def stop_after_bundle_publication(
        path: Path, value: object
    ) -> None:
        real_atomic_write(path, value)
        if path == bundle_path:
            raise InjectedStop

    monkeypatch.setattr(
        bundle_module,
        "atomic_write_json",
        stop_after_bundle_publication,
    )
    with pytest.raises(InjectedStop):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )

    assert bundle_path.exists()
    interrupted = progress_from_document(
        read_canonical_json(
            progress_path, context="post-publication progress"
        )
    )
    assert interrupted.event.algorithm != "complete"
    fits_before_repair = len(fits)

    def forbid_fit(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("progress repair performed a physical fit")

    monkeypatch.setattr(bundle_module, "atomic_write_json", real_atomic_write)
    monkeypatch.setattr(cache_module, "fit_predict", forbid_fit)
    emitted: list[EstimatorProgressEvent] = []
    repaired_bundle = ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=emitted.append,
    )
    repaired = progress_from_document(
        read_canonical_json(progress_path, context="repaired progress")
    )

    assert len(fits) == fits_before_repair
    assert repaired.event.algorithm == "complete"
    assert (
        repaired.event.physical_model_fit_count
        == repaired_bundle.physical_model_fit_count
    )
    assert (
        repaired.event.cache_hit_count
        == repaired_bundle.cache_hit_count
    )
    assert repaired.event.elapsed_s == repaired_bundle.group_runtime_s
    assert [event.algorithm for event in emitted] == ["complete"]

    ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=emitted.append,
    )
    assert [event.algorithm for event in emitted] == ["complete"]


def test_completed_bundle_rejects_conflicting_post_publication_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module

    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    fits = _install_constant_binary_fit(monkeypatch)
    bundle_path = tmp_path / bundle_relpath(request)
    progress_path = tmp_path / progress_relpath(request)
    real_atomic_write = bundle_module.atomic_write_json

    class InjectedStop(RuntimeError):
        pass

    def stop_after_bundle_publication(
        path: Path, value: object
    ) -> None:
        real_atomic_write(path, value)
        if path == bundle_path:
            raise InjectedStop

    monkeypatch.setattr(
        bundle_module,
        "atomic_write_json",
        stop_after_bundle_publication,
    )
    with pytest.raises(InjectedStop):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )

    progress_document = read_canonical_json(
        progress_path, context="post-publication progress"
    )
    assert isinstance(progress_document, dict)
    progress_document["cache_hit_count"] += 1
    real_atomic_write(progress_path, progress_document)
    monkeypatch.setattr(bundle_module, "atomic_write_json", real_atomic_write)
    fits_before_recovery = len(fits)

    with pytest.raises(ValueError, match="progress.*bundle"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )

    assert len(fits) == fits_before_recovery


def test_progress_state_persists_each_actual_cache_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module

    request = replace(
        _builder_request(4),
        methods=("exact",),
        checkpoint_every_fits=2,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    real_atomic_write = bundle_module.atomic_write_json
    persisted_fit_counts: list[int] = []

    def record_progress(path: Path, value: object) -> None:
        if path.name == "progress.json":
            assert isinstance(value, dict)
            persisted_fit_counts.append(
                value["physical_model_fit_count"]
            )
        real_atomic_write(path, value)

    monkeypatch.setattr(
        bundle_module, "atomic_write_json", record_progress
    )

    class InjectedStop(RuntimeError):
        pass

    def stop_after_five(count: int) -> None:
        if count >= 5:
            raise InjectedStop

    with pytest.raises(InjectedStop):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
            fit_hook=stop_after_five,
        )

    assert {2, 4, 5} <= set(persisted_fit_counts)
    durable = read_canonical_json(
        tmp_path / progress_relpath(request),
        context="test progress",
    )
    assert isinstance(durable, dict)
    assert durable["physical_model_fit_count"] == 5


def test_feature_major_progress_persists_real_boundary_not_global_k(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module

    request = replace(
        _builder_request(3),
        methods=("random_mc",),
        checkpoint_every_fits=100,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    real_atomic_write = bundle_module.atomic_write_json
    boundaries: list[dict[str, object]] = []

    def record_progress(path: Path, value: object) -> None:
        if (
            path.name == "progress.json"
            and isinstance(value, dict)
            and value.get("algorithm") == "random"
        ):
            boundaries.append(dict(value))
        real_atomic_write(path, value)

    monkeypatch.setattr(
        bundle_module, "atomic_write_json", record_progress
    )

    ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )

    feature_one_k_one = next(
        item
        for item in boundaries
        if item["feature_index"] == 1 and item["budget"] == 1
    )
    assert feature_one_k_one["last_completed_trial_index"] == 1
    assert feature_one_k_one["last_completed_trial_index"] != (
        len(request.ordered_features) * 1
    )


def test_empty_matching_cache_without_progress_recovers_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module

    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    identity = bundle_module._cache_identity(request, *frames)
    cache_path = (
        tmp_path
        / "cache"
        / "selection_validation"
        / request.dataset_id
        / f"s{request.seed}"
        / "coalitions.sqlite3"
    )
    with SQLiteCoalitionStore(
        cache_path,
        identity,
        commit_every_fits=request.checkpoint_every_fits,
    ) as store:
        assert store.entry_count == 0
    _install_constant_binary_fit(monkeypatch)

    bundle = ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )

    assert bundle.request_digest == request.digest
    assert (tmp_path / progress_relpath(request)).exists()
    assert (tmp_path / bundle_relpath(request)).exists()


def test_nonempty_cache_without_progress_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    _interrupt_after_first_fit(
        request=request, frames=frames, run_root=tmp_path
    )
    (tmp_path / progress_relpath(request)).unlink()

    with pytest.raises(ValueError, match="without validated progress"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


def _write_zero_progress(
    *,
    request: EstimatorGroupRequest,
    run_root: Path,
    physical_fits: int = 0,
    cache_hits: int = 0,
    last_completed_trial_index: int = 0,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module

    schedules = bundle_module._schedule_authority(request)
    event = EstimatorProgressEvent(
        request_digest=request.digest,
        schedule_digest=schedules.master_digest,
        dataset_id=request.dataset_id,
        seed=request.seed,
        algorithm="exact",
        feature_index=0,
        feature_count=len(request.ordered_features),
        budget=0,
        max_budget=2 ** (len(request.ordered_features) - 1),
        physical_model_fit_count=physical_fits,
        cache_hit_count=cache_hits,
        elapsed_s=0.0,
        eta_s=None,
    )
    atomic_write_json(
        run_root / progress_relpath(request),
        progress_to_document(
            event,
            last_completed_trial_index=last_completed_trial_index,
            build_owner_run_id=run_root.name,
            build_owner_job_id=request.digest,
            initial_fitted_entry_count=0,
        ),
    )


def test_zero_progress_without_cache_recovers_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _write_zero_progress(request=request, run_root=tmp_path)
    _install_constant_binary_fit(monkeypatch)

    bundle = ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )

    assert bundle.request_digest == request.digest
    assert (tmp_path / bundle_relpath(request)).exists()


@pytest.mark.parametrize(
    ("physical_fits", "cache_hits", "trial_index"),
    [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
def test_nonzero_progress_without_cache_fails_closed(
    tmp_path: Path,
    physical_fits: int,
    cache_hits: int,
    trial_index: int,
) -> None:
    request = replace(_builder_request(3), methods=("exact",))
    _write_zero_progress(
        request=request,
        run_root=tmp_path,
        physical_fits=physical_fits,
        cache_hits=cache_hits,
        last_completed_trial_index=trial_index,
    )

    with pytest.raises(ValueError, match="without its coalition cache"):
        ensure_estimator_bundle(
            request=request,
            frames=_valid_ensure_frames(request),
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


@pytest.mark.parametrize(
    ("corruption", "replacement"),
    [
        ("utilities", {"accuracy": 0.5}),
        ("prediction_shape", [999, 2]),
        ("prediction_dtype", "object"),
        ("fitted", False),
    ],
)
def test_completed_reuse_semantically_validates_every_cache_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corruption: str,
    replacement: object,
) -> None:
    import minppi_mofs.coalition_cache as cache_module

    request = replace(
        _builder_request(3),
        methods=("exact",),
        checkpoint_every_fits=1,
    )
    frames = _valid_ensure_frames(request)
    _install_constant_binary_fit(monkeypatch)
    ensure_estimator_bundle(
        request=request,
        frames=frames,
        run_root=tmp_path,
        progress_sink=lambda event: None,
    )
    cache_path = (
        tmp_path
        / "cache"
        / "selection_validation"
        / request.dataset_id
        / f"s{request.seed}"
        / "coalitions.sqlite3"
    )
    with sqlite3.connect(cache_path) as connection:
        coalition_key, payload_text = connection.execute(
            "SELECT coalition_key, payload_json "
            "FROM coalition_utility WHERE coalition_key != '[]' LIMIT 1"
        ).fetchone()
        payload = json.loads(payload_text)
        payload[corruption] = replacement
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            "UPDATE coalition_utility "
            "SET payload_json = ?, payload_sha256 = ? "
            "WHERE coalition_key = ?",
            (
                canonical,
                hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                coalition_key,
            ),
        )

    monkeypatch.setattr(
        cache_module,
        "fit_predict",
        lambda *args, **kwargs: pytest.fail(
            "semantic cache audit performed a physical fit"
        ),
    )
    with pytest.raises(ValueError, match="cached record|numeric"):
        ensure_estimator_bundle(
            request=request,
            frames=frames,
            run_root=tmp_path,
            progress_sink=lambda event: None,
        )


@pytest.mark.parametrize(
    "changed_request",
    [
        replace(_builder_request(3), protocol_hash="f" * 64),
        replace(
            _builder_request(3),
            split_hashes={"train": "d" * 64, "validation": "c" * 64},
        ),
        replace(
            _builder_request(3),
            model_params={"n_estimators": 3, "n_jobs": 1},
        ),
        replace(_builder_request(3), alpha=0.9),
        replace(_builder_request(3), checkpoint_every_fits=3),
        replace(_builder_request(3), progress_every_seconds=31),
    ],
)
def test_master_schedule_digest_is_sensitive_to_full_request_identity(
    changed_request: EstimatorGroupRequest,
) -> None:
    import experiments.icomp2026_estimator_bundle as bundle_module

    request = _builder_request(3)

    assert request.digest != changed_request.digest
    assert (
        bundle_module._schedule_authority(request).master_digest
        != bundle_module._schedule_authority(changed_request).master_digest
    )
