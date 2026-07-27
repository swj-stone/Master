from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.validate_grouped_acceleration as validation_module
from experiments.validate_grouped_acceleration import (
    GateResult,
    abalone_exact_gate,
    classification_gate,
    compression_gate,
    main,
    parse_named_runs,
    prediction_digest_gate,
    r2_gate,
    read_run_metadata,
    reject_test_artifact_fields,
    speed_gate,
)
from minppi_mofs.experiment_protocol import read_canonical_json


def test_gate_result_is_frozen() -> None:
    result = classification_gate(legacy=0.9, candidate=0.895)

    with pytest.raises(FrozenInstanceError):
        result.passed = False


@pytest.mark.parametrize(
    ("candidate", "passed"),
    [(0.895, True), (0.894999999999, True), (0.8949, False)],
)
def test_classification_noninferiority_threshold(
    candidate: float, passed: bool
) -> None:
    assert classification_gate(legacy=0.9, candidate=candidate).passed is passed


@pytest.mark.parametrize(
    ("candidate", "passed"),
    [(0.69, True), (0.689999999999, True), (0.689, False)],
)
def test_r2_noninferiority_threshold(candidate: float, passed: bool) -> None:
    assert r2_gate(legacy=0.7, candidate=candidate).passed is passed


def test_compression_gate_accepts_either_compression_or_utility_branch() -> None:
    assert compression_gate(0.3, 0.28, 0.9, 0.9).passed
    assert compression_gate(0.3, 0.27, 0.9, 0.905).passed
    assert not compression_gate(0.3, 0.27, 0.9, 0.9049).passed


@pytest.mark.parametrize(
    ("dataset_id", "legacy", "candidate", "passed"),
    [
        ("spambase", 300.0, 120.0, True),
        ("spambase", 300.0, 120.1, False),
        ("madelon", 350.0, 100.0, True),
        ("madelon", 350.0, 100.1, False),
    ],
)
def test_speed_gate_uses_exact_cold_thresholds(
    dataset_id: str,
    legacy: float,
    candidate: float,
    passed: bool,
) -> None:
    assert speed_gate(dataset_id, legacy, candidate, "cold").passed is passed


def test_speed_gate_rejects_warm_or_unauthorized_evidence() -> None:
    with pytest.raises(ValueError, match="cold"):
        speed_gate("spambase", 300.0, 120.0, "warm")
    with pytest.raises(ValueError, match="authorized"):
        speed_gate("isolet", 300.0, 120.0, "cold")


def test_prediction_digest_gate_requires_exact_digest_equality() -> None:
    digest = "a" * 64
    assert prediction_digest_gate(digest, digest).passed
    assert not prediction_digest_gate(digest, "b" * 64).passed


def test_abalone_exact_gate_uses_five_seed_means_and_zero_harmful_convention() -> None:
    seeds = (42, 123, 456, 789, 1024)
    exact = {
        seed: {"f0": 0, "f1": -1 if seed != 42 else 0}
        for seed in seeds
    }
    legacy = {
        seed: {"f0": 0, "f1": -1 if seed != 42 else 0}
        for seed in seeds
    }
    candidate = {
        seed: {"f0": 0, "f1": -1 if seed != 42 else 0}
        for seed in seeds
    }
    assert abalone_exact_gate(exact, legacy, candidate).passed

    candidate[42] = {"f0": -1, "f1": 0}
    failed = abalone_exact_gate(exact, legacy, candidate)
    assert not failed.passed
    assert "harmful_recall" in failed.detail


def test_named_run_parser_requires_exact_names_and_absolute_paths(
    tmp_path: Path,
) -> None:
    paths = {
        name: (tmp_path / name).resolve()
        for name in ("abalone", "spambase", "madelon", "isolet")
    }
    parsed = parse_named_runs(
        [f"{name}={path}" for name, path in paths.items()]
    )
    assert parsed == paths

    with pytest.raises(ValueError, match="exact"):
        parse_named_runs([f"abalone={paths['abalone']}"])
    with pytest.raises(ValueError, match="absolute"):
        parse_named_runs(
            [
                "abalone=relative",
                *(f"{name}={paths[name]}" for name in ("spambase", "madelon", "isolet")),
            ]
        )


def _report(passed: bool) -> dict[str, object]:
    gate = GateResult("synthetic", passed, 1.0, 1.0, "synthetic")
    return {
        "passed": passed,
        "gates": [gate],
        "evidence": {"held_out_test_access": False},
    }


@pytest.mark.parametrize(("passed", "expected_exit"), [(True, 0), (False, 1)])
def test_cli_writes_canonical_report_and_maps_valid_gate_result_to_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
    expected_exit: int,
) -> None:
    runs = [
        f"{name}={(tmp_path / name).resolve()}"
        for name in ("abalone", "spambase", "madelon", "isolet")
    ]
    monkeypatch.setattr(
        validation_module,
        "build_acceptance_report",
        lambda **_kwargs: _report(passed),
    )
    report = tmp_path / "acceptance.json"

    code = main(
        [
            "--config-dir",
            str(tmp_path),
            *sum(
                (
                    ["--legacy-run", item, "--candidate-run", item]
                    for item in runs
                ),
                [],
            ),
            "--report",
            str(report),
        ]
    )

    assert code == expected_exit
    payload = read_canonical_json(report, context="acceptance report")
    assert set(payload) == {"passed", "gates", "evidence"}
    assert payload["passed"] is passed


def test_cli_returns_two_and_does_not_publish_for_malformed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = [
        f"{name}={(tmp_path / name).resolve()}"
        for name in ("abalone", "spambase", "madelon", "isolet")
    ]
    monkeypatch.setattr(
        validation_module,
        "build_acceptance_report",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("malformed evidence")),
    )
    report = tmp_path / "acceptance.json"

    code = main(
        [
            "--config-dir",
            str(tmp_path),
            *sum(
                (
                    ["--legacy-run", item, "--candidate-run", item]
                    for item in runs
                ),
                [],
            ),
            "--report",
            str(report),
        ]
    )

    assert code == 2
    assert not report.exists()


def test_validation_command_never_calls_test_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.icomp2026_stages as stages

    monkeypatch.setattr(
        stages,
        "_task9_load_test",
        lambda *_args, **_kwargs: pytest.fail("validation read held-out test"),
    )
    monkeypatch.setattr(
        validation_module,
        "build_acceptance_report",
        lambda **_kwargs: _report(True),
    )
    runs = [
        f"{name}={(tmp_path / name).resolve()}"
        for name in ("abalone", "spambase", "madelon", "isolet")
    ]

    assert (
        main(
            [
                "--config-dir",
                str(tmp_path),
                *sum(
                    (
                        ["--legacy-run", item, "--candidate-run", item]
                        for item in runs
                    ),
                    [],
                ),
                "--report",
                str(tmp_path / "acceptance.json"),
            ]
        )
        == 0
    )


def test_reader_rejects_missing_internal_run_authority(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolved protocol"):
        read_run_metadata(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        {"test_utility": 0.9},
        {"artifacts": {"results": "final_test/results.json"}},
        {"rows": [{"test_front": []}]},
    ],
)
def test_reader_rejects_test_result_fields_or_paths(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="held-out test"):
        reject_test_artifact_fields(payload)


def _scientific_protocol(**changes: object) -> dict[str, object]:
    protocol: dict[str, object] = {
        "protocol_version": "1.0",
        "run_kind": "formal",
        "datasets": [
            {
                "dataset_id": "abalone",
                "task": "regression",
                "utility": "r2",
            }
        ],
        "seeds": [42, 123, 456, 789, 1024],
        "backend": "cuda",
        "model_params": {"max_depth": 4, "n_estimators": 100},
        "alpha_default": 0.95,
        "alpha_grid": [0.85, 0.9, 0.95, 0.99],
        "estimator_methods": [
            "exact",
            "random_mc",
            "mi_correlation_guided",
            "hybrid",
        ],
        "estimator_budgets": [10, 25, 50, 100],
        "exact_max_features": 16,
        "nsga2": {},
        "removal": {},
        "ablations": {},
        "smoke": {},
        "execution": {
            "mode": "legacy",
            "cache_schema": "coalition_utility_v1",
            "checkpoint_every_fits": 50,
            "progress_every_seconds": 30,
        },
    }
    protocol.update(changes)
    return protocol


def test_cross_mode_protocol_allows_only_execution_policy_difference() -> None:
    legacy = _scientific_protocol(
        datasets=[
            {
                "dataset_id": "abalone",
                "task": "regression",
                "utility": "r2",
            },
            {
                "dataset_id": "spambase",
                "task": "binary",
                "utility": "roc_auc",
            },
        ],
        seeds=[42, 123, 456, 789, 1024],
    )
    candidate = _scientific_protocol(
        datasets=[
            {
                "dataset_id": "abalone",
                "task": "regression",
                "utility": "r2",
            }
        ],
        seeds=[42],
        execution={
            "mode": "grouped_nested_v1",
            "cache_schema": "coalition_utility_v1",
            "checkpoint_every_fits": 10,
            "progress_every_seconds": 5,
        }
    )

    validation_module._validate_scientific_protocol_equivalence(
        legacy, candidate, dataset_id="abalone"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha_default", 0.9),
        ("exact_max_features", 15),
        ("estimator_budgets", [10, 25, 50, 75]),
        (
            "estimator_methods",
            ["exact", "random_mc", "hybrid"],
        ),
        ("removal", {"threshold": 0.1}),
    ],
)
def test_cross_mode_protocol_rejects_any_scientific_mismatch(
    field: str, value: object
) -> None:
    legacy = _scientific_protocol()
    candidate = _scientific_protocol(
        execution={
            "mode": "grouped_nested_v1",
            "cache_schema": "coalition_utility_v1",
            "checkpoint_every_fits": 50,
            "progress_every_seconds": 30,
        },
        **{field: value},
    )

    with pytest.raises(ValueError, match="scientific protocol"):
        validation_module._validate_scientific_protocol_equivalence(
            legacy, candidate, dataset_id="abalone"
        )


def test_abalone_evidence_requires_all_four_canonical_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_score_evidence(
        _metadata: object,
        *,
        dataset_id: str,
        seed: int,
        method: str,
        budget: int,
    ) -> dict[str, object]:
        assert (dataset_id, seed, budget) == ("abalone", 42, 100)
        calls.append(method)
        return {"scores": {method: 0}}

    monkeypatch.setattr(
        validation_module, "_score_evidence", fake_score_evidence
    )

    evidence = validation_module._abalone_method_evidence(
        object(), seed=42, budget=100
    )

    assert tuple(evidence) == (
        "exact",
        "random_mc",
        "mi_correlation_guided",
        "hybrid",
    )
    assert calls == list(evidence)


@pytest.mark.parametrize(
    ("bad_method", "failure"),
    [
        ("random_mc", "missing random score artifact"),
        ("mi_correlation_guided", "guided score artifact is malformed"),
    ],
)
def test_abalone_evidence_rejects_missing_or_corrupt_required_artifact(
    monkeypatch: pytest.MonkeyPatch,
    bad_method: str,
    failure: str,
) -> None:
    def fake_score_evidence(
        _metadata: object,
        *,
        dataset_id: str,
        seed: int,
        method: str,
        budget: int,
    ) -> dict[str, object]:
        del dataset_id, seed, budget
        if method == bad_method:
            raise ValueError(failure)
        return {"scores": {method: 0}}

    monkeypatch.setattr(
        validation_module, "_score_evidence", fake_score_evidence
    )

    with pytest.raises(ValueError, match=failure):
        validation_module._abalone_method_evidence(
            object(), seed=42, budget=100
        )


def _cold_owner_authority(
    *,
    owner_job_id: str,
    owner_dataset_id: str = "spambase",
    owner_seed: int = 42,
    owner_attempt: int = 1,
    owner_recovery: object = None,
) -> tuple[object, object, object]:
    owner = SimpleNamespace(
        job_id=owner_job_id,
        stage="estimator",
        dataset_id=owner_dataset_id,
        seed=owner_seed,
        method="random_mc",
        parameters={"budget": 100},
    )
    hybrid = SimpleNamespace(
        job_id="b" * 64,
        stage="estimator",
        dataset_id="spambase",
        seed=42,
        method="hybrid",
        parameters={"budget": 100},
    )
    metadata = SimpleNamespace(
        manifest=(owner, hybrid),
        status_entries={
            owner_job_id: {
                "state": "success",
                "attempt": owner_attempt,
                "recovery": owner_recovery,
            },
            hybrid.job_id: {
                "state": "success",
                "attempt": 1,
                "recovery": None,
            },
        },
    )
    progress = SimpleNamespace(build_owner_job_id=owner_job_id)
    request = SimpleNamespace(
        dataset_id="spambase",
        seed=42,
        methods=("random_mc", "hybrid"),
        budgets=(100,),
    )
    return metadata, progress, request


@pytest.mark.parametrize(
    ("attempt", "recovery"),
    [(2, None), (1, {"kind": "resume"})],
)
def test_cold_bundle_rejects_nonclean_owner_even_when_hybrid_is_clean(
    attempt: int, recovery: object
) -> None:
    metadata, progress, request = _cold_owner_authority(
        owner_job_id="a" * 64,
        owner_attempt=attempt,
        owner_recovery=recovery,
    )

    with pytest.raises(ValueError, match="owner.*cold"):
        validation_module._validate_cold_owner_job(
            metadata, progress=progress, request=request
        )


@pytest.mark.parametrize(
    ("progress_owner", "owner_dataset_id", "owner_seed"),
    [
        ("c" * 64, "spambase", 42),
        ("a" * 64, "madelon", 42),
        ("a" * 64, "spambase", 123),
    ],
)
def test_cold_bundle_rejects_missing_tampered_or_foreign_group_owner(
    progress_owner: str,
    owner_dataset_id: str,
    owner_seed: int,
) -> None:
    metadata, _progress, request = _cold_owner_authority(
        owner_job_id="a" * 64,
        owner_dataset_id=owner_dataset_id,
        owner_seed=owner_seed,
    )
    progress = SimpleNamespace(build_owner_job_id=progress_owner)

    with pytest.raises(ValueError, match="owner"):
        validation_module._validate_cold_owner_job(
            metadata, progress=progress, request=request
        )
