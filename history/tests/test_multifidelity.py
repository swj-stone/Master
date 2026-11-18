from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from minppi_mofs.multifidelity import (
    CalibrationRules,
    CandidateCalibration,
    DatasetModelPolicy,
    ModelPolicy,
    campaign_evidence_digest,
    load_model_policy,
    model_params_for,
    select_candidate_round,
    write_model_policy,
)


def _candidate(rounds: int, **overrides: float | int) -> CandidateCalibration:
    values: dict[str, float | int] = {
        "n_estimators": rounds,
        "full_utility": 0.9,
        "full_utility_loss": 0.0,
        "spearman": 1.0,
        "winning_agreement": 1.0,
        "marginal_sign_agreement": 1.0,
        "harmful_recall": 1.0,
        "false_harmful_rate": 0.0,
        "seconds_per_fit": 0.1,
        "physical_fits": 10,
    }
    values.update(overrides)
    return CandidateCalibration(**values)


def _policy() -> ModelPolicy:
    rules = CalibrationRules()
    return ModelPolicy(
        schema_version=1,
        calibration_seed=42,
        candidate_rounds=(10, 20, 40, 100),
        rules=rules,
        datasets={
            "isolet": DatasetModelPolicy(
                selected_n_estimators=20,
                reference_n_estimators=100,
                calibrations=(
                    _candidate(10, full_utility_loss=0.02),
                    _candidate(20),
                    _candidate(40),
                    _candidate(100),
                ),
            ),
            "madelon": DatasetModelPolicy(
                selected_n_estimators=40,
                reference_n_estimators=100,
                calibrations=(
                    _candidate(10, spearman=0.8),
                    _candidate(20, harmful_recall=0.8),
                    _candidate(40),
                    _candidate(100),
                ),
            ),
        },
    )


def test_selects_smallest_candidate_that_meets_every_rule():
    rows = (
        _candidate(10, full_utility_loss=0.02),
        _candidate(20),
        _candidate(40),
        _candidate(100),
    )

    assert select_candidate_round(rows, CalibrationRules()) == 20


def test_selection_falls_back_to_reference_when_lower_rounds_fail():
    rows = (
        _candidate(10, spearman=0.80),
        _candidate(20, marginal_sign_agreement=0.50),
        _candidate(40, false_harmful_rate=0.20),
        _candidate(100),
    )

    assert select_candidate_round(rows, CalibrationRules()) == 100


def test_policy_round_trip_is_canonical_and_fail_closed(tmp_path: Path):
    path = tmp_path / "model_policy.json"
    policy = _policy()

    write_model_policy(path, policy)
    loaded = load_model_policy(
        path, expected_datasets=("isolet", "madelon")
    )

    assert loaded == policy
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_policy_rejects_dataset_scope_mismatch(tmp_path: Path):
    path = tmp_path / "model_policy.json"
    write_model_policy(path, _policy())

    with pytest.raises(ValueError, match="dataset scope"):
        load_model_policy(path, expected_datasets=("isolet",))


def test_policy_rejects_selected_round_outside_candidates():
    policy = _policy()

    with pytest.raises(ValueError, match="selected_n_estimators"):
        replace(
            policy,
            datasets={
                **policy.datasets,
                "isolet": replace(
                    policy.datasets["isolet"], selected_n_estimators=30
                ),
            },
        )


def test_model_params_for_applies_only_dataset_round_override():
    protocol = SimpleNamespace(
        model_params={
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_jobs": 1,
        },
        model_params_by_dataset={
            "isolet": {
                "n_estimators": 20,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_jobs": 1,
            }
        },
        model_policy_digest="a" * 64,
    )

    assert model_params_for(protocol, "isolet") == {
        "n_estimators": 20,
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_jobs": 1,
    }
    with pytest.raises(ValueError, match="outside model policy"):
        model_params_for(protocol, "unknown")


def test_campaign_digest_ignores_active_seed_shard():
    base = dict(
        protocol_version="1.0",
        run_kind="formal",
        datasets=(SimpleNamespace(dataset_id="isolet", task="multiclass"),),
        backend="cuda",
        execution=SimpleNamespace(mode="grouped_nested_v1"),
        model_params_by_dataset={"isolet": {"n_estimators": 20}},
        model_policy_digest="a" * 64,
        alpha_default=0.95,
        alpha_grid=(0.85, 0.9, 0.95, 0.99),
        estimator_methods=("random_mc", "mi_correlation_guided", "hybrid"),
        estimator_budgets=(10, 25, 50),
        estimator_budgets_by_dataset={"isolet": (10, 25, 50)},
        exact_max_features=16,
        removal={"controls": ("full", "minppi"), "random_repeats": 30},
        nsga2={"population": 40, "evaluations": 800},
        ablations={"guidance": ("random", "mi_correlation")},
        stage_scopes={
            "alpha": {"datasets": ("isolet",), "seeds": (42,)},
            "ablation": {"datasets": ("isolet",), "seeds": (42,)},
        },
    )
    screen = SimpleNamespace(**base, seeds=(42,))
    confirm = SimpleNamespace(**base, seeds=(123, 456, 789, 1024))

    assert campaign_evidence_digest(screen) == campaign_evidence_digest(confirm)


def test_candidate_rejects_non_finite_metric():
    with pytest.raises(ValueError, match="finite"):
        _candidate(20, spearman=float("nan"))
