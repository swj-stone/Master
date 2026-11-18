"""Frozen model calibration and cross-seed campaign identities.

The model policy is intentionally independent from an active seed shard.  A
seed-42 screen run and a four-seed confirmation run may therefore be combined
only when every scientific setting represented by :func:`campaign_evidence_digest`
is identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


_POLICY_SCHEMA_VERSION = 1
_REFERENCE_ESTIMATORS = 100
def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _fraction(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CalibrationRules:
    """Thresholds a lower-cost model must meet against the 100-round reference."""

    max_full_utility_loss: float = 0.005
    min_spearman: float = 0.95
    min_winning_agreement: float = 0.95
    min_marginal_sign_agreement: float = 0.90
    min_harmful_recall: float = 0.90
    max_false_harmful_rate: float = 0.02

    def __post_init__(self) -> None:
        loss = _finite(self.max_full_utility_loss, "max_full_utility_loss")
        if loss < 0.0:
            raise ValueError("max_full_utility_loss must be non-negative")
        object.__setattr__(self, "max_full_utility_loss", loss)
        for field in (
            "min_spearman",
            "min_winning_agreement",
            "min_marginal_sign_agreement",
            "min_harmful_recall",
            "max_false_harmful_rate",
        ):
            object.__setattr__(self, field, _fraction(getattr(self, field), field))


@dataclass(frozen=True)
class CandidateCalibration:
    """Measured fidelity and cost for one candidate boosting-round count."""

    n_estimators: int
    full_utility: float
    full_utility_loss: float
    spearman: float
    winning_agreement: float
    marginal_sign_agreement: float
    harmful_recall: float
    false_harmful_rate: float
    seconds_per_fit: float
    physical_fits: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_estimators", _positive_int(self.n_estimators, "n_estimators")
        )
        for field in ("full_utility", "full_utility_loss", "seconds_per_fit"):
            number = _finite(getattr(self, field), field)
            if field != "full_utility" and number < 0.0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, number)
        spearman = _finite(self.spearman, "spearman")
        if not -1.0 <= spearman <= 1.0:
            raise ValueError("spearman must be in [-1, 1]")
        object.__setattr__(self, "spearman", spearman)
        for field in (
            "winning_agreement",
            "marginal_sign_agreement",
            "harmful_recall",
            "false_harmful_rate",
        ):
            object.__setattr__(self, field, _fraction(getattr(self, field), field))
        object.__setattr__(
            self, "physical_fits", _positive_int(self.physical_fits, "physical_fits")
        )

    def meets(self, rules: CalibrationRules) -> bool:
        """Return whether this candidate satisfies every frozen fidelity rule."""
        if not isinstance(rules, CalibrationRules):
            raise TypeError("rules must be CalibrationRules")
        return (
            self.full_utility_loss <= rules.max_full_utility_loss
            and self.spearman >= rules.min_spearman
            and self.winning_agreement >= rules.min_winning_agreement
            and self.marginal_sign_agreement
            >= rules.min_marginal_sign_agreement
            and self.harmful_recall >= rules.min_harmful_recall
            and self.false_harmful_rate <= rules.max_false_harmful_rate
        )


def select_candidate_round(
    calibrations: Sequence[CandidateCalibration], rules: CalibrationRules
) -> int:
    """Select the smallest passing round count, falling back to 100."""
    rows = tuple(calibrations)
    if not rows or not all(isinstance(row, CandidateCalibration) for row in rows):
        raise ValueError("calibrations must contain CandidateCalibration values")
    rounds = tuple(row.n_estimators for row in rows)
    if tuple(sorted(rounds)) != rounds or len(set(rounds)) != len(rounds):
        raise ValueError("calibrations must be strictly ordered by n_estimators")
    if _REFERENCE_ESTIMATORS not in rounds:
        raise ValueError("calibrations must contain the 100-round reference")
    for row in rows:
        if row.meets(rules):
            return row.n_estimators
    return _REFERENCE_ESTIMATORS


@dataclass(frozen=True)
class DatasetModelPolicy:
    """Frozen calibration evidence and selected rounds for one dataset."""

    selected_n_estimators: int
    reference_n_estimators: int
    calibrations: tuple[CandidateCalibration, ...]

    def __post_init__(self) -> None:
        selected = _positive_int(
            self.selected_n_estimators, "selected_n_estimators"
        )
        reference = _positive_int(
            self.reference_n_estimators, "reference_n_estimators"
        )
        rows = tuple(self.calibrations)
        if reference != _REFERENCE_ESTIMATORS:
            raise ValueError("reference_n_estimators must be 100")
        if (
            not rows
            or not all(isinstance(row, CandidateCalibration) for row in rows)
            or selected not in {row.n_estimators for row in rows}
            or reference not in {row.n_estimators for row in rows}
        ):
            raise ValueError(
                "selected_n_estimators and reference must occur in calibrations"
            )
        object.__setattr__(self, "selected_n_estimators", selected)
        object.__setattr__(self, "reference_n_estimators", reference)
        object.__setattr__(self, "calibrations", rows)


@dataclass(frozen=True)
class ModelPolicy:
    """Canonical per-dataset model policy produced by validation-only calibration."""

    schema_version: int
    calibration_seed: int
    candidate_rounds: tuple[int, ...]
    rules: CalibrationRules
    datasets: Mapping[str, DatasetModelPolicy]

    def __post_init__(self) -> None:
        if self.schema_version != _POLICY_SCHEMA_VERSION:
            raise ValueError("model policy schema_version must be 1")
        if (
            isinstance(self.calibration_seed, bool)
            or not isinstance(self.calibration_seed, int)
            or self.calibration_seed < 0
        ):
            raise ValueError("calibration_seed must be a non-negative integer")
        rounds = tuple(self.candidate_rounds)
        if (
            not rounds
            or tuple(sorted(rounds)) != rounds
            or len(set(rounds)) != len(rounds)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in rounds
            )
            or rounds[-1] != _REFERENCE_ESTIMATORS
        ):
            raise ValueError(
                "candidate_rounds must be strictly increasing and end at 100"
            )
        if not isinstance(self.rules, CalibrationRules):
            raise TypeError("rules must be CalibrationRules")
        if (
            not isinstance(self.datasets, Mapping)
            or not self.datasets
            or not all(
                isinstance(key, str)
                and key
                and isinstance(value, DatasetModelPolicy)
                for key, value in self.datasets.items()
            )
        ):
            raise ValueError("datasets must map dataset IDs to policies")
        for dataset_id, policy in self.datasets.items():
            actual = tuple(row.n_estimators for row in policy.calibrations)
            if actual != rounds:
                raise ValueError(
                    f"dataset {dataset_id} calibrations conflict with candidate_rounds"
                )
            if policy.selected_n_estimators not in rounds:
                raise ValueError(
                    f"dataset {dataset_id} selected_n_estimators is invalid"
                )
        object.__setattr__(self, "candidate_rounds", rounds)
        object.__setattr__(
            self,
            "datasets",
            {key: self.datasets[key] for key in sorted(self.datasets)},
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(model_policy_to_document(self))


def model_policy_to_document(policy: ModelPolicy) -> dict[str, object]:
    if not isinstance(policy, ModelPolicy):
        raise TypeError("policy must be ModelPolicy")
    return {
        "schema_version": policy.schema_version,
        "calibration_seed": policy.calibration_seed,
        "candidate_rounds": list(policy.candidate_rounds),
        "rules": asdict(policy.rules),
        "datasets": {
            dataset_id: {
                "selected_n_estimators": item.selected_n_estimators,
                "reference_n_estimators": item.reference_n_estimators,
                "calibrations": [asdict(row) for row in item.calibrations],
            }
            for dataset_id, item in policy.datasets.items()
        },
    }


def _model_policy_from_document(value: object) -> ModelPolicy:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "calibration_seed",
        "candidate_rounds",
        "rules",
        "datasets",
    }:
        raise ValueError("model policy schema is invalid")
    rules_value = value["rules"]
    if not isinstance(rules_value, dict) or set(rules_value) != {
        "max_full_utility_loss",
        "min_spearman",
        "min_winning_agreement",
        "min_marginal_sign_agreement",
        "min_harmful_recall",
        "max_false_harmful_rate",
    }:
        raise ValueError("model policy rules schema is invalid")
    datasets_value = value["datasets"]
    if not isinstance(datasets_value, dict):
        raise ValueError("model policy datasets schema is invalid")
    datasets: dict[str, DatasetModelPolicy] = {}
    for dataset_id, item in datasets_value.items():
        if not isinstance(item, dict) or set(item) != {
            "selected_n_estimators",
            "reference_n_estimators",
            "calibrations",
        }:
            raise ValueError(f"model policy dataset {dataset_id!r} is invalid")
        calibrations = item["calibrations"]
        if not isinstance(calibrations, list):
            raise ValueError(
                f"model policy dataset {dataset_id!r} calibrations are invalid"
            )
        try:
            rows = tuple(CandidateCalibration(**row) for row in calibrations)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model policy dataset {dataset_id!r} calibrations are invalid"
            ) from exc
        datasets[str(dataset_id)] = DatasetModelPolicy(
            selected_n_estimators=item["selected_n_estimators"],
            reference_n_estimators=item["reference_n_estimators"],
            calibrations=rows,
        )
    try:
        rules = CalibrationRules(**rules_value)
        candidate_rounds = tuple(value["candidate_rounds"])
    except (TypeError, ValueError) as exc:
        raise ValueError("model policy scalar fields are invalid") from exc
    return ModelPolicy(
        schema_version=value["schema_version"],
        calibration_seed=value["calibration_seed"],
        candidate_rounds=candidate_rounds,
        rules=rules,
        datasets=datasets,
    )


def write_model_policy(path: Path, policy: ModelPolicy) -> None:
    """Write one canonical policy atomically."""
    from minppi_mofs.experiment_protocol import atomic_write_json

    atomic_write_json(Path(path), model_policy_to_document(policy))


def load_model_policy(
    path: Path, *, expected_datasets: Sequence[str]
) -> ModelPolicy:
    """Read and validate one canonical policy against an exact dataset scope."""
    from minppi_mofs.experiment_protocol import read_canonical_json

    expected = tuple(expected_datasets)
    if (
        not expected
        or any(not isinstance(item, str) or not item for item in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("expected_datasets must contain unique dataset IDs")
    policy = _model_policy_from_document(
        read_canonical_json(Path(path), context="model policy")
    )
    if tuple(policy.datasets) != tuple(sorted(expected)):
        raise ValueError(
            "model policy dataset scope does not match expected dataset scope"
        )
    return policy


def model_params_for(protocol: object, dataset_id: str) -> dict[str, object]:
    """Return the exact frozen model parameters for one dataset."""
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    mapping = getattr(protocol, "model_params_by_dataset", None)
    policy_digest = getattr(protocol, "model_policy_digest", None)
    if policy_digest is not None:
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError("protocol model policy mapping is missing")
        if dataset_id not in mapping:
            raise ValueError(f"dataset {dataset_id!r} is outside model policy")
        params = mapping[dataset_id]
    else:
        params = getattr(protocol, "model_params", None)
    if (
        not isinstance(params, Mapping)
        or not params
        or "n_estimators" not in params
        or isinstance(params["n_estimators"], bool)
        or not isinstance(params["n_estimators"], int)
        or params["n_estimators"] < 1
    ):
        raise ValueError("protocol model parameters are invalid")
    return dict(params)


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return value


def campaign_evidence_payload(protocol: object) -> dict[str, object]:
    """Return scientific settings shared by compatible seed shards."""
    fields = (
        "protocol_version",
        "datasets",
        "backend",
        "execution",
        "model_params_by_dataset",
        "model_policy_digest",
        "alpha_default",
        "alpha_grid",
        "estimator_methods",
        "estimator_budgets",
        "estimator_budgets_by_dataset",
        "exact_max_features",
        "removal",
        "nsga2",
        "ablations",
        "stage_scopes",
    )
    missing = tuple(field for field in fields if not hasattr(protocol, field))
    if missing:
        raise ValueError(
            "protocol is missing campaign evidence fields: " + ", ".join(missing)
        )
    return {field: _json_safe(getattr(protocol, field)) for field in fields}


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def campaign_evidence_digest(protocol: object) -> str:
    """Return a seed-shard-independent scientific campaign digest."""
    return _canonical_digest(campaign_evidence_payload(protocol))


__all__ = [
    "CalibrationRules",
    "CandidateCalibration",
    "DatasetModelPolicy",
    "ModelPolicy",
    "campaign_evidence_digest",
    "campaign_evidence_payload",
    "load_model_policy",
    "model_params_for",
    "model_policy_to_document",
    "select_candidate_round",
    "write_model_policy",
]
