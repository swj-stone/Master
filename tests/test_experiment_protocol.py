"""Tests for the frozen ICOMP 2026 experiment protocol."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from minppi_mofs.experiment_protocol import (
    canonical_hash,
    load_resolved_protocol,
)


GLOBAL_CONFIG = {
    "protocol_version": "1.0",
    "datasets": ["spambase", "abalone"],
    "seeds": [42, 123],
    "backend": "cpu",
    "model": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_jobs": 1,
    },
    "alpha": {"default": 0.95, "grid": [0.85, 0.90, 0.95, 0.99]},
    "estimator": {
        "methods": ["exact", "random_mc", "mi_correlation_guided", "hybrid"],
        "budgets": [10, 25],
        "exact_max_features": 16,
    },
    "removal": {
        "controls": ["full", "minppi", "random_matched"],
        "random_repeats": 30,
    },
    "nsga2": {
        "variants": ["full", "minppi_prefilter"],
        "population": 40,
        "evaluations": 800,
        "ref_point": [1.1, 1.1],
    },
    "ablations": {
        "utility": ["accuracy", "ovr_auc"],
        "guidance": ["random", "mi", "correlation", "mi_correlation"],
        "dispatcher": ["exact", "random_mc", "mi_correlation_guided", "hybrid"],
    },
    "smoke": {
        "datasets": ["spambase", "abalone"],
        "seeds": [42],
        "estimator_budgets": [2],
        "random_repeats": 2,
        "nsga2_population": 4,
        "nsga2_evaluations": 8,
    },
}

DATASET_CONFIGS = {
    "spambase": {
        "dataset": {
            "id": "spambase",
            "display_name": "Spambase",
            "raw_path": "data/raw/spambase.csv",
            "target": "is_spam",
            "task": "binary",
            "utility": "roc_auc",
        },
        "preprocessing": {"max_features": 100, "variance_threshold": 0.01},
        "split": {"train_frac": 0.64, "val_frac": 0.16, "test_frac": 0.20},
    },
    "abalone": {
        "dataset": {
            "id": "abalone",
            "display_name": "Abalone",
            "raw_path": "data/raw/abalone.csv",
            "target": "Rings",
            "task": "regression",
            "utility": "r2",
        },
        "preprocessing": {"max_features": None, "variance_threshold": 0.0},
        "split": {"train_frac": 0.64, "val_frac": 0.16, "test_frac": 0.20},
    },
}


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_protocol(
    config_dir: Path,
    *,
    global_config: dict | None = None,
    dataset_configs: dict[str, dict] | None = None,
) -> Path:
    config_dir.mkdir()
    _write_yaml(
        config_dir / "icomp2026_pipeline.yaml",
        deepcopy(global_config if global_config is not None else GLOBAL_CONFIG),
    )
    configs = dataset_configs if dataset_configs is not None else DATASET_CONFIGS
    for dataset_id, config in configs.items():
        _write_yaml(config_dir / f"{dataset_id}.yaml", deepcopy(config))
    return config_dir


def test_load_resolved_protocol_merges_global_and_dataset_configs(
    tmp_path: Path,
) -> None:
    config_dir = _write_protocol(tmp_path / "configs")

    protocol = load_resolved_protocol(
        config_dir,
        seeds=(42,),
        backend="cpu",
        smoke=True,
    )

    assert protocol.protocol_version == "1.0"
    assert protocol.seeds == (42,)
    assert protocol.backend == "cpu"
    assert protocol.run_kind == "smoke"
    assert tuple(d.dataset_id for d in protocol.datasets) == (
        "spambase",
        "abalone",
    )
    assert protocol.datasets[0].task == "binary"
    assert protocol.datasets[0].utility == "roc_auc"
    assert protocol.datasets[1].task == "regression"
    assert protocol.datasets[1].utility == "r2"
    assert protocol.estimator_budgets == (2,)
    assert protocol.removal["random_repeats"] == 2
    assert protocol.nsga2["population"] == 4
    assert protocol.nsga2["evaluations"] == 8


def test_canonical_hash_is_independent_of_mapping_order() -> None:
    left = {"b": [2, 3], "a": {"y": 1, "x": 0}}
    right = {"a": {"x": 0, "y": 1}, "b": [2, 3]}

    assert canonical_hash(left) == canonical_hash(right)


def test_repository_protocol_configs_are_valid() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    formal = load_resolved_protocol(
        config_dir,
        seeds=None,
        backend=None,
        smoke=False,
    )
    smoke = load_resolved_protocol(
        config_dir,
        seeds=(42,),
        backend="cuda",
        smoke=True,
    )

    assert len(formal.datasets) == 7
    assert formal.seeds == (42, 123, 456, 789, 1024)
    assert tuple(dataset.dataset_id for dataset in smoke.datasets) == (
        "spambase",
        "abalone",
    )
    assert smoke.backend == "cuda"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda g, _d: g.update({"unknown": True}), "unknown"),
        (lambda g, _d: g["alpha"].update({"default": 1.2}), "alpha"),
        (lambda g, _d: g["estimator"].update({"budgets": [10, 0]}), "budget"),
        (lambda g, _d: g.update({"seeds": [42, 42]}), "seed"),
        (
            lambda g, _d: g["nsga2"].update({"ref_point": [1.0, 1.0]}),
            "ref_point",
        ),
        (
            lambda g, _d: g.update({"datasets": ["spambase", "spambase"]}),
            "duplicate",
        ),
        (
            lambda _g, d: d["spambase"]["dataset"].pop("target"),
            "target",
        ),
        (
            lambda _g, d: d["spambase"]["dataset"].update(
                {"task": "regression", "utility": "roc_auc"}
            ),
            "utility",
        ),
    ],
)
def test_invalid_protocol_is_rejected(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    global_config = deepcopy(GLOBAL_CONFIG)
    dataset_configs = deepcopy(DATASET_CONFIGS)
    mutator(global_config, dataset_configs)
    config_dir = _write_protocol(
        tmp_path / "configs",
        global_config=global_config,
        dataset_configs=dataset_configs,
    )

    with pytest.raises(ValueError, match=message):
        load_resolved_protocol(
            config_dir,
            seeds=None,
            backend=None,
            smoke=False,
        )
