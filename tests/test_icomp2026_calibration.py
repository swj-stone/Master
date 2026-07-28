from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.calibrate_icomp2026_models import (
    MarginalPair,
    _evaluate_checkpoints,
    calibration_metrics,
    load_prepared_train_validation,
    make_marginal_panel,
    panel_coalitions,
)
from minppi_mofs.experiment_protocol import DatasetSpec


def test_panel_has_two_deterministic_scales_per_feature() -> None:
    features = ("a", "b", "c", "d", "e")

    panel = make_marginal_panel(features, seed=42)

    assert panel == make_marginal_panel(features, seed=42)
    assert len(panel) == 2 * len(features)
    for feature in features:
        rows = [row for row in panel if row.feature == feature]
        assert tuple(row.fraction for row in rows) == (0.25, 0.75)
        assert tuple(len(row.without) for row in rows) == (1, 3)
        assert all(feature not in row.without for row in rows)
        assert all(
            row.with_feature
            == tuple(item for item in features if item in {*row.without, feature})
            for row in rows
        )


def test_panel_coalitions_are_reusable_and_include_full_coalition() -> None:
    features = ("a", "b", "c", "d", "e")
    panel = make_marginal_panel(features, seed=7)

    coalitions = panel_coalitions(features, panel)

    assert coalitions == panel_coalitions(features, panel)
    assert features in coalitions
    assert len(coalitions) == len(set(coalitions))
    assert all(
        row.without in coalitions and row.with_feature in coalitions
        for row in panel
    )


def test_calibration_metrics_detect_harmful_recall_and_false_harmful() -> None:
    full = ("a", "b", "c")
    panel = (
        MarginalPair("a", 0.25, ("b",), ("a", "b")),
        MarginalPair("b", 0.25, ("c",), ("b", "c")),
    )
    reference = {
        full: 1.0,
        ("b",): 0.98,
        ("a", "b"): 0.90,
        ("c",): 0.70,
        ("b", "c"): 0.80,
    }
    identical = dict(reference)
    degraded = {
        full: 0.99,
        ("b",): 0.80,
        ("a", "b"): 0.90,
        ("c",): 0.97,
        ("b", "c"): 0.80,
    }

    perfect = calibration_metrics(
        reference, identical, panel=panel, full=full, alpha=0.95
    )
    changed = calibration_metrics(
        reference, degraded, panel=panel, full=full, alpha=0.95
    )

    assert perfect["full_utility_loss"] == 0.0
    assert perfect["spearman"] == 1.0
    assert perfect["winning_agreement"] == 1.0
    assert perfect["marginal_sign_agreement"] == 1.0
    assert perfect["harmful_recall"] == 1.0
    assert perfect["false_harmful_rate"] == 0.0
    assert changed["full_utility_loss"] == 0.01
    assert changed["harmful_recall"] == 0.0
    assert changed["false_harmful_rate"] == 1.0


def test_prepared_loader_never_opens_test_partition(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "run"
    split_root = root / "preprocess" / "toy" / "s42"
    split_root.mkdir(parents=True)
    train = b"a,b,target\n1,2,0\n3,4,1\n"
    validation = b"a,b,target\n5,6,0\n"
    (split_root / "train.csv").write_bytes(train)
    (split_root / "val.csv").write_bytes(validation)
    (split_root / "test.csv").write_bytes(b"must-not-be-opened")
    row = {
        "dataset_id": "toy",
        "seed": 42,
        "task": "binary",
        "target": "target",
        "selected_features_json": json.dumps(
            ["a", "b"], separators=(",", ":")
        ),
        "train_relpath": "preprocess/toy/s42/train.csv",
        "val_relpath": "preprocess/toy/s42/val.csv",
        "test_relpath": "preprocess/toy/s42/test.csv",
        "train_file_sha256": hashlib.sha256(train).hexdigest(),
        "val_file_sha256": hashlib.sha256(validation).hexdigest(),
        "train_row_count": 2,
        "val_row_count": 1,
    }
    opened: list[str] = []
    real_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        opened.append(path.name)
        if path.name == "test.csv":
            raise AssertionError("calibration opened the sealed test partition")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    X_train, y_train, X_val, y_val = load_prepared_train_validation(root, row)

    assert list(X_train.columns) == ["a", "b"]
    assert list(X_val.columns) == ["a", "b"]
    assert y_train.tolist() == [0, 1]
    assert y_val.tolist() == [0]
    assert opened == ["train.csv", "val.csv"]
    assert isinstance(X_train, pd.DataFrame)


def test_checkpoint_calibration_fits_each_nonempty_coalition_once() -> None:
    X_train = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    X_val = pd.DataFrame({"a": [0.5], "b": [0.5]})
    y_train = pd.Series([0, 1]).to_numpy()
    y_val = pd.Series([0]).to_numpy()
    coalitions = (("a", "b"), ("a",), ())
    calls: list[tuple[str, ...]] = []

    def predictor(X_fit, y_fit, X_eval, *, checkpoints, **kwargs):
        calls.append(tuple(X_fit.columns))
        return {
            rounds: pd.DataFrame([[0.6, 0.4]]).to_numpy()
            for rounds in checkpoints
        }

    dataset = DatasetSpec(
        dataset_id="toy",
        display_name="Toy",
        raw_path=Path("toy.csv"),
        target="target",
        task="binary",
        utility="accuracy",
        max_features=None,
        variance_threshold=0.0,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
    )
    values, _, physical_fits = _evaluate_checkpoints(
        X_train,
        y_train,
        X_val,
        y_val,
        coalitions=coalitions,
        dataset=dataset,
        base_model_params={
            "n_estimators": 100,
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_jobs": 1,
        },
        rounds=(10, 20, 40, 100),
        backend="cpu",
        seed=42,
        checkpoint_predictor=predictor,
    )

    assert calls == [("a", "b"), ("a",)]
    assert physical_fits == 2
    assert tuple(values) == (10, 20, 40, 100)
    assert all(tuple(mapping) == coalitions for mapping in values.values())
