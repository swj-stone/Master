"""Replay full-feature validation and compare MAE with both frozen selectors."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from minppi_mofs.modeling import fit_predict
from minppi_mofs.utility import compute_utility


def validation_mae(y, prediction, task):
    if task == "regression":
        if y.shape != prediction.shape:
            raise ValueError("Regression target and prediction dimensions differ")
        return float(np.abs(y - prediction).mean())
    if (prediction.ndim != 2 or len(prediction) != len(y)
            or not np.allclose(prediction.sum(axis=1), 1., atol=1e-5)):
        raise ValueError("Expected class probabilities in the original label order")
    targets = np.eye(prediction.shape[1])[y.astype(int)]
    return float(np.abs(targets - prediction).mean())


def run(campaign, supplement):
    jobs, catalog = audit_prepared_catalog()
    protocol = json.loads((campaign / "protocol.json").read_bytes())
    if catalog["sha256"] != protocol["catalog_hash"]:
        raise ValueError("Dataset catalog differs from the formal run")
    prior = pd.read_csv(supplement / "arms.csv").set_index(["dataset", "seed", "method"], verify_integrity=True)
    manifests = {}
    for shard in set(jobs.values()):
        with (shard / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for part in ("train", "val", "test"):
                    row[part + "_row_count"] = int(row[part + "_row_count"])
                manifests[row["dataset_id"], int(row["seed"])] = row
    out = campaign / "full_validation"
    out.mkdir(exist_ok=True)
    records = []
    for (dataset, seed), source in jobs.items():
        row = manifests[dataset, seed]
        train, yt, val, y = load_prepared_train_validation(source, row)
        base = campaign / f"seed-{seed}" / dataset
        rank = json.loads((base / "direct_gain" / "ranking.json").read_bytes())
        full_piv1 = json.loads((base / "PIv1" / "frozen.json").read_bytes())["full_validation"][0]
        expected = float(rank["full_validation"][0])
        policy = dict(seed=seed, task=row["task"], backend=protocol["backend"], model_params=protocol["model_params"])
        if (rank["policy"] != policy or rank["feature_order"] != list(train.columns)
                or full_piv1 != expected or rank["partition_sha256"]["val"] != row["val_file_sha256"]):
            raise ValueError(f"Full-feature source mismatch for {dataset}/{seed}")
        if row["task"] != "regression" and not np.array_equal(np.unique(yt), np.arange(len(np.unique(yt)))):
            raise ValueError("Class labels must match zero-based probability column order")
        pred = fit_predict(train, yt, val, **policy)
        primary = compute_utility(y, pred, metric=row["utility"])
        if not np.isclose(primary, expected, rtol=0, atol=1e-12):
            raise ValueError(f"Replayed validation score differs for {dataset}/{seed}")
        entry = dict(dataset=dataset, seed=seed, width=len(train.columns), metric=row["utility"],
                     full_validation_primary=primary, full_validation_mae=validation_mae(y, pred, row["task"]))
        for method, col in (("PIv1", "piv1"), ("direct_gain", "gain")):
            old = prior.loc[dataset, seed, method]
            archive = base / method
            if old.frozen_hash != json.loads((archive / "result.json").read_bytes())["frozen_hash"]:
                raise ValueError("Wrong selector reference")
            with np.load(supplement / old.validation_predictions, allow_pickle=False) as saved:
                assert np.array_equal(saved["y"], y)
                subset = saved["selected"]
                assert np.isclose(compute_utility(y, subset, metric=row["utility"]), old.validation_primary,
                                  rtol=0, atol=1e-12)
                if len(saved["columns"]) == len(train.columns):
                    assert np.array_equal(pred, subset)
                entry[col + "_validation_mae"] = validation_mae(y, subset, row["task"])
        np.savez_compressed(out / f"{dataset}-{seed}.npz", y=y, prediction=pred, columns=np.asarray(train.columns))
        records.append(entry)
    data = pd.DataFrame(records)
    assert len(data) == 30 and data.groupby("dataset").size().eq(5).all()
    data.to_csv(out / "by_seed.csv", index=False, encoding="utf-8-sig")
    print(f"Verified full-model validation scores for {len(data)} paired splits: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("accuracy_supplement", type=Path)
    args = parser.parse_args()
    run(args.campaign, args.accuracy_supplement)
