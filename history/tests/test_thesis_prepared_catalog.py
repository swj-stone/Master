import csv
import hashlib
import json
import shutil

import pytest
import yaml

from experiments.thesis_prepared_catalog import FORMAL_DATASETS, FORMAL_SEEDS, audit_prepared_catalog, discover_prepared_jobs
from minppi_mofs.experiment_protocol import canonical_hash


def write_fixture(root):
    """Create two frozen shards with 30 small binary dataset/seed fixtures."""
    campaign = root / "icomp2026-formal-v1"
    campaign.mkdir(parents=True)
    shards = [("shard-a", FORMAL_SEEDS[:2]), ("shard-b", FORMAL_SEEDS[2:])]
    (campaign / "campaign_plan.json").write_text(json.dumps({
        "campaign_id": campaign.name, "run_ids": [name for name, _ in shards],
        "seed_shards": [list(seeds) for _, seeds in shards],
    }), encoding="utf-8")
    for name, seeds in shards:
        source = root / name
        source.mkdir()
        protocol = {"run_kind": "formal", "backend": "cpu", "seeds": list(seeds),
                    "datasets": [{"dataset_id": ds, "task": "binary", "utility": "roc_auc", "target": "y",
                                  "train_frac": .64, "val_frac": .16, "test_frac": .2, "max_features": 100,
                                  "variance_threshold": 0., "display_name": ds}
                                 for ds in (*FORMAL_DATASETS, "crimedata")],
                    "model_params_by_dataset": {ds: {"n_estimators": 4, "max_depth": 2, "learning_rate": .1, "n_jobs": 1}
                                                for ds in (*FORMAL_DATASETS, "crimedata")},
                    "alpha_default": .95, "removal": {"regression_noninferiority_margin": .01,
                                                      "classification_noninferiority_margin": .005}}
        protocol["model_params"] = protocol["model_params_by_dataset"]["abalone"].copy()
        (source / "resolved_protocol.yaml").write_text(yaml.safe_dump(protocol), encoding="utf-8")
        digest = canonical_hash(protocol)
        (source / "metadata_complete.json").write_text(json.dumps({"protocol_hash": digest}))
        rows = []
        for dataset, seed in ((d, s) for d in FORMAL_DATASETS for s in seeds):
            row = {"protocol_hash": digest, "dataset_id": dataset, "seed": seed, "task": "binary",
                   "utility": "roc_auc", "target": "y", "selected_features_json": '["x","z"]',
                   "selected_feature_count": 2}
            for part in ("train", "val", "test"):
                relpath = f"preprocess/{dataset}/s{seed}/{part}.csv"
                p = source / relpath
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x,z,y\n" + "".join(f"{i},{i%3},{i%2}\n" for i in range(32)),
                             encoding="utf-8")
                row.update({f"{part}_relpath": relpath, f"{part}_row_count": 32,
                            f"{part}_file_sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
            rows.append(row)
        with (source/"data_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return root


def test_catalog_uses_only_campaign_shards(tmp_path):
    root = write_fixture(tmp_path)
    shutil.copytree(root/"shard-a", root/"bootstrap-or-smoke")
    # The publication exclusion also applies to historical manifest rows whose
    # obsolete audit-only files are no longer installed.
    path = root/"shard-a"/"data_manifest.csv"
    with path.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    row.update(dataset_id="crimedata", test_relpath="retired-crime-partition.csv")
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=row).writerow(row)
    jobs = discover_prepared_jobs(root)
    assert list(jobs) == [(ds, seed) for seed in FORMAL_SEEDS for ds in FORMAL_DATASETS]
    assert jobs["abalone", 123] == root/"shard-a" and jobs["isolet", 1024] == root/"shard-b"


def test_audit_includes_30_job_witnesses_and_reproducible_effective_settings(tmp_path):
    root = write_fixture(tmp_path)
    jobs, audit = audit_prepared_catalog(root)
    assert audit["status"] == "PASS" and audit["shared"]["checkpoints"] == [4, 6, 8]
    assert audit["shared"]["model_params"]["max_depth"] == 2
    assert len(jobs) == len(audit["jobs"]) == 30
    assert audit["sha256"] == canonical_hash({k:v for k,v in audit.items() if k != "sha256"})
    assert audit["compared_protocol_keys"] and "seeds" not in audit["compared_protocol_keys"]


@pytest.mark.parametrize("field", ["model", "train_frac"])
def test_all_shards_with_same_dataset_specific_hyperparameter_drift_are_rejected(tmp_path, field):
    root = write_fixture(tmp_path)
    for shard in root.glob("shard-*"):
        p = shard/"resolved_protocol.yaml"
        proto = yaml.safe_load(p.read_text())
        if field == "model":
            proto["model_params_by_dataset"]["abalone"]["n_estimators"] = 7
        else:
            next(d for d in proto["datasets"] if d["dataset_id"] == "abalone")["train_frac"] = .33
        digest = canonical_hash(proto)
        p.write_text(yaml.safe_dump(proto))
        (shard/"metadata_complete.json").write_text(json.dumps({"protocol_hash": digest}))
        manifest = shard/"data_manifest.csv"
        with manifest.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["protocol_hash"] = digest
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError):
        audit_prepared_catalog(root)


@pytest.mark.parametrize("failure", ["duplicate", "missing_seed", "partition", "traversal", "model_drift", "unapproved"])
def test_catalog_fails_before_training_for_invalid_source(tmp_path, failure):
    root = write_fixture(tmp_path)
    shard = root/"shard-b"
    manifest = shard/"data_manifest.csv"
    with manifest.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if failure == "duplicate":
        rows.append(rows[0].copy())
    elif failure == "missing_seed":
        rows.pop()
    elif failure == "partition":
        (shard/rows[0]["val_relpath"]).unlink()
    elif failure == "traversal":
        rows[0]["train_relpath"] = "../shard-a/preprocess/abalone/s42/train.csv"
    elif failure == "unapproved":
        rows[0]["dataset_id"] = "unknown"
    else:
        protocol = yaml.safe_load((shard/"resolved_protocol.yaml").read_text())
        protocol["model_params_by_dataset"]["abalone"]["max_depth"] = 6
        digest = canonical_hash(protocol)
        (shard/"metadata_complete.json").write_text(json.dumps({"protocol_hash": digest}))
        (shard/"resolved_protocol.yaml").write_text(yaml.safe_dump(protocol))
        for row in rows:
            row["protocol_hash"] = digest
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        discover_prepared_jobs(root)
