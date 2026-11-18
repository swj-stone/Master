"""Load paired 100- and 500-feature views of the frozen Madelon seed-42 split."""

import csv
import hashlib
import json
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from experiments.calibrate_icomp2026_models import _safe_partition_path, load_prepared_train_validation
from experiments.thesis_sampling_worker import SOURCE
from minppi_mofs.experiment_protocol import canonical_hash


RAW = Path(__file__).resolve().parents[2] / "data/raw/madelon.csv"
ROLES = ("train", "val", "test")


def _metadata(width, alpha):
    if width not in (100, 500):
        raise ValueError("Madelon width must be 100 or 500")
    manifest_bytes = (SOURCE / "data_manifest.csv").read_bytes()
    rows = csv.DictReader(StringIO(manifest_bytes.decode("utf-8-sig")))
    row, = (r for r in rows if r["dataset_id"] == "madelon" and r["seed"] == "42")
    protocol = yaml.safe_load((SOURCE / "resolved_protocol.yaml").read_bytes())
    spec, = (s for s in protocol["datasets"] if s["dataset_id"] == "madelon")
    if row["protocol_hash"] != canonical_hash(protocol) or row["raw_file_sha256"] != spec["raw_sha256"]:
        raise ValueError("Madelon protocol or raw identity disagrees with the manifest")
    split = json.loads((SOURCE / "preprocess/madelon/s42/split_metadata.json").read_bytes())
    top = json.loads(row["selected_features_json"])
    cols = split["variance_features"]
    if (len(top) != 100 or len(set(top)) != 100 or top != split["selected_features"]
            or len(cols) != 500 or len(set(cols)) != 500 or not set(top) <= set(cols)):
        raise ValueError("Madelon feature schema differs from the archived split")
    all_indices = []
    for role in ROLES:
        ids = split[f"{role}_indices"]
        row[f"{role}_row_count"] = int(row[f"{role}_row_count"])
        if len(ids) != row[f"{role}_row_count"] or canonical_hash(ids) != row[f"{role}_index_sha256"]:
            raise ValueError(f"Madelon {role} index mismatch")
        all_indices.extend(ids)
    if sorted(all_indices) != list(range(int(row["raw_row_count"]))):
        raise ValueError("Madelon role indices do not partition the raw rows")
    audit = dict(dataset="madelon", seed=42, width=width, source=SOURCE.name,
                 alpha=alpha, classification_margin=protocol["removal"]["classification_noninferiority_margin"],
                 protocol_hash=row["protocol_hash"], manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                 raw_sha256=row["raw_file_sha256"], raw_rows=int(row["raw_row_count"]),
                 partition_sha256={p: row[f"{p}_file_sha256"] for p in ROLES},
                 index_sha256={p: row[f"{p}_index_sha256"] for p in ROLES},
                 top100_sha256=canonical_hash(top), feature_order_sha256=canonical_hash(top if width == 100 else cols),
                 policy=dict(task=row["task"], utility=row["utility"], seed=42, backend=protocol["backend"],
                             model_params=protocol["model_params_by_dataset"]["madelon"]))
    return row, split, top, audit


def _raw_roles(row, split, roles):
    # Hash all original bytes. Only approved row bytes enter the numeric CSV parser.
    wanted = {i for role in roles for i in split[f"{role}_indices"]}
    raw_rows = {}
    digest = hashlib.sha256()
    with RAW.open("rb") as handle:
        header = handle.readline()
        digest.update(header)
        for i, line in enumerate(handle):
            digest.update(line)
            if i in wanted:
                raw_rows[i] = line
    if digest.hexdigest() != row["raw_file_sha256"] or len(raw_rows) != len(wanted):
        raise ValueError("Madelon raw source bytes or row count differ from formal data")
    result = {}
    for role in roles:
        data = BytesIO(header + b"".join(raw_rows[i] for i in split[f"{role}_indices"]))
        frame = pd.read_csv(data)
        labels = frame.pop(row["target"]).to_numpy()
        if list(frame.columns) != split["variance_features"] or not np.isin(labels, [0, 1]).all():
            raise ValueError("Madelon raw features or class encoding differ from the archive")
        result[role] = (frame, labels)
    return result


def _scaled(frame, scaler):
    return pd.DataFrame(scaler.transform(frame), columns=frame.columns)


def _check_overlap(full, labels, archived, archived_labels):
    diff = np.max(np.abs(full[archived.columns].to_numpy() - archived.to_numpy()))
    if not np.isfinite(diff) or diff > 1e-12 or not np.array_equal(labels, archived_labels):
        raise ValueError("Madelon raw reconstruction differs from the archived 100-column view")
    return float(diff)


def load_arm(width, *, alpha=.99):
    """Prepare training and validation only. Both arms use the archived row order."""
    row, split, top, audit = _metadata(width, alpha)
    train, yt, val, yv = load_prepared_train_validation(SOURCE, row)
    if width == 500:
        raw = _raw_roles(row, split, ("train", "val"))
        scaler = StandardScaler().fit(raw["train"][0])
        wide_train = _scaled(raw["train"][0], scaler)
        wide_val = _scaled(raw["val"][0], scaler)
        audit["top100_max_abs_diff"] = dict(
            train=_check_overlap(wide_train, raw["train"][1], train, yt),
            val=_check_overlap(wide_val, raw["val"][1], val, yv))
        audit["scaler_sha256"] = canonical_hash([scaler.mean_.tolist(), scaler.scale_.tolist()])
        train, val = wide_train, wide_val
    return train, yt, val, yv, top, audit


def load_heldout(width, audit):
    """Called by the worker after its model and feature decisions have been frozen."""
    row, split, top, expected = _metadata(width, audit["alpha"])
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen Madelon data identity changed before held-out evaluation")
    payload = _safe_partition_path(SOURCE, row["test_relpath"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected["partition_sha256"]["test"]:
        raise ValueError("Madelon archived test SHA-256 mismatch")
    test = pd.read_csv(BytesIO(payload))
    ye = test.pop(row["target"]).to_numpy()
    if list(test.columns) != top or len(ye) != row["test_row_count"]:
        raise ValueError("Madelon archived test schema mismatch")
    if width == 500:
        raw = _raw_roles(row, split, ("train", "test"))
        scaler = StandardScaler().fit(raw["train"][0])
        if canonical_hash([scaler.mean_.tolist(), scaler.scale_.tolist()]) != audit["scaler_sha256"]:
            raise ValueError("Madelon training-only scaler changed after freezing")
        wide_test = _scaled(raw["test"][0], scaler)
        _check_overlap(wide_test, raw["test"][1], test, ye)
        test = wide_test
    return test, ye
