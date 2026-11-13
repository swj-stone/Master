"""Full-width, train-standardized inputs on the archived six seed-42 splits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from experiments.icomp2026_stages import _coerce_target
from experiments.thesis_sampling_worker import DATASETS, SOURCE
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


ROLES = ("train", "val", "test")
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT / "results/test/thesis_sampling/PIv3-six-raw-seed42-input"


def _sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _authority(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown thesis dataset {dataset!r}")
    manifest = SOURCE / "data_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        row, = (r for r in csv.DictReader(f) if r["dataset_id"] == dataset and r["seed"] == "42")
    protocol = yaml.safe_load((SOURCE / "resolved_protocol.yaml").read_bytes())
    spec, = (s for s in protocol["datasets"] if s["dataset_id"] == dataset)
    if row["protocol_hash"] != canonical_hash(protocol) or row["raw_file_sha256"] != spec["raw_sha256"]:
        raise ValueError(f"{dataset} frozen raw/protocol identity mismatch")
    metadata_path = SOURCE / f"preprocess/{dataset}/s42/split_metadata.json"
    split = json.loads(metadata_path.read_bytes())
    for key in ("selected_features", "variance_features", "scaler_mean", "scaler_scale", "target_encoding"):
        if split[key] != json.loads(row["transform_metadata_json"])[key]:
            raise ValueError(f"{dataset} frozen transform mismatch for {key}")
    indices = np.concatenate([split[f"{role}_indices"] for role in ROLES])
    if not np.array_equal(np.sort(indices), np.arange(int(row["raw_row_count"]))):
        raise ValueError(f"{dataset} frozen indices do not partition the raw rows")
    for role in ROLES:
        ids = split[f"{role}_indices"]
        if len(ids) != int(row[f"{role}_row_count"]) or canonical_hash(ids) != row[f"{role}_index_sha256"]:
            raise ValueError(f"{dataset} frozen {role} index mismatch")
    policy = dict(task=row["task"], utility=row["utility"], seed=42,
                  backend=protocol["backend"], model_params=protocol["model_params_by_dataset"][dataset])
    meta = dict(dataset=dataset, seed=42, source=SOURCE.name, task=row["task"], utility=row["utility"],
                policy=policy, target=row["target"], raw_sha256=row["raw_file_sha256"],
                raw_rows=int(row["raw_row_count"]), width=int(row["raw_feature_count"]),
                protocol_hash=row["protocol_hash"], manifest_sha256=_sha(manifest),
                source_split_metadata_sha256=_sha(metadata_path),
                index_sha256={p: row[f"{p}_index_sha256"] for p in ROLES},
                partition_sha256={p: row[f"{p}_file_sha256"] for p in ROLES})
    return row, split, meta


def _verify_archived_role(role, row, selected, offsets, values, labels):
    path = SOURCE / row[f"{role}_relpath"]
    if _sha(path) != row[f"{role}_file_sha256"]:
        raise ValueError(f"{path} differs from frozen SHA-256")
    checked = 0
    max_diff = 0.0
    for chunk in pd.read_csv(path, chunksize=4096):
        archived_y = chunk.pop(row["target"]).to_numpy()
        stop = checked + len(chunk)
        if list(chunk.columns) != selected or not np.array_equal(archived_y, labels[checked:stop]):
            raise ValueError(f"{path} feature schema or labels differ from reconstruction")
        actual = values[checked:stop, offsets]
        expected = chunk.to_numpy()
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{path} standardized values differ; largest error {np.max(np.abs(actual - expected))}")
        max_diff = max(max_diff, float(np.max(np.abs(actual - expected))))
        checked = stop
    if checked != len(values):
        raise ValueError(f"{path} row count differs")
    return max_diff


def _build(directory, raw, row, split, meta):
    frame = pd.read_csv(raw)
    labels, encoding = _coerce_target(frame.pop(meta["target"]), meta["task"])
    features = list(frame.columns)
    selected = split["selected_features"]
    if (encoding != split["target_encoding"] or frame.shape != (meta["raw_rows"], meta["width"])
            or len(set(features)) != len(features) or selected != json.loads(row["selected_features_json"])):
        raise ValueError(f"{raw} raw shape, targets or features differ from the formal source")
    offsets = [features.index(c) for c in selected]
    values = frame.to_numpy(dtype=np.float64, copy=False)
    del frame
    if not np.isfinite(values).all():
        raise ValueError(f"{raw} contains non-finite predictors")
    scaler = StandardScaler(copy=False)
    parts = {}
    hashes = {}
    for role in ROLES:
        ids = np.asarray(split[f"{role}_indices"], dtype=np.int64)
        role_y = labels[ids]
        # Fortran order matches the archived column-wise sums in StandardScaler.
        role_x = np.asfortranarray(values[ids])
        if role == "train":
            scaler.fit(role_x)
        role_x = scaler.transform(role_x)
        error = _verify_archived_role(role, row, selected, offsets, role_x, role_y)
        for name, array in ((role, role_x), (f"{role}_y", role_y), (f"{role}_indices", ids)):
            path = directory / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            hashes[path.name] = _sha(path)
        parts[role] = dict(rows=len(ids), archived_columns_checked=len(offsets), archived_max_abs_diff=error)
        del role_x
    meta.update(features=features, feature_order_sha256=canonical_hash(features), partitions=parts,
                scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                target_encoding=encoding, selected_in_archive=selected, dtype="float64", sha256=hashes)
    atomic_write_json(directory / "metadata.json", meta)


def prepare(root, dataset):
    """Build or verify the immutable per-dataset cache; return its provenance."""
    row, split, expected = _authority(dataset)
    raw = PROJECT / "data" / "raw" / f"{dataset}.csv"
    if _sha(raw) != expected["raw_sha256"]:
        raise ValueError(f"{raw} digest differs from the official raw source")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / dataset
    if not folder.exists():
        with TemporaryDirectory(prefix=f".{dataset}-", dir=root) as temp:
            build = Path(temp) / dataset
            build.mkdir()
            _build(build, raw, row, split, dict(expected))
            try:
                build.rename(folder)
            except FileExistsError:
                pass  # Another process prepared the same input concurrently.
    meta = json.loads((folder / "metadata.json").read_bytes())
    if (any(meta.get(k) != v for k, v in expected.items())
            or canonical_hash(meta["features"]) != meta["feature_order_sha256"]
            or len(meta["features"]) != meta["width"]):
        raise ValueError(f"{folder} provenance differs from archived source")
    for role in ROLES:
        for name in (role, f"{role}_y", f"{role}_indices"):
            path = folder / f"{name}.npy"
            if _sha(path) != meta["sha256"][path.name]:
                raise ValueError(f"{path} differs from verified binary cache")
    return meta


def load_split(root, dataset, part):
    """Memory-map just the named role; callers freeze selection before loading test."""
    if dataset not in DATASETS or part not in ROLES:
        raise ValueError(f"Unknown dataset/partition: {dataset}/{part}")
    folder = Path(root) / dataset
    meta = json.loads((folder / "metadata.json").read_bytes())
    x = np.load(folder / f"{part}.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(folder / f"{part}_y.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (meta["partitions"][part]["rows"], meta["width"]) or y.shape != (len(x),):
        raise ValueError(f"{dataset}/{part} has unexpected array dimensions")
    return pd.DataFrame(x, columns=meta["features"], copy=False), y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for name in args.datasets:
        metadata = prepare(args.root, name)
        print(f"{name} {metadata['raw_rows']} x {metadata['width']} verified", flush=True)
