"""Materialize the seed-42 training-only standardized Madelon view before the Top-100 filter."""

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from experiments.test.thesis_madelon_alpha_data import (
    ROLES, SOURCE, _check_overlap, _metadata, _raw_roles, _scaled,
)
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


def _write_view(path, frame, labels, target):
    """Publish one deterministic CSV; resume only from identical completed inputs."""
    frame = frame.assign(**{target: labels})
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing 500-column input differs from the reconstruction: {path}")
    else:
        temp = path.with_suffix(".csv.tmp")
        temp.write_bytes(data)
        os.replace(temp, path)
    return digest, pd.read_csv(BytesIO(data))


def prepare_run(root):
    """Return (run/input, manifest row) in the format used by the paired selector worker.

    This is one-time data preparation. Held-out rows are transformed and checked
    for identity here. Neither model fitting nor feature selection takes place.
    """
    source = Path(root).resolve() / "input"
    source.mkdir(parents=True, exist_ok=True)
    reference, split, top, archived = _metadata(500, None)
    raw = _raw_roles(reference, split, ROLES)
    columns = split["variance_features"]
    target = reference["target"]
    scaler = StandardScaler().fit(raw["train"][0])
    scaler_hash = canonical_hash([scaler.mean_.tolist(), scaler.scale_.tolist()])
    identity = dict(dataset="madelon", seed=42, width=500, target=target,
                    transform="StandardScaler fitted on frozen training rows; retain every original feature",
                    raw_sha256=archived["raw_sha256"], index_sha256=archived["index_sha256"],
                    columns_sha256=canonical_hash(columns), scaler_sha256=scaler_hash,
                    reference_protocol_hash=reference["protocol_hash"])
    row = dict(dataset_id="madelon", seed=42, task=reference["task"], utility=reference["utility"],
               target=target, protocol_hash=canonical_hash(identity), raw_file_sha256=archived["raw_sha256"],
               selected_feature_count=len(columns), selected_features_json=json.dumps(columns),
               raw_row_count=archived["raw_rows"], raw_feature_count=len(columns))
    max_diff = {}
    for role in ROLES:
        x, labels = raw[role]
        scaled = _scaled(x, scaler)
        path = source / f"{role}.csv"
        digest, reloaded = _write_view(path, scaled, labels, target)
        raw_reference = (SOURCE / reference[f"{role}_relpath"]).read_bytes()
        if hashlib.sha256(raw_reference).hexdigest() != archived["partition_sha256"][role]:
            raise ValueError(f"Official Top-100 {role} file fingerprint mismatch")
        ref = pd.read_csv(BytesIO(raw_reference))
        y100, y500 = ref.pop(target).to_numpy(), reloaded.pop(target).to_numpy()
        if list(ref.columns) != top or list(reloaded.columns) != columns or len(ref) != len(x):
            raise ValueError(f"Invalid {role} schema or partition size")
        if not np.array_equal(y500, labels):
            raise ValueError(f"Serialized {role} labels differ from the source rows")
        np.testing.assert_allclose(reloaded, scaled, rtol=0, atol=1e-12)
        max_diff[role] = _check_overlap(reloaded, y500, ref, y100)
        row.update({f"{role}_relpath":path.name, f"{role}_file_sha256":digest,
                    f"{role}_row_count":len(x), f"{role}_index_sha256":archived["index_sha256"][role]})

    provenance = dict(dataset="madelon", seed=42, width=len(columns), columns=columns, top100=top,
                      raw_sha256=archived["raw_sha256"], index_sha256=archived["index_sha256"],
                      reference_source=SOURCE.name, reference_protocol_hash=reference["protocol_hash"],
                      reference_manifest_sha256=archived["manifest_sha256"],
                      reference_partition_sha256=archived["partition_sha256"],
                      top100_max_abs_diff=max_diff, scaler_sha256=scaler_hash, input_identity=identity,
                      row=row, policy=archived["policy"])
    metadata = source / "provenance.json"
    if metadata.exists():
        if json.loads(metadata.read_bytes()) != provenance:
            raise ValueError("Existing Madelon data provenance disagrees with the frozen source")
    else:
        atomic_write_json(metadata, provenance)
    return source, row
