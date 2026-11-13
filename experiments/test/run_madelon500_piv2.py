"""Compare selectors with PIv2 on the frozen Madelon seed-42 Top-100 or 500-column view."""

import argparse
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from experiments.run_piv0_benchmark import exclusive_run, read, sha
from experiments.run_thesis_sampling import MSK, ROOT
from experiments.test.piv2_madelon500_data import prepare_run
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from experiments.test.run_piv2_prefilter import (
    ADDITIONAL, LIMIT, METHODS, OUTPUT, SEED, SMALL_ONLY, SOURCES, run_campaign_tasks,
)
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


# Alternate the order within pairs. The full Gain scan runs last.
BACKENDS = tuple(m for m in (*METHODS, *ADDITIONAL) if m not in SMALL_ONLY | {"gain"}) + ("gain",)
AUC_BACKENDS = tuple(m for m in BACKENDS if m not in {"gain", "pfi", "l1", "elastic_net"})
SNAPSHOT = (*SOURCES, "experiments/test/piv2_conventional.py", "experiments/test/piv2_conventional_filters.py",
            "experiments/test/piv2_conventional_models.py", "experiments/test/run_madelon500_piv2.py",
            "experiments/test/piv2_madelon500_data.py", "experiments/test/thesis_madelon_alpha_data.py",
            "src/minppi_mofs/piv1.py", "src/minppi_mofs/baselines/shap_top_k.py",
            "src/minppi_mofs/experiment_protocol.py")


def prepared_input(root, width):
    if width == 500:
        source, row = prepare_run(root)
        provenance = read(source / "provenance.json")
        if len(provenance["columns"]) != 500 or len(provenance["top100"]) != 100:
            raise ValueError("Unexpected raw-width Madelon feature schema")
        return source, row, provenance["policy"], source/"provenance.json", row["protocol_hash"]
    if width != 100:
        raise ValueError("Expected the official 100 or 500 column Madelon view")
    jobs, catalog = audit_prepared_catalog()
    source = jobs["madelon", SEED]
    manifest = source/"data_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        row, = (r for r in csv.DictReader(stream) if r["dataset_id"] == "madelon" and int(r["seed"]) == SEED)
    for role in ("train", "val", "test"):
        row[f"{role}_row_count"] = int(row[f"{role}_row_count"])
    if len(set(json.loads(row["selected_features_json"]))) != 100:
        raise ValueError("Invalid official Madelon Top-100 schema")
    policy = dict(task=row["task"], seed=SEED, backend=catalog["shared"]["backend"],
                  model_params=catalog["shared"]["model_params"])
    return source, row, policy, manifest, catalog["sha256"]


def run(root, *, validation_k=False, width=500):
    started = time.monotonic()
    source, row, policy, metadata, catalog_hash = prepared_input(root, width)
    data_time = root / "input_preparation.json"
    if not data_time.exists():
        atomic_write_json(data_time, {"elapsed_s": time.monotonic()-started,
                                     "scope": "shared preparation or lookup and audit of the frozen inputs"})
    if int(row["selected_feature_count"]) != width or policy["seed"] != SEED or policy["task"] != "binary":
        raise ValueError("Unexpected Madelon view or model policy")
    methods = AUC_BACKENDS if validation_k else BACKENDS
    snapshot = SNAPSHOT
    if validation_k:
        snapshot += ("experiments/test/piv2_prefix_rankings.py", "experiments/test/piv2_auc_prefix.py")
    tasks = [("madelon", method, arm) for i, method in enumerate(methods)
             for arm in (("plain", "piv2") if i % 2 == 0 else ("piv2", "plain"))]
    protocol = dict(version=f"PIv2-Madelon-{width}", dataset="madelon", width=width, seed=SEED,
                    methods=list(methods), tasks=[list(t) for t in tasks], arm_limit_s=LIMIT,
                    hardware_backend=policy["backend"], model_params=policy["model_params"], cold_start=True,
                    catalog_sha256=catalog_hash, input_sha256=sha(metadata),
                    input_partition_sha256={part: row[f"{part}_file_sha256"] for part in ("train", "val", "test")},
                    source_sha256={name: sha(ROOT/name) for name in snapshot},
                    timing="Independent end-to-end process wall from spawn to exit, including PI, validation, "
                           "final refit, test prediction and output; shared dataset preparation itemized separately.",
                    exclusions=["exact", "mc32"])
    if validation_k:
        protocol.update(version=f"PIv2-Madelon-{width}-validation-k", k_selection="validation_auc_all_prefixes",
                        candidate_sizes="all integers 1..backend input width inclusive",
                        tie_rule="smallest k within 1e-14 of the maximum validation AUC",
                        cache="no predictions or fitted models shared across arms or runs",
                        exclusions=["gain", "pfi", "l1", "elastic_net", "exact", "mc32"])
    else:
        protocol.update(original_width_for_fixed_k=width, fixed_k=23 if width == 500 else 10)
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("Cannot resume with different source code, settings, or data")
    if not protocol_path.exists():
        atomic_write_json(protocol_path, protocol)
    for name, digest in protocol["source_sha256"].items():
        target = root/"source"/name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        if sha(target) != digest:
            raise ValueError(f"Invalid source snapshot for {name}")
    if validation_k:
        row = dict(row, k_selection=protocol["k_selection"])
    return run_campaign_tasks(root, tasks, tasks, {("madelon",SEED):source}, {"madelon":row},
                              protocol, canonical_hash(protocol))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="New or resumable output directory")
    parser.add_argument("--width", type=int, choices=(100, 500), default=500)
    parser.add_argument("--validation-k", action="store_true",
                        help="Choose k by whole-validation AUC across every nonempty prefix for the 11 ranking methods")
    args = parser.parse_args()
    label = f"PIv2-madelon{args.width}" + ("-validation-k" if args.validation_k else "")
    root = (args.run or OUTPUT/datetime.now(MSK).strftime(label+"-seed42-%Y%m%d-%H%M%S-%f")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        raise SystemExit(run(root, validation_k=args.validation_k, width=args.width))
