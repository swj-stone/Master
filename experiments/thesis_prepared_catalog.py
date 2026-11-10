"""Resolve the official five-seed thesis inputs from the frozen local campaign."""

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

import yaml

from experiments.run_thesis_sampling import DATASETS as FORMAL_DATASETS, ROOT
from minppi_mofs.experiment_protocol import canonical_hash


FORMAL_ROOT = ROOT / "results/icomp2026_formal"
FORMAL_SEEDS = (42, 123, 456, 789, 1024)
AUDIT_ONLY_DATASETS = frozenset({"crimedata"})
CAMPAIGN = "icomp2026-formal-v1"


def _check_row(source, row, spec, protocol_hash):
    dataset, seed = row["dataset_id"], int(row["seed"])
    if row["protocol_hash"] != protocol_hash or any(row[k] != spec[k] for k in ("task", "utility", "target")):
        raise ValueError(f"{source.name}/{dataset}/s{seed}: manifest and frozen protocol disagree")
    features = json.loads(row["selected_features_json"])
    if (not isinstance(features, list) or not features or not all(isinstance(f, str) and f for f in features)
            or len(set(features)) != len(features) or spec["target"] in features
            or int(row["selected_feature_count"]) != len(features)):
        raise ValueError(f"{dataset}/s{seed}: invalid frozen feature set")
    for part in ("train", "val", "test"):
        relative = Path(row[f"{part}_relpath"])
        expected = Path("preprocess", dataset, f"s{seed}", f"{part}.csv")
        path = (source / relative).resolve()
        if (relative != expected or source not in path.parents
                or int(row[f"{part}_row_count"]) < 1
                or re.fullmatch(r"[0-9a-f]{64}", row[f"{part}_file_sha256"]) is None):
            raise ValueError(f"{dataset}/s{seed}: invalid frozen {part} partition")
        # Each worker hashes train/val on load and test at authorized final evaluation.
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty prepared partition: {path}")


def audit_prepared_catalog(formal_root=FORMAL_ROOT):
    """Verify the configuration of every scheduled job, without opening partition data."""
    root = Path(formal_root).resolve()
    plan_path = root / CAMPAIGN / "campaign_plan.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan["campaign_id"] != CAMPAIGN:
        raise ValueError(f"Unexpected campaign in {plan_path}")
    seed_groups = plan["seed_shards"]
    declared = [seed for group in seed_groups for seed in group]
    if not all(type(s) is int for s in declared) or sorted(declared) != list(FORMAL_SEEDS):
        raise ValueError("The campaign must contain each of the five formal seeds exactly once")
    jobs, evidence, shards = {}, {}, {}
    common_protocol = None
    allowed_dataset_fields = {
        "dataset_id", "display_name", "raw_path", "raw_sha256", "raw_byte_count",
        "task", "utility", "target", "variance_threshold",
    }
    # Other directories contain smoke, bootstrap or archived copies of these rows.
    for name, group in zip(plan["run_ids"], seed_groups, strict=True):
        source = (root / name).resolve()
        if Path(name).name != name or source.parent != root:
            raise ValueError(f"Invalid campaign shard path: {name}")
        protocol_bytes = (source / "resolved_protocol.yaml").read_bytes()
        protocol = yaml.safe_load(protocol_bytes)
        digest = canonical_hash(protocol)
        marker_bytes = (source / "metadata_complete.json").read_bytes()
        marker = json.loads(marker_bytes)
        if (marker["protocol_hash"] != digest or protocol["run_kind"] != "formal"
                or protocol["seeds"] != group or not group):
            raise ValueError(f"{name}: frozen protocol hash, kind or seeds disagree with the campaign")
        comparable = {k:v for k,v in protocol.items() if k != "seeds"}
        if common_protocol is None:
            common_protocol = comparable
        else:
            changed = sorted(k for k in common_protocol.keys() | comparable.keys()
                             if k not in common_protocol or k not in comparable or common_protocol[k] != comparable[k])
            if changed:
                raise ValueError(f"{name}: frozen protocol differs across seed shards for {changed}")
        specs = {spec["dataset_id"]: spec for spec in protocol["datasets"]}
        supported = set(specs) - AUDIT_ONLY_DATASETS
        if len(specs) != len(protocol["datasets"]) or supported != set(FORMAL_DATASETS):
            raise ValueError(f"{name}: dataset scope differs from the six approved formal benchmarks")
        params = protocol["model_params"]
        if not isinstance(params, dict) or type(params.get("n_estimators")) is not int or params["n_estimators"] < 1:
            raise ValueError(f"{name}: invalid XGBoost base parameters")
        common_preparation = {k:v for k,v in specs[FORMAL_DATASETS[0]].items() if k not in allowed_dataset_fields}
        for dataset in FORMAL_DATASETS:
            if protocol["model_params_by_dataset"][dataset] != params:
                raise ValueError(f"{name}/{dataset}: XGBoost settings differ from the common model parameters")
            if {k:v for k,v in specs[dataset].items() if k not in allowed_dataset_fields} != common_preparation:
                raise ValueError(f"{name}/{dataset}: preprocessing settings differ outside permitted dataset fields")

        manifest_path = source / "data_manifest.csv"
        manifest_bytes = manifest_path.read_bytes()
        shards[name] = dict(seeds=list(group), protocol_hash=digest,
                            protocol_file_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                            metadata_complete_sha256=hashlib.sha256(marker_bytes).hexdigest(),
                            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
        with StringIO(manifest_bytes.decode("utf-8-sig"), newline="") as stream:
            for row in csv.DictReader(stream, strict=True):
                dataset = row["dataset_id"]
                if dataset in AUDIT_ONLY_DATASETS:
                    continue
                seed = int(row["seed"])
                if dataset not in supported or seed not in group or (dataset, seed) in jobs:
                    raise ValueError(f"{name}/{dataset}/s{seed}: duplicate or unexpected manifest key")
                _check_row(source, row, specs[dataset], digest)
                jobs[dataset, seed] = source
                evidence[dataset, seed] = dict(dataset=dataset, seed=seed, shard=name, protocol_hash=digest,
                    task=row["task"], utility=row["utility"], target=row["target"],
                    selected_feature_count=int(row["selected_feature_count"]),
                    feature_order_sha256=canonical_hash(json.loads(row["selected_features_json"])),
                    rows={p:int(row[f"{p}_row_count"]) for p in ("train","val","test")},
                    declared_partition_sha256={p:row[f"{p}_file_sha256"] for p in ("train","val","test")})

    expected = [(d, s) for s in FORMAL_SEEDS for d in FORMAL_DATASETS]
    missing = [key for key in expected if key not in jobs]
    if missing:
        raise ValueError(f"Missing frozen dataset/seed combinations: {missing}")
    base_rounds = params["n_estimators"]
    active_keys = {"backend", "model_params", "model_params_by_dataset", "alpha_default", "datasets",
                   "removal", "run_kind", "seeds"}
    audit = dict(
        status="PASS", seeds=list(FORMAL_SEEDS), datasets=list(FORMAL_DATASETS),
        campaign_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        shared=dict(backend=protocol["backend"], model_params=params, alpha=protocol["alpha_default"],
                    checkpoints=[base_rounds, (3*base_rounds+1)//2, 2*base_rounds],
                    classification_margin=protocol["removal"]["classification_noninferiority_margin"],
                    regression_margin=protocol["removal"]["regression_noninferiority_margin"],
                    preprocessing=common_preparation),
        permitted_dataset_fields=sorted(allowed_dataset_fields),
        dataset_settings={d:{k:v for k,v in specs[d].items() if k in allowed_dataset_fields and k != "raw_path"}
                          for d in FORMAL_DATASETS},
        permitted_shard_differences=["seeds"], compared_protocol_keys=sorted(common_protocol),
        archived_keys_not_used_to_choose_thesis_models=sorted(common_protocol.keys() - active_keys),
        archived_removal_keys_not_used=sorted(set(protocol["removal"]) - {
            "classification_noninferiority_margin", "regression_noninferiority_margin"}),
        shards=shards, jobs=[evidence[key] for key in expected],
    )
    audit["sha256"] = canonical_hash(audit)
    return {key: jobs[key] for key in expected}, audit


def discover_prepared_jobs(formal_root=FORMAL_ROOT):
    """Return the (dataset, seed) to source mapping after the full configuration audit."""
    jobs, _ = audit_prepared_catalog(formal_root)
    return jobs
