"""Compare rankings at each PIv0 validation-selected k on the frozen five-seed splits."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.run_thesis_sampling import MSK, ROOT
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance
from minppi_mofs.experiment_protocol import (
    AuditLog, AuditedSplitAccess, JobSpec, PreparedSplitPaths, atomic_write_json, canonical_hash,
)
from minppi_mofs.modeling import fit_predict, make_model
from minppi_mofs.utility import compute_utility


CAMPAIGN = ROOT / "results/thesis_benchmark/PIv0-all-seeds-20261111-213840-398718"
OUTPUT = ROOT / "results/test/thesis_sampling"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(campaign, source, row, out):
    dataset, seed = row["dataset_id"], int(row["seed"])
    arm = campaign / f"seed-{seed}" / dataset
    train, yt, _, _ = load_prepared_train_validation(source, row)
    features = list(train.columns)
    hashes = {s: row[f"{s}_file_sha256"] for s in ("train", "val", "test")}
    previous = {}
    for method in ("PIv0", "direct_gain"):
        ranking_path = arm / method / "ranking.json"
        rank = read(ranking_path)
        frozen = read(arm / method / "frozen.json")
        if (rank["feature_order"] != features or rank["partition_sha256"] != hashes
                or frozen["dataset"] != dataset or frozen["seed"] != seed or frozen["method"] != method
                or frozen["metric"] != row["utility"] or digest(ranking_path) != frozen["ranking_sha256"]
                or canonical_hash(frozen) != read(arm/method/"result.json")["frozen_hash"]):
            raise ValueError(f"Frozen inputs disagree for {dataset}/{seed}/{method}")
        previous[method] = (rank, frozen)
    gain, _ = previous["direct_gain"]
    piv0, decision = previous["PIv0"]
    policy = gain["policy"]
    if policy != piv0["policy"] or policy["seed"] != seed:
        raise ValueError("Frozen model policies disagree")
    k = decision["retained"]
    safe = train.set_axis([f"f{i}" for i in range(len(features))], axis=1)
    model = make_model(**policy)
    model.fit(safe, yt)
    importance = fitted_tree_shap_importance(model, safe, seed=seed, max_rows=1024)
    order = np.argsort(-importance, kind="stable")
    selections = {
        "piv0": set(decision["selected"]),
        "gain": set(gain["descending_gain"][:k]),
        "treeshap": {features[i] for i in order[:k]},
    }
    columns = {m: [f for f in features if f in subset] for m, subset in selections.items()}
    if not all(len(c) == k for c in columns.values()):
        raise ValueError("Fixed-k comparison requires three masks of equal cardinality")
    frozen = dict(dataset=dataset, seed=seed, k=k, metric=row["utility"], policy=policy,
                  columns=columns, partition_sha256=hashes, feature_order=features,
                  training_mean_abs_shap=importance.tolist(), piv0_hash=canonical_hash(decision))
    out.mkdir(parents=True)
    atomic_write_json(out / "frozen.json", frozen)
    return canonical_hash(frozen)


def evaluate(campaign, source, row, out, identity):
    frozen = read(out / "frozen.json")
    if canonical_hash(frozen) != identity:
        raise ValueError("Frozen selection changed before evaluation")
    dataset, seed = frozen["dataset"], frozen["seed"]
    paths = PreparedSplitPaths(dataset, seed, row["target"],
                               *(source/row[f"{s}_relpath"] for s in ("train", "val", "test")))
    job = JobSpec(identity, row["protocol_hash"], "removal", dataset, seed, "fixed_k",
                  {"k": frozen["k"]}, (), "result.json")
    audit = AuditLog.from_manifest(out/"test_access.json", [job])
    train, yt, val, yv = load_prepared_train_validation(source, row)
    test, ye = AuditedSplitAccess(paths, stage="removal", job_id=identity, audit=audit).load_test(
        reason="final_removal_evaluation", expected_sha256=frozen["partition_sha256"]["test"])
    if list(test.columns) != frozen["feature_order"] or len(ye) != row["test_row_count"]:
        raise ValueError("Unexpected test split")
    joint, labels = pd.concat([train, val], ignore_index=True), np.concatenate([yt, yv])
    with np.load(campaign/f"seed-{seed}"/dataset/"PIv0"/"predictions.npz", allow_pickle=False) as old:
        if not np.array_equal(old["y"], ye):
            raise ValueError("Historical PIv0 labels disagree")
        predictions = {"piv0": old["selected"].copy()}
    evaluated = {tuple(frozen["columns"]["piv0"]): predictions["piv0"]}
    for name in ("gain", "treeshap"):
        columns = frozen["columns"][name]
        mask = tuple(columns)
        if mask not in evaluated:
            evaluated[mask] = fit_predict(joint[columns], labels, test[columns], **frozen["policy"])
        predictions[name] = evaluated[mask]
    scores = {m: float(compute_utility(ye, pred, metric=row["utility"])) for m, pred in predictions.items()}
    if abs(scores["piv0"] - read(campaign/f"seed-{seed}"/dataset/"PIv0"/"result.json")["test_utility"]) > 1e-12:
        raise ValueError("PIv0 test utility did not reproduce")
    np.savez_compressed(out/"predictions.npz", y=ye, **predictions)
    result = dict(dataset=dataset, seed=seed, metric=row["utility"], k=frozen["k"], **scores,
                  piv0_minus_gain=scores["piv0"]-scores["gain"],
                  piv0_minus_treeshap=scores["piv0"]-scores["treeshap"], fresh_final_fits=len(evaluated)-1)
    atomic_write_json(out/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    campaign = parser.parse_args().campaign.resolve()
    jobs, catalog = audit_prepared_catalog()
    if read(campaign/"protocol.json")["catalog_sha256"] != catalog["sha256"]:
        raise ValueError("Prepared catalog differs from the PIv0 campaign")
    rows = {}
    for source in set(jobs.values()):
        with (source/"data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for split in ("train", "val", "test"):
                    row[f"{split}_row_count"] = int(row[f"{split}_row_count"])
                rows[row["dataset_id"], int(row["seed"])] = row
    root = OUTPUT / datetime.now(MSK).strftime("piv0-fixed-k-controls-%Y%m%d-%H%M%S-%f")
    root.mkdir()
    print(root, flush=True)
    atomic_write_json(root/"protocol.json", dict(campaign=str(campaign), catalog_sha256=catalog["sha256"],
                      k="Original PIv0 validation-selected k for each matched dataset/seed",
                      gain="Original full-model gain ranking",
                      treeshap="Mean absolute exact training TreeSHAP across rows and outputs, up to 1024 rows "
                               "sampled without replacement by split seed; original feature order breaks ties.",
                      source_sha256=digest(Path(__file__))))
    identities = {}
    for key, source in jobs.items():
        dataset, seed = key
        out = root/f"seed-{seed}"/dataset
        identities[key] = freeze(campaign, source, rows[key], out)
        print(f"Frozen {dataset}/{seed}: {len(identities)}/{len(jobs)}", flush=True)
    atomic_write_json(root/"frozen_identities.json", {f"{d}/{s}":v for (d,s),v in identities.items()})
    records = []
    for key, source in jobs.items():
        dataset, seed = key
        records.append(evaluate(campaign, source, rows[key], root/f"seed-{seed}"/dataset, identities[key]))
        print(f"Evaluated {dataset}/{seed}: {len(records)}/{len(jobs)}", flush=True)
    data = pd.DataFrame(records)
    data.to_csv(root/"paired.csv", index=False)
    data.groupby(["dataset","metric"])[["k","piv0","gain","treeshap","piv0_minus_gain","piv0_minus_treeshap"]].mean().to_csv(
        root/"dataset_means.csv")


if __name__ == "__main__":
    main()
