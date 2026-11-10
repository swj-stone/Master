"""Pair a training-partition sensitivity check with the official seed-42 round-gate results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.compare_fixed_predictor import PRIMARY, validate_decision
from experiments.compare_local_refinement import FORMAL, _rows, paired_interval
from experiments.compare_thesis_sign_aware import TOL, check, close, read
from experiments.run_thesis_predictor import preparation_budget
from experiments.run_thesis_sampling import DATASETS
from minppi_mofs.experiment_protocol import canonical_hash
from minppi_mofs.utility import compute_utility


def band(value, margin):
    if value >= 0.1 * margin - TOL:
        return "improved"
    if value < -0.1 * margin - TOL:
        return "declined"
    return "within_threshold"


def _indices(y, task):
    labels = y if task != "regression" else None
    development, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=labels)
    development = np.sort(development)
    fit, val = train_test_split(
        development,
        test_size=0.25,
        random_state=42,
        stratify=labels[development] if labels is not None else None,
    )
    return [np.sort(group) for group in (fit, val, test)]


def _check_indices(meta, parts, task):
    digests = [hashlib.sha256(group.astype("<i8").tobytes()).hexdigest() for group in parts]
    check(
        meta["seed"] == 42
        and meta["stratified"] == (task != "regression")
        and meta["index_sha256"] == digests
        and meta["sizes"] == [len(a) for a in parts],
        "internal split row identities",
    )


def verify_sensitivity(folder, watch, manifest, frozen, digest):
    d = read(folder / "result.json")
    ds, task, metric = manifest["dataset_id"], manifest["task"], manifest["utility"]
    check(
        watch["exit_code"] == 0
        and watch["result"] == d
        and d["phase"] == "complete"
        and d["dataset"] == ds
        and d["seed"] == 42,
        f"{ds}: completed internal run",
    )
    fs = json.loads(manifest["selected_features_json"])
    mask = frozen["selected_mask"]
    check(
        d["features"] == fs
        and d["mask_manifest_sha256"] == digest
        and d["guarded_mask"] == mask
        and canonical_hash(fs) == frozen["feature_order_sha256"],
        f"{ds}: frozen feature set",
    )
    train_path = FORMAL / manifest["train_relpath"]
    check(
        d["source_partition"] == "train"
        and d["source_partition_sha256"]
        == manifest["train_file_sha256"]
        == hashlib.sha256(train_path.read_bytes()).hexdigest(),
        f"{ds}: original training-only data provenance",
    )
    target = pd.read_csv(train_path, usecols=[manifest["target"]]).iloc[:, 0].to_numpy()
    fit, val, test = _indices(target, task)
    check(
        d["training_split"]["index_space"] == "zero-based positions in frozen training CSV"
        and d["training_split"]["order"] == ["fit", "validation", "evaluation"]
        and len(np.unique(np.concatenate((fit, val, test)))) == len(target),
        f"{ds}: disjoint original rows",
    )
    _check_indices(d["training_split"], [fit, val, test], task)
    halves = [
        np.sort(group)
        for group in train_test_split(
            np.arange(len(val)),
            test_size=0.5,
            random_state=42,
            stratify=target[val] if task != "regression" else None,
        )
    ]
    _check_indices(d["validation_halves"], halves, task)
    params = d["parameters"]
    nrounds = d["policy"]["model_params"]["n_estimators"]
    check(
        params["original_rounds"] == nrounds
        and params["blend_checkpoints"] == [(3 * nrounds + 1) // 2, 2 * nrounds]
        and params["blend_weights"] == [0.5, 0.5],
        f"{ds}: checkpoint recipe",
    )
    check(d["test_fits"] == d["validation_fits"] == 1, f"{ds}: nested checkpoint fit count")
    lockfile = folder / "frozen_decision.json"
    check(hashlib.sha256(lockfile.read_bytes()).hexdigest() == d["decision_sha256"], f"{ds}: decision checksum")
    check(
        read(lockfile)
        == {
            k: d[k]
            for k in (
                "dataset",
                "seed",
                "source_partition_sha256",
                "mask_manifest_sha256",
                "training_split",
                "validation_halves",
                "parameters",
                "decisions",
            )
        },
        f"{ds}: final recipes differ from frozen sensitivity decision",
    )
    decision = d["decisions"]["guarded"]
    selected = validate_decision(decision, decision["original_scores"], mask)
    archive = folder / "paired_test_predictions.npz"
    check(
        d["paired_test_predictions"] == archive.name
        and hashlib.sha256(archive.read_bytes()).hexdigest() == d["paired_test_predictions_sha256"],
        f"{ds}: paired predictions archive",
    )
    with np.load(archive, allow_pickle=False) as arr:
        keys = ["guarded_original", "guarded_blend", "guarded_chosen"]
        check(
            set(arr.files) == {"y", *keys} and set(d["method_results"]) == set(keys), f"{ds}: prediction keys"
        )
        y = arr["y"]
        check(
            y.shape == target[test].shape and np.allclose(y, target[test], rtol=1e-12, atol=1e-12),
            f"{ds}: frozen evaluation rows in original training partition",
        )
        scores = {}
        for key in keys:
            entry = d["method_results"][key]
            recipe = selected if key == "guarded_chosen" else key.split("_")[1]
            scores[key] = float(compute_utility(y, arr[key], metric=metric))
            check(
                entry["recipe"] == recipe
                and entry["mask"] == mask
                and close(entry["test_utility"], scores[key])
                and entry["scores"] == decision[recipe + "_scores"]
                and np.array_equal(arr[key], arr["guarded_" + recipe]),
                f"{ds}/{key}: predictions and scores",
            )
        base = scores["guarded_original"]
        for key in keys:
            check(
                close(scores[key] - base, d["method_results"][key]["test_delta_vs_original"]),
                f"{ds}/{key}: paired delta",
            )
        low, high = paired_interval(y, arr["guarded_chosen"], arr["guarded_original"], metric, draws=500)
    return dict(
        metric=metric,
        evaluation_rows=len(test),
        fit_rows=len(fit),
        validation_rows=len(val),
        recipe=selected,
        guarded_mask=mask,
        removed=len(fs) - int(mask, 16).bit_count(),
        original=base,
        selected=scores["guarded_chosen"],
        delta=scores["guarded_chosen"] - base,
        ci_low=low,
        ci_high=high,
    )


def compare(run, rounds, prior):
    cfg = read(run / "protocol.json")
    official_cfg = read(rounds / "protocol.json")
    mask_path = prior / "comparison/guarded_validation_manifest.json"
    digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    frozen = read(mask_path)
    official = {
        r["dataset"]: r
        for r in _rows(rounds / "comparison/paired_predictor_results.csv")
        if r["contrast"] == PRIMARY
    }
    records = {r["dataset_id"]: r for r in _rows(FORMAL / "data_manifest.csv") if r["seed"] == "42"}
    check(
        cfg["seed"] == official_cfg["seed"] == 42
        and official_cfg["predictor_recipe"] == cfg["predictor_recipe"] == "rounds"
        and cfg["evaluation_partition"] == "training_check"
        and cfg["practical_change_fraction_of_margin"] == 0.1
        and cfg["paired_bootstrap_seed"] == 42
        and cfg["paired_bootstrap_replicates"] == 500
        and cfg["minimum_improved_datasets"] == 2
        and cfg["mask_manifest_sha256"] == official_cfg["mask_manifest_sha256"] == digest
        and frozen["source_run"] == prior.name,
        "confirmation source, seed or input manifest",
    )
    logs = (prior / "run.log").read_text(encoding="utf-8").splitlines()
    rows = []
    for ds in DATASETS:
        folder = run / ds
        watch = read(folder / "watchdog.json") if (folder / "watchdog.json").is_file() else {}
        state = watch.get("status", "missing")
        base = dict(dataset=ds, status=state, charged_s=watch.get("total_charged_s"))
        if state != "completed":
            rows.append(base)
            continue
        charge, basis = preparation_budget(ds, read(prior / ds / "watchdog.json"), logs)
        check(
            close(watch["inherited_preparation_s"], charge)
            and close(cfg["preparation"][ds]["charged_s"], charge)
            and cfg["preparation"][ds]["basis"] == basis
            and close(watch["elapsed_s"] + charge, watch["total_charged_s"])
            and watch["total_charged_s"] <= watch["dataset_limit_s"] == cfg["dataset_seconds"] <= 900,
            f"{ds}: inherited preparation and run budget",
        )
        check(ds in cfg["datasets"] and official[ds]["status"] == "completed", f"{ds}: missing official result")
        check(
            watch["result"]["policy"] == read(rounds / ds / "result.json")["policy"],
            f"{ds}: model policy drift",
        )
        row = verify_sensitivity(folder, watch, records[ds], frozen["datasets"][ds], digest)
        official_delta, margin = float(official[ds]["delta"]), float(official[ds]["margin"])
        internal = band(row["delta"], margin)
        prior_band = band(official_delta, margin)
        rows.append(
            dict(
                **base,
                **row,
                margin=margin,
                official_delta=official_delta,
                internal_band=internal,
                official_band=prior_band,
                both_positive=official_delta > TOL and row["delta"] > TOL,
                both_improved=internal == prior_band == "improved",
                both_declined=internal == prior_band == "declined",
            )
        )
    done = [r for r in rows if r["status"] == "completed"]

    def names(test):
        return [r["dataset"] for r in done if test(r)]

    gains = names(lambda r: r["internal_band"] == "improved")
    losses = names(lambda r: r["internal_band"] == "declined")
    summary = dict(
        completed=len(done),
        expected=len(DATASETS),
        primary=PRIMARY,
        paired_bootstrap_replicates=500,
        meaningful_internal_improvements=gains,
        material_internal_declines=losses,
        both_positive=names(lambda r: r["both_positive"]),
        both_improved=names(lambda r: r["both_improved"]),
        both_declined=names(lambda r: r["both_declined"]),
        positive_paired_intervals=names(lambda r: r["ci_low"] > 0),
        negative_paired_intervals=names(lambda r: r["ci_high"] < 0),
        internal_change_criterion_met=len(done) == len(DATASETS) and len(gains) >= 2 and not losses,
    )
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("official_rounds", type=Path)
    parser.add_argument("confirmed", type=Path)
    args = parser.parse_args()
    rows, summary = compare(args.run, args.official_rounds, args.confirmed)
    out = args.run / "comparison"
    out.mkdir(exist_ok=True)
    with (out / "paired_replication.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    encoded = json.dumps(summary, indent=2, allow_nan=False)
    (out / "paired_replication_summary.json").write_text(encoded, encoding="utf-8")
    print(encoded)
    return int(summary["completed"] != summary["expected"])


if __name__ == "__main__":
    raise SystemExit(main())
