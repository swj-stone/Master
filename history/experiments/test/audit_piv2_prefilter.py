"""Verify archived PIv2 predictions, votes and paired results without rerunning selection."""

import argparse
from collections import Counter
import csv
import json
from math import ceil, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

from experiments.run_piv0_benchmark import read, sha
from experiments.run_thesis_sampling import ROOT
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


DATASETS = ("madelon", "spambase", "abalone", "gisette", "credit_card_fraud", "isolet")
BACKENDS = ("gain", "pfi", "tree_shap", "f_score", "mutual_info", "l1", "banzhaf", "exact", "mc32")
FORMAL = ROOT / "results/thesis_benchmark/PIv1-all-seeds-20261112-003506-318341"
ACCURACY = FORMAL / "accuracy-20261113-105715-784709"
PFI = ROOT / "results/test/thesis_sampling/sampled-permutation-seed42-20261113-113612-912266"
MEASURES = ("prefilter_s", "backend_s", "final_s", "prefilter_retained", "retained", "fits",
            "validation_utility", "validation_secondary", "test_utility", "test_secondary")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(value, expected):
    np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)


def ordered_subset(values, universe):
    membership = set(values)
    return bool(values) and values == [c for c in universe if c in membership]


def validate_votes(votes, columns, candidates):
    width = len(columns)
    require(votes["columns"] == columns and votes["retained"] == candidates, "PI support mismatch")
    require(votes["input_count"] == width and 0 <= votes["sample_rows"] <= 1024, "PI sample mismatch")
    if votes["bypass"] is not None:
        require(candidates == columns and not votes["min_vote"], "Invalid PI bypass")
        return
    folds = np.asarray(votes["fold_tier_net"])
    require(folds.shape == (2, 3, width) and np.issubdtype(folds.dtype, np.integer), "Invalid votes")
    require(np.abs(folds).max() <= 4, "Voting count exceeds repetitions")
    tiers = np.clip(np.rint((width-1)*np.asarray([1/6, 1/2, 5/6])).astype(int), 1, width-1)
    np.testing.assert_array_equal(votes["tier_sizes"], tiers)
    rates = folds.sum(axis=0)/8
    worst = rates.min(axis=0)
    near(votes["tier_rates"], rates)
    near(votes["min_vote"], worst)
    strength = np.asarray(votes["standalone"])
    require(np.isfinite(strength).all() and strength.shape == (width,), "Invalid standalone strengths")
    protected = sorted(range(width), key=lambda j: -strength[j])[:ceil(sqrt(width))]
    require(votes["protected"] == [columns[j] for j in protected], "Invalid protected columns")
    eligible = [j for j in range(width) if j not in protected and worst[j] < 0
                and any(folds[0, h, j] < 0 and folds[1, h, j] < 0 for h in range(3))]
    removed = set(sorted(eligible, key=lambda j: (worst[j], strength[j], j))[:width//2])
    require(votes["removed"] == [c for j, c in enumerate(columns) if j in removed], "Invalid PI removals")
    require(candidates == [c for j, c in enumerate(columns) if j not in removed], "Invalid PI output")


def metric_pair(labels, predictions, metric):
    require(np.isfinite(predictions).all(), "Nonfinite predictions")
    if metric == "r2":
        return max(0.0, r2_score(labels, predictions)), mean_absolute_error(labels, predictions)
    require(predictions.ndim == 2 and predictions.shape[0] == len(labels), "Invalid probabilities")
    require(np.all((predictions >= 0) & (predictions <= 1)) and
            np.allclose(predictions.sum(axis=1), 1, rtol=0, atol=2e-6), "Invalid probability simplex")
    if metric == "roc_auc":
        auc = roc_auc_score(labels, predictions[:, 1])
    else:
        require(metric == "ovr_auc", "Unknown classification metric")
        auc = roc_auc_score(labels, predictions, multi_class="ovr", average="macro")
    return auc, accuracy_score(labels, predictions.argmax(axis=1))


def dataset_inputs(jobs):
    data = {}
    for dataset in DATASETS:
        source = jobs[dataset, 42]
        with (source/"data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            row, = [r for r in csv.DictReader(stream) if r["dataset_id"] == dataset and int(r["seed"]) == 42]
        labels, digests = {}, {}
        for split in ("train", "val", "test"):
            path = source/row[split+"_relpath"]
            digests[split] = sha(path)
            require(digests[split] == row[split+"_file_sha256"], f"{dataset} changed {split} data")
            if split != "train":
                labels[split] = pd.read_csv(path, usecols=[row["target"]])[row["target"]].to_numpy()
        data[dataset] = (row, digests, labels)
    return data


def check_history(backend, dataset, frozen, new, accuracy_rows):
    if backend == "gain":
        old_dir = FORMAL/"seed-42"/dataset/"direct_gain"
        val_file = ACCURACY / accuracy_rows.loc[dataset, "validation_predictions"]
        with np.load(val_file, allow_pickle=False) as old:
            for k in ("selected", "y"):
                np.testing.assert_array_equal(old[k], new["validation" if k == "selected" else "validation_y"])
    else:
        old_dir = PFI/dataset
    require(frozen["selected"] == read(old_dir/"frozen.json")["selected"], "Historical selected columns changed")
    with np.load(old_dir/"predictions.npz", allow_pickle=False) as old:
        for k in old.files:
            if k in new.files:
                np.testing.assert_array_equal(old[k], new[k])


def check_selector(backend, result, frozen):
    d, k, retained = result["width"], result["prefilter_retained"], result["retained"]
    diag, names = frozen["selector"], frozen["candidates"]
    if backend in ("tree_shap", "f_score", "mutual_info", "banzhaf"):
        require(diag["k"] == retained == min(k, ceil(sqrt(d))), "Inconsistent fixed-k selection")
    if backend == "gain":
        curve, order = diag["curve"], diag["gain_order"]
        require(len(curve) == len(order) == len(set(order)) == k and set(order) == set(names), "Invalid gain scan")
        winner = 0
        for j in range(1, len(curve)):
            if curve[j] > curve[winner] + 1e-14:
                winner = j
        require(winner+1 == retained and set(frozen["selected"]) == set(order[:retained]), "Invalid gain prefix")
        near(curve[winner], result["validation_utility"])
        fits = k+1
    elif backend == "pfi":
        drops = diag["mean_drops"]
        positive = [c for c, v in zip(names, drops, strict=True) if v > 0]
        require(diag["repeats"] == 5 and frozen["selected"] == (positive or [names[int(np.argmax(drops))]]),
                "Invalid PFI selection")
        fits = 2 + int(retained < k)
    elif backend == "exact":
        require(diag["coalition_count"] == 2**k-1, "Incomplete exact search")
        fits = 2**k
    elif backend == "mc32":
        require(diag["paths"] == 32, "MC path count mismatch")
        positive = [c for c, v in zip(names, diag["mean_marginal"], strict=True) if v > 0]
        require(frozen["selected"] == (positive or [names[int(np.argmax(diag["mean_marginal"]))]]),
                "Invalid MC selection")
        require(result["fits"] in (diag["coalition_count"]+1,diag["coalition_count"]+2), "MC fit count mismatch")
        return
    elif backend == "l1":
        require(diag["selector_fits"] == 1, "L1 model fit count mismatch")
        fits = 3
    elif backend in ("f_score", "mutual_info"):
        fits = 2
    else:
        fits = 2 + int(retained < k)
    require(fits == result["fits"], f"{backend} fit count mismatch")


def inspect_arm(run, key, summary, protocol, fingerprint, inputs, supports, counts, accuracy_rows, *, selector_check=check_selector):
    dataset, backend, arm = key
    source, digests, labels = inputs[dataset]
    location = run / dataset / backend / arm
    watchdog = read(location/"watchdog.json")
    status = watchdog["status"]
    require(watchdog["limit_s"] == protocol["arm_limit_s"] == 900 and watchdog["elapsed_s"] > 0, f"{key} timer")
    require(summary["status"] == status and int(summary["seed"]) == 42, f"{key} summary identity")
    require(summary["metric"] == source["utility"] and int(summary["width"]) == int(source["selected_feature_count"]),
            f"{key} summary dataset")
    near(float(summary["wall_s"]), watchdog["elapsed_s"])
    row = dict(status=status, wall_s=watchdog["elapsed_s"], **dict.fromkeys(MEASURES))
    counts[status] += 1
    if status != "completed":
        require(status in ("timed_out", "error"), f"{key} unknown terminal status")
        return row
    require(watchdog["exit_code"] == 0 and watchdog["kill_requested_s"] is None, f"{key} completed status")
    frozen, result = read(location/"frozen.json"), read(location/"result.json")
    require(watchdog["result"] == result and canonical_hash(frozen) == result["frozen_hash"], f"{key} digest")
    require(frozen["protocol_fingerprint"] == fingerprint and frozen["partition_sha256"] == digests, f"{key} inputs")
    for record in (result, frozen):
        require((record["dataset"], record["backend"], record["arm"], record["seed"]) == (*key, 42), f"{key} key")
        require(record["metric"] == source["utility"], f"{key} metric")
    require(frozen["model_policy"] == dict(task=source["task"], model_params=protocol["model_params"],
                                            backend=protocol["hardware_backend"], seed=42), f"{key} model")
    features, candidates, selected = (frozen[k] for k in ("features", "candidates", "selected"))
    require(features == json.loads(source["selected_features_json"]) and ordered_subset(candidates, features)
            and ordered_subset(selected, candidates), f"{key} column order or identity")
    require(result["retained"] == len(selected) and result["prefilter_retained"] == len(candidates)
            and result["width"] == len(features), f"{key} dimensions")
    if arm == "piv2":
        validate_votes(frozen["prefilter"], features, candidates)
        digest = canonical_hash(frozen["prefilter"])
        require(supports.setdefault(dataset, digest) == digest, f"{key} inconsistent prefilter")
    else:
        require(frozen["prefilter"] is None and candidates == features and result["prefilter_s"] == 0,
                f"{key} plain support")
    events = read(location/"test_access.json")["events"]
    require(len(events) == 1 and events[0]["seed"] == 42 and events[0]["dataset_id"] == dataset
            and events[0]["job_id"] == result["frozen_hash"] and events[0]["decision"] == "allowed"
            and events[0]["reason"] == "final_removal_evaluation", f"{key} test access")
    progress = read(location/"progress.json")
    require(progress["phase"] == "complete" and progress["fits"] == result["fits"], f"{key} final fit count")
    selector_check(backend, result, frozen)
    for k in MEASURES:
        near(float(summary[k]), result[k])
        row[k] = result[k]
    require(min(row[k] for k in ("prefilter_s","backend_s","final_s")) >= 0
            and sum(row[k] for k in ("prefilter_s","backend_s","final_s")) < row["wall_s"], f"{key} stage times")
    with np.load(location/"predictions.npz", allow_pickle=False) as predictions:
        for split, target, response, score_name in (("val","validation_y","validation","validation"),
                                                    ("test","y","selected","test")):
            np.testing.assert_array_equal(predictions[target], labels[split])
            scored = metric_pair(predictions[target], predictions[response], result["metric"])
            near(scored, [result[score_name+"_utility"], result[score_name+"_secondary"]])
            if split == "val":
                near(scored, [frozen["validation_utility"], frozen["validation_secondary"]])
        if arm == "plain" and backend in ("pfi", "gain"):
            check_history(backend, dataset, frozen, predictions, accuracy_rows)
            counts["historical_prediction_matches"] += 1
    return row


def audit(run):
    protocol = read(run/"protocol.json")
    fingerprint = canonical_hash(protocol)
    require(protocol["seed"] == 42 and protocol["cold_start"] is True, "Expected seed-42 cold-start run")
    for path, digest in protocol["source_sha256"].items():
        require(sha(run/"source"/path) == digest, f"Source snapshot changed: {path}")
    jobs, catalog = audit_prepared_catalog()
    require(protocol["catalog_sha256"] == catalog["sha256"], "Prepared catalog mismatch")
    expected = {(d,b,a) for d in DATASETS for b in BACKENDS if d == "abalone" or b not in ("exact","mc32")
                for a in ("plain", "piv2")}
    tasks = [tuple(t) for t in protocol["tasks"]]
    require(set(tasks) == expected and len(tasks) == len(expected) == 88, "Unbalanced task protocol")
    with (run/"summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summary = list(csv.DictReader(stream))
    by_key = {(r["dataset"], r["backend"], r["arm"]):r for r in summary}
    require(set(by_key) == expected and len(summary) == len(expected), "Incomplete or duplicate terminal records")
    evidence = dataset_inputs(jobs)
    archive = pd.read_csv(ACCURACY/"arms.csv")
    accuracy = archive[(archive.seed == 42) & (archive.method == "direct_gain")].set_index("dataset")
    supports, counts = {}, Counter()
    results = {key:inspect_arm(run, key, by_key[key], protocol, fingerprint, evidence, supports, counts, accuracy)
               for key in tasks}
    pairs = []
    for dataset, backend in sorted({(d,b) for d,b,a in expected}):
        plain, piv2 = (results[dataset,backend,a] for a in ("plain","piv2"))
        r = by_key[dataset,backend,"plain"]
        pair = dict(dataset=dataset, backend=backend, seed=42, metric=r["metric"], secondary=r["secondary_metric"],
                    width=int(r["width"]), complete=(plain["status"] == piv2["status"] == "completed"),
                    seconds_saved=None, speedup=None, test_delta=None)
        for arm, result in (("plain",plain),("piv2",piv2)):
            pair.update({arm+"_"+k:v for k,v in result.items()})
        if pair["complete"]:
            pair.update(seconds_saved=plain["wall_s"]-piv2["wall_s"], speedup=plain["wall_s"]/piv2["wall_s"],
                        test_delta=piv2["test_utility"]-plain["test_utility"])
        pairs.append(pair)
    paired = pd.DataFrame(pairs)
    aggregated = []
    for backend, group in paired.groupby("backend", sort=False):
        complete = group[group.complete]
        near_zero = 1e-14
        aggregated.append(dict(backend=backend, scheduled=len(group), completed_pairs=len(complete),
            piv2_faster=int((complete.seconds_saved > 0).sum()) if len(complete) else 0,
            higher_test=int((complete.test_delta > near_zero).sum()) if len(complete) else 0,
            tied_test=int((complete.test_delta.abs() <= near_zero).sum()) if len(complete) else 0,
            lower_test=int((complete.test_delta < -near_zero).sum()) if len(complete) else 0,
            plain_wall_s=complete.plain_wall_s.sum(), piv2_wall_s=complete.piv2_wall_s.sum(),
            prefilter_s=complete.piv2_prefilter_s.sum(), plain_fits=complete.plain_fits.sum(),
            piv2_fits=complete.piv2_fits.sum()))
    paired.to_csv(run/"paired.csv", index=False)
    pd.DataFrame(aggregated).to_csv(run/"by_backend.csv", index=False)
    report = dict(status="pass", protocol_sha256=fingerprint, summary_file_sha256=sha(run/"summary.csv"),
                  paired_file_sha256=sha(run/"paired.csv"), planned_arms=len(tasks), scheduled_pairs=len(paired),
                  completed_pairs=int(paired.complete.sum()), counts=dict(counts), prefilter_sha256=supports,
                  partition_files_checked=len(evidence)*3,
                  historical_arms={"gain":str(FORMAL.relative_to(ROOT)), "pfi":str(PFI.relative_to(ROOT))})
    atomic_write_json(run/"audit.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run
    try:
        audit(run)
    except Exception as error:
        if run.is_dir():
            atomic_write_json(run/"audit.json", dict(status="fail", error=repr(error)))
        raise
