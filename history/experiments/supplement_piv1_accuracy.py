"""Add accuracy at the AUC-selected masks of a completed PIv1 versus Gain benchmark."""

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.calibrate_icomp2026_models import load_prepared_train_validation
from experiments.run_thesis_sampling import MSK
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash
from minppi_mofs.modeling import fit_predict
from minppi_mofs.utility import compute_utility


METHODS = ("PIv1", "direct_gain")
FIELDS = ("validation_primary", "test_primary", "validation_accuracy", "test_accuracy")
REG_FIELDS = ("validation_mae", "validation_within_one", "test_mae", "test_within_one")


def read(path):
    return json.loads(path.read_bytes())


def check_score(observed, expected, context):
    if not np.isfinite(observed) or abs(observed - float(expected)) > 1e-12:
        raise ValueError(f"{context}: expected {expected}, observed {observed}")


def classification_metrics(y, prob, classes, metric):
    if (prob.shape != (len(y), len(classes)) or not np.isfinite(prob).all()
            or not np.allclose(prob.sum(axis=1), 1., atol=1e-5)
            or not np.isin(y, classes).all()):
        raise ValueError("Invalid frozen classifier probabilities or labels")
    predicted = classes[np.argmax(prob, axis=1)]
    correct = int(np.count_nonzero(predicted == y))
    return compute_utility(y, prob, metric=metric), correct, correct / len(y)


def paired_reports(arms, out, fields, pair_filename, summary_filename):
    paired = []
    for (dataset, seed, metric), group in arms.groupby(["dataset", "seed", "primary_metric"]):
        methods = group.set_index("method")
        if set(methods.index) != set(METHODS) or len(methods) != 2:
            raise ValueError(f"Incomplete accuracy pair for {dataset}/{seed}")
        row = dict(dataset=dataset, seed=seed, primary_metric=metric)
        for field in fields:
            first, second = methods.loc["PIv1", field], methods.loc["direct_gain", field]
            row["piv1_" + field] = first
            row["gain_" + field] = second
            row["delta_" + field] = first - second
        paired.append(row)
    pairs = pd.DataFrame(paired)
    pairs.to_csv(out/pair_filename, index=False, encoding="utf-8-sig")
    groups = pairs.groupby(["dataset", "primary_metric"], as_index=True)
    stats = groups[[k for k in pairs if k.startswith(("piv1_", "gain_", "delta_"))]].agg(["mean", "std"])
    stats.columns = ["_".join(col) for col in stats.columns]
    stats.insert(0, "n_pairs", groups.size())
    for field in fields:
        if field.endswith(("accuracy", "within_one")):
            stats[f"{field}_delta_pp"] = 100 * stats[f"delta_{field}_mean"]
    stats.reset_index().to_csv(out/summary_filename, index=False, encoding="utf-8-sig")


def supplement(run, out, validation_cache=None):
    jobs, catalog = audit_prepared_catalog()
    formal = read(run/"protocol.json")
    audit = read(run/"independent_audit.json")
    if (formal["version"] != "PIv1" or formal["catalog_hash"] != catalog["sha256"]
            or set(formal["methods"]) != set(METHODS) or audit["status"] != "pass"
            or audit["verified_pairs"] != len(jobs)):
        raise ValueError("Expected an audited, completed five-seed PIv1/Gain formal run")
    summary = pd.read_csv(run/"summary.csv").set_index(["dataset", "seed", "method"])
    if not summary.index.is_unique or len(summary) != 2*len(jobs) or not summary.status.eq("completed").all():
        raise ValueError("Incomplete source experiment")
    prior = None
    if validation_cache is not None:
        cache_info = read(validation_cache/"protocol.json")
        if (cache_info["source_protocol_hash"] != canonical_hash(formal)
                or Path(cache_info["source"]).resolve() != run or read(validation_cache/"audit.json")["status"] != "pass"):
            raise ValueError("Validation cache is from a different or incomplete source study")
        prior = pd.read_csv(validation_cache/"arms.csv").set_index(["dataset","seed","method"])
    manifests = {}
    for source in set(jobs.values()):
        with (source/"data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for split in ("train", "val", "test"):
                    row[f"{split}_row_count"] = int(row[f"{split}_row_count"])
                manifests[row["dataset_id"], int(row["seed"])] = row
    out.mkdir(exist_ok=False, parents=True)
    atomic_write_json(out/"protocol.json", dict(
        source=str(run), source_protocol_hash=canonical_hash(formal), seed_splits=catalog["seeds"],
        metric="ordinary multiclass accuracy; argmax over ascending class labels; first class wins ties",
        validation="Replay the fixed mask on train only and score validation; share identical masks within a split",
        test="Score original archived probabilities from frozen train+validation refits",
        regression="MAE and fraction with absolute continuous prediction error <=1 ring; no rounding",
        prior_validation_cache=str(validation_cache) if prior is not None else None,
    ))
    arms = []
    refit_count = 0
    for (dataset, seed), source in jobs.items():
        row = manifests[dataset, seed]
        metric = row["utility"]
        expected_policy = dict(task=row["task"], seed=seed, model_params=formal["model_params"], backend=formal["backend"])
        classification = row["task"] != "regression"
        train, yt, val, yv = load_prepared_train_validation(source, row)
        if classification:
            classes = np.unique(yt)
            if not np.array_equal(classes, np.arange(len(classes))):
                raise ValueError("Archived probability columns require consecutive zero-based encoded labels")
        cached = {}
        test_labels = None
        for method in METHODS:
            path = run/f"seed-{seed}"/dataset/method
            frozen = read(path/"frozen.json")
            rank = read(path/"ranking.json")
            saved = read(path/"result.json")
            if (saved["frozen_hash"] != canonical_hash(frozen) or rank["policy"] != expected_policy
                    or rank["partition_sha256"] != {p: row[f"{p}_file_sha256"] for p in ("train", "val", "test")}):
                raise ValueError(f"Archived source changed for {dataset}/{seed}/{method}")
            winner = frozen["choices"]["vote"] if method == "PIv1" else frozen
            cols = tuple(winner["selected"])
            record = summary.loc[(dataset, seed, method)]
            if (len(cols) != record.retained or list(cols) != [
                    f for f in json.loads(row["selected_features_json"]) if f in set(cols)]):
                raise ValueError("Frozen selected features or order mismatch")
            with np.load(path/"predictions.npz", allow_pickle=False) as old:
                ye, pt = old["y"], old["selected"]
            if len(ye) != row["test_row_count"] or (test_labels is not None and not np.array_equal(ye, test_labels)):
                raise ValueError("Paired held-out labels differ")
            test_labels = ye
            res = dict(dataset=dataset, seed=seed, method=method, primary_metric=metric,
                       val_n=row["val_row_count"], test_n=len(ye), retained=len(cols),
                       frozen_hash=saved["frozen_hash"], validation_primary=winner["validation"][0],
                       test_predictions=str((path/"predictions.npz").relative_to(run)),
                       validation_accuracy=np.nan, test_accuracy=np.nan)
            if cols not in cached:
                rel = Path(f"seed-{seed}")/dataset/f"validation-{method}.npz"
                (out/rel.parent).mkdir(exist_ok=True, parents=True)
                if prior is not None and classification:
                    entry = prior.loc[(dataset,seed,method)]
                    if entry.frozen_hash != saved["frozen_hash"]:
                        raise ValueError("Cached validation mask identity differs")
                    cache_path = validation_cache/entry.validation_predictions
                    with np.load(cache_path, allow_pickle=False) as old:
                        if (list(old["columns"]) != list(cols) or not np.array_equal(old["y"], yv)
                                or not np.array_equal(old["classes"], classes)):
                            raise ValueError("Cached validation inputs differ")
                        pv = old["selected"].copy()
                    shutil.copyfile(cache_path, out/rel)
                else:
                    pv = fit_predict(train.loc[:, list(cols)], yt, val.loc[:, list(cols)], **expected_policy)
                    np.savez_compressed(out/rel, y=yv, selected=pv, columns=np.asarray(cols),
                                        **({"classes": classes} if classification else {}))
                    refit_count += 1
                cached[cols] = (pv, str(rel), method)
            pv, rel, fitted_by = cached[cols]
            res.update(validation_predictions=rel, validation_refit_by=fitted_by)
            if classification:
                va, vc, vacc = classification_metrics(yv, pv, classes, metric)
                ta, tc, tacc = classification_metrics(ye, pt, classes, metric)
                check_score(va, winner["validation"][0], f"{dataset}/{seed}/{method} frozen validation")
                res.update(validation_primary=va, test_primary=ta, validation_correct=vc, test_correct=tc,
                           validation_accuracy=vacc, test_accuracy=tacc,
                           test_majority_fraction=np.bincount(ye).max()/len(ye))
            else:
                for split, true, prediction in (("validation", yv, pv), ("test", ye, pt)):
                    if true.shape != prediction.shape:
                        raise ValueError("Unexpected regression prediction shape")
                    absolute = np.abs(true - prediction)
                    res[split + "_mae"] = absolute.mean()
                    res[split + "_within_one"] = np.mean(absolute <= 1.)
                res["validation_primary"] = compute_utility(yv, pv, metric=metric)
                check_score(res["validation_primary"], winner["validation"][0], "frozen regression validation")
                res["test_primary"] = compute_utility(ye, pt, metric=metric)
            check_score(res["validation_primary"], record.validation_utility, "formal validation utility")
            check_score(res["test_primary"], record.test_utility, "formal test utility")
            arms.append(res)
        print(f"{seed} {dataset}: frozen masks and scores verified", flush=True)
    table = pd.DataFrame(arms)
    table.to_csv(out/"arms.csv", index=False, encoding="utf-8-sig")
    paired_reports(table, out, FIELDS, "paired_accuracy.csv", "dataset_accuracy.csv")
    paired_reports(table.loc[table.primary_metric == "r2"], out, REG_FIELDS,
                   "regression_paired.csv", "regression_summary.csv")
    atomic_write_json(out/"audit.json", dict(status="pass", paired_splits=len(jobs), arms=len(arms),
                      classification_arms=int(table["test_accuracy"].count()), validation_refits=refit_count))
    print(f"Results: {out}, validation refits={refit_count}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed PIv1 versus Gain formal output")
    parser.add_argument("--validation-cache", type=Path, help="Audited classification validation replays for the same run")
    args = parser.parse_args()
    run_root = args.run.resolve()
    timestamp = datetime.now(MSK).strftime("%Y%m%d-%H%M%S-%f")
    supplement(run_root, run_root/f"accuracy-{timestamp}", args.validation_cache.resolve() if args.validation_cache else None)

