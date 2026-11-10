"""Paired test-set comparisons of prediction recipes on two frozen feature masks."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_local_refinement import FORMAL, _rows, paired_interval
from experiments.compare_thesis_sign_aware import TOL, check, close, read
from experiments.run_thesis_sampling import DATASETS
from experiments.run_thesis_predictor import preparation_budget
from minppi_mofs.experiment_protocol import canonical_hash
from minppi_mofs.utility import compute_utility


CONTRASTS = (
    ("guarded_chosen", "guarded_original"),
    ("full_chosen", "full_original"),
    ("guarded_blend", "guarded_original"),
    ("full_blend", "full_original"),
    ("guarded_blend", "full_blend"),
    ("guarded_original", "full_original"),
)
PRIMARY = "guarded_chosen_vs_guarded_original"


def validate_decision(decision, historical, mask):
    original = np.asarray(decision["original_scores"], dtype=float)
    blend = np.asarray(decision["blend_scores"], dtype=float)
    check(original.shape == blend.shape == (3,) and np.isfinite(blend).all(), "invalid validation triples")
    check(decision["mask"] == mask and np.allclose(original, historical, atol=1e-10, rtol=0),
          "original validation/mask differs from pre-frozen result")
    delta = blend-original
    check(np.allclose(delta, decision["validation_deltas"], atol=TOL, rtol=0), "incorrect validation deltas")
    expected = bool(np.min(delta) >= -TOL and np.mean(delta[1:]) > TOL)
    check(decision["adopt_blend"] is expected, "validation adoption rule mismatch")
    return "blend" if expected else "original"


def summarize(rows, expected):
    finished = [r for r in rows if r["status"] == "completed"]
    gains = [r["dataset"] for r in finished if r["delta"] >= .1*r["margin"]-TOL]
    losses = [r["dataset"] for r in finished if r["delta"] < -.1*r["margin"]-TOL]
    breaches = [r["dataset"] for r in finished if r["new_breach"]]
    return dict(completed=len(finished), meaningful_improvements=gains, material_declines=losses,
                new_original_full_margin_breaches=breaches,
                below_original_full_margin=[r["dataset"] for r in finished if not r["within_original_full_margin"]],
                positive_paired_intervals=[r["dataset"] for r in finished if r["ci_low"] > 0],
                negative_paired_intervals=[r["dataset"] for r in finished if r["ci_high"] < 0],
                success=len(finished) == expected and len(gains) >= 2 and not losses and not breaches)


def recipe_parameters(recipe, policy):
    if recipe == "column_blend":
        return {"colsample_bynode": .5}, 2
    check(recipe == "rounds", "unknown predictor recipe")
    rounds = policy["model_params"]["n_estimators"]
    return dict(predictor_recipe="rounds", original_rounds=rounds,
                blend_checkpoints=[math.ceil(1.5*rounds),2*rounds]), 1


def verify_record(directory, watch, manifest, historical, guarded_row, frozen, digest, recipe):
    ds = manifest["dataset_id"]
    d = read(directory / "result.json")
    check(watch["result"] == d and watch["exit_code"] == 0 and
          (d["phase"], d["dataset"], d["seed"]) == ("complete", ds, 42), f"{ds}: invalid completed worker")
    fs = json.loads(manifest["selected_features_json"])
    check(d["features"] == fs and d["policy"] == historical["policy"] and
          d["validation_halves"] == historical["validation_halves"], f"{ds}: feature/model/validation provenance")
    check(d["partition_sha256"] == {p:manifest[f"{p}_file_sha256"] for p in ("train","val","test")},
          f"{ds}: frozen partition hashes")
    masks = {"full": hex((1 << len(fs))-1), "guarded": frozen["selected_mask"]}
    check(d["guarded_mask"] == guarded_row["selected_mask"] == masks["guarded"] and
          canonical_hash(fs) == frozen["feature_order_sha256"] and d["mask_manifest_sha256"] == digest,
          f"{ds}: frozen mask evidence")
    expected_recipes = {}
    for role, mask in masks.items():
        old_scores = frozen["full_validation"] if role == "full" else frozen["official_validation"][frozen["selected_source"]]
        expected_recipes[role] = validate_decision(d["decisions"][role], old_scores, mask)
    params, per_mask_fits = recipe_parameters(recipe, d["policy"])
    expected_params = dict(mask_manifest_sha256=digest, masks=masks, seed=42, **params,
                           blend_weights=[.5,.5], adopt_blend={r:s == "blend" for r,s in expected_recipes.items()})
    expected_job = canonical_hash(dict(dataset=ds, seed=42, protocol=manifest["protocol_hash"], parameters=expected_params))
    events = read(directory/"test_access.json")["events"]
    check(len(events) == 1 and (events[0]["job_id"], events[0]["dataset_id"], events[0]["decision"], events[0]["reason"]) ==
          (expected_job, ds, "allowed", "final_removal_evaluation"), f"{ds}: test access/decision binding")
    check(d["parameters"] == expected_params and d["job_id"] == expected_job
          and d["test_training_partitions"] == ["train","val"], f"{ds}: fitted recipe/partitions")
    check(d["validation_fits"] == d["test_fits"] == per_mask_fits*len(set(masks.values())), f"{ds}: distinct model fit count")
    keys = {f"{r}_{m}" for r in masks for m in ("original","blend","chosen")}
    check(set(d["method_results"]) == keys, f"{ds}: method keys")
    archive = directory / "paired_test_predictions.npz"
    check(d["paired_test_predictions"] == archive.name and
          hashlib.sha256(archive.read_bytes()).hexdigest() == d["paired_test_predictions_sha256"], f"{ds}: paired archive hash")
    test_path = FORMAL / manifest["test_relpath"]
    check(hashlib.sha256(test_path.read_bytes()).hexdigest() == manifest["test_file_sha256"], f"{ds}: on-disk test hash")
    target = pd.read_csv(test_path, usecols=[manifest["target"]]).iloc[:,0].to_numpy()
    with np.load(archive,allow_pickle=False) as packet:
        check(set(packet.files) == keys | {"y"}, f"{ds}: prediction keys")
        y = packet["y"]
        check(y.shape == target.shape and np.allclose(y,target,rtol=1e-12,atol=1e-12), f"{ds}: test labels/order")
        predictions = {key:packet[key] for key in keys}
    metric = manifest["utility"]
    test_scores, identities = {}, {}
    for key in sorted(keys):
        role, label = key.split("_")
        recipe = expected_recipes[role] if label == "chosen" else label
        record = d["method_results"][key]
        check(record["mask"] == masks[role] and record["recipe"] == recipe and
              record["scores"] == d["decisions"][role][recipe+"_scores"], f"{ds}/{key}: recorded recipe")
        identity = (masks[role], recipe)
        pred = predictions[key]
        check(np.array_equal(predictions[identities.setdefault(identity, key)], pred) and np.isfinite(pred).all(),
              f"{ds}/{key}: inconsistent predictions for same model")
        test_scores[key] = float(compute_utility(y, pred, metric=metric))
        check(close(test_scores[key], record["test_utility"]), f"{ds}/{key}: held-out utility")
    check(close(float(guarded_row["selected_test"]), test_scores["guarded_original"]) and
          close(float(guarded_row["full_test"]), test_scores["full_original"]), f"{ds}: original test parity")
    return d, y, predictions, test_scores, identities


def compare(run, prior):
    config = read(run/"protocol.json")
    check(config["seed"] == config["paired_bootstrap_seed"] == 42 and config["primary_method"] == "guarded_chosen"
          and config["practical_change_fraction_of_margin"] == .1 and config["paired_bootstrap_replicates"] == 500
          and 0 < config["dataset_seconds"] <= 900, "unexpected predictor analysis protocol")
    frozen_file = prior/"comparison/guarded_validation_manifest.json"
    frozen = read(frozen_file)
    digest = hashlib.sha256(frozen_file.read_bytes()).hexdigest()
    recipe = config.get("predictor_recipe", "column_blend")
    check(config["mask_source"] == frozen["source_run"] == prior.name and digest == config["mask_manifest_sha256"],
          "mask source and protocol hash mismatch")
    log_lines = (prior/"run.log").read_text(encoding="utf-8").splitlines()
    authority = {r["dataset_id"]:r for r in _rows(FORMAL/"data_manifest.csv") if r["seed"] == "42"}
    previous = {r["dataset"]:r for r in _rows(prior/"comparison/guarded_replay.csv")}
    rows, statuses = [], {}
    for ds in DATASETS:
        directory = run/ds
        watch = read(directory/"watchdog.json") if (directory/"watchdog.json").is_file() else {}
        status = watch.get("status", "missing")
        charged = watch.get("total_charged_s")
        meta = dict(dataset=ds, metric=authority[ds]["utility"], predictor_recipe=recipe, status=status, charged_s=charged)
        statuses[ds] = dict(status=status, charged_s=charged, phase=watch.get("progress",{}).get("phase"))
        if status != "completed":
            rows.extend(dict(**meta, contrast=a+"_vs_"+b) for a,b in CONTRASTS)
            continue
        base_charge, basis = preparation_budget(ds, read(prior/ds/"watchdog.json"), log_lines)
        check(ds in config["datasets"] and config["preparation"][ds]["basis"] == basis and
              close(base_charge, config["preparation"][ds]["charged_s"]) and close(base_charge, watch["inherited_preparation_s"])
              and 0 <= charged <= config["dataset_seconds"] and watch["dataset_limit_s"] == config["dataset_seconds"]
              and close(charged, base_charge + watch["elapsed_s"]), f"{ds}: charged deadline")
        guarded = previous[ds]
        d,y,pred,scores,ids = verify_record(directory, watch, authority[ds], read(prior/ds/"result.json"),
                                           guarded, frozen["datasets"][ds], digest, recipe)
        margin = float(guarded["margin"])
        cache = {}
        for new, old in CONTRASTS:
            rn, ro = (d["method_results"][k] for k in (new,old))
            pair = (ids[rn["mask"],rn["recipe"]], ids[ro["mask"],ro["recipe"]])
            if pair not in cache:
                cache[pair] = paired_interval(y, pred[new], pred[old], meta["metric"])
            ci_low,ci_high = cache[pair]
            full_original = scores["full_original"]
            within = scores[new] >= full_original - margin - TOL
            rows.append(dict(**meta, contrast=new+"_vs_"+old, method=new, comparator=old, feature_count=len(d["features"]),
                             removed=len(d["features"])-int(rn["mask"],16).bit_count(), mask=rn["mask"], comparator_mask=ro["mask"],
                             recipe=rn["recipe"], comparator_recipe=ro["recipe"], test=scores[new], comparator_test=scores[old],
                             delta=scores[new]-scores[old], ci_low=ci_low, ci_high=ci_high, margin=margin,
                             test_delta_vs_original_full=scores[new]-full_original, within_original_full_margin=within,
                             within_comparator_margin=scores[new] >= scores[old]-margin-TOL,
                             new_breach=not within and scores[old]>=full_original-margin-TOL,
                             inherited_preparation_s=watch["inherited_preparation_s"], predictor_wall_s=watch["elapsed_s"]))
    contrasts = {a+"_vs_"+b:summarize([r for r in rows if r["contrast"] == a+"_vs_"+b],len(DATASETS))
                 for a,b in CONTRASTS}
    return rows, dict(primary=PRIMARY, predictor_recipe=recipe, primary_success=contrasts[PRIMARY]["success"], statuses=statuses,
                      contrasts=contrasts, bootstrap_replicates=500, bootstrap_seed=42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("previous_confirmed_run",type=Path)
    args = parser.parse_args()
    rows, summary = compare(args.run,args.previous_confirmed_run)
    out = args.run/"comparison"
    out.mkdir(exist_ok=True)
    with (out/"paired_predictor_results.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    (out/"paired_predictor_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False))
    return int(any(v["status"] != "completed" for v in summary["statuses"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
