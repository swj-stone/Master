"""Post-selection audit of seed-42 feature recovery using paired held-out predictions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.test.compare_thesis_sign_aware import MATCHED, TOL, check, close, operating_point, read, verified_result
from experiments.run_thesis_sampling import DATASETS, ROOT
from minppi_mofs.utility import compute_utility, is_classification_metric


METHODS = ("local_robust", "validation_frontier", "local_pooled", "local_drop", "guarded_incumbent")
INTERVAL_METHODS = ("local_robust", "validation_frontier")
FOLD_METHODS = ("fold_consensus", "guarded_incumbent")
FORMAL = ROOT / "results/icomp2026_formal/icomp2026-formal-v1-s42-123"


def paired_interval(y, prediction, incumbent, metric, *, draws=500):
    """95% test-row percentile interval for utility(model) minus utility(incumbent)."""
    if np.array_equal(prediction, incumbent):
        return 0., 0.
    groups = ([np.flatnonzero(y == c) for c in np.unique(y)] if is_classification_metric(metric)
              else [np.arange(len(y))])
    rng = np.random.default_rng(42)
    diffs = np.empty(draws)
    for i in range(draws):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        diffs[i] = (compute_utility(y[indices], prediction[indices], metric=metric)
                    - compute_utility(y[indices], incumbent[indices], metric=metric))
    return tuple(float(v) for v in np.quantile(diffs, [.025,.975]))


def stability_summary(points, expected):
    improved = sum(p["test_delta_vs_incumbent"] >= .1 * p["margin"] - TOL for p in points)
    declined = sum(p["test_delta_vs_incumbent"] < -.1 * p["margin"] - TOL for p in points)
    lost_compression = sum(p["removed"] < p["incumbent_removed"] for p in points)
    breaches = sum(not p["within_test_margin"] for p in points)
    return dict(completed=len(points), expected=expected, meaningful_improvements=improved,
                material_declines=declined, compression_losses=lost_compression, full_margin_breaches=breaches,
                ci_lower_above_zero=sum(p.get("ci_low") is not None and p["ci_low"] > 0 for p in points),
                removed_total=sum(p["removed"] for p in points),
                success=len(points) == expected and improved >= 2 and not (declined or lost_compression or breaches))


def _rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def verify_fold_halves(y, task, split, fits):
    """Reconstruct each training half and the opposite half's exact SHAP subsample."""
    def digest(rows):
        return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()

    stratified = task != "regression"
    check(split["seed"] == 42 and split["stratified"] == stratified
          and split["index_space"] == "zero-based positions in frozen training CSV", "training split protocol")
    halves = [np.sort(values) for values in train_test_split(
        np.arange(len(y)), test_size=.5, random_state=42, stratify=y if stratified else None)]
    check(split["sizes"] == [len(v) for v in halves] and split["index_sha256"] == [digest(v) for v in halves],
          "training half sizes or hashes disagree")
    shap_rows = []
    for fit, hold in list(zip(halves, reversed(halves), strict=True))[:fits]:
        if len(hold) > 512:
            sample = np.sort(np.random.default_rng(42).choice(len(hold), 512, replace=False))
            hold = hold[sample]
        check(not np.intersect1d(fit, hold).size, "SHAP rows overlap fit rows")
        shap_rows.append(hold)
    check(split["shap_rows"] == [len(v) for v in shap_rows]
          and split["shap_index_sha256"] == [digest(v) for v in shap_rows], "opposite-half SHAP hashes disagree")


def _verify_fold_search(result, manifest):
    ds = result["dataset"]
    search = result["fold_consensus"]
    fs = result["features"]
    fits = search["ranking_fits"]
    ranks = search["fold_rankings"]
    check(fits in (0,1,2) and len(ranks) == fits == result["extra_ranking_fits"], f"{ds}: fold fit count")
    split = search["training_split"]
    if split is not None:
        train_path = FORMAL / manifest["train_relpath"]
        check(hashlib.sha256(train_path.read_bytes()).hexdigest() == result["partition_sha256"]["train"],
              f"{ds}: frozen training file")
        y = pd.read_csv(train_path, usecols=[manifest["target"]]).iloc[:,0].to_numpy()
        verify_fold_halves(y, result["policy"]["task"], split, fits)
    else:
        check(fits == 0 and not search["budget_stopped"], f"{ds}: absent fold partition")
    for record in ranks:
        gains, shap, scores = (record[k] for k in ("gain", "shap", "normalized_mean_rank"))
        check(set(gains) == set(shap) == set(scores) == set(fs), f"{ds}: rank feature coverage")
        ranks_input = pd.DataFrame({"gain":gains, "shap":shap}).loc[fs]
        check(np.isfinite(ranks_input.to_numpy()).all() and (ranks_input >= 0).all().all(),
              f"{ds}: invalid gain or SHAP importances")
        expected = ranks_input.rank(pct=True, method="average").mean(axis=1)
        check(all(close(scores[f], expected[f]) for f in fs), f"{ds}: unstable fold ranks")
    incumbent = search["incumbent_mask"]
    baseline = result["method_masks"]["guarded_incumbent"]
    check(incumbent == baseline, f"{ds}: search incumbent")
    plans = search["planned_masks"]
    trials = search["trials"]
    done = search["evaluated_count"]
    check(len(plans) == len(set(plans)) <= 5 and done == len(trials)
          and plans[:done] == [row["mask"] for row in trials], f"{ds}: evaluated plan prefix")
    if fits < 2 and split is not None:
        check(search["budget_stopped"] and search["planned_count"] is None and not plans and not trials,
              f"{ds}: partial ranking used for candidate evaluations")
    else:
        check(search["planned_count"] == len(plans) and search["budget_stopped"] == (done < len(plans)),
              f"{ds}: evaluation budget status")
    signs = result["sample_minima"]
    if fits == 2:
        first, second = (r["normalized_mean_rank"] for r in ranks)
        order = sorted((f for f in fs if signs[f] in (-1, 0)),
                       key=lambda f: (max(first[f], second[f]), (first[f]+second[f])/2))
        check(order == search["consensus_order"], f"{ds}: consensus order")
    for mask in plans:
        check(mask and mask.bit_count() <= incumbent.bit_count(), f"{ds}: empty/less compressed alternative")
        for i,f in enumerate(fs):
            if signs[f] not in (-1,0):
                check(mask & (1 << i), f"{ds}: protected candidate deletion")
            if signs[f] == -1 and not incumbent & (1 << i):
                check(not mask & (1 << i), f"{ds}: restored witness")
    base_scores = search["incumbent_scores"]
    full_scores = result["full_validation_scores"]
    margin = result["gate"]["noninferiority_margin"]
    incumbent_removed = len(fs)-incumbent.bit_count()
    scored = [(incumbent, (0., 0., incumbent_removed))]
    for row in trials:
        delta = [a-b for a,b in zip(row["scores"], base_scores, strict=True)]
        check(len(delta) == 3, f"{ds}: candidate score dimensions")
        full_pass = min(a-b for a,b in zip(row["scores"], full_scores, strict=True)) >= -margin-TOL
        nonworse = min(delta) >= -TOL
        positive = (delta[1]+delta[2])/2 > TOL
        check(row["full_gate_passed"] == full_pass and row["incumbent_gate_passed"] == nonworse
              and row["accepted"] == (full_pass and nonworse and positive), f"{ds}: candidate gate")
        if row["accepted"]:
            scored.append((row["mask"], (min(delta[1:]), delta[0], len(fs)-row["mask"].bit_count())))
    check(max(scored,key=lambda p:p[1])[0] == search["selected_mask"] == result["method_masks"]["fold_consensus"],
          f"{ds}: selected consensus mask disagrees with frozen objective")


def compare(run, historical):
    config = read(run / "protocol.json")
    policy = config["selection_policy"]
    check(config["seed"] == 42 and policy in ("local_refine", "fold_consensus")
          and 0 < config["dataset_seconds"] <= 900, "unexpected seed/policy/budget")
    fold = policy == "fold_consensus"
    methods = FOLD_METHODS if fold else METHODS
    primary = methods[0]
    interval_methods = ("fold_consensus",) if fold else INTERVAL_METHODS
    check(config["primary_method"] == primary and config["minimum_improved_datasets"] == 2
          and config["practical_change_fraction_of_margin"] == .1
          and config["paired_bootstrap_replicates"] == 500 and config["paired_bootstrap_seed"] == 42,
          "frozen analysis protocol disagrees with the audit")
    refs = {r["dataset"]: r for r in _rows(historical / "comparison/guarded_replay.csv")}
    controls = {(r["dataset"],r["method"]): r for r in _rows(historical / "comparison/operating_points.csv")
                if r["method"] in MATCHED and r["status"] == "completed"}
    manifest = {r["dataset_id"]:r for r in _rows(FORMAL / "data_manifest.csv") if r["seed"] == "42"}
    rows, statuses = [], {}
    for ds in DATASETS:
        directory = run / ds
        watch = read(directory / "watchdog.json") if (directory / "watchdog.json").is_file() else {}
        status = watch.get("status", "missing")
        charged = watch.get("elapsed_s", 0.) + watch.get("inherited_sampling_s", 0.) if watch else None
        statuses[ds] = dict(status=status, charged_s=charged, phase=watch.get("progress", {}).get("phase"))
        if status != "completed":
            rows.extend(dict(dataset=ds, method=m, status=status, charged_s=charged) for m in methods)
            continue
        check(charged <= config["dataset_seconds"] and close(watch["dataset_limit_s"],config["dataset_seconds"]),
              f"{ds}: dataset exceeded its budget")
        result = verified_result(directory, ds, manifest[ds], policy, config, watch)
        search = result["fold_consensus" if fold else "local_refinement"]
        if fold:
            _verify_fold_search(result, manifest[ds])
            statuses[ds].update(fold_evaluations=search["evaluated_count"], planned=search["planned_count"],
                                fold_budget_stopped=search["budget_stopped"], fold_ranking_fits=search["ranking_fits"])
        else:
            statuses[ds].update(local_evaluations=search["evaluated_count"], planned=search["planned_count"],
                                local_budget_stopped=search["budget_stopped"])
        check(set(result["method_masks"]) == set(methods) and result["selected_method"] == primary and
              result["selection"]["selected_mask"] == result["method_masks"][primary], f"{ds}: method identity")
        metric = result["policy"]["utility"]
        margin = result["gate"]["noninferiority_margin"]
        guarded = refs[ds]
        old = read(historical / ds / "result.json")
        for k in ("features", "partition_sha256", "policy", "validation_halves", "full_validation_scores"):
            check(result[k] == old[k], f"{ds}: mismatched {k}")
        paired = all(result[k] == old[k] for k in ("sample_minima", "witnesses", "draws_per_feature"))
        incumbent = operating_point(result, "guarded_incumbent", margin)
        parity = (incumbent["mask"] == guarded["selected_mask"] and
                  close(incumbent["test_utility"], float(guarded["selected_test"])))
        if paired:
            check(parity, f"{ds}: previous guarded mask/utility differed under identical sampling evidence")
            locked = old["confirmation"]
            prev = next(t for t in locked["trials"] if t["mask"] == int(guarded["selected_mask"], 16))
            check(prev["scores"] == result["method_results"]["guarded_incumbent"]["scores"],
                  f"{ds}: guarded validation parity")
        archive = directory / "paired_test_predictions.npz"
        check(result["paired_test_predictions"] == archive.name and
              hashlib.sha256(archive.read_bytes()).hexdigest() == result["paired_test_predictions_sha256"],
              f"{ds}: prediction archive checksum")
        target_path = FORMAL / manifest[ds]["test_relpath"]
        check(hashlib.sha256(target_path.read_bytes()).hexdigest() == result["partition_sha256"]["test"],
              f"{ds}: frozen test checksum")
        targets = pd.read_csv(target_path, usecols=[manifest[ds]["target"]]).iloc[:,0].to_numpy()
        with np.load(archive, allow_pickle=False) as values:
            check(set(values.files) == {"y", "full_set", *methods}, f"{ds}: prediction archive keys")
            y = values["y"]
            # CSV serialization of floating-point regression labels can round on readback.
            check(y.shape == targets.shape and np.allclose(y, targets, rtol=1e-12, atol=1e-12),
                  f"{ds}: frozen test target order")
            check(close(compute_utility(y, values["full_set"], metric=metric), result["full_test_utility"])
                  and close(result["full_test_utility"], float(guarded["full_test"])), f"{ds}: full-set utility")
            pred0 = values["guarded_incumbent"]
            interval_cache = {}
            predictions_by_mask = {(1 << len(result["features"])) - 1: values["full_set"]}
            incumbent_mask = result["method_masks"]["guarded_incumbent"]
            base_scores = result["method_results"]["guarded_incumbent"]["scores"]
            removed_harmful = [i for i,f in enumerate(result["features"])
                               if not (incumbent_mask & (1 << i)) and result["sample_minima"][f] == -1]
            for method in methods:
                p = operating_point(result, method, margin)
                pred = values[method]
                check(close(compute_utility(y, pred, metric=metric), p["test_utility"]), f"{ds}: {method} score")
                mask = result["method_masks"][method]
                check(np.array_equal(predictions_by_mask.setdefault(mask, pred), pred),
                      f"{ds}: inconsistent predictions for the same selected mask")
                check(p["removed_witnesses"] + p["removed_sampled_zeros"] == p["removed"],
                      f"{ds}: {method} removed protected feature")
                check(not any(mask & (1 << i) for i in removed_harmful), f"{ds}: restored a negative witness")
                selected_scores = result["method_results"][method]["scores"]
                check(all(s >= b - TOL for s,b in zip(selected_scores, base_scores, strict=True))
                      and p["removed"] >= incumbent["removed"], f"{ds}: regression on official validation/compression")
                point = dict(dataset=ds, seed=42, metric=metric, method=method, status=status, charged_s=charged,
                             paired_sampling=paired, historical_guarded_parity=parity, features=len(result["features"]),
                             margin=margin, incumbent_removed=incumbent["removed"],
                             incumbent_test=incumbent["test_utility"],
                             test_delta_vs_incumbent=p["test_utility"] - incumbent["test_utility"], **{
                                 k:v for k,v in p.items() if k != "method"})
                if method in interval_methods:
                    if p["mask"] not in interval_cache:
                        interval_cache[p["mask"]] = paired_interval(y, pred, pred0, metric)
                    point["ci_low"], point["ci_high"] = interval_cache[p["mask"]]
                for control in MATCHED:
                    baseline = controls[ds,control]
                    check(close(float(baseline["full_test"]), result["full_test_utility"]) and
                          baseline["metric"] == metric, f"{ds}: {control} reference")
                    point[f"removed_vs_{control}"] = p["removed"] - int(baseline["removed"])
                    point[f"test_delta_vs_{control}"] = p["test_utility"] - float(baseline["test_utility"])
                rows.append(point)
    summaries = {m:stability_summary([p for p in rows if p["method"] == m and p["status"] == "completed"], len(DATASETS))
                 for m in methods}
    return rows, dict(statuses=statuses, methods=summaries, paired_interval_draws=500, paired_interval_seed=42,
                      expected_datasets=list(DATASETS), primary_method=primary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("previous_confirmed_run", type=Path)
    args = parser.parse_args()
    rows, summary = compare(args.run, args.previous_confirmed_run)
    output = args.run / "comparison"
    output.mkdir(exist_ok=True)
    label = "fold" if summary["primary_method"] == "fold_consensus" else "local"
    with (output / f"paired_{label}_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    (output / f"paired_{label}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return int(any(v["status"] != "completed" for v in summary["statuses"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
