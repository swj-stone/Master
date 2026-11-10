"""Summaries and within-seed marginal effects for a single nested K=1,2,3 run."""

from collections import defaultdict
import csv
import math
import os
from statistics import mean, stdev


KS = (1, 2, 3)
SCORES = ("guarded_original_validation", "guarded_chosen_validation", "guarded_original_test", "guarded_chosen_test")
POINT_METRICS = (
    "n_features", "removed_count", "compression_fraction", "nominees", "witnesses",
    "sampled_draws", "sample_fits", "sampling_s", "sampling_cache_hits_from_selection",
    "checkpoint_selection_s", "checkpoint_selection_fits", "checkpoint_test_s", "checkpoint_test_fits",
    *SCORES, "full_original_test", "full_chosen_test", "selection_delta", "predictor_delta",
    "net_delta", "matched_full_delta",
)
POINT_FIELDS = ("dataset", "seed", "k", "status", "metric", "margin", *POINT_METRICS,
                "planned_trials", "resolved_trials", "sampling_stop", "guarded_recipe", "within_full_margin")
PAIR_METRICS = (
    "removed_delta", "compression_fraction_delta", "removal_set_symmetric_difference", "removal_set_jaccard",
    "new_negative_witnesses", "additional_draws", "additional_sample_fits", "additional_sampling_s",
    "additional_selection_cache_reuses", "checkpoint_selection_s", "checkpoint_test_s",
    *(name + "_delta" for name in SCORES),
)
PAIR_FIELDS = ("dataset", "seed", "from_k", "to_k", "status", "metric", "margin", *PAIR_METRICS,
               "guarded_chosen_test_gain_per_sampling_minute", "within_full_margin", "new_full_margin_breach")
TOL = 1e-12


def _point(record):
    scores = record["method_test_utilities"]
    guard = record["predictor_decisions"]["guarded"]
    recipe = "blend" if guard["adopt_blend"] else "original"
    removed = record["removed_count"]
    n = record["n_features"]
    if n <= 0 or removed != len(set(record["removed_features"])) or not 0 <= removed < n:
        raise ValueError("Invalid checkpoint feature counts")
    net = scores["guarded_chosen"] - scores["full_original"]
    return dict(metric=record["policy"]["utility"], margin=record["margin"], n_features=n,
                removed_count=removed, compression_fraction=removed/n, nominees=record["proposed_count"],
                witnesses=record["witness_count"], sampled_draws=record["sampled_draws"],
                sample_fits=record["sample_fits"], sampling_s=record["sample_elapsed_s"],
                sampling_cache_hits_from_selection=record["sampling_cache_hits_from_selection"],
                checkpoint_selection_s=record["checkpoint_selection_s"], checkpoint_selection_fits=record["checkpoint_selection_fits"],
                checkpoint_test_s=record["checkpoint_test_s"], checkpoint_test_fits=record["checkpoint_test_fits"],
                planned_trials=record["planned_trials"],
                resolved_trials=record["resolved_trials"], sampling_stop=record["sampling_stop"],
                guarded_recipe=recipe, guarded_original_validation=guard["original_scores"][0],
                guarded_chosen_validation=guard[recipe + "_scores"][0],
                guarded_original_test=scores["guarded_original"], guarded_chosen_test=scores["guarded_chosen"],
                full_original_test=scores["full_original"], full_chosen_test=scores["full_chosen"],
                selection_delta=scores["guarded_original"] - scores["full_original"],
                predictor_delta=scores["guarded_chosen"] - scores["guarded_original"],
                net_delta=net, matched_full_delta=scores["guarded_chosen"] - scores["full_chosen"],
                within_full_margin=net >= -record["margin"] - TOL)


def _pair(before, after, old, new):
    if any(before[key] != after[key] for key in ("n_features", "margin", "metric", "full_original_test", "full_chosen_test")):
        raise ValueError("Inconsistent reference for adjacent K values")
    old_set, new_set = set(old["removed_features"]), set(new["removed_features"])
    old_neg, new_neg = set(old["witnesses"]), set(new["witnesses"])
    if not old_neg <= new_neg:
        raise ValueError("A negative witness disappeared in a later prefix")
    elapsed = after["sampling_s"] - before["sampling_s"]
    draws = after["sampled_draws"] - before["sampled_draws"]
    fits = after["sample_fits"] - before["sample_fits"]
    if min(elapsed, draws, fits) < 0:
        raise ValueError("Nested sampling costs must be cumulative")
    union = old_set | new_set
    diffs = {key + "_delta": after[key] - before[key] for key in SCORES}
    return dict(metric=after["metric"], margin=after["margin"], removed_delta=len(new_set)-len(old_set),
                compression_fraction_delta=after["compression_fraction"] - before["compression_fraction"],
                removal_set_symmetric_difference=len(old_set ^ new_set),
                removal_set_jaccard=len(old_set & new_set)/len(union) if union else 1.,
                new_negative_witnesses=len(new_neg-old_neg), additional_draws=draws,
                additional_sample_fits=fits, additional_sampling_s=elapsed,
                additional_selection_cache_reuses=after["sampling_cache_hits_from_selection"] - before["sampling_cache_hits_from_selection"],
                checkpoint_selection_s=after["checkpoint_selection_s"], checkpoint_test_s=after["checkpoint_test_s"], **diffs,
                guarded_chosen_test_gain_per_sampling_minute=diffs["guarded_chosen_test_delta"]*60/elapsed if elapsed else None,
                within_full_margin=after["within_full_margin"],
                new_full_margin_breach=before["within_full_margin"] and not after["within_full_margin"])


def _stats(record, name, values):
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite metric {name}")
    record[name + "_mean"] = mean(values) if values else None
    record[name + "_sd"] = stdev(values) if len(values) >= 2 else None


def _aggregate(rows, metrics):
    done = [row for row in rows if row["status"] == "completed"]
    if len({(r["metric"], r["margin"]) for r in done}) > 1:
        raise ValueError("Cannot aggregate different metrics or margins for one dataset")
    report = dict(metric=done[0]["metric"] if done else None, margin=done[0]["margin"] if done else None,
                  expected=len(rows), completed=len(done), all_completed=len(rows) == len(done),
                  completed_seeds=",".join(str(r["seed"]) for r in done),
                  pending=sum(r["status"] == "pending" for r in rows),
                  interrupted=sum(r["status"] == "interrupted" for r in rows),
                  errors=sum(r["status"] == "error" for r in rows),
                  full_margin_breaches=sum(not r["within_full_margin"] for r in done))
    for metric in metrics:
        _stats(report, metric, [r[metric] for r in done])
    return report, done


def _write_csv(path, rows, fields=None):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def update_k_reports(run_dir, expected_jobs, outcomes):
    """Refresh four atomic CSVs. Only jobs with all three final held-out results enter means."""
    points, pairs = [], []
    point_groups, pair_groups = defaultdict(list), defaultdict(list)
    curvature = defaultdict(lambda: defaultdict(list))
    for ds, seed in expected_jobs:
        outcome = outcomes.get((ds, seed), {})
        status = outcome.get("status", "pending")
        completed = status == "completed"
        records = outcome["result"]["checkpoints"] if completed else {}
        if completed and set(records) != {"1", "2", "3"}:
            raise ValueError(f"Incomplete nested result for {ds}/{seed}")
        by_k, pair_by_k = {}, {}
        for k in KS:
            row = dict(dataset=ds, seed=seed, k=k, status=status)
            if completed:
                record = records[str(k)]
                if (record["dataset"], record["seed"], record["k"], record["phase"]) != (ds, seed, k, "complete"):
                    raise ValueError(f"Checkpoint identity mismatch for {ds}/{seed}/K={k}")
                row.update(_point(record))
            by_k[k] = row
            points.append(row)
            point_groups[ds,k].append(row)
        for a,b in zip(KS[:-1], KS[1:], strict=True):
            row = dict(dataset=ds, seed=seed, from_k=a, to_k=b, status=status)
            if completed:
                row.update(_pair(by_k[a], by_k[b], records[str(a)], records[str(b)]))
            pairs.append(row)
            pair_groups[ds,a,b].append(row)
            pair_by_k[b] = row
        if completed:
            for key in (*SCORES, "compression_fraction"):
                curvature[ds][key].append(pair_by_k[3][key + "_delta"] - pair_by_k[2][key + "_delta"])

    grouped = []
    for (ds, k), group in point_groups.items():
        aggregate, _ = _aggregate(group, POINT_METRICS)
        grouped.append(dict(dataset=ds, k=k, **aggregate))
    margins = []
    for (ds,a,b), group in pair_groups.items():
        agg, done = _aggregate(group, PAIR_METRICS)
        gains = sum(r["guarded_chosen_test_delta"] >= .1*r["margin"]-TOL for r in done)
        losses = sum(r["guarded_chosen_test_delta"] < -.1*r["margin"]-TOL for r in done)
        raw_gains = sum(r["guarded_chosen_test_delta"] > TOL for r in done)
        raw_losses = sum(r["guarded_chosen_test_delta"] < -TOL for r in done)
        sampling_minutes = sum(r["additional_sampling_s"] for r in done) / 60
        gain = sum(r["guarded_chosen_test_delta"] for r in done)
        report = dict(dataset=ds, from_k=a, to_k=b, **agg,
                      test_gains=raw_gains, test_losses=raw_losses, test_ties=len(done)-raw_gains-raw_losses,
                      pooled_test_gain_per_sampling_minute=gain/sampling_minutes if sampling_minutes else None,
                      test_practical_gains=gains,
                      test_practical_losses=losses, test_practical_ties=len(done)-gains-losses,
                      unchanged_masks=sum(r["removal_set_symmetric_difference"] == 0 for r in done),
                      new_full_margin_breaches=sum(r["new_full_margin_breach"] for r in done),
                      second_difference_n=len(done) if b == 3 else None)
        for key in (*SCORES, "compression_fraction"):
            _stats(report, "second_difference_" + key, curvature[ds][key] if b == 3 else [])
        margins.append(report)
    _write_csv(run_dir / "summary.csv", points, POINT_FIELDS)
    _write_csv(run_dir / "dataset_summary.csv", grouped)
    _write_csv(run_dir / "marginal.csv", pairs, PAIR_FIELDS)
    _write_csv(run_dir / "marginal_summary.csv", margins)
