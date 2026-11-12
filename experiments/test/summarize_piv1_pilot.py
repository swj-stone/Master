"""Audit locked PIv1 pilot predictions and report paired utility, compression, and cost."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

from experiments.thesis_prepared_catalog import audit_prepared_catalog
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


TIE = 1e-14
ROLES = ("piv1", "anchor", "piv0", "gain")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def score(metric, y, predictions):
    if metric == "r2":
        return max(0., r2_score(y, predictions))
    if metric == "roc_auc":
        return roc_auc_score(y, predictions[:, 1])
    if metric == "ovr_auc":
        return roc_auc_score(y, predictions, multi_class="ovr", average="macro")
    raise ValueError(f"Unsupported metric: {metric}")


def records(folder):
    rank, frozen, result, timer = (
        read(folder / name) for name in ("ranking.json", "frozen.json", "result.json", "watchdog.json")
    )
    require(timer["status"] == "completed" and timer["exit_code"] == 0
            and timer["kill_requested_s"] is None and timer["elapsed_s"] > 0, f"Incomplete: {folder}")
    require(canonical_hash(frozen) == result["frozen_hash"] and timer["result"] == result, f"Freeze: {folder}")
    return rank, frozen, result, timer


def verify_pilot_search(folder, final):
    rank, frozen, result, timer = records(folder)
    require(frozen["ranking_hash"] == canonical_hash(rank), f"Ranking: {folder}")
    require(timer["limit_s"] == 900 and timer["elapsed_s"] < 900, f"Timeout: {folder}")
    evaluated = read(folder / "validation.json")
    scores = {tuple(r["indices"]): r["scores"] for r in evaluated}
    feature_count = len(rank["features"])
    require(len(scores) == len(evaluated) and tuple(range(feature_count)) in scores, f"Mask cache: {folder}")
    require(result["fits"] == len(scores) + int(final), f"Fit count: {folder}")
    require(frozen["choices"] == result["choices"], f"Reported choices: {folder}")
    for recipe, chosen in frozen["choices"].items():
        trials = read(folder / f"trials-{recipe}.json")
        masks = [tuple(t["indices"]) for t in trials]
        require(len(set(masks)) == len(trials) == chosen["checked"] <= chosen["budget"], f"Budget: {folder}/{recipe}")
        for mask, t in zip(masks, trials, strict=True):
            require(mask == tuple(sorted(set(mask))) and mask and mask in scores
                    and abs(t["utility"] - scores[mask][0]) <= TIE, f"Mask utility: {folder}")
        maximum = max(t["utility"] for t in trials)
        best = min((t for t in trials if t["utility"] >= maximum-TIE),
                   key=lambda t: (len(t["indices"]), t["indices"]))
        indices = tuple(best["indices"])
        require(list(indices) == chosen["indices"] and chosen["source"] == best["source"]
                and chosen["retained"] == len(indices) and chosen["selected"] == [rank["features"][i] for i in indices]
                and scores[indices] == chosen["validation"], f"Selected mask: {folder}")
    if not final:
        require(result["test_utility"] is None and not (folder/"test_access.json").exists()
                and not (folder/"predictions.npz").exists(), f"Held-out artifacts in development: {folder}")
    return rank, frozen, result, timer


def check_development(run, protocol, lock):
    rows, choices = [], {}
    weak, strong = "vote", "vote_strong"
    rank_totals = {weak: 0., strong: 0.}
    for dataset in protocol["datasets"]:
        folder = run / "development" / f"seed-{protocol['development_seed']}" / dataset / "shared"
        _, frozen, _, _ = verify_pilot_search(folder, False)
        require(lock["decisions"][dataset] == canonical_hash(frozen), f"Locked development source: {dataset}")
        choices[dataset] = frozen["choices"]
        weak_score, strong_score = (choices[dataset][v]["validation"][0] for v in (weak, strong))
        delta = weak_score - strong_score
        rank_totals[weak] += 1.5 if abs(delta) <= TIE else (1 if delta > 0 else 2)
        rank_totals[strong] += 1.5 if abs(delta) <= TIE else (1 if delta < 0 else 2)
        for name, v in choices[dataset].items():
            rows.append(dict(dataset=dataset, seed=frozen["seed"], metric=frozen["metric"], recipe=name,
                             validation=v["validation"][0], retained=v["retained"], candidates=v["checked"],
                             budget=v["budget"], winner_source=v["source"],
                             delta_vs_anchor=v["validation"][0] - choices[dataset]["anchor"]["validation"][0]))
    expected = strong if rank_totals[strong] < rank_totals[weak] else weak
    require(lock["ranks"] == rank_totals and expected == lock["primary_recipe"], "Global recipe selection differs")
    return choices, pd.DataFrame(rows)


def verify_racing(run, lock):
    if not (run/"racing_replay.csv").exists():
        return {}
    decision_path = run/"racing-lock.json"
    decision = read(decision_path) if decision_path.exists() else None
    if decision is not None:
        require(decision["primary"] == lock["primary_recipe"] and decision["test_access_before_lock"] == 0,
                "Racing decision and global recipe disagree")
        for name, recorded in decision["proposal_sha256"].items():
            require(digest(run/name) == recorded, f"Racing proposal or curve changed: {name}")
    results = pd.read_csv(run/"racing_replay.csv")
    reports = {}
    if decision is not None:
        require(set(results.variant) == set(decision["rejected_racing"]), "Incomplete racing variants")
    for variant, group in results.groupby("variant"):
        failures = set()
        for row in group.itertuples():
            arm = run/"development"/f"seed-{row.seed}"/row.dataset/"shared"
            trials = read(arm/"trials-vote.json")
            require(len(trials) == row.budget and row.saved == row.budget-row.checked
                    and 0 < row.checked <= row.budget, f"Racing budget: {variant}/{row.dataset}")
            prefix = trials[:row.checked]
            high = max(t["utility"] for t in prefix)
            winner = min((t for t in prefix if t["utility"] >= high-TIE),
                         key=lambda t: (len(t["indices"]),t["indices"]))
            require(winner["indices"] == json.loads(row.selected_indices) and abs(winner["utility"]-row.val)<TIE
                    and abs(row.val-row.full_val-row.delta)<TIE, f"Racing winner: {variant}/{row.dataset}")
            if row.delta < (-.005 if row.metric == "r2" else -.001)-TIE:
                failures.add(row.dataset)
        reports[variant] = dict(total=int(group.budget.sum()), checked=int(group.checked.sum()),
                                outside_margin=sorted(failures), verified_lock=decision is not None)
        if decision is not None:
            recorded = decision["rejected_racing"][variant]
            require(reports[variant]["total"] == recorded["total"] and reports[variant]["checked"] == recorded["checked"]
                    and failures == {v["dataset"] for v in recorded["outside_margin"]}, f"Racing replay: {variant}")
    return reports


def verify_baseline(folder, direct):
    rank, frozen, result, timer = records(folder)
    require(digest(folder/"ranking.json") == frozen["ranking_sha256"]
            and digest(folder/"validation_curve.csv") == frozen["curve_sha256"], f"Prior source: {folder}")
    width = len(rank["feature_order"])
    retained = result["retained"]
    negatives = len(rank["negative"])
    curve = pd.read_csv(folder/"validation_curve.csv")
    require(len(curve) == width-negatives and result["fits"] == len(curve)+1+int(negatives > 0)
            and len(frozen["selected"]) == retained and (not direct or negatives == 0), f"Prior fits: {folder}")
    require(set(frozen["selected"]) == set(rank["descending_gain"][:retained]), f"Prior selected features: {folder}")
    return rank, frozen, result, timer


def verify_heldout(folder, y, metric, result):
    audit = read(folder/"test_access.json")["events"]
    require(len(audit) == 1 and audit[0]["decision"] == "allowed"
            and audit[0]["job_id"] == result["frozen_hash"], f"Test access: {folder}")
    with np.load(folder/"predictions.npz", allow_pickle=False) as file:
        require(np.array_equal(y, file["y"]), f"Test labels: {folder}")
        observed = score(metric, y, file["selected"])
    require(abs(observed-result["test_utility"]) < 1e-12, f"Test utility: {folder}")
    return float(observed)


def run_audit(run):
    protocol, lock = read(run/"protocol.json"), read(run/"lock.json")
    baseline = Path(protocol["prior_run"])
    jobs, catalog = audit_prepared_catalog()
    require(protocol["catalog_hash"] == catalog["sha256"] == read(baseline/"protocol.json")["catalog_sha256"],
            "Dataset catalog identity differs")
    require(canonical_hash(protocol) == lock["protocol_hash"], "Pilot protocol differs from global recipe lock")
    for name, expected in protocol["source_hashes"].items():
        require(digest(run/"source"/name) == expected, f"Archived PIv1 source hash: {name}")
    chosen = lock["primary_recipe"]
    dev_choices, ablation = check_development(run, protocol, lock)
    racing = verify_racing(run, lock)
    min_check = []
    for dataset, choices in dev_choices.items():
        for recipe in ("vote", "vote_strong"):
            voted, averaged = choices[recipe], choices["mean_"+recipe]
            min_check.append(dict(dataset=dataset, seed=protocol["development_seed"], recipe=recipe,
                primary=recipe == chosen, min_validation=voted["validation"][0], mean_validation=averaged["validation"][0],
                delta_min_minus_mean=voted["validation"][0]-averaged["validation"][0],
                min_budget=voted["budget"], mean_budget=averaged["budget"],
                min_retained=voted["retained"], mean_retained=averaged["retained"],
                overlap=len(set(voted["selected"]) & set(averaged["selected"]))))
    manifest = {}
    for source in set(jobs.values()):
        with (source/"data_manifest.csv").open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                manifest[row["dataset_id"], int(row["seed"])] = row
    rows, long_rows = [], []
    for seed in lock["seeds"]:
        for dataset in protocol["datasets"]:
            src = jobs[dataset, seed]
            m = manifest[dataset, seed]
            features = json.loads(m["selected_features_json"])
            partition = {p: m[f"{p}_file_sha256"] for p in ("train", "val", "test")}
            for part, h in partition.items():
                require(digest(src/m[f"{part}_relpath"]) == h, f"Partition checksum: {dataset}/{seed}/{part}")
            y = pd.read_csv(src/m["test_relpath"], usecols=[m["target"]])[m["target"]].to_numpy()
            expected_policy = dict(backend=catalog["shared"]["backend"], seed=seed, task=m["task"],
                                   model_params=catalog["shared"]["model_params"])
            pair = dict(dataset=dataset, seed=seed, metric=m["utility"], width=len(features))
            selected = {}
            role_rows = []
            halves = None
            for role, recipe in zip(ROLES, (chosen, "anchor", "PIv0", "direct_gain"), strict=True):
                pilot = role in ("piv1", "anchor")
                parent = run/"final" if pilot else baseline
                folder = parent/f"seed-{seed}"/dataset/recipe
                if pilot:
                    rank, frozen, result, timer = verify_pilot_search(folder, True)
                    require(set(frozen["choices"]) == {recipe}, f"Unexpected final recipe: {folder}")
                    choice = frozen["choices"][recipe]
                    require((rank["votes"] is None) == (role == "anchor"), f"Unexpected voting state: {folder}")
                    if seed == protocol["development_seed"]:
                        require(choice == dev_choices[dataset][recipe], f"Development replay mismatch: {folder}")
                    columns = rank["features"]
                    selected[role] = choice["selected"]
                    val = choice["validation"][0]
                    source = choice["source"]
                    candidates = choice["checked"]
                else:
                    rank, frozen, result, timer = verify_baseline(folder, role == "gain")
                    columns = rank["feature_order"]
                    selected[role] = frozen["selected"]
                    source = recipe + "_prefix"
                    val = result["validation_utility"]
                    candidates = result["candidates"]
                require(rank["partition_sha256"] == partition and columns == features
                        and rank["policy"] == expected_policy and rank["metric"] == m["utility"]
                        and rank["dataset"] == dataset and rank["seed"] == seed, f"Paired input identity: {folder}")
                if halves is None:
                    halves = rank["halves"]
                require(halves == rank["halves"], f"Validation halves: {folder}")
                info = dict(test=verify_heldout(folder, y, m["utility"], result), retained=len(selected[role]),
                            validation=val, source=source, candidates=candidates, fits=result["fits"],
                            seconds=timer["elapsed_s"], voting_seconds=result["voting_s"])
                pair.update({f"{role}_{k}":v for k,v in info.items()})
                role_rows.append(dict(dataset=dataset, seed=seed, method=role, metric=m["utility"], **info))
            for role in ROLES[1:]:
                pair[f"piv1_minus_{role}"] = pair["piv1_test"] - pair[f"{role}_test"]
                pair[f"piv1_seconds_saved_vs_{role}"] = pair[f"{role}_seconds"] - pair["piv1_seconds"]
            pair["anchor_minus_gain"] = pair["anchor_test"] - pair["gain_test"]
            pair["anchor_seconds_saved_vs_gain"] = pair["gain_seconds"] - pair["anchor_seconds"]
            pair["overlap_with_anchor"] = len(set(selected["piv1"]) & set(selected["anchor"]))
            rows.append(pair)
            long_rows.extend(role_rows)
    require(len(rows) == 12 and len(long_rows) == 48, "Require 12 pairs across two seeds and four workflows")
    pairs = pd.DataFrame(rows)
    means = pd.DataFrame(long_rows).groupby(["dataset", "metric", "method"], as_index=False).agg(
        seeds=("seed", "count"), mean_test=("test", "mean"), mean_retained=("retained", "mean"),
        mean_candidates=("candidates", "mean"), mean_fits=("fits", "mean"), mean_seconds=("seconds", "mean"))
    totals = []
    for seed, frame in [*pairs.groupby("seed"), ("all", pairs)]:
        for role in ROLES[1:]:
            delta = frame[f"piv1_minus_{role}"]
            t1, tc = frame.piv1_seconds.sum(), frame[f"{role}_seconds"].sum()
            totals.append(dict(seed=seed, comparator=role, datasets=len(frame), wins=(delta>TIE).sum(),
                               ties=(delta.abs()<=TIE).sum(), losses=(delta< -TIE).sum(),
                               faster=(frame.piv1_seconds < frame[f"{role}_seconds"]).sum(),
                               piv1_fits=frame.piv1_fits.sum(), comparator_fits=frame[f"{role}_fits"].sum(),
                               piv1_seconds=t1, comparator_seconds=tc, percent_time_saved=100*(tc-t1)/tc))
    for file, frame in (("paired_comparison.csv", pairs), ("dataset_comparison.csv", means),
                        ("by_seed.csv", pd.DataFrame(totals)), ("development_ablations.csv", ablation),
                        ("min_vs_mean_validation.csv", pd.DataFrame(min_check))):
        frame.to_csv(run/file, encoding="utf-8-sig", index=False)
    atomic_write_json(run/"independent_audit.json", dict(status="pass", primary_recipe=chosen,
        source_hashes=protocol["source_hashes"], seeds=lock["seeds"], paired_splits=len(rows),
        development_validation_only=len(dev_choices), cold_final_arms=24, archived_baseline_arms=24,
        verified_partition_hashes=36, recomputed_heldout_scores=len(long_rows), development_ranks=lock["ranks"],
        racing_replays=racing))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    checked = run_audit(run)
    print(f"Verified {len(checked)} four-way pairs. Reports: {run}")
