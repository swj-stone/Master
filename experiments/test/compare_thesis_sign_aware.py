"""Audit frozen seed-42 decisions and compare test outcomes at the selected subset sizes."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.run_thesis_sampling import DATASETS, ROOT
from minppi_mofs.experiment_protocol import canonical_hash


VARIANTS = ("sign_worst", "sign_pooled")
CONTROLS = ("witness_proxy_gain", "gain_only", "treeshap_only")
MATCHED = ("matched_gain", "matched_treeshap")
TOL = 1e-12


def read(path):
    return json.loads(path.read_bytes())


def check(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=TOL)


def relation(a, b):
    """Pareto comparison of achieved removal counts and test utilities (higher is better)."""
    count = a["removed"] - b["removed"]
    score = a["test_utility"] - b["test_utility"]
    if count == 0 and abs(score) <= TOL:
        return "equivalent"
    if count >= 0 and score >= -TOL:
        return "dominates"
    if count <= 0 and score <= TOL:
        return "dominated"
    return "tradeoff"


def operating_point(result, method, margin):
    features = result["features"]
    r = result["method_results"][method]
    mask = result["method_masks"][method]
    check(0 < mask < 1 << len(features) and r["mask"] == mask, f"{method}: mask")
    removed = [f for i, f in enumerate(features) if not mask & (1 << i)]
    check(len(removed) == r["removed_count"] and set(removed) == set(r["removed_features"]),
          f"{method}: removal identities")
    h = sum(result["sample_minima"][f] == -1 for f in removed)
    z = sum(result["sample_minima"][f] == 0 for f in removed)
    if method in (*VARIANTS, "confirmed", "witness_proxy_gain"):
        check(h + z == len(removed), f"{method}: removal of protected feature")
    deltas = [a - b for a, b in zip(r["scores"], result["full_validation_scores"], strict=True)]
    check(len(deltas) == 3 and min(deltas) >= -margin-TOL, f"{method}: failed validation gate")
    test = r["test_utility"]
    delta = test - result["full_test_utility"]
    check(close(test, result["method_test_utilities"][method]) and close(delta, r["test_delta"]),
          f"{method}: test delta")
    return dict(method=method, removed=len(removed), fraction=len(removed)/len(features), mask=hex(mask),
                selected_source=r.get("selected_source", r.get("source")),
                removed_witnesses=h, removed_sampled_zeros=z, validation_gate_passed=True,
                pooled_validation_delta=deltas[0], worst_half_validation_delta=min(deltas[1:]),
                full_test=result["full_test_utility"], test_utility=test, test_delta=delta,
                within_test_margin=delta >= -margin-TOL)


def verified_result(directory, dataset, manifest, policy, protocol, watchdog):
    d = read(directory/"result.json")
    check(watchdog["exit_code"] == 0 and watchdog["result"] == d, f"{dataset}: watchdog result")
    check(d["phase"] == "complete" and (d["dataset"], d["seed"], d["selection_policy"]) ==
          (dataset, 42, policy), f"{dataset}: run identity")
    fs = d["features"]
    check(fs == json.loads(manifest["selected_features_json"]) and d["n_features"] == len(fs),
          f"{dataset}: feature manifest")
    check(d["partition_sha256"] == {p: manifest[f"{p}_file_sha256"] for p in ("train","val","test")},
          f"{dataset}: split hashes")
    selection = d["selection"]
    mask = selection["selected_mask"]
    check(d["removed_count"] == len(fs)-mask.bit_count() and d["selected_gate_passed"], f"{dataset}: final mask")
    check(d["test_training_partitions"] == ["train", "val"], f"{dataset}: refit partitions")
    check(d["policy"]["utility"] == manifest["utility"] and d["policy"]["task"] == manifest["task"],
          f"{dataset}: manifest utility")
    check(d["requested_rounds"] == protocol["rounds"] and d["sample_seconds"] == protocol["sample_seconds"],
          f"{dataset}: sampling protocol")
    params = dict(rounds=d["requested_rounds"], sample_seconds=d["sample_seconds"], retained_mask=hex(mask),
                  selection_policy=policy, method_masks=d["method_masks"])
    job_id = canonical_hash(dict(dataset=dataset, seed=42, protocol=manifest["protocol_hash"], parameters=params))
    event, = read(directory/"test_access.json")["events"]
    check(event["job_id"] == job_id and event["decision"] == "allowed" and
          event["reason"] == "final_removal_evaluation", f"{dataset}: held-out access/mask binding")
    return d


def audit_confirmation(result, source, authority):
    record = result["confirmation"]
    ds = result["dataset"]
    candidates = record["candidate_masks"]
    check(set(candidates) == {"structured", "sign_worst"}, f"{ds}: confirmation candidates")
    full = (1 << len(result["features"])) - 1
    trials = {t["mask"]:t for t in record["trials"]}
    check(len(trials) == result["confirmation_fits"] == len(record["trials"]) and
          set(trials) == {full, *candidates.values()}, f"{ds}: confirmation fits")
    check(set(result["method_masks"]) == {"confirmed"} and
          result["method_masks"]["confirmed"] == record["selected_mask"] == result["selection"]["selected_mask"],
          f"{ds}: confirmation decision")
    if record["selected_mask"] == full:
        check(close(result["retained_test_utility"], result["full_test_utility"]), f"{ds}: full-set fallback")
    reference = result["full_validation_scores"]
    margin = result["gate"]["noninferiority_margin"]
    for mask, trial in trials.items():
        official = [v-r for v,r in zip(trial["scores"], reference, strict=True)]
        extra = trial["confirmation_score"] - trials[full]["confirmation_score"]
        check(len(official) == 3 and all(close(a,b) for a,b in zip(official, trial["official_deltas"], strict=True))
              and close(extra, trial["confirmation_delta"]) and trial["official_gate_passed"] ==
              (min(official) >= -margin-TOL) and trial["confirmation_gate_passed"] == (extra >= -margin-TOL)
              and trial["accepted"] == (min(*official,extra) >= -margin-TOL), f"{ds}: confirmation gates")
        check(trial["removed_count"] == len(result["features"]) - mask.bit_count(), f"{ds}: confirmation size")
        check(close(trial["confirmation_score"], result["confirmation_scores"][hex(mask)]),
              f"{ds}: training refit scores")
    selected = max((t for t in record["trials"] if t["accepted"]),
                   key=lambda t: (t["removed_count"], t["confirmation_delta"], min(t["official_deltas"][1:])))
    check(record["selected_mask"] == selected["mask"] and record["selected_source"] == selected["sources"][0],
          f"{ds}: confirmation selection rule")
    targets = pd.read_csv(source/authority["train_relpath"], usecols=[authority["target"]]).iloc[:,0].to_numpy()
    stratify = targets if result["policy"]["task"] != "regression" else None
    partitions = train_test_split(np.arange(len(targets)), test_size=.25, random_state=42, stratify=stratify)
    split = record["split"]
    hashes = [hashlib.sha256(np.sort(part).astype("<i8").tobytes()).hexdigest() for part in partitions]
    check(split["index_sha256"] == hashes and split["sizes"] == list(map(len, partitions)) and split["seed"] == 42
          and split["stratified"] == (stratify is not None), f"{ds}: training confirmation fold")
    return dict(confirmation_delta=selected["confirmation_delta"], confirmation_gate_passed=selected["accepted"],
                confirmation_source=record["selected_source"], confirmation_fits=len(trials))


def compare(run, reference, confirmed=None):
    """Return all six datasets, including an explicit status for each absent or failed run."""
    new_config, old_config = (read(p/"protocol.json") for p in (run, reference))
    configs = [(reference, old_config, "wide", CONTROLS), (run, new_config, "sign_aware", (*VARIANTS, *MATCHED))]
    if confirmed is not None:
        configs.append((confirmed, read(confirmed/"protocol.json"), "confirmed", ("confirmed",)))
    for _, config, policy, _ in configs:
        check(config["seed"] == 42 and config["selection_policy"] == policy and
              0 < config["dataset_seconds"] <= 900, "Unexpected comparison protocol")
    source = ROOT/"results/icomp2026_formal/icomp2026-formal-v1-s42-123"
    with (source/"data_manifest.csv").open(encoding="utf-8-sig") as stream:
        manifest = {r["dataset_id"]:r for r in csv.DictReader(stream) if r["seed"] == "42"}
    rows, problems = [], {}
    for ds in DATASETS:
        authority = manifest[ds]
        base = dict(dataset=ds, seed=42, metric=authority["utility"], features=int(authority["selected_feature_count"]))
        for part in ("train", "val", "test"):
            digest = hashlib.sha256((source/authority[f"{part}_relpath"]).read_bytes()).hexdigest()
            check(digest == authority[f"{part}_file_sha256"], f"{ds}: on-disk {part} partition")
        pair = {}
        for root, config, policy, methods in configs:
            directory = root/ds
            watch_path = directory/"watchdog.json"
            watch = read(watch_path) if watch_path.exists() else {}
            status = watch.get("status", "missing")
            charged = watch["elapsed_s"] + watch["inherited_sampling_s"] if watch else None
            meta = dict(**base, status=status, source_run=root.name,
                        new_joint_run_charged_s=charged if policy == "sign_aware" else None)
            if policy == "confirmed":
                meta["confirmed_run_charged_s"] = charged
            if status != "completed":
                problems[f"{ds}/{policy}"] = status
                rows.extend(dict(**meta, method=m) for m in methods)
                continue
            check(ds in config["datasets"] and charged <= config["dataset_seconds"] and
                  close(watch["dataset_limit_s"], config["dataset_seconds"]), f"{ds}: charged budget")
            result = verified_result(directory, ds, authority, policy, config, watch)
            pair[policy] = result
            margin = result["gate"]["noninferiority_margin"]
            signs = Counter(result["sample_minima"].values())
            meta.update(sampled_negative=signs[-1], sampled_zero=signs[0], sampled_positive=signs[1],
                        unsampled=signs[None])
            if policy == "sign_aware":
                check(result["selected_method"] == "sign_worst" and result["selection"]["selected_mask"] ==
                      result["method_masks"]["sign_worst"], f"{ds}: predeclared primary method")
                check(all(result["sample_minima"][f] in (-1, 0) for f in result["proposed_features"]),
                      f"{ds}: protected nomination")
                candidates = set()
                for trial in result["selection_trials"]:
                    if trial["is_sign_candidate"]:
                        candidates.update(f for i, f in enumerate(result["features"]) if not trial["mask"] & (1 << i))
                check(candidates == set(result["proposed_features"]), f"{ds}: examined nomination union")
            if policy == "confirmed":
                meta.update(audit_confirmation(result, source, authority))
            rows.extend(dict(**meta, margin=margin, **operating_point(result, m, margin)) for m in methods)
        for name, new in pair.items():
            if name == "wide" or "wide" not in pair:
                continue
            old = pair["wide"]
            for key in ("features", "partition_sha256", "policy", "source_run_id", "validation_halves",
                        "full_validation_scores", "full_test_utility", "alpha"):
                check(old[key] == new[key], f"{ds}: reference mismatch for {key}")
            check(close(old["gate"]["noninferiority_margin"], new["gate"]["noninferiority_margin"]),
                  f"{ds}: margin mismatch")
            if old["sampling_source_sha256"] and old["sampling_source_sha256"] == new["sampling_source_sha256"]:
                for key in ("witnesses", "sample_minima", "draws_per_feature"):
                    check(old[key] == new[key], f"{ds}: reused sample mismatch for {key}")
        points = {r["method"]:r for r in rows if r["dataset"] == ds and r["status"] == "completed"}
        sample_fields = ("witnesses", "sample_minima", "draws_per_feature")
        identical = len(pair) == len(configs) and all(d[k] == pair["wide"][k] for d in pair.values()
                                                       for k in sample_fields)
        for p in points.values():
            p["identical_sampling_evidence"] = identical
        if "confirmed" in pair and identical:
            rec = pair["confirmed"]["confirmation"]
            for name, policy, method in (("structured", "wide", "witness_proxy_gain"),
                                         ("sign_worst", "sign_aware", "sign_worst")):
                candidate = rec["candidate_masks"][name]
                check(candidate == pair[policy]["method_masks"][method], f"{ds}: candidate parity for {name}")
                trial = next(t for t in rec["trials"] if t["mask"] == candidate)
                check(trial["scores"] == pair[policy]["method_results"][method]["scores"],
                      f"{ds}: candidate official scores for {name}")
                if rec["selected_mask"] == candidate:
                    check(close(points["confirmed"]["test_utility"], points[method]["test_utility"]),
                          f"{ds}: held-out parity for {name}")
        for variant in (*VARIANTS, "confirmed"):
            if variant in points:
                comparators = (*CONTROLS, *MATCHED, "sign_worst") if variant == "confirmed" else (*CONTROLS, *MATCHED)
                for control in comparators:
                    if control in points:
                        points[variant][f"vs_{control}"] = relation(points[variant], points[control])
    return rows, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("wide_reference", type=Path)
    parser.add_argument("--confirmed", type=Path, help="Optional second experiment with a train-only confirmation fold.")
    args = parser.parse_args()
    rows, problems = compare(args.run, args.wide_reference, args.confirmed)
    out = (args.confirmed or args.run)/"comparison"
    out.mkdir(exist_ok=True)
    with (out/"operating_points.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(complete=not problems, incomplete_datasets=problems, methods={})
    if not problems:
        for method in dict.fromkeys(r["method"] for r in rows):
            points = [r for r in rows if r["method"] == method]
            removed, n = sum(p["removed"] for p in points), sum(p["features"] for p in points)
            summary["methods"][method] = dict(removed=removed, features=n, fraction=removed/n,
                worst_test_delta=min(p["test_delta"] for p in points),
                test_breaches=[p["dataset"] for p in points if not p["within_test_margin"]],
                relations={k[3:]:dict(Counter(p[k] for p in points)) for k in points[0] if k.startswith("vs_")})
    encoded = json.dumps(summary, indent=2, allow_nan=False)
    (out/"summary.json").write_text(encoded+"\n", encoding="utf-8")
    print(encoded)
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
