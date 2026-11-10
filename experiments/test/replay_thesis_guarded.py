"""Retrospective guarded update using six previously trained seed-42 pairs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.test.compare_thesis_sign_aware import CONTROLS, MATCHED, check, compare, read, relation
from experiments.run_thesis_sampling import DATASETS, ROOT
from minppi_mofs.experiment_protocol import canonical_hash
from minppi_mofs.guarded_update import guarded_update


def freeze_validation_decisions(run, out):
    """Only features, candidate identities and original validation triples enter the selection."""
    result = {}
    for ds in DATASETS:
        record = read(run/ds/"result.json")
        features = record["features"]
        confirmation = record["confirmation"]
        masks = confirmation["candidate_masks"]
        scores = {trial["mask"]:trial["scores"] for trial in confirmation["trials"]}
        baseline = record["full_validation_scores"]
        margin = record["gate"]["noninferiority_margin"]
        check(record["dataset"] == ds and record["seed"] == 42 and
              record["selection_policy"] == "confirmed", f"{ds}: validation provenance")
        winner = guarded_update(features, masks["structured"], masks["sign_worst"], scores, baseline, margin)
        result[ds] = dict(selected_mask=hex(winner["selected_mask"]), selected_source=winner["selected_source"],
                          candidate_masks={k:hex(v) for k,v in masks.items()},
                          official_validation={k:scores[m] for k,m in masks.items()}, full_validation=baseline,
                          validation_difference=[v-b for v,b in zip(scores[masks["sign_worst"]],
                                                                    scores[masks["structured"]], strict=True)],
                          margin=margin, feature_order_sha256=canonical_hash(features))
    rule = ROOT/"src/minppi_mofs/guarded_update.py"
    manifest = dict(design="Retrospective on previously examined seed-42 test splits", seed=42,
                    source_run=run.name, rule_sha256=hashlib.sha256(rule.read_bytes()).hexdigest(), datasets=result)
    path = out/"guarded_validation_manifest.json"
    if path.exists():
        check(read(path) == manifest, "Existing frozen guarded manifest differs")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return path


def lookup_test_outcomes(manifest_path, confirmed, signed, wide):
    """Bind a saved decision to the previously evaluated method mask and its test score."""
    manifest = read(manifest_path)
    rows, problems = compare(signed, wide, confirmed)
    check(not problems, f"Cannot compare all six paired results: {problems}")
    report = []
    for ds in DATASETS:
        decision = manifest["datasets"][ds]
        points = {r["method"]:r for r in rows if r["dataset"] == ds}
        source = {"structured":"witness_proxy_gain", "sign_worst":"sign_worst"}[decision["selected_source"]]
        point = points[source]
        check(point["mask"] == decision["selected_mask"] and point["identical_sampling_evidence"],
              f"{ds}: selected archived test mask")
        full, half_a, half_b = decision["validation_difference"]
        entry = dict(dataset=ds, source=decision["selected_source"], selected_mask=point["mask"],
                     utility_source_run=point["source_run"], metric=point["metric"], n=point["features"],
                     removed=point["removed"], removed_witnesses=point["removed_witnesses"],
                     removed_zeros=point["removed_sampled_zeros"], full_test=point["full_test"],
                     selected_test=point["test_utility"], test_delta=point["test_delta"],
                     margin=point["margin"], within_test_margin=point["within_test_margin"],
                     challenger_minus_incumbent_val=full, challenger_minus_incumbent_half_a=half_a,
                     challenger_minus_incumbent_half_b=half_b,
                     confirmed_run_charged_s_reference=points["confirmed"]["confirmed_run_charged_s"])
        for method in (*CONTROLS, *MATCHED, "sign_worst"):
            entry[f"vs_{method}"] = relation(point, points[method])
        report.append(entry)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmed", type=Path)
    parser.add_argument("sign_aware", type=Path)
    parser.add_argument("wide", type=Path)
    args = parser.parse_args()
    out = args.confirmed/"comparison"
    out.mkdir(exist_ok=True)
    manifest = freeze_validation_decisions(args.confirmed, out)
    report = lookup_test_outcomes(manifest, args.confirmed, args.sign_aware, args.wide)
    with (out/"guarded_replay.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report[0])
        writer.writeheader()
        writer.writerows(report)
    print(json.dumps(dict(datasets=len(report), new_fits=0, new_test_model_evaluations=0,
                          removed=sum(r["removed"] for r in report),
                          test_breaches=[r["dataset"] for r in report if not r["within_test_margin"]])))


if __name__ == "__main__":
    main()
