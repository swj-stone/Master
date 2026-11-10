"""Per-dataset summaries over the planned seed matrix of a single thesis run."""

import csv
import os
from statistics import mean, stdev


METRICS = (
    "removed_count", "compression_fraction", "full_original_test", "guarded_original_test",
    "full_chosen_test", "guarded_chosen_test", "selection_delta", "predictor_delta",
    "net_delta", "matched_full_delta", "wall_s",
)


def write_dataset_summary(output_path, rows, expected_jobs):
    """Aggregate completed seeds; retain timeout/error/pending counts and spent time."""
    plan = {}
    for dataset, seed in expected_jobs:
        plan.setdefault(dataset, []).append(seed)
    by_job = {(row["dataset"], row["seed"]): row for row in rows}
    summaries = []
    for dataset, seeds in plan.items():
        attempted = [by_job[dataset, s] for s in seeds if (dataset, s) in by_job]
        complete = [row for row in attempted if row["status"] == "completed"]
        record = dict(dataset=dataset, metric=complete[0].get("metric") if complete else None,
                      expected=len(seeds), completed=len(complete), pending=len(seeds) - len(attempted),
                      timed_out=sum(r["status"] == "timed_out" for r in attempted),
                      errors=sum(r["status"] == "error" for r in attempted),
                      interrupted=sum(r["status"] == "interrupted" for r in attempted),
                      all_completed=len(complete) == len(seeds),
                      completed_seeds=",".join(str(r["seed"]) for r in complete),
                      guarded_extended_count=sum(r.get("guarded_recipe") == "extended_average" for r in complete),
                      attempted_wall_s=sum(r["wall_s"] for r in attempted))
        for metric in METRICS:
            values = [r[metric] for r in complete if r.get(metric) is not None]
            record[f"{metric}_mean"] = mean(values) if values else None
            record[f"{metric}_sd"] = stdev(values) if len(values) > 1 else None
        summaries.append(record)
    path = output_path / "dataset_summary.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    os.replace(temp, path)
