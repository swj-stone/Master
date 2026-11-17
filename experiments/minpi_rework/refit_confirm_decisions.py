"""Describe failures of locked policies within the observed splits and sources."""
from pathlib import Path

import numpy as np
import pandas as pd


EVIDENCE = Path(__file__).resolve().parents[2] / "thesis-paper/evidence/minpi_rework/refit-outage-confirmation"


def paired_metrics(data, method, reference):
    metrics = ["k", "deployment_s", "accuracy", "auc", "r2", "scaled_mae", "worst"]
    candidate = data[data.method == method].set_index(["source", "seed"])[metrics]
    control = data[data.method == reference].set_index(["source", "seed"])[metrics]
    assert candidate.index.equals(control.index)
    result = candidate - control
    for field in ["k", "deployment_s"]:
        result[field] = np.log(candidate[field] / control[field])
    return result


def quality(delta, task):
    if task == "classification":
        return (delta.auc >= -.01) & (delta.accuracy >= -.01)
    return (delta.r2 >= -.02) & (delta.scaled_mae <= .02)


def decisions(delta, full, task, workflow):
    matched = quality(delta, task)
    if workflow == "outage":
        return {"miss_matched_quality": ~matched, "miss_joint": ~(matched & (delta.worst < 0))}
    full_ok = quality(full, task)
    efficient = (delta.k <= np.log(.90)) | ((delta.k <= 0) & (delta.deployment_s <= np.log(.90)))
    return {"miss_matched_quality": ~matched, "miss_full_quality": ~full_ok,
            "miss_joint": ~(matched & full_ok & efficient)}


def summarize(path=EVIDENCE):
    data = pd.read_csv(path / "heldout_scores.csv")
    main = data[data.weight == "rows"]
    rows = []
    for panel, panel_data in [*main.groupby("panel"), ("pooled", main)]:
        for (workflow, task), group in panel_data.groupby(["workflow", "task"]):
            primary = workflow == "refit"
            group = group[group.policy == ("adaptive" if primary else "fixed_half")]
            method = "width" if primary else "exchange_min"
            ref = "greedy" if primary else "exchange_mean"
            delta = paired_metrics(group, method, ref)
            full = paired_metrics(group, method, "full")
            source_summaries = decisions(delta.groupby(level=0).mean(), full.groupby(level=0).mean(),
                                         task, workflow)
            for metric, failures in decisions(delta, full, task, workflow).items():
                fractions = failures.groupby(level=0).mean().to_numpy()
                rng = np.random.default_rng(377)
                boot = rng.choice(fractions, size=(10000, len(fractions))).mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                failed_sources = source_summaries[metric]
                rows.append(dict(panel=panel, workflow=workflow, task=task, metric=metric,
                                 split_failures=int(failures.sum()), splits=len(failures),
                                 split_failure_rate=fractions.mean(), lo=lo, hi=hi,
                                 source_failures=int(failed_sources.sum()), sources=len(fractions),
                                 sources_passing=";".join(failed_sources.index[~failed_sources])))
    result = pd.DataFrame(rows)
    result.to_csv(path / "heldout_decisions.csv", index=False)


if __name__ == "__main__":
    summarize()
