"""Recalculate the paired timing claims from independent run receipts."""
import json

import numpy as np
import pandas as pd

from experiments.context_study.data import ROOT, config, digest

OUT = ROOT/"results/context_review/timing-process"


def main():
    protocol = json.loads((OUT/"protocol.json").read_bytes())
    for name, expected in protocol.items():
        if digest(ROOT/name) != expected:
            raise ValueError(f"Changed timing source: {name}")
    rows = []
    for dataset in config()["datasets"]:
        for fold in range(3):
            payload = json.loads((OUT/dataset/f"fold-{fold}"/"completed.json").read_bytes())
            assert payload["protocol"] == digest(OUT/"protocol.json")
            rows.extend(payload["records"])
    data = pd.DataFrame(rows)
    assert len(data)==378 and not data.duplicated(["dataset","fold","repeat","arm"]).any()
    keys = ["dataset","fold","arm"]
    assert data.groupby(keys).position.nunique().eq(3).all()
    assert data.groupby(keys).vote_hash.nunique().eq(1).all()
    assert data.groupby(["dataset","fold"]).apply(
        lambda g:len(set(g[g.arm!="fixed"].vote_hash))==1,
        include_groups=False).all()
    assert data.logical_cpu.nunique()==1
    for meter in ["cpu","wall"]:
        stages = [stage+"_"+meter for stage in ["prep","fit","anchor","delete","refit"]]
        np.testing.assert_allclose(data[stages].sum(axis=1),data["total_"+meter],atol=1e-8,rtol=0)
    numeric = [n for n in data if n.endswith(("_wall","_cpu"))]+[
        "masks","dispatches","full_scores","quick_veto","verified_panels"]
    folds = data.groupby(keys)[numeric].median()
    tasks = folds.groupby(["dataset","arm"]).mean()
    rows = []
    for metric in ["total_wall","total_cpu","delete_wall","delete_cpu"]:
        t = tasks[metric].unstack("arm")
        for a,b in [("recent","fixed"),("recent_lazy","fixed"),("recent_lazy","recent")]:
            delta = t[a]-t[b]
            ratio = (t[a]/t[b]).where((t[a]>0)&(t[b]>0)).dropna()
            rows.append(dict(metric=metric,method=a,control=b,ratio_of_means=t[a].mean()/t[b].mean(),
                             ratio_eligible_tasks=len(ratio),median_ratio=ratio.median(),mean_delta_seconds=delta.mean(),
                             median_delta_seconds=delta.median(),faster_tasks=int((delta<0).sum()),
                             fastest_ratio=ratio.min(),slowest_ratio=ratio.max()))
    pd.DataFrame(rows).to_csv(OUT/"paired-comparisons.csv",index=False)
    work = folds.groupby("arm")[["masks","dispatches","full_scores","quick_veto","verified_panels"]].sum()
    work.to_csv(OUT/"one-pass-counts.csv")
    gains = folds.masks.unstack()
    saved = (gains.fixed-gains.recent).groupby("dataset").sum()
    saved.to_csv(OUT/"unique-mask-savings.csv",header=["fixed_minus_recent"])
    overall = data.groupby(keys)[["total_wall","total_cpu"]]
    variation = (overall.max()/overall.min()).groupby("arm").median()
    variation.to_csv(OUT/"within-fold-range-ratios.csv")
    print(pd.DataFrame(rows).to_string(index=False))
    print(work.to_string())
    print(variation.to_string())


if __name__ == "__main__":
    main()
