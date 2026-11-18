"""Derive paired counts for each learner and budget from the raw crossed panel."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize_speed_tradeoffs(cases, folder):
    tasks = {
        "balanced_accuracy": ("digits", "wine", "breast_cancer", "spambase", "banknote", "madelon"),
        "r2": ("diabetes", "abalone", "airfoil", "crimedata"),
    }
    metric_by_source = {source: metric for metric, sources in tasks.items() for source in sources}
    if set(cases.source) != set(metric_by_source):
        raise ValueError("The ten-source panel has changed. Verify its score units.")
    records = []
    for (name, metric), group in cases.groupby(["candidate", cases.source.map(metric_by_source)]):
        faster = group[group.faster_reduced]
        record = {
            "candidate": name,
            "metric": metric,
            "cases": len(group),
            "reduced_and_faster_full": len(faster),
            "also_within_002": int(faster.joint.sum()),
            "median_score_gap_when_faster": faster.gap.median(),
            "worst_score_gap_when_faster": faster.gap.min(),
            "median_time_ratio_when_faster": faster.time_ratio.median(),
        }
        for comparator in ("F", "random", "PFI"):
            record["also_better_and_faster_" + comparator] = int(faster["beats_" + comparator].sum())
        records.append(record)
    pd.DataFrame(records).to_csv(folder / "speed-tradeoffs.csv", index=False)


def summarize(folder):
    raw=pd.read_csv(folder/"outcomes.csv")
    keys=["source","seed","model"]
    if len(raw)!=1080 or raw.duplicated(keys+["arm"]).any() or raw.score.isna().any():
        raise ValueError("Expected all 1080 scored deployments")
    base=raw[raw.arm=="full"].set_index(keys)
    records=[]
    for candidate,suffix in (("minpi80","80"),("minpi_auto","_match")):
        selected=raw[raw.arm==candidate].set_index(keys).sort_index()
        paired=pd.DataFrame(index=selected.index)
        paired["k"]=selected.k
        paired["p"]=selected.p
        paired["minpi"]=selected.score
        paired["full"]=base.score
        paired["gap"]=selected.score-base.score
        paired["time_ratio"]=selected.total_ms/base.total_ms
        paired["faster_reduced"]=(selected.k<selected.p)&(paired.time_ratio<1)
        paired["joint"]=(selected.k<selected.p)&(paired.gap>=-.02-1e-12)&(paired.time_ratio<1)
        for name in ("F","random","PFI"):
            other=raw[raw.arm==name+suffix].set_index(keys).sort_index()
            assert selected.k.equals(other.k)
            gap=selected.score-other.score
            paired["gap_"+name]=gap
            paired["time_ratio_"+name]=selected.total_ms/other.total_ms
            paired["wins_"+name]=gap>1e-12
            paired["beats_"+name]=paired["wins_"+name]&(selected.total_ms<other.total_ms)
        paired["candidate"]=candidate
        records.append(paired.reset_index())
    cases=pd.concat(records,ignore_index=True)
    cases.to_csv(folder/"paired.csv",index=False)
    summarize_speed_tradeoffs(cases, folder)
    groups=cases.groupby(["candidate","model"])
    table=groups.agg(cases=("joint","size"),reduced=("k",lambda col:int((col<cases.loc[col.index,"p"]).sum())),
                     faster_no_score_cap=("faster_reduced","sum"),worst_gap=("gap","min"),
                     quality=("gap",lambda col:int((col>=-.02-1e-12).sum())),joint=("joint","sum"),
                     median_gap=("gap","median"),median_ratio=("time_ratio","median"),
                     wins_F=("wins_F","sum"),wins_random=("wins_random","sum"),wins_PFI=("wins_PFI","sum"),
                     both_F=("beats_F","sum"),both_random=("beats_random","sum"),both_PFI=("beats_PFI","sum"))
    table.to_csv(folder/"summary.csv")
    print(table.to_string())
    print("MLP cap hits per method",raw.loc[raw.model.str.startswith("mlp")].groupby("arm").converged.apply(lambda x:(~x).sum()).to_dict())


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    summarize(parser.parse_args().folder)
