"""Reduce paired trials and retain all losses as well as successes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cohort","dataset","fold","seed"]
MAIN = ("minpi","full","F","random","marginal")


def quality_table(quality):
    rows = pd.read_csv(quality/"scores.csv")
    starts = rows[rows.method=="minpi"].set_index(KEY)
    table = starts[["task","family","train","test","p","width"]].copy()
    scores = rows.pivot(index=KEY,columns="method",values="score")
    table = table.join(scores[list(MAIN)].add_suffix("_score"))
    draws = [s for s in scores if s.startswith("random_")]
    table["median_random_score"] = scores[draws].median(axis=1)
    table["random_draws_beaten"] = scores[draws].lt(scores["minpi"] - 1e-12, axis=0).sum(axis=1)
    table["reduced"] = table.width < table.p
    assert len(table)==126 and len(draws)==5
    return table


def summarize(quality, costs, output):
    cases = quality_table(quality)
    if costs:
        assert len(costs)==2
        timings=[]
        for i,directory in enumerate(costs,1):
            complete = json.loads((directory/"complete.json").read_text(encoding="utf-8"))
            assert complete["stage"]=="cost" and complete["pass_id"]==i
            assert (directory/"protocol.json").read_bytes()==(quality/"protocol.json").read_bytes()
            raw = pd.read_csv(directory/"times.csv")
            assert len(raw)==126*5*5
            group = raw.groupby(KEY+["method"])
            assert (group.size()==5).all() and (group.position.nunique()==5).all() and (group.repeat.nunique()==5).all()
            timings.append(group[["select_ms","total_ms"]].median().unstack("method"))
        for arm in MAIN:
            cases[arm+"_total_ms"] = np.maximum(timings[0]["total_ms",arm],timings[1]["total_ms",arm])
            cases[arm+"_select_ms"] = np.maximum(timings[0]["select_ms",arm],timings[1]["select_ms",arm])
            if arm != "full":
                cases[arm+"_to_full_ratio"] = np.maximum(*[t["total_ms",arm]/t["total_ms","full"] for t in timings])
            if arm != "minpi":
                cases["minpi_to_"+arm+"_ratio"] = np.maximum(*[t["total_ms","minpi"]/t["total_ms",arm] for t in timings])
    cases=cases.reset_index()
    output.mkdir(exist_ok=False,parents=True)
    cases.to_csv(output/"paired-cases.csv",index=False)
    totals=[]
    for grain in (["cohort"], ["cohort","task"],["cohort","dataset"]):
        for key, group in cases.groupby(grain):
            labels=dict(zip(grain,key,strict=True))
            for arm in ["minpi","F","random","marginal"]:
                score=group[arm+"_score"]
                row=dict(grain="+".join(grain), **labels, method=arm, cases=len(group),
                  reduced=int(group.reduced.sum()), median_retained=(group.width/group.p).median(),
                  worst_gap=(score-group.full_score).min())
                for alternative in ("full","F","random","marginal","median_random"):
                    delta=score-group[alternative+"_score"]
                    row[alternative+"_wins"]=int((delta>1e-12).sum())
                    row[alternative+"_ties"]=int((delta.abs()<=1e-12).sum())
                    row[alternative+"_losses"]=int((delta< -1e-12).sum())
                for tol in (0,.005,.01,.02):
                    passes=group.reduced & (score>=group.full_score-tol-1e-12)
                    row[f"quality_{tol}"]=int(passes.sum())
                    if costs:
                        row[f"joint_{tol}"]=int((passes & (group[arm+"_to_full_ratio"]<1)).sum())
                if costs:
                    row["median_time_ratio_full"]=group[arm+"_to_full_ratio"].median()
                    if arm=="minpi":
                        for alternative in ("F","random","marginal"):
                            relative=group["minpi_to_"+alternative+"_ratio"]
                            row[alternative+"_faster"]=int((relative<1).sum())
                            row[alternative+"_pareto"]=int((group.reduced & (relative<1)
                                 & (score>=group[alternative+"_score"]-1e-12)).sum())
                totals.append(row)
    pd.DataFrame(totals).to_csv(output/"summary.csv",index=False)
    print(pd.DataFrame(totals).query("grain=='cohort'").to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quality",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--costs",type=Path,nargs=2)
    args=parser.parse_args()
    summarize(args.quality,args.costs,args.output)
