"""Paired exploratory effects, with equal weight for each source."""
import argparse

import numpy as np
import pandas as pd

from experiments.minpi_compression import benchmark as b


def effects(frame, arms, controls):
    rows = []
    for (original,task), group in frame.groupby(["original","task"]):
        for arm in arms:
            for control in controls:
                if arm == control:
                    continue
                scores = {}
                for metric in ["k","selection_s","deployment_s","auc","accuracy","r2","scaled_mae"]:
                    wide = group.pivot(index=["dataset","seed"],columns="method",values=metric)
                    if not {arm,control} <= set(wide):
                        continue
                    wide = wide[[arm,control]].dropna()
                    if not len(wide):
                        continue
                    ratio = metric in ["k","selection_s","deployment_s"]
                    if ratio and control == "full" and metric == "selection_s":
                        continue
                    d = np.log(wide[arm]/wide[control]) if ratio else wide[arm]-wide[control]
                    avg = d.groupby(level="dataset").mean().mean()
                    scores[metric] = np.exp(avg) if ratio else avg
                if scores:
                    rows.append(dict(original=original,task=task,method=arm,control=control,**scores))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",nargs="+",required=True)
    parser.add_argument("--arms",nargs="+",default=["integral","signed","signed_gcv"])
    parser.add_argument("--controls",nargs="+",default=["full","coefficient","pooled_coefficient","full_marginal","rfe","gain"])
    parser.add_argument("--csv")
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(b.ROOT/"results/minpi_rework"/run/"scores.csv") for run in args.runs])
    summary = effects(data,args.arms,args.controls)
    if args.csv:
        summary.to_csv(args.csv,index=False)
    print(summary.round(4).to_string(index=False))
