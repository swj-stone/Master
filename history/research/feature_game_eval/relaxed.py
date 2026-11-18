"""Retain all observed score losses when only smaller width and time are required."""
from pathlib import Path
import argparse
import pandas as pd


def relaxed(history,confirmation,output):
    history=pd.read_csv(history)
    confirmation=pd.read_csv(confirmation)
    groups=list(history.groupby("cohort"))
    groups += [(f"new_{mech}_{task}",g) for (mech,task),g in confirmation.groupby(["mechanism","task"])]
    rows=[]
    for name,g in groups:
        for baseline in ["full","F","random","marginal","ridge_weights","representatives"]:
            ratio="minpi_to_"+baseline+"_ratio" if name in ("original","extra") else "minpi_relative_"+baseline
            if ratio not in g:
                continue
            reduced=g.width<g.p
            passed=reduced & (g[ratio]<1)
            gap=g.minpi_score-g[baseline+"_score"]
            rows.append(dict(cohort=name,baseline=baseline,cases=len(g),true_reduction=reduced.sum(),
                             reduced_and_faster=passed.sum(),accuracy_restricted=(passed & (gap>=-.02-1e-12)).sum(),
                             gap_median_when_faster=gap[passed].median(),worst_gap_when_faster=gap[passed].min(),
                             ratio_median_when_faster=g[ratio][passed].median()))
    df=pd.DataFrame(rows)
    df.to_csv(output,index=False,mode="x")
    print(df[df.baseline=="full"].to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("history","confirmation","output"):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    relaxed(args.history,args.confirmation,args.output)
