"""Describe raw paired score costs alongside reduction and end-to-end time."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def historical():
    cases=pd.read_csv(ROOT/"results/minpi-study/residual-126-corrected/summary/paired-cases.csv")
    rows=[]
    for (cohort, task),sub in cases.groupby(["cohort","task"]):
        for arm in ("full","F","random","marginal"):
            ratio=sub["minpi_to_"+arm+"_ratio"]
            gap=sub.minpi_score-sub[arm+"_score"]
            faster=sub.reduced & (ratio<1)
            rows.append(dict(cohort=cohort,task=task,comparator=arm,n=len(sub),
                reduced=int(sub.reduced.sum()),faster=int(faster.sum()),within_002_and_faster=int((faster&(gap>=-.02-1e-12)).sum()),
                score_up_and_faster=int((faster&(gap>1e-12)).sum()),median_retained=np.median(sub.width/sub.p),
                median_time_ratio=ratio.median(),median_score_gap=gap.median(),
                median_score_gap_when_faster=gap[faster].median(),worst_score_gap_when_faster=gap[faster].min()))
    return pd.DataFrame(rows)


if __name__=="__main__":
    path=ROOT/"research/four_stage/historical-tradeoffs.csv"
    historical().to_csv(path,index=False)
    print(historical().to_string(index=False))
