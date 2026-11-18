"""Descriptive paired comparisons of the five measured pipelines."""
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence"
ARMS = ("current", "F", "random", "full")
TOL = 1e-12


def performance(frame, arm):
    gap, ratio = frame[f"gap_{arm}"], frame[f"ratio_{arm}"]
    cut = frame.k_pair < frame.p
    faster = ratio < 1
    faster_cut = cut & faster
    return {
        "n": len(frame), "cut": cut.sum(), "changed": frame.cols_pair.ne(frame[f"cols_{arm}"]).sum(),
        "score_up": (gap > TOL).sum(), "score_tie": (gap.abs() <= TOL).sum(),
        "score_down": (gap < -TOL).sum(), "faster_both": faster.sum(),
        "cut_faster": faster_cut.sum(),
        "strict_joint": (faster_cut & (gap > TOL)).sum(),
        "no_loss_joint": (faster_cut & (gap >= -TOL)).sum(),
        "loss_001_joint": (faster_cut & (gap >= -.01-TOL)).sum(),
        "loss_002_joint": (faster_cut & (gap >= -.02-TOL)).sum(),
        "loss_005_joint": (faster_cut & (gap >= -.05-TOL)).sum(),
        "median_worst_total_ratio": ratio.median(),
        "median_worst_selection_ratio": np.maximum(
            frame.select_ms1_pair / frame[f"select_ms1_{arm}"],
            frame.select_ms2_pair / frame[f"select_ms2_{arm}"],
        ).median(),
    }


def main():
    d = pd.read_csv(OUT / "paired-cases.csv", keep_default_na=False)
    d["metric"] = d.metric.str.upper()
    archive = d[d.cohort.ne("external")]
    low = d[d.p.le(32)]
    ext = low[low.cohort.eq("external")]
    historical_low = low[low.cohort.ne("external")]
    scopes = {
        "all": d, "original69": archive[archive.cohort.eq("original")],
        "extra57": archive[archive.cohort.eq("extra")], "archive161": archive,
        "wide_dispatch": archive[archive.p.gt(32)], "historical_low": historical_low,
        "external": ext, "all_low": low, "all_low_K_gt_2": low[low.k_pair.gt(2)],
        "historical_low_binary": historical_low[historical_low.task.eq("binary")],
        "historical_low_regression": historical_low[historical_low.task.eq("regression")],
    }
    rows = [dict(scope=name, comparator=arm, **performance(frame, arm))
            for name, frame in scopes.items() for arm in ARMS]
    pd.DataFrame(rows).to_csv(OUT / "comparisons.csv", index=False)

    # Average primary scores only for one source, protocol and metric at a time.
    source_rows = []
    for key, frame in low.groupby(["cohort", "dataset", "metric"]):
        for arm in (*ARMS, "pair"):
            row = dict(zip(["cohort", "dataset", "metric"], key))
            row.update(method=arm, n=len(frame), p=frame.p.iloc[0],
                       k_min=frame[f"k_{arm}"].min(), k_max=frame[f"k_{arm}"].max(),
                       mean_score=frame[f"score_{arm}"].mean())
            for col in ["total_ms", "select_ms"]:
                row[col] = ((frame[f"{col}1_{arm}"] + frame[f"{col}2_{arm}"]) / 2).median()
            if arm != "pair":
                row.update(mean_pair_score_difference=frame[f"gap_{arm}"].mean(),
                           **performance(frame, arm))
            source_rows.append(row)
    pd.DataFrame(source_rows).to_csv(OUT / "by-source.csv", index=False)


if __name__ == "__main__":
    main()
