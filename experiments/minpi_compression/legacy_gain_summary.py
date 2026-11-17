"""Paired historical comparisons and exact source-bootstrap ranges for classification."""
import argparse
from itertools import product
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.minpi_compression import legacy_gain as study


def intervals(panel):
    dest = study.DEST if panel=="ridge" else study.DEST/"xgb-receiver"
    frame = pd.read_csv(dest/"scores.csv")
    if panel == "ridge":
        labels = []
        with ZipFile(dest/"forecasts.zip") as traces:
            for key in json.loads(traces.read("lock.json")):
                state = json.loads(traces.read(key+"/selection.json"))
                r = next(r for r in state["rows"] if r["method"]=="minpi")
                labels.append(dict(dataset=state["dataset"], seed=state["seed"], p=r["p"],
                    negative=r["minimum"].count(-1), neutral=r["minimum"].count(0),
                    positive=r["minimum"].count(1), **r["work"]))
        pd.DataFrame(labels).to_csv(dest/"minima.csv",index=False)
        archive = pd.read_csv(study.bench.ROOT / "docs/thesis/archives/piv1/final_results_by_seed.csv")
        prior = []
        for method in ("gain","piv1"):
            data = archive[["dataset","seed"]].copy()
            for metric in ("auc","accuracy","r2","mae"):
                data[metric] = archive[f"{method}_test_{metric}"]
            data["method"] = "archive_"+method
            data["task"] = np.where(data.auc.notna(),"classification","regression")
            prior.append(data)
        frame = pd.concat([frame,*prior],ignore_index=True)
        frame[frame.method.isin(["minpi","archive_gain","archive_piv1"])].groupby(
            ["dataset","method"])[["auc","accuracy","r2","mae"]].mean().to_csv(dest/"historical-scores.csv")
        comparisons = [("minpi",m) for m in ("full","gain","coefficient","archive_gain","archive_piv1")]
    else:
        frame = frame[frame.part=="primary"]
        comparisons = [("guard_minpi",m) for m in ("guard_gain","guard_full","guard_coefficient","old_direct_gain","old_PIv1")]
        comparisons += [("fixed_minpi","fixed_gain"),("fixed_minpi","old_direct_gain")]
    paired = frame[frame.task=="classification"].set_index(["dataset","seed","method"])
    rows, per_source = [], []
    for arm,baseline in comparisons:
        a = paired.xs(arm,level="method")
        b = paired.xs(baseline,level="method").reindex(a.index)
        diffs = pd.DataFrame({m:a[m]-b[m] for m in ("auc","accuracy")})
        ratios = []
        for m in ("k","selection_s","deployment_s"):
            if a[m].notna().all() and b[m].notna().all() and b[m].gt(0).all() and not (
                    m == "selection_s" and baseline in ("full","guard_full")):
                ratios.append(m)
                diffs[m] = np.log(a[m]/b[m])
        source = diffs.groupby(level="dataset").mean()
        assert len(source) == 5
        by_source = source.copy()
        by_source[ratios] = np.exp(by_source[ratios])
        per_source.append(by_source.assign(arm=arm,baseline=baseline).reset_index())
        draws = np.array(list(product(range(5), repeat=5)))
        for metric in source.columns:
            values = source[metric].to_numpy()
            est,lo,hi = values.mean(),*np.quantile(values[draws].mean(axis=1),[.025,.975])
            if metric in ratios:
                est,lo,hi = np.exp([est,lo,hi])
            rows.append(dict(arm=arm,baseline=baseline,metric=metric,estimate=est,lo=lo,hi=hi))
    out = pd.DataFrame(rows)
    out.to_csv(dest/"classification-source-intervals.csv",index=False)
    pd.concat(per_source).to_csv(dest/"paired-classification-sources.csv",index=False)
    print(out[out.baseline.isin(["gain","guard_gain","archive_gain","old_direct_gain"])].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",choices=("ridge","receiver"))
    intervals(parser.parse_args().panel)
