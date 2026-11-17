"""Aggregate the independently timed closure arms; preserve source-level differences."""
import json

import numpy as np
import pandas as pd

from experiments.minpi_compression.benchmark import EVIDENCE, prepare
from experiments.minpi_compression.oracle_audit import complete_votes
from threadpoolctl import threadpool_limits


def verify_extra_edges(folder):
    bounds = pd.read_csv(folder/"bounds.csv")
    extra = bounds[(bounds.closed == -1) & (bounds.refined != -1) & bounds.exact.notna()]
    for (name, seed), rows in extra.groupby(["source","seed"]):
        args = prepare(name, seed)
        vote = complete_votes(args["X_fit"], args["y_fit"], args["X_vote"], args["y_vote"], args["task"])
        for row in rows.itertuples():
            mask = sum(1 << j for j in json.loads(row.closed_witness))
            assert not mask & (1 << row.feature)
            assert vote[mask] and not vote[mask | (1 << row.feature)]


def main():
    folder = EVIDENCE/"dictator"
    with threadpool_limits(1):
        verify_extra_edges(folder)
    clocks = pd.read_csv(folder/"runs.csv")
    records = []
    for key, frame in clocks.groupby(["panel","source","task","seed","method"]):
        assert len(frame) == 5 and frame.position.nunique() == 5
        assert all(frame[k].nunique() == 1 for k in ("order_sha256","columns","forecast_sha256"))
        row = frame.iloc[0].to_dict()
        for k in ("rank_s","selection_s","deployment_s"):
            row[k] = frame[k].median()
        records.append(row)
    frame = pd.DataFrame(records).set_index(["panel","source","task","seed","method"])
    rows = []
    for method in ("bound","closure"):
        chosen = frame.xs(method, level="method")
        for baseline in ("original","cached","coefficient","bound"):
            if baseline == method:
                continue
            other = frame.xs(baseline, level="method")
            data = pd.DataFrame(index=chosen.index)
            for k in ("rank_s","selection_s","deployment_s","k"):
                data["log_ratio_"+k] = np.log(chosen[k]/other[k])
            for k in ("auc","accuracy","r2","mae","scaled_mae"):
                data["delta_"+k] = chosen[k] - other[k]
            for k in ("order_sha256","columns","forecast_sha256"):
                data["changed_"+k] = (chosen[k] != other[k]).astype(int)
            data["splits"] = 1
            source = data.groupby(["panel","source","task"]).agg(
                {k: "sum" if k.startswith("changed_") or k=="splits" else "mean" for k in data})
            source["method"] = method
            source["baseline"] = baseline
            rows.append(source.reset_index())
    sources = pd.concat(rows,ignore_index=True)
    keys = ["method","baseline","panel","task"]
    numeric = sources.select_dtypes("number")
    domains = sources.groupby(keys).agg({k:"sum" if k.startswith("changed_") or k=="splits" else "mean"
                                        for k in numeric}).reset_index()
    domains = domains.drop(columns="delta_mae")
    for table, name in ((sources,"source-effects"),(domains,"domain-effects")):
        for k in ("rank_s","selection_s","deployment_s","k"):
            table["ratio_"+k] = np.exp(table.pop("log_ratio_"+k))
        table.to_csv(folder/f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
