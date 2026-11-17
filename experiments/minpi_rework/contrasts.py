"""Equal-source contrasts; resample complete tasks, preserving paired splits."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired(frame, method, reference):
    keys = ["dataset", "task", "seed"]
    left = frame[frame.method.eq(method)].set_index(keys)
    right = frame[frame.method.eq(reference)].set_index(keys)
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError("Duplicate method/split entry")
    if not left.index.symmetric_difference(right.index).empty:
        raise ValueError("Missing paired split")
    out = pd.DataFrame(index=left.index)
    for column in ("k", "selection_s", "deployment_s"):
        out[column] = np.log(left[column] / right[column])
    for column in ("auc", "accuracy", "r2", "scaled_mae"):
        if column in left:
            out[column] = left[column] - right[column]
    return out.reset_index().groupby(["task", "dataset"]).mean(numeric_only=True).drop(columns="seed")


def summarize(frame, method, refs):
    rows = []
    for ref in refs:
        pairs = paired(frame, method, ref)
        for task, group in pairs.groupby(level="task"):
            data = group.dropna(axis=1, how="all")
            rng = np.random.default_rng(377)
            boot = data.to_numpy()[rng.integers(len(data),size=(10000,len(data)))].mean(axis=1)
            for j,key in enumerate(data.columns):
                values = [data.iloc[:,j].mean(), *np.quantile(boot[:,j],[.025,.975])]
                if key in ("k","selection_s","deployment_s"):
                    values = np.exp(values)
                rows.append((task,ref,len(data),key,*values))
    return pd.DataFrame(rows,columns=["task","reference","tasks","metric","point","lo","hi"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("csv",type=Path,nargs="+")
    cli.add_argument("--method",default="min")
    cli.add_argument("--refs",nargs="+",default=["coefficient","rfe","mean","current","fit"])
    cli.add_argument("--output",type=Path)
    args = cli.parse_args()
    report = summarize(pd.concat([pd.read_csv(path) for path in args.csv]),args.method,args.refs)
    if args.output:
        report.to_csv(args.output,index=False)
    print(report.pivot(index=["task","reference"],columns="metric",values="point").round(4).to_string())
