"""Measure one-time native loading and selection alone on fixed synthetic inputs."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

from research.conditional_dense.original_tasks.regression import regression_support
from research.conditional_dense.vote_width.method import vote_support
from research.lowdim.method import _kernel, fast_support


def profile(folder):
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261118)
    first = perf_counter_ns()
    _kernel()
    load_ms = (perf_counter_ns()-first)/1e6
    records = []
    for task in ("regression", "classification"):
        for n,p in ((300,10),(3000,10),(768,32)):
            X = rng.normal(size=(n,p)).astype(np.float32)
            y = X[:,0] + 0.6*X[:,2] + 0.2*rng.normal(size=n)
            if task == "classification":
                y = np.searchsorted(np.quantile(y,[1/3,2/3]),y)
            previous = regression_support if task == "regression" else vote_support
            order = ["new", "previous"]
            for i in range(-1,31):
                order.reverse()
                for method in order:
                    start = perf_counter_ns()
                    if method == "new":
                        columns,_ = fast_support(X,y,task=task,seed=1847)
                    else:
                        columns,_ = previous(X,y,seed=1847)
                    duration = (perf_counter_ns()-start)/1e6
                    if i>=0:
                        records.append(dict(task=task,n=n,p=p,method=method,repeat=i,
                                            select_ms=duration,k=len(columns)))
    with (folder/"selection.csv").open("x",newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (folder/"startup.json").write_text(json.dumps(dict(native_load_ms=load_ms,thread_limit=1)))
    print(f"Native load {load_ms:.3f} ms; {len(records)} selection observations")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        profile(args.folder)
