"""Replay frozen inner curves using the independently registered zero-tolerance sensitivity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression.benchmark import DESIGN, EVIDENCE, RUN, digest, metrics, predict, split, stamp


def acceptable(row, full, task):
    if task == "classification":
        return row["auc"] >= full["auc"] and row["accuracy"] >= full["accuracy"]
    return row["r2"] >= full["r2"] and row["scaled_mae"] <= full["scaled_mae"]


def select(phase):
    primary = RUN / phase
    dest = RUN / "zero_tolerance" / phase
    dest.mkdir(parents=True, exist_ok=True)
    lock = json.loads((primary/"lock.json").read_bytes())
    sealed = {}
    for key, receipt in lock.items():
        path = primary / key
        assert digest(path/"selection.json") == receipt
        state = json.loads((path/"selection.json").read_bytes())
        assert state["sources"] == stamp()
        x,y,task,idx = split(state["dataset"],state["seed"])
        assert all(np.array_equal(idx[k], v) for k,v in state["splits"].items())
        dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
        forecasts, rows = {}, []
        for r in state["rows"]:
            if r["repeat"] != 0:
                continue
            order = np.array(r["order"])
            inner = {int(k):v for k,v in r["inner"].items()}
            k = len(order)
            if inner:
                k = next(i for i in sorted(inner) if acceptable(inner[i],inner[k],task))
            assert k >= r["k"]
            forecasts[r["method"]] = predict(x,y,task,dev,idx["test"],order,[k])[k]
            rows.append(dict(method=r["method"],p=r["p"],k=k,original_k=r["k"],
                             selected=sorted(order[:k].tolist()),inner=inner.get(k),original_inner=inner.get(r["k"])))
        target = dest/key
        target.mkdir(exist_ok=True)
        np.savez_compressed(target/"predictions.npz", **forecasts)
        meta = dict(dataset=state["dataset"],seed=state["seed"],rows=rows,primary=receipt,
                    predictions=digest(target/"predictions.npz"),test=idx["test"].tolist())
        (target/"selection.json").write_text(json.dumps(meta),encoding="utf8")
        sealed[key] = digest(target/"selection.json")
    (dest/"lock.json").write_text(json.dumps(sealed,indent=2), encoding="utf8")
    print(f"{phase} sealed {len(sealed)} sensitivity selections", flush=True)


def score(phase):
    dest = RUN / "zero_tolerance" / phase
    lock = json.loads((dest/"lock.json").read_bytes())
    assert len(lock) == len(DESIGN[phase])*len(DESIGN["seeds"])
    rows = []
    for key, receipt in lock.items():
        path = dest / key
        assert digest(path/"selection.json") == receipt
        state = json.loads((path/"selection.json").read_bytes())
        assert digest(path/"predictions.npz") == state["predictions"]
        assert state["primary"] == digest(RUN/phase/key/"selection.json")
        x,y,task,idx = split(state["dataset"],state["seed"])
        assert np.array_equal(idx["test"],state["test"])
        dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
        with np.load(path/"predictions.npz") as pred:
            for r in state["rows"]:
                outcome = metrics(y[idx["test"]],pred[r["method"]],task,max(np.std(y[dev]),1e-12))
                inner_delta = {f"inner_delta_{m}":v-r["original_inner"][m] for m,v in (r["inner"] or {}).items()}
                rows.append(dict(dataset=state["dataset"],seed=state["seed"],method=r["method"],task=task,
                                 p=x.shape[1],k=r["k"],original_k=r["original_k"],retained=r["k"]/x.shape[1],
                                 selected=" ".join(map(str,r["selected"])),**inner_delta,**outcome))
    target = EVIDENCE/"zero_tolerance"/phase
    target.mkdir(parents=True,exist_ok=True)
    (target/"lock.json").write_text(json.dumps(lock,indent=2),encoding="utf8")
    raw = pd.DataFrame(rows)
    raw.to_csv(target/"scores.csv",index=False)
    paired = raw.set_index(["task","dataset","seed","method"]).sort_index()
    comparisons = []
    for task in raw.task.unique():
        group = paired.xs(task)
        minpi = group.xs("minpi",level="method")
        for method in DESIGN["methods"]:
            if method == "minpi":
                continue
            baseline = group.xs(method,level="method")
            fields = ["auc","accuracy"] if task=="classification" else ["r2","scaled_mae"]
            contrasts = minpi[fields]-baseline[fields]
            contrasts["log_width_ratio"] = np.log(minpi.retained/baseline.retained)
            s = contrasts.groupby("dataset").mean().mean()
            width = np.exp(s.pop("log_width_ratio"))
            quality = (s.auc >= -.01 and s.accuracy >= -.01 if task == "classification"
                       else s.r2 >= -.02 and s.scaled_mae <= .02)
            comparisons.append(dict(task=task,baseline=method,width_ratio=width,**s.to_dict(),
                                    within_quality_bounds=quality))
    frame = pd.DataFrame(comparisons)
    frame.to_csv(target/"contrasts.csv",index=False)
    print(frame.round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["select","score"])
    p.add_argument("phase",choices=["pilot","confirmation"])
    args=p.parse_args()
    with threadpool_limits(limits=1):
        (select if args.action=="select" else score)(args.phase)
