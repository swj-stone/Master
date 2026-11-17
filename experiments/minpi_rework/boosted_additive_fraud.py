"""Forecast a historical five-seed source after its compression signal replicated."""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark, legacy_gain, legacy_gain_receiver
from experiments.minpi_rework.boosted_additive import OUT, EVIDENCE


NAME = "credit_card_fraud"
METHODS = ("gain","shap_abs","raw_min","raw_pooled","raw_point","raw_energy","full")
SCOPE = OUT / "fraud-forecasts"


def deploy():
    jobs = legacy_gain.jobs()
    lock = {}
    SCOPE.mkdir(parents=True,exist_ok=True)
    for seed in legacy_gain.CONFIG["seeds"]:
        key = f"{NAME}-{seed}"
        receipt = OUT / f"{key}.json"
        selection = json.loads(receipt.read_bytes())["arms"]
        directory,row = jobs[NAME,seed]
        x,y,_,idx = legacy_gain.read_split(directory,row,seed)
        dev = np.sort(np.r_[idx["fit"],idx["vote"],idx["width"]])
        forecasts, measured, cache = {},{},{}
        for m in METHODS:
            columns = (np.arange(x.shape[1]) if m=="full" else
                       np.sort(selection[m]["order"][:selection[m]["k"]]))
            support = tuple(columns)
            if support not in cache:
                begin = perf_counter()
                pred = legacy_gain_receiver.predict(x,y,dev,idx["test"],columns,row["task"],seed)
                cache[support] = pred, perf_counter()-begin
            forecasts[m], measured[m] = cache[support]
        archive = legacy_gain_receiver.OUT/key/"predictions.npz"
        with np.load(archive) as old:
            for m in ("gain","full"):
                np.testing.assert_array_equal(forecasts[m],old["guard_"+m])
        predictions = SCOPE/f"{key}.npz"
        np.savez_compressed(predictions,**forecasts)
        lock[key] = dict(selection=benchmark.digest(receipt), predictions=benchmark.digest(predictions),
                         inputs={part:row[part+"_file_sha256"] for part in ("train","val","test")},
                         deployment_component_s=measured)
        print("sealed", key, flush=True)
    (SCOPE/"lock.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")


def score():
    jobs = legacy_gain.jobs()
    lock = json.loads((SCOPE/"lock.json").read_bytes())
    assert set(lock)=={f"{NAME}-{s}" for s in legacy_gain.CONFIG["seeds"]}
    for seed in legacy_gain.CONFIG["seeds"]:
        key = f"{NAME}-{seed}"
        assert benchmark.digest(OUT/f"{key}.json")==lock[key]["selection"]
        assert benchmark.digest(SCOPE/f"{key}.npz")==lock[key]["predictions"]
        with np.load(SCOPE/f"{key}.npz") as predictions:
            assert set(predictions)==set(METHODS)
    output=[]
    for seed in legacy_gain.CONFIG["seeds"]:
        key = f"{NAME}-{seed}"
        directory,row = jobs[NAME,seed]
        held = directory/row["test_relpath"]
        assert benchmark.digest(held)==lock[key]["inputs"]["test"]
        y = pd.read_csv(held,usecols=[row["target"]])[row["target"]].to_numpy()
        selection = json.loads((OUT/f"{key}.json").read_bytes())["arms"]
        with np.load(SCOPE/f"{key}.npz") as predictions:
            for m in METHODS:
                v = predictions[m]
                k = selection["gain"]["p"] if m=="full" else selection[m]["k"]
                output.append(dict(seed=seed,method=m,k=k,**benchmark.metrics(y,v,"classification",1.),
                    average_precision=average_precision_score(y,v[:,1]),
                    deployment_component_s=lock[key]["deployment_component_s"][m]))
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    result = pd.DataFrame(output)
    result.to_csv(EVIDENCE/"fraud-historical.csv",index=False)
    (EVIDENCE/"fraud-lock.json").write_bytes((SCOPE/"lock.json").read_bytes())
    print(result.groupby("method")[["k","auc","accuracy","average_precision"]].mean().round(5))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["deploy","score"])
    args=parser.parse_args()
    with threadpool_limits(1):
        (deploy if args.action=="deploy" else score)()
