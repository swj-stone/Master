"""Audit R3 on the six original CUDA splits against contemporaneous R2 direct controls."""
import argparse
from itertools import product
import json

import numpy as np
import pandas as pd

from experiments.anchored_vote import audit, run
from experiments.minpi_compression import legacy_gain as legacy
from experiments.split_vote import report, run as r2


SEEDS = legacy.CONFIG["seeds"]
SOURCES = r2.ORIGINAL
ANCHOR_ARMS = ("min", "mean", "gain")
CONTROL_ARMS = ("exhaustive", "gain_grid")
OUT = r2.ROOT / "thesis-paper/evidence/anchored_vote"


def verify_files():
    signatures = {}
    records = {}
    for name,seed,arm in product(SOURCES, SEEDS, ANCHOR_ARMS + CONTROL_ARMS):
        anchored = arm in ANCHOR_ARMS
        tag = "original-r3" if anchored else "six-paired"
        folder = (run.RUN if anchored else r2.RUN)/tag/f"{name}-{seed}"/arm
        file = folder/"receipt.json"
        forecast = folder/"forecast.npz"
        rec = json.loads(file.read_bytes())
        if (rec["dataset"],rec["seed"],rec["method"]) != (name,seed,arm):
            raise ValueError(f"Mismatched receipt {file}")
        if not r2.digest(forecast) == rec["forecast" if anchored else "forecast_sha256"]:
            raise ValueError(f"Forecast hash mismatch {forecast}")
        sig = rec["signature" if anchored else "version"]
        if tag in signatures and sig != signatures[tag]:
            raise ValueError(f"Mixed code identity within {tag}")
        signatures[tag] = sig
        if anchored and rec["device"]!="cuda":
            raise ValueError(f"Unexpected backend {file}")
        records[name,seed,arm] = (rec,forecast,r2.digest(file))
    if signatures["six-paired"]["direct_gain.py"] != signatures["original-r3"]["direct_gain.py"]:
        raise ValueError("The two runs used different native fit helpers")
    return records


def score():
    records = verify_files()
    tasks = legacy.jobs()
    rows = []
    for n,s in product(SOURCES,SEEDS):
        folder,manifest = tasks[n,s]
        x,_,task,idx = r2.load(n,s)
        path = folder/manifest["test_relpath"]
        if r2.digest(path) != manifest["test_file_sha256"]:
            raise ValueError(f"Corrupt test split {path}")
        y = pd.read_csv(path,usecols=[manifest["target"]]).iloc[:,0].to_numpy()
        control = records[n,s,"exhaustive"][0]["record"]
        order = control["gain_order"]
        np.testing.assert_array_equal(sorted(order),np.arange(x.shape[1]))
        grid = records[n,s,"gain_grid"][0]["record"]
        np.testing.assert_array_equal(order,grid["order"])
        for k,util in grid["scores"].items():
            np.testing.assert_allclose(util,control["width_curve"][int(k)-1],atol=1e-12,rtol=0)
        anchor_ref = records[n,s,"gain"][0]["record"]
        same = {}
        for m in ANCHOR_ARMS+CONTROL_ARMS:
            report_,fc,receipt_hash = records[n,s,m]
            if (report_["p"],report_["task"]) != (x.shape[1],task):
                raise ValueError("Split identity changed")
            r = report_["record"]
            support = r["support"] if m=="exhaustive" else r["selected"]
            k = len(support)
            audit.validate_support(support,x.shape[1])
            if m in ANCHOR_ARMS:
                for field in ("anchor","anchor_curve","floor"):
                    if r[field] != anchor_ref[field]:
                        raise ValueError(f"Unequal native anchors at {n}/{s}/{m}")
                np.testing.assert_array_equal(r["anchor"],sorted(order[:len(r["anchor"])]))
                if not set(support)<=set(r["anchor"]) or r["utility"] < r["floor"]-1e-14:
                    raise ValueError(f"Violated inner guarantee {n}/{s}/{m}")
                for width,value in r["anchor_curve"].items():
                    np.testing.assert_allclose(value,control["width_curve"][int(width)-1],rtol=0,atol=1e-12)
                if not r["fits"] <= 7:
                    raise ValueError(f"Exceeded registered refits {n}/{s}/{m}")
            with np.load(fc) as a:
                pred = a["primary"]
                if m in ANCHOR_ARMS:
                    np.testing.assert_array_equal(a["indices"],idx["test"])
            if len(pred)!=len(y) or not np.isfinite(pred).all():
                raise ValueError(f"Forecast alignment {fc}")
            cols = tuple(support)
            if cols in same:
                np.testing.assert_allclose(same[cols],pred,rtol=0,atol=1e-7)
            same[cols] = pred
            rows.append(dict(dataset=n,seed=s,method=m,task=task,p=x.shape[1],k=k,
                             support=json.dumps(support,separators=(",",":")),fits=r["fits"],
                             anchor_k=len(r.get("anchor",[])) or None,
                             seconds=r.get("seconds",r.get("worker_s")),
                             selection_s=r.get("selection_s"),**audit.score(y,pred,task),
                             seconds_source="original-r3" if m in ANCHOR_ARMS else "six-paired",
                             near_width=r.get("near_width"),receipt_sha256=receipt_hash,
                             forecast_sha256=r2.digest(fc),test_sha256=manifest["test_file_sha256"]))
    frame = pd.DataFrame(rows)
    out = []
    for b in ("mean","gain",*CONTROL_ARMS):
        pair = report.compare(frame,"min",b)
        pair["timing_session"] = "within" if b in ANCHOR_ARMS else "separate_same_device"
        out.append(pair)
    sources = pd.concat(out,ignore_index=True)
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/"prepared-seeds.csv",index=False)
    sources.to_csv(OUT/"prepared-sources.csv",index=False)
    print(sources.groupby("baseline")[["pass_joint","pass_with_quality","paired_pass_seeds","quality_pass_seeds"]].sum())


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    score()
