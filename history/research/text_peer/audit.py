"""Audit both frozen runs and produce row-paired score, time and width evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from research.text_peer.bench import HERE, original


def run():
    base=original.ROOT/"results/minpi-study"
    first,second=[base/("text-peer-log-20261118"+suffix) for suffix in ("a","b")]
    data=pd.concat([pd.read_csv(p/"trials.csv") for p in (first,second)],ignore_index=True)
    assert len(data)==3600 and data.groupby(["run","dataset","fold","method"]).size().eq(5).all()
    metas=json.loads((first/"panels.json").read_text(encoding="utf8"))
    arrays=0
    for meta in metas:
        key=meta["key"]; part=data[(data.dataset==meta["dataset"]) & (data.fold==meta["fold"])]
        with np.load(first/(key+".npz")) as x,np.load(second/(key+".npz")) as y:
            assert set(x.files)==set(y.files)
            for name in x.files:
                np.testing.assert_array_equal(x[name],y[name]);arrays+=1
            for method,records in part.groupby("method"):
                assert (records.width==len(x[method+"_cols"])).all()
                assert (abs(records.ba-balanced_accuracy_score(x["gold"],x[method+"_pred"]))<1e-13).all()
    summary=data.groupby(["dataset","fold","method","run"]).agg(total_ms=("total_ms","median"),select_ms=("select_ms","median"),ba=("ba","first"),width=("width","first"),p=("p","first")).reset_index()
    index=["dataset","fold","run"]
    candidate=summary[summary.method=="candidate"].set_index(index).drop(columns="method")
    full=summary[summary.method=="full"].set_index(index)
    candidate["vs_full"]=candidate.ba-full.ba
    candidate["full_time_ratio"]=candidate.total_ms/full.total_ms
    candidate["joint"]=(candidate.width<candidate.p)&(candidate.vs_full>=-.02-1e-12)&(candidate.full_time_ratio<1)
    out=[]
    for method,group in summary.groupby("method"):
        frame=candidate.copy()
        control=group.set_index(index)
        frame["baseline"]=method
        frame["delta"]=frame.ba-control.ba
        frame["time_ratio"]=frame.total_ms/control.total_ms
        frame["width_ratio"]=frame.width/control.width
        frame["better_ba"]=frame.delta>1e-12
        frame["relative_joint"]=frame.joint & (frame.delta>=-.02-1e-12) & (frame.time_ratio<1) & (frame.width_ratio<=1)
        out.append(frame.reset_index())
    paired=pd.concat(out).groupby(["dataset","fold","baseline"]).agg(
        ba=("ba","first"),width=("width","first"),p=("p","first"),full_delta=("vs_full","first"),
        full_time_ratio=("full_time_ratio","max"),joint=("joint","all"),delta=("delta","first"),better_ba=("better_ba","all"),
        time_ratio=("time_ratio","max"),width_ratio=("width_ratio","first"),relative_joint=("relative_joint","all")).reset_index()
    summary.to_csv(HERE/"case-times.csv",index=False)
    paired.to_csv(HERE/"comparisons.csv",index=False)
    print(f"{arrays} arrays identical; {len(data)} timings")
    for name,g in paired[paired.baseline=="F_match"].groupby("dataset"):
        print(name,len(g), round(g.ba.mean(),4),round(g.full_delta.mean(),4),g.joint.sum(), round(g.delta.mean(),4),round(g.full_time_ratio.median(),3),round((g.width/g.p).median(),3))
    for target, subset in (("old",~paired.dataset.str.startswith("cornell")),("new",paired.dataset.str.startswith("cornell"))):
        print(target,paired[subset].groupby("baseline").agg(ba_wins=("better_ba","sum"),mean_delta=("delta","mean"),joint=("relative_joint","sum")).round(4))


if __name__=="__main__":
    run()
