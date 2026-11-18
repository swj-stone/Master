"""Audit archived pairs and report absolute score, selection and end-to-end costs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.feature_game_eval import panels as bench
from research.lowdim.panel_study import KEY, METHODS


def load_raw(first,second):
    frames=[]
    for run,folder in enumerate((first,second),1):
        state=json.loads((folder/"complete.json").read_text(encoding="utf-8"))
        assert state["process"]==run and tuple(state["arms"])==METHODS
        trials=pd.read_csv(folder/"times.csv")
        assert len(trials)==126*len(METHODS)**2
        assert not trials.duplicated(KEY+["method","repeat"]).any()
        assert trials.groupby(KEY+["method"]).position.nunique().eq(len(METHODS)).all()
        assert trials.select_ms.ge(0.).all() and trials.total_ms.ge(trials.select_ms).all()
        trials["process"]=run
        frames.append(trials)
    raw=pd.concat(frames,ignore_index=True)
    assert raw.groupby(KEY+["method"]).score.nunique().eq(1).all()
    for filename in ("inputs.csv","f-audit.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(first/filename),pd.read_csv(second/filename))
    return raw


def audit_scores(first,second,key,medians):
    name="-".join(map(str,key))+".npz"
    with np.load(first/name) as arrays, np.load(second/name) as replay:
        assert set(arrays.files)==set(replay.files)
        for entry in arrays.files:
            np.testing.assert_array_equal(arrays[entry],replay[entry])
        if medians.loc["new","p"]>32:
            for suffix in ("_cols","_pred"):
                np.testing.assert_array_equal(arrays["new"+suffix],arrays["previous"+suffix])
        units=arrays["units"] if "units" in arrays.files else None
        dataset=key[1]
        for arm,row in medians.iterrows():
            decision=arrays[arm+"_decision"] if dataset=="secom" else None
            score,_,ap=bench.deployment.score_predictions(dataset,arrays["truth"],arrays[arm+"_pred"],
                                                          units,row.task,decision)
            np.testing.assert_allclose(score,row.score,rtol=0,atol=1e-12)
            assert len(arrays[arm+"_cols"])==row.k
            if dataset=="secom":
                np.testing.assert_allclose(ap,row.ap,rtol=0,atol=1e-12)
        return len(arrays.files)


def summarize(first,second,out):
    raw=load_raw(first,second)
    med=raw.groupby(KEY+["method","process"],as_index=False).agg(
        **{field:(field,"first") for field in ("p","k","task","classes","metric","family","source_kind",
                                               "n_train","n_test","score","ap")},
        select_ms=("select_ms","median"),total_ms=("total_ms","median"))
    cases=[]
    array_count=0
    for key,group in med.groupby(KEY,sort=False):
        times=[group[group.process==run].set_index("method") for run in (1,2)]
        m=times[0]
        array_count+=audit_scores(first,second,key,m)
        new=m.loc["new"]
        case=dict(zip(KEY,key))
        for field in ("task","metric","classes","family","source_kind","n_train","n_test","p","k"):
            case[field]=new[field]
        case["removed"]=bool(new.k<new.p)
        for arm in METHODS:
            case["score_"+arm]=m.loc[arm,"score"]
            case["width_"+arm]=int(m.loc[arm,"k"])
            if new.task=="classification":
                case["ap_"+arm]=m.loc[arm,"ap"]
            gap=new.score-m.loc[arm,"score"]
            case["gap_"+arm]=gap
            delta,ratio,select_delta=[],[],[]
            for run,t in enumerate(times,1):
                a,b=t.loc["new"],t.loc[arm]
                delta.append(a.total_ms-b.total_ms)
                ratio.append(a.total_ms/b.total_ms)
                select_delta.append(a.select_ms-b.select_ms)
                case[f"ms{run}_{arm}"]=b.total_ms
                case[f"select_ms{run}_{arm}"]=b.select_ms
            case["ratio_"+arm]=max(ratio)
            case["time_diff_ms_"+arm]=max(delta)
            case["select_diff_ms_"+arm]=max(select_delta)
            case["faster_"+arm]=bool(max(ratio)<1)
            case["better_"+arm]=bool(gap>1e-12)
            case["both_"+arm]=case["faster_"+arm] and case["better_"+arm]
        case["cut_faster"]=case["removed"] and case["faster_full"]
        case["cut_faster_within_002"]=case["cut_faster"] and case["gap_full"]>=-.02-1e-12
        cases.append(case)
    rows=pd.DataFrame(cases)
    assert len(rows)==126 and rows.cohort.value_counts().to_dict()=={"original":69,"extra":57}
    out.mkdir(parents=True,exist_ok=False)
    raw_inputs=pd.read_csv(first/"inputs.csv")
    rows=rows.merge(raw_inputs,on=KEY,how="inner",validate="one_to_one")
    med.to_csv(out/"process-medians.csv",index=False)
    rows.to_csv(out/"cases.csv",index=False)
    agg=dict(n=("k","size"),k_min=("k","min"),k_max=("k","max"),reduced=("removed","sum"),
             cut_faster=("cut_faster","sum"),cut_faster_within_002=("cut_faster_within_002","sum"))
    for arm in METHODS:
        for qty,stat in (("score","mean"),("gap","mean"),("ratio","median"),("faster","sum"),
                         ("better","sum"),("both","sum"),("time_diff_ms","median"),("select_diff_ms","median")):
            field=qty+"_"+arm
            agg[field]=(field,stat)
    rows.groupby(["cohort","source_kind","dataset","task","metric","p"]).agg(**agg).to_csv(out/"sources.csv")
    rows["low_dim"]=rows.p.le(32)
    rows.groupby(["cohort","task","low_dim","metric"]).agg(**agg).to_csv(out/"strata.csv")
    audits=pd.read_csv(first/"f-audit.csv")
    audits.to_csv(out/"f-audit.csv",index=False)
    summary=dict(cohorts=rows.cohort.value_counts().to_dict(),sources=rows.dataset.nunique(),
                 distinct_test_matrices=rows.test_sha.nunique(),cross_process_arrays=array_count,
                 unchanged_wide_supports=int(rows.p.gt(32).sum()),
                 scored_vectors=len(rows)*len(METHODS),timed_calls=len(raw),
                 F_exact_sets=int(audits.exact_F_set.sum()),
                 F_max_border_violation=audits.independent_F_gap.max())
    (out/"verification.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("first","second","output"):
        parser.add_argument(name,type=Path)
    opts=parser.parse_args()
    summarize(opts.first,opts.second,opts.output)
