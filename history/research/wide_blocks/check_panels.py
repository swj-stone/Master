"""Recover archived inputs and independently replay historical controls, outside timing."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from research.wide_blocks.panels import HASHES, KEY, cases, registry


def check(wanted, out):
    index = registry()
    reports = []
    for case in cases(wanted):
        row = index.loc[case.key]
        with np.load(case.archive) as saved:
            np.testing.assert_array_equal(case.truth, saved["truth"])
            width = len(saved["new_cols"])
            supports = {"new":case.select_old(), "full":np.arange(case.X.shape[1]),
                        "random":case.select_random(width), "F":case.select_F(width)}
            for arm, cols in supports.items():
                np.testing.assert_array_equal(cols, saved[arm+"_cols"], err_msg=str(case.key))
                pred, decision = case.predict(cols)
                np.testing.assert_array_equal(pred, saved[arm+"_pred"], err_msg=str(case.key))
                if arm+"_decision" in saved:
                    np.testing.assert_array_equal(decision, saved[arm+"_decision"], err_msg=str(case.key))
                np.testing.assert_allclose(case.score(pred, decision), row["score_"+arm],
                                           rtol=0, atol=1e-12, err_msg=str(case.key))
        reports.append(dict(zip(KEY,case.key),dataset_width=row.p,n_train=len(case.y),n_test=len(case.truth),
                            sparse=sparse.issparse(case.X),full_feature_count=len(supports["full"]),
                            controls_replayed=len(supports)))
        if len(reports)%30==0:
            print(f"{len(reports)} sets of inputs and predictions checked",flush=True)
    frame = pd.DataFrame(reports)
    assert len(frame)==len(wanted)
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/"panel-replay.csv",index=False)
    finish(out, wanted)


def finish(out,wanted):
    frame=pd.read_csv(out/"panel-replay.csv",keep_default_na=False)
    keys=set(frame[KEY].itertuples(index=False,name=None))
    assert len(frame)==len(keys) and keys==wanted
    assert (frame.controls_replayed==4).all()
    counts=dict(cases=len(frame),fingerprints=4*len(frame),
                support_prediction_and_score_checks=int(frame.controls_replayed.sum()),
                sparse=int(frame["sparse"].sum()))
    registered=registry().reset_index(drop=True)
    audited=frame.merge(registered[KEY+["source","source_kind","metric",*HASHES]],
                        on=KEY,validate="one_to_one")
    audited["input_id"]=audited.groupby(list(HASHES),sort=False).ngroup()
    audited.groupby(["cohort","source","dataset","source_kind","metric"]).agg(
        cases=("seed","size"),unique_inputs=("input_id","nunique"),sparse=("sparse","sum"),
        min_p=("dataset_width","min"),max_p=("dataset_width","max"),
        min_n=("n_train","min"),max_n=("n_train","max")
    ).to_csv(out/"sources.csv")
    (out/"panel-replay.json").write_text(json.dumps(counts,indent=2),encoding="utf8")
    print(counts)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--cohort")
    args=parser.parse_args()
    frame=registry()
    wanted=set(frame[frame.cohort.eq(args.cohort)].index if args.cohort else frame.index)
    with threadpool_limits(1):
        check(wanted,args.out)
