"""Verify that the combined case table identifies the primary support and actual trace."""
import json

import numpy as np
import pandas as pd

from research.lowdim.combined import HERE, ARMS
from research.feature_game_eval.panels import ARCHIVE


def trace(r):
    bare=f"{r.dataset}-{r.fold}-{r.seed}.npz"
    if r.cohort in ("original","extra"):
        return ARCHIVE/"lowdim-core-20261118a-pass1"/(r.cohort+"-"+bare)
    if r.cohort=="formal":
        return ARCHIVE/"lowdim-formal-20261118a-pass1"/f"{r.dataset}-{r.seed}.npz"
    if r.cohort=="text":
        name=r.dataset if r.dataset.startswith("bydate_") else f"{r.dataset}-{r.fold}"
        return ARCHIVE/"lowdim-wide-text/process1"/(name+".npz")
    if r.cohort in ("archive_supplement","conference_fraud"):
        label="survey" if r.cohort=="archive_supplement" else "fraud"
        return ARCHIVE/f"lowdim-{label}-20261118a-pass1"/bare
    return ARCHIVE/"lowdim-breadth-20261118a"/(r.cohort+"1")/bare


def run():
    out=HERE/"evidence"
    d=pd.read_csv(out/"all-cases.csv",keep_default_na=False)
    for r in d.itertuples(index=False):
        with np.load(trace(r)) as a:
            for arm in ARMS:
                cols=a[arm+"_cols"]
                assert len(cols)==getattr(r,"k_"+arm)
                assert len(np.unique(cols))==len(cols) and cols.min()>=0 and cols.max()<r.p
                assert len(a[arm+"_pred"])==len(a["truth"])
            if r.p>32:
                for suffix in ("_cols","_pred"):
                    np.testing.assert_array_equal(a["new"+suffix],a["previous"+suffix])
            if r.dataset in ("abalone","diabetes"):
                np.testing.assert_array_equal(a["new_cols"],a["F_cols"])
        ratio=max(r.ms1_new/r.ms1_full,r.ms2_new/r.ms2_full)
        assert (r.k<r.p and ratio<1)==r.cut_faster
    for group,ref in (("all",d),("lowdim",d[d.p<=32])):
        assert len(ref)>0 and (ref.better_faster_F & ref.better_faster_random).sum()==ref.beat_both_matched.sum()
    totals=pd.read_csv(out/"cohort-counts.csv").set_index("cohort").loc["ALL"]
    for col,val in totals.items():
        assert val==(len(d) if col=="cases" else d[col].sum())
    dedup=pd.read_csv(out/"one-row-per-input-seed-metric.csv",keep_default_na=False)
    hashes=["train_sha","target_sha","test_sha","truth_sha"]
    assert len(dedup)==len(d.drop_duplicates(hashes+["seed","metric"]))
    records=dict(configurations=len(d),source_protocol_metric_rows=len(pd.read_csv(out/"all-sources.csv",keep_default_na=False)),
                 actual_supports_checked=len(d)*len(ARMS),unchanged_wide_cols_predictions=int((d.p>32).sum()),
                 exact_F_sets_diabetes_abalone=int(d.dataset.isin(["abalone","diabetes"]).sum()),
                 distinct_inputs=d.groupby(hashes).ngroups,distinct_input_seed_metric=len(dedup),
                 pareto_both_full_speed_within_002=int((d.pareto_both_and_full_speed & d.cut_faster_002).sum()))
    (out/"verification.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    print(records)


if __name__=="__main__":
    run()

