"""Recompute scores and support widths, then join speed and quality on the same case."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.conditional_dense.transfer.audit import score
from research.conditional_dense.vote_width.study import METHODS, REPEATS


def audit(first, second, output):
    output.mkdir(parents=True, exist_ok=False)
    states = [json.loads((f/"protocol.json").read_text()) for f in (first,second)]
    assert [s.pop("process") for s in states] == [1,2]
    assert states[0] == states[1] and len(states[0]["panels"]) == 16
    data = pd.concat([pd.read_csv(f/"trials.csv").assign(process=i)
                      for i,f in enumerate((first,second),1)])
    assert len(data) == 16 * len(METHODS) * 2 * REPEATS
    assert data.source.nunique() == 6 and set(data.method) == set(METHODS)
    for _, group in data.groupby(["panel","method","process"]):
        assert sorted(group.position) == sorted(group.repeat) == list(range(REPEATS))
    stage_cols = ["select_ms","slice_ms","fit_ms","predict_ms","total_ms"]
    assert (data.total_ms + 1e-8 >= data[stage_cols[:-1]].sum(axis=1)).all()
    medians = data.groupby(["panel","source","process","method"])[["score","p","k",*stage_cols]].median()
    medians.reset_index().to_csv(output/"process-medians.csv",index=False)
    details, paired, count = [], [], 0
    for panel in states[0]["panels"]:
        key = panel["panel"]
        group = data[data.panel == key]
        assert group.groupby("method").score.nunique().eq(1).all()
        scores = group.groupby("method").score.first()
        timings = medians.xs((key,group.source.iloc[0]),level=("panel","source")).reset_index()
        times = timings.pivot(index="method",columns="process",values="total_ms")
        widths = group.groupby("method").k.first()
        with np.load(first/(key+".npz")) as a, np.load(second/(key+".npz")) as b:
            assert set(a.files) == set(b.files)
            for field in a:
                np.testing.assert_array_equal(a[field],b[field])
            count += len(a.files)
            units = a["units"] if "units" in a else None
            for arm in METHODS:
                assert widths[arm] == len(np.unique(a[arm+"_cols"]))
                np.testing.assert_allclose(score(a["truth"],a[arm+"_pred"],units),scores[arm],rtol=0,atol=1e-12)
            votes = a["votes"].astype(np.float64)
            prob = votes/votes.sum() if votes.sum() > 0 else None
            diversity = float(1./sum(x*x for x in prob)) if prob is not None else None
            k0 = int(widths["conditional"])
            k = k0 if diversity is None else min(k0, int(np.ceil(diversity)))
            assert k0 == panel["original_width"] and k == panel["k"] == widths["vote_width"]
            ranking = np.lexsort((np.arange(len(votes)), -votes))
            np.testing.assert_array_equal(a["vote_width_cols"],np.sort(ranking[:k]))
            assert all(widths[m] == k for m in ("F","sketch_F","random"))
        source, _, _ = key.rsplit("-",2)
        row = dict(panel=key,source=source,p=panel["p"],k0=k0,k=k,
                   tightened=k<k0,effective_count=diversity,
                   score=scores["vote_width"],full_loss=scores["full"]-scores["vote_width"])
        for arm in METHODS:
            row["k_"+arm] = widths[arm]
            row["score_"+arm] = scores[arm]
            row["delta_"+arm] = scores["vote_width"]-scores[arm]
            row["win_"+arm] = row["delta_"+arm] > 1e-12
            for i in (1,2):
                row[f"ms{i}_"+arm] = times.loc[arm,i]
                row[f"ratio{i}_"+arm] = times.loc["vote_width",i]/times.loc[arm,i]
            row["ratio_"+arm] = max(row["ratio1_"+arm],row["ratio2_"+arm])
            row["fast_"+arm] = row["ratio_"+arm] < 1
        row["cut_fast_full"] = k<panel["p"] and row["fast_full"]
        row["old_tolerance"] = row["full_loss"] <= .02+1e-12
        row["joint_full_old"] = row["cut_fast_full"] and row["old_tolerance"]
        for arm in METHODS:
            if arm=="vote_width":
                continue
            joint = row["cut_fast_full"] and row["fast_"+arm]
            paired.append(dict(panel=key,source=source,arm=arm,k=k,baseline_k=widths[arm],
                delta=row["delta_"+arm],worst_ratio=row["ratio_"+arm],
                faster=row["fast_"+arm],better=row["win_"+arm],
                win_both=row["fast_"+arm] and row["win_"+arm],
                cut_faster_both=joint,cut_faster_both_and_full_within_2pp=joint and row["old_tolerance"],
                cut_faster_both_and_beat_arm=joint and row["win_"+arm],
                cut_faster_both_and_beat_arm_and_full_within_2pp=joint and row["win_"+arm] and row["old_tolerance"]))
        details.append(row)
    cases = pd.DataFrame(details)
    cases.to_csv(output/"cases.csv",index=False)
    pairs = pd.DataFrame(paired)
    pairs.to_csv(output/"paired.csv",index=False)
    flags = ["faster","better","win_both","cut_faster_both","cut_faster_both_and_full_within_2pp",
             "cut_faster_both_and_beat_arm","cut_faster_both_and_beat_arm_and_full_within_2pp"]
    summary = pairs.groupby("arm")[flags].sum()
    summary.to_csv(output/"comparator-counts.csv")
    bysource = cases.groupby("source").agg(configurations=("panel","size"),
         original_width_min=("k0","min"), original_width_max=("k0","max"), width_min=("k","min"),
         width_max=("k","max"), tightened=("tightened","sum"), mean_score=("score","mean"),
         mean_full_delta=("delta_full","mean"), worst_full_delta=("delta_full","min"),
         mean_old_delta=("delta_conditional","mean"), mean_F_delta=("delta_F","mean"),
         mean_sketch_F_delta=("delta_sketch_F","mean"),mean_half_F_delta=("delta_F_half","mean"),
         mean_random_delta=("delta_random","mean"),cut_fast_full=("cut_fast_full","sum"),
         cut_fast_full_old=("joint_full_old","sum"))
    bysource.to_csv(output/"sources.csv")
    verified = dict(timed_pipelines=len(data),sources=6,panels=len(cases),processes=2,
        repeats_per_arm=REPEATS,arrays_compared=count,all_predictions_rescored=True,
        all_widths_and_sorted_supports_verified=True,balanced_schedule=True,
        max_F_boundary_gap=max(max(p["F_boundary_gaps"].values()) for p in states[0]["panels"]))
    (output/"verification.json").write_text(json.dumps(verified,indent=2))
    print(bysource.round(4).to_string())
    print(summary.to_string())
    print(verified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first","second","output"):
        parser.add_argument(name,type=Path)
    params=parser.parse_args()
    audit(params.first,params.second,params.output)
