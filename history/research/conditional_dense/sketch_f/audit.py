"""Rescore saved predictions and compare balanced rotations in both processes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.conditional_dense.sketch_f.study import METHODS, REPEATS
from research.conditional_dense.transfer.audit import score


def audit(first, second, output):
    output.mkdir(parents=True, exist_ok=False)
    states = [json.loads((folder/"protocol.json").read_text()) for folder in (first,second)]
    assert [s.pop("process") for s in states] == [1,2]
    assert states[0] == states[1] and len(states[0]["panels"]) == 8
    data = pd.concat([pd.read_csv(p/"trials.csv").assign(process=i) for i,p in enumerate((first,second),1)])
    assert len(data) == 8 * len(METHODS) * 2 * REPEATS
    for _, g in data.groupby(["panel","method","process"]):
        assert sorted(g.position) == sorted(g.repeat) == list(range(REPEATS))
    groups = data.groupby(["panel","source","process","method"])
    medians = groups[["score","p","k","select_ms","slice_ms","fit_ms","predict_ms","total_ms"]].median()
    medians.reset_index().to_csv(output/"process-medians.csv",index=False)
    rows = []
    arrays = 0
    for panel in states[0]["panels"]:
        key = panel["panel"]
        selected = data[data.panel==key]
        subset = medians.xs((key,selected.source.iloc[0]),level=("panel","source"))
        pivot = subset.reset_index().pivot(index="method",columns="process",values="total_ms")
        assert selected.groupby("method").score.nunique().eq(1).all()
        scores = selected.groupby("method").score.first()
        with np.load(first/(key+".npz")) as a, np.load(second/(key+".npz")) as b:
            assert set(a.files) == set(b.files)
            for field in a.files:
                np.testing.assert_array_equal(a[field],b[field])
            arrays += len(a.files)
            units = a["units"] if "units" in a else None
            for arm in METHODS:
                np.testing.assert_allclose(score(a["truth"],a[arm+"_pred"],units),scores[arm],rtol=0,atol=1e-12)
            assert len(a["rows"]) == panel["rows"] == 768
            for arm in ("conditional","sketch_F","zero"):
                assert len(a[arm+"_cols"]) == panel["width"] < panel["p"]
            assert len(a["full_cols"]) == panel["p"]
        row = dict(panel=key,source=selected.source.iloc[0],p=panel["p"],k=panel["width"])
        for arm in METHODS:
            row["score_"+arm] = scores[arm]
            row["delta_"+arm] = scores["conditional"] - scores[arm]
            for i in (1,2):
                row[f"ms{i}_"+arm] = pivot.loc[arm,i]
                row[f"ratio{i}_"+arm] = pivot.loc["conditional",i] / pivot.loc[arm,i]
            row["ratio_"+arm] = max(row[f"ratio{i}_"+arm] for i in (1,2))
        row["full_joint"] = row["delta_full"] >= -.02-1e-12 and row["ratio_full"] < 1
        row["both_sketch_F"] = row["delta_sketch_F"] > 1e-12 and row["ratio_sketch_F"] < 1
        row["all_criteria"] = row["full_joint"] and row["both_sketch_F"]
        rows.append(row)
    cases = pd.DataFrame(rows)
    cases.to_csv(output/"cases.csv",index=False)
    summary = []
    for source,g in cases.groupby("source"):
        row = dict(source=source,n=len(g),width_min=g.k.min(),width_max=g.k.max(),
                   mean_conditional=g.score_conditional.mean(),mean_sketch_F=g.score_sketch_F.mean())
        for arm in ("full","sketch_F","zero"):
            row["mean_delta_"+arm] = g["delta_"+arm].mean()
            row["win_quality_"+arm] = (g["delta_"+arm] > 1e-12).sum()
            row["two_process_faster_"+arm] = (g["ratio_"+arm] < 1).sum()
        row["full_joint"] = g.full_joint.sum()
        row["both_sketch_F"] = g.both_sketch_F.sum()
        row["all_criteria"] = g.all_criteria.sum()
        summary.append(row)
    pd.DataFrame(summary).to_csv(output/"sources.csv",index=False)
    verified = dict(timed_pipelines=len(data),panels=len(cases),arrays_compared=arrays,processes=2,
        repeats_per_arm=REPEATS,paired_position_balanced=True,same_sketch_verified=True,
        independent_F_exact_supports=sum(p["n_equal_effect_substitutions"]==0 for p in states[0]["panels"]),
        max_F_boundary_gap=max(p["max_F_boundary_violation"] for p in states[0]["panels"]))
    (output/"verification.json").write_text(json.dumps(verified,indent=2))
    print(cases[["panel","score_conditional","score_sketch_F","delta_sketch_F","ratio1_sketch_F",
                 "ratio2_sketch_F","ratio_full","ratio_zero","all_criteria"]].round(4).to_string(index=False))
    print(verified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("first","second","output"):
        parser.add_argument(key,type=Path)
    args = parser.parse_args()
    audit(args.first,args.second,args.output)
