"""Verify stored predictions and aggregate paired process medians by source and task."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, r2_score

from experiments.stable_minpi_stress import digest
from research.conditional_dense.original_tasks.study import BINARY, KEY, METHODS, REGRESSION
from research.feature_game_eval.panels import ARCHIVE


def summarize(first, second, out):
    out.mkdir(parents=True, exist_ok=False)
    streams = []
    for process, path in enumerate((first, second), 1):
        trials = pd.read_csv(path/"times.csv")
        assert len(trials) == 45 * len(METHODS)**2
        assert not trials.duplicated(KEY+["method","repeat"]).any()
        assert trials.groupby(KEY+["method"]).position.nunique().eq(len(METHODS)).all()
        assert trials.select_ms.le(trials.total_ms).all()
        trials["process"] = process
        streams.append(trials)
    data = pd.concat(streams, ignore_index=True)
    assert data.groupby(KEY+["method"]).score.nunique().eq(1).all()
    medians = data.groupby(KEY+["task","p","method","process"],as_index=False).agg(
        k=("k","first"),score=("score","first"),ap=("average_precision","first"),
        select_ms=("select_ms","median"),total_ms=("total_ms","median"))
    medians.to_csv(out/"process-medians.csv",index=False)
    cases, arrays_verified = [], 0
    frozen = pd.read_csv(ARCHIVE/"adaptive-69-20261117a"/"inputs.csv")
    frozen = frozen[frozen.dataset.isin(BINARY+REGRESSION)].set_index(KEY)
    for key, part in medians.groupby(KEY):
        method = part[part.process==1].set_index("method")
        name,fold,seed = key
        record = dict(zip(KEY,key))
        record.update(task=part.task.iloc[0],p=int(part.p.iloc[0]),k=int(method.loc["vote_width","k"]),
                      geometric_k=int(method.loc["geometric","k"]))
        file = f"{name}-{fold}-{seed}.npz"
        with np.load(first/file) as generated, np.load(second/file) as replay:
            assert set(generated.files) == set(replay.files)
            for entry in generated.files:
                np.testing.assert_array_equal(generated[entry], replay[entry])
                arrays_verified += 1
            # The archive's dtype/shape-inclusive hash matches original.input_hashes().
            assert frozen.loc[key,"truth_sha256"] == digest(generated["truth"])
            for arm in METHODS:
                metric = r2_score if name in REGRESSION else balanced_accuracy_score
                score = metric(generated["truth"], generated[arm+"_pred"])
                np.testing.assert_allclose(score, method.loc[arm,"score"], atol=1e-13, rtol=0)
                record["score_"+arm] = score
                if name == "secom":
                    ap = average_precision_score(generated["truth"], generated[arm+"_decision"])
                    np.testing.assert_allclose(ap, method.loc[arm,"ap"], atol=1e-13, rtol=0)
                    record["ap_"+arm] = ap
        for arm in METHODS:
            ratios = []
            for process in (1,2):
                times = part[part.process==process].set_index("method").total_ms
                ratios.append(times["vote_width"]/times[arm])
                record[f"ms{process}_{arm}"] = times[arm]
            gap = record["score_vote_width"]-record["score_"+arm]
            record["gap_"+arm] = gap
            record["ratio_"+arm] = max(ratios)
            record["faster_"+arm] = max(ratios) < 1
            record["better_"+arm] = gap > 1e-12
            record["both_"+arm] = record["faster_"+arm] and record["better_"+arm]
        record["removed"] = record["k"] < record["p"]
        record["cut_faster"] = record["removed"] and record["faster_full"]
        record["cut_faster_within_002"] = record["cut_faster"] and record["gap_full"] >= -.02-1e-12
        cases.append(record)
    rows = pd.DataFrame(cases)
    assert set(frozen.index) == set(rows[KEY].itertuples(index=False,name=None))
    rows.to_csv(out/"cases.csv",index=False)
    frozen.reset_index().to_csv(out/"inputs.csv",index=False)
    comparisons = ["full","geometric","F","random"]
    summary = dict(n=("k","size"),k_min=("k","min"),k_max=("k","max"),reduced=("removed","sum"),
                   faster_and_cut=("cut_faster","sum"),within_002_faster_and_cut=("cut_faster_within_002","sum"),
                   score=("score_vote_width","mean"))
    for arm in comparisons:
        summary.update({f"score_{arm}":("score_"+arm,"mean"), f"gap_{arm}":("gap_"+arm,"mean"),
                        f"worst_gap_{arm}":("gap_"+arm,"min"), f"ratio_{arm}":("ratio_"+arm,"median"),
                        f"wins_{arm}":("better_"+arm,"sum"), f"faster_{arm}":("faster_"+arm,"sum"),
                        f"both_{arm}":("both_"+arm,"sum")})
    rows.groupby(["task","dataset","p"]).agg(**summary).to_csv(out/"sources.csv")
    rows.groupby("task").agg(**summary).to_csv(out/"tasks.csv")
    f_check = pd.read_csv(first/"f-audit.csv")
    pd.testing.assert_frame_equal(f_check, pd.read_csv(second/"f-audit.csv"))
    f_check.to_csv(out/"f-audit.csv",index=False)
    (out/"verification.json").write_text(json.dumps(dict(
        sources=rows.dataset.nunique(), configurations=len(rows), distinct_train_test=frozen.test_sha256.nunique(),
        total_timings=len(data), cross_process_arrays=arrays_verified, scored_predictions=len(rows)*len(METHODS),
        exact_F_sets=int(f_check.exact_F_set.sum()), max_F_boundary_violation=f_check.independent_F_gap.max()
    ),indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first","second","out"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    summarize(args.first,args.second,args.out)
