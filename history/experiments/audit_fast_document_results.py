"""Compare two independent timings on identical document folds and an external mail batch."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]
HALF = ["F_50","chi_50","random_50"]
MATCH = ["F_match","chi_match","random_match"]
FLAGS = ["joint", "joint_5pct", "joint_10pct", "joint_no_loss", "joint_extra_width",
         "both_windows", "direct_joint", "direct_time_win", "direct_quality", "direct_width"]


def read_pass(path):
    raw = pd.read_csv(path/"trials.csv")
    cols = KEY + ["method"]
    methods = set(raw.method)
    assert {"fast_vote","full","F_match","chi_match","random_match","df_match",
            "df2","F_50","chi_50","random_50","pooled_match","pooled_50"}.issubset(methods)
    assert not raw.duplicated(cols+["repeat"]).any()
    assert not raw.duplicated(KEY+["repeat","position"]).any()
    assert raw.groupby(cols).position.nunique().eq(len(methods)).all()
    assert raw.groupby(cols)[["score","p","width"]].nunique().eq(1).all().all()
    assert raw.select_ms.ge(0.).all() and raw.total_ms.gt(0.).all()
    groups = raw.groupby(cols)
    f = groups.agg(width=("width","first"), p=("p","first"), score=("score","first"),
                   ms=("total_ms","median"), select_ms=("select_ms","median"),
                   refit_ms=("refit_ms","median"), timed_repeats=("repeat","size"))
    f["q25_ms"] = groups.total_ms.quantile(.25)
    f["q75_ms"] = groups.total_ms.quantile(.75)
    f["rel_iqr"] = (f.q75_ms-f.q25_ms)/f.ms
    n = len(methods)
    assert f.timed_repeats.eq(n).all() and len(f)==20*n
    for field, mask in (("early_ms",raw.repeat<n//2),("late_ms",raw.repeat>=n-n//2)):
        f[field] = raw[mask].groupby(cols).total_ms.median()
    saved = pd.read_csv(path/"cases.csv").set_index(cols).sort_index()
    pd.testing.assert_index_equal(f.index,saved.index)
    np.testing.assert_allclose(f.ms,saved.total_ms,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(f.width,saved.width)
    np.testing.assert_allclose(f.score,saved.score,atol=1e-12,rtol=0)
    meta = pd.read_csv(path/"panels.csv").set_index(KEY).sort_index()
    assert meta.suite.value_counts().to_dict() == {"uci":17,"mail":3}
    f = f.join(meta[["suite","task","common_prepare_ms"]],validate="many_to_one")
    f["kept_fraction"] = f.width/f.p
    f["pipeline_with_common_ms"] = f.ms+f.common_prepare_ms
    f["ms_pass"] = path.name
    return f


def score_draws(root):
    data = pd.concat([pd.read_csv(root/d/"random-draws.csv") for d in
                      ("document-vote-20261117b","mail-vote-20261117a")],ignore_index=True)
    assert not data.duplicated(KEY+["draw"]).any()
    return data


def contrasts(f, random):
    methods = set(f.index.get_level_values("method"))
    vote = f.xs("fast_vote",level="method")
    groups = {arm:[arm] for arm in sorted(methods-{"fast_vote"})}
    groups.update(half_F_random=["F_50","random_50"], half_F_chi_random=HALF,
                  half_including_pooled=HALF+["pooled_50"], matched_F_chi=["F_match","chi_match"],
                  matched_supervised=["F_match","chi_match","pooled_match"], matched_F_chi_random=MATCH,
                  matched_with_pool_and_frequency=MATCH+["pooled_match","df_match"],
                  all_external=sorted(methods-{"fast_vote","full","document_vote"}))
    draw = random.merge(vote[["score","width"]].rename(columns={"score":"a_score","width":"a_width"}),
                        left_on=KEY,right_index=True,validate="many_to_one")
    assert draw.width.eq(draw.a_width).all() and len(draw)==400
    draw["strict_win"] = draw.a_score-draw.score>1e-12
    np.testing.assert_array_equal(draw.strict_win,draw.win)
    safe = draw.groupby(KEY).agg(random_wins=("strict_win","sum"),draws=("draw","size"),
                                 random_median=("score","median"))
    assert safe.draws.eq(20).all()
    output=[]
    for label, names in groups.items():
        arms = list(dict.fromkeys(["full",*names]))
        direct = f[f.index.get_level_values("method").isin(names)].groupby(level=KEY)
        all_base = f[f.index.get_level_values("method").isin(arms)].groupby(level=KEY)
        t = vote.join(safe)
        t["group"] = label
        t["controls"] = ",".join(arms)
        t["score_gap"] = vote.score-all_base.score.max()
        t["ratio"] = vote.ms/all_base.ms.min()
        t["width_ratio"] = vote.width/all_base.width.min()
        t["ratio_with_common"] = ((vote.ms+vote.common_prepare_ms)/
                                  (all_base.ms.min()+vote.common_prepare_ms))
        t["direct_score_gap"] = vote.score-direct.score.max()
        t["direct_time_ratio"] = vote.ms/direct.ms.min()
        t["direct_width_ratio"] = vote.width/direct.width.min()
        t["direct_width"] = vote.width<=direct.width.min()
        t["direct_quality"] = t.direct_score_gap>=-.02
        t["direct_time_win"] = t.direct_time_ratio<1
        t["direct_joint"] = (vote.width<vote.p) & t.direct_width & t.direct_quality & t.direct_time_win
        guard = (vote.width<vote.p) & (t.score_gap>=-.02) & (t.ratio<1)
        t["joint"] = guard & (vote.width<=all_base.width.min())
        t["joint_5pct"] = t.joint & (t.ratio<=.95)
        t["joint_10pct"] = t.joint & (t.ratio<=.90)
        t["joint_no_loss"] = t.joint & (t.score_gap>=0.)
        t["joint_extra_width"] = guard & (10*vote.width<=11*all_base.width.min())
        t["both_windows"] = t.joint & (vote.early_ms<all_base.early_ms.min()) & (vote.late_ms<all_base.late_ms.min())
        output.append(t.reset_index())
    return pd.concat(output,ignore_index=True)


def repeat_intersection(cases):
    keys = [*KEY,"group"]
    g=cases.groupby(keys)
    assert g.size().eq(2).all() and g[["score","p","width","random_wins"]].nunique().eq(1).all().all()
    b = g.first()
    for field in FLAGS:
        b[field] = g[field].all()
    for field in ["ratio","direct_time_ratio","ratio_with_common"]:
        b[field]=g[field].max()
    b["ms"] = g.ms.mean()
    b["select_ms"] = g.select_ms.mean()
    b["rel_iqr"] = g.rel_iqr.max()
    b["repeat_time_difference_pct"] = (g.ms.max()-g.ms.min())/g.ms.mean()
    b["ms_pass"] = "both"
    return b.reset_index()


def aggregate(frame, fields):
    g=frame.groupby(fields)
    return g.agg(configs=("width","size"),distinct_folds=("held_fold","nunique"), sources=("dataset","nunique"),
                 **{c:(c,"sum") for c in FLAGS}, votes_above_random75=("strong_votes","sum"),
                 min_random_wins=("random_wins","min"), median_keep=("kept_fraction","median"),
                 median_score=("score","median"), median_p=("p","median"), median_width=("width","median"),
                 median_total_ms=("ms","median"), median_select_ms=("select_ms","median"),
                 median_within_run_relative_iqr=("rel_iqr","median"),
                 median_between_run_relative_difference=("repeat_time_difference_pct","median"),
                 median_joint_time_ratio=("ratio","median"),median_worst_time_with_prepare=("ratio_with_common","median"),
                 median_direct_score_delta=("direct_score_gap","median"),
                 median_direct_time_ratio=("direct_time_ratio","median"),
                 median_direct_width_ratio=("direct_width_ratio","median"))


def main(args):
    root = args.root
    data = []
    draws = score_draws(root)
    for name in args.runs:
        data.append(read_pass(root/name))
    pd.testing.assert_series_equal(data[0].width,data[1].width)
    np.testing.assert_allclose(data[0].score,data[1].score,rtol=0,atol=1e-12)
    traces=sorted((root/args.runs[0]).glob("*.npz"))
    assert len(traces)==20
    for path in traces:
        with np.load(path) as first, np.load(root/args.runs[1]/path.name) as second:
            assert set(first.files)==set(second.files)
            for field in first:
                np.testing.assert_array_equal(first[field],second[field],err_msg=path.name+field)
    methods = pd.concat(data).reset_index()
    cases = pd.concat([contrasts(f,draws) for f in data],ignore_index=True)
    both = repeat_intersection(cases)
    cases = pd.concat([cases,both],ignore_index=True)
    cases["held_fold"] = cases.dataset+"-"+cases.fold.astype(str)
    cases["strong_votes"] = cases.random_wins>=15
    args.output.mkdir(parents=True,exist_ok=True)
    methods.to_csv(args.output/"method-cases.csv",index=False)
    cases.to_csv(args.output/"cases.csv",index=False)
    aggregate(cases,["ms_pass","suite","task","dataset","group"]).to_csv(args.output/"by-source.csv")
    aggregate(cases,["ms_pass","suite","group"]).to_csv(args.output/"summary.csv")
    methods.groupby(["ms_pass","suite","dataset","method"]).agg(configs=("width","size"),median_p=("p","median"),
        median_k=("width","median"), median_keep=("kept_fraction","median"),median_BA=("score","median"),
        median_time_ms=("ms","median"),median_select_ms=("select_ms","median"),median_refit_ms=("refit_ms","median"),
        median_rel_iqr=("rel_iqr","median")).to_csv(args.output/"methods-by-source.csv")
    # Pick a UCI illustration by increasing dataset/fold, among successful
    # folds under the more demanding group of half and matched supervised arms.
    eligible = both[both.suite.eq("uci")].pivot(index=KEY,columns="group",values="joint").sort_index()
    choices = eligible.index[eligible.half_including_pooled & eligible.matched_supervised]
    if not len(choices):
        choices = eligible.index[eligible.half_including_pooled]
    if len(choices):
        idx = choices[0]
        methods.set_index(KEY).loc[[idx]].to_csv(args.output/"example.csv")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("results/minpi-study"))
    parser.add_argument("--runs",nargs=2,default=["fast-document-20261117a","fast-document-20261117b"])
    parser.add_argument("--output",type=Path,default=Path("results/minpi-study/fast-document-comparison-20261117a"))
    main(parser.parse_args())
