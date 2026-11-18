"""Audit compressed ridge pipelines against full, fixed-width and same-width baselines."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "fold", "seed"]
SUITES = {
    "next-vote-69": (["budget", "ridge"], 69),
    "next-vote-transfer": (["budget", "ridge"], 57),
    "capped-vote-69": (["capped"], 69),
    "capped-vote-transfer": (["capped"], 57),
    "nir-band-vote": (["band_vote"], 21),
    "calibrated-nir": (["calibrated"], 30),
    "allocation-nir": (["allocation"], 36),
}


def amounts(directory):
    trials = pd.read_csv(directory / "trials.csv")
    keys = KEY + ["method"]
    groups = trials.groupby(keys)
    assert not trials.duplicated(keys + ["repeat"]).any()
    assert groups[["width", "p", "score"]].nunique().eq(1).all().all()
    assert groups.size().nunique() == 1
    n = groups.size().iloc[0]
    assert set(trials.repeat) == set(range(n))
    assert trials.total_ms.gt(0).all() and trials.select_ms.ge(0).all()
    frame = groups.agg(width=("width","first"), p=("p","first"), score=("score","first"),
                       time=("total_ms","median"), select=("select_ms","median"))
    for name, mask in [("early", trials.repeat < n//2), ("late", trials.repeat >= n-n//2)]:
        frame[name] = trials[mask].groupby(keys).total_ms.median()
    saved = pd.read_csv(directory/"cases.csv").set_index(keys).sort_index()
    pd.testing.assert_index_equal(frame.index, saved.index)
    for actual, old in [("width","width"), ("p","p"), ("score","score"), ("time","total_ms")]:
        np.testing.assert_allclose(frame[actual], saved[old], rtol=0, atol=1e-11)
    replay = directory / "fresh-process-audit.csv"
    if replay.exists():
        checked = pd.read_csv(replay).set_index(keys).sort_index()
        pd.testing.assert_index_equal(checked.index, frame.index)
        for field in ("prediction_sha256","support_sha256"):
            assert checked[field].str.fullmatch("[0-9a-f]{64}").all()
    return frame, saved, replay.exists()


def random_quality(directory, candidate, vote):
    draws = pd.read_csv(directory/"random-draws.csv")
    if "candidate" in draws:
        draws = draws[draws.candidate == candidate]
    assert not draws.duplicated(KEY+["draw"]).any()
    score_column = "random_score" if "random_score" in draws else "score"
    draws = draws.merge(vote[["score", "width"]].rename(columns={"score":"audited_score","width":"audited_width"}),
                        left_on=KEY, right_index=True, validate="many_to_one")
    assert draws.width.eq(draws.audited_width).all()
    if "vote_score" in draws:
        np.testing.assert_allclose(draws.vote_score, draws.audited_score, rtol=0, atol=1e-12)
    draws["strict_win"] = draws.audited_score - draws[score_column] > 1e-12
    by_case = draws.groupby(KEY)[score_column]
    result = by_case.quantile([.25,.5,.75]).unstack()
    result.columns = ["random_q25","random_median","random_q75"]
    result["random_wins"] = draws.groupby(KEY).strict_win.sum()
    result["n_random"] = by_case.size()
    assert result.n_random.eq(20).all() and result.index.equals(vote.index)
    return result


def comparators(methods, candidate):
    domain = candidate in ("document_vote","band_vote","calibrated","allocation")
    f, r = ("F_match","random_match") if domain else ("F_"+candidate,"random_"+candidate)
    f50, r50 = ("F_50","random_50") if domain else ("F50","random50")
    old = [name for name in ("signal_auto","signal_vote") if name in methods]
    arms = methods - {"budget","ridge","capped"} if not domain else methods-{candidate}
    # The other candidate has its own dataset-dependent width and controls.
    if not domain:
        arms = {"full",f,r,f50,r50,*old}
    groups = {"full":["full"], "fixed_F50_random50":[f50,r50], "matched_F_random":[f,r],
              "fixed_and_previous":[f50,r50,*old], "all_baselines":sorted(arms-{"full"})}
    for name in sorted(arms-{"full"}):
        groups["vs_"+name] = [name]
    return {key:list(dict.fromkeys(["full", *names])) for key,names in groups.items()}, f, r


def audit_suite(directory, candidates, expected):
    times, saved, replayed = amounts(directory)
    methods = set(times.index.get_level_values("method"))
    tables = []
    for candidate in candidates:
        vote = times.xs(candidate, level="method")
        assert len(vote)==expected
        groups, f, r = comparators(methods,candidate)
        for matched in [f, r] + [m for m in methods if m.endswith("_match")]:
            np.testing.assert_array_equal(vote.width,times.xs(matched,level="method").width)
        draws = random_quality(directory,candidate,vote)
        raw = vote.rename(columns={"time":"total_ms","select":"select_ms"}).join(draws)
        raw["above_random_median"] = raw.score - raw.random_median > 1e-12
        raw["competitive_random"] = raw.random_wins >= 15
        raw["vote_minus_random_median"] = raw.score-raw.random_median
        raw["eligible"] = True
        raw["stage"] = ""
        panels = directory/"panels.csv"
        if panels.exists():
            meta = pd.read_csv(panels)
            if "eligible" in meta:
                raw["eligible"] = raw.index.get_level_values("dataset").map(meta.set_index("dataset").eligible)
                if "validation_set" in meta:
                    raw["stage"] = raw.index.get_level_values("dataset").map(meta.set_index("dataset").validation_set)
            elif candidate == "document_vote":
                raw["stage"] = meta.set_index(KEY).task
        raw["reduced"] = vote.width<vote.p
        raw["keep_fraction"] = vote.width/vote.p
        for name, arms in groups.items():
            # Each criterion must hold against every arm, including full.
            selected = times[times.index.get_level_values("method").isin(arms)].groupby(level=KEY)
            criteria = raw.copy()
            criteria["group"] = name
            criteria["baselines"] = ",".join(arms)
            criteria["candidate"] = candidate
            criteria["run"] = directory.name
            criteria["fresh_replay"] = replayed
            criteria["score_gap"] = vote.score-selected.score.max()
            criteria["time_ratio"] = vote.time/selected.time.min()
            criteria["width_ratio"] = vote.width/selected.width.min()
            criteria["passes_score"] = criteria.score_gap>=-.02
            criteria["passes_time"] = criteria.time_ratio<1
            criteria["passes_width"] = criteria.width_ratio<=1
            quality_time = criteria.reduced & criteria.passes_score & criteria.passes_time
            criteria["joint"] = quality_time & criteria.passes_width
            criteria["joint_allow_10pct_width"] = quality_time & (10*vote.width<=11*selected.width.min())
            criteria["joint_no_loss"] = criteria.joint & (criteria.score_gap>=0.)
            criteria["joint_time_5pct"] = criteria.joint & (criteria.time_ratio<=.95)
            criteria["joint_time_10pct"] = criteria.joint & (criteria.time_ratio<=.90)
            criteria["both_windows"] = criteria.joint & (vote.early<selected.early.min()) & (vote.late<selected.late.min())
            criteria["joint_beat_random75"] = criteria.joint & criteria.competitive_random
            if name=="full":
                np.testing.assert_array_equal(criteria.joint, saved.xs(candidate,level="method").joint)
            # Report direct comparator distributions, free of the full-baseline intersection.
            contrast = times.xs(arms[-1],level="method") if len(arms)<=2 else None
            for field,numerator,denominator in [("direct_score_gap","score","score"),("direct_time_ratio","time","time"),
                                                ("direct_width_ratio","width","width")]:
                if contrast is None:
                    criteria[field] = np.nan
                elif field=="direct_score_gap":
                    criteria[field] = vote[numerator]-contrast[denominator]
                else:
                    criteria[field] = vote[numerator]/contrast[denominator]
            criteria["direct_score_win"] = criteria.direct_score_gap>1e-12
            criteria["direct_score_tie"] = criteria.direct_score_gap.abs()<=1e-12
            criteria["direct_time_win"] = criteria.direct_time_ratio<1
            criteria["direct_joint"] = (criteria.reduced & (criteria.direct_score_gap>=-.02)
                                        & criteria.direct_time_win & (criteria.direct_width_ratio<=1))
            tables.append(criteria.reset_index())
    return pd.concat(tables,ignore_index=True)


def summary(frame, keys):
    counts = ("reduced","passes_score","passes_time","passes_width","joint","joint_no_loss",
              "joint_allow_10pct_width","joint_time_5pct","joint_time_10pct","both_windows",
              "above_random_median","joint_beat_random75","direct_score_win","direct_score_tie","direct_time_win",
              "direct_joint")
    grouped = frame.groupby(keys,sort=False)
    output = grouped.agg(configs=("width","size"), sources=("dataset","nunique"),
                         **{k:(k,"sum") for k in counts}, median_ratio=("time_ratio","median"),
                         median_keep=("keep_fraction","median"), median_worst_gap=("score_gap","median"),
                         median_vote_minus_random_median=("vote_minus_random_median","median"))
    for col in ("direct_time_ratio","direct_width_ratio","direct_score_gap"):
        for q in (.25,.5,.75):
            output[f"{col}_q{int(100*q)}"] = grouped[col].quantile(q)
    return output


def intersection_of_repeats(cases, names):
    """Summarize success on identical text cases in every independent clock run."""
    keys = ["candidate","group",*KEY]
    repeated = cases[cases.run.isin(names)].set_index(keys)
    assert repeated.groupby(level=keys).size().eq(len(names)).all()
    groups = repeated.groupby(level=keys)
    for field in ["score","p","width","random_median","random_wins","baselines"]:
        assert groups[field].nunique().eq(1).all()
    both = groups.first()
    for field in repeated.select_dtypes(include="bool"):
        both[field] = groups[field].all()
    for field in ("time_ratio","direct_time_ratio","total_ms","select_ms"):
        both[field] = groups[field].max()
    both["run"] = "document-vote-intersection"
    return both.reset_index()


def main(args):
    all_cases = []
    for suite,(candidates,n) in SUITES.items():
        directory = args.root/(suite+"-20261117a")
        all_cases.append(audit_suite(directory,candidates,n))
    if not args.without_document:
        for suffix in args.document_runs:
            directory=args.root/("document-vote-"+suffix)
            all_cases.append(audit_suite(directory,["document_vote"],17))
    cases = pd.concat(all_cases,ignore_index=True)
    args.output.mkdir(parents=True,exist_ok=True)
    cases.to_csv(args.output/"cases.csv",index=False)
    if not args.without_document and len(args.document_runs)>1:
        intersection = intersection_of_repeats(cases, ["document-vote-"+s for s in args.document_runs])
        cases = pd.concat([cases,intersection],ignore_index=True)
    keys = ["run","candidate","group"]
    summary(cases,keys+["dataset"]).to_csv(args.output/"by-source.csv")
    strata = [cases.assign(subset="all")]
    spectral = cases[cases.candidate.isin(["band_vote","calibrated","allocation"])]
    strata.append(spectral.assign(subset=np.where(spectral.eligible,"eligible","ineligible")))
    named = cases[cases.stage.ne("")]
    strata.append(named.assign(subset="stage: "+named.stage))
    summary(pd.concat(strata,ignore_index=True),keys+["subset"]).to_csv(args.output/"summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("results/minpi-study"))
    parser.add_argument("--output",type=Path,default=Path("results/minpi-study/next-vote-comparison-20261117a"))
    parser.add_argument("--without-document",action="store_true")
    parser.add_argument("--document-runs",nargs="+",default=["20261117b","20261117c"])
    main(parser.parse_args())
