"""Combine paired end-to-end records and retain separate timing batches and metrics."""
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence"
KEY = ["cohort", "dataset", "fold", "seed"]
HASH = ["train_sha", "target_sha", "test_sha", "truth_sha"]
ARMS = ("old", "F", "random", "full")
FIELDS = ("score", "k", "select_ms1", "total_ms1", "select_ms2", "total_ms2")
EPS = 1e-12
BANDS = ["≤32", "33–128", "129–512", "513–2048", "2049–10000", ">10000"]


def combine():
    wide = pd.read_csv(OUT / "wide-method-cases.csv", keep_default_na=False)
    old_run = pd.read_csv(HERE.parent / "block_selection/evidence/method-cases.csv", keep_default_na=False)
    archive = pd.read_csv(HERE.parent / "lowdim/evidence/all-cases.csv", keep_default_na=False)
    low = old_run[(old_run.p <= 32) & old_run.method.ne("pair")].copy()
    low["method"] = low.method.replace({"block": "wide", "current": "old"})
    low = low.merge(archive[KEY + ["source", "source_kind"]], how="left", on=KEY, validate="many_to_one")
    add = low.cohort.isin(["external", "fresh"])
    low.loc[add, "source"] = low.loc[add, "dataset"]
    low.loc[add, "source_kind"] = "observational"
    low.loc[low.dataset.eq("uci_magic_gamma"), "source_kind"] = "simulated_observatory"
    low["width_cap"] = low.set_index(KEY).index.map(
        low[low.method.eq("old")].set_index(KEY).k.to_dict())
    low["screen_width"] = low.p
    low["route"] = "original_block"
    low["timing_batch"] = "block-completion-20261119a"
    wide["timing_batch"] = "wide_replay"
    long = pd.concat([low, wide], ignore_index=True)
    long["metric"] = long.metric.str.upper().replace({"BALANCED_ACCURACY": "BA", "ROC_AUC": "AUC"})
    assert len(wide) == 331 * 6 and len(low) == 74 * 5 and not long.duplicated(KEY + ["method"]).any()
    assert long.source.notna().all() and long.source.ne("").all()
    # Include every historical key once and all six external datasets.
    focal = long[long.method.eq("wide")].set_index(KEY)
    assert len(focal) == 405 and set(archive.set_index(KEY).index) <= set(focal.index)
    assert focal.p.max() == 54877 and focal.p.min() == 4
    for h in HASH:
        assert focal[h].str.fullmatch("[a-fA-F0-9]{64}").all()
    meta = ["p", "task", "metric", "n_train", "n_test", "source", "source_kind", "timing_batch",
            "width_cap", "screen_width", "route", *HASH]
    data = focal[meta].copy()
    for a in ("wide", *ARMS, "cap"):
        one = long[long.method.eq(a)].set_index(KEY)
        data = data.join(one[list(FIELDS)].add_suffix("_" + a), how="left", validate="one_to_one")
    assert (data.k_wide == data.k_F).all() and (data.k_wide == data.k_random).all()
    assert (data.k_full == data.p).all() and data.loc[data.p <= 32, "k_cap"].isna().all()
    for a in ARMS:
        data["gap_" + a] = data.score_wide - data["score_" + a]
        data["ratio_" + a] = np.maximum(data.total_ms1_wide / data["total_ms1_" + a],
                                        data.total_ms2_wide / data["total_ms2_" + a])
    data["retained"] = data.k_wide / data.p
    data["width_band"] = pd.cut(data.p, [0, 32, 128, 512, 2048, 10000, np.inf], labels=BANDS)
    data = data.reset_index()
    data.to_csv(OUT / "all-paired.csv", index=False)
    return data


def comparison(g, a):
    gap = g["gap_" + a]
    ratio = g["ratio_" + a]
    cut = g.k_wide < g.p
    joint = cut & (ratio < 1)
    return dict(n=len(g), cut=int(cut.sum()), score_up=int((gap > EPS).sum()),
                score_equal=int((gap.abs() <= EPS).sum()), score_down=int((gap < -EPS).sum()),
                cut_faster=int(joint.sum()), strict_up_faster=int((joint & (gap > EPS)).sum()),
                no_loss_faster=int((joint & (gap >= -EPS)).sum()),
                within_001=int((joint & (gap >= -.01-EPS)).sum()),
                within_002=int((joint & (gap >= -.02-EPS)).sum()),
                within_005=int((joint & (gap >= -.05-EPS)).sum()),
                median_ratio=ratio.median(), median_retained=g.retained.median(),
                median_k=g.k_wide.median())


def groups(d):
    wide = d[d.p > 32]
    yield "all", d
    yield "all_wide", wide
    yield "screened", wide[wide.p > 128]
    yield "wide_dense", wide[wide.cohort.ne("text")]
    for band, g in d.groupby("width_band", observed=True):
        yield "p_" + str(band), g
    for task, g in wide.groupby("task"):
        yield "wide_" + task, g
    for kind, g in wide.groupby("source_kind"):
        yield "wide_source_" + kind, g
    for mechanism in ("proxy", "diffuse"):
        simulation = wide[wide.dataset.str.startswith("sim_" + mechanism + "_")]
        for task, g in simulation.groupby("task"):
            yield mechanism + "_" + task, g
    for cohort, g in d.groupby("cohort"):
        yield "cohort_" + cohort, g


def summarize():
    data = combine()
    summary, joint, budgets, ablation, weighted, versus_full = [], [], [], [], [], []
    for name, g in groups(data):
        flags = pd.DataFrame(index=g.index)
        for arm in ARMS:
            summary.append(dict(scope=name, comparator=arm, **comparison(g, arm)))
            flags[arm] = g["ratio_" + arm] < 1
        for controls in (("F", "random"), ARMS):
            fast = flags[list(controls)].all(axis=1) & (g.k_wide < g.p)
            gap = g[["gap_" + c for c in controls]].min(axis=1)
            joint.append(dict(scope=name, comparators="+".join(controls), n=len(g),
                              faster_all=int(fast.sum()), no_loss=int((fast & (gap >= -EPS)).sum()),
                              within_002=int((fast & (gap >= -.02-EPS)).sum())))
        digest = g[HASH].copy()
        digest["cut_faster"] = (g.k_wide < g.p) & (g.ratio_full < 1)
        digest["within_002"] = digest.cut_faster & (g.gap_full >= -.02-EPS)
        by_input = digest.groupby(HASH)[["cut_faster", "within_002"]].mean()
        weighted.append(dict(scope=name, unique_inputs=len(by_input), **by_input.mean().to_dict()))
        for arm in ("wide", "cap", *ARMS):
            subset = g[g["k_" + arm].notna()]
            if subset.empty:
                continue
            select = (subset["select_ms1_" + arm] + subset["select_ms2_" + arm]) / 2
            total = (subset["total_ms1_" + arm] + subset["total_ms2_" + arm]) / 2
            budgets.append(dict(scope=name, method=arm, n=len(subset), k_median=subset["k_"+arm].median(),
                                select_ms=select.median(), total_ms=total.median()))
            if arm != "full":
                gap = subset["score_" + arm] - subset.score_full
                ratio = np.maximum(subset["total_ms1_" + arm] / subset.total_ms1_full,
                                   subset["total_ms2_" + arm] / subset.total_ms2_full)
                cut = subset["k_" + arm] < subset.p
                faster_cut = cut & (ratio < 1)
                versus_full.append(dict(scope=name, method=arm, n=len(subset), cut=cut.sum(),
                                        cut_faster=faster_cut.sum(), no_loss=(faster_cut & (gap >= -EPS)).sum(),
                                        within_002=(faster_cut & (gap >= -.02-EPS)).sum(),
                                        within_002_5pct=(cut & (ratio < .95) & (gap >= -.02-EPS)).sum(),
                                        median_time_ratio=ratio.median()))
        available = g[g.k_cap.notna()]
        for changed, subset in (("all", available), ("kept_more", available[available.k_cap > available.k_wide])):
            if subset.empty:
                continue
            gap = subset.score_cap - subset.score_wide
            ratio = np.maximum(subset.total_ms1_cap / subset.total_ms1_wide,
                               subset.total_ms2_cap / subset.total_ms2_wide)
            fulldiff = subset.score_cap - subset.score_full
            fast_full = ((subset.k_cap < subset.p) & (subset.total_ms1_cap < subset.total_ms1_full)
                         & (subset.total_ms2_cap < subset.total_ms2_full))
            ablation.append(dict(scope=name, subset=changed, n=len(subset), k_adaptive=subset.k_wide.median(),
                                 k_filled=subset.k_cap.median(), score_up=int((gap > EPS).sum()),
                                 same=int((gap.abs() <= EPS).sum()), score_down=int((gap < -EPS).sum()),
                                 median_cap_to_adaptive_time=ratio.median(), faster_full=int(fast_full.sum()),
                                 within_002_full=int((fast_full & (fulldiff >= -.02-EPS)).sum())))
    for name, rows in [("comparisons", summary), ("simultaneous", joint), ("absolute-costs", budgets),
                       ("more-features", ablation), ("input-weighted", weighted),
                       ("pipelines-vs-full", versus_full)]:
        pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)

    sources, tradeoffs = [], []
    for key, g in data.groupby(["cohort", "dataset", "metric"]):
        for a in ARMS:
            sources.append(dict(zip(["cohort", "dataset", "metric"], key), comparator=a,
                                p=g.p.iloc[0], mean_score=g.score_wide.mean(),
                                mean_control=g["score_"+a].mean(), mean_gap=g["gap_"+a].mean(),
                                **comparison(g, a)))
    for name, g in groups(data):
        for metric, part in g.groupby("metric"):
            for a in ARMS:
                gap = part["gap_"+a]
                tradeoffs.append(dict(scope=name, metric=metric, comparator=a, n=len(part),
                                      median_gap=gap.median(), mean_gap=gap.mean(), min_gap=gap.min(),
                                      max_gap=gap.max(), median_time_ratio=part["ratio_"+a].median()))
    pd.DataFrame(sources).to_csv(OUT / "by-source.csv", index=False)
    pd.DataFrame(tradeoffs).to_csv(OUT / "by-metric.csv", index=False)
    print(len(data), "paired cases", data[HASH].drop_duplicates().shape[0], "distinct inputs")


if __name__ == "__main__":
    summarize()
