"""Validate all timed rotations and summarize adaptive block selection at paired widths."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.block_selection.benchmark import ARCHIVE, ARMS, HERE, KEY


UNIT = KEY + ["method"]
EPS = 1e-12


def read_suite(run, suite, count):
    medians = []
    for process in (1,2):
        side = run / f"{suite}-pass{process}"
        info=json.loads((side/"complete.json").read_text())
        assert info["cases"]==count and info["process"]==process and tuple(info["arms"])==ARMS
        assert all(x["num_threads"]==1 for x in info["threads"])
        raw=pd.read_csv(side/"times.csv",keep_default_na=False)
        assert len(raw)==count*len(ARMS)**2
        assert not raw.duplicated(UNIT+["repeat"]).any() and not raw.duplicated(UNIT+["position"]).any()
        assert not raw.duplicated(KEY+["repeat","position"]).any() and (raw.total_ms>=raw.select_ms).all()
        medians.append(raw.groupby(UNIT)[["select_ms","total_ms"]].median().add_suffix(str(process)))
        quality=pd.read_csv(side/"quality.csv",keep_default_na=False).set_index(UNIT).sort_index()
        assert len(quality)==count*len(ARMS) and quality.index.is_unique
        if process==1:
            first=quality
        else:
            pd.testing.assert_frame_equal(first,quality,check_exact=True)
    location=run if suite=="fresh" else ARCHIVE
    inputs=pd.read_csv(location/f"{suite}-pass1/inputs.csv",keep_default_na=False)
    return first.join(medians,how="inner").reset_index().merge(inputs,on=KEY,validate="many_to_one")


def paired(methods):
    meta=KEY+["task","metric","p","n_train","n_test","train_sha","target_sha","test_sha","truth_sha"]
    cases=methods[meta].drop_duplicates().set_index(KEY)
    fields=["k","score","cols","select_ms1","select_ms2","total_ms1","total_ms2"]
    for arm in ARMS:
        one=methods[methods.method.eq(arm)].set_index(KEY)
        cases=cases.join(one[fields].add_suffix("_"+arm),how="inner")
    for arm in ARMS[1:]:
        cases["gap_"+arm] = cases.score_block - cases["score_"+arm]
        cases["ratio_"+arm] = np.maximum(cases.total_ms1_block / cases["total_ms1_"+arm],
                                         cases.total_ms2_block / cases["total_ms2_"+arm])
    assert cases.k_F.equals(cases.k_block) and cases.k_random.equals(cases.k_block)
    assert cases.k_current.equals(cases.k_pair) and cases.k_full.equals(cases.p)
    return cases.reset_index()


def vs_full(g,name):
    cut=g["k_"+name] < g.p
    diff=g["score_"+name]-g.score_full
    ratio=np.maximum(g["total_ms1_"+name]/g.total_ms1_full,g["total_ms2_"+name]/g.total_ms2_full)
    joint=cut & (ratio<1)
    return dict(method=name,n=len(g),cut=int(cut.sum()),cut_faster=int(joint.sum()),
                no_loss=int((joint&(diff>=-EPS)).sum()),within_001=int((joint&(diff>=-.01-EPS)).sum()),
                within_002=int((joint&(diff>=-.02-EPS)).sum()),within_005=int((joint&(diff>=-.05-EPS)).sum()),
                within_010=int((joint&(diff>=-.10-EPS)).sum()),median_ratio=ratio.median())


def vs_comparator(g,arm):
    diff,ratio=g["gap_"+arm],g["ratio_"+arm]
    cut=g.k_block < g.p
    joint=cut & (ratio<1)
    return dict(comparator=arm,n=len(g),smaller_width=int((g.k_block<g["k_"+arm]).sum()),
                equal_width=int((g.k_block==g["k_"+arm]).sum()),
                strict_score_up=int((diff>EPS).sum()),score_equal=int((diff.abs()<=EPS).sum()),
                score_down=int((diff< -EPS).sum()),cut_faster=int(joint.sum()),
                strict_up_faster=int((joint&(diff>EPS)).sum()),
                no_loss_faster=int((joint&(diff>=-EPS)).sum()),
                within_002_faster=int((joint&(diff>=-.02-EPS)).sum()),median_ratio=ratio.median(),
                margin_5pct_no_loss=int((cut&(ratio<.95)&(diff>=-EPS)).sum()))


def save(run,out):
    cases_long=pd.concat([read_suite(run,suite,n) for suite,n in (("archive",161),("external",12),("fresh",6))],ignore_index=True)
    out.mkdir(exist_ok=True,parents=True)
    cases_long.to_csv(out/"method-cases.csv",index=False)
    c=paired(cases_long)
    assert len(c)==179 and (c.p<=32).sum()==74
    c.to_csv(out/"paired-cases.csv",index=False)
    low=c[c.p<=32]
    scopes={"all":c,"low":low,"archived-low":low[~low.cohort.isin(["external","fresh"])],
            "external":low[low.cohort=="external"],"fresh":low[low.cohort=="fresh"],"wide":c[c.p>32],
            "low-K>2":low[low.k_block>2],"original":c[c.cohort=="original"],"extra":c[c.cohort=="extra"]}
    scopes.update({str(task)+"-low":g for task,g in low.groupby("task")})
    full,comparisons,per_source,tradeoffs,concurrent=[],[],[],[],[]
    for title,g in scopes.items():
        full.extend(dict(scope=title,**vs_full(g,name)) for name in ARMS[:-1])
        comparisons.extend(dict(scope=title,**vs_comparator(g,arm)) for arm in ARMS[1:])
        for arm in ARMS[1:]:
            for metric,sub in g.groupby("metric"):
                gap=sub["gap_"+arm]
                ratio=sub["ratio_"+arm]
                tradeoffs.append(dict(scope=title,comparator=arm,metric=metric,n=len(sub),
                    mean_score_gap=gap.mean(),median_score_gap=gap.median(),min_score_gap=gap.min(),
                    max_score_gap=gap.max(),median_time_ratio=ratio.median()))
        cut=g.k_block<g.p
        for comparators in [("F","random"),("F","random","full"),("F","random","full","current","pair")]:
            gap=g[["gap_"+a for a in comparators]].min(axis=1)
            ratio=g[["ratio_"+a for a in comparators]].max(axis=1)
            concurrent.append(dict(scope=title,comparators="+".join(comparators),n=len(g),
                                   cut_faster=int((cut&(ratio<1)).sum()),
                                   strict_up_faster=int((cut&(ratio<1)&(gap>EPS)).sum()),
                                   no_loss_faster=int((cut&(ratio<1)&(gap>=-EPS)).sum()),
                                   within_002_faster=int((cut&(ratio<1)&(gap>=-.02-EPS)).sum()),
                                   no_loss_5pct=int((cut&(ratio<.95)&(gap>=-EPS)).sum())))
    for (cohort,dataset,task,metric),g in low.groupby(["cohort","dataset","task","metric"]):
        meta=dict(cohort=cohort,dataset=dataset,task=task,metric=metric,min_k=g.k_block.min(),max_k=g.k_block.max())
        for arm in ARMS[1:]:
            per_source.append(dict(meta,**vs_comparator(g,arm),mean_score_delta=g["gap_"+arm].mean(),
                                   min_score_delta=g["gap_"+arm].min(),max_score_delta=g["gap_"+arm].max()))
    for name,rows in [("joint-vs-full",full),("paired-summary",comparisons),("by-source",per_source),
                      ("within-metric",tradeoffs),("simultaneous-controls",concurrent)]:
        pd.DataFrame(rows).to_csv(out/(name+".csv"),index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--out",type=Path,default=HERE/"evidence")
    args=p.parse_args()
    save(args.run,args.out)
