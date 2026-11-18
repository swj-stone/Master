"""Combine the paired studies with one definition of elapsed time and score change."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ("new","previous","full","F","random")
INPUTS = {"core_evidence":126,"formal_evidence":40,"extra_evidence":149,"survey_evidence":33,"text_evidence":38,"fraud_evidence":1}
PRIMARY = ["spambase","digits","madelon","abalone","crimedata","gas","supercon","har","diabetes","isolet","secom",
           "breast_cancer","gisette","multiple_features","spoken_arabic_digit","fsdd"]


def standardized(folder,expected):
    # The archived random-label simulation is literally named 'null'.
    data=pd.read_csv(HERE/folder/"cases.csv",keep_default_na=False,na_values=[""])
    if expected is not None and len(data)!=expected:
        raise ValueError(f"{folder} contains {len(data)} cases (expected {expected})")
    if "cohort" not in data:
        data["cohort"]=folder.replace("_evidence","")
    if folder=="formal_evidence":
        ids=pd.read_csv(HERE/folder/"input-fingerprints.csv")
        data=data.merge(ids,on=["dataset","fold","seed"],how="left",validate="one_to_one")
    result=data[["cohort","dataset","fold","seed","task","metric","p","k"]].copy()
    for field in ("classes","n_train","n_test","family","source","train_sha","target_sha","test_sha","truth_sha","test_positive","prevalence"):
        if field in data:
            result[field]=data[field]
    result["source_kind"]=data.get("source_kind","observational")
    result.loc[result.dataset.eq("madelon"),"source_kind"]="artificial_challenge"
    if "family" in data:
        result.loc[data.family.eq("formal_simulation"),"source_kind"]="generated"
        result.loc[data.family.str.contains("semi|augment",na=False,case=False),"source_kind"]="augmented"
    result["source"]=result["source"].fillna(result.dataset) if "source" in result else result.dataset
    result["metric"]=result.metric.replace({"r2":"R2","balanced_accuracy":"BA","auc":"AUC","ba":"BA"})
    if "classes" in result:
        mask=result.task.eq("classification")
        if result.loc[mask,"classes"].isna().any():
            raise ValueError(f"missing class counts in {folder}")
        result.loc[mask,"task"]=np.where(result.loc[mask,"classes"]==2,"binary","multiclass")
    for arm in ARMS:
        for i in (1,2):
            dest=f"ms{i}_{arm}"
            col=dest if dest in data else f"total_ms{i}_{arm}"
            result[dest]=data[col]
            result[f"select_ms{i}_{arm}"]=data[f"select_ms{i}_{arm}"]
        result["score_"+arm]=data["score_"+arm]
        for prefix in ("ap","average_precision","fraud_recall","fraud_detected","auc","accuracy","balanced_accuracy","mae"):
            metric_col=prefix+"_"+arm
            if metric_col in data:
                result[metric_col]=data[metric_col]
        k=next((x for x in ("width_"+arm,"k_"+arm) if x in data),None)
        if k:
            result["k_"+arm]=data[k]
        elif arm=="full":
            result["k_full"]=result.p
        elif arm!="previous":
            result["k_"+arm]=result.k
        else:
            raise ValueError(f"missing old budget in {folder}")
    result["file_group"]=folder
    return result


TITLES={"original":"原 69 组","extra":"额外 57 组","formal":"原论文","four_stage_ridge":"四阶段 Ridge","dense":"稠密与语音",
        "spectra":"近红外光谱","mechanism":"模拟机制","archive_supplement":"调查、广告与原稿模拟","text":"稀疏文本与广告","conference_fraud":"信用卡历史扩展"}


def readable_sources(data,out):
    lines=["# 所有来源与协议的同场比较","",
           "每行是同一数据、同一协议、同一测试指标。得分为配置均值，时间为配置中位数，单位为毫秒。每个配置先对两个进程各取中位数，再平均，得到表中的时间。删列且加速要求两个进程内都比 Full 更快。相同数据在不同协议下保留独立行。",
           "","F 为同宽单变量 F 检验，随机为同宽均匀无放回抽列。分类 BA、AUC、按组 BA 及回归 R² 分开报告。配对分差和时间差另存于[全部配置](all-cases.csv)。",
           "","| 数据 | 协议 | 指标 | 组数 | 选列 k / 原 p | 新得分 | 旧得分 | Full 得分 | F 得分 | 随机得分 | 新总 ms | 旧总 ms | Full ms | F ms | 随机 ms | 删列且加速 |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data.itertuples():
        width=str(r.k_min) if r.k_min==r.k_max else f"{r.k_min}–{r.k_max}"
        original=str(r.p_min) if r.p_min==r.p_max else f"{r.p_min}–{r.p_max}"
        lines.append(f"| {r.dataset} | {TITLES.get(r.cohort,r.cohort)} | {r.metric} | {r.cases} | {width} / {original} | "
                     f"{r.mean_score_new:.4f} | {r.mean_score_previous:.4f} | {r.mean_score_full:.4f} | {r.mean_score_F:.4f} | {r.mean_score_random:.4f} | "
                     f"{r.median_total_ms_new:.3f} | {r.median_total_ms_previous:.3f} | {r.median_total_ms_full:.3f} | "
                     f"{r.median_total_ms_F:.3f} | {r.median_total_ms_random:.3f} | {r.cut_faster}/{r.cases} |")
    (out/"all-results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def derive(frame):
    fields={}
    for arm in ARMS:
        per_process=np.stack([frame[f"ms{i}_{arm}"] for i in (1,2)])
        selection=np.stack([frame[f"select_ms{i}_{arm}"] for i in (1,2)])
        assert (selection>=0).all() and (per_process>=selection).all()
        fields["total_ms_"+arm]=per_process.mean(axis=0)
        fields["select_ms_"+arm]=selection.mean(axis=0)
        if arm=="new":
            continue
        ratio=np.stack([frame[f"ms{i}_new"]/frame[f"ms{i}_{arm}"] for i in (1,2)])
        gap=frame.score_new-frame["score_"+arm]
        fields["gap_"+arm]=gap
        fields["delta_ms_"+arm]=fields["total_ms_new"]-fields["total_ms_"+arm]
        fields["worst_ratio_"+arm]=ratio.max(axis=0)
        fields["faster_"+arm]=ratio.max(axis=0)<1
        fields["better_"+arm]=gap>1e-12
        fields["tie_"+arm]=gap.abs()<=1e-12
        fields["better_faster_"+arm]=fields["faster_"+arm]&fields["better_"+arm]
        fields["no_loss_faster_"+arm]=fields["faster_"+arm]&(gap>=-1e-12)
    result=pd.concat([frame,pd.DataFrame(fields,index=frame.index)],axis=1)
    assert np.isfinite(result[["score_"+arm for arm in ARMS]].values).all()
    assert (result.k_F==result.k).all() and (result.k_random==result.k).all() and (result.k_full==result.p).all()
    assert ((result.k>0)&(result.k<=result.p)).all()
    result["cut"]=result.k<result.p
    result["cut_faster"]=result.cut & result.faster_full
    result["cut_faster_002"]=result.cut_faster & (result.gap_full>=-.02-1e-12)
    result["beat_both_matched"]=result.cut & result.better_faster_F & result.better_faster_random
    result["beat_both_and_full_speed"]=result.beat_both_matched & result.faster_full
    result["pareto_both_matched"]=result.cut & result.no_loss_faster_F & result.no_loss_faster_random
    result["pareto_both_and_full_speed"]=result.pareto_both_matched & result.faster_full
    return result


def summarize(frame,group):
    agg=dict(cases=("k","size"),p_min=("p","min"),p_max=("p","max"),k_min=("k","min"),k_max=("k","max"),
             cut=("cut","sum"),cut_faster=("cut_faster","sum"),cut_faster_002=("cut_faster_002","sum"),
             beat_both_matched=("beat_both_matched","sum"),beat_both_and_full_speed=("beat_both_and_full_speed","sum"),
             pareto_both_matched=("pareto_both_matched","sum"),pareto_both_and_full_speed=("pareto_both_and_full_speed","sum"))
    for arm in ARMS:
        agg["mean_score_"+arm]=("score_"+arm,"mean")
        for kind in ("total_ms","select_ms"):
            agg["median_"+kind+"_"+arm]=(kind+"_"+arm,"median")
        if arm=="new":
            continue
        for field,op in (("gap","mean"),("worst_ratio","median"),("delta_ms","median"),("faster","sum"),("better","sum"),("tie","sum"),("better_faster","sum"),("no_loss_faster","sum")):
            agg[field+"_"+arm]=(field+"_"+arm,op)
    return frame.groupby(group,as_index=False,dropna=False,sort=False).agg(**agg)


def run():
    dfs=[standardized(k,v) for k,v in INPUTS.items()]
    all_cases=derive(pd.concat(dfs,ignore_index=True))
    assert not all_cases.duplicated(["cohort","dataset","fold","seed"]).any()
    out=HERE/"evidence"
    out.mkdir(exist_ok=True)
    all_cases.to_csv(out/"all-cases.csv",index=False,encoding="utf-8-sig")
    per_source=summarize(all_cases,["cohort","source_kind","source","dataset","task","metric"]).sort_values(["dataset","cohort","metric"])
    per_source.to_csv(out/"all-sources.csv",index=False,encoding="utf-8-sig")
    readable_sources(per_source,out)
    hashes=["train_sha","target_sha","test_sha","truth_sha"]
    assert all_cases[hashes].notna().all().all(), "All inputs need traceable fingerprints"
    rates=["cut_faster","cut_faster_002","better_faster_F","better_faster_random","beat_both_and_full_speed"]
    groups=all_cases.groupby(hashes,sort=False)
    units=groups[rates].mean().add_suffix("_fraction")
    units["configurations"]=groups.size()
    for c in ("source","cohort","seed","metric"):
        units[c]=groups[c].agg(lambda a:" | ".join(map(str,pd.unique(a))))
    units.to_csv(out/"distinct-inputs.csv",encoding="utf-8-sig")
    # Same seed on the same input may also be a repeated protocol. Keep the historical
    # original 69 occurrence first, then supplement in the declared study order.
    priority=["original","extra","formal","dense","four_stage_ridge","spectra","mechanism","archive_supplement","text"]
    ordered=all_cases.assign(protocol_order=all_cases.cohort.map({c:i for i,c in enumerate(priority)}).fillna(len(priority)))
    repeated=ordered.sort_values("protocol_order",kind="stable").duplicated(hashes+["seed","metric"])
    first=ordered.loc[repeated.index[~repeated]]
    first.drop(columns="protocol_order").to_csv(out/"one-row-per-input-seed-metric.csv",index=False,encoding="utf-8-sig")
    # One protocol per named historical source for a compact index; all alternatives remain above.
    order={"original":0,"extra":1,"formal":2,"dense":3,"four_stage_ridge":4}
    compact=per_source[per_source.dataset.isin(PRIMARY)].copy()
    compact["priority"]=compact.cohort.map(order).fillna(99)
    compact=compact.sort_values(["dataset","priority"]).drop_duplicates("dataset")
    assert len(compact)==len(PRIMARY)
    compact.drop(columns="priority").to_csv(out/"prior-16-sources.csv",index=False,encoding="utf-8-sig")
    keys=["cohort","task","metric"]
    summarize(all_cases,keys).to_csv(out/"by-task-and-metric.csv",index=False,encoding="utf-8-sig")
    count_fields=["cut","cut_faster","cut_faster_002","faster_previous","better_previous",
                  "faster_F","better_F","better_faster_F","no_loss_faster_F",
                  "faster_random","better_random","better_faster_random","no_loss_faster_random",
                  "beat_both_matched","beat_both_and_full_speed","pareto_both_matched","pareto_both_and_full_speed"]
    totals=all_cases.groupby("cohort")[count_fields].sum()
    totals.insert(0,"cases",all_cases.groupby("cohort").size())
    totals.loc["ALL"]=totals.sum()
    totals.to_csv(out/"cohort-counts.csv",encoding="utf-8-sig")
    tradeoffs=all_cases[all_cases.cut_faster].groupby(["task","metric"]).agg(
        cases=("k","size"),median_gap=("gap_full","median"),min_gap=("gap_full","min"),
        median_paired_delta_ms=("delta_ms_full","median"),median_worst_time_ratio=("worst_ratio_full","median"),
        within_002=("cut_faster_002","sum"))
    tradeoffs.to_csv(out/"accuracy-time-among-faster.csv",encoding="utf-8-sig")
    focus=all_cases[(all_cases.p<=32)].copy()
    summarize(focus,["cohort","source","dataset","task","metric"]).to_csv(out/"low-dimensional.csv",index=False,encoding="utf-8-sig")
    print("Paired cases",len(all_cases),"source/protocol/metric entries",len(per_source))
    print(all_cases.groupby("cohort")[["cut","cut_faster","cut_faster_002","better_faster_F","better_faster_random","beat_both_and_full_speed"]].agg(["size","sum"]).to_string())


if __name__=="__main__":
    run()

