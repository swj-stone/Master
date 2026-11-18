"""Audit held-out predictions and report the matched compact-selection pilot."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.test.audit_piv3_lite import _fails, _read, _require, _score, _sha, _subset, _verify_scores
from experiments.test.piv3_raw_splits import DEFAULT_ROOT as INPUT, PROJECT
from experiments.test.run_piv3_lite_six import DATASETS
from experiments.test.report_piv3_lite_six import csv_table, losses
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


METHODS = ("tree_min", "tree_mean", "compact_gain", "compact_pfi",
           "pair_min", "pair_mean", "pair_relief", "tree_min_pfi", "pair_min_pfi")
BASELINES = ("full", "pfi", "gain", "pi", "pi_pfi")
PAIR_NAMES = tuple(("legacy_"+m, "legacy_"+m) for m in BASELINES) + (
    ("compact_gain", "compact_gain"), ("compact_pfi","compact_pfi"), ("pair_relief","pair_relief"))
TIMED = ("tree_min","pair_min","tree_min_pfi","pair_min_pfi")


def metrics_text(scores):
    if "r2" in scores:
        return f"{scores['r2']:.4f} / {scores['mae']:.4f}"
    s = f"{scores['auc']:.4f} / {100*scores['accuracy']:.2f}%"
    return s + (f" / {scores['ap']:.4f}" if "ap" in scores else "")


def route_info(folder, d, m, metadata, valid_y, guard_idx, test_y):
    timer = _read(folder/"watchdog.json")
    state = timer["status"]
    _require(state in {"completed", "timed_out"} and 0 < timer["limit_s"] <= 300., "Unexpected deadline/status")
    if state == "timed_out":
        _require(timer["kill_requested_s"] is not None, "Missing timer kill")
        return None
    r, frozen = _read(folder/"result.json"), _read(folder/"frozen.json")
    _require(timer["result"] == r and timer["exit_code"] == 0 and timer["kill_requested_s"] is None,
             "Watchdog or result changed")
    _require(0 < r["elapsed_s"] <= timer["elapsed_s"] + 1e-8 and r["elapsed_s"] <= 300.,
             "Invalid timing")
    _require(r["dataset"] == d and r["method"] == m and r["seed"] == 42, "Route identity")
    _require(r["selected"] == frozen["selected"] and len(r["selected"]) == r["k"] and
             _subset(r["selected"], metadata["features"]), "Feature support/order mismatch")
    _require(r["width"] == metadata["width"] and r["front_fits"] == 0 and r["fit_count"] >= 2,
             "Feature width or fit record mismatch")
    ident = canonical_hash(frozen)
    _require(ident == r["frozen_hash"], "Frozen support fingerprint mismatch")
    access = _read(folder/"test_access.json")
    _require(access["frozen_hash"] == ident and access["test_y_sha256"] == metadata["sha256"]["test_y.npy"]
             and access["test_x_sha256"] == metadata["sha256"]["test.npy"], "Test access provenance mismatch")
    rank_indices=np.setdiff1d(np.arange(len(valid_y)),guard_idx)
    _require(frozen["ranking_rows_hash"] == canonical_hash(rank_indices.tolist()) and
             frozen["guard_rows_hash"] == canonical_hash(guard_idx.tolist()),"Ranking/guard split differs")
    _require(frozen["policy"]=={k:v for k,v in metadata["policy"].items() if k!="utility"},"Model policy differs")
    with np.load(folder/"predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["guard_rows"],guard_idx)
        np.testing.assert_array_equal(z["y_guard"],valid_y[guard_idx])
        np.testing.assert_array_equal(z["y_test"],test_y)
        full = _score(z["y_guard"],z["guard_full"],d)
        guarded = _score(z["y_guard"],z["guard_selected"],d)
        test = _score(test_y,z["test_pred"],d)
    _verify_scores(r["validation_full"],full)
    _verify_scores(r["guard_selected"],guarded)
    _verify_scores(r["test_scores"],test)
    cols=metadata["features"]
    front,initial,checks=frozen["front"],frozen["initial"],frozen["checks"]
    diag=_read(folder/"front.json")
    order=diag["order"]
    downstream=m in ("tree_min_pfi","pair_min_pfi")
    _require(_subset(front,cols) and _subset(initial,front) and len(front)==r["front_k"] and
             len(initial)==r["initial_k"] and len(set(order))==len(order) and set(order)<=set(cols) and
             set(order[:len(front)])==set(front) and (downstream or initial==front),"Prefilter/ranking support")
    if m in ("tree_min","tree_mean","tree_min_pfi") and "tier_rates" in diag:
        rates=np.asarray(diag["tier_rates"])
        _require(rates.shape==(3,len(diag["vote_columns"])) and np.all(np.abs(rates)<=1),"Path vote shape")
        reduce=np.mean if m=="tree_mean" else np.min
        np.testing.assert_allclose(reduce(rates,axis=0),diag["vote_score"],rtol=0,atol=1e-14)
        idx=np.lexsort((np.arange(rates.shape[1]),-reduce(diag["relative_gains"],axis=0),-reduce(rates,axis=0)))
        _require(order==[diag["vote_columns"][i] for i in idx],"Path order inconsistent with recorded votes")
    if m in ("pair_min","pair_mean","pair_min_pfi") and "tier_rates" in diag:
        rates=np.asarray(diag["tier_rates"])
        _require(rates.shape==(3,len(cols)) and np.all(np.abs(rates)<=1+1e-9),"Pair votes shape")
        reduce=np.mean if m=="pair_mean" else np.min
        np.testing.assert_allclose(reduce(rates,axis=0),diag["aggregate_scores"],rtol=0,atol=1e-9)
    if "masked_proposals" in diag:
        endpoint=diag["exact_endpoint_width"]
        observed={endpoint:diag["baseline"]}
        candidates=[endpoint]
        for entry in diag["masked_proposals"]:
            n=entry["k"]
            _require(1<=n<endpoint and n not in observed,"Repeated masked size or bad endpoint")
            observed[n]=entry["scores"]
            ok=not _fails(diag["baseline"],entry["scores"],diag["mask_tolerance"],diag["mask_tolerance"])
            _require(ok==entry["passes"],"Masked feasibility record differs")
            if ok:
                candidates.append(n)
        _require(min(candidates)==r["front_k"]==diag["k"] and len(observed)<=13,
                 "Masked prefix was not smallest observed feasible size")
        _verify_scores(observed[diag["k"]],diag["selected_mask_scores"])
    repair_members=set(order[:2*len(front)])
    repair=front if downstream and initial!=front else [c for c in cols if c in repair_members]
    expected=[initial]
    if repair!=initial and repair!=cols:
        expected.append(repair)
    _require(1<=len(checks)<=len(expected) and r["checks"]==len(checks),"Guard attempt count")
    for pos,check in enumerate(checks):
        failed=_fails(full,check["metrics"],.03,.05)
        _require(check["names"]==expected[pos] and check["k"]==len(check["names"]) and
                 sorted(check["failed"])==failed,"Guard proposal or failure labels")
        _require(pos==len(checks)-1 or bool(failed),"Passed first proposal but kept searching")
    fallback=bool(checks[-1]["failed"])
    _require((not fallback or len(checks)==len(expected)) and r["fallback"]==fallback and
             r["selected"]==(cols if fallback else checks[-1]["names"]) and
             r["repaired"]==(len(checks)>1 and not fallback),"Guard delivery/fallback")
    _verify_scores(full if fallback else checks[-1]["metrics"],guarded)
    expected_guard=sum(c["names"]!=cols and not(downstream and c["names"]==front) for c in checks)
    _require(r["guard_fits"]==expected_guard and r["fit_count"]==2+int(downstream and front!=cols)+expected_guard,
             "Fitted model ledger")
    return dict(r,validation_selected=guarded,front=front)


def audit_and_report(root, allow_pending):
    root = Path(root)
    proto = _read(root/"protocol.json")
    for key, digest in proto.get("source_sha256",{}).items():
        _require(_sha(PROJECT/key) == digest, f"Changed source {key}")
    _require(set(proto["data"]) == set(DATASETS) and set(proto["methods"]) == set(METHODS) and
             proto["seed"]==42 and proto["limit_s"]==300.,
             "Experiment grid mismatch")
    legacy = root.parent/"PIv3-lite-raw-six-seed42"
    with (legacy/"timing-repeats/timing-median.csv").open(encoding="utf-8-sig",newline="") as f:
        times = {(s["dataset"],s["method"]):s for s in csv.DictReader(f)}
    _require(proto["benchmark_sha256"]==_sha(legacy/"protocol.json"),"Archived protocol identity")
    rows, comparisons, baselines, ablation, repetitions = [], [], [], [], []
    status = Counter()
    for d in DATASETS:
        metadata = _read(INPUT/d/"metadata.json")
        _require(proto["data"][d]["hash"]==_sha(INPUT/d/"metadata.json"),"Raw input metadata differs")
        vy = np.load(INPUT/d/"val_y.npy",allow_pickle=False)
        ty = np.load(INPUT/d/"test_y.npy",allow_pickle=False)
        _require(all(_sha(INPUT/d/name)==metadata["sha256"][name] for name in ("val_y.npy","test_y.npy")),
                 "Changed label file")
        _, guard = train_test_split(np.arange(len(vy)),test_size=.5,random_state=42,
                                    stratify=None if metadata["task"]=="regression" else vy)
        guard = np.sort(guard)
        records, clocks, games = {}, {}, {}
        for m in BASELINES:
            path = legacy/d/m
            r,timer = _read(path/"result.json"),_read(path/"watchdog.json")
            _require(timer["status"] == "completed" and timer["result"] == r, "Archived baseline changed")
            with np.load(path/"predictions.npz",allow_pickle=False) as z:
                np.testing.assert_array_equal(z["y_test"],ty)
                _verify_scores(r["test_scores"],_score(ty,z["test_pred"],d))
            key = "legacy_"+m
            records[key]=r
            time_row=times.get((d,m))
            clocks[key]=float(time_row["median_active_s"]) if time_row else r["elapsed_s"]
            baselines.append(dict(dataset=d, baseline=key, returned_k=r["k"], active_s=clocks[key],
                              time_measure="median_of_three" if time_row else "single",
                              cold_s=float(time_row["median_cold_s"]) if time_row else timer["elapsed_s"],
                              fits=r["fits"], fallback=r["fallback"], **r["test_scores"]))
        for m in METHODS:
            folder=root/d/m
            row=dict(dataset=d,method=m,width=metadata["width"],status="pending")
            if (folder/"watchdog.json").exists():
                timer=_read(folder/"watchdog.json")
                row.update(status=timer["status"],cold_s=timer["elapsed_s"])
                try:
                    r=route_info(folder,d,m,metadata,vy,guard,ty)
                except (ValueError, KeyError, AssertionError) as err:
                    raise ValueError(f"{d}/{m}: {err}") from err
                if r is not None:
                    records[m]=r
                    games[m]=_read(folder/"front.json")
                    obs=[(r["elapsed_s"],timer["elapsed_s"])]
                    spent=timer["elapsed_s"]
                    if m in TIMED:
                        repetitions.append(dict(dataset=d,method=m,repeat=1,status="completed",
                                                 active_s=r["elapsed_s"],cold_s=timer["elapsed_s"]))
                        for rep in (2,3):
                            dest=root/f"repeat-{rep}"/d/m
                            if not (dest/"watchdog.json").exists():
                                continue
                            extra=_read(dest/"watchdog.json")
                            state=extra["status"]
                            details=dict(dataset=d,method=m,repeat=rep,status=state,cold_s=extra["elapsed_s"])
                            if state=="budget_exhausted":
                                _require(300.-spent<=1 and extra["elapsed_s"]==0.,"Spurious exhausted budget")
                            else:
                                _require(abs(extra["limit_s"]-(300.-spent))<1e-6,"Repeat budget accounting")
                                copy=route_info(dest,d,m,metadata,vy,guard,ty)
                                if copy:
                                    for k in ("selected","front","fit_count","fallback","repaired","initial_k","frozen_hash"):
                                        _require(copy[k]==r[k],f"Repeat {rep} changed {k} for {d}/{m}")
                                    _verify_scores(copy["test_scores"],r["test_scores"])
                                    obs.append((copy["elapsed_s"],extra["elapsed_s"]))
                                    details["active_s"]=copy["elapsed_s"]
                            spent+=extra["elapsed_s"]
                            repetitions.append(details)
                    clocks[m]=median(a for a,_ in obs)
                    row.update(observations=len(obs),median_active_s=clocks[m],median_cold_s=median(b for _,b in obs),
                               cumulative_cold_s=spent)
                    _verify_scores(r["validation_full"],records["legacy_full"]["validation_full"])
                    for k in ("front_k","initial_k","k","fit_count","guard_fits","front_fits","selector_s","elapsed_s","fallback","repaired"):
                        row[k]=r[k]
                    row.update(uses_fitted_model=r["uses_fitted_model"],checks=r["checks"],front_s=r["front_s"],
                               downstream_s=r["rank_s"],reference_s=r["reference_s"],guard_s=r["guard_s"],final_s=r["final_s"])
                    for phase in ("validation_full","validation_selected","test_scores"):
                        row.update({phase+"_"+k:v for k,v in r[phase].items()})
                    row["validation"]=metrics_text(r["validation_selected"])
                    row["test"]=metrics_text(r["test_scores"])
            _require(allow_pending or row["status"]!="pending",f"Incomplete at {d}/{m}")
            rows.append(row)
            status[row["status"]]+=1
        for m in METHODS:
            if m not in records:
                continue
            r=records[m]
            for key,name in PAIR_NAMES:
                if key not in records or key==m:
                    continue
                other=records[key]
                loss = losses(r["test_scores"],other["test_scores"])
                relative=clocks[m]/clocks[key]
                comparisons.append(dict(dataset=d,method=m,baseline=name,k=r["k"],baseline_k=other["k"],
                                        smaller=r["k"]<other["k"],columns_delta=r["k"]-other["k"],
                                        active_s=clocks[m],baseline_s=clocks[key],time_ratio=relative,
                                        speedup=1/relative,at_most_10pct_slower=relative<=1.1,
                                        baseline_fits=other.get("fit_count",other.get("fits")),
                                        fits=r["fit_count"],worst_relative_loss=max(loss.values()),
                                        pass_all_5pct=max(loss.values())<=.05+1e-12,
                                        both_full_and_ref_5pct=(max(loss.values())<=.05+1e-12 and
                                            max(losses(r["test_scores"],records["legacy_full"]["test_scores"]).values())<=.05+1e-12),
                                        **{"relative_loss_"+k:v for k,v in loss.items()},
                                        **{"test_delta_"+k:r["test_scores"][k]-v for k,v in other["test_scores"].items()}))
        for first,second in (("tree_min","tree_mean"),("pair_min","pair_mean"),("tree_min","compact_gain"),
                             ("pair_min","pair_relief")):
            if first not in records or second not in records:
                continue
            a,b=records[first],records[second]
            x,y=set(a["selected"]),set(b["selected"])
            if second in ("tree_mean","pair_mean"):
                np.testing.assert_array_equal(games[first].get("tier_rates",[]),games[second].get("tier_rates",[]))
            cross=set(a["front"])&set(b["front"])
            combined=set(a["front"])|set(b["front"])
            ablation.append(dict(dataset=d,first=first,second=second,k_first=len(x),k_second=len(y),
                                 initial_k_first=a["initial_k"],initial_k_second=b["initial_k"],
                                 initial_screens_equal=a["front"]==b["front"],front_jaccard=len(cross)/len(combined),
                                 shared=len(x&y),union=len(x|y),jaccard=len(x&y)/len(x|y),
                                 final_masks_equal=x==y,
                                 **{"delta_"+k:v-b["test_scores"][k] for k,v in a["test_scores"].items()}))
    csv_table(root/"compact-routes.csv",rows)
    csv_table(root/"compact-baselines.csv",baselines)
    if comparisons:
        csv_table(root/"compact-comparisons.csv",comparisons)
    if ablation:
        csv_table(root/"compact-ablation.csv",ablation)
    if repetitions:
        csv_table(root/"compact-timing.csv",repetitions)
    joint = {(x["dataset"],x["method"]):x for x in comparisons if x["baseline"]=="legacy_pfi"}
    body=["# 紧凑特征选择实验","",
          "六个原始维度数据集，数据与模型 seed 42。活动耗时含读入、筛选、验证与最终重训，冷耗时另含进程启动。直接快速 PFI 用此前三次独立进程计时的中位数。新方法的活动及冷耗时列用已完成计时的中位数，筛选耗时取首次。比值大于 1 表示快于直接 PFI。验证为独立半区，测试列使用冻结子集在训练集与完整验证集合并重训后的预测。分类指标按 AUC / 准确率排列，欺诈集另列 AP，ISOLET 为宏平均 OvR AUC。Abalone 为零截断 R² / MAE。","",
          "|数据集|方法|状态|初始 / 交付列|验证 / 总拟合|筛选 / 全程 / 冷耗时（秒）|相对 PFI 加速倍数|验证指标|测试指标|相对全列及 PFI 各项损失 ≤5%|全列回退|",
          "|---|---|---|---:|---:|---:|---:|---|---|---|---|"]
    for r in rows:
        if r["status"]!="completed":
            body.append(f"|{r['dataset']}|{r['method']}|{r['status']}|—|—|—|—|—|—|—|—|")
            continue
        rel=joint[r["dataset"],r["method"]]
        body.append(f"|{r['dataset']}|{r['method']}|完成|{r['initial_k']} / {r['k']}|{r['guard_fits']} / {r['fit_count']}|"
                    f"{r['selector_s']:.2f} / {r['median_active_s']:.2f} / {r['median_cold_s']:.2f}|{rel['speedup']:.2f}|"
                    f"{r['validation']}|{r['test']}|{'是' if rel['both_full_and_ref_5pct'] else '否'}|{'是' if r['fallback'] else ('补列后通过' if r['repaired'] else '否')}|")
    body.extend(["","[原始对照指标](compact-baselines.csv) · [逐组指标与耗时](compact-routes.csv) · [各参照的逐指标差值](compact-comparisons.csv) · [min、mean 和简单排序的子集重合](compact-ablation.csv)"])
    (root/"compact-results.md").write_text("\n".join(body)+"\n",encoding="utf-8")
    aggregates=[]
    for method in METHODS:
        family=[v for v in comparisons if v["method"]==method and v["baseline"]=="legacy_pfi"]
        if not family:
            continue
        aggregates.append(dict(method=method,pairs=len(family),
            fewer_than_pfi=sum(v["smaller"] for v in family),
            faster_than_pfi=sum(v["time_ratio"]<1 for v in family),
            within_time_margin=sum(v["at_most_10pct_slower"] for v in family),
            quality_both=sum(v["both_full_and_ref_5pct"] for v in family),
            smaller_faster_quality=sum(v["smaller"] and v["time_ratio"]<1 and v["both_full_and_ref_5pct"] for v in family),
            smaller_time_margin_quality=sum(v["smaller"] and v["at_most_10pct_slower"] and v["both_full_and_ref_5pct"] for v in family),
            median_speedup=median(v["speedup"] for v in family)))
    if aggregates:
        csv_table(root/"compact-summary.csv",aggregates)
    atomic_write_json(root/"compact-audit.json",dict(status=dict(status),total=len(rows),
                      re_scored=status["completed"],pairwise_comparisons=len(comparisons),
                      timing_status=dict(Counter(v["status"] for v in repetitions)),
                      triplicates=sum(r.get("observations")==3 for r in rows)))
    return dict(status)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--allow-pending",action="store_true")
    args=p.parse_args()
    print(audit_and_report(args.run,args.allow_pending))
