"""Independently check and summarize the Madelon PIv3, sampled and prefix controls."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_piv0_benchmark import read, sha
from experiments.run_thesis_sampling import ROOT
from experiments.test.audit_piv2_prefilter import metric_pair, near, ordered_subset, require
from experiments.test.report_madelon500_piv2 import verify_inputs
from experiments.test.report_piv3 import check_new_search, jaccard, passes
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


METHODS = ("piv3", "pfi", "piv3_pfi", "rf_prefix", "piv3_rf_prefix")
NAMES = {"piv3":"PIv3", "pfi":"PFI", "piv3_pfi":"PIv3＋PFI",
         "rf_prefix":"随机森林排序＋前缀遍历", "piv3_rf_prefix":"PIv3＋随机森林前缀遍历"}
SEEDS = (11, 23, 47)
TOLS = (.03, .05, .08)
HISTORICAL = ROOT/"results/test/thesis_sampling/PIv3-madelon500-adaptive-seed42"


def check_replay(folder, seed, selected, steps, archive, is_standalone):
    history = HISTORICAL/"piv3"/"tol-05"/f"s{seed}"
    frozen, result = read(history/"frozen.json"), read(history/"result.json")
    require(canonical_hash(frozen) == result["frozen_hash"] and frozen["selected"] == selected,
            f"{folder}: 5% PIv3 winner diverged from the previous experiment")
    old_steps = frozen["diagnostics"]["steps"]
    require(len(old_steps) == len(steps), "Different replay length")
    for step, old in zip(steps, old_steps, strict=True):
        require(step["proposed"] == old["proposed"] and step["accepted"] == old["accepted"],
                "Changed replay trajectory")
        near([step["auc"], step["accuracy"]], [old["auc"], old["accuracy"]])
    with np.load(history/"predictions.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(old["candidate_validation"], archive["candidate_validation"][:len(steps)])
        for key in (("validation","selected","validation_full","full") if is_standalone else ("validation_full","full")):
            np.testing.assert_array_equal(old[key], archive[key])


def inspect(folder, protocol, fingerprint, columns, labels, hashes, method, tol, seed, summary):
    clock = read(folder/"watchdog.json")
    row = dict(method=method, tolerance=tol, search_seed=seed, status=clock["status"], wall_s=clock["elapsed_s"])
    require(clock["limit_s"] == 900 and clock["elapsed_s"] > 0 and summary["status"] == row["status"],
            "Missing deadline or inconsistent status")
    near(summary["wall_s"], row["wall_s"])
    if row["status"] != "completed":
        require(row["status"] == "timed_out" and clock["kill_requested_s"] is not None, f"Failed arm {folder}")
        row["last_known_fits"] = clock["progress"].get("fits")
        return row, None
    result, frozen = read(folder/"result.json"), read(folder/"frozen.json")
    columns = list(columns)
    keep = frozen["selected"]
    require(clock["result"] == result and clock["exit_code"] == 0 and clock["kill_requested_s"] is None,
            "Watchdog does not match the selector record")
    require(frozen["method"] == method and frozen["tolerance"] == tol and frozen["search_seed"] == seed,
            "Different search identity")
    require(frozen["dataset"]=="madelon" and frozen["data_seed"]==frozen["predictor_seed"]==42
            and (result["method"],result["tolerance"],result["search_seed"])==(method,tol,seed),
            "Different dataset, split, predictor or recorded arm")
    require(frozen["columns"] == columns and ordered_subset(keep, columns) and frozen["partition_sha256"] == hashes,
            "Invalid final support or data split")
    require(frozen["protocol_fingerprint"] == fingerprint and frozen["model_policy"] == protocol["model_policy"]
            and result["frozen_hash"] == canonical_hash(frozen), "Changed protocol/model/frozen record")
    event, = read(folder/"test_access.json")["events"]
    require(event["job_id"] == result["frozen_hash"] and event["decision"] == "allowed"
            and event["reason"] == "final_removal_evaluation" and event["seed"] == 42
            and event["dataset_id"] == "madelon", "Unexpected test access")
    diag = frozen["diagnostics"]
    front, backend = diag["front"], diag["backend"]
    support = frozen["support"]
    front_steps = front["steps"] if front is not None else []
    backend_steps = backend["proposals"] if backend is not None else []
    has_front = method.startswith("piv3")
    require(has_front == (front is not None) and (method == "piv3") == (backend is None),
            "Missing or extra pipeline stage")
    require(ordered_subset(support,columns) and ordered_subset(keep,support), "Unexpected stage supports")
    steps = [*front_steps, *backend_steps]
    expected_masks = list(dict.fromkeys(tuple(s["proposed"]) for s in steps if s["proposed"] != columns))
    actual_masks = [tuple(s) for s in diag["evaluated_masks"]]
    require(expected_masks == actual_masks and all(ordered_subset(list(m),columns) for m in actual_masks),
            "Real candidate fits do not match stage proposals")
    with np.load(folder/"predictions.npz", allow_pickle=False) as p:
        np.testing.assert_array_equal(p["validation_y"], labels["val"])
        np.testing.assert_array_equal(p["y"], labels["test"])
        for key in ("validation_full","validation","candidate_validation","full","selected"):
            prob=p[key]
            n=len(labels["test" if key in ("full","selected") else "val"])
            shape=(len(actual_masks),n,2) if key=="candidate_validation" else (n,2)
            require(prob.shape==shape and np.all((prob>=0)&(prob<=1)) and
                    np.allclose(prob.sum(axis=-1),1.,rtol=0,atol=1e-6),"Invalid class probabilities: "+key)
        ref_val = metric_pair(labels["val"],p["validation_full"],"roc_auc")
        val = metric_pair(labels["val"],p["validation"],"roc_auc")
        ref_test = metric_pair(labels["test"],p["full"],"roc_auc")
        test = metric_pair(labels["test"],p["selected"],"roc_auc")
        for key, actual in (("reference_scores",ref_val),("validation_scores",val),
                            ("test_reference_scores",ref_test),("test_scores",test)):
            near(result[key],actual)
            if key in ("reference_scores","validation_scores"):
                near(frozen[key],actual)
        predictions = dict(zip(actual_masks,p["candidate_validation"],strict=True))
        predictions[tuple(columns)] = p["validation_full"]
        scores = {mask:metric_pair(labels["val"], pred, "roc_auc") for mask,pred in predictions.items()}
        for s in steps:
            pred_scores = scores[tuple(s["proposed"])]
            require(len(s["proposed"]) == s["k"] and s["accepted"] == passes(pred_scores,ref_val,tol),
                    "Wrong stage validation decision")
            near([s["auc"],s["accuracy"]],pred_scores)
        front_score = scores[tuple(support)]
        near(frozen["front_scores"],front_score)
        require(passes(front_score,ref_val,tol),"Front support did not meet the same standard")
        if has_front:
            require(len(front_steps)<=12 and len({tuple(s["proposed"]) for s in front_steps})==len(front_steps)
                    and all(s["proposed"]!=columns for s in front_steps),"PIv3 candidate budget or duplicate")
            require(check_new_search(front,columns,ref_val,tol)[0] == support, "PIv3 vote/selection mismatch")
            if tol == .05:
                check_replay(folder,seed,support,front_steps,p,method=="piv3")
        else:
            require(support == columns,"Unexpected preliminary screening")
        if backend is not None:
            check_backend(method,backend,support,keep,seed,front_score)
        else:
            require(keep==support,"Standalone PIv3 differs from front choice")
        np.testing.assert_array_equal(predictions[tuple(keep)],p["validation"])
        if keep == columns:
            np.testing.assert_array_equal(p["selected"],p["full"])
    aux = int(backend is not None and backend["backend"]=="rf_prefix")
    require(result["aux_fits"]==aux and result["fits"]==len(actual_masks)+3+aux,"Wrong total estimator fits")
    require(result["k"]==len(keep) and result["front_k"]==len(support)
            and result["candidates"]==len(actual_masks) and result["front_candidates"]==len(front_steps),
            "Inconsistent candidate or feature counts")
    require(result["validation_pass"] and passes(val,ref_val,tol) and result["test_pass"]==passes(test,ref_test,tol),
            "Different validation or held-out pass status")
    require(summary["k"]==len(keep) and summary["fits"]==result["fits"],"Watchdog summary counts")
    row.update(k=len(keep),candidates=len(actual_masks),fits=result["fits"],front_k=len(support),
               front_candidates=len(front_steps),backend_proposals=len(backend_steps),aux_fits=aux,
               endpoint_fallback=(backend["endpoint_fallback"] if backend else None),
               front_s=result["front_s"],backend_s=result["backend_s"],
               front_fingerprint=canonical_hash(front) if front else None,
               backend_ranking_sha256=canonical_hash(backend["ranking"]) if backend else None,
               validation_auc=val[0],validation_accuracy=val[1],test_auc=test[0],test_accuracy=test[1],
               test_pass=result["test_pass"],test_auc_pass=test[0]>=ref_test[0]*(1-tol)-1e-12,
               test_accuracy_pass=test[1]>=ref_test[1]*(1-tol)-1e-12,validation_pass=True,
               full_validation_auc=ref_val[0],full_validation_accuracy=ref_val[1],
               full_test_auc=ref_test[0],full_test_accuracy=ref_test[1])
    require(min(row["front_s"],row["backend_s"]) >= 0 and row["front_s"]+row["backend_s"]<=row["wall_s"],
            "Stage timing exceeds end-to-end time")
    return row,set(keep)


def check_backend(method,diag,support,keep,search_seed,front_score):
    score=np.asarray(diag["importance"],dtype=float)
    rank=[support[i] for i in np.argsort(-score,kind="stable")]
    require(score.shape==(len(support),) and np.isfinite(score).all() and diag["ranking"]==rank
            and diag["support"]==support and diag["backend"]==method.removeprefix("piv3_"),
            "Wrong backend scores, order or stage inputs")
    proposals=diag["proposals"]
    if diag["backend"]=="pfi":
        drops=np.asarray(diag["drops"])
        require(drops.shape==(len(support),5) and diag["repeats"]==5 and diag["ranking_seed"]==search_seed
                and diag["row_indices"]==list(range(416)) and diag["selector_aux_fits"]==0,
                "Different PFI setup")
        near(drops.mean(axis=1),score)
        near(front_score[0],diag["sample_full_auc"])
        wanted=[name for name, drop in zip(support,score,strict=True) if drop>0] or rank[:1]
        p,=proposals
        require(p["proposed"]==wanted==diag["native"] and keep==(wanted if p["accepted"] else support)
                and diag["endpoint_fallback"]==(not p["accepted"]),"Invalid PFI support or reference fallback")
    else:
        require(diag["trees"]==100 and diag["ranking_seed"]==42 and diag["ranking_sample_rows"]==1664
                and diag["selector_aux_fits"]==1,"Different forest ranking policy")
        for k,p in enumerate(proposals,1):
            wanted=set(rank[:k])
            require(p["proposed"]==[f for f in support if f in wanted] and p["k"]==k,
                    "Discontinuous RF traversal")
        require(proposals and proposals[-1]["accepted"] and not any(p["accepted"] for p in proposals[:-1])
                and keep==proposals[-1]["proposed"] and diag["endpoint_fallback"]==(keep==support),
                "Traversal did not stop at its smallest dual-feasible prefix")


def summarize(frame,masks):
    summaries, overlaps = [], []
    for (tol,method), all_rows in frame.groupby(["tolerance","method"],sort=False):
        complete=all_rows[all_rows.status=="completed"]
        jac=[]
        for a,b in combinations(complete.search_seed,2):
            left,right=masks[method,tol,a],masks[method,tol,b]
            jac.append(jaccard(left,right))
            overlaps.append(dict(tolerance=tol,method=method,seed_a=a,seed_b=b,intersection=len(left&right),
                                 union=len(left|right),jaccard=jac[-1]))
        summary=dict(tolerance=tol,method=method,scheduled=len(all_rows),completed=len(complete),
                     min_k=complete.k.min(),median_k=complete.k.median(),max_k=complete.k.max(),
                     mean_k=complete.k.mean(),mean_jaccard=np.mean(jac) if jac else None,
                     mean_wall_s=complete.wall_s.mean(),total_observed_wall_s=all_rows.wall_s.sum(),
                     mean_fits=complete.fits.mean(),mean_test_auc=complete.test_auc.mean(),
                     mean_front_k=complete.front_k.mean(),mean_front_s=complete.front_s.mean(),
                     mean_backend_s=complete.backend_s.mean(),
                     fallback_count=int(complete.endpoint_fallback.eq(True).sum()),
                     mean_test_accuracy=complete.test_accuracy.mean(),test_passes=complete.test_pass.sum())
        summaries.append(summary)
    return pd.DataFrame(summaries),pd.DataFrame(overlaps)


def paired_frontends(frame,masks):
    rows=[]
    for tol in TOLS:
        for downstream in ("pfi","rf_prefix"):
            native=downstream
            combined="piv3_"+downstream
            for seed in SEEDS:
                subset=frame[(frame.tolerance==tol)&(frame.search_seed==seed)].set_index("method")
                base, front = subset.loc[native],subset.loc[combined]
                row=dict(tolerance=tol,backend=native,search_seed=seed,
                         plain_status=base.status,piv3_status=front.status,
                         plain_wall_s=base.wall_s,piv3_wall_s=front.wall_s)
                if base.status==front.status=="completed":
                    left,right=masks[native,tol,seed],masks[combined,tol,seed]
                    row.update(plain_k=base.k,piv3_k=front.k,seconds_saved=base.wall_s-front.wall_s,
                               fraction_time_saved=1-front.wall_s/base.wall_s,
                               delta_test_auc=front.test_auc-base.test_auc,
                               delta_test_accuracy=front.test_accuracy-base.test_accuracy,
                               plain_test_pass=base.test_pass,piv3_test_pass=front.test_pass,
                               intersection=len(left&right),union=len(left|right),jaccard=jaccard(left,right))
                rows.append(row)
    return pd.DataFrame(rows)


def render(frame, summary, paired):
    first=frame[frame.status=="completed"].iloc[0]
    lines=["# Madelon 原始 500 列：PIv3 与两种传统代表方法", "",
           "数据划分和下游 XGBoost 均为 seed 42。容差为相对损失，AUC 与准确率都要达标。"
           "选列只使用训练与验证数据，锁定的子集在测试集上单独检查。"
           "原始特征模型的验证 AUC 和准确率为 "
           f"{first.full_validation_auc:.5f} 和 {first.full_validation_accuracy:.2%}，"
           f"测试为 {first.full_test_auc:.5f} 和 {first.full_test_accuracy:.2%}。","",
           "采样代表 PFI 在拟合模型上每列置换 5 次，取验证 AUC 平均下降为正的列，然后重训这个子集。"
           "若验证未通过，回退到本阶段输入。遍历代表以含 100 棵树的随机森林的训练重要性排序，从一个特征起逐列增加，"
           "首个验证双达标的前缀即停止。PIv3＋后端运行会在 PIv3 保留列上重新计算后端排名，"
           "并沿用最初完整模型的门槛。","",
           "下表为同一数据划分下各次冷启动的算术均值。列数以中位数及范围表示。"
           "Jaccard 为各随机种子两两比较时的交集大小除以并集大小的均值。原生遍历的随机森林训练固定为 seed 42，三次冷启动只复测时长，重合度应为 1。","",
           "| 容差 | 方法 | 保留列数 中位数［范围］ | 平均 Jaccard | 平均拟合次数 | 平均总耗时（秒） | 平均测试 AUC | 平均测试准确率 | 测试双达标 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        j=f"{r.mean_jaccard:.3f}" if pd.notna(r.mean_jaccard) else "—"
        if not r.completed:
            lines.append(f"| {r.tolerance:.0%} | {NAMES[r.method]} | — | — | — | — | — | — | 无完整结果 |")
            continue
        lines.append(f"| {r.tolerance:.0%} | {NAMES[r.method]} | {r.median_k:g}［{r.min_k:g}–{r.max_k:g}］ | {j} | "
                     f"{r.mean_fits:.2f} | {r.mean_wall_s:.2f} | {r.mean_test_auc:.5f} | {r.mean_test_accuracy:.2%} | "
                     f"{r.test_passes:.0f}/{r.completed} |")
    lines += ["","## 加入 PIv3 前置后的配对变化","",
              "按相同容差与搜索种子配对后取均值。节省时间为原方法耗时减去加入前置后的耗时。"
              "其余变化量均为加入前置后减去原值。此处的 Jaccard 比较同一次配对中两种方法各自保留的特征。","",
              "| 容差 | 后端 | 完整配对 | 保留列变化 | 节省秒数 | AUC 变化 | 准确率变化（百分点） | 配对 Jaccard |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (tol, method), group in paired.groupby(["tolerance","backend"], sort=False):
        complete = group[(group.plain_status=="completed") & (group.piv3_status=="completed")]
        if len(complete):
            delta_k = (complete.piv3_k - complete.plain_k).mean()
            lines.append(f"| {tol:.0%} | {NAMES[method]} | {len(complete)}/{len(group)} | {delta_k:+.1f} | "
                         f"{complete.seconds_saved.mean():+.2f} | {complete.delta_test_auc.mean():+.5f} | "
                         f"{100*complete.delta_test_accuracy.mean():+.2f} | {complete.jaccard.mean():.3f} |")
        else:
            lines.append(f"| {tol:.0%} | {NAMES[method]} | 0/{len(group)} | — | — | — | — | — |")
    lines += ["","## 各次运行","",
              "| 容差 | 方法 | 搜索种子 | 保留列 | 总拟合 | 秒 | 验证 AUC | 验证准确率 | 测试 AUC | 测试准确率 | 测试双达标 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in frame.itertuples():
        head=f"| {r.tolerance:.0%} | {NAMES[r.method]} | {r.search_seed} |"
        if r.status != "completed":
            lines.append(head+f" — | — | {r.wall_s:.2f} | — | — | — | — | 超时 |")
            continue
        lines.append(head+f" {r.k:g} | {r.fits:g} | {r.wall_s:.2f} | {r.validation_auc:.5f} | {r.validation_accuracy:.2%} | "
                     f"{r.test_auc:.5f} | {r.test_accuracy:.2%} | {'是' if r.test_pass else '否'} |")
    lines += ["", "拟合数包含各独立运行的完整模型验证、实际验证过的不同候选和完整／入选模型的两次最终重训；"
              "随机森林后端另计 1 次排序拟合。端到端耗时计入这些步骤以及投票、置换和保存结果。"
              "同一运行中重复提出的子集复用已有拟合。同一搜索种子在不同容差中可能选出相同结果。",
              "", "[方法汇总](method_summary.csv) · [逐次结果](per_arm.csv) · [是否加入 PIv3 的逐种子配对](paired_frontends.csv) · [特征重合度](overlap.csv) · "
              "[比较协议](protocol.json) · [核验记录](audit.json)"]
    return "\n".join(lines)+"\n"


def main(run):
    protocol=read(run/"protocol.json")
    require(protocol["tolerances"]==list(TOLS) and protocol["search_seeds"]==list(SEEDS)
            and protocol["data_seed"]==protocol["predictor_seed"]==42 and protocol["arm_limit_s"]==900,
            "Different experimental scope")
    for name,digest in protocol["source_sha256"].items():
        require(sha(run/"source"/name)==digest,"Invalid source snapshot: "+name)
    columns, _,labels,hashes,_=verify_inputs(run,protocol)
    tasks=[tuple(t) for t in protocol["tasks"]]
    expected={(m,t,s) for m in METHODS for t in TOLS for s in SEEDS}
    require(len(tasks)==len(set(tasks))==45 and set(tasks)==expected, "Unbalanced task list")
    summary=pd.read_csv(run/"summary.csv").set_index(["method","tolerance","search_seed"])
    require(summary.index.is_unique and set(summary.index)==set(tasks), "Incomplete monitored results")
    masks, rows={}, []
    fingerprint=canonical_hash(protocol)
    for m,t,s in sorted(tasks,key=lambda v:(v[1],METHODS.index(v[0]),v[2])):
        folder=run/m/f"tol-{round(t*100):02d}"/f"s{s}"
        row,mask=inspect(folder,protocol,fingerprint,columns,labels,hashes,m,t,s,summary.loc[m,t,s])
        rows.append(row)
        masks[m,t,s]=mask
    frame=pd.DataFrame(rows)
    complete=frame[frame.status=="completed"]
    require(len(complete)>0,"No complete arm")
    for field in ("full_validation_auc","full_validation_accuracy","full_test_auc","full_test_accuracy"):
        near(complete[field],np.full(len(complete),complete[field].iloc[0]))
    for _, group in complete[complete.method.str.startswith("piv3")].groupby(["tolerance","search_seed"]):
        require(group.front_fingerprint.nunique()==1, "PIv3 decisions diverge between downstream methods")
    forests=complete[complete.method=="rf_prefix"]
    require(forests.backend_ranking_sha256.nunique()<=1, "Raw forest ranking changes across tolerances")
    for _, group in forests.groupby("tolerance"):
        for field in ("k","validation_auc","validation_accuracy","test_auc","test_accuracy","backend_ranking_sha256"):
            require(group[field].nunique()==1, "Raw forest traversal varies with its replicate label")
    for _, group in complete[complete.method=="pfi"].groupby("search_seed"):
        require(group.backend_ranking_sha256.nunique()==1, "Same search seed has different PFI ranks across tolerances")
    groups,overlaps=summarize(frame,masks)
    paired=paired_frontends(frame,masks)
    frame.to_csv(run/"per_arm.csv",index=False)
    groups.to_csv(run/"method_summary.csv",index=False)
    overlaps.to_csv(run/"overlap.csv",index=False)
    paired.to_csv(run/"paired_frontends.csv",index=False)
    (run/"comparison.md").write_text(render(frame,groups,paired),encoding="utf-8")
    outputs=["per_arm.csv","method_summary.csv","overlap.csv","paired_frontends.csv","comparison.md"]
    report=dict(status="pass",completed=len(complete),planned=len(tasks),protocol_fingerprint=fingerprint,
                candidate_predictions_verified=int(complete.candidates.sum()),
                historical_piv3_replays=int((complete.method.str.startswith("piv3")&(complete.tolerance==.05)).sum()),
                artifacts={key:sha(run/key) for key in outputs})
    atomic_write_json(run/"audit.json",report)
    print(report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run)
