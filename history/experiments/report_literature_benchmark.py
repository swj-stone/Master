"""Audit the sampled game-theoretic benchmark and publish its two comparable tables."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.run_literature_benchmark import CONTROLS, DATASETS, DEFAULT, ESTIMATORS, LIMIT
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, _sha
from experiments.test.piv3_lite_arm import _metrics
from minppi_mofs.experiment_protocol import canonical_hash

LABEL={"full":"全列模型","minpi":"最新 minPI","gain":"紧凑 Gain","pfi":"快速 PFI",
       "minshap":"MinShap","shapley":"排列 Shapley","banzhaf":"采样 Banzhaf","shapg":"ShapG"}

def read(path):
    return json.loads(path.read_bytes())

def write_csv(path,rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig",newline="") as handle:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def fail(why):
    raise ValueError(why)

def match(a,b):
    if set(a)!=set(b) or any(not np.isclose(a[k],b[k],rtol=0,atol=1e-12) for k in a):
        fail(f"Score mismatch {a}, {b}")

def relative(scores, full):
    return {k: ((v-full[k]) if k=="mae" else (full[k]-v))/abs(full[k]) if full[k] else
            (0. if (v<=0 if k=="mae" else v>=0) else float("inf")) for k,v in scores.items()}

def notation(scores):
    if "mae" in scores:
        return f"{scores['r2']:.4f} / {scores['mae']:.4f}"
    out=f"{scores['auc']:.4f} / {scores['accuracy']:.2%}"
    return out+(f" / {scores['ap']:.4f}" if "ap" in scores else "")

def ledger(out):
    file=out/"fits.jsonl"
    records=[]
    if file.is_file():
        lines=file.read_bytes().splitlines()
        for idx,line in enumerate(lines):
            try: records.append(json.loads(line))
            except ValueError:
                if idx!=len(lines)-1: raise
    counts={}
    for row in records:
        k=str(row.get("event",row.get("status","")))
        counts[k]=counts.get(k,0)+1
    return counts

def report(root, *, partial=False):
    meta=read(root/"protocol.json")
    enforce_guard=meta.get("enforce_guard",True)
    routes={}
    retained=[]
    for d in DATASETS:
        dx=DEFAULT_ROOT/d
        source=read(dx/"metadata.json")
        if _sha(dx/"metadata.json")!=meta["inputs"][d]["metadata_sha256"]:
            fail("Changed input metadata")
        ytest=np.load(dx/"test_y.npy",mmap_mode="r")
        yval=np.load(dx/"val_y.npy",mmap_mode="r")
        _,guard=train_test_split(np.arange(len(yval)),test_size=.5,random_state=42,
                                 stratify=None if source["task"]=="regression" else yval)
        guard=np.sort(guard)
        for mode,methods in (("shared",CONTROLS),("frozen",ESTIMATORS),("native",ESTIMATORS)):
            for m in methods:
                folder=root/d/mode/m
                tfile=folder/"watchdog.json"
                row=dict(dataset=d,source_mode=mode,method=m,width=source["width"],status="pending")
                routes[d,mode,m]=row
                features=dict(dataset=d,source_mode=mode,method=m,status="pending",candidate_k=None,k=None,
                              retained_features_json=None)
                retained.append(features)
                if not tfile.is_file():
                    receipt=folder/"interruption.json"
                    if receipt.is_file():
                        stop=read(receipt)
                        counts=ledger(folder)
                        if (stop["status"]!="interrupted" or stop["reason"]!="stopped_by_user"
                            or (d,mode,m)!=(stop["dataset"],stop["mode"],stop["method"])
                            or (stop["fits_completed"],stop["fits_started"])!=(counts.get("completed",0),counts.get("started",0))
                            or (folder/"result.json").exists()):
                            fail(f"Inconsistent interruption receipt {folder}")
                        p=read(folder/"progress.json")
                        row.update(status="interrupted",stop_reason=stop["reason"],fits=stop["fits_completed"],
                                   fits_started=stop["fits_started"],last_heartbeat_wall_s=stop["last_heartbeat_wall_s"],
                                   phase=p["phase"])
                    elif not folder.exists():
                        row["status"]="not_started"
                    if not partial: fail(f"No finished route at {folder} ({row['status']})")
                    features["status"]=row["status"]
                    continue
                timer=read(tfile)
                row.update(status=timer["status"],wall_s=timer["elapsed_s"])
                features["status"]=timer["status"]
                if timer["limit_s"]!=LIMIT: fail("Budget mismatch")
                counts=ledger(folder)
                if timer["status"]=="timed_out":
                    if timer["kill_requested_s"] is None: fail("Missing watchdog stop")
                    p=timer["progress"]
                    # A killed fit may be started but not completed.
                    row.update(fits=counts.get("completed",p.get("fits_completed",p.get("fits"))),
                               fits_started=counts.get("started",p.get("fits_started")),phase=p.get("phase"),
                               timeout_progress=json.dumps(p,ensure_ascii=False))
                    continue
                if timer["status"]!="completed": fail(f"Unexpected {timer['status']} at {folder}")
                r,frozen,access=map(read,(folder/"result.json",folder/"frozen.json",folder/"test_access.json"))
                if (r!=timer["result"] or r["frozen_hash"]!=canonical_hash(frozen)
                    or r["frozen_hash"]!=access["frozen_hash"] or r["k"]!=len(r["selected"])
                    or len(set(r["selected"]))!=r["k"] or frozen["selected"]!=r["selected"]
                    or r.get("enforce_guard",True)!=enforce_guard or frozen.get("enforce_guard",True)!=enforce_guard
                    or r["candidate_k"]!=len(r["candidate"]) or frozen["candidate"]!=r["candidate"]
                    or r["selected"]!=(source["features"] if r["fallback"] else r["candidate"])):
                    fail(f"Invalid freeze/timeout result {folder}")
                score=lambda y,pred:_metrics(y,pred,source["utility"],d=="credit_card_fraud")
                with np.load(folder/"predictions.npz",allow_pickle=False) as z:
                    np.testing.assert_array_equal(ytest,z["y_test"])
                    np.testing.assert_array_equal(guard,z["guard_rows"])
                    np.testing.assert_array_equal(yval[guard],z["y_guard"])
                    match(score(ytest,z["test_pred"]),r["test_scores"])
                    match(score(yval[guard],z["guard_full"]),r["validation_full"])
                    match(score(yval[guard],z["guard_candidate"]),r["validation_candidate"])
                    np.testing.assert_array_equal(z["guard_selected"],z["guard_full" if r["fallback"] else "guard_candidate"])
                failures=[name for name,base in r["validation_full"].items()
                          if ((r["validation_candidate"][name]-base) if name=="mae" else
                              (base-r["validation_candidate"][name]))>.05*abs(base)+1e-12]
                if ((enforce_guard and bool(failures))!=r["fallback"] or set(failures)!=set(r["failed_metrics"])):
                    fail(f"Inconsistent gate {folder}")
                features.update(candidate_k=r["candidate_k"],k=r["k"],
                                retained_features_json=json.dumps(r["selected"],ensure_ascii=False,separators=(",",":")))
                row.update(candidate_k=r["candidate_k"],k=r["k"],deletions=r["width"]-r["k"],
                           enforce_guard=enforce_guard,validation_failed_metrics=",".join(failures),
                           fits=r["fits"],fits_started=r["fits"],fallback=r["fallback"],active_s=r["elapsed_s"],
                           selector_s=r.get("selector_s"),phase="complete",test_metrics=notation(r["test_scores"]))
                for k in ("validation_full","validation_candidate","test_scores"):
                    row.update({k+"_"+name:val for name,val in r[k].items()})
                for k in ("selection_fits","estimate_s","sizing_s","guard_s","final_s","reference_s","coalition_calls",
                          "actual_coalitions","oracle_batches","ranking_prediction_rows","utility_rows",
                          "effective_split_features","cached_calls","constant_calls"):
                    if k in r: row[k]=r[k]
    for (d,mode,m),r in routes.items():
        base=routes[d,"shared","full"]
        if r["status"]!="completed" or base["status"]!="completed": continue
        keys=[k for k in base if k.startswith("test_scores_")]
        losses=relative({k[12:]:r[k] for k in keys},{k[12:]:base[k] for k in keys})
        r.update(worst_relative_loss=max(losses.values()),test_pass=max(losses.values())<=.05+1e-12)
    view=[]
    wins=[]
    for mode in ("native","frozen"):
        for d in DATASETS:
            group=[dict(table_mode=mode,**routes[d,"shared" if m in CONTROLS else mode,m])
                   for m in (*CONTROLS,*ESTIMATORS)]
            completed=[r for r in group if r["method"]!="full" and r["status"]=="completed"]
            passing=[r for r in completed if r.get("test_pass")]
            valid=passing if enforce_guard else completed
            best_k=min((r["k"] for r in valid),default=None)
            fastest=min((r["wall_s"] for r in valid),default=None)
            passing_k=min((r["k"] for r in passing),default=None)
            passing_s=min((r["wall_s"] for r in passing),default=None)
            for r in group:
                r["min_k_among_passing"]=r in passing and r["k"]==passing_k
                r["fastest_among_passing"]=r in passing and r["wall_s"]==passing_s
                r["min_k_in_ranking"]=r in valid and r["k"]==best_k
                r["fastest_in_ranking"]=r in valid and r["wall_s"]==fastest
            wins.append(dict(table_mode=mode,dataset=d,finished=len(completed),passed=len(passing),eligible=len(valid),
                             ranking_policy="test_pass" if enforce_guard else "all_completed",smallest_k=best_k,
                             smallest_methods=" / ".join(LABEL[r["method"]] for r in valid if r["min_k_in_ranking"]),
                             fastest_s=fastest,fastest_method=" / ".join(LABEL[r["method"]] for r in valid if r["fastest_in_ranking"])))
            view.extend(group)
    write_csv(root/"routes.csv",list(routes.values()))
    write_csv(root/"retained-features.csv",retained)
    for mode in ("native","frozen"):
        write_csv(root/(mode+".csv"),[r for r in view if r["table_mode"]==mode])
    write_csv(root/"winners.csv",wins)
    counts=Counter(r["status"] for r in routes.values())
    status_label={"completed":"已完成","timed_out":"超时","interrupted":"用户停止","not_started":"未启动",
                  "pending":"已启动，尚无终值"}
    progress="，".join(f"{label} {counts[state]} 条" for state,label in status_label.items() if counts[state])
    text=["# 采样博弈特征选择主实验","",
          f"共 {len(routes)} 条独立路线，{progress}。","",
          "原始维度、数据/模型/抽样 seed 42。每条独立路线冷启动限 300 秒。两张表中的全列参考、minPI、紧凑 Gain 和快速 PFI 各实测一次，在两表引用同一结果。总秒含加载、首个全列模型、前置/建图/估计、验证、冻结后最终重训及测试。","",
          "分类依次为 AUC 与准确率，ISOLET 用宏平均 OvR AUC，欺诈检测再列 AP。Abalone 为零截断 R² 与 MAE，MAE 越低越好。超时行显示截至终止已完成的拟合数。",
          ("验证集逐项检查 5% 相对损失，超限即回退全列。下表在最终测试各项均达标的路线中比较速度和列数。" if enforce_guard else
           "本轮关闭 5% 验证回退。每条完成的路线均用实际候选列测试，零列时使用截距预测。验证和测试上的 5% 超限均作诊断，排名包含全部完成的非全列参考方法。"),
          "实际列名见 retained-features.csv。零列记为 []，尚无终值的路线留空。","",
          "## 删列与运行速度","","|口径|数据集|参与排名方法数|最少列|并列最少的方法|最快方法|冷启动秒|","|---|---|---:|---:|---|---|---:|"]
    for r in wins:
        text.append(f"|{r['table_mode']}|{r['dataset']}|{r['eligible']}|{r['smallest_k'] if r['smallest_k'] is not None else '—'}|{r['smallest_methods'] or '—'}|{r['fastest_method'] or '—'}|{r['fastest_s']:.2f}|" if r["fastest_s"] is not None else f"|{r['table_mode']}|{r['dataset']}|0|—|—|—|—|")
    for mode,title in (("native","文献估计量，联盟重新拟合"),("frozen","统一冻结预测模型与遮盖效用")):
        text+=["","## "+title,"","|数据集|方法|原始列|候选列|最终列|拟合次数|含前置总秒|测试指标|5% 达标|回退全列|","|---|---|---:|---:|---:|---:|---:|---|---|---|"]
        for r in (x for x in view if x["table_mode"]==mode):
            if r["status"]=="completed":
                text.append(f"|{r['dataset']}|{LABEL[r['method']]}|{r['width']}|{r['candidate_k']}|{r['k']}|{r['fits']}|{r['wall_s']:.2f}|{r['test_metrics']}|{'是' if r.get('test_pass') else '否'}|{'是' if r['fallback'] else '否'}|")
            else:
                wall=f"{r['wall_s']:.2f}" if "wall_s" in r else "—"
                if r["status"]=="interrupted":
                    wall=f"最后心跳 {r['last_heartbeat_wall_s']:.1f}"
                text.append(f"|{r['dataset']}|{LABEL[r['method']]}|{r['width']}|—|—|{r.get('fits','—')}|{wall}|{status_label[r['status']]}|—|—|")
    text += ["", "## 如何解读这两张表", "",
             "1. 同一数据集的路线使用相同原始列、数据划分和 XGBoost 配置，分别从头拟合。验证集等分为排序与独立评估两部分。特征冻结后才读取测试集。表中“5% 达标”根据同一数据集全列模型的测试成绩事后判定。",
             "2. 文献估计量采用本实验的预测器和数据划分。MinShap 用四条完整随机排列取最小观测边际，以配对逐行损失方差做显著性筛选（α=0.05）。回归用平方误差，分类用 Brier 损失。排列 Shapley 在同样四条排列上取平均边际。Banzhaf 复用 32 对互补的均匀随机联盟。ShapG 从全部训练行构建 Pearson 图，以深度 1、邻居上限 3 做局部估值，分类使用 F1，回归使用 R²。这里的采样预算固定，结果仅代表这一组近似精度。 [MinShap](https://arxiv.org/html/2604.15107v2#S4) · [排列 Shapley](https://www.sciencedirect.com/science/article/pii/S0305054808000804) · [Banzhaf 样本复用](https://proceedings.mlr.press/v206/wang23e.html) · [ShapG](https://arxiv.org/html/2407.00506v2#S3.SS2.SSS2)",
             "3. 冻结表的四种博弈估计量用同一个全列模型定义联盟效用。用训练中位数遮盖联盟外的列，在最多 256 个排序样本上评估，对从未出现在树分裂中的列可直接赋零边际。此时 MinShap 用最小观测边际排序。需要将排序转为列集的方法统一用冻结模型检查最多 12 个前缀，内部代理允许 8% 损失。这一步决定候选列数，独立验证和测试各自另行计算实际损失。",
             "4. 最新 minPI 先用紧凑 Gain 得到候选，再在三种规模的上下文中对成块删除投票，取三组净票率的最小值。快速 PFI 对树实际用到的列分别打乱三次，按正的重要性选列。表中的拟合数为实际完成的 XGBoost 拟合次数；一次模型拟合可伴随许多遮盖预测。运行总秒也包含进程启动和退出，超时行的少量额外墙钟时间来自终止并回收子进程。"]
    (root/"results.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(dict(counts))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path,nargs="?",default=DEFAULT)
    p.add_argument("--allow-pending",action="store_true")
    args=p.parse_args()
    report(args.run,partial=args.allow_pending)

