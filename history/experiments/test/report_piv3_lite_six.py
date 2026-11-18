"""Publish seed-42 scoped benchmark comparisons using actual costs and all applicable metrics."""
import argparse
import csv
import json
from pathlib import Path

from experiments.test.run_piv3_lite_six import DATASETS, METHODS

DOWNSTREAM = ("gain","extra_trees","mi","banzhaf","pfi","f_score")

def read(path):
    return json.loads(path.read_bytes())

def losses(scores, ref):
    """Positive values represent deterioration; 0 reference requires no deterioration."""
    out={}
    for key,base in ref.items():
        loss=(scores[key]-base) if key=="mae" else (base-scores[key])
        out[key]=loss/abs(base) if base else (0. if loss<=0 else float("inf"))
    return out

def csv_table(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)

def show(x,d=2):
    return "—" if x is None else f"{x:.{d}f}"

def report(root):
    scores_by_arm={}
    rows=[]
    for d in DATASETS:
        for m in METHODS:
            folder=root/d/m
            timer=read(folder/"watchdog.json")
            row=dict(dataset=d,method=m,status=timer["status"],cold_wall_s=timer["elapsed_s"])
            if timer["status"]=="completed":
                a=read(folder/"result.json")
                scores_by_arm[d,m]=a
                row.update({k:a[k] for k in ("width","front_k","candidate_k","k","fits","front_fits","front_s","rank_s","selector_s","proposal_s","guard_s","final_s","elapsed_s","fallback")})
                reason=read(folder/"front.json").get("bypass")
                row["bypass"]=bool(reason)
                row["bypass_reason"]=reason or ""
                for key in ("validation_full","validation_candidate","validation_selected","test_scores"):
                    row.update({key+"_"+k:v for k,v in a[key].items()})
            rows.append(row)
    pairs=[]
    for d in DATASETS:
        full=scores_by_arm[d,"full"]
        for m in DOWNSTREAM:
            baseline=scores_by_arm.get((d,m))
            for arm in ("pi","pi_"+m):
                a=scores_by_arm.get((d,arm))
                row=dict(dataset=d,baseline=m,challenger=arm,status="complete" if a and baseline else "incomplete",
                         width=full["width"])
                if a and baseline:
                    vs_full=losses(a["test_scores"],full["test_scores"])
                    vs_base=losses(a["test_scores"],baseline["test_scores"])
                    ratio=a["elapsed_s"]/baseline["elapsed_s"]
                    cold={r["method"]:r["cold_wall_s"] for r in rows if r["dataset"]==d}
                    row.update(k_base=baseline["k"],k=a["k"],base_s=baseline["elapsed_s"],s=a["elapsed_s"],
                               time_ratio=ratio, speedup=1/ratio, cold_time_ratio=cold[arm]/cold[m],
                               selector_s_base=baseline["selector_s"],selector_s=a["selector_s"],
                               fits_base=baseline["fits"],fits=a["fits"],fallback_base=baseline["fallback"],
                               fallback=a["fallback"],same_columns=a["selected"]==baseline["selected"],
                               pass_vs_full=max(vs_full.values())<=.05+1e-12,
                               pass_vs_base=max(vs_base.values())<=.05+1e-12,
                               worst_loss_vs_full=max(vs_full.values()),worst_loss_vs_base=max(vs_base.values()),
                               faster=ratio<1,within_10pct=ratio<=1.10,
                               meets_both=ratio<=1.10 and max(*vs_full.values(),*vs_base.values())<=.05+1e-12)
                    for k in vs_full:
                        row["base_test_"+k]=baseline["test_scores"][k]
                        row["test_"+k]=a["test_scores"][k]
                        row["loss_vs_base_"+k]=vs_base[k]
                        row["loss_vs_full_"+k]=vs_full[k]
                pairs.append(row)
    csv_table(root/"methods.csv",rows)
    csv_table(root/"paired.csv",pairs)
    lines=["# PIv3-Lite 单 seed 原始维度实验","",
           "输入、模型和搜索使用 seed 42。每个方法每个数据集限 300 秒。全列基准与每条路线使用同一最终模型协议。独立执行各路线，完整活动耗时包含读入、前置、后端、验证、重训和测试，另在 CSV 保留含 Python 冷启动的墙钟耗时。","",
           "各指标允许相对损失 5%。相对全列和对应原法均达标才记为效果达标。筛选方法的验证门槛有额外余量，候选未通过时回退全列。3 个低维集直接通过前置。","",
           "## 与下游方法逐组比较","",
           "|数据集|后端|列数 原法 / 加 PI|活动秒 原法 / 加 PI|加速倍数|测试主指标 原法 / 加 PI|准确率或 MAE 原法 / 加 PI|欺诈 AP 原法 / 加 PI|加 PI 回退|",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for x in pairs:
        if x["challenger"]=="pi": continue
        d=x["dataset"]
        if x["status"]!="complete":
            lines.append(f"|{d}|{x['baseline']}|未完成|—|—|—|—|—|—|")
            continue
        def metrics(k):
            return show(x.get("base_test_"+k),4)+" / "+show(x.get("test_"+k),4)
        lines.append(f"|{d}|{x['baseline']}|{x['k_base']} / {x['k']}|{show(x['base_s'])} / {show(x['s'])}|{show(x['speedup'])}|{metrics('r2' if d=='abalone' else 'auc')}|{metrics('mae' if d=='abalone' else 'accuracy')}|{metrics('ap') if d=='credit_card_fraud' else '—'}|{'是' if x['fallback'] else '否'}|")
    lines.extend(["","## 全部路线","",
                  "|数据集|方法|状态|前置留列|候选列|交付列|拟合|前置秒|活动秒|含启动秒|测试 AUC / R²|测试准确率 / MAE|欺诈 AP|回退|",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for x in rows:
        vals=[x['dataset'],x['method'],x['status']]
        vals += [str(x.get(k,"—")) for k in ('front_k','candidate_k','k','fits')]
        vals += [show(x.get(k),3 if k=='front_s' else 2) for k in ('front_s','elapsed_s','cold_wall_s')]
        vals += [show(x.get("test_scores_"+k),4) for k in (('r2','mae','ap') if x['dataset']=='abalone' else ('auc','accuracy','ap'))]
        vals += ["是" if x.get("fallback") else "否" if x["status"]=="completed" else "—"]
        lines.append("|"+"|".join(vals)+"|")
    (root/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return rows,pairs

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    rows,pairs=report(parser.parse_args().run)
    print(f"{len(rows)} routes, {len(pairs)} comparison rows")

