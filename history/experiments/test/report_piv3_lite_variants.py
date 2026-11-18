"""Combined readout of the two lightweight PIv3 experiments and fixed-seed timing replicates."""
import argparse
import csv
import json
from pathlib import Path

def read(root,d,m):
    return json.loads((root/d/m/"result.json").read_bytes())

def csvrows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def metrics(record):
    s=record["test_scores"]
    if "mae" in s:
        return f"{s['r2']:.5f} / {s['mae']:.5f}"
    value=f"{s['auc']:.5f} / {100*s['accuracy']:.2f}%"
    return value+(f" / {s['ap']:.5f}" if "ap" in s else "")

def report(main,peer):
    timings={(x["dataset"],x["method"]):x for x in csvrows(main/"timing-repeats/timing-median.csv")}
    active=lambda d,m:float(timings[d,m]["median_active_s"])
    peer_comparisons=csvrows(peer/"peer-comparison.csv")
    rows=[]
    for d in dict.fromkeys(x["dataset"] for x in peer_comparisons):
        base=read(main,d,"pfi")
        for m in ("pi","pi_pfi"):
            lite=read(main,d,m)
            richer=read(peer,d,m)
            diag=json.loads((peer/d/m/"front.json").read_bytes())
            row=dict(dataset=d,route=m,width=base["width"],plain_pfi_k=base["k"],shadow_front_k=lite["front_k"],
                     peer_front_k=richer["front_k"],substitutions=diag["peer_removed_count"],
                     shadow_k=lite["k"],peer_k=richer["k"],peer_fallback=richer["fallback"],
                     pfi_median_s=active(d,"pfi"),shadow_median_s=active(d,m),
                     peer_s=richer["elapsed_s"],peer_front_s=richer["front_s"],
                     peer_cold_s=json.loads((peer/d/m/"watchdog.json").read_bytes())["elapsed_s"],
                     pfi_test=metrics(base),shadow_test=metrics(lite),peer_test=metrics(richer))
            for category,rec in (("pfi",base),("shadow",lite),("peer",richer)):
                for key,val in rec["test_scores"].items(): row[category+"_"+key]=val
            rows.append(row)
    with (peer/"combined.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)
    out=["# PIv3-Lite 单 seed 对照","",
         "六个数据集沿用完整原始预测列和存档数据划分，数据、预测模型与抽样 seed 均为 42。活动总时间从读入缓存数据到拟合、冻结特征、最终重训和测试。随机对照版与快速 PFI 的秒数取三次独立进程的中位数；增加相关替代检查的变体只执行一次。各次冷启动和 5 分钟上限见逐组记录。","",
         "测试指标分类为 AUC / 准确率，ISOLET 为宏平均 OvR AUC，欺诈集另列 AP，Abalone 为零截断 R² / MAE。","",
         "|数据集|路线|列数 PFI / 随机对照 / 加替代检查|活动秒 PFI / 随机对照 / 加替代检查|原 PFI 测试|随机对照版测试|加替代检查测试|",
         "|---|---|---:|---:|---|---|---|"]
    for x in rows:
        out.append(f"|{x['dataset']}|{x['route']}|{x['plain_pfi_k']} / {x['shadow_k']} / {x['peer_k']}|{x['pfi_median_s']:.2f} / {x['shadow_median_s']:.2f} / {x['peer_s']:.2f}|{x['pfi_test']}|{x['shadow_test']}|{x['peer_test']}|")
    out+=["","## 原始证据","",
          "[全部 90 条路线与 36 个配对的结果表](../PIv3-lite-raw-six-seed42/results.md) · [运行日志](run.log) · [三次计时](../PIv3-lite-raw-six-seed42/timing-repeats/timing-median.csv) · [替代检查实验 30 条明细](peer-comparison.csv) · [测试行 bootstrap 的损失上界](joint.csv)。"]
    (peer/"comparison.md").write_text("\n".join(out)+"\n",encoding="utf-8")
    return rows

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main",type=Path)
    parser.add_argument("peer",type=Path)
    args=parser.parse_args()
    print(len(report(args.main,args.peer)),"combined rows")

