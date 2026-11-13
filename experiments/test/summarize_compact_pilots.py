"""Build a comparison across all compact candidates from the audited measurements."""
from pathlib import Path
import argparse
import csv
import json
from statistics import median

from experiments.test.run_compact_minpi import DATASETS, BASE, DEFAULT
from experiments.test.report_piv3_lite_six import losses,csv_table

def read(path):
    return json.loads(path.read_bytes())

def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def fmt(v):
    if 'mae' in v:
        return f"{v['r2']:.4f} / {v['mae']:.4f}"
    line=f"{v['auc']:.4f} / {v['accuracy']:.2%}"
    return line + (f" / {v['ap']:.4f}" if 'ap' in v else "")

def report(primary,supp):
    by={}
    history={r['dataset']:float(r['median_active_s']) for r in read_csv(BASE/'timing-repeats/timing-median.csv') if r['method']=='pfi'}
    for run,filename in [(primary,'compact-routes.csv'),(supp,'witness-comparison.csv')]:
        for row in read_csv(run/filename):
            d,m=row['dataset'],row['method']
            by[d,m]=dict(status=row['status'])
            if row['status']!='completed': continue
            result=read(run/d/m/'result.json')
            t=[result['elapsed_s']]
            for n in (2,3):
                path=run/f'repeat-{n}'/d/m/'result.json'
                if path.exists(): t.append(read(path)['elapsed_s'])
            by[d,m]=dict(status='completed',k=result['k'],t=median(t),n=len(t),scores=result['test_scores'],fallback=result['fallback'])
    methods=sorted({m for d,m in by})
    cohort=[]; summary=[]
    for m in methods:
        data=[]
        for d in DATASETS:
            ref=read(BASE/d/'pfi/result.json')
            full=read(BASE/d/'full/result.json')
            r=by[d,m]
            if r['status']!='completed': continue
            a=max(losses(r['scores'],ref['test_scores']).values())
            b=max(losses(r['scores'],full['test_scores']).values())
            q=max(a,b)<=.05+1e-12
            compact=r['k']<ref['k']
            fast=r['t']<history[d]
            parity=r['t']<=1.10*history[d]
            row=dict(dataset=d,method=m,initial_width=ref['width'],pfi_k=ref['k'],delivered_k=r['k'],
                     pfi_s=history[d],active_s=r['t'],timing_observations=r['n'],fallback=r['fallback'],
                     pfi_test=fmt(ref['test_scores']),test=fmt(r['scores']),full_test=fmt(full['test_scores']),
                     max_loss_vs_full=b,max_loss_vs_pfi=a,all_metrics_pass=q,smaller=compact,faster=fast,
                     within_time_margin=parity,smaller_quality_within_time=q and compact and parity,
                     smaller_quality_faster=q and compact and fast)
            row.update({f'test_{key}':val for key,val in r['scores'].items()})
            data.append(row)
        cohort+=data
        count=lambda key:sum(x[key] for x in data)
        summary.append(dict(method=m,n=len(data),smaller=count('smaller'),faster=count('faster'),quality=count('all_metrics_pass'),
                            smaller_quality_faster=count('smaller_quality_faster'),smaller_quality_within_time=count('smaller_quality_within_time')))
    csv_table(supp/'all-methods.csv',cohort)
    csv_table(supp/'all-methods-summary.csv',summary)
    md=["# 紧凑 minPI 与快速 PFI 的六数据集对照","",
        "两批实验均使用相同的 seed 42、原始维度、预测模型和验证/测试划分。快速 PFI、主实验四条 min 路线以及补充实验三条接 PFI 的路线采用三次活动耗时中位数，其余为单次。各行列出实际计时次数。活动耗时包含读入、整模型拟合、筛选、独立验证、最终重训和测试。五分钟上限用含进程启动的实际墙钟执行。预测达标要求全部适用测试指标相对全列与快速 PFI 的损失均不超过 5%。时间接近预先定义为不超过快速 PFI 的 1.10 倍。","",
        "|路线|配对数|列数更少|总时间更短|预测达标|列更少＋达标＋更快|列更少＋达标＋时间在 1.10 倍内|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for x in summary:
        md.append('|'+ '|'.join(str(v) for v in x.values())+'|')
    for d in DATASETS:
        md+=["",f"## {d}","","分类报告 AUC / 准确率，欺诈集另列 AP。回归报告 R² / MAE。","",
             "|方法|最终列数|总秒|计时次数|测试指标|预测达标|回退全列|","|---|---:|---:|---:|---|---|---|"]
        base=read(BASE/d/'pfi/result.json')
        md.append(f"|快速 PFI|{base['k']}|{history[d]:.2f}|3|{fmt(base['test_scores'])}|参照|{'是' if base['fallback'] else '否'}|")
        for x in (x for x in cohort if x['dataset']==d):
            md.append(f"|{x['method']}|{x['delivered_k']}|{x['active_s']:.2f}|{x['timing_observations']}|{x['test']}|{'是' if x['all_metrics_pass'] else '否'}|{'是' if x['fallback'] else '否'}|")
    md+=["","[逐路线 CSV](all-methods.csv) · [逐项比较与消融](../PIv3-compact-raw-six-seed42/compact-results.md) · [删块投票与配对控制的实验明细](comparison.md)。"]
    (supp/'benchmark.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,default=DEFAULT)
    p.add_argument('--supplement',type=Path,default=DEFAULT.parent/'PIv3-compact-witness-raw-six-seed42')
    a=p.parse_args()
    report(a.primary,a.supplement)

