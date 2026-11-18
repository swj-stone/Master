"""Verify witnessed pruning outputs, fixed candidate order, timing, metrics and actual reductions."""
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from experiments.test.audit_piv3_lite import _fails, _score, _verify_scores
from experiments.test.report_compact_minpi import metrics_text
from experiments.test.report_piv3_lite_six import csv_table,losses
from experiments.test.run_compact_witness import CONTROLS,METHODS,guard_proposals,parent,read
from minppi_mofs.experiment_protocol import canonical_hash


def _require(ok,reason):
    if not ok:
        raise ValueError(reason)


def verify(folder, dataset, method, meta, labels, valid_labels):
    clock=read(folder/"watchdog.json")
    _require(clock["limit_s"]==300.,"Changed wall deadline")
    if clock["status"]=="timed_out":
        _require(clock["kill_requested_s"] is not None,"Missing actual time limit")
        return None,clock
    result=read(folder/"result.json")
    _require(clock["status"]=="completed" and clock["result"]==result and clock["exit_code"]==0
             and result["dataset"]==dataset and result["method"]==method and result["seed"]==42,
             "Mismatched route result")
    frozen,diag,access=(read(folder/f) for f in ("frozen.json","front.json","test_access.json"))
    _require(canonical_hash(frozen)==result["frozen_hash"]==access["frozen_hash"] and
             meta["sha256"]["test_y.npy"]==access["test_y_sha256"] and
             meta["sha256"]["test.npy"]==access["test_x_sha256"],"Test boundary changed")
    _require(0<result["elapsed_s"]<=clock["elapsed_s"] and clock["kill_requested_s"] is None and
             result["elapsed_s"]<=300 and result["front_fits"]==0,"Broken timing/fit count")
    cols=meta["features"]
    front,anchor,initial=frozen["front"],frozen["anchor"],frozen["initial"]
    _require(diag.get("anchor",front)==anchor and set(anchor)<=set(cols) and set(initial)<=set(front)<=set(anchor) and
             len(front)==result["front_k"] and len(anchor)==result["anchor_k"] and len(initial)==result["initial_k"],
             "Recorded anchor, witness or initial support disagrees")
    live=set(anchor)
    anchor_order=diag.get("anchor_order",diag["order"])
    _require(len(set(anchor_order))==len(anchor_order) and set(anchor_order[:len(anchor)])==live,"Anchor prefix")
    suffix=set(anchor_order[len(anchor):])
    medium=set(anchor_order[len(anchor):2*len(anchor)])
    aggregate=method.split("_")[1]
    for witness in diag.get("witnesses",[]):
        block=set(witness["block"])
        vote=np.asarray(witness["votes"])
        _require(block and block<=live and len(block)==len(witness["block"]) and
                 vote.shape==(3,4) and np.all(np.isin(vote,[-1,1])),"Invalid witness block")
        contexts=[live,live|medium,live|suffix]
        _require(witness["context_widths"]==[len(s) for s in contexts],"Wrong witness contexts")
        for h in range(3):
            fp=_fails(witness["before_point"][h],witness["after_point"][h],.03,.05)
            _require((not fp)==(vote[h,0]==1),"Point vote inconsistent")
        rates=vote.mean(1)
        np.testing.assert_allclose(rates,witness["net"],rtol=0.,atol=0.)
        score=(rates.min() if aggregate=="min" else rates.mean() if aggregate=="mean" else rates[0])
        _require(abs(score-witness["score"])<1e-12 and
                 (not _fails(diag["anchor_sample_scores"],witness["after_point"][0],.03,.05))==witness["cumulative_anchor_pass"]
                 and bool(score>0 and witness["cumulative_anchor_pass"])==witness["accepted"],
                 "Weakest stratum or cumulative quota inconsistency")
        if witness["accepted"]:
            live-=block
    _require(front==[c for c in cols if c in live] and len(diag.get("witnesses",[]))<=5,"Witness chain inconsistent")
    ri,gi=train_test_split(np.arange(len(valid_labels)),test_size=.5,random_state=42,
                           stratify=None if dataset=="abalone" else valid_labels)
    ri,gi=np.sort(ri),np.sort(gi)
    _require(frozen["ranking_rows_hash"]==canonical_hash(ri.tolist()) and
             frozen["guard_rows_hash"]==canonical_hash(gi.tolist()),"Ranking/guard identities")
    expected=list(guard_proposals(cols,initial,front,anchor,anchor_order,method.endswith("_pfi"),method in CONTROLS))
    expected_fits={tuple(cols)}
    if method.endswith("_pfi"):
        expected_fits.add(tuple(front))
    rank_fits=len(expected_fits)-1
    checks=frozen["checks"]
    _require(1<=len(checks)<=len(expected) and len(checks)==result["checks"],"Unexpected guard count")
    with np.load(folder/"predictions.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(z["y_test"],labels)
        np.testing.assert_array_equal(z["guard_rows"],gi)
        np.testing.assert_array_equal(z["y_guard"],valid_labels[gi])
        full=_score(z["y_guard"],z["guard_full"],dataset)
        test=_score(labels,z["test_pred"],dataset)
        _verify_scores(full,result["validation_full"])
        _verify_scores(test,result["test_scores"])
        _require(len(z["guard_checks"])==len(checks),"Guard predictions missing")
        selected,accepted=cols,"full_fallback"
        for n,check in enumerate(checks):
            role,coords=expected[n]
            g=_score(z["y_guard"],z["guard_checks"][n],dataset)
            _verify_scores(g,check["metrics"])
            fail=_fails(full,g,.03,.05)
            _require((role,coords)==(check["route"],check["names"]) and sorted(check["failed"])==fail,
                     "Guard repair order, support or threshold changed")
            expected_fits.add(tuple(coords))
            if not fail:
                _require(n==len(checks)-1,"Continued after accepted repair")
                selected,accepted=coords,role
        _require(accepted!="full_fallback" or len(expected)==len(checks),"Skipped repair")
        np.testing.assert_array_equal(z["guard_selected"],z["guard_full"] if accepted=="full_fallback" else z["guard_checks"][-1])
        _verify_scores(_score(z["y_guard"],z["guard_selected"],dataset),result["validation_selected"])
    _require(selected==frozen["selected"]==result["selected"] and len(selected)==result["k"] and
             accepted==result["accepted_route"]==frozen["accepted_route"] and result["fits"]==len(expected_fits)+1 and
             result["rank_fits"]==rank_fits and result["guard_fits"]==len(expected_fits)-rank_fits-1,
             "Frozen support, accepted route or fit ledger disagrees")
    return result,clock


def report(root,primary):
    root,primary=Path(root),Path(primary)
    proto=read(root/"protocol.json")
    _require(proto["primary_sha256"]==parent._sha(primary/"protocol.json") and
             proto["benchmark_sha256"]==parent._sha(parent.BASE/"protocol.json"),"Parent provenance")
    with (parent.BASE/"timing-repeats/timing-median.csv").open(encoding="utf-8-sig",newline="") as f:
        medians={(x["dataset"],x["method"]):float(x["median_active_s"]) for x in csv.DictReader(f)}
    rows=[]
    text=["# Compact Gain 起点的三层删除投票","",
          "六个原始数据集，seed 42。该分支基于前一轮初步结果提出。新路线独立冷启动，300 秒超时，活动秒和冷启动秒均实测一次。PFI 基线计时取既有三次中位数，其他比较为单次。","",
          "|数据集|方法|Gain 起点|投票后列数|最终列数|通过阶段|活动秒|冷启动秒|拟合|测试 AUC / 准确率 / AP，或 R² / MAE|",
          "|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for d in parent.DATASETS:
        datadir=parent.DEFAULT_ROOT/d
        meta=read(datadir/"metadata.json")
        y,v=np.load(datadir/"test_y.npy",mmap_mode="r"),np.load(datadir/"val_y.npy",mmap_mode="r")
        full=read(parent.BASE/d/"full/result.json")
        pfi=read(parent.BASE/d/"pfi/result.json")
        for m in METHODS:
            result,clock=verify(root/d/m,d,m,meta,y,v)
            row=dict(dataset=d,method=m,status=clock["status"],cold_wall_s=clock["elapsed_s"])
            rows.append(row)
            if result is None:
                text.append(f"|{d}|{m}|—|—|—|超时|—|{clock['elapsed_s']:.2f}|—|—|")
                continue
            for key in ("width","anchor_k","front_k","initial_k","k","fits","front_fits","rank_fits","guard_fits","front_s","selector_s",
                        "guard_s","reference_s","final_s","elapsed_s","accepted_route","fallback","repaired","witness_predictions"):
                row[key]=result[key]
            for key,value in result["test_scores"].items():
                row["test_"+key]=value
            for name,ref,seconds in (("full",full,full["elapsed_s"]),("fast_pfi",pfi,medians[d,"pfi"])):
                add_compare(row,result,name,ref,seconds)
            for name in ("compact_gain","compact_pfi","tree_min","pair_min"):
                folder=primary/d/name
                clock_base=read(folder/"watchdog.json")
                row[name+"_status"]=clock_base["status"]
                if clock_base["status"]=="completed":
                    ref=read(folder/"result.json")
                    add_compare(row,result,name,ref,ref["elapsed_s"])
            text.append(f"|{d}|{m}|{result['anchor_k']}|{result['front_k']}|{result['k']}|{result['accepted_route']}|"
                        f"{result['elapsed_s']:.2f}|{clock['elapsed_s']:.2f}|{result['fits']}|{metrics_text(result['test_scores'])}|")
    csv_table(root/"witness-comparison.csv",rows)
    (root/"comparison.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(f"{sum(r['status']=='completed' for r in rows)}/{len(rows)} audited witness routes.",flush=True)
    return rows


def add_compare(row,result,name,ref,seconds):
    l=losses(result["test_scores"],ref["test_scores"])
    row.update({name+"_k":ref["k"],name+"_s":seconds,name+"_active_speedup":seconds/result["elapsed_s"],
                name+"_worst_relative_loss":max(l.values()),name+"_within_5pct":max(l.values())<=.05+1e-12})
    for key,value in ref["test_scores"].items():
        row[name+"_test_"+key]=value

