"""Validate the seed-42 three-part comparison and publish measured routes, pairs and fronts."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.test.audit_piv3_lite import _fails, _read, _require, _score, _sha, _subset, _verify_scores
from experiments.test.piv3_raw_splits import DEFAULT_ROOT, PROJECT
from experiments.test.report_compact_minpi import metrics_text
from experiments.test.report_piv3_lite_six import csv_table, losses
from experiments.test.run_piv3_lite_six import DATASETS
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


BASES = ("gain", "pfi", "nsga")
VARIANTS = ("plain", "min", "mean")
METHODS = ("full", "gain", "min_gain", "mean_gain", "pfi", "min_pfi", "mean_pfi",
           "nsga", "min_nsga", "mean_nsga", "mi", "extra_trees")
NAMES = dict(full="全列模型", gain="Gain", pfi="快速 PFI", nsga="NSGA-II", mi="互信息", extra_trees="ExtraTrees")
STATUSES = dict(completed="完成", timed_out="超时", pending="待运行")


def identifier(backend, variant):
    return backend if variant == "plain" else f"{variant}_{backend}"


def describe(method):
    parts = method.split("_", 1)
    return f"{parts[0]}＋{NAMES[parts[1]]}" if parts[0] in ("min", "mean") else NAMES[method]


def same_effect(a, b):
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= 1e-12 for k in a)


def _witness(diag, result):
    """Verify the actual block decisions, with common context sizes and no stale removals."""
    mode = result["method"].split("_")[0]
    if mode not in ("min", "mean"):
        _require(result["front_k"] == result["width"] and not diag.get("witnesses"), "Unexpected PI")
        return
    _require(diag["aggregate"] == mode and diag["front_fits"] == 0, "Frontend mode changed")
    anchor = diag["anchor"]
    order = diag["anchor_order"]
    rest = set(order[len(anchor):])
    medium = set(order[len(anchor):2*len(anchor)])
    live = set(anchor)
    _require(len(anchor) == diag["anchor_k"] and live == set(order[:len(anchor)]), "Gain anchor")
    for step in diag["witnesses"]:
        block = set(step["block"])
        votes = np.asarray(step["votes"])
        _require(block and block <= live and votes.shape == (3,4) and np.isin(votes,(-1,1)).all(),
                 "Invalid deletion votes")
        _require(step["context_widths"] == [len(live),len(live | medium),len(live | rest)],
                 "Wrong coalition context")
        rates = votes.mean(axis=1)
        np.testing.assert_array_equal(rates,step["net"])
        agg = float(min(rates) if mode == "min" else rates.mean())
        _require(abs(step["score"]-agg) < 1e-12, "Incorrect vote reducer")
        for j in range(3):
            ok = not _fails(step["before_point"][j],step["after_point"][j],.03,.05)
            _require(ok == (votes[j,0] == 1), "Point vote disagrees with scores")
        anchor_ok = not _fails(diag["anchor_sample_scores"],step["after_point"][0],.03,.05)
        _require(anchor_ok == step["cumulative_anchor_pass"] and
                 bool(agg > 0 and anchor_ok) == step["accepted"], "Quota witness error")
        if step["accepted"]:
            live -= block
    _require(len(diag["witnesses"]) <= 5 and set(result["front"]) == live, "Stale front")


def pareto_audit(backend, result):
    """Rebuild nondominance and compute two-dimensional area by horizontal strips."""
    rows = backend["evaluations"]
    primary = "r2" if result["dataset"] == "abalone" else "auc"
    points = []
    _require(backend["budget"] == 24 and backend["population"] == 8, "Search budget mismatch")
    _require(backend["support_width"]==result["front_k"] and backend["original_width"]==result["width"] and
             backend["unique_evaluations"]==result["nsga_count"] and len(backend["front"])==result["nsga_front_size"],
             "Search metadata mismatch")
    theoretical = 2**len(result["front"])-1 if result["front_k"] < 5 else 24
    target = min(24,theoretical)
    _require(backend["unique_evaluations"] == len(rows) == target, "Unique evaluation budget")
    _require(backend["exhaustive"] == (result["front_k"]<=4), "Exhaustive flag vs budget")
    _require(len({tuple(r["names"]) for r in rows}) == len(rows) and
             len({r["id"] for r in rows}) == len(rows), "Repeated subset evaluation")
    _require(any(r["names"] == result["front"] for r in rows), "Missing support endpoint")
    for r in rows:
        _require(r["k"] == len(r["names"]) and _subset(r["names"],result["front"]), "Search support")
        point = (r["k"]/result["width"],1.-r["scores"][primary])
        np.testing.assert_allclose(r["objectives"],point,rtol=0.,atol=1e-12)
        points.append(point)
    frontier = [i for i, p in enumerate(points)
                if not any(q[0]<=p[0] and q[1]<=p[1] and (q[0]<p[0] or q[1]<p[1]) for q in points)]
    expected = {rows[i]["id"]:rows[i] for i in frontier}
    _require(expected == {r["id"]:r for r in backend["front"]} and
             len(expected)==len(backend["front"]), "Pareto archive mismatch")
    ref = tuple(backend["ref_point"])
    _require(ref == (1.1,1.1), "Unexpected hypervolume reference")
    ordered = sorted(set(points[i] for i in frontier))
    hv = sum((next_x-x)*(ref[1]-y)
             for ((x,y),next_x) in zip(ordered,[p[0] for p in ordered[1:]]+[ref[0]],strict=True))
    _require(abs(backend["hv"]-hv) <= 1e-10 and abs(result["nsga_hv"]-hv) <= 1e-10,
             "Hypervolume mismatch")
    _require(backend["hv_history"] and np.all(np.diff(backend["hv_history"])>=-1e-12) and
             abs(backend["hv_history"][-1] - hv)<=1e-10, "Search trajectory mismatch")
    reference = backend["reference_ranking"]
    eligible = [r for r in backend["front"] if not _fails(reference,r["scores"],.03,.05)]
    if eligible:
        winner = min(eligible,key=lambda r:(r["k"],-r["scores"][primary],r["id"]))
    else:
        winner = min(backend["front"],key=lambda r:(max(0.,max(losses(r["scores"],reference).values())),
                                                    r["k"],r["id"]))
    _require(winner["id"] == backend["selected_id"] and winner["names"] == result["candidate"],
             "Representative must be chosen on ranking data")
    _require(backend["selection_mode"]==("smallest_feasible" if eligible else "least_worst_relative_loss"),
             "Selection-mode mismatch")
    return [dict(dataset=result["dataset"],method=result["method"],candidate_id=r["id"],
                 generation=r["generation"],origin=r["origin"],k=r["k"],fraction_original=r["objectives"][0],
                 ranking_primary_loss=r["objectives"][1],rank_chosen=r["id"]==winner["id"],
                 feature_names_json=json.dumps(r["names"],ensure_ascii=False,separators=(",",":")),
                 **{"ranking_"+k:v for k,v in r["scores"].items()}) for r in backend["front"]]


def read_route(folder, d, method, meta, y_val, rank, guard, y_test):
    timer = _read(folder/"watchdog.json")
    _require(timer["limit_s"] == 300. and timer["elapsed_s"] > 0, "Unexpected timer")
    row = dict(dataset=d,method=method,width=meta["width"],status=timer["status"],cold_s=timer["elapsed_s"])
    if timer["status"] == "timed_out":
        _require(timer["kill_requested_s"] is not None,"Missing actual timeout")
        row.update(timeout_phase=timer.get("progress",{}).get("phase"))
        return row, None, None, []
    r, f, access, diag, backend = (_read(folder/file) for file in
                                   ("result.json","frozen.json","test_access.json","front.json","backend.json"))
    _require(timer["status"]=="completed" and timer["result"]==r and timer["exit_code"]==0 and
             timer["kill_requested_s"] is None and 0<r["elapsed_s"]<=min(300.,timer["elapsed_s"]+1e-6),
             "Watchdog and result disagree")
    _require((r["dataset"],r["method"],r["width"],r["seed"])==(d,method,meta["width"],42), "Method identity")
    h = canonical_hash(f)
    _require(h==r["frozen_hash"]==access["frozen_hash"] and
             access["test_x_sha256"]==meta["sha256"]["test.npy"] and
             access["test_y_sha256"]==meta["sha256"]["test_y.npy"], "Test read before freeze or wrong SHA")
    _require(f["policy"]=={k:v for k,v in meta["policy"].items() if k!="utility"} and
             f["rank_rows_sha256"]==canonical_hash(rank.tolist()) and
             f["guard_rows_sha256"]==canonical_hash(guard.tolist()),"Split or model differs")
    _require(f["selected"]==r["selected"] and _subset(r["front"],meta["features"]) and
             _subset(r["candidate"],r["front"]) and _subset(r["selected"],meta["features"]) and
             [len(r[k]) for k in ("front","candidate","selected")]==[r["front_k"],r["candidate_k"],r["k"]],
             "Support or column order mismatch")
    _require(r["front_fits"]==0 and r["fits"]==2+r["backend_fits"]+r["guard_fits"],"Fit ledger")
    _require(f["columns_sha256"]==canonical_hash(meta["features"]) and
             all(r[key]==f[key] for key in ("front","candidate","selected","failed_metrics","reference_ranking",
                                            "validation_full","validation_candidate","validation_selected")),
             "Freeze, scores or mask identities differ")
    name=r["backend"]
    if name=="nsga":
        expected_backend=backend["unique_evaluations"]-int(r["front"]==meta["features"])
        expected_guard=0
        _verify_scores(backend["reference_ranking"],r["reference_ranking"])
    else:
        support_model=(name in ("gain","pfi"))
        expected_backend=int(name=="extra_trees" or (support_model and r["front"]!=meta["features"]))
        expected_guard=int(r["candidate"]!=meta["features"] and not(support_model and r["candidate"]==r["front"]))
    _require(r["backend_fits"]==expected_backend and r["guard_fits"]==expected_guard,"Cached fits mismatch")
    for key in ("front_s","backend_s","reference_s","guard_s","final_s"):
        _require(0<=r[key]<=r["elapsed_s"],f"Invalid timing {key}")
    _require(sum(r[k] for k in ("front_s","backend_s","reference_s","guard_s","final_s"))
             <=r["elapsed_s"]+1e-5,"Double-counted time")
    with np.load(folder/"predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["guard_rows"],guard)
        np.testing.assert_array_equal(z["ranking_rows"],rank)
        np.testing.assert_array_equal(z["y_guard"],y_val[guard])
        np.testing.assert_array_equal(z["y_test"],y_test)
        full=_score(y_val[guard],z["guard_full"],d)
        candidate=_score(y_val[guard],z["guard_candidate"],d)
        failed=_fails(full,candidate,.03,.05)
        np.testing.assert_array_equal(z["guard_selected"],z["guard_full"] if failed else z["guard_candidate"])
        test=_score(y_test,z["test_pred"],d)
    for key, actual in (("validation_full",full),("validation_candidate",candidate),
                        ("validation_selected",full if failed else candidate),("test_scores",test)):
        _verify_scores(r[key],actual)
    _require(r["fallback"] == bool(failed) and r["selected"] == (meta["features"] if failed else r["candidate"])
             and sorted(f["failed_metrics"])==failed,"Guard decision mismatch")
    _witness(diag,r)
    pareto = pareto_audit(backend,r) if method.endswith("nsga") else []
    for key in ("front_k","candidate_k","k","fits","backend_fits","guard_fits","front_s","backend_s",
                "reference_s","guard_s","final_s","elapsed_s","fallback","nsga_count","nsga_front_size","nsga_hv"):
        row[key]=r.get(key)
    row["anchor_k"]=diag.get("anchor_k")
    row["pfi_shuffled"]=backend.get("shuffled_features")
    row["permutation_s"]=r["permutation_s"]
    row["pfi_repeats"]=r["pfi_repeats"]
    if name=="pfi":
        _require(0<=r["permutation_s"]<=r["backend_s"] and r["pfi_repeats"]==3 and
                 r["pfi_shuffled_features"]==row["pfi_shuffled"]<=r["front_k"],"Permutation ledger")
    row["nsga_exhaustive"]=backend.get("exhaustive") if pareto else None
    row["failed_metrics"]=",".join(failed)
    for key in ("validation_full","validation_candidate","validation_selected","test_scores"):
        row.update({key+"_"+k:v for k,v in r[key].items()})
    row["test"]=metrics_text(test)
    return row,r,diag,pareto


def paired(d,back,results):
    direct,full = results.get(back),results.get("full")
    triplet=dict(dataset=d,backend=back)
    pairs=[]
    for variant in VARIANTS:
        obj=results.get(identifier(back,variant))
        if obj:
            triplet.update({variant+"_"+key:obj[key] for key in ("k","front_k","candidate_k","elapsed_s","fits","fallback")})
            triplet.update({variant+"_test_"+k:v for k,v in obj["test_scores"].items()})
            if full and direct:
                triplet[variant+"_passes_5pct_vs_full_and_plain"]=not(_fails(full["test_scores"],obj["test_scores"],.05,.05)
                                                                     or _fails(direct["test_scores"],obj["test_scores"],.05,.05))
    for variant,other in (("min","plain"),("mean","plain"),("min","mean")):
        key,ref=identifier(back,variant),identifier(back,other)
        x,y=results.get(key),results.get(ref)
        out=dict(dataset=d,backend=back,challenger=key,reference=ref,
                 status="completed" if x and y else "incomplete")
        if x and y:
            worst = losses(x["test_scores"],y["test_scores"])
            a,b=set(x["front"]),set(y["front"])
            sx,sy=set(x["selected"]),set(y["selected"])
            out.update(k=x["k"],ref_k=y["k"],delta_k=x["k"]-y["k"],delta_active_s=x["elapsed_s"]-y["elapsed_s"],
                       speedup=y["elapsed_s"]/x["elapsed_s"],front_jaccard=len(a&b)/len(a|b),
                       pfi_shuffled_difference=(x["pfi_shuffled_features"]-y["pfi_shuffled_features"]
                                                if back=="pfi" else None),
                       selected_jaccard=len(sx&sy)/len(sx|sy),same_selected=sx==sy,
                       same_front=a==b,same_test=same_effect(x["test_scores"],y["test_scores"]),
                       max_relative_loss_vs_reference=max(worst.values()),
                       **{"diff_test_"+k:x["test_scores"][k]-v for k,v in y["test_scores"].items()},
                       **{"loss_vs_ref_"+k:v for k,v in worst.items()})
            if full and direct:
                plain_loss=losses(x["test_scores"],direct["test_scores"])
                full_loss=losses(x["test_scores"],full["test_scores"])
                out.update(max_relative_loss_vs_plain=max(plain_loss.values()),max_relative_loss_vs_full=max(full_loss.values()),
                           pass_5pct_vs_full_and_plain=max(*plain_loss.values(),*full_loss.values())<=.05+1e-12)
            if back=="nsga":
                out.update(hv_difference=x["nsga_hv"]-y["nsga_hv"])
        pairs.append(out)
    return triplet,pairs


def conventional(d, results):
    rows=[]
    for variant in VARIANTS:
        m=identifier("nsga",variant)
        for b in ("full","gain","pfi","mi","extra_trees"):
            x,y=results.get(m),results.get(b)
            row=dict(dataset=d,method=m,baseline=b,status="completed" if x and y else "incomplete")
            if x and y:
                loss=losses(x["test_scores"],y["test_scores"])
                full=results.get("full")
                row.update(k=x["k"],baseline_k=y["k"],delta_k=x["k"]-y["k"],
                           active_s=x["elapsed_s"],baseline_s=y["elapsed_s"],speedup=y["elapsed_s"]/x["elapsed_s"],
                           max_relative_loss_vs_baseline=max(loss.values()),
                           test_pass_vs_baseline=max(loss.values())<=.05+1e-12,
                           **{"test_delta_"+key:x["test_scores"][key]-v for key,v in y["test_scores"].items()})
                if full:
                    row["test_pass_vs_full_and_baseline"]=(row["test_pass_vs_baseline"] and
                        not _fails(full["test_scores"],x["test_scores"],.05,.05))
            rows.append(row)
    return rows


def markdown(rows,pairs):
    by={(r["dataset"],r["method"]):r for r in rows}
    lines=["# 六个原始数据集的 minPI 三部分实验","",
           "单 seed 42，每个数据集、每个方法的独立冷进程最多 300 秒。活动总秒涵盖读入、全列训练、前置、后端、验证、最终重训和测试。冷总秒另含进程启动与退出。每格的列数为验证后实际交付列数。","",
           "分类指标按 AUC / 准确率列出，ISOLET 使用宏平均 OvR AUC。信用卡欺诈集额外列 AP。Abalone 为零截断 R² / MAE，MAE 越小越好。前置先确定紧凑 Gain 起点，在三个特征上下文中投票删除块；min 和 mean 共用模型、样本和种子。排名半区与守门验证半区分开。候选若相对验证全列分数损失超过 3%（AP 为 5%），最终交付全列模型。",""]
    for title,methods in (("一　Gain",("gain","min_gain","mean_gain")),("二　快速 PFI",("pfi","min_pfi","mean_pfi")),
                          ("三　多目标特征选择",("nsga","min_nsga","mean_nsga","full","gain","pfi","mi","extra_trees"))):
        lines += ["","## "+title,""]
        if title.startswith("三"):
            lines += ["NSGA-II 同时最小化排名半区上的首要预测损失与原始宽度中的特征占比。各路线使用相同群体和候选预算；搜索空间不足时穷尽其所有非空子集。表中测试分数属于最终交付模型，每条搜索路线的排名前沿另见下表。",""]
        lines+=["|数据集|方法|前置留列|候选列|交付列|拟合|含前置活动秒|冷总秒|测试指标|回退全列|",
                "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
        for d in DATASETS:
            for m in methods:
                x=by[d,m]
                if x["status"]!="completed":
                    lines.append(f"|{d}|{describe(m)}|—|—|—|—|—|{x.get('cold_s','—')}|{STATUSES[x['status']]}|—|")
                    continue
                lines.append(f"|{d}|{describe(m)}|{x['front_k']}|{x['candidate_k']}|{x['k']}|{x['fits']}|"
                             f"{x['elapsed_s']:.2f}|{x['cold_s']:.2f}|{x['test']}|{'是' if x['fallback'] else '否'}|")
    lines+=["","### 排名半区 Pareto 前沿","","HV 在同一原始宽度尺度上计算，使用参考点 (1.1, 1.1)，越大越好。HV 衡量本轮已探索点的前沿覆盖度。","",
            "|数据集|方法|搜索维度|已评估不同子集|前沿子集数|HV|","|---|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        for v in VARIANTS:
            r=by[d,identifier("nsga",v)]
            if r["status"]=="completed":
                lines.append(f"|{d}|{describe(r['method'])}|{r['front_k']}|{r['nsga_count']}|{r['nsga_front_size']}|{r['nsga_hv']:.5f}|")
    lines+=["","### min 与 mean 的直接消融","","两条路线使用同一 Gain 起点、抽样行和三层上下文。Jaccard 是两组特征交集与并集的数量之比。时间比为 mean 的活动耗时除以 min 的活动耗时。","",
            "|数据集|后端|前置 Jaccard|最终列 min / mean|最终 Jaccard|测试指标是否相同|mean/min 时间比|",
            "|---|---|---:|---:|---:|---|---:|"]
    for p in pairs:
        if (p["challenger"].startswith("min_") and p["reference"].startswith("mean_")
                and p["status"]=="completed"):
            lines.append(f"|{p['dataset']}|{NAMES[p['backend']]}|{p['front_jaccard']:.3f}|"
                         f"{p['k']} / {p['ref_k']}|{p['selected_jaccard']:.3f}|"
                         f"{'是' if p['same_test'] else '否'}|{p['speedup']:.2f}|")
    lines+=["","[全部路线](routes.csv) · [18 组直接、min 与 mean 横向对照](triplets.csv) · [逐对差值](paired.csv) · [分组计数](summary.csv) · [多目标路线与五种参照](mofs-conventional.csv) · [完整排名 Pareto 点](pareto.csv) · [审计计数](audit.json)"]
    return "\n".join(lines)+"\n"


def summary(pairs):
    rows=[]
    for backend in BASES:
        for challenger,reference in (("min","plain"),("mean","plain"),("min","mean")):
            subset=[p for p in pairs if p["challenger"]==identifier(backend,challenger) and
                    p["reference"]==identifier(backend,reference) and p["status"]=="completed" and
                    "pass_5pct_vs_full_and_plain" in p]
            row=dict(backend=backend,challenger=identifier(backend,challenger),reference=identifier(backend,reference),
                     n_pairs=len(subset),k_smaller=sum(x["delta_k"]<0 for x in subset),
                     faster=sum(x["speedup"]>1 for x in subset),same_selected=sum(x["same_selected"] for x in subset),
                     same_front=sum(x["same_front"] for x in subset),
                     test_pass_vs_full_and_plain=sum(x["pass_5pct_vs_full_and_plain"] for x in subset),
                     smaller_faster_test_pass=sum(x["delta_k"]<0 and x["speedup"]>1 and
                                                 x["pass_5pct_vs_full_and_plain"] for x in subset))
            if backend=="nsga":
                row["larger_ranking_hv"]=sum(x["hv_difference"]>1e-12 for x in subset)
                row["same_ranking_hv"]=sum(abs(x["hv_difference"])<=1e-12 for x in subset)
            rows.append(row)
    return rows


def report(root, allow_pending=False):
    root=Path(root)
    spec=_read(root/"protocol.json")
    _require(spec["seed"]==42 and set(METHODS)==set(spec["methods"]) and set(DATASETS)==set(spec["datasets"]),
             "Incorrect scheduled experiment grid")
    for name,h in spec["source_sha256"].items():
        _require(_sha(PROJECT/name)==h,f"Changed implementation: {name}")
    all_rows,triplets,pairs,fronts,conventional_rows=[],[],[],[],[]
    for d in DATASETS:
        data=DEFAULT_ROOT/d
        meta=_read(data/"metadata.json")
        _require(meta["seed"]==42 and meta["dataset"]==d and spec["data"][d]["hash"]==_sha(data/"metadata.json"),
                 f"Dataset provenance: {d}")
        for file in ("val_y.npy","test_y.npy"):
            _require(meta["sha256"][file]==_sha(data/file),f"Changed {d}/{file}")
        valid,test=(np.load(data/file,allow_pickle=False) for file in ("val_y.npy","test_y.npy"))
        ri,gi=train_test_split(np.arange(len(valid)),test_size=.5,random_state=42,stratify=None if d=="abalone" else valid)
        rank,guard=np.sort(ri),np.sort(gi)
        results,front_keys={},{}
        for m in METHODS:
            folder=root/d/m
            if not (folder/"watchdog.json").is_file() and allow_pending:
                all_rows.append(dict(dataset=d,method=m,width=meta["width"],status="pending"))
                continue
            try:
                row,result,diag,pareto=read_route(folder,d,m,meta,valid,rank,guard,test)
            except (AssertionError,KeyError,ValueError,OSError) as err:
                raise ValueError(f"{folder}: {err}") from err
            all_rows.append(row)
            if result:
                results[m]=result
                fronts.extend(pareto)
                mode=m.split("_")[0]
                if mode in ("min","mean"):
                    curr=(tuple(result["front"]),canonical_hash(diag))
                    _require(front_keys.setdefault(mode,curr)==curr,"Frontend differs across matched backends")
                    # Identical reweighting, sample rows, initial soft mask and its prediction
                    # checks ensure the ablation is paired before the first reduction decision.
                    context={k:diag.get(k) for k in ("anchor","anchor_order","witness_rows","anchor_sample_scores",
                                                     "masked_proposals","initial_width","ranking_rows")}
                    h=canonical_hash(context)
                    _require(front_keys.setdefault("anchor",h)==h,"Min/mean Gain anchor differs")
        if "full" in results:
            for m,r in results.items():
                _verify_scores(r["validation_full"],results["full"]["validation_full"])
                _verify_scores(r["reference_ranking"],results["full"]["reference_ranking"])
                if r["selected"]==meta["features"]:
                    _verify_scores(r["test_scores"],results["full"]["test_scores"])
        for b in BASES:
            group,comparisons=paired(d,b,results)
            triplets.append(group)
            pairs.extend(comparisons)
        conventional_rows.extend(conventional(d,results))
    csv_table(root/"routes.csv",all_rows)
    csv_table(root/"triplets.csv",triplets)
    csv_table(root/"paired.csv",pairs)
    csv_table(root/"summary.csv",summary(pairs))
    csv_table(root/"mofs-conventional.csv",conventional_rows)
    if fronts:
        csv_table(root/"pareto.csv",fronts)
    stats=dict(routes=dict(Counter(r["status"] for r in all_rows)),paired_comparisons=len(pairs),triplets=len(triplets),
               complete_pairs=sum(r["status"]=="completed" for r in pairs),pareto_points=len(fronts))
    atomic_write_json(root/"audit.json",stats)
    (root/"three-part-results.md").write_text(markdown(all_rows,pairs),encoding="utf-8")
    return stats


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--allow-pending",action="store_true")
    args=parser.parse_args()
    print(report(args.run,args.allow_pending))

