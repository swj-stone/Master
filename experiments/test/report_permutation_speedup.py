"""Audit the permutation speed pilot against the archived no-fallback experiment. No fits."""
import argparse
import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from experiments.audit_literature_methods import json_hash, metrics, require
from experiments.test.piv3_raw_splits import DEFAULT_ROOT as DATA


ROOT = Path(__file__).resolve().parents[2]/"results/test/thesis_sampling"
OLD = ROOT/"literature-games-raw-six-seed42-no-gate"
PILOT = ROOT/"permutation-speedup-raw42-20261114083150"
METHODS = ("minshap","shapley")
FROZEN = ("madelon","gisette","isolet")
NATIVE = ("spambase","madelon")


def read(path):
    return json.loads(path.read_bytes())


def ledger(folder, status):
    path = folder/"fits.jsonl"
    lines = path.read_bytes().splitlines() if path.exists() else []
    records = []
    for i,line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except ValueError:
            require(i == len(lines)-1 and status != "completed", "Incomplete fit record")
    inflight, completed, started, peak = {}, [], 0, 0
    for i,event in enumerate(records):
        if i:
            require(event["elapsed_s"] >= records[i-1]["elapsed_s"], "Fit clock")
        fit_id = event["fit_id"]
        if event["event"]=="started":
            started += 1
            require(fit_id == started, "Fit start sequence")
            inflight[fit_id] = event
            peak = max(peak,len(inflight))
        else:
            require(fit_id in inflight and event["event"] in ("completed","failed"), "Orphan or duplicate fit")
            prev = inflight.pop(fit_id)
            require(all(event.get(k)==prev.get(k) for k in ("phase","k","methods","prefix_sha256","thread_id","pid")), "Fit pairing")
            if event["event"]=="completed":
                completed.append(event)
    require(status != "completed" or len(completed)==started, "Unfinished or failed fit on completed arm")
    return completed,started,peak


def timer(folder):
    if (folder/"watchdog.json").is_file():
        record = read(folder/"watchdog.json")
        require(record["status"] in ("completed","timed_out","error") and record["limit_s"] == 300, "Watchdog")
        return record["status"],record["elapsed_s"]
    return ("running_or_interrupted" if folder.is_dir() else "not_started"),None


def fit_audit(path, fit_rows, m, paired, dataset, mode, meta):
    r,info = read(path/"result.json"),read(path/"estimator.json")
    require((r["dataset"],r["mode"],r["method"],r["width"],r["seed"]) ==
            (dataset,mode,m,meta["width"],42), "Result identity")
    require(r["enforce_guard"] is False and r["fallback"] is False and
            r["candidate"] == r["selected"] and r["candidate_k"] == r["k"] == len(r["selected"]),
            "Unexpected guard or dimensions")
    members = set(r["selected"])
    require(r["selected"] == [name for name in meta["features"] if name in members], "Feature names")
    frozen,access = read(path/"frozen.json"),read(path/"test_access.json")
    require(json_hash(frozen) == r["frozen_hash"] == access["frozen_hash"] and frozen["selected"] == r["selected"]
            and access["test_x_sha256"] == meta["sha256"]["test.npy"]
            and access["test_y_sha256"] == meta["sha256"]["test_y.npy"], "Test leakage checkpoint")
    owned = [e for e in fit_rows if not paired or m in e["methods"]]
    counts = Counter(e["phase"] for e in owned)
    require(len(owned) == r["fits"] == r["fits_started"] and counts["reference"] == 1 and
            counts["guard"] == (0<r["k"]<r["width"]) and counts["final"] == bool(r["k"]) and
            counts["coalition"] == r["selection_fits"], "Method fit accounting")
    require(all(e["k"] == (r["width"] if e["phase"]=="reference" else r["k"])
                for e in owned if e["phase"]!="coalition"), "Refit support widths")
    require(r["actual_coalitions"] == info["oracle_prefixes"] and
            (mode=="native" or r["selection_fits"]==0), "Oracle fit accounting")
    with np.load(path/"predictions.npz",allow_pickle=False) as z:
        for role,expected,target in (("test_pred","test_scores","y_test"),("guard_full","validation_full","y_guard"),
                                     ("guard_candidate","validation_candidate","y_guard")):
            measured = metrics(z[target],z[role],meta["task"],dataset=="credit_card_fraud")
            require(measured.keys()==r[expected].keys() and
                    all(np.isclose(v,r[expected][k],rtol=0,atol=1e-12) for k,v in measured.items()), "Metrics")
        np.testing.assert_array_equal(z["guard_selected"],z["guard_candidate"])
    return r,info


def comparison(new, old, new_info, out, ref, support):
    a,b = read(new/"result.json"),read(old/"result.json")
    out["reference"] = ref
    out["identical_candidate"] = a["candidate"] == b["candidate"] and a["selected"] == b["selected"]
    out["identical_metrics"] = all(a[k] == b[k] for k in ("test_scores","validation_candidate","validation_full"))
    with np.load(new/"predictions.npz") as y,np.load(old/"predictions.npz") as x:
        out["identical_test_predictions"] = np.array_equal(y["test_pred"],x["test_pred"])
        out["identical_archived_prediction_arrays"] = (set(y.files) == set(x.files) and
                                                       all(np.array_equal(y[k],x[k]) for k in x.files))
    archive = read(old/"estimator.json")
    require(new_info["permutation_sha256"] == archive["permutation_sha256"], "Permutation mismatch")
    if support:
        source = np.asarray(archive["sampled_marginals"])
        first = np.where((source<0).any(axis=0),(source<0).argmax(axis=0),-1)
        steps = np.where(first<0,source.shape[0],first+1)
        allowed = np.arange(source.shape[0])[:,None] < steps
        observed = np.where(allowed,source,np.inf)
        min_idx = observed.argmin(axis=0)
        sd = np.asarray(archive["variance_of_mean"])[min_idx,np.arange(source.shape[1])]
        threshold = np.sqrt(-2*np.log(new_info["alpha"])*sd)
        out["identical_certificates"] = all(np.array_equal(new_info[k],v) for k,v in (
            ("per_feature_marginals",steps),("negative_certificate_path",first),
            ("observed_minimum",observed.min(axis=0)),("threshold_of_observed_minimum",threshold)))
    else:
        float_keys = ["scores","sampled_marginals"]
        if a["mode"]=="native" and a["method"]=="minshap":
            float_keys += ["variance_of_mean","thresholds","worst_path"]
        for key in float_keys:
            current,previous = np.asarray(new_info[key]),np.asarray(archive[key])
            require(current.shape == previous.shape,"Score/trace shape")
            out["identical_"+key] = np.array_equal(current,previous)
            out["max_difference_"+key] = np.abs(current-previous).max().item() if current.size else 0.
    out["all_compared_fields_bitwise_equal"] = all(v for k,v in out.items() if k.startswith("identical_"))


def check_evidence(info,r,names):
    n = len(names)
    rng = np.random.default_rng(42)
    paths = np.asarray([rng.permutation(n) for _ in range(4)],dtype="<u4")
    require(sha256(paths.tobytes()).hexdigest() == info["permutation_sha256"] and info["paths"] == 4,
            "Seeded paths")
    if info.get("support_only"):
        counts = np.asarray(info["per_feature_marginals"])
        cert = np.asarray(info["negative_certificate_path"])
        mins = np.asarray(info["observed_minimum"])
        thresholds = np.asarray(info["threshold_of_observed_minimum"])
        require(all(v.shape == (n,) for v in (counts,cert,mins,thresholds))
                and np.array_equal(counts,np.where(cert<0,4,cert+1)) and ((cert>=-1)&(cert<4)).all()
                and np.array_equal(mins<0,cert>=0) and info["sampled_marginal_count"] == counts.sum()
                and np.isfinite(mins).all() and np.isfinite(thresholds).all() and np.all(thresholds>=0),
                "Negative certificate/coverage")
        keep = (cert<0)&(mins>=thresholds)
        require([c for c,k in zip(names,keep) if k]==r["selected"],"Certified selection")
        return
    matrix = np.asarray(info["sampled_marginals"])
    require(matrix.shape == (4,n) and np.isfinite(matrix).all(), "Complete trace")
    calc = matrix.min(axis=0) if r["method"] == "minshap" else matrix.mean(axis=0)
    require(np.array_equal(calc,info["scores"]),"Recomputed score")
    if "thresholds" in info:
        j = matrix.argmin(axis=0)
        threshold = np.sqrt(-2*np.log(info["alpha"])*np.asarray(info["variance_of_mean"])[j,np.arange(n)])
        require(np.array_equal(j,info["worst_path"]) and np.array_equal(threshold,info["thresholds"])
                and [c for c,k in zip(names,calc>=threshold) if k]==r["selected"],"MinShap decision")


def check_parallel_masks(events,width,complete):
    """Verify every recorded original-column mask independently against the prescribed four paths."""
    recorded=[e for e in events if e["phase"]=="coalition"]
    if not recorded or "prefix_sha256" not in recorded[0]:
        return
    rng=np.random.default_rng(42)
    expected={}
    for _ in range(4):
        prefix=[]
        for j in rng.permutation(width)[:-1]:
            prefix.append(int(j))
            digest=sha256(np.asarray(sorted(prefix),dtype="<u4").tobytes()).hexdigest()
            expected[digest]=len(prefix)
    ids={e["prefix_sha256"] for e in recorded}
    require(len(ids)==len(recorded) and all(expected.get(e["prefix_sha256"])==e["k"] for e in recorded),
            "Duplicate or invalid physical coalition fit in native joint ledger")
    require(not complete or ids==set(expected),"Incomplete native mask schedule")


def write_table(path,rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path,routes,results):
    def fmt(v,places=2):
        return "—" if v is None else f"{v:.{places}f}"
    status={"completed":"完成","timed_out":"超时","running_or_interrupted":"未取得终态",
            "not_started":"未启动","error":"错误"}
    forms={"single":"独跑","pair":"两方法联合","support":"MinShap 证书"}
    lines=["# 排列方法加速实验", "",
           "冷启动总秒包含数据加载、全列模型、估计、选列、候选验证及最终重训。联合运行只计一份时间与物理拟合。所有路线关闭验证损失回退。", "",
           "|数据集|口径|运行方式|旧独跑秒¹|新冷启动秒|加速比¹|相对新独跑合计|新物理拟合²|状态|",
           "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in routes:
        method = r["method"] if r["run"]=="single" else ""
        lines.append(f"|{r['dataset']}|{r['mode']}|{forms[r['run']]} {method}|{fmt(r['old_complete_seconds'])}|"
                     f"{fmt(r['cold_seconds'])}|{fmt(r['old_speedup'])}|{fmt(r.get('speedup_over_new_standalone_sum'))}|"
                     f"{r['fits_completed']}|{status[r['status']]}|")
        for prefix, title in (("first_thread","首次双线程尝试"),("thread_retry","双线程复跑")):
            if prefix+"_status" in r:
                lines.append(f"|{r['dataset']}|{r['mode']}|{title}|—|{fmt(r[prefix+'_seconds'])}|"
                             f"—|—|{r[prefix+'_completed_fits']}|{status[r[prefix+'_status']]}|")
    lines.extend(["","¹ 联合行的旧秒数为两种旧方法分别完成的冷启动时间之和。旧路线超时时不计算速度比。",
                  "² 拟合以完成事件计数，超时路线的未完成拟合另存于 CSV 的启动次数。","",
                  "|数据集|口径|运行方式|估计量|保留列|测试 AUC|准确率|完整参照与输出复核|",
                  "|---|---|---|---|---:|---:|---:|---|"])
    for r in results:
        if r["status"]!="completed":
            verify=status[r["status"]]
        elif "all_compared_fields_bitwise_equal" not in r:
            verify="内部证据通过，缺少完整参照"
        else:
            verify="逐值相同" if r["all_compared_fields_bitwise_equal"] else "存在差异，见 CSV"
        lines.append(f"|{r['dataset']}|{r['mode']}|{forms[r['run']]}|{r['method']}|{r.get('k','—')}|"
                     f"{fmt(r.get('auc'),5)}|{fmt(r.get('accuracy'),5)}|{verify}|")
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def pair_folder(root,dataset,mode,native_root=None):
    if mode=="native" and native_root:
        return native_root/dataset/mode
    candidates = [root/dataset/mode, root/mode/dataset/mode,
                  root.with_name(root.name+"-"+mode)/dataset/mode]
    available = [p for p in candidates if p.is_dir()]
    require(len(available)<=1, f"Ambiguous pair artifacts for {dataset}/{mode}: {available}")
    return next(iter(available),candidates[0])


def report(old,single,pair,support,output,*,pair_native=None,single_native=None):
    require(not output.resolve().is_relative_to(old.resolve()),"Output must be outside the read-only baseline")
    if pair_native is None:
        pair_native = next((d for tag in ("-native-process","-native-verified")
                            if (d:=pair.with_name(pair.name+tag)).is_dir()),None)
    routes,results,seen=[],[],{}
    roots = dict(single=single,pair=pair,support=support)
    planned = ([("single",d,"frozen",m) for d in FROZEN for m in METHODS] +
               [("single",d,"native","shapley") for d in NATIVE] +
               [("pair",d,mode,"permutation_pair") for mode,ds in (("frozen",FROZEN),("native",NATIVE)) for d in ds] +
               [("support",d,"native","minshap") for d in NATIVE])
    for kind,d,mode,method in planned:
        arm = pair_folder(pair,d,mode,pair_native) if kind=="pair" else roots[kind]/d/mode/method
        if kind=="single" and mode=="native":
            arm=(single_native or single.with_name(single.name+"-native"))/d/mode/method
        status,wall = timer(arm)
        physical,started,peak = ledger(arm,status)
        members = METHODS if kind=="pair" else [method]
        prior = [timer(old/d/mode/m) for m in members]
        complete_refs = all(s=="completed" for s,_ in prior)
        old_seconds = sum(t for _,t in prior) if complete_refs else None
        loc = dict(run=kind,dataset=d,mode=mode,method=method)
        rec = dict(**loc,status=status,cold_seconds=wall,fits_completed=len(physical),fits_started=started,
                   peak_simultaneous_fits=peak,
                   old_status="+".join(s for s,_ in prior),old_complete_seconds=old_seconds,
                   old_speedup=(old_seconds/wall if status=="completed" and old_seconds is not None else None))
        ledger_old = [ledger(old/d/mode/m,prior[i][0])[0] for i,m in enumerate(members)]
        rec["old_completed_fits"] = sum(map(len,ledger_old))
        if mode=="native" and kind != "support":
            check_parallel_masks(physical,read(DATA/d/"metadata.json")["width"],status=="completed")
        if kind=="pair":
            if mode=="native":
                for suffix,prefix in (("-native","first_thread"),("-native-verified","thread_retry")):
                    previous = pair.with_name(pair.name+suffix)/d/mode
                    if previous != arm and (previous/"watchdog.json").is_file():
                        previous_status,previous_seconds=timer(previous)
                        events,started_then,peak_then=ledger(previous,previous_status)
                        rec.update({prefix+"_status":previous_status,prefix+"_seconds":previous_seconds,
                                    prefix+"_completed_fits":len(events),prefix+"_started_fits":started_then,
                                    prefix+"_peak_fits":peak_then})
            solo = [timer(single/d/mode/m) for m in members]
            if all(s=="completed" for s,_ in solo):
                rec["new_standalone_sum_seconds"] = sum(t for _,t in solo)
                if status=="completed":
                    rec["speedup_over_new_standalone_sum"] = rec["new_standalone_sum_seconds"]/wall
        routes.append(rec)
        if status!="completed":
            for m in members:
                results.append(dict(run=kind,dataset=d,mode=mode,method=m,status=status))
            continue
        summary = read(arm/"result.json")
        require(summary == read(arm/"watchdog.json")["result"] and len(physical)==summary["fits"], "Route ledger")
        meta=read(DATA/d/"metadata.json")
        for m in members:
            path = arm/m if kind=="pair" else arm
            r,info = fit_audit(path,physical,m,kind=="pair",d,mode,meta)
            check_evidence(info,r,meta["features"])
            if kind=="pair":
                require(summary["method_fit_counts"][m] == r["fits"] and
                        summary["methods"][m]["frozen_hash"] == r["frozen_hash"], "Joint subresult")
            row = dict(run=kind,dataset=d,mode=mode,method=m,status=status,k=r["k"],dependent_fits=r["fits"],
                       oracle_calls=r["actual_coalitions"],marginals=info["sampled_marginal_count"],
                       estimate_s=r["estimate_s"],**r["test_scores"])
            reference = old/d/mode/m
            if timer(reference)[0] == "completed":
                comparison(path,reference,info,row,"archived standalone",kind=="support")
            elif (d,mode,m) in seen:
                comparison(path,seen[d,mode,m],info,row,"new complete estimator",kind=="support")
            else:
                row["reference"]="unavailable: no completed exact reference"
            if kind=="pair" and (d,mode,m) in seen:
                second={}
                comparison(path,seen[d,mode,m],info,second,"new standalone",False)
                row["also_matches_new_standalone"] = second["all_compared_fields_bitwise_equal"]
            if kind != "support":
                seen[d,mode,m] = path
            results.append(row)
    output.mkdir(parents=True,exist_ok=True)
    write_table(output/"speed-timings.csv",routes)
    write_table(output/"speed-equivalence.csv",results)
    write_summary(output/"speed-summary.md",routes,results)
    return routes,results


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old",type=Path,default=OLD)
    parser.add_argument("--single",type=Path,default=PILOT/"single")
    parser.add_argument("--pair",type=Path,default=PILOT/"pair")
    parser.add_argument("--pair-native",type=Path,help="Native joint results if under a separate retry directory")
    parser.add_argument("--single-native",type=Path,help="Native standalone results if under a separate retry directory")
    parser.add_argument("--support",type=Path,default=PILOT/"support")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    rows,evidence=report(args.old,args.single,args.pair,args.support,args.output or args.single.parent,
                         pair_native=args.pair_native,single_native=args.single_native)
    print(dict(Counter(r["status"] for r in rows)))
    print(f"Bitwise-equal against complete references: "
          f"{sum(r.get('all_compared_fields_bitwise_equal',False) for r in evidence)} / "
          f"{sum('all_compared_fields_bitwise_equal' in r for r in evidence)}")
