"""Verify the seed-42 Madelon paired pruning trials and summarize search-seed stability."""

import argparse
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_piv0_benchmark import read, sha
from experiments.run_thesis_sampling import ROOT
from experiments.test.audit_piv2_prefilter import metric_pair, near, ordered_subset, require
from experiments.test.report_madelon500_piv2 import verify_inputs
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


METHODS = {"old": "上一版单路径", "piv3": "PIv3"}
SEEDS = (11, 23, 47)
TOLS = (.01, .05)
OLD = ROOT / "results/test/thesis_sampling/adaptive-pruning-validation-20261113-175433-248"


def passes(scores, reference, tol):
    return bool(np.all(np.asarray(scores) >= np.asarray(reference) * (1-tol) - 1e-12))


def jaccard(a, b):
    return len(a & b) / len(a | b)


def check_history(steps, selected, seed, tolerance):
    historical = read(OLD / f"trace-t{round(tolerance*100)}-s{seed}.json")
    require(selected == historical["final_columns"] and len(steps) == len(historical["steps"]),
            "Previous strategy diverged from its archived seed trajectory")
    for now, old in zip(steps, historical["steps"], strict=True):
        for key in ("k", "accepted", "removed"):
            require(now[key] == old[key], f"Historical trace changed at {key}")
        near([now["auc"], now["accuracy"]], [old["auc"], old["accuracy"]])


def check_new_search(diag, columns, reference, tolerance):
    rates = np.asarray(diag["tier_rates"])
    require(rates.shape == (3,len(columns)) and len(set(diag["vote_seeds"])) == 3,
            "Missing PI vote replicates or coalition strata")
    near(rates.min(axis=0), diag["min_vote"])
    require(len(diag["nominations"]) == 3 and
            diag["initial_step"] == max(1, int(np.median(diag["nominations"]))), "Incorrect initial PI step")
    probs = np.asarray(diag["association_p"])
    require(probs.shape == (len(columns),) and np.all((0 <= probs) & (probs <= 1)), "Invalid training p-values")
    ranked = np.argsort(probs, kind="stable")
    passing = np.flatnonzero(probs[ranked] <= diag["bh_q"]*np.arange(1,len(columns)+1)/len(columns))
    core = ranked[:int(passing[-1])+1] if len(passing) else ranked[:1]
    require(diag["training_core"] == [columns[j] for j in sorted(core)], "Changed training-only anchor")
    masks = [columns]
    eligible = [(len(columns),0.,columns,0)]
    repair = accepted_repair = 0
    for i, step in enumerate(diag["steps"], start=1):
        parent = step["parent"]
        require(step["id"] == i and 0 <= parent < i, "Invalid proposal lineage")
        previous, current = set(masks[parent]), set(step["proposed"])
        added, removed = current-previous, previous-current
        require(step["added"] == [c for c in columns if c in added] and
                step["removed"] == [c for c in columns if c in removed], "Different edits from proposal parent")
        kind = step["operation"]
        require(kind in ("training_core","prune","refill","exchange"), "Unknown proposal")
        if kind == "training_core":
            require(parent == 0 and step["proposed"] == diag["training_core"], "Invalid anchor proposal")
        elif kind == "refill":
            require(bool(added) and not removed, "Invalid refill")
        elif kind == "exchange":
            require(added and len(added) == len(removed), "Invalid equal-size exchange")
        else:
            require(removed and not added, "Invalid deletion")
        repair += int(kind in ("refill","exchange"))
        accepted_repair += int(kind in ("refill","exchange") and step["accepted"])
        ratio = max((a-b)/(tolerance*a) for a,b in zip(reference,[step["auc"],step["accuracy"]],strict=True))
        near(ratio,step["tightest_relative_budget_used"])
        if step["accepted"]:
            eligible.append((len(current),ratio,step["proposed"],i))
        masks.append(step["proposed"])
    winner = min(eligible)
    require(diag["winner_id"] == winner[3], "Invalid winner risk/tie-break")
    return winner[2], repair, accepted_repair


def inspect(run, protocol, columns, labels, digests, method, tolerance, seed):
    folder = run/method/f"tol-{round(tolerance*100):02}"/f"s{seed}"
    clock = read(folder/"watchdog.json")
    row = dict(method=method, tolerance=tolerance, search_seed=seed, status=clock["status"],
               wall_s=float(clock["elapsed_s"]))
    require(clock["limit_s"] == 900 and row["wall_s"] > 0, "Unexpected watchdog")
    if clock["status"] != "completed":
        require(clock["status"] == "timed_out" and clock["kill_requested_s"] is not None,
                f"{folder}: failed run")
        row["last_reported_fits"] = clock["progress"].get("fits")
        return row, None
    require(clock["kill_requested_s"] is None and clock["exit_code"] == 0, "Invalid completion")
    frozen, result = read(folder/"frozen.json"), read(folder/"result.json")
    selected = frozen["selected"]
    require((frozen["method"], frozen["tolerance"], frozen["search_seed"]) == (method, tolerance, seed)
            and frozen["columns"] == columns and ordered_subset(selected, columns), "Different arm identity or mask")
    require(frozen["model_policy"] == protocol["model_policy"] and frozen["model_policy"]["seed"] == 42 and frozen["model_policy"]["task"] == "binary"
            and frozen["model_policy"]["backend"] == "cuda", "Unexpected validation model")
    require(frozen["partition_sha256"] == digests and frozen["protocol_fingerprint"] == canonical_hash(protocol)
            and result["frozen_hash"] == canonical_hash(frozen) and result == clock["result"],
            "Mask, partition or protocol integrity check failed")
    event, = read(folder/"test_access.json")["events"]
    require(event["dataset_id"] == "madelon" and event["seed"] == 42 and event["decision"] == "allowed"
            and event["job_id"] == result["frozen_hash"] and event["reason"] == "final_removal_evaluation",
            "Test access is not bound to the frozen mask")
    steps = frozen["diagnostics"]["steps"]
    require(len(steps) <= protocol["candidate_budget"] and result["fits"] == len(steps)+3,
            "Incorrect reference, candidate, or final fit count")
    with np.load(folder/"predictions.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["validation_y"], labels["val"])
        np.testing.assert_array_equal(archive["y"], labels["test"])
        validation_ref = metric_pair(labels["val"], archive["validation_full"], "roc_auc")
        test_ref = metric_pair(labels["test"], archive["full"], "roc_auc")
        validation = metric_pair(labels["val"], archive["validation"], "roc_auc")
        test = metric_pair(labels["test"], archive["selected"], "roc_auc")
        for field, scores in (("reference_scores", validation_ref), ("validation_scores", validation),
                              ("test_reference_scores", test_ref), ("test_scores", test)):
            near(result[field], scores)
        near(frozen["reference_scores"], validation_ref)
        near(frozen["validation_scores"], validation)
        require(passes(validation, validation_ref, tolerance), "Final validation mask failed acceptance")
        candidates = archive["candidate_validation"]
        seen = {tuple(columns)}
        feasible = [(columns, validation_ref)]
        require(len(candidates) == len(steps), "Missing candidate predictions")
        for step, prediction in zip(steps, candidates, strict=True):
            mask = step["proposed"]
            key = tuple(mask)
            require(ordered_subset(mask, columns) and step["k"] == len(mask) and key not in seen,
                    "Candidate was duplicated or misidentified")
            seen.add(key)
            score = metric_pair(labels["val"], prediction, "roc_auc")
            near(score, (step["auc"], step["accuracy"]))
            require(bool(step["accepted"]) == passes(score, validation_ref, tolerance),
                    "Dual validation acceptance differs from predictions")
            if step["accepted"]:
                feasible.append((mask, score))
            if mask == selected:
                np.testing.assert_array_equal(prediction, archive["validation"])
        require(len(selected) == min(len(mask) for mask, _ in feasible)
                and any(selected == mask for mask, _ in feasible), "A smaller accepted subset exists")
        if selected == columns:
            np.testing.assert_array_equal(archive["validation"], archive["validation_full"])
            np.testing.assert_array_equal(archive["selected"], archive["full"])
    if method == "old":
        check_history(steps, selected, seed, tolerance)
        repairs, valid_repairs = 0, 0
    else:
        winner, repairs, valid_repairs = check_new_search(frozen["diagnostics"],columns,validation_ref,tolerance)
        require(winner == selected, "Final mask differs from winning proposal")
    require(result["validation_pass"] and result["test_pass"] == passes(test,test_ref,tolerance)
            and result["k"] == len(selected) and result["candidates"] == len(steps), "Incorrect headline record")
    row.update(retained=len(selected), candidate_fits=len(steps), fits=result["fits"],
               repair_proposals=repairs, accepted_repairs=valid_repairs,
               validation_auc=validation[0], validation_accuracy=validation[1],
               validation_ref_auc=validation_ref[0], validation_ref_accuracy=validation_ref[1],
               validation_auc_floor=validation_ref[0]*(1-tolerance),
               validation_accuracy_floor=validation_ref[1]*(1-tolerance),
               test_auc=test[0], test_accuracy=test[1], test_ref_auc=test_ref[0], test_ref_accuracy=test_ref[1],
               test_auc_floor=test_ref[0]*(1-tolerance), test_accuracy_floor=test_ref[1]*(1-tolerance),
               validation_pass=True, test_auc_pass=test[0] >= test_ref[0]*(1-tolerance)-1e-12,
               test_accuracy_pass=test[1] >= test_ref[1]*(1-tolerance)-1e-12,
               test_pass=passes(test,test_ref,tolerance))
    return row, set(selected)


def summarize(frame, masks):
    summaries, overlaps = [], []
    for (method,tol), group in frame.groupby(["method","tolerance"], sort=False):
        group = group[group.status == "completed"]
        scores = []
        for a,b in combinations(group.search_seed, 2):
            left, right = masks[method,tol,a], masks[method,tol,b]
            scores.append(jaccard(left,right))
            overlaps.append(dict(comparison="between_search_seeds", method_a=method, method_b=method,
                                 tolerance=tol, seed_a=a, seed_b=b, intersection=len(left & right),
                                 union=len(left | right), jaccard=scores[-1]))
        summaries.append(dict(method=method, tolerance=tol, completed=len(group),
                              min_k=group.retained.min() if len(group) else None,
                              median_k=group.retained.median() if len(group) else None,
                              max_k=group.retained.max() if len(group) else None,
                              mean_jaccard=np.mean(scores) if scores else None, overlap_pairs=len(scores),
                              mean_fits=group.fits.mean() if len(group) else None,
                              mean_wall_s=group.wall_s.mean() if len(group) else None,
                              test_passes=int(group.test_pass.sum()) if len(group) else 0))
    for tol, seed in product(TOLS,SEEDS):
        left, right = masks["old",tol,seed], masks["piv3",tol,seed]
        if left is not None and right is not None:
            overlaps.append(dict(comparison="paired_methods", method_a="old", method_b="piv3",
                                 tolerance=tol, seed_a=seed, seed_b=seed, intersection=len(left & right),
                                 union=len(left | right), jaccard=jaccard(left,right)))
    return pd.DataFrame(summaries), pd.DataFrame(overlaps)


def render(frame, summary, overlaps):
    ref = frame[frame.status=="completed"]
    lines = ["# Madelon 原始 500 列的自适应裁剪对照", "",
             "数据和 XGBoost 均固定 seed 42。11、23、47 仅控制投票与搜索。每组独立冷启动，限时 900 秒、"
             "最多考察 12 个不同候选。两个指标分别相对该分区的完整模型计算损失容差，须同时达标。"]
    if len(ref):
        row = ref.iloc[0]
        lines += ["", f"完整模型的验证 AUC 为 {row.validation_ref_auc:.5f}、准确率为 {row.validation_ref_accuracy:.2%}；"
                  f"测试 AUC 为 {row.test_ref_auc:.5f}、准确率为 {row.test_ref_accuracy:.2%}。"
                  "候选只在验证集上筛选。表中测试是否达标是冻结后的独立记分，不参与重选。"]
    lines += ["", "| 相对容差 | 方法 | 搜索种子 | 最终列数 | 总拟合数 | 总秒数 | 验证 AUC | 验证准确率 | 测试 AUC | 测试准确率 | 测试双达标 |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in frame.itertuples():
        prefix = f"| {r.tolerance:.0%} | {METHODS[r.method]} | {r.search_seed} |"
        if r.status != "completed":
            lines.append(prefix+f" — | — | {r.wall_s:.2f} | — | — | — | — | 超时 |")
        else:
            lines.append(prefix+f" {r.retained:.0f} | {r.fits:.0f} | {r.wall_s:.2f} | {r.validation_auc:.5f} | "
                         f"{r.validation_accuracy:.2%} | {r.test_auc:.5f} | {r.test_accuracy:.2%} | "
                         f"{'是' if r.test_pass else '否'} |")
    lines += ["", "## 跨搜索种子的稳定性", "",
              "Jaccard 是两次最终列集合的交集大小除以并集大小。均值对每种方法、每个容差下已完成的种子两两等权计算。",
              "", "| 相对容差 | 方法 | 完成数 | 列数 最小／中位／最大 | 平均 Jaccard | 平均拟合数 | 平均秒数 | 测试双达标 |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        sizes = f"{r.min_k:.0f}／{r.median_k:g}／{r.max_k:.0f}" if r.completed else "—"
        def fmt(x):
            return f"{x:.3f}" if pd.notna(x) else "—"
        lines.append(f"| {r.tolerance:.0%} | {METHODS[r.method]} | {r.completed}/3 | {sizes} | {fmt(r.mean_jaccard)} | "
                     f"{fmt(r.mean_fits)} | {fmt(r.mean_wall_s)} | {r.test_passes}/{r.completed} |")
    lines += ["", "总拟合次数包括完整模型验证参考、各候选验证拟合及两个最终重训（入选子集与完整模型）。"
              "总耗时计入冷启动、投票、搜索、验证及最终测试。三次搜索共用一个数据划分，重合度反映算法随机性；"
              "接近保留全部 500 列时，重合度容易随保留比例升高。", "",
              "[逐次运行与双指标门槛](per_arm.csv) · [稳定性](stability.csv) · "
              "[每对集合的交并及同种子两方法对照](overlap.csv) · [审核记录](audit.json) · [运行协议](protocol.json)"]
    return "\n".join(lines)+"\n"


def main(run):
    protocol = read(run/"protocol.json")
    require(protocol["data_seed"] == protocol["predictor_seed"] == 42 and protocol["arm_limit_s"] == 900
            and protocol["candidate_budget"] == 12 and tuple(protocol["search_seeds"]) == SEEDS
            and set(protocol["tolerances"]) == set(TOLS), "Unexpected seed/budget/tolerance")
    expected = set(product(METHODS,TOLS,SEEDS))
    require(len(protocol["tasks"]) == len(expected) and set(map(tuple,protocol["tasks"])) == expected,
            "Missing or duplicate arms in frozen protocol")
    require(protocol["old_probe_sha256"] == sha(OLD/"probe.py"), "Historical baseline probe changed")
    for source, digest in protocol["source_sha256"].items():
        require(sha(run/"source"/source) == digest, "Changed source snapshot "+source)
    columns, _, labels, digests, initial = verify_inputs(run, protocol)
    records, masks = [], {}
    for tol, sd, method in product(TOLS, SEEDS, METHODS):
        row, mask = inspect(run, protocol, columns, labels, digests, method, tol, sd)
        records.append(row)
        masks[method,tol,sd] = mask
    frame = pd.DataFrame(records)
    completed = frame[frame.status=="completed"]
    for column in ("test_ref_auc","test_ref_accuracy","validation_ref_auc","validation_ref_accuracy"):
        if len(completed):
            near(completed[column].to_numpy(), np.repeat(completed[column].iloc[0],len(completed)))
    summary, overlaps = summarize(frame,masks)
    frame.to_csv(run/"per_arm.csv",index=False)
    summary.to_csv(run/"stability.csv",index=False)
    overlaps.to_csv(run/"overlap.csv",index=False)
    (run/"comparison.md").write_text(render(frame,summary,overlaps),encoding="utf-8")
    evidence = dict(status="pass", completed=len(completed), arms=len(frame),
                    old_archived_traces_verified=int((completed.method=="old").sum()),
                    candidate_predictions_verified=int(completed.candidate_fits.sum()) if len(completed) else 0,
                    initial_feature_overlap_check=initial, protocol_fingerprint=canonical_hash(protocol),
                    artifacts={name:sha(run/name) for name in ("comparison.md","per_arm.csv","stability.csv","overlap.csv")})
    atomic_write_json(run/"audit.json",evidence)
    print(run, evidence)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run)
