"""Audit two nonoverlapping PIv2 seed-42 campaigns and produce their combined comparison."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from math import ceil, sqrt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_piv0_benchmark import read, sha
from experiments.test.audit_piv2_prefilter import (
    MEASURES, check_selector, dataset_inputs, inspect_arm, require, validate_votes,
)
from experiments.test.report_piv2_prefilter import DATASETS, METHODS as ORIGINAL, load_rows
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


NEW = dict(spearman="Spearman 秩相关", mrmr="mRMR（MID）", chi2="分箱卡方",
           variance="方差 Top-k", random_forest="随机森林", extra_trees="极端随机树",
           rfe_linear="线性 RFE", elastic_net="弹性网")
FILTERS = {"spearman", "mrmr", "chi2", "variance"}
COALITIONS = {"exact", "mc32"}
METHODS = {**ORIGINAL, **NEW}
TIE = 1e-14


def check_new(backend, result, frozen):
    diag, names, selected = frozen["selector"], frozen["candidates"], frozen["selected"]
    if backend in COALITIONS:
        check_selector(backend, result, frozen)
        return
    require(backend in NEW, "Unexpected new method")
    k = min(len(names), ceil(sqrt(result["width"])))
    if backend != "elastic_net":
        require(len(selected) == k, f"{backend}: invalid fixed k")
    if backend in FILTERS:
        require(diag["selected"] == selected and len(diag["choice_order"]) == k
                and set(diag["choice_order"]) == set(selected), f"{backend}: filter order")
        scores = np.asarray(diag["ranking_scores"])
        require(scores.shape == (len(names),) and np.isfinite(scores).all(), f"{backend}: scores")
        require(diag["k"] == k and diag["row_limit"] == 4096 and diag["ranking_sample_rows"] <= 4096,
                f"{backend}: rows or width")
        if backend == "mrmr":
            require(len(diag["mid_by_step"]) == k and np.isfinite(diag["mid_by_step"]).all(), "mRMR choices")
            require(diag["choice_order"][0] == names[int(scores.argmax())], "mRMR first choice")
        else:
            require(diag["choice_order"] == [names[i] for i in np.argsort(-scores, kind="stable")[:k]],
                    f"{backend}: scores and chosen names disagree")
        if backend in {"chi2", "mrmr"}:
            require(diag["max_bins"] == 8 and diag["target_binned"] == (result["metric"] == "r2"),
                    f"{backend}: discretization")
    else:
        require(diag["selected"] == selected and diag["row_limit"] == 4096 and diag["ranking_sample_rows"] <= 4096,
                f"{backend}: model support or capacity")
        if backend == "rfe_linear":
            step = max(1, len(names)//5)
            require(diag["k"] == k and diag["elimination_step"] == step and
                    selected == [c for c,r in zip(names, diag["rankings"], strict=True) if r == 1] and
                    diag["selector_fits"] == ceil((len(names)-k)/step)+1, "RFE step or support")
        else:
            scores = np.asarray(diag["scores"])
            require(scores.shape == (len(names),) and np.isfinite(scores).all() and diag["selector_fits"] == 1,
                    f"{backend}: model scores or fit count")
            if backend == "elastic_net":
                cols = [col for i,col in enumerate(names) if scores[i] > 0]
                require(diag["fallback_top1"] == (not cols) and selected == (cols or [names[int(scores.argmax())]]),
                        "ElasticNet coefficient support")
                require(0 < diag["iterations"] <= diag["iteration_limit"] == 2500 and
                        diag["hit_iteration_limit"] == (diag["iterations"] == 2500), "ElasticNet convergence record")
            else:
                indices = np.sort(np.argsort(-scores, kind="stable")[:k])
                require(diag["trees"] == 100 and diag["k"] == k and selected == [names[i] for i in indices],
                        "Random tree support or tree count")
    require(result["fits"] == diag["selector_fits"] + 2, f"{backend}: fitted model count")


def planned_coalitions(backend, columns, kept, seed):
    if backend == "exact":
        return 2**len(kept) - 1
    bits = {c:1<<i for i,c in enumerate(kept)}
    masks = set()
    rng = np.random.default_rng(seed)
    for _ in range(32):
        mask = 0
        for j in rng.permutation(len(columns)):
            name = columns[j]
            if name in bits:
                mask |= bits[name]
                masks.add(mask)
    return len(masks)


def coalition_evidence(folder, key, inputs, fingerprint, supports, timer):
    dataset, backend, arm = key
    state = timer["progress"]
    path = folder/"backend_input.json"
    if not path.exists() and timer["status"] != "completed" and state.get("phase") in ("loading","prefilter",None):
        return {}
    entry = read(path)
    cols = json.loads(inputs[dataset][0]["selected_features_json"])
    kept = entry["candidates"]
    require(entry["protocol_fingerprint"] == fingerprint, "Coalition source protocol")
    if arm == "plain":
        require(entry["prefilter"] is None and kept == cols, "Wrong full-space coalition columns")
    else:
        validate_votes(entry["prefilter"], cols, kept)
        require(canonical_hash(entry["prefilter"]) == supports[dataset], "Changed coalition prefilter")
    planned = planned_coalitions(backend, cols, kept, 42)
    if timer["status"] == "completed":
        data = read(folder/"frozen.json")
        diag = data["selector"]
        require(data["candidates"] == kept and diag["coalition_count"] == planned, "Incorrect full coalition schedule")
        checked = int(diag["unique_scored"])
        require(checked in {planned, planned + int(backend == "mc32")}, "Invalid distinct mask count")
        return dict(checked=str(checked), target=str(checked), paths_completed=32 if backend == "mc32" else None)
    if state.get("phase") == backend:
        checked, total = state["checked"], state["candidates"]
        require(type(checked) == type(total) == int and 0 <= checked <= total, "Invalid coalition progress")
        require(total in {planned, planned+int(backend=="mc32")}, "Incorrect coalition target")
        if last := state.get("last_mask"):
            require(1 <= int(last,16) < 2**len(kept), "Out-of-bounds saved coalition mask")
            if backend == "exact":
                require(int(last,16) == checked, "Enumeration count does not match final visited mask")
        if backend == "mc32":
            require(state["paths_planned"] == 32 and 0 <= state["paths_completed"] <= 32, "Invalid path progress")
        return dict(checked=str(checked), target=str(total), paths_completed=state.get("paths_completed"))
    return dict(target=str(planned))


def scope_key(item):
    return tuple(item[k] for k in ("dataset", "backend", "arm"))


def open_arm(folder, key, scope, partial, inputs, fingerprint, supports):
    state = read(folder/"progress.json") if (folder/"progress.json").exists() else {}
    status = "running" if state else "pending"
    cutoffs = [scope["interrupted"]]
    not_started = set()
    if "exact_request" in scope:
        cutoffs.append(scope["exact_request"]["interrupted"])
        not_started = {scope_key(item) for item in scope["exact_request"]["not_started"]}
    interruptions = {scope_key(item):item for item in cutoffs}
    outcome = dict.fromkeys(MEASURES)
    outcome.update(wall_s=None, checked="", target="", paths_completed=None, phase=state.get("phase","pending"))
    if key in interruptions:
        interrupted = interruptions[key]
        status = "user_skipped"
        require(not (folder/"watchdog.json").exists() and not (folder/"result.json").exists()
                and not (folder/"test_access.json").exists(), "Interrupted arm has a terminal result or test access")
        for k,alias in (("checked","checked"),("candidates","target"),("paths_completed","paths_completed"),
                        ("last_mask","last_mask")):
            if alias == "paths_completed" and key[1] != "mc32":
                continue
            require(state[k] == interrupted[alias], "Interrupted arm differs from scope note")
        seconds = float(interrupted["last_observed_wall_s"])
        log = (folder.parents[2]/"run.log").read_text(encoding="utf-8")
        require(0 < seconds < 900 and any("/".join(key) in line and f"elapsed={seconds:.1f}s" in line
                                         for line in log.splitlines()), "Unverified last heartbeat")
        outcome["last_observed_wall_s"] = seconds
    elif key in not_started:
        require(not folder.exists(), f"Marked unstarted but run artifacts exist: {folder}")
        status = "user_skipped_before_start"
        outcome["phase"] = "not_started"
    else:
        require(partial and key[1] == "exact", f"Unexpected missing terminal result: {key}")
    outcome["status"] = status
    if status not in {"pending","user_skipped_before_start"}:
        outcome.update(coalition_evidence(folder, key, inputs, fingerprint, supports, dict(status=status, progress=state)))
        for name in ("prefilter_retained", "prefilter_s"):
            if name in state:
                outcome[name] = state[name]
    return outcome


def audited_pairs(old, new, partial):
    old_arms, old_protocol = load_rows(old)
    old_audit = read(old/"audit.json")
    protocol = read(new/"protocol.json")
    for name in ("seed", "arm_limit_s", "catalog_sha256", "model_params", "hardware_backend", "cold_start"):
        require(old_protocol[name] == protocol[name], f"Different {name} between runs")
    for run, config in ((old, old_protocol), (new, protocol)):
        for path, digest in config["source_sha256"].items():
            require(sha(run/"source"/path) == digest, f"Source snapshot mismatch: {run.name}/{path}")
    shared = old_protocol["source_sha256"].keys() & protocol["source_sha256"].keys()
    require({"src/minppi_mofs/piv2.py", "src/minppi_mofs/modeling.py", "src/minppi_mofs/utility.py"} <= shared,
            "Missing common prefilter or model source")
    for path in shared - {"experiments/test/run_piv2_prefilter.py"}:
        require(protocol["source_sha256"][path] == old_protocol["source_sha256"][path], f"Different {path}")

    old_pairs = pd.read_csv(old/"paired.csv")
    old_keys = {(d, b) for d, b, _ in old_arms}
    require(len(old_pairs) == len(old_keys) and set(zip(old_pairs.dataset, old_pairs.backend)) == old_keys,
            "Old pair identities differ from audited source arms")
    old_pairs["source_run"], old_pairs["batch"] = old.name, "original"
    for arm in ("plain","piv2"):
        for key in COALITIONS:
            folder = old/"abalone"/key/arm
            diag = read(folder/"frozen.json")["selector"]
            idx = (old_pairs.dataset=="abalone") & (old_pairs.backend==key)
            old_pairs.loc[idx, arm+"_phase"] = "complete"
            old_pairs.loc[idx, arm+"_checked"] = old_pairs.loc[idx, arm+"_target"] = str(diag["coalition_count"])
            if key == "mc32":
                old_pairs.loc[idx, arm+"_paths_completed"] = diag["paths"]
    jobs, catalog = audit_prepared_catalog()
    require(catalog["sha256"] == protocol["catalog_sha256"], "Data catalog changed")
    inputs = dataset_inputs(jobs)
    fingerprint = canonical_hash(protocol)
    expected = {(d, b, a) for d in DATASETS for b in {*NEW, *COALITIONS} if d != "abalone" or b in NEW
                for a in ("plain", "piv2")}
    require(len(protocol["tasks"]) == len(expected) and set(map(tuple, protocol["tasks"])) == expected,
            "Incomplete new run protocol")
    with (new/"summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summary = list(csv.DictReader(stream))
    indexed = {(r["dataset"], r["backend"], r["arm"]): r for r in summary}
    scope = read(new/"scope_change.json")
    require(scope["excluded_from_main"] == ["mc32"] and scope["reason"] == "user_requested_skip_too_slow",
            "Unexpected change of scope")
    cut = tuple(scope["interrupted"][k] for k in ("dataset", "backend", "arm"))
    require(cut == ("isolet", "mc32", "plain") and cut not in indexed, "Unexpected interrupted arm")
    cancelled = {cut}
    if "exact_request" in scope:
        request = scope["exact_request"]
        require(request["reason"] == "user_requested_skip_remaining_combinatorial_cost", "Exact decision reason")
        stopping = scope_key(request["interrupted"])
        unstarted = {scope_key(item) for item in request["not_started"]}
        require(stopping == ("spambase", "exact", "plain") and len(unstarted)==len(request["not_started"])==7
                and all(key[1]=="exact" for key in unstarted), "Exact decision scope")
        cancelled.update({stopping, *unstarted})
        require(cancelled <= expected and cancelled.isdisjoint(indexed), "Decision overlaps terminal records")
    missing = expected-set(indexed)-cancelled
    require(set(indexed) <= expected and len(indexed) == len(summary)
            and all(k[1] == "exact" for k in missing) and (partial or not missing), "Missing terminal records")
    counts = Counter()
    require(set(old_audit["prefilter_sha256"]) == set(DATASETS), "Missing old prefilter fingerprints")
    supports = old_audit["prefilter_sha256"].copy()
    pairs = []
    converged_warnings = []
    paired_fit_rows = {}
    for dataset in DATASETS:
        for backend in (*NEW,"exact","mc32"):
            if (dataset, backend, "plain") not in expected:
                continue
            pair = dict(dataset=dataset, backend=backend, seed=protocol["seed"], batch="extension", source_run=new.name)
            for arm in ("plain", "piv2"):
                key = dataset, backend, arm
                location = new / dataset / backend / arm
                pair.update(width=int(inputs[dataset][0]["selected_feature_count"]),
                            metric=inputs[dataset][0]["utility"],
                            secondary="mae" if inputs[dataset][0]["task"] == "regression" else "accuracy")
                if key not in indexed:
                    outcome = open_arm(location,key,scope,partial,inputs,fingerprint,supports)
                    counts[outcome["status"]] += 1
                    pair.update({arm+"_"+k:v for k, v in outcome.items()})
                    continue
                row = indexed[key]
                outcome = inspect_arm(new, key, row, protocol, fingerprint, inputs, supports, counts, None,
                                      selector_check=check_new)
                timer = read(location/"watchdog.json")
                state = timer["progress"]
                outcome["phase"] = state.get("phase", "starting")
                outcome.update(checked="", target="", paths_completed=None)
                if backend in COALITIONS:
                    outcome.update(coalition_evidence(location, key, inputs, fingerprint, supports, timer))
                if outcome["status"] == "completed":
                    diag = read(location/"frozen.json")["selector"]
                    if backend not in COALITIONS:
                        require(diag["ranking_sample_rows"] == min(4096, int(inputs[dataset][0]["train_row_count"])),
                                f"{key}: ranking sample size")
                        if backend not in FILTERS:
                            stamp = diag["sample_rows_sha256"]
                            require(paired_fit_rows.setdefault(dataset, stamp) == stamp, f"{key}: tree/linear rows")
                        if backend == "elastic_net" and diag["hit_iteration_limit"]:
                            converged_warnings.append(list(key))
                elif outcome["status"] == "timed_out":
                    require(timer["kill_requested_s"] is not None and timer["kill_requested_s"] >= protocol["arm_limit_s"],
                            f"{key}: missing hard-deadline event")
                    for k in ("prefilter_retained", "prefilter_s"):
                        if k in state:
                            outcome[k] = state[k]
                    require(not (location/"test_access.json").exists() or outcome["phase"] in {"final_model","complete"},
                            f"{key}: unexpected test access before selection")
                pair.update(width=int(row["width"]), metric=row["metric"], secondary=row["secondary_metric"])
                pair.update({arm+"_"+k:v for k, v in outcome.items()})
            pair["complete"] = pair["plain_status"] == pair["piv2_status"] == "completed"
            if pair["complete"]:
                pair.update(seconds_saved=pair["plain_wall_s"] - pair["piv2_wall_s"],
                            speedup=pair["plain_wall_s"]/pair["piv2_wall_s"],
                            test_delta=pair["piv2_test_utility"] - pair["plain_test_utility"])
            pairs.append(pair)
    combined = pd.concat([old_pairs, pd.DataFrame(pairs)], ignore_index=True)
    require(len(combined) == 102 and not combined.duplicated(["dataset","backend"]).any(), "Duplicated paired keys")
    combined["time_saved_pct"] = 100 * combined.seconds_saved / combined.plain_wall_s
    combined["faster_and_not_lower"] = ((combined.seconds_saved > 0) & (combined.test_delta >= -TIE)).astype("boolean")
    combined.loc[~combined.complete, "faster_and_not_lower"] = pd.NA
    report = dict(status="pass", snapshot=partial, snapshot_utc=datetime.now(timezone.utc).isoformat(),
                  scope_change_file_sha256=sha(new/"scope_change.json"), excluded_from_main=scope["excluded_from_main"],
                  seed=42, original=old.name, extension=new.name,
                  old_audit_sha256=sha(old/"audit.json"), old_protocol_hash=canonical_hash(old_protocol),
                  extension_protocol_hash=fingerprint, extension_summary_sha256=sha(new/"summary.csv"),
                  extension_terminal_status_counts=dict(counts), pairs=len(combined), completed_pairs=int(combined.complete.sum()),
                  original_arms=len(old_arms), extension_arms=len(expected), partition_files_checked=3*len(inputs),
                  matching_prefilter_sha256=supports, elastic_net_iteration_limits=converged_warnings)
    return combined, report


def backend_summary(pairs):
    rows = []
    for method, group in pairs.groupby("backend", sort=False):
        data = group[group.complete]
        plain, pi = data.plain_wall_s.sum(), data.piv2_wall_s.sum()
        rows.append(dict(backend=method, name=METHODS[method], batch="+".join(sorted(set(group.batch))),
                         scheduled=len(group), completed_pairs=len(data), incomplete=len(group)-len(data),
                         plain_completed=int((group.plain_status == "completed").sum()),
                         piv2_completed=int((group.piv2_status == "completed").sum()),
                         plain_timeouts=int((group.plain_status == "timed_out").sum()),
                         piv2_timeouts=int((group.piv2_status == "timed_out").sum()),
                         plain_total_observed_wall_s=group.plain_wall_s.sum(),
                         piv2_total_observed_wall_s=group.piv2_wall_s.sum(),
                         plain_wall_s=plain, piv2_wall_s=pi, prefilter_s=data.piv2_prefilter_s.sum(),
                         speedup=plain/pi if pi else None, time_saved_pct=100*(1-pi/plain) if plain else None,
                         piv2_faster=int((data.seconds_saved > 0).sum()),
                         higher_test=int((data.test_delta > TIE).sum()), tied_test=int((data.test_delta.abs() <= TIE).sum()),
                         lower_test=int((data.test_delta < -TIE).sum()), faster_and_not_lower=int(data.faster_and_not_lower.sum()),
                         plain_fits=data.plain_fits.sum(), piv2_fits=data.piv2_fits.sum(),
                         plain_mean_retained=data.plain_retained.mean(), piv2_mean_retained=data.piv2_retained.mean()))
    return pd.DataFrame(rows).set_index("backend").loc[[m for m in METHODS if m in set(pairs.backend)]].reset_index()


def link(path, out):
    return Path(os.path.relpath(path, out)).as_posix()


def fmt(val, pattern):
    return format(val, pattern) if pd.notna(val) else "—"


def arrow(row, name, pattern):
    return fmt(row["plain_"+name], pattern) + " → " + fmt(row["piv2_"+name], pattern)


def signed_percent(value):
    return f"{value:+.1f}%" if pd.notna(value) else "—"


STATUS = dict(completed="已完成", user_skipped="用户中止（耗时过长）", user_skipped_before_start="用户跳过（未启动）", timed_out="15 分钟超时",
              running="进行中", pending="待运行", error="运行错误")


def render(pairs, groups, deferred, report, old, new, out):
    valid = pairs[pairs.complete]
    plain, pi = valid.plain_wall_s.sum(), valid.piv2_wall_s.sum()
    lines = ["# PIv2 通用前置筛选的合并测试（seed 42）", "",
             f"原计划包括 17 种后端、102 组配对。按最新要求，耗时过长的 32 排列重训式 Shapley 单列附录。"
             f"主表覆盖 6 个预处理数据集和 16 种后端，共 {len(pairs)} 组配对，已完成 {len(valid)} 组。"
             f"审计时间为 {report['snapshot_utc']}。"
             "每个配对的两端使用相同的数据、后端规则和预测器，各独立冷启动一次，单次限时 900 秒。"
             "按已完成配对加总，原方法与加 PIv2 的端到端耗时分别为 "
             f"{plain:.2f} 和 {pi:.2f} 秒。其中 {int((valid.seconds_saved > 0).sum())} 组实测更快，"
             f"{int(valid.faster_and_not_lower.sum())} 组同时更快且测试主分数持平或上升。"
             "时间总和按实际秒数计，耗时较长的后端占比较大。", "",
             "## 按方法汇总", "",
             "节时率为（原法秒数 − 加 PIv2 秒数）÷ 原法秒数。胜、平、负按同一数据集上的测试主分数计算，平局容差为 1e−14。"
             "每列时间和胜负计数仅纳入两端均完成的配对。单种子和单次计时不能估计结果波动。", "",
             "| 方法 | 完成／计划 | 总耗时秒 原法→+PI | 节时率 | 更快组数 | 主分数 胜／平／负 | 更快且主分数不降 | 最终列数均值 原法→+PI |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in groups.itertuples():
        lines.append(f"| {r.name} | {r.completed_pairs}/{r.scheduled} | {r.plain_wall_s:.2f} → {r.piv2_wall_s:.2f} | "
                     f"{signed_percent(r.time_saved_pct)} | {r.piv2_faster} | {r.higher_test}/{r.tied_test}/{r.lower_test} | "
                     f"{r.faster_and_not_lower} | {fmt(r.plain_mean_retained,'.1f')} → {fmt(r.piv2_mean_retained,'.1f')} |")
    lines += ["", "## 逐数据集的全部配对", "",
              "箭头左侧为原法，右侧为加入 PIv2 的同一后端。总时长包含进程启动、加载、前置、选列、最终模型重训、测试预测和落盘。"
              "主分数为测试 ROC-AUC（ISOLET 为宏平均 OvR）或 Abalone 的非负截断 R²。"
              "副指标为分类准确率或 Abalone 的平均绝对误差（圈，越低越好）。", "",
              "| 数据集 | 后端 | 输入→前置列数 | 前置秒 | 总耗时秒 原法→+PI | 节时率 | 最终列数 原法→+PI | 测试 AUC／R² 原法→+PI | 准确率／MAE 原法→+PI |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, display in DATASETS.items():
        group = pairs[pairs.dataset == dataset].set_index("backend")
        for method, name in METHODS.items():
            if method not in group.index:
                continue
            row = group.loc[method]
            status = "" if row.complete else f"（{STATUS[row.plain_status]} → {STATUS[row.piv2_status]}）"
            lines.append(f"| {display} | {name}{status} | {row.width} → {fmt(row.piv2_prefilter_retained,'.0f')} | "
                         f"{fmt(row.piv2_prefilter_s,'.3f')} | {arrow(row,'wall_s','.2f')} | "
                         f"{signed_percent(row.time_saved_pct)} | {arrow(row,'retained','.0f')} | "
                         f"{arrow(row,'test_utility','.5f')} | "
                         f"{arrow(row,'test_secondary','.4f' if dataset=='abalone' else '.3%')} |")
    if len(valid) < len(pairs):
        lines += ["", "## 未完整配对的运行进度", "",
                  "两端只要有一端未完成，配对涨跌和节时率即留空。上表仍列各自实测用时以及已完成一端的测试结果。"
                  "下表保留最后一次成功写盘的计数，计划规模按完整整数记录。"
                  "枚举的进度单位为不同非空联盟；Shapley 的单位为计划缓存的不同联盟，必要时额外拟合一次获胜方案。", "",
                  "| 数据集 | 方法 | 运行 | 状态／阶段 | 已验／计划数 | 完成路径数 | 实测秒数 |",
                  "|---|---|---|---|---:|---:|---:|"]
        for _, row in pairs[~pairs.complete].iterrows():
            for arm, label in (("plain","原法"),("piv2","+ PIv2")):
                seconds = fmt(row[arm+'_wall_s'],'.2f')
                if row[arm+"_status"]=="user_skipped":
                    seconds = "末次观测 ≥"+fmt(row[arm+"_last_observed_wall_s"],".1f")
                lines.append(f"| {DATASETS[row.dataset]} | {METHODS[row.backend]} | {label} | "
                             f"{STATUS[row[arm+'_status']]} / {row[arm+'_phase']} | {row[arm+'_checked'] or '—'} / "
                             f"{row[arm+'_target'] or '—'} | {fmt(row[arm+'_paths_completed'],'.0f')} | "
                             f"{seconds} |")
    lines += ["", "## 附录：重训式 Shapley（耗时过长，已从主统计中移出）", "",
              "首轮和扩展轮已经完成的结果照录。ISOLET 原法显示中断前最后一次心跳耗时，"
              "当时最近落盘的特征选择进度单独保留。ISOLET 加 PIv2 的 900 秒 watchdog 已触发。", "",
              "| 数据集 | 运行 | 状态 | 耗时秒 | 已验／计划数 | 路径 | 最终列数 | 测试 AUC／R² | 准确率／MAE |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in deferred.iterrows():
        for arm, label in (("plain","原法"),("piv2","+ PIv2")):
            value = fmt(row[arm+"_wall_s"], ".2f")
            if row[arm+"_status"] == "user_skipped":
                value = "末次观测 ≥"+fmt(row[arm+"_last_observed_wall_s"],".1f")
            lines.append(f"| {DATASETS[row.dataset]} | {label} | {STATUS[row[arm+'_status']]} | {value} | "
                         f"{row[arm+'_checked'] or '—'}／{row[arm+'_target'] or '—'} | "
                         f"{fmt(row[arm+'_paths_completed'],'.0f')}/32 | {fmt(row[arm+'_retained'],'.0f')} | "
                         f"{fmt(row[arm+'_test_utility'],'.5f')} | "
                         f"{fmt(row[arm+'_test_secondary'],'.4f' if row.dataset=='abalone' else '.3%')} |")
    lines += ["", "## 比较口径与数据来源", "",
              "PIv2 仅使用训练分区。两折训练代理在小、中、大联盟中投票并取最低净票率。"
              "同一数据集的所有后端收到一致的 PIv2 候选列。本轮继续使用相同的 CUDA XGBoost 预测器、固定的预处理和评估指标。"
              "两轮使用过同一 seed 42 测试划分，比较用于探索。", "",
              "原计划的 17 种方法各安排 6 个数据集。两种真实重训法在首轮测了 Abalone，其他五集属于扩展批次。"
              "两端各自受 900 秒限时约束。固定列数的方法均以原始宽度确定 ⌈√d⌉，"
              "两臂目标列数一致。Gain 扫描全部排序前缀，PFI 留下平均置换掉分为正的列。", "",
              "扩展的四种过滤法最多使用 4,096 条训练行。分类按类别分层。"
              "Spearman 在多分类上按类别频率加权汇总绝对秩相关；卡方使用分箱列联表，回归目标也分箱；"
              "mRMR 依据信息相关性减去对已选列的平均冗余逐步选列；方差 Top-k 在既有预处理量纲上排序，结果受列尺度影响。"
              "随机森林、极端随机树依训练所得特征重要性排序；RFE 递归剔除线性模型中的弱列；"
              "弹性网依正则化线性模型的非零系数选列。验证分数、验证副指标与分阶段时间见配对 CSV。", "",
              f"[主表配对 CSV](paired.csv) · [方法汇总 CSV](by_backend.csv) · [Shapley 附录 CSV](shapley_deferred.csv) · [合并审计](audit.json) · "
              f"[首轮审计]({link(old/'audit.json',out)}) · [扩展协议]({link(new/'protocol.json',out)}) · "
              f"[扩展运行日志]({link(new/'run.log',out)})"]
    return "\n".join(lines) + "\n"


def main(old, new, out, partial=False):
    old, new, out = (p.resolve() for p in (old, new, out))
    require(not any(out == r or out.is_relative_to(r) or r.is_relative_to(out) for r in (old,new)),
            "Place combined results in a separate output directory")
    planned, report = audited_pairs(old, new, partial)
    deferred = planned[planned.backend == "mc32"]
    pairs = planned[planned.backend != "mc32"]
    require(len(pairs)==96 and len(deferred)==6 and len(set(pairs.backend))==16, "Unexpected scope sizes")
    report.update(main_pairs=len(pairs), main_completed_pairs=int(pairs.complete.sum()),
                  deferred_shapley_pairs=len(deferred), deferred_shapley_completed_pairs=int(deferred.complete.sum()))
    groups = backend_summary(pairs)
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "The combined output directory must be empty")
    pairs.to_csv(out/"paired.csv", index=False)
    deferred.to_csv(out/"shapley_deferred.csv", index=False)
    groups.to_csv(out/"by_backend.csv", index=False)
    (out/"comparison.md").write_text(render(pairs, groups, deferred, report, old, new, out), encoding="utf-8")
    report["artifacts"] = {name:sha(out/name) for name in ("paired.csv","by_backend.csv","comparison.md","shapley_deferred.csv")}
    atomic_write_json(out/"audit.json", report)
    print(f"{out} | core {report['main_completed_pairs']}/{len(pairs)} complete, Shapley {int(deferred.complete.sum())}/{len(deferred)} complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("original", "extension", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--partial", action="store_true", help="Include ongoing or scheduled exact-enumeration arms")
    args = parser.parse_args()
    main(args.original, args.extension, args.output, args.partial)
