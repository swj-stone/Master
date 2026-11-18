"""Audit the complete head-to-head Madelon-500 validation-selected prefix scans."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_piv0_benchmark import read, sha
from experiments.run_thesis_sampling import ROOT
from experiments.test.audit_piv2_prefilter import inspect_arm, near, require
from experiments.test.report_madelon500_piv2 import ARMS, SELECTORS, verify_inputs
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


FIXED = ROOT / "results/test/thesis_sampling/PIv2-madelon500-seed42"
METHODS = {k:v for k,v in SELECTORS.items() if k not in {"gain", "pfi", "elastic_net", "l1"}}
TOL = 1e-14


def verify_protocol(run):
    protocol = read(run/"protocol.json")
    old = read(FIXED/"protocol.json")
    audit = read(FIXED/"audit.json")
    require(audit["status"] == "pass" and canonical_hash(old) == audit["protocol_sha256"]
            and sha(FIXED/"paired.csv") == audit["artifacts"]["paired.csv"], "Changed fixed-k baseline")
    for name in ("dataset", "width", "seed", "cold_start", "arm_limit_s", "hardware_backend",
                 "model_params", "catalog_sha256", "input_sha256", "input_partition_sha256"):
        require(protocol[name] == old[name], f"Different test configuration: {name}")
    require(protocol["dataset"] == "madelon" and protocol["width"] == 500 and protocol["seed"] == 42
            and protocol["arm_limit_s"] == 900 and protocol["cold_start"] is True
            and "k_selection" in protocol and protocol.get("fixed_k") is None, "Invalid selection protocol")
    expected = {("madelon", method, arm) for method in METHODS for arm in ARMS}
    require(len(protocol["tasks"]) == len(expected) and set(map(tuple, protocol["tasks"])) == expected
            and len(protocol["methods"]) == len(METHODS) and set(protocol["methods"]) == set(METHODS),
            "Invalid 11-method arm schedule")
    for path, digest in protocol["source_sha256"].items():
        require(sha(run/"source"/path) == digest, f"Changed source snapshot {path}")
    for path in ("src/minppi_mofs/piv2.py", "src/minppi_mofs/modeling.py", "src/minppi_mofs/utility.py"):
        require(protocol["source_sha256"][path] == old["source_sha256"][path], f"Changed selector/model core: {path}")
    return protocol, expected


def check_curve(method, result, frozen):
    diag = frozen["selector"]
    candidates, order = frozen["candidates"], diag["order"]
    width = len(candidates)
    require(len(order) == len(set(order)) == width and set(order) == set(candidates), "Invalid ranking")
    curve = np.asarray(diag["curve"], dtype=float)
    seconds = np.asarray(diag["seconds_by_k"], dtype=float)
    require(curve.shape == seconds.shape == (width,) and np.isfinite(curve).all()
            and np.all((curve >= 0) & (curve <= 1)), "Missing or invalid validation AUC")
    require(np.isfinite(seconds).all() and np.all(seconds >= 0)
            and 0 <= seconds.sum() <= diag["scan_s"] + 1e-5 and diag["ranking_s"] >= 0,
            "Invalid prefix timing")
    expected_k = int(np.flatnonzero(curve >= curve.max() - TOL)[0]) + 1
    require(diag["selected_k"] == result["retained"] == expected_k, "AUC maximum/tie rule violated")
    selected = set(order[:expected_k])
    require([c for c in candidates if c in selected] == frozen["selected"], "Selected mask differs from winning prefix")
    near(curve[expected_k - 1], result["validation_utility"])
    require(type(diag["reused_full_validation"]) is bool and type(diag["ranking_fits"]) is int
            and diag["ranking_fits"] >= int(diag["reused_full_validation"])
            and diag["prefix_fits"] == width - int(diag["reused_full_validation"])
            and result["fits"] == diag["ranking_fits"] + diag["prefix_fits"] + 1,
            "Invalid ranking/prefix/final fit counts")


def verify_fixed_reference(run, method, arm, out, top, full_scores):
    folder = FIXED/"madelon"/method/arm
    old, old_result = read(folder/"frozen.json"), read(folder/"result.json")
    require(canonical_hash(old) == old_result["frozen_hash"] and old_result == read(folder/"watchdog.json")["result"]
            and len(old["selected"]) == 23, "Invalid k=23 archive")
    out.update(fixed23_test_auc=old_result["test_utility"], fixed23_test_accuracy=old_result["test_secondary"],
               fixed23_validation_auc=old_result["validation_utility"])
    if out["status"] == "timed_out":
        location = run/"madelon"/method/arm
        watchdog = read(location/"watchdog.json")
        require(watchdog["kill_requested_s"] is not None, "No recorded deadline for timeout")
        progress = watchdog["progress"]
        out.update(phase=progress.get("phase"), checked=progress.get("checked"),
                   prefilter_s=progress.get("prefilter_s"))
        return
    require(out["status"] == "completed", f"Unresolved method failure: {method}/{arm}")
    current = read(run/"madelon"/method/arm/"frozen.json")
    for field in ("partition_sha256", "model_policy", "features", "candidates", "prefilter"):
        require(current[field] == old[field], f"{method}/{arm} differs at {field}")
    order, curve = current["selector"]["order"], current["selector"]["curve"]
    require(len(order) == (500 if arm == "plain" else 322), "Unexpected prefix count")
    require(set(order[:23]) == set(old["selected"]), f"{method}/{arm}: different k=23 mask")
    near(curve[22], old_result["validation_utility"])
    near(curve[-1], full_scores.setdefault(arm, curve[-1]))
    out.update(checked=len(curve), validation_auc_at_23=curve[22], outside_top100=len(set(current["selected"])-top),
               candidate_count=len(order), phase="complete")


def text_report(pairs, evidence):
    lines = ["# Madelon 原始 500 列，seed 42，按完整验证集 AUC 选择列数", "",
             "每个方法先对可用特征排序，再用相同配置的 XGBoost 逐个检验所有非空前缀。原法检查 500 种列数，"
             "加 PIv2 后检查 322 种。完整验证集 AUC 最高的子集入选，差值在 1e−14 内时取较少列。"
             "随后在训练集加验证集上重新拟合，并在测试集上评价一次。", "",
             "箭头左侧为原方法，右侧为同一方法加 PIv2。表中总时间从独立进程启动计到结束，含前置、完整验证搜索、"
             "最终重训、测试预测和结果保存。每次冷启动限时 900 秒。", "",
             "| 方法 | 实际保留列数 | 总耗时（秒） | PIv2 前置（秒） | 所选子集验证 AUC | 测试 AUC | 测试准确率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]

    for row in pairs.to_dict("records"):
        def fmt(key, spec):
            number = row.get(key)
            return format(number, spec) if pd.notna(number) else "—"

        def both(field, spec):
            return " → ".join(fmt(a+"_"+field, spec) for a in ARMS)

        name = METHODS[row["backend"]]
        if not row["complete"]:
            name += "（含超时；仅已完成侧给出分数）"
        lines.append("| " + " | ".join([name, both("retained", ".0f"), both("wall_s", ".2f"),
                     fmt("piv2_prefilter_s", ".3f"), both("validation_utility", ".5f"),
                     both("test_utility", ".5f"), both("test_secondary", ".2%")]) + " |")
    lines += ["", f"每种方法只用一个划分、各计时一次。共用输入的准备耗时 {evidence['input_preparation_s']:.2f} 秒，"
              "单独记录。同一个验证集承担多列数比较，最佳验证 AUC 可能偏乐观；测试集衡量所选子集的预测表现。"
              "方差法的输入列已经按训练集标准化，各列方差近似为 1，排序对浮点舍入和并列处理敏感。", "",
              "本次保留 k=23 作为历史复核点。每条完整曲线在该位置的列集合和验证 AUC 均与上一轮一致。"
              "配对明细同时列出原固定 23 列和本次入选子集的分数。", "",
              "[配对明细与固定 23 列对照](paired.csv) · [逐次运行](summary.csv) · "
              "[独立核验](audit.json) · [冻结协议](protocol.json) · [运行日志](run.log)"]
    return "\n".join(lines) + "\n"


def main(run):
    protocol, expected = verify_protocol(run)
    columns, top, labels, hashes, overlap = verify_inputs(run, protocol)
    source_row = read(run/"input/provenance.json")["row"]
    inputs = {"madelon": (source_row, hashes, labels)}
    summary = pd.read_csv(run/"summary.csv").set_index(["dataset", "backend", "arm"], drop=False)
    require(summary.index.is_unique and set(summary.index) == expected, "Incomplete or duplicate terminal arms")
    fingerprint = canonical_hash(protocol)
    counts, supports, full_scores, records = Counter(), {}, {}, []
    for method in METHODS:
        row = dict(backend=method)
        for arm in ARMS:
            key = ("madelon", method, arm)
            out = inspect_arm(run, key, summary.loc[key], protocol, fingerprint, inputs, supports, counts, None,
                              selector_check=check_curve)
            verify_fixed_reference(run, method, arm, out, top, full_scores)
            row.update({arm+"_"+k:v for k,v in out.items()})
        row["complete"] = row["plain_status"] == row["piv2_status"] == "completed"
        if row["complete"]:
            row.update(time_saved_fraction=1-row["piv2_wall_s"]/row["plain_wall_s"],
                       test_auc_difference=row["piv2_test_utility"]-row["plain_test_utility"],
                       test_accuracy_difference=row["piv2_test_secondary"]-row["plain_test_secondary"])
        records.append(row)
    pairs = pd.DataFrame(records)
    evidence = dict(status="pass", method_count=len(METHODS), completed_pairs=int(pairs.complete.sum()),
                    counts=dict(counts), fixed23_masks_and_validation_auc_verified=counts["completed"],
                    protocol_sha256=fingerprint, summary_sha256=sha(run/"summary.csv"),
                    fixed23_archive=FIXED.name, fixed23_archive_protocol_sha256=sha(FIXED/"protocol.json"),
                    full_input_validation_auc=full_scores, prefilter_sha256=supports,
                    original_top100_max_abs_difference=overlap, original_width=len(columns),
                    rows={role:len(y) for role,y in labels.items()},
                    input_preparation_s=float(read(run/"input_preparation.json")["elapsed_s"]))
    pairs.to_csv(run/"paired.csv", index=False)
    (run/"comparison.md").write_text(text_report(pairs, evidence), encoding="utf-8")
    evidence["artifacts"] = {p:sha(run/p) for p in ("paired.csv", "comparison.md")}
    atomic_write_json(run/"audit.json", evidence)
    print(f"{run} | audited {counts['completed']} completed arms, {evidence['completed_pairs']} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
