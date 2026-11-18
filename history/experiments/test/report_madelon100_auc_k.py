"""Verify the official Top-100 Madelon validation-selected prefix scans and their two controls."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_piv0_benchmark import read, sha
from experiments.run_thesis_sampling import ROOT
from experiments.test.audit_piv2_prefilter import inspect_arm, near, require
from experiments.test.report_madelon500_auc_k import ARMS, METHODS, check_curve
from experiments.thesis_prepared_catalog import audit_prepared_catalog
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


BASE = ROOT / "results/test/thesis_sampling"
FIXED = BASE / "PIv2-seed42-all-methods-summary"
WIDE = BASE / "PIv2-madelon500-validation-k-seed42"


def audited_history(folder):
    audit = read(folder/"audit.json")
    require(audit["status"] == "pass" and sha(folder/"paired.csv") == audit["artifacts"]["paired.csv"],
            f"Changed historical results in {folder}")
    history = pd.read_csv(folder/"paired.csv")
    if "dataset" in history.columns:
        history = history[(history.dataset == "madelon") & (history.seed == 42)]
    history = history[history.backend.isin(METHODS)].set_index("backend")
    require(history.index.is_unique and set(history.index) == set(METHODS) and history.complete.all(),
            f"Missing historical arm in {folder}")
    return audit, history


def verify_inputs(run, protocol):
    jobs, catalog = audit_prepared_catalog()
    source = jobs["madelon", 42]
    manifest = source/"data_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        row, = [r for r in csv.DictReader(stream) if r["dataset_id"] == "madelon" and int(r["seed"]) == 42]
    require(protocol["catalog_sha256"] == catalog["sha256"] and protocol["input_sha256"] == sha(manifest)
            and int(row["selected_feature_count"]) == 100, "Unexpected official Top-100 input")
    wide = read(WIDE/"protocol.json")
    source_wide = read(WIDE/"input/provenance.json")
    require(sha(WIDE/"input/provenance.json") == wide["input_sha256"]
            and sha(manifest) == source_wide["reference_manifest_sha256"]
            and json.loads(row["selected_features_json"]) == source_wide["top100"],
            "Widths use different source rows or reference columns")
    for item in ("seed", "model_params", "hardware_backend", "cold_start", "arm_limit_s", "k_selection", "tie_rule"):
        require(protocol[item] == wide[item], f"Different {item} between widths")
    for item in ("src/minppi_mofs/piv2.py", "src/minppi_mofs/modeling.py", "src/minppi_mofs/utility.py",
                 "experiments/test/piv2_auc_prefix.py", "experiments/test/piv2_prefix_rankings.py"):
        require(protocol["source_sha256"][item] == wide["source_sha256"][item], f"Different calculation {item}")
    for name, digest in protocol["source_sha256"].items():
        require(sha(run/"source"/name) == digest, f"Changed source snapshot {name}")
    labels, digests = {}, {}
    for role in ("train", "val", "test"):
        path = source/row[role+"_relpath"]
        digests[role] = sha(path)
        require(digests[role] == row[role+"_file_sha256"] == source_wide["reference_partition_sha256"][role]
                and row[role+"_index_sha256"] == source_wide["index_sha256"][role], f"Changed {role} partition")
        data = pd.read_csv(path)
        labels[role] = data.pop(row["target"]).to_numpy()
        wide_path = WIDE/"input"/source_wide["row"][role+"_relpath"]
        require(sha(wide_path) == wide["input_partition_sha256"][role], f"Changed 500-column {role} rows")
        same_rows = pd.read_csv(wide_path, usecols=data.columns.tolist()+[row["target"]])
        np.testing.assert_array_equal(labels[role], same_rows.pop(row["target"]).to_numpy())
        np.testing.assert_array_equal(data.to_numpy(), same_rows[data.columns].to_numpy())
    require(digests == protocol["input_partition_sha256"], "Protocol partition hashes disagree")
    return {"madelon": (row, digests, labels)}


def align_reference(run, method, arm, result, fixed, wide, full_endpoints):
    old_folder = BASE/fixed.source_run/"madelon"/method/arm
    old = read(old_folder/"frozen.json")
    old_result = read(old_folder/"result.json")
    require(canonical_hash(old) == old_result["frozen_hash"] and old_result == read(old_folder/"watchdog.json")["result"]
            and old_result["retained"] == len(old["selected"]) == 10, "Changed historical fixed-10 mask")
    for key in ("wall_s", "retained", "validation_utility", "test_utility", "test_secondary"):
        result["fixed10_"+key] = fixed[arm+"_"+key]
        result["width500_"+key] = wide[arm+"_"+key]
    near(old_result["validation_utility"], result["fixed10_validation_utility"])
    for key in ("test_utility", "test_secondary"):
        near(old_result[key], result["fixed10_"+key])
    folder = run/"madelon"/method/arm
    if result["status"] == "timed_out":
        timer = read(folder/"watchdog.json")
        require(timer["kill_requested_s"] is not None, "Unconfirmed timeout")
        for key in ("phase", "checked", "prefilter_s"):
            result[key] = timer["progress"].get(key)
        return
    require(result["status"] == "completed", f"Unresolved error for {method}/{arm}")
    current = read(folder/"frozen.json")
    for key in ("partition_sha256", "model_policy", "features", "candidates", "prefilter"):
        require(old[key] == current[key], f"Changed {method}/{arm} {key}")
    path = current["selector"]
    require(len(path["order"]) == (100 if arm == "plain" else 64)
            and set(path["order"][:10]) == set(old["selected"]), f"Changed {method}/{arm} at k=10")
    near(path["curve"][9], old_result["validation_utility"])
    near(path["curve"][-1], full_endpoints.setdefault(arm, path["curve"][-1]))
    result.update(checked=len(path["curve"]), validation_auc_at_10=path["curve"][9], phase="complete")


def report_table(pairs):
    lines = ["# Madelon 预处理 Top-100，seed 42，按验证 AUC 自选列数", "",
             "原法在 100 列上逐一验证 k=1 至 100。PIv2 从训练集筛出 64 列，同一后端随后检查 k=1 至 64。"
             "每个子集在完整训练分区拟合，在完整验证分区评估，取得最高 AUC 的最小 k 入选（同分容差 1e−14）。"
             "冻结后在训练加验证分区重训，在测试分区评分。三份分区分别为 1,664、416、520 行。", "",
             "箭头左侧为原法，右侧为同一方法加 PIv2。每次独立冷启动限时 15 分钟，总耗时含前置、排名、"
             "所有候选的验证、最终重训、测试预测与落盘。", "",
             "| 方法 | 最终列数 | 总耗时（秒） | PIv2 前置（秒） | 验证 AUC | 测试 AUC | 测试准确率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in pairs.to_dict("records"):
        def fmt(key, spec):
            return format(row[key], spec) if pd.notna(row[key]) else "—"
        def pair(key, spec):
            return " → ".join(fmt(arm+"_"+key, spec) for arm in ARMS)
        name = METHODS[row["backend"]] + ("" if row["complete"] else "（含超时）")
        lines.append("| "+" | ".join([name, pair("retained",".0f"), pair("wall_s",".2f"),
                     fmt("piv2_prefilter_s",".3f"), pair("validation_utility",".5f"),
                     pair("test_utility",".5f"), pair("test_secondary",".2%")])+" |")
    lines += ["", "## 与同一划分的两次对照", "",
              "各格依次为原法和原法＋PIv2 的测试 AUC。固定列数实验在这份 Top-100 输入上使用 k=10。"
              "原始 500 列的自选 k 实验使用相同训练、验证、测试样本、预测器与计时规则。", "",
              "| 方法 | Top-100 固定 10 列 | Top-100 验证选 k | 全 500 列验证选 k |",
              "|---|---:|---:|---:|"]
    for _, row in pairs.iterrows():
        scores = []
        for tag in ("fixed10_", "", "width500_"):
            values = [row[arm+"_"+tag+"test_utility"] for arm in ARMS]
            scores.append(" / ".join(f"{v:.5f}" if pd.notna(v) else "—" for v in values))
        lines.append("| "+" | ".join([METHODS[row.backend], *scores])+" |")
    lines += ["", "每个组合仅运行一个 seed、一次冷启动。大量候选共用一份验证集，最高验证分数可能偏乐观。"
              "方差排序使用训练集标准化后的输入，近似并列的方差对舍入和原始列序敏感。", "",
              "[包含三轮结果的 CSV](paired.csv) · [原始计时](summary.csv) · [审计](audit.json) · [运行日志](run.log)"]
    return "\n".join(lines)+"\n"


def main(run):
    fixed_audit, fixed = audited_history(FIXED)
    wide_audit, wide = audited_history(WIDE)
    protocol = read(run/"protocol.json")
    expected = {("madelon", m, a) for m in METHODS for a in ARMS}
    require(len(protocol["tasks"]) == len(expected) and set(map(tuple, protocol["tasks"])) == expected
            and set(protocol["methods"]) == set(METHODS) and protocol["width"] == 100
            and protocol["dataset"] == "madelon" and protocol["seed"] == 42 and protocol["cold_start"]
            and protocol["arm_limit_s"] == 900 and protocol.get("fixed_k") is None, "Unexpected Top-100 protocol")
    require(wide_audit["protocol_sha256"] == canonical_hash(read(WIDE/"protocol.json")), "Changed 500-column protocol")
    inputs = verify_inputs(run, protocol)
    summary = pd.read_csv(run/"summary.csv").set_index(["dataset","backend","arm"], drop=False)
    require(summary.index.is_unique and set(summary.index) == expected, "Missing or duplicate terminal arms")
    fingerprint = canonical_hash(protocol)
    supports = {"madelon": fixed_audit["matching_prefilter_sha256"]["madelon"]}
    counts, full, records = Counter(), {}, []
    for method in METHODS:
        row = dict(backend=method)
        for arm in ARMS:
            key = ("madelon", method, arm)
            result = inspect_arm(run, key, summary.loc[key], protocol, fingerprint, inputs, supports, counts, None,
                                 selector_check=check_curve)
            align_reference(run, method, arm, result, fixed.loc[method], wide.loc[method], full)
            row.update({arm+"_"+k:v for k,v in result.items()})
        row["complete"] = row["plain_status"] == row["piv2_status"] == "completed"
        if row["complete"]:
            row["time_saved_fraction"] = 1-row["piv2_wall_s"]/row["plain_wall_s"]
            row["test_auc_difference"] = row["piv2_test_utility"]-row["plain_test_utility"]
        records.append(row)
    pairs = pd.DataFrame(records)
    pairs.to_csv(run/"paired.csv", index=False)
    (run/"comparison.md").write_text(report_table(pairs), encoding="utf-8")
    audit = dict(status="pass", protocol_sha256=fingerprint, method_count=len(pairs), counts=dict(counts),
                 completed_pairs=int(pairs.complete.sum()), fixed10_masks_and_validation_auc_verified=counts["completed"],
                 prefilter_sha256=supports, full_input_validation_auc=full, partition_sha256=inputs["madelon"][1],
                 summary_sha256=sha(run/"summary.csv"), fixed10_audit_sha256=sha(FIXED/"audit.json"),
                 width500_audit_sha256=sha(WIDE/"audit.json"), archived_width500_values_and_rows_matched=True,
                 artifacts={p:sha(run/p) for p in ("paired.csv","comparison.md")})
    atomic_write_json(run/"audit.json", audit)
    print(f"{run} | verified {counts['completed']} completed arms, {audit['completed_pairs']} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
