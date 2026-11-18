"""Independently verify seed-42 Madelon-500 results and write the paired table."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_piv0_benchmark import read, sha
from experiments.run_thesis_sampling import ROOT
from experiments.test.audit_piv2_prefilter import (
    MEASURES, check_selector, metric_pair, near, ordered_subset, require, validate_votes,
)
from experiments.test.summarize_piv2_extended import METHODS, NEW, check_new
from experiments.test.thesis_madelon_alpha_data import RAW, ROLES, SOURCE
from minppi_mofs.experiment_protocol import atomic_write_json, canonical_hash


SELECTORS = {key: name for key, name in METHODS.items() if key not in {"mc32", "exact"}}
ARMS = ("plain", "piv2")
OLD_GAIN = ROOT / "results/test/thesis_sampling/madelon-all-sizes-20261111-162002-353446/500-direct_gain"


def verify_inputs(run, protocol):
    root = run / "input"
    provenance = read(root / "provenance.json")
    require(sha(root / "provenance.json") == protocol["input_sha256"], "Input provenance changed")
    row = provenance["row"]
    identity = canonical_hash(provenance["input_identity"])
    require(identity == row["protocol_hash"] == protocol["catalog_sha256"], "Invalid 500-column identity")
    raw = pd.read_csv(RAW)
    target = row["target"]
    raw_labels = raw.pop(target)
    require(sha(RAW) == row["raw_file_sha256"] == provenance["raw_sha256"], "Raw file identity changed")
    columns = raw.columns.tolist()
    require(len(columns) == len(set(columns)) == 500 and columns == provenance["columns"]
            == json.loads(row["selected_features_json"]), "Invalid original feature set")
    manifest = SOURCE / "data_manifest.csv"
    require(sha(manifest) == provenance["reference_manifest_sha256"], "Archived source manifest changed")
    records = pd.read_csv(manifest)
    official, = records[(records.dataset_id == "madelon") & (records.seed == 42)].to_dict("records")
    split = read(SOURCE / "preprocess/madelon/s42/split_metadata.json")
    top = json.loads(official["selected_features_json"])
    require(len(set(top)) == 100 and top == provenance["top100"] == split["selected_features"],
            "Different reference Top-100")
    train = raw.iloc[split["train_indices"]].to_numpy(dtype=float)
    mean, scale = train.mean(axis=0), train.std(axis=0)
    require((scale > 0).all(), "Unexpected constant source column")
    labels, observed, digests = {}, {}, {}
    for role in ROLES:
        indices = split[f"{role}_indices"]
        require(canonical_hash(indices) == official[f"{role}_index_sha256"] == provenance["index_sha256"][role],
                f"Different {role} indices")
        part = root / row[f"{role}_relpath"]
        digests[role] = sha(part)
        require(digests[role] == row[f"{role}_file_sha256"], f"Different {role} CSV")
        wide = pd.read_csv(part)
        labels[role] = wide.pop(target).to_numpy()
        require(wide.columns.tolist() == columns and len(indices) == len(wide) == row[f"{role}_row_count"],
                f"Unexpected {role} feature order or size")
        np.testing.assert_array_equal(labels[role], raw_labels.iloc[indices].to_numpy())
        np.testing.assert_allclose(wide, (raw.iloc[indices].to_numpy()-mean)/scale, atol=1e-12, rtol=0)
        old_path = SOURCE / official[f"{role}_relpath"]
        require(sha(old_path) == official[f"{role}_file_sha256"], f"Archived Top-100 {role} changed")
        old = pd.read_csv(old_path)
        np.testing.assert_array_equal(labels[role], old.pop(target))
        require(old.columns.tolist() == top, f"Archived {role} has a different column order")
        observed[role] = float(np.max(np.abs(wide[top].to_numpy() - old.to_numpy())))
        require(observed[role] <= 1e-12 and observed[role] == provenance["top100_max_abs_diff"][role],
                f"{role} Top-100 mismatch")
    require(digests == protocol["input_partition_sha256"], "Frozen partitions differ across protocol and data")
    return columns, set(top), labels, digests, observed


def verify_arm(run, method, arm, summary, protocol, identity, inputs, supports):
    columns, top, labels, digests, _ = inputs
    folder = run / "madelon" / method / arm
    clock = read(folder / "watchdog.json")
    require(tuple(summary[k] for k in ("dataset", "backend", "arm", "seed")) == ("madelon", method, arm, 42)
            and summary["status"] == clock["status"] and clock["limit_s"] == 900, f"Invalid {method}/{arm} identity")
    near(summary["wall_s"], clock["elapsed_s"])
    out = dict(status=clock["status"], wall_s=clock["elapsed_s"])
    if clock["status"] != "completed":
        require(clock["status"] == "timed_out" and clock["kill_requested_s"] is not None,
                f"{method}/{arm}: unexpected execution failure")
        out.update({k:clock["progress"].get(k) for k in ("phase","checked","candidates","prefilter_s")})
        return out
    result, frozen = read(folder/"result.json"), read(folder/"frozen.json")
    require(clock["exit_code"] == 0 and clock["kill_requested_s"] is None and clock["result"] == result,
            "Watchdog/result disagreement")
    require(result["frozen_hash"] == canonical_hash(frozen) and frozen["protocol_fingerprint"] == identity
            and frozen["partition_sha256"] == digests and frozen["features"] == columns, "Changed mask or data")
    for item in (frozen, result):
        require((item["dataset"], item["seed"], item["backend"], item["arm"], item["metric"]) ==
                ("madelon", 42, method, arm, "roc_auc"), "Different result identity")
    require(frozen["model_policy"] == dict(task="binary", seed=42, backend=protocol["hardware_backend"],
            model_params=protocol["model_params"]), "Different predictor")
    candidates, selected = frozen["candidates"], frozen["selected"]
    require(ordered_subset(candidates, columns) and ordered_subset(selected, candidates)
            and result["width"] == 500 and result["retained"] == len(selected)
            and result["prefilter_retained"] == len(candidates), "Invalid selection")
    if arm == "piv2":
        validate_votes(frozen["prefilter"], columns, candidates)
        supports.add(canonical_hash(frozen["prefilter"]))
        require(len(supports) == 1, "PIv2 columns or votes changed across backends")
    else:
        require(candidates == columns and frozen["prefilter"] is None and result["prefilter_s"] == 0,
                "Unexpected plain-arm front end")
    (check_new if method in NEW else check_selector)(method, result, frozen)
    for k in MEASURES:
        near(summary[k], result[k])
        out[k] = result[k]
    stages = [result[k] for k in ("backend_s","final_s","prefilter_s")]
    require(min(stages) >= 0 and sum(stages) <= clock["elapsed_s"] <= 900, "Invalid stage timers")
    event, = read(folder/"test_access.json")["events"]
    require(event["seed"] == 42 and event["job_id"] == result["frozen_hash"] and event["dataset_id"] == "madelon"
            and event["reason"] == "final_removal_evaluation" and event["decision"] == "allowed",
            "Held-out access not bound to the frozen mask")
    out["outside_top100"] = len(set(selected) - top)
    out["prefilter_outside_top100"] = len(set(candidates) - top)
    with np.load(folder/"predictions.npz", allow_pickle=False) as pred:
        for split, y, p, prefix in (("val","validation_y","validation","validation"), ("test","y","selected","test")):
            np.testing.assert_array_equal(pred[y], labels[split])
            pair = metric_pair(pred[y], pred[p], "roc_auc")
            near(pair, [result[prefix+"_utility"], result[prefix+"_secondary"]])
            if split == "val":
                near(pair, [frozen["validation_utility"], frozen["validation_secondary"]])
        if method == "gain" and arm == "plain":
            archived = read(OLD_GAIN/"frozen.json")
            curve_path = OLD_GAIN/"validation_curve.csv"
            require(sha(curve_path) == archived["curve_sha256"], "Historical Gain curve fingerprint mismatch")
            old_curve = pd.read_csv(curve_path)
            require(old_curve.retained.tolist() == list(range(1,501))
                    and selected == archived["selected"], "Old Gain result differs")
            near(old_curve.pooled_auc, frozen["selector"]["curve"])
            code = 0
            positions = {name:i for i,name in enumerate(columns)}
            for col, mask in zip(frozen["selector"]["gain_order"], old_curve["mask"], strict=True):
                code |= 1 << positions[col]
                require(code == int(mask,16), "Different historical Gain prefix")
            with np.load(OLD_GAIN/"predictions.npz", allow_pickle=False) as old:
                np.testing.assert_array_equal(old["y"], pred["y"])
                np.testing.assert_array_equal(old["selected"], pred["selected"])
    if method == "variance":
        out["variance_spread"] = float(np.ptp(frozen["selector"]["ranking_scores"]))
    return out


def render(pairs, report):
    def value(data, key, spec):
        number = data.get(key)
        return format(number, spec) if pd.notna(number) else "—"

    lines = ["# Madelon 原始 500 列，seed 42：各方法与加入 PIv2 的配对结果", "",
             "使用相同的训练、验证、测试样本及顺序。保留全部 500 列，按训练集均值和标准差变换，"
             "未做原先的互信息 Top-100 截断。每组独立冷启动一次，截止 900 秒。"
             "箭头左侧是原方法，右侧是相同后端加入 PIv2。", "",
             "端到端时间包含进程启动、读入与核验预先准备的分区、前置、后端、最终重训、测试预测与落盘。"
             "固定列数的方法在 500 列实验中两侧均以原宽度确定 k=23；此前 100 列实验的同一规则给出 k=10。"
             "Top-100 外列数仅用于与上轮输入范围对照。", "",
             "| 方法 | 前置秒 | 总耗时（秒） | 节时率 | 最终保留列数 | 其中原 Top-100 外 | 测试 AUC | 测试准确率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pairs.to_dict("records"):
        def both(key, spec):
            return " → ".join(value(r, arm+"_"+key, spec) for arm in ARMS)
        name = SELECTORS[r["backend"]]
        if not r["complete"]:
            name += f"（{r['plain_status']} → {r['piv2_status']}）"
        cells = [name, value(r,"piv2_prefilter_s",".3f"), both("wall_s",".2f"),
                 value(r,"time_saved_fraction","+.1%"), both("retained",".0f"), both("outside_top100",".0f"),
                 both("test_utility",".5f"), both("test_secondary",".2%")]
        lines.append("| "+" | ".join(cells)+" |")
    lines += ["", f"本次共用输入的重建、标准化和身份核验用时 {report['shared_input_preparation_s']:.2f} 秒，"
              "单独计时一次。[数据准备计时](input_preparation.json)"]
    span = report.get("plain_variance_range")
    if span is not None and span < 1e-10:
        lines.extend(["", f"在完整训练行上，标准化使各列方差约为 1，原方差排序的最大值与最小值只相差 "
                      f"{span:.2e}。该行选列对浮点舍入及并列顺序敏感。"])
    lines.extend(["", "[配对明细与验证指标](paired.csv) · [逐次运行](summary.csv) · [独立核验](audit.json) · "
                  "[数据身份](input/provenance.json) · [固定协议](protocol.json) · [日志](run.log)"])
    return "\n".join(lines)+"\n"


def main(run):
    protocol = read(run/"protocol.json")
    require(protocol["dataset"] == "madelon" and protocol["seed"] == 42 and protocol["width"] == 500
            and protocol["fixed_k"] == 23 and protocol["cold_start"] and protocol["arm_limit_s"] == 900,
            "Unexpected protocol")
    for name, digest in protocol["source_sha256"].items():
        require(sha(run / "source" / name) == digest, f"Changed source snapshot {name}")
    expected = {("madelon", backend, arm) for backend in SELECTORS for arm in ARMS}
    require(len(protocol["tasks"]) == len(expected) and set(map(tuple, protocol["tasks"])) == expected
            and set(protocol["methods"]) == set(SELECTORS), "Unexpected tasks")
    inputs = verify_inputs(run, protocol)
    prep_s = float(read(run/"input_preparation.json")["elapsed_s"])
    require(prep_s >= 0 and np.isfinite(prep_s), "Invalid shared input preparation timing")
    summary = pd.read_csv(run/"summary.csv").set_index(["dataset","backend","arm"], drop=False)
    require(summary.index.is_unique and set(summary.index) == expected, "Missing or duplicated arms")
    identity, supports, rows = canonical_hash(protocol), set(), []
    for backend in SELECTORS:
        results = {arm:verify_arm(run, backend, arm, summary.loc[("madelon",backend,arm)], protocol, identity,
                                  inputs, supports) for arm in ARMS}
        a, b = results.values()
        complete = a["status"] == b["status"] == "completed"
        row = dict(backend=backend, complete=complete)
        for arm, values in results.items():
            row.update({arm+"_"+k:v for k,v in values.items()})
        if complete:
            row.update(time_saved_fraction=1-b["wall_s"]/a["wall_s"], speedup=a["wall_s"]/b["wall_s"],
                       test_auc_difference=b["test_utility"]-a["test_utility"],
                       test_accuracy_difference=b["test_secondary"]-a["test_secondary"])
        rows.append(row)
    pairs = pd.DataFrame(rows)
    var = next(row for row in rows if row["backend"] == "variance")
    report = dict(status="pass", protocol_sha256=identity, method_count=len(SELECTORS), shared_input_preparation_s=prep_s,
                  completed_pairs=int(pairs.complete.sum()), summary_sha256=sha(run/"summary.csv"),
                  counts=dict(Counter(summary.status)), prefilter_diagnostics_sha256=sorted(supports),
                  original_top100_max_abs_difference=inputs[-1], original_partition_rows={k:len(v) for k,v in inputs[2].items()},
                  plain_variance_range=var.get("plain_variance_spread"),
                  historical_500_gain_verified=next(r["plain_status"]=="completed" for r in rows if r["backend"]=="gain"),
                  historical_500_gain=str(OLD_GAIN.relative_to(ROOT)))
    pairs.to_csv(run/"paired.csv", index=False)
    (run/"comparison.md").write_text(render(pairs, report), encoding="utf-8")
    report["artifacts"] = {n:sha(run/n) for n in ("paired.csv","comparison.md")}
    atomic_write_json(run/"audit.json", report)
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
