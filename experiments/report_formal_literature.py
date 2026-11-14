"""Audit independent formal runs and aggregate seed-level results without refitting."""
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from experiments.report_literature_benchmark import LABEL, ledger, match, read
from experiments.run_formal_literature import DEFAULT
from experiments.test.piv3_lite_arm import _metrics
from experiments.test.piv3_raw_splits import _sha
from minppi_mofs.experiment_protocol import canonical_hash

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ("seed", "dataset", "mode", "method")
SCORES = ("auc", "accuracy", "ap", "r2", "mae")
MEASURES = ("k", "deleted", "fits", "wall_s", *SCORES)
PREDICTIONS = ("sample_prediction_rows", "sample_prediction_calls",
               "selection_prediction_rows", "selection_prediction_calls")
FORMAL_LABEL = {**LABEL, "sampling_shap": "SamplingSHAP"}
MODES = {"shared": "共享参照", "native": "文献估计量", "frozen": "统一冻结效用"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=48)
def _source(directory, seed):
    meta = read(directory / "metadata.json")
    require(meta["policy"]["seed"] == seed, f"Unexpected input seed at {directory}")
    for name in ("val_y.npy", "test_y.npy"):
        require(_sha(directory / name) == meta["sha256"][name], f"Input changed: {directory / name}")
    yval = np.load(directory / "val_y.npy", allow_pickle=False)
    ytest = np.load(directory / "test_y.npy", allow_pickle=False)
    rank, guard = train_test_split(np.arange(len(yval)), test_size=.5, random_state=seed,
                                   stratify=None if meta["task"] == "regression" else yval)
    return meta, np.sort(rank), np.sort(guard), yval, ytest


def _route(root, job, pins):
    seed, dataset, mode, method = (job[k] for k in KEYS)
    source = root / "inputs" / f"seed-{seed}" / dataset
    meta, rank, guard, yval, ytest = _source(source, seed)
    cols = meta["features"]
    require(len(cols) == meta["width"] == len(set(cols)), f"Invalid schema: {source}")
    if "metadata_sha256" in pins:
        require(_sha(source / "metadata.json") == pins["metadata_sha256"], f"Metadata changed: {source}")
    folder = root / "runs" / f"seed-{seed}" / dataset / mode / method
    row = {**{k:job[k] for k in KEYS}, "width":len(cols)}
    if not (folder / "watchdog.json").exists():
        if (folder / "interrupted.json").is_file():
            row["status"] = "interrupted"
        else:
            row["status"] = "in_progress" if folder.is_dir() else "not_started"
        return row, []
    timer = read(folder / "watchdog.json")
    row.update(status=timer["status"], wall_s=timer["elapsed_s"])
    require(np.isfinite(row["wall_s"]) and row["wall_s"] > 0, f"Invalid cold time: {folder}")
    counts = ledger(folder)
    row.update(fits=counts.get("completed", 0), fits_started=counts.get("started", 0))
    if timer["status"] != "completed":
        return row, []
    res, frozen, access = [read(folder / name) for name in ("result.json", "frozen.json", "test_access.json")]
    require(res == timer["result"] and all(res[k] == row[k] for k in KEYS), f"Wrong route: {folder}")
    require(all(res[k] == row[k] for k in ("width","fits","fits_started"))
            and res["fits"] == res["fit_count"] == res["fits_started"], f"Fit ledger disagreement: {folder}")
    require(canonical_hash(frozen) == res["frozen_hash"] == access["frozen_hash"]
            and access["test_y_sha256"] == meta["sha256"]["test_y.npy"]
            and access["test_x_sha256"] == meta["sha256"]["test.npy"], f"Test access mismatch: {folder}")
    selected = res["selected"]
    require(selected == frozen["selected"] == res["candidate"] == frozen["candidate"]
            and not res["fallback"] and not res["enforce_guard"]
            and len(selected) == res["candidate_k"] == res["k"] == len(set(selected))
            and set(selected).issubset(cols), f"Incorrect delivered subset: {folder}")
    if method == "full":
        require(selected == cols, f"Full reference has missing columns: {folder}")
    with np.load(folder / "predictions.npz", allow_pickle=False) as pred:
        np.testing.assert_array_equal(ytest, pred["y_test"])
        np.testing.assert_array_equal(rank, pred["rank_rows"])
        np.testing.assert_array_equal(guard, pred["guard_rows"])
        np.testing.assert_array_equal(yval[guard], pred["y_guard"])
        np.testing.assert_array_equal(pred["guard_candidate"], pred["guard_selected"])
        for name, key in (("test_scores","test_pred"), ("validation_selected","guard_selected"),
                          ("validation_candidate","guard_candidate"), ("validation_full","guard_full")):
            true = ytest if name == "test_scores" else yval[guard]
            match(_metrics(true, pred[key], meta["utility"], dataset == "credit_card_fraud"), res[name])
        scores = dict(res["test_scores"])
        if meta["task"] == "regression":
            scores["r2_protocol"] = scores["r2"]
            scores["r2"] = float(r2_score(ytest, pred["test_pred"]))
    row.update(k=len(selected), deleted=len(cols)-len(selected), selector_s=res["selector_s"], **scores)
    for key in PREDICTIONS:
        if key in res:
            value = res[key]
            require(type(value) is int and value >= 0, f"Invalid {key}: {folder}")
            row[key] = value
    if "sampling_policy" in res:
        policy = res["sampling_policy"]
        row["sampling_policy"] = policy if isinstance(policy, str) else json.dumps(
            policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    positions = {c:i for i,c in enumerate(cols, 1)}
    features = [{**{k:row[k] for k in KEYS}, "selected_position":i, "source_position":positions[c],
                 "feature":c} for i,c in enumerate(selected, 1)]
    return row, features


def _write(path, rows, fields):
    with NamedTemporaryFile(mode="w", newline="", encoding="utf-8-sig", dir=path.parent,
                            prefix=path.stem+".", suffix=".tmp", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(f.name, path)


def _fmt(row, key):
    avg, sd = row.get(key+"_mean"), row.get(key+"_sd")
    if avg is None:
        return "—"
    digits = 2 if key in ("wall_s","k","deleted","fits",*PREDICTIONS) else 4
    value = f"{avg:.{digits}f}"
    return value if sd is None else f"{value} ± {sd:.{digits}f}"


def report(root=DEFAULT, *, partial=False, xlsx=False):
    root = Path(root)
    _source.cache_clear()
    protocol = read(root / "protocol.json")
    jobs = protocol["jobs"]
    keys = [tuple(j[k] for k in KEYS) for j in jobs]
    require(keys and len(keys) == len(set(keys)), "Protocol contains no jobs or duplicate jobs")
    rows, features = [], []
    by_group = defaultdict(list)
    for job in jobs:
        pins = protocol["inputs"][str(job["seed"])][job["dataset"]]
        row, cols = _route(root, job, pins)
        rows.append(row)
        features.extend(cols)
        by_group[(row["dataset"],row["mode"],row["method"])].append(row)
    has_sampling = any(r["method"] == "sampling_shap" or any(key in r for key in (*PREDICTIONS, "sampling_policy"))
                       for r in rows)
    prediction_fields = PREDICTIONS if has_sampling else ()
    measures = (*MEASURES, *prediction_fields)
    states = Counter(r["status"] for r in rows)
    if not partial:
        require(states == {"completed": len(jobs)}, f"Incomplete formal experiment: {dict(states)}")
    references = {(r["seed"],r["dataset"]):r for r in rows if r["mode"] == "shared"
                  and r["method"] == "full" and r["status"] == "completed"}
    for row in rows:
        base = references.get((row["seed"],row["dataset"]))
        if base and row["status"] == "completed":
            row.update({"full_"+k:base[k] for k in SCORES if k in base})
    groups = []
    for (dataset,mode,method), cohort in by_group.items():
        done = [r for r in cohort if r["status"] == "completed"]
        group = dict(dataset=dataset, mode=mode, method=method, width_min=min(r["width"] for r in cohort),
                     width_max=max(r["width"] for r in cohort),
                     n_completed=len(done), n_planned=len(cohort))
        for field in measures:
            numbers = [r[field] for r in done if field in r]
            if field in prediction_fields and len(numbers) != len(done):
                continue
            if numbers:
                require(len(numbers) == len(done) and all(map(math.isfinite,numbers)), f"Inconsistent {field}")
                group[field+"_mean"] = fmean(numbers)
                group[field+"_sd"] = stdev(numbers) if len(numbers)>1 else None
        groups.append(group)
    _write(root/"per-seed.csv",rows,[*KEYS,"width","status","k","deleted","fits","fits_started","wall_s",
           "selector_s",*SCORES,"r2_protocol",*["full_"+k for k in SCORES],
           *prediction_fields,*(["sampling_policy"] if has_sampling else [])])
    _write(root/"five-seed-summary.csv",groups,["dataset","mode","method","width_min","width_max","n_completed","n_planned",
           *[k+"_"+s for k in measures for s in ("mean","sd")]])
    _write(root/"retained-features.csv",features,[*KEYS,"selected_position","source_position","feature"])
    lines = ["# 正式实验 · 预处理数据集","",
             f"{len(jobs)} 条计划路线，{states['completed']} 条完成。数据集与方法均按 protocol.json 指定的 seed 汇总。",
             "", "表中为已完成 seed 的算术均值 ± 样本标准差。只有一个观测时仅列均值。总秒为从启动独立进程到结束的墙钟时间。"
             "分类报告测试 AUC 与准确率，欺诈检测另列 AP，ISOLET 使用宏平均 OvR AUC。回归报告未截断的测试 R² 与 MAE。",
             "", "需要把排名转为子集的方法在冻结模型上检查有限前缀，内部代理容差为 8%。最终交付各自选出的子集，"
             "原实验 5% 的独立验证回退已关闭。共享方法的每个数据集与 seed 组合独立计时一次。",
             "", "[逐 seed 明细](per-seed.csv) · [跨 seed 数值](five-seed-summary.csv) · "
             "[实际保留特征](retained-features.csv)"]
    if has_sampling:
        lines.extend(["", "采样预测行数按模型实际处理的输入行累加；同一行在不同遮盖下可重复计数。"
                      "多分类一次前向同时返回所有类别。选择总预测量包含采样与选列定宽，逐 seed 表另列实际调用数和采样配置。"])
    game_modes = [mode for mode in ("native", "frozen") if any(g["mode"] == mode for g in groups)]
    for mode in game_modes or ["shared"]:
        label = MODES[mode]
        subset = [r for r in groups if r["mode"] == mode]
        if not subset:
            continue
        mode_data = {r["dataset"] for r in subset}
        if mode != "shared":
            subset.extend(r for r in groups if r["mode"] == "shared" and r["dataset"] in mode_data)
        labels = tuple(FORMAL_LABEL)
        order = list(dict.fromkeys(r["dataset"] for r in groups))
        subset.sort(key=lambda r:(order.index(r["dataset"]), labels.index(r["method"])))
        prediction_columns = ("sample_prediction_rows", "selection_prediction_rows") if has_sampling else ()
        displayed = ("k", "fits", *prediction_columns, "wall_s", *SCORES)
        heading = "采样预测行 | 选择总预测行 | " if has_sampling else ""
        lines.extend(["", "## "+label, "", "| 数据集 | 方法 | 完成 seed | 原始列 | 保留列 | 拟合次数 | "+heading+
                      "总秒 | AUC | 准确率 | AP | R² | MAE |", "|---|---|"+"---:|"*(2+len(displayed))])
        for r in subset:
            width = str(r["width_min"]) if r["width_min"] == r["width_max"] else f"{r['width_min']}–{r['width_max']}"
            fields = [r["dataset"], FORMAL_LABEL[r["method"]],f"{r['n_completed']}/{r['n_planned']}",width]
            fields.extend(_fmt(r,k) for k in displayed)
            lines.append("|"+"|".join(fields)+"|")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix="summary.", suffix=".tmp",delete=False) as out:
        out.write("\n".join(lines)+"\n")
    os.replace(out.name,root/"results.md")
    if not partial or xlsx:
        from experiments.formal_literature_excel import build
        build(root)
    return dict(states)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=DEFAULT)
    p.add_argument("--allow-pending",action="store_true")
    p.add_argument("--xlsx",action="store_true",help="Build and independently verify the Excel workbook after auditing the runs")
    args=p.parse_args()
    print(report(args.run, partial=args.allow_pending, xlsx=args.xlsx),flush=True)
