"""Audit each seed-42 outcome and publish the full raw-data PIv3 paired comparison."""

import argparse
import csv
import json
import math
from collections import Counter
from io import StringIO
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile


WIDTHS = dict(abalone=8, credit_card_fraud=29, gisette=5000, isolet=617, madelon=500, spambase=57)
NAMES = dict(banzhaf="采样 Banzhaf", chi2="分箱卡方", extra_trees="极端随机树", f_score="F 分数",
             mrmr="mRMR（MID）", mutual_info="互信息", random_forest="随机森林", rfe_linear="线性 RFE",
             spearman="Spearman 秩相关", tree_shap="TreeSHAP", variance="方差")
TOLERANCES = (.03, .05, .08)
ARMS = ("plain", "piv3")
STATES = dict(completed="完成", timeout="超时", pending="待出结果")
MEASURES = ("validation", "reference", "test", "test_reference")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pass(scores, reference, tolerance, regression):
    for i, (score, base) in enumerate(zip(scores, reference, strict=True)):
        delta = tolerance * abs(base) + 1e-12
        if regression and i == 1:
            if score > base + delta:
                return False
        elif score < base - delta:
            return False
    return True


def _read_arm(path, dataset, method, tolerance, arm):
    if not path.is_file():
        return {"status": "pending"}
    data = json.loads(path.read_bytes())

    def check(value, issue):
        if not value:
            raise ValueError(f"{path}: {issue}")

    width = WIDTHS[dataset]
    regression = dataset == "abalone"
    check((data["dataset"], data["method"], data["arm"], data["tolerance"], data["width"]) ==
          (dataset, method, arm, tolerance, width), "outcome does not match scheduled arm")
    complete = data["status"] == "completed"
    check(complete or data["status"] == "timeout", "unexpected terminal status")
    check(_finite(data["wall_s"]) and data["wall_s"] > 0, "invalid elapsed time")
    for key in ("front_k", "k", "fits"):
        v = data.get(key)
        if v is not None:
            check(type(v) is int and (v >= 0 if key == "fits" else 1 <= v <= width), f"invalid {key}")
    if complete:
        check(data["k"] is not None and data["fits"] is not None, "missing completed k or fits")
        check(data["wall_s"] <= 900 + 1e-6, "completed after deadline")
    if data.get("front_k") is not None and data.get("k") is not None:
        check(data["k"] <= data["front_k"], "backend retained more columns than it received")
    if data.get("front_s") is not None:
        check(_finite(data["front_s"]) and 0 <= data["front_s"] <= data["wall_s"] + 1e-6,
              "invalid prefilter duration")
    for phase in MEASURES:
        vals = data.get(f"{phase}_scores")
        if complete or vals is not None:
            check(isinstance(vals, list) and len(vals) == 2 and all(_finite(v) for v in vals),
                  f"invalid {phase} scores")
            check(0 <= vals[0] <= 1 and vals[1] >= 0 and (regression or vals[1] <= 1),
                  f"{phase} metrics out of range")
    if complete:
        check(_pass(data["validation_scores"], data["reference_scores"], tolerance, regression),
              "selected model failed validation tolerances")
        check(type(data.get("test_pass")) is bool and data["test_pass"] ==
              _pass(data["test_scores"], data["test_reference_scores"], tolerance, regression),
              "reported test pass disagrees with held-out predictions' metrics")
    else:
        check(data.get("test_pass") is None and data.get("test_scores") is None,
              "timeout has an unevaluated test result")
    return data


def collect(run):
    rows = []
    references = {}
    for dataset, width in sorted(WIDTHS.items()):
        regression = dataset == "abalone"
        for tol in TOLERANCES:
            for method in sorted(NAMES):
                row = dict(dataset=dataset, tolerance=tol, method=method, width=width,
                           metric_1="R2" if regression else "AUC", metric_2="MAE" if regression else "accuracy")
                for arm in ARMS:
                    rel = Path("arms") / dataset / method / arm / f"tol-{round(100*tol):02d}" / "outcome.json"
                    data = _read_arm(run / rel, dataset, method, tol, arm)
                    for key in ("status", "k", "wall_s", "fits", "front_k", "front_s", "test_pass"):
                        row[f"{arm}_{key}"] = data.get(key)
                    row[f"{arm}_phase"] = data.get("phase") or data.get("timeout_phase")
                    row[f"{arm}_source"] = rel.as_posix() if data["status"] != "pending" else None
                    for phase in MEASURES:
                        scores = data.get(f"{phase}_scores")
                        for i in (1, 2):
                            row[f"{arm}_{phase}_{i}"] = scores[i-1] if scores is not None else None
                        if phase in ("reference", "test_reference") and scores is not None:
                            previous = references.setdefault((dataset, phase), scores)
                            if any(not math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)
                                   for a, b in zip(scores, previous, strict=True)):
                                raise ValueError(f"{rel}: full-width {phase} differs across arms")
                paired = all(row[f"{a}_status"] == "completed" for a in ARMS)
                row["both_completed"] = paired
                row["speedup_plain_over_piv3"] = row["plain_wall_s"] / row["piv3_wall_s"] if paired else None
                rows.append(row)
    return rows


def _group(dataset, tol, group):
    out = dict(dataset=dataset, tolerance=tol, scheduled_pairs=len(group))
    for arm in ARMS:
        counts = Counter(r[f"{arm}_status"] for r in group)
        for status in STATES:
            out[f"{arm}_{status}"] = counts[status]
        out[f"{arm}_test_passes"] = sum(r[f"{arm}_test_pass"] is True for r in group)
    paired = [r for r in group if r["both_completed"]]
    speeds = [r["speedup_plain_over_piv3"] for r in paired]
    out.update(both_completed=len(paired), faster=sum(v > 1 for v in speeds),
               median_speedup=median(speeds) if speeds else None,
               fewer_features=sum(r["piv3_k"] < r["plain_k"] for r in paired))
    return out


def _csv(rows):
    text = StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue()


def _fmt(value, spec):
    return "—" if value is None else format(value, spec)


def _side_by_side(row, key, spec):
    return " / ".join(_fmt(row[f"{arm}_{key}"], spec) for arm in ARMS)


def _status(row, arm):
    result = row[f"{arm}_status"]
    if result == "timeout" and row[f"{arm}_phase"]:
        return f"超时（{str(row[f'{arm}_phase']).replace('|', '/')}）"
    return STATES[result]


def _markdown(rows, groups):
    overall = _group("全部", None, rows)
    lines = ["# PIv3 前置与 11 种原方法的配对结果", "",
             "六个原始数据集，数据 seed 42，验证集容差 3%、5%、8%。共 198 个配对，每条路线各有 900 秒预算。"
             "验证集同时检查两项指标，以同一数据集的全宽模型为基准。分类的 AUC 与准确率、回归的 R² 允许相对下降至给定容差，"
             "回归的 MAE 允许相对上升至该容差。R² 沿用项目的零下限。测试列展示冻结所选特征后的结果。", "",
             "耗时（秒）累计每条路线实际使用的独立阶段，包括全宽基准、PIv3 前置、后端排序与候选验证、最终预测。"
             "多条路线复用相同阶段时，每条都计入该阶段的完整实测耗时。", "",
             "## 进度与配对汇总", "",
             "每格前后依次为原方法 / PIv3＋原方法。测试达标数以对应路线已完成数为分母。"
             "加速比为原方法总秒数除以加前置总秒数。中位数和加速对数仅统计两侧完成的配对。", "",
             "| 数据集 | 容差 | 完成 | 超时 | 待出结果 | 双侧完成 | PIv3 加速对数 | 加速比中位数 | 测试双指标达标 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in [overall, *groups]:
        tol = "全部" if g["tolerance"] is None else f"{g['tolerance']:.0%}"
        cells = [g["dataset"], tol, *[_side_by_side(g, k, "d") for k in STATES], str(g["both_completed"]),
                 str(g["faster"]), _fmt(g["median_speedup"], ".2f"), _side_by_side(g, "test_passes", "d")]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## 完整配对", "", "同一格内沿用原方法 / PIv3＋原方法的顺序。前置列数和前置秒数仅对应 PIv3。"
                  "表中“—”表示该项尚无可报告的测量值。每对结果来自一次固定划分。"])
    for dataset, width in sorted(WIDTHS.items()):
        m1, m2 = ("R²", "MAE") if dataset == "abalone" else ("AUC", "准确率")
        lines.extend(["", f"### {dataset}（{width:,} 列）", "",
                      f"| 容差 | 方法 | 状态 | 保留列 | 前置留列 / 秒 | 拟合次数 | 总耗时（秒） | 加速比 | 测试 {m1} | 测试 {m2} | 测试达标 |",
                      "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for r in (r for r in rows if r["dataset"] == dataset):
            passes = " / ".join({True:"是", False:"否", None:"—"}[r[f"{a}_test_pass"]] for a in ARMS)
            cells = [f"{r['tolerance']:.0%}", NAMES[r["method"]], " / ".join(_status(r, a) for a in ARMS),
                     _side_by_side(r, "k", "d"), f"{_fmt(r['piv3_front_k'], 'd')} / {_fmt(r['piv3_front_s'], '.2f')}",
                     _side_by_side(r, "fits", "d"), _side_by_side(r, "wall_s", ".2f"),
                     _fmt(r["speedup_plain_over_piv3"], ".2f"), _side_by_side(r, "test_1", ".5f"),
                     _side_by_side(r, "test_2", ".5f" if dataset == "abalone" else ".2%"), passes]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _publish(path, content, encoding):
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".", suffix=".tmp",
                            encoding=encoding, newline="", delete=False) as handle:
        try:
            handle.write(content)
        except BaseException:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)
            raise
    tmp = Path(handle.name)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def report(run, *, require_complete=False):
    run = Path(run)
    if not run.is_dir():
        raise ValueError(f"Run directory does not exist: {run}")
    rows = collect(run)
    overall = _group("all", None, rows)
    pending = sum(overall[f"{arm}_pending"] for arm in ARMS)
    if require_complete and pending:
        raise ValueError(f"{pending} of {2*len(rows)} scheduled arms are pending")
    groups = [_group(d, t, [r for r in rows if r["dataset"] == d and r["tolerance"] == t])
              for d in sorted(WIDTHS) for t in TOLERANCES]
    _publish(run/"paired.csv", _csv(rows), "utf-8-sig")
    _publish(run/"group-summary.csv", _csv(groups), "utf-8-sig")
    _publish(run/"paired.md", _markdown(rows, groups), "utf-8")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--require-complete", action="store_true", help="Require a completed or timeout outcome for every arm")
    args = parser.parse_args()
    print(json.dumps(report(args.run, require_complete=args.require_complete), allow_nan=False))
