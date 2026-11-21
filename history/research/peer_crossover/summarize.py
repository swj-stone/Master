"""Summarize the paired single-peer/nested-peer replay on the paper's inputs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parents[1] / "lowdim/evidence/all-cases.csv"
ARMS = ("single", "nested", "full", "F_single", "random_single", "F_nested", "random_nested")
KEY = ["cohort", "dataset", "fold", "seed", "task", "metric", "p"]
STABLE = ["k", "score", "cols_sha", "pred_sha"]
EPS = 1e-12
TOLERANCES = {"0": 0, "002": .02, "005": .05, "010": .10}


def paired_cases(run_dir):
    archive = pd.read_csv(ARCHIVE, keep_default_na=False)
    archive.index.name = "case_id"
    rows = []
    for process in (1, 2):
        files = sorted((run_dir / f"pass{process}").glob("case-*.csv"))
        found = {int(path.stem.split("-")[-1]) for path in files}
        if found != set(archive.index) or len(files) != len(found):
            raise ValueError(f"Pass {process} has {len(found)} case IDs, expected {len(archive)}")
        for path in files:
            case_id = int(path.stem.split("-")[-1])
            data = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
            if set(data.arm) != set(ARMS) or set(data.process) != {process}:
                raise ValueError(f"Missing or unexpected arm/process in {path}")
            if not data[KEY].eq(archive.loc[case_id, KEY]).all().all():
                raise ValueError(f"Input configuration mismatch in {path}")
            if data.duplicated(["arm", "repeat"]).any():
                raise ValueError(f"Duplicate arm repetitions in {path}")
            batches = data.groupby("arm")
            if batches.size().nunique() != 1 or batches[STABLE].nunique().ne(1).any().any():
                raise ValueError(f"Repetitions disagree in {path}")
            values = data[["score", "select_ms", "total_ms", "k"]].to_numpy()
            if (not np.isfinite(values).all() or data.total_ms.le(0).any()
                    or data.select_ms.lt(0).any() or data.k.lt(1).any() or data.k.gt(data.p).any()):
                raise ValueError(f"Invalid score, width, or runtime in {path}")
            medians = batches[["select_ms", "total_ms"]].median().join(batches[STABLE].first())
            medians["process"] = process
            medians["case_id"] = case_id
            rows.append(medians.reset_index())
    per_pass = pd.concat(rows, ignore_index=True).set_index(["case_id", "arm", "process"])
    first, second = (per_pass.xs(i, level="process") for i in (1, 2))
    if not first[STABLE].equals(second[STABLE]):
        raise ValueError("The two processes disagree on width, score, selected columns, or predictions")
    records = first[STABLE].copy()
    for process, frame in enumerate((first, second), start=1):
        for stage in ("select_ms", "total_ms"):
            records[f"{stage}{process}"] = frame[stage]
    wide = records.unstack("arm")
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    cases = archive[[*KEY, "source", "family", "n_train", "n_test"]].join(wide)
    cases["width_group"] = np.where(cases.p <= 32, "p<=32", "p>32")
    for rule in ("single", "nested"):
        for baseline in ("F", "random"):
            if not cases[f"k_{rule}"].equals(cases[f"k_{baseline}_{rule}"]):
                raise ValueError(f"{baseline} does not match the width of {rule}")
    if not cases.k_full.eq(cases.p).all():
        raise ValueError("Full dropped columns")
    return cases


def compare(cases, arm, other):
    delta = cases[f"score_{arm}"] - cases[f"score_{other}"]
    ratios = pd.concat(
        [cases[f"total_ms{i}_{arm}"] / cases[f"total_ms{i}_{other}"] for i in (1, 2)], axis=1)
    return pd.DataFrame({
        "k": cases[f"k_{arm}"], "retained_pct": 100 * cases[f"k_{arm}"] / cases.p,
        "cut": cases[f"k_{arm}"] < cases.p, "delta": delta, "ratio": ratios.max(axis=1),
        "fast": ratios.lt(1).all(axis=1), "other_fast": ratios.gt(1).all(axis=1),
        "select_ms": (cases[f"select_ms1_{arm}"] + cases[f"select_ms2_{arm}"]) / 2,
        "total_ms": (cases[f"total_ms1_{arm}"] + cases[f"total_ms2_{arm}"]) / 2,
    })


def summarize(comparison):
    c = comparison
    faster_cut = c.cut & c.fast
    higher, lower = c.delta > EPS, c.delta < -EPS
    row = {
        "n": len(c), "cut": int(c.cut.sum()), "cut_faster": int(faster_cut.sum()),
        "faster": int(c.fast.sum()), "other_faster": int(c.other_fast.sum()),
        "inconsistent_or_tied_time": int((~c.fast & ~c.other_fast).sum()),
        "score_higher": int(higher.sum()), "score_tied": int((~higher & ~lower).sum()),
        "score_lower": int(lower.sum()), "higher_faster": int((higher & c.fast).sum()),
        "no_loss_faster": int((~lower & c.fast).sum()), "median_k": c.k.median(),
        "median_retained_pct": c.retained_pct.median(), "median_select_ms": c.select_ms.median(),
        "median_total_ms": c.total_ms.median(), "median_worst_time_ratio": c.ratio.median(),
    }
    for label, loss in TOLERANCES.items():
        row[f"joint_{label}"] = int((faster_cut & (c.delta >= -loss - EPS)).sum())
    return row


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join([*lines, *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def report(output, cells, policies, comparisons):
    def formatted(r):
        return [int(r.n), f"{r.median_retained_pct:.1f}", f"{r.median_select_ms:.3f}",
                f"{r.median_total_ms:.3f}", f"{r.median_worst_time_ratio:.3f}",
                *[int(r[key]) for key in ("joint_0", "joint_002", "joint_005", "joint_010", "cut_faster")]]

    headers = ["配置", "保留列 %", "选列 ms", "总耗时 ms", "对 Full 时间比", "损失 ≤0", "≤0.02",
               "≤0.05", "≤0.10", "不限损失"]
    main_cells = cells[cells.arm.isin(["single", "nested"])]
    by_arm = cells.set_index(["width_group", "arm"])
    baseline_rows = []
    for policy in policies.itertuples():
        values = [policy.low, policy.wide]
        for criterion in ("joint_0", "joint_002", "cut_faster"):
            for prefix in ("", "F_", "random_"):
                values.append(int(by_arm.at[("p<=32", prefix + policy.low), criterion]
                                  + by_arm.at[("p>32", prefix + policy.wide), criterion]))
        baseline_rows.append(values)
    interpretations = []
    for criterion, definition in [("joint_002", "允许损失 0.02"), ("cut_faster", "不设损失上限")]:
        maximum = policies[criterion].max()
        winners = policies.loc[policies[criterion] == maximum]
        routes = "、".join(f"{r.low}/{r.wide}" for _, r in winners.iterrows())
        interpretations.append(f"{definition}时，达标最多的搭配为 {routes}（{maximum}/{int(winners.n.iloc[0])}）。")
    lines = [
        "# Single-peer 与 Nested-peer 的交叉实验", "",
        "按论文原始特征数划分为两个区间。每个配置使用相同训练集和测试集评估两种规则，"
        "两种规则独立决定各自保留的列数。时间包括选列、提取子矩阵、训练和预测。", "",
        "## 按输入维度和规则", "",
        "最后五列为达标配置数。每个配置须删列，两次独立进程中均比 Full 快，"
        "且主测试得分下降在指定范围内。时间比取每个配置两次配对比值中较大者再求组中位数。"
        "毫秒数先平均每个配置在两次进程中的耗时中位数，再取跨配置中位数。", "",
        table(["维度", "规则", *headers],
              ([r.width_group, r.arm, *formatted(r)] for _, r in main_cells.iterrows())), "",
        "## 四种固定搭配", "",
        "按同一套结果分别计算四种路由。每行低维和较高维各用一种固定规则。", "",
        table(["低维规则", "较高维规则", *headers],
              ([r.low, r.wide, *formatted(r)] for _, r in policies.iterrows())), "",
        "".join(interpretations) + "此处按原有数据上的达标配置数排序，两种规则可选择不同列数。", "",
        "## 与同宽 F、随机选列比较", "",
        "每项均以 Full 为参照，分母均为 387。F 和随机逐配置使用该行规则选出的列数。全留计为失败。", "",
        table(["低维规则", "较高维规则", "零损失·本方法", "零损失·F", "零损失·随机",
               "≤0.02·本方法", "≤0.02·F", "≤0.02·随机", "不限·本方法", "不限·F", "不限·随机"],
              baseline_rows), "",
        "## 同一配置内直接比较", "",
        "速度胜场要求两次进程中均更快。分数胜、平、负按本配置的主指标计数，数值平分容差为 10⁻¹²。"
        "F 和随机对照分别使用被比较规则选定的列数。", "",
        table(["维度", "规则", "对照", "配置", "分数胜/平/负", "两次更快", "两次更慢", "更快且更高分"],
              ([r.width_group, r.arm, r.other, int(r.n),
                f"{int(r.score_higher)}/{int(r.score_tied)}/{int(r.score_lower)}", int(r.faster),
                int(r.other_faster), int(r.higher_faster)] for _, r in comparisons.iterrows())), "",
        "按原数据任务、指标和协议分组的逐项比较见 [by-dataset.csv](by-dataset.csv)，按预测任务与指标汇总见 "
        "[by-metric.csv](by-metric.csv)。其中分差等于本行规则得分减对照得分，分差只在同类指标内汇总。"
        "所有规则及同宽基线相对 Full 的结果见 [width-groups.csv](width-groups.csv)，"
        "原始两次进程的配对中位数见 [paired-cases.csv](paired-cases.csv)。"
        "同一数据来源可贡献多个配置，各行是这组基准的描述性结果。", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="Default: summary directory under the run")
    args = parser.parse_args()
    cases = paired_cases(args.run_dir)
    out = args.out or args.run_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out / "paired-cases.csv")

    cells, direct, by_source, by_metric, policies = [], [], [], [], []
    for width, group in cases.groupby("width_group", sort=True):
        for arm in ARMS:
            if arm != "full":
                cells.append({"width_group": width, "arm": arm, **summarize(compare(group, arm, "full"))})
        for arm in ("single", "nested"):
            others = (["nested"] if arm == "single" else []) + [f"F_{arm}", f"random_{arm}"]
            for other in others:
                result = compare(group, arm, other)
                direct.append({"width_group": width, "arm": arm, "other": other, **summarize(result)})
            for other in ["full", *others]:
                result = compare(group, arm, other)
                for grouping, output in [(["cohort", "source", "dataset", "task", "metric"], by_source),
                                         (["task", "metric"], by_metric)]:
                    for key, grp in group.groupby(grouping, sort=True):
                        sub = result.loc[grp.index]
                        output.append({**dict(zip(grouping, key)), "width_group": width,
                                       "p_min": grp.p.min(), "p_max": grp.p.max(),
                                       "arm": arm, "other": other, "mean_score_delta": sub.delta.mean(),
                                       "median_score_delta": sub.delta.median(), **summarize(sub)})
    low = cases.p <= 32
    for low_rule in ("single", "nested"):
        for wide_rule in ("single", "nested"):
            merged = pd.concat([compare(cases.loc[low], low_rule, "full"),
                                compare(cases.loc[~low], wide_rule, "full")])
            policies.append({"low": low_rule, "wide": wide_rule, **summarize(merged)})
    frames = [pd.DataFrame(records) for records in (cells, policies, direct, by_source, by_metric)]
    for name, frame in zip(("width-groups", "routes", "head-to-head", "by-dataset", "by-metric"), frames):
        frame.to_csv(out / f"{name}.csv", index=False)
    report(out, *frames[:3])
    print(f"Wrote paired results for {len(cases)} configurations to {out}")


if __name__ == "__main__":
    main()
