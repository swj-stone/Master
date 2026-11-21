"""Validate both timing passes, then compare the sampled and exact Single-peer rules."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.peer_crossover.summarize import EPS, KEY, compare, summarize, table


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "peer_crossover/evidence/paired-cases.csv"
INPUTS = HERE.parent / "lowdim/evidence/all-cases.csv"
HASHES = ["train_sha", "target_sha", "test_sha", "truth_sha"]
RULES = ("fast", "single")
ARMS = ("fast", "single", "full", "F_fast", "random_fast", "F_single", "random_single")
STABLE = ["k", "score", "cols_sha", "pred_sha", "decision_sha"]
REF_ARMS = ("single", "full", "F_single", "random_single")
PAIRS = [(arm, "full") for arm in ARMS if arm != "full"] + [
    (rule, f"{base}_{rule}") for rule in RULES for base in ("F", "random")
] + [("fast", "single")]
NUMBERS = {"fast": "候选版", "single": "论文版", "full": "Full",
           "F_fast": "候选同宽 F", "F_single": "论文同宽 F",
           "random_fast": "候选同宽随机", "random_single": "论文同宽随机"}


def read_case(path, case_id, process, expected, repeats):
    rows = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    if (set(rows.arm) != set(ARMS) or not rows.case_id.eq(case_id).all()
            or not rows.process.eq(process).all() or not rows[KEY].eq(expected[KEY]).all().all()):
        raise ValueError(f"Unregistered case, arm or pass in {path}")
    counts = rows.groupby(["arm", "repeat"]).size()
    if set(rows.repeat) != set(range(repeats)) or len(counts) != len(ARMS)*repeats or counts.ne(1).any():
        raise ValueError(f"Missing or repeated measurements in {path}")
    finite = np.isfinite(rows[["score", "select_ms", "total_ms", "k"]]).all().all()
    if (not finite or not rows.k.between(1, rows.p).all() or not rows.k.eq(rows.k.astype(int)).all()
            or rows.select_ms.lt(0).any() or rows.total_ms.le(0).any()
            or rows.select_ms.gt(rows.total_ms).any() or rows.cols_sha.eq("").any()
            or rows.pred_sha.eq("").any()):
        raise ValueError(f"Invalid width, score, timing or output fingerprint in {path}")
    arms = rows.groupby("arm")
    if arms[STABLE].nunique().ne(1).any().any():
        raise ValueError(f"Non-reproducible outcomes within process in {path}")
    first = arms[STABLE].first()
    if first.at["full", "k"] != expected.p:
        raise ValueError(f"Full dropped input columns in {path}")
    for arm in REF_ARMS:
        for field in ("cols_sha", "pred_sha", "k"):
            if first.at[arm, field] != expected[f"{field}_{arm}"]:
                raise ValueError(f"Archived {arm} {field} differs in {path}")
        if abs(first.at[arm, "score"] - expected[f"score_{arm}"]) > 2e-12:
            raise ValueError(f"Archived {arm} score differs in {path}")
    for rule in RULES:
        if any(first.at[arm, "k"] != first.at[rule, "k"]
               for arm in (f"F_{rule}", f"random_{rule}")):
            raise ValueError(f"Width mismatch for {rule} in {path}")
    aliases = {arm: arm for arm in ARMS}
    if first.at["fast", "k"] == first.at["single", "k"]:
        aliases.update(F_single="F_fast", random_single="random_fast")
    if not rows.timed_arm.eq(rows.arm.map(aliases)).all():
        raise ValueError(f"Incorrect shared timing for a width-matched control in {path}")
    same_run = rows.groupby(["timed_arm", "repeat"])[[*STABLE, "position", "select_ms", "total_ms"]]
    if same_run.nunique().ne(1).any().any():
        raise ValueError(f"Duplicated control labels disagree in {path}")
    schedule = rows.drop_duplicates(["timed_arm", "repeat"])
    if (schedule.duplicated(["repeat", "position"]).any()
            or set(schedule.position) != set(range(len(set(aliases.values()))))):
        raise ValueError(f"Invalid measurement order in {path}")
    return first.join(arms[["select_ms", "total_ms"]].median()).assign(case_id=case_id)


def pair_cases(run):
    paper = pd.read_csv(PAPER, keep_default_na=False, float_precision="round_trip").set_index("case_id")
    protocol = json.loads((run/"protocol.json").read_text(encoding="utf8"))
    repeats = protocol["repeats"]
    if repeats < 3 or protocol["threads"] != 1 or set(protocol["arms"]) != set(ARMS):
        raise ValueError("Unexpected benchmark protocol")
    runs = []
    for process in (1, 2):
        folder = run / f"pass{process}"
        files = list(folder.glob("case-*.csv"))
        paths = {int(p.stem.split("-")[1]): p for p in files}
        if set(paths) != set(paper.index) or len(files) != len(paths):
            raise ValueError(f"Incomplete pass {process}, found {len(paths)} of {len(paper)} cases")
        records = [read_case(paths[i], i, process, row, repeats) for i, row in paper.iterrows()]
        runs.append(pd.concat(records).reset_index().set_index(["case_id", "arm"]).sort_index())
    a, b = runs
    if not a[STABLE].equals(b[STABLE]):
        raise ValueError("The two processes disagree on widths, scores or outputs")
    result = a[STABLE].copy()
    for i, frame in enumerate(runs, 1):
        for field in ("select_ms", "total_ms"):
            result[f"{field}{i}"] = frame[field]
    wide = result.unstack()
    wide.columns = [f"{field}_{arm}" for field, arm in wide.columns]
    inputs = pd.read_csv(INPUTS, usecols=KEY+HASHES, keep_default_na=False)
    inputs = paper[KEY].merge(inputs, on=KEY, validate="one_to_one", how="left").set_axis(paper.index)
    if inputs[HASHES].isna().any().any() or not all(
            inputs[h].str.fullmatch("[a-fA-F0-9]{64}").all() for h in HASHES):
        raise ValueError("Invalid archived input fingerprints")
    return paper[[*KEY, "source", "family", "n_train", "n_test"]].join(inputs[HASHES]).join(wide)


def group_masks(cases):
    return {"全部": cases.p.gt(0), "≤32": cases.p.le(32), ">32": cases.p.gt(32),
            "33–128": cases.p.between(33, 128), "129–512": cases.p.between(129, 512),
            "513–2048": cases.p.between(513, 2048), ">2048": cases.p.gt(2048),
            "稀疏文本与广告": cases.cohort.eq("text")}


def metrics(c, detail, one_metric):
    result = summarize(c)
    for key, threshold in [("100", 1.), ("105", 1.05), ("110", 1.10), ("120", 1.20)]:
        near = c.ratio.le(threshold) & c.cut
        result["cut_ratio_le_"+key] = int(near.sum())
        result["cut_ratio_le_"+key+"_full_loss_002"] = int((near & detail.full_002).sum())
        result["joint_full_002_ratio_le_"+key] = int((near & detail.full_joint_002).sum())
    result["smaller_k_than_comparator"] = int((c.k < detail.other_k).sum())
    result["equal_k_to_comparator"] = int((c.k == detail.other_k).sum())
    result["same_selected_columns"] = int(detail.same_support.sum())
    result["median_conservative_select_ratio"] = detail.select_ratio.median()
    result["cut_faster_vs_comparator_within002_full"] = int((c.cut & c.fast & detail.full_002).sum())
    if one_metric:
        result["mean_loss_vs_comparator"] = -c.delta.mean()
        result["median_loss_vs_comparator"] = -c.delta.median()
        result["p90_loss_vs_comparator"] = (-c.delta).quantile(.90)
    return result


def summaries(cases):
    widths, datasets, task_metrics = [], [], []
    groups = group_masks(cases)
    dataset_keys = ["cohort", "source", "dataset", "task", "metric"]
    for arm, other in PAIRS:
        c = compare(cases, arm, other)
        full = compare(cases, arm, "full")
        loss_ok = full.delta.ge(-.02-EPS)
        ratios = [cases[f"select_ms{i}_{arm}"] /
                  cases[f"select_ms{i}_{other}"].replace(0, np.nan) for i in (1, 2)]
        details = pd.DataFrame({"other_k": cases[f"k_{other}"], "full_002": loss_ok,
                                "full_joint_002": full.cut & full.fast & loss_ok,
                                "same_support": cases[f"cols_sha_{arm}"].eq(cases[f"cols_sha_{other}"]),
                                "select_ratio": np.maximum(*ratios)})
        label = {"arm": arm, "other": other}
        for name, mask in groups.items():
            if mask.any():
                widths.append({"scope": name, **label, **metrics(c.loc[mask], details.loc[mask], False)})
        for cols, target in ((dataset_keys, datasets), (["task", "metric"], task_metrics)):
            for key, idx in cases.groupby(cols).groups.items():
                target.append({**dict(zip(cols, key)), **label, **metrics(c.loc[idx], details.loc[idx], True)})
    return tuple(map(pd.DataFrame, (widths, datasets, task_metrics)))


def input_weighted(cases):
    rows = []
    for scope in ("全部", ">32"):
        c = cases.loc[group_masks(cases)[scope]]
        for rule in RULES:
            full = compare(c, rule, "full")
            random = compare(c, rule, f"random_{rule}")
            faster = full.cut & full.fast
            joint = faster & full.delta.ge(-.02-EPS)
            near = full.cut & random.ratio.le(1.10)
            flags = c[HASHES].assign(cut_faster=faster, joint_002=joint,
                                     faster_random=full.cut & random.fast,
                                     near_random=near, both=joint & near)
            units = flags.groupby(HASHES).mean()
            rows.append([scope, NUMBERS[rule], len(units), *[f"{100*x:.1f}" for x in units.mean()]])
    return rows


def markdown(widths, task_metrics, weighted, output):
    grouped = widths.set_index(["scope", "arm", "other"])
    scoped = ["全部", "≤32", ">32", "33–128", "129–512", "513–2048", ">2048", "稀疏文本与广告"]
    overview = []
    for group in scoped:
        for rule in RULES:
            row = grouped.loc[(group, rule, "full")]
            random = grouped.loc[(group, rule, f"random_{rule}")]
            overview.append([group, NUMBERS[rule], int(row.n), f"{row.median_retained_pct:.1f}",
                             f"{row.median_select_ms:.3f}", f"{row.median_total_ms:.3f}",
                             int(row.cut_faster), int(row.joint_0), int(row.joint_002),
                             f"{random.median_worst_time_ratio:.2f}"])
    speed = []
    for group in ["全部", "≤32", ">32", "稀疏文本与广告"]:
        for rule in RULES:
            row = grouped.loc[(group, rule, f"random_{rule}")]
            speed.append([group, NUMBERS[rule], int(row.n), int(row.cut_faster), int(row.cut_ratio_le_105),
                          int(row.cut_ratio_le_110), int(row.cut_ratio_le_120), int(row.joint_full_002_ratio_le_110),
                          f"{int(row.score_higher)}/{int(row.score_tied)}/{int(row.score_lower)}"])
    wide = grouped.loc[(">32", "fast", "single")]
    main = grouped.loc[("全部", "fast", "single")]
    small = grouped.loc[("≤32", "fast", "single")]
    main_rows = []
    for arm in ARMS:
        if arm != "full":
            row = grouped.loc[("全部", arm, "full")]
            main_rows.append([NUMBERS[arm], int(row.cut), int(row.cut_faster), int(row.joint_0), int(row.joint_002),
                              int(row.joint_005), int(row.joint_010), f"{row.median_worst_time_ratio:.2f}"])
    scores = []
    for r in task_metrics[task_metrics.arm.eq("fast") & task_metrics.other.isin(["full", "single"])].itertuples():
        scores.append([r.task, r.metric, NUMBERS[r.other], r.n, f"{r.median_loss_vs_comparator:.4f}",
                       f"{r.p90_loss_vs_comparator:.4f}", f"{r.score_higher}/{r.score_tied}/{r.score_lower}"])
    lines = [
        "# 候选 Single-peer 的配对实验", "",
        "每个配置使用同一训练集、测试集、种子和下游模型。每进程先取重复测量的中位数。绝对耗时为两个进程中位数的平均值，再跨配置取中位数。时间比先取单个配置中两个配对比值的较大者，再跨配置取中位数。保留比例逐配置按保留列数除以原始列数计算。", "",
        "## 保留列数、选列和预测", "",
        "末四列依次统计删列且快于 Full 的次数、再要求得分不降的次数、允许下降 0.02 的次数，以及与该方法同宽随机的总时间比。0.02 是描述性容差。稀疏文本与广告单独列出，和按列数划分的区间有重叠。配置包含同一数据集上的不同种子或分割。", "",
        table(["维度", "方法", "配置", "保留 %", "选列 ms", "端到端 ms", "删列提速", "损失≤0",
               "损失≤0.02", "对同宽随机时间比"], overview), "",
        "## 全部方法相对 Full", "",
        table(["方法", "删列", "删列提速", "损失≤0", "≤0.02", "≤0.05", "≤0.10", "时间比"],
              main_rows), "",
        "## 同宽随机的速度与精度", "",
        "所有速度计数均要求删列。严格更快指两个进程都少于随机的时间。允许 5%、10%、20% 开销的三列为累积数，含实际更快的配置。联合列还要求两个进程都快于 Full，且相对 Full 的得分损失最多为 0.02。末列为本方法对同宽随机的得分胜/平/负。", "",
        table(["维度", "方法", "配置", "严格更快", "≤1.05倍", "≤1.10倍", "≤1.20倍", "≤1.10倍且对 Full 三项达标", "分数胜/平/负"], speed), "",
        "## 按不同输入等权", "",
        "训练输入、训练标签、测试输入、测试标签的四个指纹相同即归为一组。先算每组内的达标比例，再对每组等权平均。下表列出百分比，最后一列要求同时满足 Full 三项标准和同宽随机 1.10 倍时间限制。", "",
        table(["维度", "方法", "不同输入", "删列快于 Full %", "且损失≤0.02 %", "严格快于随机 %", "≤1.10倍随机 %", "同时满足 %"], weighted), "",
        "## 与论文版直接比较", "",
        f"全部 {int(main.n)} 组中，候选版在 {int(main.smaller_k_than_comparator)} 组保留更少列，在 {int(main.equal_k_to_comparator)} 组保留相同列数。"
        f"候选版的得分有 {int(main.score_higher)} 胜、{int(main.score_tied)} 平、{int(main.score_lower)} 负，两进程都更快的有 {int(main.faster)} 组。"
        f"在 {int(wide.n)} 组较高维配置中，相应得分为 {int(wide.score_higher)} 胜、{int(wide.score_tied)} 平、{int(wide.score_lower)} 负，两进程都更快的有 {int(wide.faster)} 组。"
        f"在最多 32 列的 {int(small.n)} 组中，候选版沿用论文规则，{int(small.same_selected_columns)} 组选列完全相同。该区间的计时差异包括测量波动。", "",
        "## 各指标预测损失", "",
        "正值表示候选得分低于对照。分位数在该任务与指标的全部配置内计算。", "",
        table(["任务", "指标", "对照", "配置", "损失中位数", "损失90分位", "分数胜/平/负"], scores), "",
        "[逐配置计时、列数及得分](paired-cases.csv) · [按数据集和实验协议比较](by-dataset.csv) · [按维度比较](by-width.csv) · [按任务指标比较](by-metric.csv)", "",
    ]
    (output/"summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = pair_cases(args.run_dir)
    widths, datasets, task_metrics = summaries(cases)
    output = args.out or (args.run_dir/"summary")
    output.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output/"paired-cases.csv")
    for filename, frame in [("by-width", widths), ("by-dataset", datasets), ("by-metric", task_metrics)]:
        frame.to_csv(output/(filename + ".csv"), index=False)
    markdown(widths, task_metrics, input_weighted(cases), output)
    print(f"Validated and summarized {len(cases)} configurations in {output}")


if __name__ == "__main__":
    main()
