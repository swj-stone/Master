"""Score the marginal-ranking and unadjusted-width controls on the 331 wider cases."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from research.fast_single.selector import select_fast
from research.peer_crossover.summarize import EPS, table
from research.wide_blocks.panels import KEY, bench, cases, registry


CONTROLS = ("marginal", "unadjusted")


def raw_width(scores):
    largest = scores.max()
    if largest <= 0:
        return len(scores)
    z = scores / largest
    return min(len(z), int(np.ceil(z.sum()**2 / (z @ z))))


def top(scores, k):
    return np.sort(np.argsort(-scores, kind="stable")[:k])


def score_support(case, cols):
    pred, decision = case.predict(cols)
    score = float(case.score(pred, decision))
    if not np.isfinite(score):
        raise ValueError(f"Nonfinite score on {case.key}")
    return dict(k=len(cols), score=score, cols_sha=bench.digest(cols), pred_sha=bench.digest(pred),
                decision_sha=bench.digest(decision) if decision is not None else "")


def evaluate(case, reference):
    fast, info = select_fast(case.selection_X(), case.y, task=case.task, seed=case.seed)
    rank = info["rank"]
    np.testing.assert_array_equal(fast, top(rank, len(fast)))
    supports = dict(fast=fast, marginal=top(info["tier_scores"][0], len(fast)),
                    unadjusted=top(rank, raw_width(rank)))
    stats = {arm: score_support(case, cols) for arm, cols in supports.items()}
    for field, observed in stats["fast"].items():
        if observed != reference[field + "_fast"]:
            raise ValueError(f"Fast {field} differs from the paired benchmark on {case.key}")
    row = {key: reference[key] for key in ["case_id", *KEY, "source", "task", "metric", "p", "score_full"]}
    row["null_scale"] = info["null_scale"]
    for arm, outcome in stats.items():
        row.update({k+"_"+arm: v for k, v in outcome.items()})
    for arm in CONTROLS:
        cols = supports[arm]
        common = np.intersect1d(fast, cols, assume_unique=True).size
        row.update({"common_"+arm: common, "only_fast_"+arm: len(fast)-common,
                    "only_control_"+arm: len(cols)-common,
                    "same_support_"+arm: np.array_equal(cols, fast),
                    "score_delta_"+arm: stats["fast"]["score"]-stats[arm]["score"]})
    if sparse.issparse(case.X):
        matrix = case.X.tocsr()
        if not matrix.has_canonical_format:
            matrix = matrix.copy()
            matrix.sum_duplicates()
        counts = np.bincount(matrix.indices[matrix.data != 0], minlength=matrix.shape[1])
        uniform = case.select_random(len(fast))
        if bench.digest(uniform) != reference["cols_sha_random_fast"]:
            raise ValueError(f"Random support differs from the paired benchmark on {case.key}")
        row.update(train_nnz_full=counts.sum(), train_nnz_fast=counts[fast].sum(),
                   train_nnz_random=counts[uniform].sum())
    return row


def summary_row(frame, arm, scope, group):
    n = len(frame)
    delta = frame["score_delta_"+arm]
    k = frame["k_"+arm]
    changed = ~frame["same_support_"+arm]
    full_ok_fast = (frame.k_fast < frame.p) & (frame.score_fast >= frame.score_full-.02-EPS)
    full_ok_control = (k < frame.p) & (frame["score_"+arm] >= frame.score_full-.02-EPS)
    return dict(scope=scope, group=group, control=arm, n=n, changed=int(changed.sum()),
                win=int((delta>EPS).sum()), tie=int((delta.abs()<=EPS).sum()), loss=int((delta<-EPS).sum()),
                median_delta=delta.median() if frame.metric.nunique()==1 else np.nan,
                mean_delta=delta.mean() if frame.metric.nunique()==1 else np.nan,
                median_fast_k=frame.k_fast.median(), median_control_k=k.median(),
                median_columns_saved=(k-frame.k_fast).median(),
                median_fast_retained_pct=(100*frame.k_fast/frame.p).median(),
                median_control_retained_pct=(100*k/frame.p).median(),
                median_fast_only_when_changed=frame.loc[changed, "only_fast_"+arm].median(),
                median_control_only_when_changed=frame.loc[changed, "only_control_"+arm].median(),
                smaller_k=int((frame.k_fast<k).sum()), larger_k=int((frame.k_fast>k).sum()),
                cut_within_002_full_fast=int(full_ok_fast.sum()), cut_within_002_full_control=int(full_ok_control.sum()))


def summarize(frame):
    groups = [("all", "p>32", frame)]
    bins = pd.cut(frame.p, [32,128,512,2048,np.inf], labels=["33–128","129–512","513–2048",">2048"])
    groups += [("width", str(group), f) for group, f in frame.groupby(bins, observed=True)]
    if "train_nnz_full" in frame:
        groups.append(("format", "稀疏输入", frame[frame.train_nnz_full.notna()]))
    for keys, scope in [(["task","metric"], "metric"), (["cohort","dataset","task","metric"], "dataset")]:
        groups += [(scope, " / ".join(group), f) for group, f in frame.groupby(keys)]
    return pd.DataFrame([summary_row(f, arm, scope, group) for scope, group, f in groups for arm in CONTROLS])


def markdown(summary, frame, path):
    rows = []
    for r in summary[summary.scope.ne("dataset")].itertuples(index=False):
        rows.append([r.group, "边际同宽" if r.control=="marginal" else "原始权重",
                     r.n, r.changed, f"{r.win}/{r.tie}/{r.loss}",
                     f"{r.mean_delta:+.4f}" if np.isfinite(r.mean_delta) else "—",
                     f"{r.median_fast_retained_pct:.1f}/{r.median_control_retained_pct:.1f}",
                     f"{r.cut_within_002_full_fast}/{r.cut_within_002_full_control}"])
    sparsity = []
    if "train_nnz_full" in frame:
        sp = frame[frame.train_nnz_full.notna()]
        ratios = sp.train_nnz_fast / sp.train_nnz_random.replace(0, np.nan)
        defined = ratios.dropna()
        sparsity = [
            "## 稀疏训练矩阵", "",
            f"{len(sp)} 个稀疏配置中，候选列的非零项数多于同宽随机的有 {int(sp.train_nnz_fast.gt(sp.train_nnz_random).sum())} 个。"
            f"剔除随机子集中非零项为零的配置后，{len(defined)} 个配置的非零项数之比中位数为 {defined.median():.2f}。"
            "计数使用归并重复条目后的实际非零值。它描述下游模型读入的数据量，不能单独确定计时差异的原因。", ""]
    path.write_text("\n".join([
        "# 采样 Single-peer 的选列消融", "",
        "两项对照共用候选版抽取的训练行和同一个下游模型。“边际同宽”只按单个特征的边际收益排序，取候选版给出的 K 列；“原始权重”保留候选版的同伴投票排序，用未经噪声修正的投票得分计算 K。每个子集单独拟合、预测。候选版的列、预测、决策值、列数和得分均与两次计时后汇总的主实验记录核对。", "",
        "列集改变指候选版与对应对照保留的列不完全相同。胜/平/负是候选版预测得分相对该对照的配对计数，绝对差在 1e−12 内记平。平均差按候选版减对照计算；混合指标的组省略均值。末两列分别按候选版/对照列出结果。保留比例逐配置计算后取中位数，达标数要求实际删列且相对 Full 得分下降最多 0.02。本表只评估预测和选列。", "",
        table(["范围", "对照", "配置", "列集改变", "分数胜/平/负", "平均得分差",
               "保留 %", "删列且损失≤0.02"], rows), "",
        *sparsity, "[逐配置结果](ablation-cases.csv) · [分组与各数据集结果](ablation-groups.csv)", ""]
    ), encoding="utf-8")


def run(evidence, output):
    main = pd.read_csv(evidence, keep_default_na=False, float_precision="round_trip").set_index(KEY, drop=False)
    original = registry()
    if not main.index.equals(original.index) or not main.p.equals(original.p):
        raise ValueError("Main evidence must cover the entire registered paper panel")
    wanted = main[main.p.gt(32)]
    if len(wanted) != 331:
        raise ValueError("Expected 331 configurations with more than 32 original features")
    records = []
    for case in cases(wanted.index):
        records.append(evaluate(case, wanted.loc[case.key]))
        if len(records) % 25 == 0:
            print(f"Scored {len(records)}/{len(wanted)}", flush=True)
    frame = pd.DataFrame(records).sort_values("case_id")
    summary = summarize(frame)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/"ablation-cases.csv", index=False)
    summary.to_csv(output/"ablation-groups.csv", index=False)
    markdown(summary, frame, output/"ablation.md")
    print(f"Verified {len(frame)} fast results and scored both ablations in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_cases", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.paired_cases, args.out or args.paired_cases.parent)
