"""Render the audited seed-42 PIv2 pilot as one paired comparison table."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from minppi_mofs.experiment_protocol import canonical_hash


DATASETS = {
    "madelon": "Madelon", "spambase": "Spambase", "gisette": "Gisette",
    "credit_card_fraud": "Credit Card Fraud", "isolet": "ISOLET", "abalone": "Abalone",
}
METHODS = {
    "gain": "Gain 全前缀", "pfi": "子采样 PFI", "f_score": "F 分数", "mutual_info": "互信息",
    "tree_shap": "TreeSHAP", "l1": "L1", "banzhaf": "采样 Banzhaf",
    "exact": "精确枚举", "mc32": "采样 Shapley",
}


def load_rows(run):
    protocol = json.loads((run / "protocol.json").read_bytes())
    audit = json.loads((run / "audit.json").read_bytes())
    summary = run / "summary.csv"
    if (audit["status"] != "pass" or protocol["seed"] != 42 or audit["completed_pairs"] != 44
            or audit["scheduled_pairs"] != 44 or audit["planned_arms"] != 88 or audit["counts"]["completed"] != 88
            or canonical_hash(protocol) != audit["protocol_sha256"]
            or hashlib.sha256(summary.read_bytes()).hexdigest() != audit["summary_file_sha256"]
            or hashlib.sha256((run/"paired.csv").read_bytes()).hexdigest() != audit["paired_file_sha256"]):
        raise ValueError("The seed-42 report requires the audited protocol, 44 completed pairs and exact summary bytes")
    with summary.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    arms = {(row["dataset"], row["backend"], row["arm"]): row for row in rows}
    expected = {(dataset, method, arm) for dataset in DATASETS for method in METHODS
                if dataset == "abalone" or method not in {"exact", "mc32"} for arm in ("plain", "piv2")}
    if (set(arms) != expected or set(map(tuple, protocol["tasks"])) != expected or len(rows) != 88
            or any(row["status"] != "completed" or row["seed"] != "42" for row in rows)):
        raise ValueError("The protocol and completed results must cover all 44 unique pairs")
    return arms, protocol


def render(run):
    arms, protocol = load_rows(run)
    params = protocol["model_params"]
    lines = [
        "# PIv2 前置筛选的 seed 42 配对测试",
        "",
        f"六个预处理数据集使用冻结的训练、验证和测试划分。测试开始前固定了前置规则、后端参数和运行清单。"
        f"每个组合独立冷启动一次，每臂最多 {protocol['arm_limit_s'] // 60} 分钟。相邻两臂按预设顺序交替运行。"
        f"各后端用相同的 XGBoost 预测器评估（{protocol['hardware_backend'].upper()}，"
        f"{params['n_estimators']} 轮，树深 {params['max_depth']}，学习率 {params['learning_rate']}）。",
        "",
        "PIv2 在最多 1,024 条训练行上交叉拟合分箱预测，比较特征在小、中、大随机联盟中的相对效用变化。"
        "同一层在两折都投出净负票且三层最低净票率为负时，该列才成为删除候选。"
        "前置保护单列得分排名最靠前的 ⌈√d⌉ 列，最多删除一半输入列。选列冻结后才读取测试集。",
        "",
        "箭头左侧为原后端，右侧为同一后端加 PIv2。前置列数列给出原始输入宽度及 PIv2 交给后端的列数。"
        "前置时间已经计入总耗时。总耗时覆盖进程启动、加载与校验数据、前置、后端筛选、最终重训、测试预测、写结果和进程退出。"
        "节时率为 1 减去两臂总耗时之比（加 PIv2 除以原法），负值表示加 PIv2 后更慢。",
        "",
        "分类主分数为测试 ROC-AUC，ISOLET 用宏平均 OvR ROC-AUC。分类副指标为准确率。"
        "Abalone 的主分数为测试 R²（下限 0），副指标为 MAE（圈，越低越好）。"
        "每个数据集和方法仅作一次计时，小差值未估计计时方差。",
        "",
        "| 数据集 | 后端 | 输入→前置列数 | 前置秒 | 总耗时秒 原法→+PI | 节时率 | 最终列数 原法→+PI | 测试 AUC／R² 原法→+PI | 准确率／MAE 原法→+PI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, dataset_name in DATASETS.items():
        for method, method_name in METHODS.items():
            if (dataset, method, "plain") not in arms:
                continue
            direct = arms[dataset, method, "plain"]
            pi = arms[dataset, method, "piv2"]
            seconds, pi_seconds = float(direct["wall_s"]), float(pi["wall_s"])
            saving = 1 - pi_seconds / seconds
            secondary = [float(r["test_secondary"]) for r in (direct, pi)]
            secondary_text = " → ".join(f"{v:.4f}" if dataset == "abalone" else f"{v:.3%}" for v in secondary)
            lines.append(
                f"| {dataset_name} | {method_name} | {direct['width']} → {pi['prefilter_retained']} | "
                f"{float(pi['prefilter_s']):.3f} | {seconds:.2f} → {pi_seconds:.2f} | {saving:+.1%} | "
                f"{direct['retained']} → {pi['retained']} | "
                f"{float(direct['test_utility']):.5f} → {float(pi['test_utility']):.5f} | {secondary_text} |"
            )
    lines.extend([
        "",
        "Gain 在各自输入上重新排序，并验证全部非空前缀。PFI 在同一验证子样本上逐列置换 5 次，保留平均掉分为正的列。"
        "F 分数、互信息、TreeSHAP、采样 Banzhaf 均按预处理后的原始宽度 d 取前 ⌈√d⌉ 列，两臂使用同一个 k。"
        "L1 采用稀疏逻辑回归或 Lasso 的非零系数选列。完整参数见[后端源码](source/experiments/test/piv2_selectors.py)"
        "与[线性模型和 Banzhaf 实现](source/experiments/test/piv2_extra_selectors.py)。",
        "",
        "采样 Banzhaf 在固定预测器上用训练集中位数遮蔽缺失特征，对 32 组互补掩码（64 个联盟）计算贡献。"
        "真实重训的全空间枚举和 32 排列采样 Shapley 仅在 8 列 Abalone 上测试。"
        "加 PIv2 后，精确枚举遍历前置保留列的全部非空子集。Shapley 沿排列访问真实重训子集，并选择平均边际贡献为正的列。",
        "",
        "[88 臂逐项结果（含验证指标）](summary.csv) · [配对数据](paired.csv) · [运行日志](run.log) · [固定协议](protocol.json) · "
        "[核验结果](audit.json) · [前置源码快照](source/src/minppi_mofs/piv2.py)",
        "",
        "在 Paper 目录重新运行一次完整测试，可使用以下命令并从日志首行找到新结果目录。",
        "",
        "```powershell",
        r".\MastersThesis\thesis-python.ps1 experiments/test/run_piv2_prefilter.py",
        "```",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run
    report = render(run)
    path = run / "comparison.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
