"""Render the two evidence panels without pooling development and prospective tasks."""
import pandas as pd

from experiments.anchored_vote.audit import OUT


def section(title,df,new):
    lines=[f"## {title}","","| 数据集 | 输入列数 | 指标 | 保留列数 | 完整耗时（秒） | 测试分数 | 准确率或 MAE | 同时通过的 seed |",
           "|---|---:|---|---:|---:|---:|---:|---:|"]
    for _,row in df.iterrows():
        pi,base=("min","control") if new else ("pi","base")
        is_cls=pd.notna(row.get(f"auc_{pi}"))
        primary,secondary=("auc","accuracy") if is_cls else ("r2","mae")
        def pair(name,precision):
            return f"{row[f'{name}_{pi}']:.{precision}f} / {row[f'{name}_{base}']:.{precision}f}"
        passed=int(row["simultaneous_seeds" if new else "paired_pass_seeds"])
        lines.append(f"| {row.dataset} | {int(row.input_width)} | {'AUC' if is_cls else 'R²'} | {pair('k',0)} | {pair('seconds',3)} | "
                     f"{pair(primary,4)} | {pair(secondary,4)} | {passed}/5 |")
    return lines


def main():
    prepared=pd.read_csv(OUT/"prepared-sources.csv").query("baseline=='exhaustive'")
    prospect=pd.read_csv(OUT/"sources.csv").query("control=='exhaustive'")
    prepared["input_width"]=prepared.p_pi
    from experiments.minpi_replication.second_panel import data
    prospect["input_width"]=prospect.dataset.map(lambda name:len(data.feature_names(name)))
    if len(prepared)!=6 or len(prospect)!=5:
        raise RuntimeError("Both panels must be complete")
    lines=["# MinPI 和直接 Gain 的逐数据集对照","",
           "每格按 MinPI / 直接 Gain 排列。直接 Gain 按训练增益排序，对每个非空前缀重新拟合，取验证得分最高的宽度。"
           "分类报告测试 AUC、准确率，回归报告测试 R²、原单位 MAE。耗时从全列模型拟合到最终子集重训并产生测试预测。"
           "列数、时间和分数均为五个 seed 的中位数。","",
           "每个数据集要求配对耗时差的中位数严格小于零，列数超过基线容许量的中位数不大于零。"
           "容许量为 1 列与基线列数的 5% 中的较大者。最右列同时检查同一 seed 的时间和宽度。","",
           *section("原六个预处理数据集",prepared,False),"",
           f"达到时间和列数标准的有 {int(prepared.pass_joint.sum())}/6 个，加入 AUC 或 R² 最多下降 0.01 的条件后为 "
           f"{int(prepared.pass_with_quality.sum())}/6 个。原六集经过开发使用。两种方法在同一 GPU 的先后两次运行中计时。"
           "尤其是 ISOLET，虽然 AUC 接近，准确率中位数仍从 84.10% 降至 77.37%。","",
           "若穷举 Gain 同样选取距验证最高分 0.005 以内的最窄前缀，MinPI 在原六集中只有 3/6 同时达到列数与测试主分数要求。"
           "当这个对照需要更换最终子集时，只实测了额外的最终重训时间。[同容差逐集结果](prepared-common-slack.csv)","",
           *section("预先指定的五个新数据集",prospect,True),"",
           "五个新数据集均达到时间和列数标准。每个 seed 在同一 CPU 上轮换方法顺序并计时三次。全部 25 组同时达到时间和列数标准，"
           "其中 23 组同时满足测试主指标容差。元音识别 seed 46 的准确率从 51.3% 降至 43.3%。","",
           "新数据集上，与几何宽度搜索的 Gain 比，4/5 同时满足时间和列数条件。换成同预算的 Gain 后为 0/5。"
           "最低票和均值票在 25 组中产生完全相同的最终预测。与 RFECV 比，5/5 同时满足时间和列数条件，加入测试分数容差后为 4/5。"
           "默认 SelectFromModel 在 25 组中均更快。","",
           "[原六集逐 seed](prepared-seeds.csv) · [新五集逐 seed](seeds.csv) · [标准方法对照](standard/sources.csv) · "
           "[不确定性分析](held-unit-uncertainty.md) · [原始预测复算](replication.md)"]
    (OUT/"comparison-zh.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
