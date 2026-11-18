# 原 69 组中二分类与回归的集中度选宽结果

最初论文的八来源面板有另一组五次固定划分，二分类以 ROC AUC 为主。其四个二分类和两个回归来源的集中度版已重新测量。[原稿六来源结果与原 10% 定宽对照](paper-results.md)

最新两层条件投票 minPI 在原 69 组中实际覆盖 3 个二分类来源与 4 个回归来源，共 45 个配置。二分类沿用冻结的分类选择器。回归首次在这套规则中使用连续标量响应上的岭损失下降，保留同样的两个同伴规模、最低贡献票与自动预算。7 个来源合计 17 个不同的外层测试矩阵。同一外层测试数据上的多个种子用于测量选列不稳定性。[运行前的协议](protocol.md) · [逐个输入和划分的指纹](evidence/inputs.csv)

| 任务和来源 | 配置 | 保留列 / 原列 | minPI 分数 | 全特征分数 | 同宽 F 分数 | 总时间 / Full | 真删列且快于 Full | 同时符合 0.02 容差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 二分类 Spambase | 9 | 17–26 / 57 | 84.32% | 87.13% | 84.25% | 0.879 | 9/9 | 1/9 |
| 二分类 Madelon | 9 | 94–162 / 500 | 55.03% | 54.38% | 55.23% | 0.508 | 9/9 | 7/9 |
| 二分类 SECOM | 3 | 91–99 / 590 | 49.95% | 49.59% | 49.73% | 0.501 | 3/3 | 3/3 |
| 回归 Abalone | 9 | 3–7 / 10 | 0.4835 | 0.5301 | 0.4270 | 1.188 | 0/9 | 0/9 |
| 回归 Communities and Crime | 9 | 8–34 / 102 | 0.3292 | 0.3718 | 0.3689 | 0.834 | 9/9 | 3/9 |
| 回归 Diabetes | 3 | 3–6 / 10 | 0.4704 | 0.4911 | 0.4545 | 1.414 | 0/3 | 0/3 |
| 回归 Superconductivity | 3 | 16–20 / 81 | 0.6416 | 0.7231 | 0.6011 | 0.286 | 3/3 | 0/3 |

分数为同一来源配置的平均测试平衡准确率或 R²。每个进程先分别取五次中位总时间，再算方法与 Full 的比值；表中取两进程中的较大比值，然后对来源内配置取中位数。Full 使用全部特征拟合相同的岭模型。F 为全训练样本上的单因素 ANOVA F 或单变量线性回归 F，保留列数与新 minPI 相同。宽度独立于测试集确定。[逐来源分数与各基线的时间比](evidence/sources.csv) · [逐配置分数和两进程的毫秒数](evidence/cases.csv)

二分类 21/21 真删列并缩短总时间，再加最多下降 2 个百分点的条件为 11/21。同宽 F 有 12/21 次同时被 minPI 超过分数及速度，单个固定随机支持有 17/21 次分数较低、21/21 次速度更快。Spambase 平衡准确率平均牺牲 2.80 个百分点，Madelon 平均提高 0.64 个百分点，但仍只有约 55%。SECOM 的三个种子共享 392 条测试记录，其中 24 条缺陷。新方法三次均漏掉全部缺陷，预测为正的条数分别为 0、1、0。平均精确率为 0.0837，全特征模型为 0.1415。这里的名义三项达标没有转化为缺陷检出能力。[逐配置分数与 SECOM 的平均精确率](evidence/cases.csv)

回归 24/24 真删列，12/24 总时间下降，3/24 还守住 0.02 的 R² 容差。新方法有 9/24 次同时比同宽 F 更准更快，对单个随机支持的分数有 16/24 次更高，总时间 0/24 次更低。两个原本只有 10 列的来源在全部 12 次配置中都无法抵消选列开销。Superconductivity 的时间降幅最大，相比同宽 F 三组都更准更快，但相比全特征平均损失 0.0815 R²。[按任务汇总](evidence/tasks.csv)

## 复跑

从 history 根目录执行。每个输出路径需使用新目录。

```powershell
.\thesis-python.ps1 -m research.conditional_dense.original_tasks.study results/minpi-study/my-orig-pass1
.\thesis-python.ps1 -m research.conditional_dense.original_tasks.study results/minpi-study/my-orig-pass2 --reference results/minpi-study/my-orig-pass1
.\thesis-python.ps1 -m research.conditional_dense.original_tasks.summarize results/minpi-study/my-orig-pass1 results/minpi-study/my-orig-pass2 results/minpi-study/my-orig-summary
```

两个进程共 2,250 条正式计时，逐项重现 510 个存档数组，另从保存的预测复算 225 个分数。所有 Full 预测逐项比对之前的原配置归档。独立公式核对 45 份同宽 F 列集全部一致。[核查记录](evidence/verification.json) · [正式计时一](../../../results/minpi-study/original-tasks-20261118a-pass1/times.csv) · [正式计时二](../../../results/minpi-study/original-tasks-20261118a-pass2/times.csv)
