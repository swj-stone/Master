# PIv1 两个 seed 的探索性对照

PIv1 从 PIv0 的 Git 提交 `cb5557f` 开发，使用同一套预处理数据和 100 轮 CUDA XGBoost。两个版本共用分类与回归的测试指标。Abalone 用非负截断 R²，ISOLET 用宏平均 OvR ROC-AUC，其余四个数据集用二分类 ROC-AUC。

## PIv1 如何选列

投票只用最多 512 条验证行。二分类尽量平衡两类并保留稀有类别；多分类按频率分配名额并覆盖所有类；回归均匀无放回抽样。从完整模型取得这些行的 TreeSHAP 贡献后，产生四条独立基础排列，对每条取四个循环偏移及其反向，共 32 条路径。由另外两个基础排列及其派生路径给每个位置估计平均效用增量。在位置 p 记该参照为 cₚ。平票范围半宽取 max(1e−12, 0.25 × median |Δ − cₚ|)，其中 Δ 取自用于校准的位置 p。评估路径上的增量落在此范围内记平票；超出上界记正票，低于下界记负票。投票用同一组抽样行，最终选列使用全部验证行。[投票实现](../../../src/minppi_mofs/piv1.py) · [排列采样研究](https://www.jmlr.org/papers/v23/21-0439.html)

每列分别在小、中、大联盟内计算净票率，即（正票数 − 负票数）/（该层总票数 + 2），取三层最小值 m。数值偏低的列降低进入候选子集的优先级，仍有机会在后续重训和回补中入选。此分数刻画同规模联盟中的相对贡献；PIv0 使用的序贯误报界不适用于这组相关路径。

在训练分区计算全模型 Gain 和最多 1,024 行上的平均绝对 TreeSHAP。把两个重要性的归一化名次平均，再加 0.30 × m，得到默认的候选顺序。先验证覆盖小子集和大子集的一组列数，再在表现较好的列数上比较纯 TreeSHAP 和纯 Gain 顺序、细化相邻列数，并试验加列、删列和交换 1–2 列。完整特征集也是候选。按真实重训后的整体验证效用选子集，同分优先保留较少列。六个数据集每组最多验证 B = min(24, max(6, ceil(2.4√d))) 个不同子集（含全模型）；d 是输入宽度。[搜索实现](../../../src/minppi_mofs/piv1_search.py) · [可回补的特征选择研究](https://library.utia.cas.cz/separaty/historie/somol-floating%20search%20methods%20in%20feature%20selection.pdf)

## 方案选择与对照

在 seed 42 的验证集比较普通投票和强投票。后者权重为 0.65，并根据两批排列在同一层都偏负的特征数缩小验证预算。两个主候选在六个数据集内按验证效用排名，名次和分别为 7 和 11，选定普通投票作为所有数据集的 PIv1。无投票对照使用相同的候选预算和搜索程序，将投票权重设为零，且省略投票计算。另做跨层平均取代最低净票率的验证诊断，未参与主方案选择。[开发协议](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/protocol.json) · [冻结选择](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/lock.json)

普通版的候选轨迹上还尝试了两项节省真实拟合的停止规则，均只使用验证分数。令 r 为汇总最低净票率为负、且两批排列至少有一个共同负向层的列数。第一项规则要求至少验证 B − floor(Br/(2d)) 个子集，且连续 ceil(√B) 次没有改善。第二项规则在此基础上保护原先预留给全局扫描和列数细化的阶段。两者分别将 110 个候选减至 79、92 个。Madelon 的验证 AUC 分别下降 0.01096 和 0.00497，超过原定 0.001 的可接受损失，默认版保留普通搜索预算。[验证回放](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/racing_replay.csv) · [最终决策](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/racing-lock.json)

## 复现

在 `Paper` 目录运行以下 PowerShell 命令。`develop` 输出本次目录，用该目录替换后两条命令中的占位符；每个数据集与方法独立限时 900 秒。

```powershell
.\MastersThesis\thesis-python.ps1 experiments/test/run_piv1_pilot.py develop
.\MastersThesis\thesis-python.ps1 experiments/test/run_piv1_pilot.py final --run 'results/test/thesis_sampling/<本次运行目录名>'
.\MastersThesis\thesis-python.ps1 experiments/test/summarize_piv1_pilot.py 'results/test/thesis_sampling/<本次运行目录名>'
```

两项提前停止试验可用 `experiments/test/piv1_racing.py --run <本次运行目录>` 从开发验证轨迹重新计算。[提前停止试验入口](../../../experiments/test/piv1_racing.py)

## 冷启动测试结果

下表为 seed 42 和 123 两次结果的均值。成对方法在每个数据集、每个 seed 使用相同的模型与冻结分区。PIv1 和无投票对照均在本次独立冷启动；PIv0 与直接 Gain 的时长和测试分取自此前同划分的完整运行。两个 seed 的数据曾在早期项目中使用。seed 42 的验证数据用于选择本次方案，seed 123 仅在方案固定后运行。

| 数据集 | PIv1 测试 | PIv0 测试 | 直接 Gain 测试 | 同预算无票测试 | 平均保留列（PIv1 / PIv0 / Gain） | 平均秒数（PIv1 / PIv0 / Gain） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Abalone | 0.53648 | 0.53648 | 0.53648 | 0.53648 | 7.5 / 7.5 / 7.5 | 4.92 / 4.84 / 4.59 |
| Spambase | 0.98753 | 0.98661 | 0.98747 | 0.98758 | 46.0 / 25.0 / 34.5 | 6.79 / 8.74 / 11.80 |
| Madelon | 0.87138 | 0.87263 | 0.87138 | 0.87314 | 8.5 / 8.5 / 8.5 | 8.11 / 12.05 / 20.45 |
| Gisette | 0.98894 | 0.98571 | 0.98902 | 0.98906 | 59.0 / 56.5 / 76.5 | 8.35 / 15.84 / 21.55 |
| Credit Card Fraud | 0.97420 | 0.97759 | 0.97805 | 0.97092 | 24.0 / 12.5 / 26.5 | 13.36 / 29.20 / 20.83 |
| ISOLET | 0.99333 | 0.99339 | 0.99343 | 0.99344 | 70.5 / 81.0 / 85.0 | 95.37 / 332.58 / 336.16 |

12 组累计，PIv1 拟合 232 次、用时 273.80 秒，PIv0 为 526 次、806.52 秒，直接 Gain 为 797 次、830.76 秒。相对 PIv0 节时 66.1%，相对直接 Gain 节时 67.0%。各划分按测试分数比较的胜/平/负次数（1e−14 数值容差）：PIv1 对 PIv0 为 6/3/3，对直接 Gain 为 2/4/6，对同预算无票为 3/5/4。seed 123 上，PIv1 对直接 Gain 为 0/2/4；Credit Card Fraud 比直接 Gain 低 0.00593，比 PIv0 低 0.00818。

同预算无投票对照也拟合 232 次，耗时 263.23 秒。实测主要加速来自限制真实子集拟合数；相对无票对照，加入 voting/min 后总用时增加 4.0%。温和方案中，最弱层与跨层平均的开发验证对照为 0 胜、5 平、1 负；在 ISOLET 上最低净票率的验证宏平均 AUC 比跨层平均低约 0.00009。本轮没有显示 min 自身能稳定提高效用。

[12 组四方法逐项明细](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/paired_comparison.csv) · [各数据集均值](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/dataset_comparison.csv) · [逐 seed 成本和胜平负](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/by_seed.csv) · [min 与均值的验证消融](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/min_vs_mean_validation.csv) · [冻结与分数复核](../../../results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781/independent_audit.json)

