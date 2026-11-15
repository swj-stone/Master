# Master's thesis workspace / 硕士论文工作区

硕士研究在 `MastersThesis/` 的 `thesis/masters` 分支进行，与 `../Master/` 共享 Git 仓库及 Python 环境。会议最终版在 `../Master/` 的 `main` 分支，供只读参考。工作树中的 `data/`、`results/` 和 `packages/` 是独立的本地文件，受 Git 忽略规则管理。

## 遮蔽损失能揭示多少删列重训后的信息？

论文当前区分了两个问题。在已知的有限样本中，单靠原模型的连续输出就能锁定全部 704 个提案的容差判定，精度降低后会出现与同一记录相容的相反判定。在实际选列时，给相同 Gain 候选和单点平票规则再加投票，只改变了常规组的一个提名，宽数据组为零。换用随机森林和 ANOVA 的局部候选后，两类评分的相对表现随数据来源与候选抽样变化。[当前论文](thesis-paper/minpi.pdf) · [新证据和复现命令](thesis-paper/evidence/robustness-release.md) · [实验协议](experiments/context_review/README.md)

在 704 个历史候选列集中，有 136 个在同一折与另一个同宽候选得到完全相同的二元选票，但重训后的损失一个超限、一个未超限。新实验让各策略从同一组半宽候选中提名一个列集，再用独立重训检查决定是否接受。首批 10 个 UCI 任务上，单次连续遮蔽评分、最差背景票数、均匀随机提名的容差内平均压缩分别为 7.29%、1.33%、5.49%。在另外 6 个较高维任务的 16 个划分上，四种主策略的压缩均为 7.34%，各有 3 次外测超限。[论文 PDF](thesis-paper/minpi.pdf) · [当前证据包与复现说明](thesis-paper/evidence/transport-release.md) · [逐任务结果](thesis-paper/evidence/chance/)

### 早期投票与重训对照

算法先用 Compact Gain 选出初始列集，再提议删除其中若干列。每次恢复一组背景列，在同一个已训练模型上比较候选与初始列集。四个验证面板投票，所有恢复情形都要有至少三票同意。这里的 min 作用于一项删列提案，会议版 MinPI 衡量的是单个特征的最差边际投票。两者的定义和输出不同。[英文论文](thesis-paper/minpi.pdf) · [论文与复现说明](thesis-paper/README.md)

在已用于比较的 14 个 PMLB 分类任务上，我们用独立于选择的外折测试行计算删列重训损失。与只检查空、全两种背景相比，加入分离设计的中间背景，在 400 个历史搜索提案上多识别了 11 个超容差提案，同时额外拒绝了 20 个重训仍在容差内的提案。在另行抽取的 379 个删列集合上，多识别 0 个，多误拒 9 个。按任务看，搜索提案的 11 个额外检出中有 7 个来自 Sonar。另一次实验给候选和 Gain 前缀相同的独立重训检查，两者的平均外折 AUC 都为 0.93327。[逐任务图](thesis-paper/figures/refit-vetoes.pdf) · [独立复核与适用范围](thesis-paper/audit/refit-review.md)

两个新外测在本地先固定方案和预测，再打开公开测试标签。活动识别的九名新受试者上，22 列投票的平均受试者准确率为 89.40%。同宽 RFE 用 15 或 21 人选列时分别为 89.47% 和 90.68%。单一气体传感器阵列的三个未来批次上，六列投票的等批次平均平衡准确率为 35.81%。同宽 RFE 用前五、六、七批选列时分别为 41.72%、52.63%、57.80%。在 PMLB 上调整否决顺序并优化预测调用，完整流程的任务配对耗时中位降幅为 2.12%，未达到预设的 5% 目标。[活动识别协议](experiments/context_review/external_har.md) · [时间漂移协议](experiments/context_review/drift.md) · [等宽基线的历史数据权限](experiments/context_review/matched_history.md) · [完整计时](experiments/context_review/timing-process.md)

另一次对照直接在删列重训后的预测上投票。在同一组 Gain 候选上，仅检查全局指标时，41 个外折中有 16 折至少一个类别的召回越过损失容差。对每个类别各检查一次后降到 8 折，改用四面板投票后为 7 折。后者只改变了 4 折的选宽，平均多保留 1.05 个百分点的特征，平均 AUC 从 0.93139 降到 0.93089。该试验复用了已研究的 14 个数据集。[方案、失败样例与结果](experiments/context_review/refit_vote_grid.md)

### 提案成本与实际选列

| 问题 | 结果 |
|---|---|
| 增加一次否决值多少钱？ | 在搜索提案上多检出 11 个失败，同时多误拒 20 个成功，故漏报与误报的成本比超过 20/11 时，这一对规则的经验成本才有利于增加背景。按任务与折重采样，交叉点的区间上界无穷。在另行抽取的 379 个提案上多检出 0 个。用其他 13 个任务选规则时，按删除比例设阈值在抽样组的四种代价下总成本都较低。[成本与区间](experiments/context_review/cluster_cost.md) · [跨任务选择](experiments/context_review/cluster_cost_transfer.md) |
| 得票能否提供超过删除比例的信息？ | 对 360 种设置作敏感性检查。将重训模型从 XGBoost 换成逻辑回归时，400 个路径提案有 111 个实际超限标签改变。在留出任务上，加入最差背景的反对票后，路径与抽样提案的风险排序 AUC 各提高 0.024 和 0.046。两组对数损失的差值区间均跨零。[规则敏感性](experiments/context_review/rule_sensitivity.md) · [增量预测](experiments/context_review/incremental_prediction.md) |
| 只有一次候选重训机会时选哪些列？ | 在 41 个 PMLB 外折上，半宽 Gain 与投票加 Gain 的容差内压缩分别为初始列集的 9.33% 和 10.12%，外折超限为 1 和 3 次。让主要风险模型自行选宽后，加入投票使容差内压缩从 8.94% 升至 11.92%，超限从 1 次升至 4 次。单独报告的 Optdigits 固定预算比较在每种预算下均选出相同列集。[预算、重训检查与逐折结果](experiments/context_review/budgeted_guard.md) |
| 完整流程耗时多少？ | 半宽策略共完成 168 次独立执行，164 次来自 PMLB，4 次来自 Optdigits。按 14 个 PMLB 任务等权，Gain 平均 0.8423 秒，投票平均 0.8658 秒，差值 +0.0235 秒，95% 任务重采样区间为 −0.0584 至 +0.1046 秒。每次均核对提名、最终选列和预测。[计时记录](results/context_review/guard-clock/runs.csv) · [汇总](results/context_review/guard-clock/summary.json) |

| 查找内容 | 入口 |
| --- | --- |
| 复现实验或查看逐项结果 | [实验协议与执行命令](experiments/context_review/README.md) · [本轮证据包](packages/thesis-transport-evidence.zip) · [早期操作策略证据](thesis-paper/evidence/operating/README.md) · [早期重训对照汇总](thesis-paper/evidence/reassessment/) |
| 稿件、复审和可验证的数学命题 | [论文 PDF](thesis-paper/minpi.pdf) · [论文源文件包](thesis-paper/minpi-source.zip) · [复审意见](thesis-paper/audit/transport-area-chair.md) · [独立复算记录](thesis-paper/audit/transport-forensic.md) · [可识别范围证明](thesis-paper/formal/IdentificationBounds.lean) |
| 投票、min 与提案搜索 | [搜索实现](src/minppi_mofs/context_vote.py) · [恢复列设计](src/minppi_mofs/context_design.py) · [历史六集的机制检验](experiments/context_diagnostics/README.md) |
| 已用于开发或对照的基准 | [14 个 PMLB 任务与高维挑战](experiments/context_study/README.md) · [早期基线核查](thesis-paper/audit/rounds/README.md) |

## 历史版本

| 版本 | 实验范围 | 查看结果 |
| --- | --- | --- |
| PIv0 | 六个预处理数据集、五个 seed，与直接 Gain 按划分配对 | [正式结果](docs/thesis/experiments/piv0-five-seeds.md) · [与会议版及其他基线的比较](docs/thesis/experiments/piv0-method-comparison.md) |
| PIv1 | 同一批 30 个划分。表中含分类 AUC 与准确率、回归 R² 与 MAE、未删列验证基线及耗时 | [数据集汇总](docs/thesis/archives/piv1/final_results_by_dataset.csv) · [逐 seed 结果](docs/thesis/archives/piv1/final_results_by_seed.csv) · [协议、对照与证据包](docs/thesis/archives/piv1/README.md) |
| PIv2 | seed 42 上，对比下游选择器单独使用和加 PI 预筛的结果 | [预筛入口](experiments/test/run_piv2_prefilter.py) |
| PIv3 | seed 42 原始宽度多后端、Lite 与 compact witness 试验 | [六集入口](experiments/test/run_piv3_raw_six.py) · [Lite](experiments/test/run_piv3_lite_six.py) · [原五块选择器](src/minppi_mofs/piv3_compact_witness.py) |
| 上一版上下文投票研究 | 原六集五 seed、线性迁移及原始宽度实验 | [历史实验索引](thesis-paper/README.md#earlier-six-dataset-study) · [历史稿件](packages/prior-context-vote/minpi.pdf) |

## 历史 PIv0 或 PIv1 正式实验

以下命令使用早期试验的本地运行目录作为输入。迁移工作树后，需要先按对应版本的协议备齐历史输入；PIv1 的可恢复范围见[封存说明](docs/thesis/archives/piv1/README.md)。本轮论文使用上方的正式对照入口。

在包含 `Master`、`MastersThesis` 和 `venv` 的 **`Paper` 目录**打开 PowerShell，按所需版本执行对应的一条命令。入口使用共享的 `venv`、导入硕士工作树的源码，并在 `MastersThesis/` 下解析实验路径。

**PIv0**

```powershell
.\MastersThesis\thesis-python.ps1 experiments/run_piv0_benchmark.py
```

**PIv1**

```powershell
.\MastersThesis\thesis-python.ps1 experiments/run_piv1_benchmark.py
```

正式实验使用 Abalone、Spambase、Madelon、Gisette、Credit Card Fraud、ISOLET，seed 为 42、123、456、789、1024。每组沿用对应 seed 的冻结数据划分和最多 100 列的预处理输入。每个版本有 30 组配对、60 条方法记录，新训练运行至完成。PIv0 复用已核验的 seed 42 结果。PIv1 使用已核验的直接 Gain 及 pilot 记录，补算其余 PIv1 划分。[PIv0 实验口径](docs/thesis/experiments/piv0-five-seeds.md) · [PIv1 实验口径](docs/thesis/experiments/piv1-five-seeds.md)

程序打印本次结果目录，并在终端和 `run.log` 中记录进度、拟合次数和用时。接续已有批次时，在对应启动命令后加 `--output` 和本次结果目录路径。要在另一个 PowerShell 窗口查看日志，在 `Paper` 目录运行下列命令，并将占位符替换为实际目录名。

```powershell
Get-Content -LiteralPath '.\MastersThesis\results\thesis_benchmark\<本次运行目录名>\run.log' -Tail 15 -Wait
```

完整批次结束后，使用对应版本的汇总脚本。

```powershell
.\MastersThesis\thesis-python.ps1 experiments/summarize_piv0.py 'results/thesis_benchmark/<PIv0本次运行目录名>'
.\MastersThesis\thesis-python.ps1 experiments/summarize_piv1.py 'results/thesis_benchmark/<PIv1本次运行目录名>'
```

PIv1 的封存标签为 `thesis/piv1-frozen`。含准确率、回归 MAE 和未删列基线的完整表及补算记录可从封存索引查阅。[PIv1 封存索引](docs/thesis/archives/piv1/README.md)

## 方案试验与历史材料

PIv0、PIv1 的开发对照、PIv2、PIv3 的前置筛选试验入口在 `experiments/test/`，单 seed 结果在 `results/test/thesis_sampling/`。各入口的用法、输入宽度及已有对照见[方案试验说明](experiments/test/README.md)。

PIv0 定名前的独立探索按清理清单移除了本地运行文件。硕士工作树仍保留论文比较引用的会议出版表、旧 guarded 汇总、PIv1 的 Abalone 联盟缓存，以及各版本复现实验用的原始数据、冻结划分和配置。[清理清单](docs/thesis/cleanup-inventory-2026-11-13.md) · [PIv0 历史比较](docs/thesis/experiments/piv0-method-comparison.md)

会议论文、海报和会议实验的使用说明在冻结参考版中可查阅。[会议项目说明](../Master/README.md) · [会议论文](conference-paper/paper/icomp2026_conference.pdf) · [海报](conference-paper/poster/ASCOMP_MinPI_Poster_Wenjie_SHI.pdf)

## 本地开发

在 `MastersThesis/` 中使用 `thesis-python.ps1`。该入口为当前进程设置源码路径，并将 Matplotlib、Numba 缓存放入本工作树的 `tmp/`。共享环境中已有的可编辑安装指向会议参考版。

```powershell
.\thesis-python.ps1 -c "import minppi_mofs; print(minppi_mofs.__file__)"
.\thesis-python.ps1 -m pytest tests -q -p no:cacheprovider --basetemp ".t$PID"
```

在本工作树提交后，如需推送硕士论文分支，可执行下列命令。

```powershell
git status --short --branch
git push -u origin thesis/masters
```
