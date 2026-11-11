# 硕士论文待解决问题

## MinPI 对纯冗余有效特征缺少直接删除机制

**状态：待解决。记录日期：2026-10-27。**

硕士论文需要解决以下问题：当多个特征提供重复或可替代的有效信息，且加入这些特征不会造成达标状态下降时，当前 MinPI 的删除规则不能主动去重。应研究如何在保留必要预测信息的同时删除多余副本，并明确这一能力与原有有害特征检测的关系。

### 当前规则及限制

对于特征集合 N、验证集效用 U 和配额 alpha，当前方法定义：

```text
tau = alpha * U(N)
v(S) = 1，当 U(S) >= tau；否则为 0
MinPI(i) = min over S ⊆ N \ {i} of [v(S ∪ {i}) - v(S)]
H = {i : estimated MinPI(i) = -1}
```

只有出现“某个组合原本达标，加入特征后不达标”的证据，特征才会进入候选删除集合 H。纯粹的“加入后没有额外作用”只产生零边际贡献，不满足这个条件。

候选集合 H 联合删除后，验证门槛检查 `U(N \ H) >= U(N) - epsilon`。通过则接受全部候选，否则撤回全部删除。这个门槛只审核已有候选，不会从零分特征中补充冗余候选，也不会自动选择重复特征中的一个代表。

### 最小反例

以下为按定义构造的说明性效用表，不是实测结果。设 A、B 提供完全相同的有效信息，完整模型 AUC 为 0.90，`alpha = 0.95`，因此达标线为 0.855。

| 特征集合 | 验证 AUC | 达标状态 v |
|---|---:|---:|
| 空集合 | 0.50 | 0 |
| 仅 A | 0.90 | 1 |
| 仅 B | 0.90 | 1 |
| A 和 B | 0.90 | 1 |

对 A 而言，在空集合中加入它得到边际贡献 +1，在已有 B 时加入它得到 0，所以 `MinPI(A) = min(+1, 0) = 0`。B 同理为 0。

保留 A 或 B 中任意一个都足够，但当前规则不会提名删除任何一个。这是评分定义与删除条件的限制，即使枚举全部组合，也不能消除这个反例中的冗余。

### 解释边界

- 若效用对增加特征始终单调不减，则所有阈值边际贡献均非负，当前规则不会提名删除任何特征；增加采样预算无法产生不存在的负边际证据。
- 实际训练中，重复特征仍可能因拟合过程等因素造成阈值下降，进而被 MinPI 标为 -1。这是在检测其负面作用，不等于直接识别重复性。
- 仅有 `U(N \ {i}) = U(N)`，不足以推出特征 i 在所有其他组合中都没有负面作用，因而也不足以推出其 MinPI 一定为 0。
- MinPI 为 0 不等于没有预测价值，也不等于可以联合删除。上例中 A、B 都是零分，但一起删除会丢失全部有效信息。
- 后续 NSGA-II 通过同时优化特征数量和性能，可能找到只保留一个副本的方案；这属于后续子集搜索的能力，有限预算搜索不保证找到最小子集。

### 硕士论文需要解决的内容

- 明确纯冗余、近似冗余和有害特征的区别，为当前 -1 候选规则未覆盖的冗余情况确定处理机制。
- 在重复或可替代特征中保留必要代表，避免把同组有效特征全部删除；不能仅将删除条件扩大为“删除全部零分特征”。
- 以完全重复有效特征、近似重复特征和依赖组合才能发挥作用的特征为验证情形，检查压缩程度、预测效用及不同随机种子下的稳定性。
- 分别报告 MinPI 单独阶段、冗余处理阶段及后续 NSGA-II 的作用，避免将后续搜索的去重收益归因于 MinPI 本身。

具体算法和实验方案尚未确定；本问题保持待解决状态。

### 当前实现依据

- [会议论文参考正文](../../conference-paper/paper/icomp2026_conference.tex)：`Min-Marginal Power Indices` 与 `Quota-Indexed Selection and Validation Gate` 两节。
- [删除流程](../../experiments/icomp2026_stages.py)：`_run_removal_plan` 仅提名分数为 -1 的特征；`_removal_safety_gate` 执行整组接受或撤回。
- [NSGA-II 子集搜索](../../src/minppi_mofs/baselines/nsga2_fs.py)：同时最小化特征数量比例与效用损失。

### English summary

Open thesis problem: MinPI's current removal rule does not directly eliminate purely redundant but useful features. Two interchangeable useful features can both have exact MinPI equal to zero, so neither is nominated by the -1-only rule, although either one could be removed. The thesis needs a mechanism that removes unnecessary copies while retaining the information required for prediction, with its contribution distinguished from harmful-witness detection and downstream NSGA-II search.

## MinPI 的效用指标敏感性及多指标保留与裁剪决策

**状态：待解决。记录日期：2026-10-27。候选方案尚未确定，也未实现。**

需要研究是否有更符合任务目标的指标，或可以联合多个指标，决定最终保留哪些特征、裁剪哪些特征。当前分类实验主要使用 ROC AUC 或宏平均 OVR AUC；保持排序能力不保证保持最终分类准确率。研究目标是明确不同指标下“有害”的含义，并防止一种指标的改善掩盖实际关心的另一种性能损失。

### 已有证据与尚不能得出的结论

ISOLET 的 seed 42 配对效用消融使用相同采样计划和同一批预测结果，预算为 10。100 个分析特征中，accuracy 下有 9 个 -1，OVR AUC 下全部为 0。两种指标的预测摘要一致，第二次评分没有增加模型拟合。

这支持“效用指标会改变 MinPI 的有害标签”，但不能证明 AUC 选错了、accuracy 普遍更好，或这 9 个特征应当被删除。该消融只有一个随机种子；配对文件保存了特征分数和预测摘要，没有直接列出每个有害见证的指标前后变化，也没有给出这 9 个特征联合删除后的独立测试效果。

AUC 下未发现 -1，可能是排序没有变差，也可能是 AUC 下降但未跨过门槛，或者有限采样没有遇到见证。零发现不能解释为所有指标均无损，也不能解释为精确证明没有 AUC 有害情境。

### 需要比较的指标

以下是按任务选择的候选组合，不是已验证的最佳配置。

| 任务或关注点 | 候选指标 | 在决策中的作用 |
|---|---|---|
| ISOLET 等多分类识别 | Accuracy、宏平均 F1、宏平均 OVR AUC | 可优先研究以识别准确率为主指标，以宏平均 F1 和 OVR AUC 约束其他性能损失；同时查看各类别结果 |
| 欺诈等稀有正类识别 | 平均精确率 AP、预定误报率上限下的召回率、ROC AUC | 比较正类识别表现与实际操作点；误报率上限和分类阈值选择规则需事先确定 |
| 概率预测质量 | Log loss | 补充对概率预测的评价；它越小越好，不是纯粹的校准指标 |

宏平均 F1 是先对每个类别计算 F1 再等权平均，不能保证任何单个类别都不退化。AP 是精确率与召回率曲线的摘要，与梯形积分得到的 PR-AUC 不是同一个计算规则。应固定精确定义，避免混用名称。

不同指标即使都在 0 到 1 之间，其基线、敏感性和实际意义也不同；统一使用同一个相对配额 alpha，不代表相同的容忍程度。Log loss 等越小越好的损失，不能直接代入当前“正效用且越大越好”的乘法门槛。

### 候选研究方案

| 方案 | 基本决策规则 | 需要验证的代价或限制 |
|---|---|---|
| 主指标加逐指标约束 | 用任务主指标评价收益；最终子集必须同时满足其他必选指标的损失上限，再按预先规定的性能与特征数量偏好选择 | 规则直观、避免指标相互抵消，但可能保留更多特征；优先比较这一方案，尚未认定它最优 |
| 多目标子集搜索 | 将特征数量和多个性能指标作为搜索目标，保留折中方案，再应用事先规定的最终选择规则 | 指标增多可能增加搜索难度；得到 Pareto 前沿本身不等于完成最终决策 |
| 归一化后加权评分 | 将多个指标合成为一个效用分数，作为比较基线 | 必须解释归一化与权重；总分上升可能掩盖某个关键指标退化，不能默认取等权平均 |

可讨论的初步决策流程如下，具体规则仍待研究确定：

1. 在实验前固定任务主指标、辅助指标、各自门槛和允许损失；分类阈值只能利用训练或验证数据确定。
2. 分指标记录有害证据及其具体背景组合。候选集合可以研究采用各指标负分集合的并集，但不同见证可能来自不同组合；并集仅是删除提名，不是安全删除结论。
3. 对拟保留的整个子集 S 重新训练模型，在验证数据上同时计算所选指标。对于越大越好的必选指标，逐一检查 `U_m(S) >= U_m(N) - epsilon_m`；对于越小越好的损失，检查 `L_m(S) <= L_m(N) + epsilon_m`。
4. 只有通过全部必选约束的候选子集，才按预先规定的性能与特征数量规则接受；未通过时保留原方案，或继续寻找不同的删除子集。不能因为一个指标改善就忽略另一项超限损失。

这些约束是候选方案在验证集上的接受条件，不是泛化性能保证，也不是统计意义上的非劣效证明。多指标判定本身仍不能消除前一个问题中的零分冗余特征；若需联合处理，应另行明确冗余候选如何产生。

### 研究验证重点

- 在原有准确率有害见证上，同时记录加入特征前后的各项指标、跨门槛幅度和所属特征组合，区分指标差异、门槛效应与采样漏检。
- 比较原始 AUC 方案、任务主指标方案及候选混合方案，分别检查单特征与联合裁剪，不能从负分数量直接推断裁剪收益。
- 扩展到多个随机种子，报告配对性能变化、压缩程度、逐类表现、选择稳定性和额外计算开销；分别分析预筛选与后续 NSGA-II 的贡献。
- 方案、指标、容忍度及最终选择规则只在开发数据上确定，控制重复使用验证集带来的选择偏差；冻结选择后再用独立测试数据评价，不能按测试结果挑权重或特征。

补充多指标报告属于实验完善；是否形成方法贡献，取决于能否给出清晰的冲突处理与接受规则，并通过上述比较验证其价值。

### 项目证据与指标定义

- [论文效用消融结果](../../conference-paper/paper/icomp2026_conference.tex)：`RQ4: Estimator, Guidance, and Utility Sensitivity`。
- [ISOLET 原始配对记录](../../results/icomp2026_formal/icomp2026-formal-v1-s42-123/ablation/isolet/s42/utility/paired/b10/scores.json)：本地历史数据，位于 Git 忽略的 results 目录。
- [配对消融与删除流程](../../experiments/icomp2026_stages.py)：`_run_utility_ablation` 与 `_removal_safety_gate`。
- [当前效用计算](../../src/minppi_mofs/utility.py)：accuracy 使用每条样本的最大概率类别，OVR AUC 使用宏平均。
- [F1 与宏平均定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)、[AP 定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)、[Log loss 定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)。

### English summary

Open thesis problem: investigate task-appropriate utility metrics and combined criteria for feature retention or pruning. ISOLET's paired ablation found nine accuracy-based harmful labels and none under OVR AUC, which demonstrates metric sensitivity but does not establish that those nine features should be removed. Candidate approaches include a primary metric with separate performance constraints, multi-objective subset search, and normalized weighted scoring. The first is proposed for initial comparison, not accepted as optimal. Metric-specific harmful witnesses only nominate candidates; final subsets require joint evaluation, with separate tolerance directions for scores and losses, multi-seed validation, and a frozen independent test evaluation.
