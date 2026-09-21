# MR1：MinPPI 大修研究定位与实验协议

## 0. 文档地位与本轮边界

版本：MR1 design v0.1，2026-09-10。用户已确认路线 A：**见证审计语义＋多场景精确参照＋条件性决策价值**。本文件把路线细化为可审查设计；尚不是已实施、已注册或已冻结的确认实验协议。

- 本轮只进行文献、协议与现有实现审查；不启动训练，不修改算法、论文正文、历史结果，不提交 git。
- 所有后续命令从 `Master` 运行，Python 仅用 `../venv/Scripts/python.exe`。不引入外部项目脚本作为运行依赖。
- 历史证据及 MR0–MR7 台账见 [MAJOR_REVISION_WORKPLAN.md](MAJOR_REVISION_WORKPLAN.md)。新增结果必须独立命名并记录协议哈希，不覆盖已有 campaign。
- 本设计优先明确 MR2–MR4 所需接口和有限小试。旧数据上的扩展验证不是新独立外部确认；MR5 的新增确认数据清单及推断计划必须另行批准并冻结。
- 用户已确认 review 来自本届正式评审，并要求同时准备 rebuttal。大修无自设截止时间与本届回复窗口是两件事。回复不得把本文件中的计划称为完成的实验。

## 1. 主问题、定位与贡献边界

### 1.1 一个主问题

**在一个固定、可重放的特征子集效用博弈中，有限计算预算能发现多少上下文有害见证；这些发现能否支持比简单选择策略更有价值的删除决策？**

前半句是主要研究问题；后半句是独立的应用检验，而不是定理推论。三个层级必须分开：

1. 见证有效：某个已评估上下文确实发生负向阈值翻转。
2. 发现充分：有限采样发现了多少真实有害特征，漏掉哪些类型。
3. 行动有用：联合删除后，是否在可接受效用损失下得到压缩或成本收益。

建议英文定位：

> Building on minimum aggregation, we study a quota-conditioned audit of negative utility-threshold crossings, retain evaluated coalitions as witnesses, assess discovery completeness against exact small-game references, and evaluate deletion as a separate decision problem.

这是研究定位，不是“新颖性已获证明”或“方法已有效”的结论。不得将 minimum aggregation、本身的有限最小值性质或非单调二值博弈声称为首次提出。

### 1.2 三个研究问题与固定评价对象

| 研究问题 | 主评价对象 | 不能替代它的结果 |
| --- | --- | --- |
| RQ1：有限预算的见证发现能力 | 多个精确小博弈中的 recall–总查询曲线及 A/B 配对差 | 高维 harmful count、训练准确率、oracle 条件曲线 |
| RQ2：删除决策是否值得 | 同验证约束下的压缩、测试效用损失、失败率与成本 | 只与随机比较、只展示 Madelon、gate 通过就称安全 |
| RQ3：解释和成本来自哪里 | 可重放的见证、阈值距离、逻辑查询与冷启动账本 | 未分离缓存的时间、事后选出的案例、固定游戏上的零误报 |

### 1.3 可保留与必须撤回的主张

| 主张 | 所需证据／当前处理 |
| --- | --- |
| 已观察到的 −1 是固定博弈内的负见证 | 定义、有限采样性质及逐条重放检查；可保留 |
| 未发现 −1 表示全局无害 | 有限采样不支持；不得使用 |
| B 比 A 更有效 | 必须由非 oracle、统一总预算的新实验决定；当前不支持 |
| MinPPI 是更好的通用特征选择器 | 必须面对简单强基线；现有总体结果不支持 |
| 压缩总体“无损”或统计非劣 | 不能由不显著差异或 validation 点估计 gate 推出 |
| 存在上下文见证即值得从 full 删除 | 不成立；用 RQ2 单独检验 |
| 理想 proposal 可提高检测概率 | 条件机制分析，可放附录；不能当作 learned guidance 的实证结论 |
| 能显著加速 NSGA-II | 需包含前置审计的独立冷启动成本；旧暖缓存数据不支持 |

## 2. 相关工作核验与必须纠正的归因

以下均按原论文或出版方核验，不以搜索摘要中的性能宣传代替本文证据。

| 工作 | 已有内容与本文关系 | 大修处理 |
| --- | --- | --- |
| Zheng、Raskutti，MinShap，2026，v2 | 已给出 minimum aggregation 的一般框架及特定假设下的冗余推断和统计程序 | 承认直接前驱；突出本文固定 quota 的负翻转审计，不宣称最小聚合创新，也不借用其统计保证。[原文](https://arxiv.org/html/2604.15107v2) |
| Lundberg、Lee，SHAP，2017 | 对预测进行加性归因；本文的 mean absolute TreeSHAP 是排序基线 | 不把负 SHAP 值等同有害性，也不把该实现叫平均重训效用边际。[原文](https://arxiv.org/abs/1705.07874) |
| Zhao、Jing Liu、Parilina，ShapG，2025 | 图结构用于 Shapley 近似；本文借用图信息构造见证 proposal | `.bib` 的 Jiapeng Liu 应改为 **Jing Liu**；补 EAAI 148, 110409 及 DOI。不能继承其效率结论。[出版方](https://www.sciencedirect.com/science/article/abs/pii/S0952197625004099) |
| Kulynych、Troncoso，Banzhaf pruning，2017 | 已研究二值分类器决策翻转及零影响特征的 lossless pruning | `.bib` 当前四位作者错误，应为 **Bogdan Kulynych、Carmela Troncoso**。其固定分类器、绝对翻转指数与本文重训效用负翻转不同。[原文](https://arxiv.org/html/1711.04992v2) |
| Lippl、Lindsey，2024 | 讨论多任务与 finetuning 中的 feature reuse、权重重缩放 | 不能作为“删掉有害输入特征改善单任务模型”的直接证据；删除这处动机归因，除非正文确实讨论其研究设置。[原文](https://arxiv.org/html/2310.02396v4) |

现有两个基线还存在名称与实现不符的问题：

- `src/minppi_mofs/baselines/min_shap.py` 计算受限采样下的最小连续效用边际，未实施 MinShap 原文的统计推断程序。历史结果应披露为 adapted minimum-marginal control，不能称忠实复现 MinShap。
- `src/minppi_mofs/baselines/banzhaf_fs.py` 只抽取较小联盟并提前停止，计算有符号平均效用边际，不是标准均匀联盟 Banzhaf，也不是 Kulynych–Troncoso 的绝对分类翻转指数。应披露为 restricted-proposal mean-marginal control。
- 本轮不回写这些代码或历史结果。MR2 先纠正文字和接口命名；若要进行真正的原方法竞争，必须另写忠实实现与验证计划，不能仅改标签后声称已复现。

这些修正属于研究完整性底线，不应等新实验取得有利结果才处理。

## 3. 固定博弈与见证契约

### 3.1 定义

令 D 为本次实际分析的有序特征集合，p=|D|。给定冻结的数据划分、预处理、学习器、模型随机种子和验证集，定义 U(S)。

\[
q=\alpha U(D),\qquad v_\alpha(S)=\mathbf 1\{U(S)\ge q\},\qquad
M_i=\min_{S\subseteq D\setminus\{i\}}[v_\alpha(S\cup\{i\})-v_\alpha(S)].
\]

主设置 α=0.95，要求 U(D)>0 且所有效用有限。条件不满足时报告 invalid-game 及原因，不悄悄换 split。二值与多分类使用相同 XGBoost 分类模型族及概率输出，utility 分别为 ROC AUC 与 macro OVR AUC；回归审计沿用 U=max(raw R²,0)，同时保存 raw R²。

空集合不是人为赋值 0：分类使用训练类别先验概率，回归使用训练均值预测，并经同一 utility 计算。阈值比较为 `>=`；gate 的数值容差不能混入博弈标签定义。

三类严格含义：−1 表示存在负翻转；0 表示不存在负翻转但也不是所有上下文均正向决定；+1 表示每个上下文均正向决定。“neutral”不等于无预测价值。

有限采样的最小值不小于精确最小值。因此已评估的 −1 可以认证该固定博弈中的存在性；有限样本的 0/+1 不能认证全局类别。GPU 或库版本导致的效用漂移也必须记录，不能将新的效用混入旧表。

### 3.2 见证输出

每条记录至少包含：game_id、目标特征、原始特征名及顺序、S 与 S∪{i}、两侧 U、两侧 raw metric、q、α、两侧与 q 的距离、model/split/proposal 标识、首次发现的累计查询数、协议与效用表哈希。

独立 verifier 只根据记录和对应效用重放判断：`U(S)>=q` 且 `U(S+i)<q`。全量重放失败必须使该运行验收失败，不能删除失败记录后继续发表。

## 4. E1：多真实数据集的完整精确参照

### 4.1 固定矩阵

| 数据集 | 面板规则 | 维数 | 数据划分数 | 冻结游戏数 |
| --- | --- | --- | --- | --- |
| Abalone | 原八特征 | 8 | 5 | 5 |
| Spambase | MI top8；MI top4＋random4 | 8 | 5 | 10 |
| Madelon | MI top8；MI top4＋random4 | 8 | 5 | 10 |
| Credit-Card-Fraud | MI top8；MI top4＋random4 | 8 | 5 | 10 |
| 合计 | 7 个数据集–面板组合 | 8 | 5 | 35 |

split 标识为 `[42,123,456,789,1024]`。面板仅从训练侧按现有预处理得到的特征池产生：MI top4 以外的所有剩余特征中无放回抽 random4，不排除 top5–8。两面板偶然相同也保留并披露，不重抽。MI 并列按原始列顺序决定；面板输出仍按原始列顺序排列。

这些是真实数据上的**重新定义八维游戏**：q 相对于各面板 U(D) 计算。它们的 exact labels 不是原 100 维游戏的全局真值。两种规则都含 MI 偏好，不能外推成任意特征面板的结论。

E1 仍用 64% train／16% validation／20% test 划分，但不访问 test 标签或测试结果。分类按现有协议分层；固定划分下不能为增加有害标签而重新抽样。

模型沿用当前 100 trees、depth=4、learning_rate=0.1、n_jobs=1 的 XGBoost 配置；其余有效参数、包版本与 backend 必须完整写入 resolved protocol。CPU/CUDA 不自动混用；MR4 前做环境检查后锁定。不得为了缩短时长擅自降到 10/20/40 trees。

### 4.2 随机性与缓存身份

分别命名 `split_seed`、`model_seed`、`panel_seed`、`mi_seed`、`schedule_seed`、`method_order_seed`，禁止仅换旧 bundle 的一个 seed 冒充同一游戏内的采样重复。

设计规则：split_seed 使用上述五个标识；其余种子按下表所列字段的规范 JSON 经 SHA256 前四字节 little-endian 导出 uint32。domain 区分各随机流；规范 JSON 键排序、UTF-8、无额外空白。固定 RNG 为 NumPy PCG64，并记录库版本。

| 随机流 | 哈希字段（均包含 dataset、split_id） | 是否随 schedule repeat 改变 |
| --- | --- | --- |
| model_seed | domain=`model` | 否，同数据集/划分的各面板使用相同模型 seed |
| panel_seed | domain=`panel`，panel_id | 否 |
| mi_seed | domain=`mi`，scope=`full_pool` 或具体 panel_id | 否，同一面板指导固定 |
| schedule_seed | domain=`schedule/A` 或 `schedule/B`，panel_id，repeat_id | 是，仅此流随采样重复改变 |
| method_order_seed | domain=`cold_method_order`，panel_id | 否，决定该冷启动块的 A/B 顺序 |

不把 repeat_id 或 method 字段注入模型、面板和 MI 的哈希。A/B 在相同 game/repeat 配对，但使用不同 proposal-domain 流；30 repeats 必须共享同一效用表和指导信息。

每个游戏做 30 个 schedule repeats，标识 0–29。它们不重新训练模型、不重新划分数据，共 1,050 对 A/B 轨迹，即 2,100 条方法轨迹；不是 1,050 个独立数据集。

### 4.3 真值与盲查询隔离

- Reference builder 完整计算 256 个集合效用。单游戏最多 255 个成功的非空 unique fits；35 个游戏最多 8,925 个。full 已包含在 255 内，不能再加一次；不含可选冷启动重复。失败/重试 attempts 另计，不能被这个成功拟合上限掩盖。
- 当前 exact label estimator 找到 −1 后早停，不能据此宣称全表完整。新 builder 必须检验 256 个唯一 mask，无缺失、无重复冲突、数值有限。
- Query runner 只接收 D、q、自己的训练侧指导信息和按 mask 查询的接口；不得访问完整表、exact labels、全缓存键列表或未查询见证。
- 统计评估器在采样结束后才合并 exact labels。标签准确率与 oracle 建表时间分开汇总。
- 历史 Abalone 表只有数据、特征、划分、模型、效用等身份完全一致且确为全表才能复用；否则重新构建，不修改旧表。

### 4.4 统一总查询预算：明确改变的是 runner，不是 proposal

本设计选择**统一 round-robin 总预算 runner**。它不同于历史 feature-major／每特征 K 的图，必须使用新协议标识，不能直接与旧曲线拼接。

1. A/B 分别为每特征生成最长 128 次的固定采样日程。A 仍是先均匀抽 coalition size、再该大小内均匀抽子集；不同 trial 允许重复。B 沿用现有训练 MI＋相关树及扰动 proposal，具体常数在 MR2 逐项写成伪代码。
2. 按原始特征顺序轮转，每次为一个尚未发现见证且日程未耗尽的特征评估一个 marginal；发现 −1 后跳过该特征，把后续槽位留给其他特征。每特征实际最多 128 次。
3. 一次 marginal query 包括 U(S) 与 U(S+i) 两个逻辑 utility 请求。重复集合照计 marginal query；distinct utility 和实际拟合另计。
4. 统一总预算检查点 Q 为 `8×[2,5,10,25,50,100]`，即 `[16,40,80,200,400,800]`。Qmax=800；full quota 初始化不算 marginal query，但计入 utility 与成本账本。
5. 全部特征已找到见证或日程耗尽时可以早停；余下预算的 recall 曲线保持最终值。报告实际 Q 和停止原因，不虚构未发生的查询。

初始化时，各方法账本登记一次 full utility，独立 seen-mask 集预置 full；后续请求 full 仍增加 utility-request 数，但不重复记 unique fit。所有方法采用同一规则，不能免费提供 quota 后漏计 full 成本。

比较的是相同总查询**上限**，不是要求方法浪费计算以获得相同实际耗时。128 次有放回抽样不等于枚举 128 个上下文。

### 4.5 主要终点与汇总

设 H 为 exact harmful 集。仅当 |H|>0 时，R(q)=已发现真实有害数/|H|。主要终点为每条轨迹的归一化 recall area：

\[
A_m=\frac{1}{800}\sum_{q=1}^{800}R_m(q),\qquad \Delta A=A_B-A_A.
\]

面积按逐查询事件计算，不用六个绘图点插值。面积高表示在给定预算范围内更早、更完整地发现见证；不意味着训练更快。主要比较只有 B−A 这一配对差。

次要终点：Q=800 的 recall、首次见证成本、达到 80% recall 的成本、见证集合大小和 proposal 质量。未达到目标记右删失；不能填成“刚好在 800 达到”。F1 保留兼容表，但在 precision=1 时不作为额外独立发现。

H 为空时 recall 与面积均为 NA，完整列出空游戏数量、exact 三类分布、报警数及误报数。不得填 recall=1 纳入 headline；不补选替代面板。固定同表上的误报应为 0，这是正确性不变量，而非有价值的竞争优势。

聚合顺序：同游戏平均 30 schedules → 同面板平均有效 splits → 同数据集等权平均面板 → 有定义的数据集等权宏平均。每层同时展示分母、空率和无效原因；某层无正类游戏则 NA，不隐性填零。主图仍展示每个数据集的效应，不能只保留总平均。

只报告固定游戏条件下的 schedule Monte Carlo 标准误及 split 范围；不得称其为新数据集泛化置信区间。四个数据集的双侧精确 sign-flip 最小 p 为 2/16=0.125，本设计不以统计显著性为 E1 验收门槛。

### 4.6 次要分析，不追加模型拟合

完整效用表可用于 α=0.90、0.99 的敏感性分析，但不能据此更改主 α。报告 quota 附近的翻转距离和所有见证集合的大小分布。

对 A，可从全表精确计算负见证在真实 size-uniform proposal 下的概率质量；不能拿 uniform-subset 密度替代。对 B，只能在明确其完整采样分布后计算，或用不查询标签的独立 proposal 抽样估计并报告误差。oracle 答案不能反过来调整 guidance。

## 5. E2：受控机制与解释案例

E2 分两层，避免再次用人为给定的答案证明指导算法好：

- 解析布尔游戏：覆盖无负见证、单一稀有见证、多个负见证及交互情形，验证枚举标签、采样支持、停止与计数。原五特征 oracle 例子只在这一层出现，不进 A/B 实证汇总。
- 非 oracle 合成表格：加性／冗余、交互、干扰变量三类生成机制。具体分布、样本量与参数先在 MR2–MR3 单独冻结，再生成数据；所有 MI／相关信息只来自训练集，harmful labels 仍由完整效用表得到。植入噪声不是自动的 harmful truth。

第二层不是本次 35 游戏的隐藏追加项，未经生成协议审查不运行。不能看 E1 后只构造让 B 获胜的分布。

解释案例从 E1 按预定规则选择：每种数据集–面板按已列 split 顺序取第一个有效游戏，固定 schedule_repeat=0，取 A 按实际事件顺序首次发现的见证；无见证则展示未发现情况，不换 repeat。另展示该游戏最接近 quota 的一条真见证作为敏感性案例：距离定义为 `min(U(S)-q, q-U(S+i))`，取最小值，并列按目标特征原始序号、S 的整数 mask 升序选择。无真见证则不构造案例。后一案例明确标为 exact 事后诊断，不混同事前采样成果；两类都不据测试集选择“好看”的案例。

## 6. E3：查询、冷启动与复用三本账

| 账本 | 可回答的问题 | 必须计入 |
| --- | --- | --- |
| 预计算表上的盲查询 | 哪种 proposal 在总查询预算内更完整 | marginal requests、utility requests、distinct masks、停止原因 |
| 独立冷启动 | 为得到一次审计或删除结果，真正付出多少时间 | full fit、MI/相关图、SHAP 解释、LOFO、子集拟合、预测、gate、最终 refit、I/O |
| 部署暖缓存 | 已有指定缓存时，额外执行的成本 | 来源哈希、命中、新增拟合及 wall-clock |

预计算 replay 可以维护每方法独立 seen-mask 集，给出反事实 unique-fit 数；这不是实测冷启动耗时。真实冷启动各方法使用独立空 store，允许方法内部复用；不共享 oracle builder 的全表缓存。

若物理共享共同 full 模型，独立成本账本仍分别归属一次，同时报告实际 campaign 的去重成本，不能把归属成本冒充实际 GPU 时间。主 cold wall-clock 使用确实独立执行的数据，不按 query 比例摊派旧时间。

MR4 冷启动验收仅选 Spambase、Madelon、Credit 的 MI top8、split42、schedule0，A/B 各一次并随机化执行顺序。每方法的非空 unique fits 上限 255，六次合计最多 1,530；它是计数与隔离验收，不足以形成稳定速度结论。

## 7. E4：使用者决策实验

### 7.1 历史 matched-k 与新策略比较分开

历史同删除数量比较继续作为描述性证据，重新标明 adapted controls。它只能回答“固定 k 时选哪些特征较好”。

新 E4 回答“在同一效用容忍规则下，哪个完整策略更值得使用”。它不是等计算预算比较，必须报告每种策略的全成本，不把成本不同藏在同一个 K 中。

### 7.2 选择、独立 gate 与 test

建议新划分为 64% train、8% V_select、8% V_gate、20% test；先按固定规则分出 80/20，再在非 test 的80%内部按80/10/10比例划分。分类分层，保存精确行号，禁止为补足有利结果换 seed。

- 预处理和 MI 只拟合 train。MinPPI 的游戏 U 只用 V_select。
- 每方法在 V_select 决定一个候选，V_gate 不用于排名、调参或候选搜索。
- 各方法只向 V_gate 提交这一个候选；拒绝即 full，不尝试第二个候选。
- 最终子集冻结后，统一在非 test 的 80% 上 refit；预处理参数仍保持原 64% train 的拟合值，再评价 test。

E4 与 E1 验证集不同，因此不能把 E1 的 exact labels 或模型效用当成 E4 的同一游戏结果。

### 7.3 共同约束与候选策略

定义政策评价指标 M：分类为对应 AUC，回归为 **raw R²**。选择与 gate 使用同一操作性容忍度：分类 ε=0.005、回归 ε=0.01；要求 `M(S)-M(D)>=-ε-1e-12`，两侧使用同一阶段评价集。LOFO 排名、前缀可行性、portfolio 并列选择与 gate 全部用 M，只有 MinPPI 审计标签用 U。回归 M 不截断，是为了避免零截断掩盖差模型损失；这与审计 U=max(R²,0) 明确分离，属于新协议变化。

这些 ε 沿用旧研究量级以避免结果导向调参，不是领域公认“无损”标准。独立 gate 也只是有限样本的点估计规则，不是统计非劣保证。

| 策略 | 在 V_select 上形成唯一候选的规则 |
| --- | --- |
| Full | 不删特征，基准 |
| MinPPI-A，主要审计策略 | round-robin 总 marginal 上限 25p，每特征日程上限128；删除所有发现 −1 的特征；若选择约束失败则 full，不搜索 harmful 子集 |
| MinPPI-B，次要策略 | 同一总预算上限与删除规则，仅 proposal 不同；不能按 test 结果改为主要策略 |
| MI | 按训练 MI 从小到大删除，评估规定前缀 |
| TreeSHAP | 同一 full 模型，在固定训练行上计算 mean absolute TreeSHAP，低分先删；说明多分类归约及背景设置 |
| 静态 LOFO | 对每个 i 计算 V_select 上 M(D\{i})−M(D)，从大到小删除；分数只计算一次，不叫 backward elimination |
| validation-only portfolio，主要实用对照 | 合并 MI、TreeSHAP、LOFO 的可行前缀；取最大压缩，依次按更高 V_select 指标 M、固定方法顺序 MI→SHAP→LOFO 打破并列；只向 gate 提交一次 |

所有 rank 方法使用统一删除数量集合：`{0,1,p} ∪ {ceil(j*p/10): j=1,...,9}`，去重并排序。八维时覆盖全部数量；高维最多 12 个前缀。每个单一 rank 方法选择满足 V_select 的 M 约束的最大删除量；若只 full 可行则提交 full。排名分数并列按原始特征顺序处理。不能因首个失败就停止，因为效用不单调。所有方法允许空集合，并使用相同常数预测规范；不存在对 MinPPI 特有的空集特判。

SHAP 的背景/行数等实现细节在 MR2 的接口与 MR5 resolved protocol 中显式固定。portfolio 必须计入三个 ranking 的完整构建成本，不是事后按 test 挑出的“最佳基线”。

### 7.4 输出、范围与分析

报告所有单元，包括无候选、选择拒绝、gate 拒绝和有效删除，不能只分析最终有压缩者。

- 压缩 C=删除数/p；按数据集的 test ΔAUC 或 Δraw R²；测试损失超过 ε 的观测比例。
- 次要联合量 `C × 1{Δmetric_test >= -ε}`，称“满足预定测试容忍度的压缩量”，必须和违反率同时展示，不能独自当安全得分。
- 与 full 及 portfolio 的配对差、完整冷启动时间、distinct fits、选择/gate/refit 分项。
- 不平均 AUC 与 R² 的原始差值作总体 headline；总体最多采用数据集等权压缩及无量纲的容忍度内结果，同时保留各数据集明细。

主要策略比较固定为 MinPPI-A 对 portfolio；B 与各单一基线是次要分析，不依据结果更换主比较。旧六数据集若重跑，统一采用 25p 总预算，包括 ISOLET；若资源不足，暂不启动 E4，而不是静默给 ISOLET 降预算后仍称完全一致。

本设计没有授权立即运行六数据集 E4。MR4 先验收低维/小规模流程并实测成本，MR5 再冻结全维数据范围、训练重复、SHAP 有效参数、独立确认数据和完整计算上限。新外部数据尚未选定，因此目前**不能宣称确认协议定稿**；这是明确的后续阶段门槛，不用临时凑几个 seeds 代替。

NSGA-II 保留为次要应用。只有 E4 有可解释的收益或值得检验的条件假设时，再设计包含前置成本、统一分母和冷缓存的搜索比较；本轮不重跑历史全部搜索。

## 8. 小试规模、恢复与停止规则

| 批次 | 运行内容 | 成功的非空 unique fits 上限 | 启动条件 |
| --- | --- | --- | --- |
| P1 | Spambase/Madelon/Credit，各 MI top8、split42 的完整表 | 765 | MR3 单元与集成验收完成，获得运行许可 |
| P2 | 三游戏 A/B 各 schedule0 的独立冷启动 | 1,530 | P1 全表、见证及计数正确 |
| P3 | 扩展至35游戏、每游戏30 schedules | 全部表总上限8,925；已完成且身份一致的 P1 包含在内 | 成本报告通过，另行批准扩展 |

每批建议 wall-clock 上限：P1 六小时、P2 十二小时；达到上限后完成当前可安全落盘单元并暂停，不抛弃已完成结果。长单次拟合不会承诺硬时限中断。P3 的时间上限在 P1/P2 实测后书面设定，不用旧平均时间承诺几天跑完。

每个 coalition 完成后原子保存，保存协议哈希、mask、utility、计数及状态；重启只补缺失单元。失败项保留错误记录，最多一次同参数重试；仍失败则停止该批验收，不靠换模型/seed 绕开。上表是成功 unique fits 上限，不是 fit attempts 上限；每个 mask 最多两次尝试，分别保存 attempts、失败、重试及耗时，均占用批准的 wall-clock 预算。若发生过重试，报告实际总尝试量，不能仍声称只进行了表中的模型调用数。

以下情况停止扩大规模，或按说明停止相应研究主张：

1. 标签与见证重放不一致、全表缺失、oracle 信息泄漏或成本不守恒。
2. 固定游戏重复计算出现跨 quota 漂移，尚未明确数值/随机性原因。
3. 实测资源超过批准上限。报告已完成/未完成分母，不以部分有利结果替代预定矩阵。
4. B 缺少稳定收益：停止为 B 必须获胜而调参或升级优势主张，记录负结果；是否研究新 proposal 属于新设计。不能因此截断已经批准的 P3 固定矩阵。
5. 相对简单策略没有决策收益：保留审计结果的独立价值，撤回通用选择器/加速器主张；不以效果方向中止已批准的固定比较。

禁止以 p<0.05 为停止条件，不删除不利 seed、不将空游戏替换、不按效果决定加 d=10。维数扩展只可基于事先批准的资源与科学覆盖需求；它不是本次核心任务。

## 9. MR2–MR3 接口和验收边界

优先复用现有模型、预处理、MI-correlation、采样日程、缓存身份和恢复组件；不顺便重写全部 pipeline。

| 组件 | 新职责 | 必须通过的验收 |
| --- | --- | --- |
| Complete reference builder | 面板完整效用表 | 256 masks、哈希、empty/full、幂等恢复、无 test 依赖 |
| Blind query runner | 统一 round-robin 与逐查询日志 | A/B 总预算守恒、重复计数、无 oracle 可见、相同日程恢复 |
| Witness verifier | 逐条认证固定游戏负翻转 | 全量真值核对、阈值边界、非法 mask 拒绝 |
| Cost ledger | 三类账本独立 | full 成本、guidance、warm 命中、真实拟合、重试与恢复不双计 |
| Policy evaluator | 一候选＋独立 gate | 选择/gate/test 隔离、拒绝 full、前缀完整评估、空集对称 |
| Report generator | 层级统计与证据图 | 空正类 NA、数据集等权、右删失、旧新协议禁止混报 |

MR2 需要写出编号 Algorithm 1（完整审计与见证输出）、Algorithm 2（A 的 proposal 与预算调度）、Algorithm 3（B 的指导构建、采样与调度），并在正文或附录给出唯一候选 gate 的确定规则。不得省略 B 的 top-half MI、邻居截断、drop/add 分布、seed 填补顺序及重试/早停。

历史 hybrid 不进入 A/B 主比较；保留旧结果读取能力，并在旧协议说明其 exact/guided/two-branch 条件及合计预算。新增功能应通过可选版本化路径接入，不能使旧 run 的恢复语义改变。

## 10. ICOMP 格式、论文结构与 rebuttal 分流

2026-09-10 查核的 [ICOMP 官网](https://www.icomp.cc/) 要求单 PDF、主文9页、参考文献与附录不限、双盲，通过 OpenReview 提交。当前页面列作者回复9月7日开始、9月14日结束。页面未给本稿回复字数上限，也不能据此确认是否允许替换 PDF 或添加新实验；须以作者账号内具体 invitation 为准。官网和公开 OpenReview 的部分投稿日期不一致，不能把公共时间表当作本稿唯一操作依据。

用户已确认这是正式评审，且本项目当前最终 PDF 就是送审版本；已按该 PDF 核对 review 涉及的摘要、§2.3、§4.1–4.4、表格与图注。证据 SHA256 见工作台账。同时准备 [REBUTTAL_DRAFT.md](REBUTTAL_DRAFT.md)，不自动提交，不承诺能用长期新实验赶上本轮回复。用户提供回复字段 `max length: 2500`，暂按字符而非单词保守处理；单位、附件与修改稿权限仍以作者账号内 invitation 为准。

建议9页主文结构：

1. 问题、动机与相关工作：明确 minimum 前驱和本文边界。
2. 固定 quota 博弈、见证语义与有限采样性质。
3. 编号 A/B 算法与统一预算、成本定义。
4. 多精确参照上的覆盖与失败机制，包含空游戏。
5. 同验证约束的删除决策与简单强基线。
6. 解释案例、成本、限制与结论。

主图优先为非 oracle recall–query 图、压缩–测试损失图、成本分解。解析 oracle 图和可选 NSGA-II 放附录，除非新证据表明其确实是主贡献。局部收益、总体效应与强基线比较在同一证据段内解释，不在摘要挑选最大效应替代总体结果。

## 11. 本阶段验收与下一步

MR1 已形成：主问题与贡献边界、文献纠错清单、35游戏精确验证设计、总查询主终点、成本隔离、新决策政策和有限小试门槛。

尚未完成：用户对本书面细化协议的审阅；2500长度限制的单位与回复权限核对（已有保守字符版）；MR2 伪代码与接口实现计划；MR3 基础设施；任何新训练或新实验结果；MR5 的独立确认数据及最终资源冻结。送审 PDF 内容核对已完成，不在待办中重复要求用户提供。

下一步先审阅本设计。确认后进入 MR2，按“修正相关工作与算法说明 → 对齐伪代码/接口 → 小规模单元验证”推进，再进入 MR3。不能把“设计完成”报告成“review 五项已解决”。
