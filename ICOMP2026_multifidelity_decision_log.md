# ICOMP 2026 多保真实验决策日志

> 状态：持续更新。本文档保存从快速决策实验到正式主表之间的实测事实、
> 被淘汰方案、最终参数、架构选择和论文结论。除非明确通过验证集冻结门，
> 本日志中的结果均为 provisional / validation-only，不得写成最终测试集结论。

## 1. 时间与目标

- 当前日期：2026-07-28。
- ICOMP 2026 最新截止时间：2026-08-05。
- 可用资源：1 张支持 CUDA 的 GPU，可连续运行。
- 最高级硬约束：最终框架必须能够从空结果开始，在 48 小时内完成可写入
  论文的完整证据链。
- 当前目标：先在所有 7 个数据集上获得可解释的方向性结论和真实耗时，
  再在 48 小时硬预算内联合冻结数据集、预算、seeds 与各证据模块。
- 约束：方向筛选阶段不得访问测试集；Git 提交由用户完成。

## 2. 当前候选总体架构

实验被拆成以下互不混淆的层级：

1. `calibration`：seed 42，只读取 train/validation，选择每个数据集的
   XGBoost 轮数。
2. `decision`：seed 42、所有 7 个数据集、K = 2/5，快速判断方向、稳定性、
   harmful witness 与验证集移除效果。
3. `freeze gate`：人工检查 decision report；在此之前测试集保持密封。
4. `formal`：原候选为 K = 10/25/50、5 seeds；实测成本表明该全组合无法
   在 48 小时内完成，最终 formal 分层方案尚待 decision 结果冻结。
5. `sensitivity`：K = 100，仅在主表结论需要时执行，不属于默认正式主表。
6. `campaign aggregation`：只允许相同模型策略、科学设置和 Python 源码指纹
   的不重叠 seed shard 合并。

关键原则：

- 分类任务始终使用同一 XGBoost 概率模型，utility 只决定概率的评分方式。
- 模型策略和 decision K 都进入协议哈希。
- decision 结果不能与 formal 主表混合。
- 断点恢复复用 SQLite coalition cache，不重新执行已完成物理拟合。
- 正式证据必须区分：
  - 全数据集广度证据；
  - 多 seed 稳定性证据；
  - 代表数据集高预算收敛证据；
  - removal 与 NSGA-II 的方法效果证据。
  不要求四种证据都使用同一个最昂贵的笛卡尔积。

### 2.1 已被否决的原始全组合

ISOLET 在 100 轮模型、K=2/5 下的前段实测约为每个特征 32 秒。由于 f=100
时 hybrid 同时需要 random 与 guided，成本近似随最大 K 线性增长：

- K=5 约 1.5–1.8 小时/seed；
- K=50 约 15–18 小时/seed；
- 仅 ISOLET 的 K=50 × 5 seeds 就约需 75–90 小时。

因此“7 数据集 × 5 seeds × K=50，再加 removal、alpha、NSGA-II、ablation”
不是 48 小时框架，不能继续作为默认执行方案。后续必须采用分层主表和
自适应追加，同时保留公平比较、测试集密封和来源可审计性。

### 2.2 48 小时正式协议候选（待 decision 结果冻结）

当前最有希望的候选不是减少主表数据集或 seeds，而是拆分证据精度：

- 核心 estimator（初始候选）：7 datasets × 5 seeds，嵌套 K = 2/5/10；
- Credit K5 全 neutral 后的修订候选：ISOLET 使用 K=2/5/10；其余
  非 exact 数据集使用 K=2/5/10/25；Abalone 使用 exact；
- 高预算收敛：ISOLET seed 42 从 K=10 晋级到 K=25；
- removal 主表：7 datasets × 5 seeds；
- alpha sweep：优先复用已缓存 coalition 概率，避免按 alpha 重训；
- NSGA-II 主表：两种变体都使用 `population=20`、`evaluations=400`
  （20 代，包括初始种群的现有计数约定需在实现测试中再次核对）；
- NSGA-II 800-evaluation sensitivity：只在 seed 42 或信号最强的代表
  数据集追加，并保存 200/400/800 嵌套收敛 checkpoint；
- guidance/utility/dispatcher ablation：seed 42；guidance 的额外物理
  拟合只在任务与信号有代表性的 3 个数据集运行。

协议对齐发现：

- 论文主文件声明 random-matched control 为 10 trials，而当前 YAML 是 30；
  正式配置应改回 10，避免实现与论文不一致。
- 论文主文件声明 NSGA-II 为 20 individuals × 20 generations，而当前 YAML
  是 population 40、evaluations 800；正式配置应采用 population 20、
  evaluations 400，并用 seed-42 800-evaluation checkpoint 做收敛敏感性。

按 calibration 的 100 轮 CUDA `seconds_per_fit` 和近似 `4 × f × K`
估算，统一 K10 的 5 seeds 约 24.2 小时；Credit K5 全 neutral 表明统一
低预算可能损害核心 witness 结论。改成“ISOLET K10、其余非 exact 数据集
K25”的 5-seed estimator 约 29–31 小时，再将 ISOLET seed42 晋级 K25
约增加 5.5 小时。它比统一 K10 略贵，却显著增强所有便宜数据集的检出
证据。最终预算需等待 Credit K10/K25 定向检查和 CPU/CUDA 基准后冻结。

### 2.3 “全流程”完整性缺口

当前 manifest pipeline 的 removal 阶段包含 SHAP、MinShap、Banzhaf 与随机
matched controls，NSGA-II 阶段包含 full/prefilter HV，但它尚未把论文主表
声明的六个 ranking baselines 作为一个正式 stage 纳入。尤其是 ShapG 与
Powershap 仍只存在于旧的 `experiments/run_main_table.py`。

旧脚本不能直接成为新论文证据，原因包括：

- 自己再次切分训练数据，没有复用冻结的 train/validation/sealed-test；
- 不进入正式 manifest、协议哈希、状态恢复和 test-access audit；
- 与当前 per-dataset 100-round 模型策略及统一概率 utility 边界不一致；
- MinShap/Banzhaf 又按旧预算重复训练，不能复用 removal 的 ranking；
- NSGA-II ranking 又会独立重新搜索，不能复用公平 full front。

因此 48 小时架构必须增加一个可恢复的 `ranking` 证据阶段：

- SHAP、MinShap、Banzhaf 排名直接复用 removal selection plan；
- NSGA-II 排名从已经冻结的 full validation front 派生；
- 只新增 ShapG 与 Powershap 的训练成本；
- 所有方法的 top-k 子集在 validation 冻结后，只对每个冻结子集做一次
  sealed-test 评估；
- 统一输出 7 datasets × 5 seeds 的 AUC@k/R²@k、Friedman/Nemenyi 表。

这一步完成前，即使现有 `figures` stage 成功，也不能称为论文实验全流程。

旧主表 `results/tables/2026-07-10_12-20-33/main_metrics.csv` 只能用于成本
估计，不能混入新论文结果。其 35 个 dataset-seed 的 importance 计算实测为：

| 方法 | 总 importance 时间 |
|---|---:|
| SHAP-Top-k | 24.5 s |
| Banzhaf | 924.2 s |
| MinShap | 761.9 s |
| ShapG | 8,494.9 s（2.36 h） |
| Powershap | 197.9 s |
| NSGA-II ranking | 3,477.7 s（0.97 h） |

旧脚本从 15:20 到 19:16 的完整 wall time 约 3.94 小时。新 ranking stage
复用 Banzhaf、MinShap 与 NSGA-II 后，新增主成本是 ShapG 与 Powershap；
暂按 2.5–3.5 小时纳入 48 小时 P0 账本，之后必须用新统一模型边界的 smoke
重新校准。

### 2.4 预算冻结门槛（防止按结果挑参数）

最终低预算不能仅因“跑得快”而被采用。正式实现应预先冻结以下门槛：

- `K10 harmful recall relative to seed42 K25 >= 0.90`；若 K25 harmful 为空，
  记录为无事件而不是伪造 100% recall；
- K5→K10 与 K10→K25 报告三值标签稳定率和 harmful-set Jaccard；
- nested minimum 下已发现的 harmful 不应撤销；若出现撤销，视为 schedule、
  cache 或实现错误；
- removal 的 paired test utility 需满足预设非劣界：分类 utility 的绝对
  delta 不低于 `-0.005`，回归 R² 的绝对 delta 不低于 `-0.010`；
- NSGA-II 400 checkpoint 的 validation HV 应达到 seed42 800 checkpoint
  的预设比例（候选 0.99），否则只对未收敛数据集追加；
- 所有未达到门槛的数据集仍保留并报告，不能按正负结果删除。

这些门槛应由代码生成机器可读的 gate report。自适应晋级只由门槛触发，
不由是否支持论文观点触发。

### 2.5 可证明无损的 harmful 吸收态优化

MinPPI checkpoint 是已观察 marginal 的最小值，且每个 marginal 只可能为
`-1/0/+1`。一旦某特征观察到 `-1`，任何后续 trial 都不可能改变其当前或
未来 checkpoint 的最小值。因此可在 random/guided 的单特征 schedule 内：

1. 首次出现 `-1` 后停止该特征的剩余 coalition 拟合；
2. 将所有更高预算 checkpoint 精确填为 `-1`；
3. 保留原 schedule digest，并记录
   `absorbing_state_skipped_evaluation_count`；
4. 用测试证明优化前后每个 checkpoint 的三值 scores 完全相同。

该优化不改变模型、采样分布、预算定义或论文结果，只删除数学上不可能影响
输出的工作。收益取决于 harmful 特征数量和 witness 首次出现位置：对全
neutral 数据集收益为零，对 Madelon/Credit-Card-Fraud 等 harmful-rich
数据集可明显降低 K10/K25 的成本。

## 3. 代码验证证据

- 全套测试：`1212 passed, 3 skipped`，耗时 132.94 秒。
- 15 条 warning 均为既有测试的 SQLite `ResourceWarning`，没有测试失败。
- 多保真协议、校准、筛选专项回归：`210 passed`。
- decision budget、bundle 与 CLI 专项回归：`361 passed`。
- XGBoost checkpoint 等价性：
  - 一次训练后读取 2/4 轮树前缀；
  - 与分别训练 2 轮、4 轮的预测逐元素一致；
  - 数值标准：`rtol=0, atol=1e-12`。

## 4. 预处理事实（seed 42）

| 数据集 | Train | Validation | Sealed test | 选中特征 |
|---|---:|---:|---:|---:|
| Spambase | 2,944 | 736 | 921 | 56 |
| Madelon | 1,664 | 416 | 520 | 100 |
| ISOLET | 4,989 | 1,248 | 1,560 | 100 |
| Gisette | 4,480 | 1,120 | 1,400 | 100 |
| Credit-Card-Fraud | 182,276 | 45,569 | 56,962 | 29 |
| CrimeData | 1,417 | 355 | 443 | 100 |
| Abalone | 2,672 | 669 | 836 | 8 |

正式 calibration 输入：

`results/icomp2026_multifidelity/bootstrap-s42-20260728-v2`

## 5. 模型轮数 calibration 结果

候选轮数：10、20、40、100。所有候选共享同一个 100 轮树前缀，每个非空
coalition 只拟合一次；相比四次独立训练，物理拟合量约减少 75%。

- 总 calibration 时间：1,753.4 秒（29 分 13 秒）。
- 数据：仅 train/validation。
- 结论：7 个数据集全部保留 `n_estimators = 100`。
- 冻结策略：
  `results/icomp2026_multifidelity/bootstrap-s42-20260728-v2/calibration/model_policy.json`
- 详细表：
  `results/icomp2026_multifidelity/bootstrap-s42-20260728-v2/calibration/calibration_summary.md`

40 轮相对 100 轮的关键证据：

| 数据集 | Full utility loss | Spearman | Winning agreement | Marginal sign | Harmful recall | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Spambase | 0.004940 | 0.9899 | 0.9822 | 0.6696 | 0.0000 | 保留 100 |
| Madelon | 0.000000 | 0.9921 | 0.9551 | 0.7100 | 0.3333 | 保留 100 |
| ISOLET | 0.001802 | 0.9789 | 1.0000 | 0.6750 | 1.0000 | 保留 100 |
| Gisette | 0.003973 | 0.9588 | 1.0000 | 0.6650 | 1.0000 | 保留 100 |
| Credit-Card-Fraud | 0.000533 | 0.9307 | 0.9915 | 0.6897 | 1.0000 | 保留 100 |
| CrimeData | 0.003166 | 0.9961 | 0.9800 | 0.6550 | 0.5000 | 保留 100 |
| Abalone | 0.015406 | 0.9497 | 0.9333 | 0.8125 | 1.0000 | 保留 100 |

解释：

- 低轮数经常几乎不损失完整特征集 utility，但不能稳定保持局部边际符号。
- Spambase、Madelon、CrimeData 的少数 harmful 事件在 40 轮下召回不足。
- 因此仅根据完整集 AUC/accuracy/R² 选择低轮数会破坏本文真正研究的
  coalition/harmful 结构；强制降到 40 轮不符合“不牺牲论文结论”的要求。

## 6. Policy smoke 结果

运行：

`results/icomp2026_multifidelity/policy-smoke-s42-20260728-v1`

- 完成状态：成功。
- 总 wall time：117.7 秒。
- Spambase，K=2：443 physical fits，6 cache hits，约 1 分钟。
- Abalone exact：255 physical fits；随后随机/引导步骤主要命中缓存。
- 证明：100 轮冻结策略、CUDA、分组 bundle、SQLite cache、逐特征进度和恢复
  边界可以共同工作。

## 7. 为什么 decision 使用 K=2/5

实测 100 轮平均单次拟合时间中，ISOLET 约为 3.48 秒。分组 random + guided
在每个特征上大致产生 `4 × K` 次物理拟合：

- K=25：ISOLET 约 10,000 fits，单数据集接近 10 小时；
- K=5：ISOLET 约 2,000 fits，约 1.9 小时；
- 七数据集 K=2/5 总体预估约 2–3 小时。

因此：

- K=10/25 仍不是“快速”层；
- K=2/5 能给出两个 checkpoint 的稳定性、真实 threshold-crossing witness
  和全数据集方向；
- K=2/5 明确标记为 `run_kind=decision`；
- 最终主表仍使用 K=10/25/50，不降低正式证据强度。

## 8. 当前运行

- Decision run：
  `results/icomp2026_multifidelity/decision-s42-k2-5-20260728-v1`
- Seed：42。
- 数据集：全部 7 个。
- 模型：每个数据集 100 轮。
- Budget：K=2/5。
- 状态：validate 与 preprocess 已完成；estimator 正在串行运行。
- 测试集状态：密封，尚未用于 calibration 或 decision。

### 8.1 已完成的数据集

#### Spambase

- Bundle runtime：180.65 秒。
- Physical fits：1,109。
- Cache hits：10。
- K=2：harmful 0、neutral 56、decisive 0。
- K=5：harmful 0、neutral 56、decisive 0。
- K2/K5 三值标签稳定度：100%。
- 当前解释：在极低 decision budget 下没有检测到非中性特征。这只能记为
  “稳定的低预算 neutral 方向”，不能解释为正式 K50 下不存在 harmful 或
  decisive 特征；若其他数据集也大量全 neutral，需要在 freeze gate 判断
  K=5 是否过低并追加一个 K=10 定向检查。

#### Madelon

- Bundle runtime：330.77 秒。
- Physical fits：1,931。
- Cache hits：14。
- K=2：harmful 6、neutral 94、decisive 0。
- K=5：harmful 15、neutral 85、decisive 0。
- K2/K5 三值标签稳定度：91%。
- K2 的 6 个 harmful 全部包含在 K5 的 15 个 harmful 中。
- Harmful 集合：K2/K5 intersection = 6，union = 15。
- 当前解释：随着预算增加，已检测 harmful 集合单调扩张，且没有撤销 K2 的
  harmful 判断。这是支持方法方向的第一条强证据，但具体 coalition witness
  与移除效用仍需在全部 estimator bundle 完成后运行 validation-only screen
  report 才能确认。

#### ISOLET

- Bundle runtime：6,506.37 秒（108.44 分钟）。
- Physical fits：1,985。
- Cache hits：16。
- Random K=2/K=5：均为 harmful 0、neutral 100、decisive 0。
- Guided K=2/K=5：均为 harmful 0、neutral 100、decisive 0。
- Hybrid K=2/K=5：均为 harmful 0、neutral 100、decisive 0。
- 三种方法各自的 K2/K5 三值标签稳定度：100%。
- Random K5：985 physical fits、15 hits、3,409.57 秒。
- Guided K5：1,000 physical fits、1 hit、3,094.98 秒。
- 当前解释：这复现了旧论文中 ISOLET 在 α=0.95 下无 harmful 的方向，
  且不是 random/guided 选择造成的差异。但 K=5 对 existential harmful
  witness 的检出能力有限，不能据此宣称真实 MinPPI 全为 neutral。正式层
  应保留 ISOLET K10 主结果，并将其列为 seed42 K25 的高成本 neutral
  sentinel，以检验更高预算是否仍无 harmful。
- 实际平均约 3.28 秒/physical fit，与 calibration 的 3.48 秒/fit 相差约
  6%，证明以 calibration 建立的 48 小时线性成本模型足够可靠。

#### Gisette

- Bundle runtime：372.37 秒（6.21 分钟）。
- Physical fits：1,985。
- Cache hits：16。
- Random、guided、hybrid 在 K=2 与 K=5 下均为 harmful 0、neutral 100、
  decisive 0。
- 三种方法各自的 K2/K5 三值标签稳定度：100%。
- 当前解释：低预算方向与旧论文“Gisette 在 α=0.95 下无 harmful”一致，
  且结果不依赖 random/guided。由于成本仅为 ISOLET 的约 5.7%，Gisette 可
  作为低成本 neutral sentinel 纳入 seed42 K25 sensitivity。

#### Credit-Card-Fraud

- Bundle runtime：349.54 秒（5.83 分钟）。
- Physical fits：565。
- Cache hits：16。
- Random、guided、hybrid 在 K=2 与 K=5 下均为 harmful 0、neutral 29、
  decisive 0。
- 三种方法各自的 K2/K5 三值标签稳定度：100%。
- 当前解释：这与旧论文“α=0.95 删除约 35% 特征、NSGA-II HV 提升 37%”
  的核心实证声明完全不一致。旧结果在新统一 XGBoost 概率模型和当前冻结
  数据协议下尚未复现，现阶段不得继续写入摘要或结论。
- K=5 仍可能漏掉稀有 harmful witness。Credit 的 K25 成本仅约
  5 × 5.83 = 29 分钟/seed，因此 decision 全部完成后，应优先运行
  seed42 K=10/25 定向检查，区分“预算不足”与“旧结果失效”。

#### CrimeData

- Bundle runtime：349.22 秒（5.82 分钟）。
- Physical fits：1,966。
- Cache hits：15。
- Random K2：harmful 1、neutral 98、decisive 1。
- Random K5：harmful 2、neutral 98、decisive 0。
- Guided K2/K5：均为 harmful 2、neutral 97、decisive 1，稳定率 100%。
- Hybrid K2：harmful 3、neutral 96、decisive 1。
- Hybrid K5：harmful 4、neutral 96、decisive 0，K2/K5 稳定率 98%。
- Hybrid K2 harmful 全部保留，K5 新增 1 个；不存在 harmful 撤销。
- Hybrid K5 harmful：
  `assaultPerPop`、`PctNotHSGrad`、`MedNumBR`、`PctLargHouseFam`。
- 当前解释：这是继 Madelon 后第二个具有清晰 K2→K5 单调扩张的强方向，
  并把论文证据扩展到回归任务。具体 witness、移除 R² 非劣性和这些变量的
  领域解释仍需 validation screen 及正式多 seed 结果确认。

#### Abalone

- Bundle runtime：40.11 秒。
- Physical fits：255；cache hits：1,814。
- Exact K2/K5：harmful 4、neutral 4、decisive 0。
- Hybrid K2/K5：与 exact 完全一致，因为 f=8 触发 exact dispatcher。
- Exact harmful：`Length`、`Diameter`、`Whole weight`、`Viscera weight`。
- Random K2/K5：harmful 0、neutral 8、decisive 0，相对 exact harmful
  recall 为 0%。
- Guided K2：harmful 0；guided K5：harmful 1（`Diameter`），相对 exact
  harmful recall 为 25%。
- 当前解释：K5 近似显著漏掉 exact harmful，证明 decision K2/5 不能作为
  正式估计预算；另一方面 hybrid exact dispatcher 得到精确结果，说明低维
  数据无需增加 K。旧论文“Abalone 无 harmful”在当前统一协议下不成立，
  必须以 exact 多 seed removal 结果重写。

### 8.2 Estimator 完成状态

- 七个数据集的 seed42 K2/K5 decision estimator 全部完成。
- Pipeline exit code：0。
- 尚未读取 test。
- 下一步：运行 validation-only screen，提取 coalition witnesses、harmful
  removal utility、matched-random 对照和方向性 claim cards。

### 8.3 监测缺陷

- ISOLET random 实际完成值：985 physical fits、15 cache hits、约 57 分钟。
- ISOLET 整个 bundle 实际为 108.44 分钟。
- 进度日志显示的 ETA 只覆盖/近似正在运行的方法阶段，不能直接当作整个
  bundle 或全七数据集的剩余时间。
- 在 guided 2/100 时日志 ETA 为 28 分钟，但基于约 926 个剩余 physical
  fits 和 3.4 秒/fit，真实 ETA 应约 53–56 分钟。当前 ETA 计算很可能把
  每个 marginal 的两次 coalition utility evaluation 只计成一次，约低估
  一半。该缺陷只影响监测显示，不改变 schedule、cache 或结果；当前 run
  完成后必须修复并增加覆盖 random→guided 总剩余成本的测试。

## 9. Validation-only screen 结果

报告：

`results/icomp2026_multifidelity/decision-s42-k2-5-20260728-v1/screening/screen_report.md`

完整 JSON：

`results/icomp2026_multifidelity/decision-s42-k2-5-20260728-v1/screening/screen_report.json`

报告生成约 51 秒，未读取 test。

| 数据集 | Grade | Harmful | Witness | K2/K5 稳定 | Validation Δ | 相对 random |
|---|---|---:|---:|---:|---:|---:|
| Spambase | neutral | 0 | 0 | 1.00 | +0.00000 | 0.00000 |
| Madelon | strong support | 15 | 15 | 0.91 | +0.00615 | +0.02084 |
| ISOLET | neutral | 0 | 0 | 1.00 | +0.00000 | 0.00000 |
| Gisette | neutral | 0 | 0 | 1.00 | +0.00000 | 0.00000 |
| Credit-Card-Fraud | neutral | 0 | 0 | 1.00 | +0.00000 | +0.00000 |
| CrimeData | strong support | 4 | 4 | 0.98 | +0.00062 | +0.00047 |
| Abalone | counterexample | 4 | 4 | 1.00 | -0.01549 | +0.12784 |

Claim matrix：strong support 2、neutral 4、counterexample 1，无 incomplete。

### 9.1 当前可保留的论文方向

- Madelon：删除 15% harmful 后 validation AUC 增加 0.00615，且比相同
  删除量的 5 个 random controls 平均高 0.02084；15/15 均有 witness。
- CrimeData：删除 4% harmful 后 validation R² 增加 0.00062，略高于
  random-matched 平均值；4/4 均有 witness。
- ISOLET/Gisette/Spambase：K2/K5 稳定 neutral，可作为“方法不会为了压缩
  强行删除”的保守性案例，但必须等待更高预算确认。
- Credit：核心旧结果没有在 K5 复现，必须先做 K10/K25 定向检查。

### 9.2 Abalone 暴露的理论—行动边界

Abalone exact 证明 4 个特征分别存在使 winning coalition 变 losing 的
contextual witness，但一次性删除四者使 validation R² 从 0.57460 降至
0.55911（delta -0.01549），未通过预注册的 -0.010 回归非劣界。它仍显著
优于随机删除 4 个特征（random mean 0.43126），但不能称为“安全删除”。

因此论文中的以下措辞必须修改：

- 旧：`MinPPI=-1 means must remove`；
- 新：`MinPPI=-1 identifies a contextually harmful candidate; deployment or
  downstream prefiltering requires a held-out validation safety gate`。

正式方法应输出两个集合：

1. `raw_harmful_set`：存在 threshold-crossing witness 的解释集合；
2. `safe_prefilter_set`：只有当冻结的 validation removal 满足非劣界时才
   接受 raw set，否则回退为空集/full features。

NSGA-II prefilter 与最终 test removal 必须使用 `safe_prefilter_set`，同时
论文完整报告 raw set 和被 safety gate 拒绝的 Abalone 反例。这个 gate 在
test 前冻结，不构成 test cherry-picking。

### 9.3 当前报告的预测字段不能用于正式 48 小时预算

screen JSON 的 `11.74 GPU h` 只是把当前 K5 decision bundle 乘以剩余
4 seeds，并不是 per-dataset K10/K25、removal、NSGA-II、ranking 和 ablation
的正式全流程预测。字段名称与 Markdown “confirmation forecast” 容易造成
误解，正式修改应将其标记为 `decision_like_remaining_seeds_forecast`，并由
独立的 stage-level P50/P90 48 小时账本取代。

## 10. Credit K10/K25 定向检查

运行：

`results/icomp2026_multifidelity/credit-probe-s42-k2-5-10-25-20260728-v2`

数据一致性：

- probe 与 decision 的 train.csv SHA-256 完全一致；
- validation.csv SHA-256 完全一致；
- sealed test.csv SHA-256 完全一致；
- 使用从正式七数据集 policy 逐字段投影的 Credit-only calibration policy；
- Python 源码未在两次 estimator 之间修改。

Estimator：

- Bundle runtime：1,554.67 秒（25.91 分钟）。
- Physical fits：2,716；cache hits：141。
- Random：K2/K5/K10 harmful 0；K25 harmful 2（`V5`, `V22`）。
- Guided：K2/K5/K10 harmful 0；K25 harmful 1（`V25`）。
- Hybrid（f=29 时采用 guided）：K2/K5/K10 harmful 0；K25 harmful 1
  （`V25`）。
- Hybrid K10→K25 label stability：0.9655。
- K10 relative-to-K25 harmful recall：0，未通过预注册的 0.90 门槛。

Validation-only screen：

- Grade：strong support。
- Witness coverage：1/1。
- Full AUC：0.99162036。
- 删除 `V25` 后 AUC：0.99141472。
- Delta：-0.00020564，通过 -0.005 非劣界。
- Compression：1/29 = 3.45%。
- 相对 5 个 random matched removals 的平均优势：+0.00156518。
- Witness：不含 `V25` 的 coalition `[V10,V12,V16]` utility 0.95081017；
  加入 `V25` 后 utility 0.94154398，跌破 threshold 0.94203934，drop
  0.00926619。

结论：

- Credit 正式最大预算必须至少 K25；K10 会漏掉当前唯一 hybrid witness。
- 旧论文的“约 35% 压缩、NSGA-II HV +37%”尚未复现，不能保留。
- 当前可检验的新版主张是：K25 找到一个可解释的 contextual harmful
  feature；安全 gate 接受 3.45% prefilter，validation 非劣且优于随机。
- 是否改善 NSGA-II HV 必须等待正式 paired NSGA experiment，不能由旧结果
  推断。

## 11. 最终要回答的论文问题

Decision report 必须回答：

1. 估计器是否在 K=2 到 K=5 之间形成可识别的稳定方向？
2. 哪些数据集出现 harmful features？
3. 每个 harmful feature 是否存在可复核的 coalition threshold-crossing witness？
4. 移除 harmful features 后，validation utility 是否保持或改善？
5. 相同压缩量下是否优于随机移除？
6. 哪些数据集是 strong support、partial support、neutral 或 counterexample？
7. K=10/25/50 的正式五 seed 运行是否能在 2026-08-05 前完成？

通过 freeze gate 后，最终论文核心结果应区分：

- 理论性质：MinPPI 三值分类与阈值定义；
- 算法性质：nested/grouped 估计的样本效率和缓存后的物理拟合成本；
- 实证性质：harmful feature 的存在、可解释 witness、移除后的预测效用；
- 多目标性质：MinPPI prefilter 对 NSGA-II Pareto/HV/运行时间的影响。

## 12. 尚未得出的结论

以下内容目前不能提前宣称：

- MinPPI 在多数数据集显著优于所有 baseline；
- harmful 移除在测试集上稳定提升；
- NSGA-II prefilter 在五 seeds 上显著改善 HV；
- K=5 已经达到正式估计精度；
- 论文核心假设已被全部数据集支持。

这些结论必须等待 decision report、freeze gate 和正式五 seed 结果。

## 13. 48 小时后端与并发基准（2026-07-28）

### 13.1 基准边界

新增 validation-only 真实切分基准：

`experiments/benchmark_icomp2026_backends.py`

产物：

`results/icomp2026_multifidelity/backend-benchmark-20260728/backend_benchmark.json`

基准只读取 decision run 冻结的 train/validation CSV，不读取 test。每个候选
执行完全相同、确定性的 25%/50%/75%/100% 四个 coalition，测量完整
`fit -> predict -> utility` 边界。参考候选为 `cuda-n1-w1`。本轮是吞吐筛查，
不是正式统计结果。

### 13.2 精确结果

ISOLET（4,989 train rows，100 features，26-class OVR-AUC）：

| 候选 | wall / 4 fits | fits/s | 相对 CUDA 单工 | 最大 utility 漂移 |
|---|---:|---:|---:|---:|
| cuda-n1-w1 | 13.369 s | 0.299 | 1.000x | 0 |
| cuda-n1-w2 | 11.137 s | 0.359 | 1.200x | 0 |
| cuda-n1-w3 | 10.692 s | 0.374 | 1.250x | 0 |
| cpu-n1-w1 | 16.808 s | 0.238 | 0.795x | 1.489e-4 |
| cpu-n4-w1 | 7.397 s | 0.541 | 1.807x | 1.262e-4 |
| cpu-n8-w1 | 6.381 s | 0.627 | 2.095x | 1.262e-4 |
| cpu-n4-w4 | 3.063 s | 1.306 | 4.365x | 1.262e-4 |

Credit（182,276 train rows，29 features，ROC-AUC）：

| 候选 | wall / 4 fits | fits/s | 相对 CUDA 单工 | 最大 utility 漂移 |
|---|---:|---:|---:|---:|
| cuda-n1-w1 | 2.761 s | 1.449 | 1.000x | 0 |
| cuda-n1-w2 | 1.536 s | 2.604 | 1.797x | 0 |
| cuda-n1-w3 | 1.387 s | 2.883 | 1.990x | 0 |
| cpu-n1-w1 | 4.575 s | 0.874 | 0.603x | 0.002625 |
| cpu-n4-w1 | 1.748 s | 2.289 | 1.579x | 0.002625 |
| cpu-n8-w1 | 1.273 s | 3.142 | 2.168x | 0.002625 |
| cpu-n4-w4 | 0.722 s | 5.539 | 3.822x | 0.002625 |

### 13.3 决策

- CUDA 并发的逐 coalition validation utility 与 CUDA 单工完全一致，三工是
  当前无科学损失的安全候选。
- ISOLET 的 CUDA 三工只得到 1.25x，说明单 GPU contention 明显；不能把
  “三个 worker”错误地按 3x 计入 48 小时预算。
- Credit 的 CUDA 三工接近 2x，适合与其他较轻数据集分片并发。
- CPU 4x4 的 ISOLET 吞吐潜力为 4.37x，但 CPU/CUDA utility 非零漂移；
  在完成边际三值结构一致性检查前，不得切换正式 backend。
- Credit CPU/CUDA 最大 AUC 漂移 0.002625，已达到论文效应量级，禁止仅凭
  速度将 Credit 切到 CPU。
- 正式框架默认保持同一 CUDA 模型/输出，并允许最多三个独立 shard 并发；
  是否给 ISOLET 启用 CPU，必须由单独的结构面板 gate 决定。

### 13.4 对旧加速计划的纠正

`nested_estimation.py` 已经在 random/guided 两条路径中实现 `best == -1`
后的吸收态停止。因此“发现 harmful 后跳过未来 trials”不是尚未实现的未来
加速，而是 decision/Credit probe 已经享受的现状。ISOLET 全 neutral，几乎
无法从该机制获益；其主要可压缩维度只能是预算、后端和跨 seed 并发。

当前 ETA 则确有系统性低估：remaining 使用 marginal trial 数，而吞吐使用
physical fit 数；一次未命中缓存的 marginal 最多需要两个 fits。修复时应将
剩余物理拟合上界显式化，并对 random 到 guided 的阶段切换做测试。

### 13.5 ISOLET CPU 结构门控结果

产物：

`results/icomp2026_multifidelity/backend-benchmark-20260728/isolet_structural_panel.json`

在完整 100-feature 顺序上预先等距抽取 20 个目标特征，对每个特征构造固定的
25%/75% 两尺度 marginal panel，共 40 个 marginal pairs、81 个唯一
coalitions。CUDA `n_jobs=1, workers=3` 是参考；CPU 候选为
`n_jobs=4, workers=4`。

- CUDA：203.432 秒，0.398 fits/s。
- CPU：45.680 秒，1.773 fits/s，吞吐 4.453x。
- 最大单 coalition utility 漂移：0.00056290。
- Full utility loss：0.00004340。
- Spearman：0.98456。
- Winning-coalition agreement：1.000。
- Marginal-sign agreement：0.725。
- Harmful recall：1.000；false-harmful rate：0（该面板无足够 positive
  harmful event 时，recall 不应被解释为具有检验力的 100%）。

冻结决策：**拒绝 ISOLET CPU backend**。0.725 的 marginal-sign agreement
说明 CPU 与 CUDA 的局部 utility landscape 并非同一实证对象；不能用吞吐
优势交换论文方法的一致性。正式主表固定 CUDA。CPU 结果只作为被否决的工程
候选保留，不进入论文效果表。

## 14. 48 小时正式框架冻结候选（2026-07-28）

权威执行文档：

`ICOMP2026_48H_EXECUTION_FRAMEWORK.md`

### 14.1 时间与预算决策

- 三个 seed shards：`(42,123)`、`(456,789)`、`(1024)`；
- estimator sequential projection：28.85 h；
- 最低实测三 shard speedup：1.25x；
- 1.30 guard 后 estimator P90：30.00 h；
- 后处理 P90 reserve：13.00 h；
- P0 total P90：43.00 h；
- 48 h margin：5.00 h；
- P1 ISOLET K25 seed42：约 9.04 h，不阻塞 P0。

NSGA-II 保留 population 40、evaluations 800。当前没有 800→400 的
convergence 证据；既然 43 h P90 已包含 3 h paired NSGA reserve，就不以降低
科学预算换取表面速度。

### 14.2 安全边界已经进入正式代码

Removal selection plan 现在同时保存：

- `raw_k` 与 `raw_harmful_features`；
- full/raw-prefilter validation utility；
- validation delta 与 task-specific noninferiority margin；
- gate accepted 状态；
- `safe_harmful_features` 与 safe `k`；
- safety validation model-fit count。

分类 margin 为 0.005，回归 margin 为 0.010。失败时 safe set 为空，
removal 与 NSGA-II prefilter 自动回退 full features。Raw witnesses 仍进入
正式 summary，用于解释 Abalone 等反例。

### 14.3 跨 shard 汇总与统计

原 per-run `summarize` 要求一个 run 同时包含完整五 seeds，因此不能直接汇总
seed shards。现增加 campaign-level summary：

- 逐 shard 重建并验证冻结 protocol/source；
- 对每个 shard 重算预期 manifest 并验证所有 P0 ResultRecords；
- 合并后重新建立唯一 manifest sequence；
- 只接受精确五 seed 非重叠联合；
- 生成一次五 seed raw/aggregate/paired/front 表；
- 生成 paired NSGA 统计、PDF figures 和 LaTeX macros；
- 生成 matched-removal 六方法 Friedman/Nemenyi 排名；
- pairwise MinPPI 使用 dataset-level Wilcoxon + Holm；
- 统计块为 7 datasets，每个 dataset 先平均 5 seeds，禁止 35-fold
  pseudo-replication。

### 14.4 兼容性修复

- grouped acceleration validator 现接受 model-policy、per-dataset K 和
  stage-scope 字段；
- ISOLET validator request 使用 `[2,5,10]`，不再错误使用全局预算；
- screen 现读取完整 shard seed authority，可在 `(42,123)` shard 上只解释
  seed 42，而不会发生 protocol hash mismatch；
- selection safety metrics 会进入最终 raw/aggregate tables。

### 14.5 Cache 决策

不将旧 decision/Credit probe cache 晋级到正式 P0。原因是正式 source、
protocol、安全门和汇总边界已经变化。旧 artifact 仅用于运行时间和方向性证据；
正式 campaign 从空结果建立自己的 cache。

### 14.6 当前验证证据

最终完整测试：

```text
1245 passed, 3 skipped in 133.47 s
```

三 shard 真实 CLI validate smoke：

```text
framework-validate-20260728-v1-s42-123: validate 1/1 success
framework-validate-20260728-v1-s456-789: validate 1/1 success
framework-validate-20260728-v1-s1024: validate 1/1 success
```

首次创建和第二次相同命令 resume 均 exit 0；campaign plan 明确
`test_access_authorized=false`。Smoke 位于
`results/icomp2026_framework_smoke/framework-validate-20260728-v1`，不进入
正式论文结果。
