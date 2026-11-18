# 126 个历史配置的统一配对评估

主比较保留既有 69 个配置和额外 57 个配置的外层划分、目标和每个来源的评分规则。运行前逐项核对四个输入数组的 SHA-256，原面板还核对训练组和测试时的部署单位。Gas 和 HAR 给部署单位等权，MHEALTH 对每位受试者取腕部与踝部较差的结果再平均。回归评估 R²，其他分类评估平衡准确率，SECOM 另外保留平均精确率。选择器只能访问外层训练特征和标签。

每组包含五项完整方法。Full 直接使用全部列。候选规则在每次部署中自行决定列数，重复运行选列。F 对分类使用单因素方差分析，对回归使用单变量线性回归 F 统计量。Random 按固定种子做均匀无放回抽样。无同伴消融以候选机制造成的相同宽度仅按边际准则选列。三个等宽对照在计时前获得候选的宽度。另抽五份确定种子的随机支持测量质量分布。每个假设使用在训练数据上拟合的相同 StandardScaler 和单位惩罚 Ridge/RidgeClassifier，SECOM 与 CNAE 的非有限值按外层训练均值补缺。每个面板按原始训练矩阵的较短维度选求解器，≤2048 时使用 Cholesky，其余使用 LSQR，所有方法采用相同求解器。原先的大多数历史结果使用 LSQR，直接比较以本批重新训练的 Full 分数为准。

先运行质量评估保存实际支持、逐行预测、输入指纹、必要的单位与决策值。两个独立进程随后逐配置预热一次，五个方法按旋转表单线程各计时五次，每个位置各一次。两进程均须与质量归档逐项重现支持和预测。完整耗时包含训练期选列及校准、切列、训练/测试预处理、模型拟合和预测；相同的输入读取、外层评分及存盘不计入耗时。联合成功要求真删列、分数较 Full 最多低 0.02，且两个进程中该组完整耗时中位数均低于 Full。另记录 0、0.005、0.01 容差。等宽严格比较要求质量不降及完整时间更短，同时报告胜、平、负。重复种子的配置只作为稳定性检查，观察单位可按来源及外层划分重新合并。此前用过这些数据，解释范围为历史复测。

从 history 根目录运行。每项输出使用一个未存在的目录。以 `q`、`c1`、`c2`、`summary` 表示位于 results/minpi-study 下新建的本轮输出目录。

```powershell
.\thesis-python.ps1 -m research.feature_game_eval.panels quality results/minpi-study/q
.\thesis-python.ps1 -m research.feature_game_eval.panels cost results/minpi-study/c1 --quality results/minpi-study/q --pass-id 1
.\thesis-python.ps1 -m research.feature_game_eval.panels cost results/minpi-study/c2 --quality results/minpi-study/q --pass-id 2
.\thesis-python.ps1 -m research.feature_game_eval.summarize results/minpi-study/q results/minpi-study/summary --costs results/minpi-study/c1 results/minpi-study/c2
```

**冻结方法的历史结果。** 在同一次单线程比较中，真实删列、得分较全特征最多低 0.02、两轮耗时均较短的计数如下。额外面板包含真实数据和人工数据。

| 方法（各列数相同） | 原 69 | 额外 57 |
| --- | ---: | ---: |
| 两种同伴数取最低收益的 minPI | 10 | 14 |
| ANOVA / 回归 F 统计量 | 10 | 9 |
| 单个固定种子的均匀随机支持 | 6 | 2 |
| 同一初始模型和评价样本，仅依边际收益排名 | 9 | 5 |

原 69 组中，minPI 60 次实际删列，其中 15 次满足精度门槛。同宽与 F 比较，计 36 胜、16 平、17 负，与额外五份随机支持的中位数比较为 57 胜、12 平、0 负。额外 57 组全部删列，其中 32 次保分，对 F 的分数是 36 胜、12 平、9 负。相对无同伴消融，minPI 在两批中各有 37 胜、18 平、14 负和 27 胜、18 平、12 负。原 69 组中有 9 次全留，对照也使用全列。这些平局计入分数统计，联合成功计零。minPI 的中位保留比例分别为 14.0% 和 6.25%。ISOLET 的典型配置只保留 1/617 列，所有三组都未同时删列保分；四种新模拟稠密信号配置（分类/回归 × 48/128 列）的 12 次实例全部未保分。这两类失败指向训练子样本和自动停止条件低估组合所需宽度。

在预先归档的四种代理模拟配置（分类/回归 × 48/128 列）中，12/12 组的 minPI 分数严格高于同宽 F 和无同伴消融，11/12 守住全特征分数，5/12 联合成功。为区分模型归纳偏好与数据选择，另用在本次历史分析后冻结的新数据生成机制检验。[两进程配对总表](../../results/minpi-study/residual-126-corrected/summary/paired-cases.csv) · [分源、分任务与容差汇总](../../results/minpi-study/residual-126-corrected/summary/summary.csv) · [质量与可恢复源码](../../results/minpi-study/residual-126-quality-a) · [计时一](../../results/minpi-study/residual-126-cost-a1) · [计时二](../../results/minpi-study/residual-126-cost-a2)

**F 基线的数值更正。** 对所有 126 组，使用独立 float64 组内/组间方差或回归相关平方检查归档 F 排名边界，发现原始尺度下三个 SECOM 配置的 `f_classif` 受 float32 计算抵消影响：第 370 列的软件统计量 45.3336，直接计算为 0.9644。改为训练期补缺、转 float64、中心化后调用同一 ANOVA 公式，正确第一列为 497，第二列为 28（列号从 0 开始）。三组各自的正确 F 模型 BA 均为 0.500，与 minPI 持平。按照相同协议两次重新计时三个完整配置的全部五臂，含稳定 F 的预处理成本；其余 123 组使用已通过边界核查的原支持与配对时间。上面各结果是更正后的合并版本。最初的 39 胜、13 平、17 负只对应未经更正的 F。原比较数据仍在[初版档案](../../results/minpi-study/residual-126-joint-a)。
 
若仅要求真删列与两进程都快于 Full，原 69 组有 39 组、额外 57 组有 26 组。实际通过组的分差中位数分别为 −0.1019、−0.00294，最差为 −0.8503、−0.9080。不同任务的分值含义不同，逐组表列出了任务和指标。[按对照松弛后的计数及损失](../../results/minpi-study/residual-126-corrected/summary/relaxed.csv) · [三配置稳定 F 修正及合并程序](secom_correction.py) · [独立列级质量检查](audit_controls.py)。

**额外的独立模拟确认。** 实验在获取任何一个新生成实例的测试结果前固定算法和验收要求。二分类和回归各用八个潜在因子生成信号，每个因子产生四个带独立高斯测量误差的代理，模拟的同组列相关系数为 1/(1+0.15²) ≈ 0.978。潜在系数绝对值分别从 [0.7,1.3] 抽取并归一化，再添加标准差为 0.35 的独立标签噪声；剩余列为独立噪声，列号随机打乱。每个任务包含 2,048×64 和 4,096×256 两种训练形状，各以八个全新生成种子得到独立实例，测试集大小均为 1,024。同时运行相同规模的稠密反事实，令全部独立特征均携带信号。两个机制共 64 组。只在评分输出阶段读取潜在因子的真实分组。

除上述五法，还设立两项监督学习强对照。第一项在全部训练样本拟合标准化单位岭，以回归系数绝对值（分类使用对应决策系数范数）选取前 k。第二项用全部训练行按 F 排序，依序保留与已选列的绝对相关系数低于 0.90 的代表；相关性用预先固定的 256 个训练样本估计，剩余预算不足时以 F 顺序补足。两者均获得被测算法的同宽预算。每个进程每个配置下七个方法预热后各轮转七次，两个进程都必须严格重放所选列和预测。对每个实际任务，要求 16 个新代理配置中至少 12 个同时删列、保分、快于 Full 且分数严格胜 F 和随机，其中每种形状至少 5/8。另报告相对强对照的速度、得分与其选中多少个不同潜在信号；逐项比较优于较弱对照不能代替这个检查。[冻结协议及生成器](confirmation.py) · [判定程序](confirmation_summary.py)

反事实可直接量化。稠密回归中的独立高斯特征系数绝对值为 1/√p，噪声方差为 0.35²。固定保留 k 列时，知道真实系数的最优子模型相对完整最优回归器，理论 R² 也下降 (1−k/p)/(1+0.35²)。要将损失控制在 0.02 内，64 列至少要保留 63 列，256 列至少要保留 251 列。当前自动宽度最多选择 64 列，除非回退全宽，因此对 256 列稠密机制难以达到压缩与保分目标。该推导说明必须按数据生成机制报告。

**新模拟结果。** 预定的代理信号层门槛在两个任务上都达到。分类 14/16 组（64 列 6/8、256 列 8/8），回归 16/16 组（两档均 8/8）同时删列、相对 Full 损失至多 0.02、两轮都加速且分数严格高于同宽 F 与随机。两种任务的两档宽度中位数均为 8。分布于全部列的稠密信号反事实在两项任务均 0/16 达到联合标准，暴露出同一训练准则会过度压缩的边界。

| 代理层（每任务 16 组） | 分类平均 BA | 回归平均 R² | 分类平均覆盖因子 / 8 | 回归平均覆盖因子 / 8 |
| --- | ---: | ---: | ---: | ---: |
| minPI | 0.8787 | 0.8705 | 7.875 | 8.000 |
| Full | 0.8729 | 0.8794 | 8.000 | 8.000 |
| 同宽 F | 0.6902 | 0.3447 | 2.063 | 2.313 |
| 同宽均匀随机 | 0.6120 | 0.2185 | 1.625 | 2.000 |
| 同样抽样、无同伴的边际消融 | 0.6958 | 0.3411 | 2.500 | 2.563 |
| 全训练岭系数强度 | 0.8233 | 0.6922 | 6.375 | 5.875 |
| F 加相关性去冗余 | 0.8761 | 0.8693 | 7.875 | 8.000 |

minPI 对 F、随机、无同伴消融、岭系数强度，在此生成机制上分数各为 32 胜。对岭系数强度，32 个配置还同时更快。最强的相关性去冗余对照同样覆盖约八个潜在信号，保分加速计数与 minPI 相等；minPI 对它的逐配置分数在分类、回归各为 10 胜 6 负，分类 15/16 更快、回归 0/16 更快。这里确认的是在产生相关代理的独立合成数据上、相对点名基线的优势，尤其是同预算下找全不同信号因子；仅从这些实验不能推出在真实数据上的同样效果。[逐配置质量、时间和潜在因子覆盖](../../results/minpi-study/residual-64-confirm-summary/paired.csv) · [强对照比较](../../results/minpi-study/residual-64-confirm-summary/method-summary.csv) · [预定门槛](../../results/minpi-study/residual-64-confirm-summary/decision.json)

若只按真删列且两轮都比 Full 快计数，代理层分类和回归各 16/16，较 Full 的时间比中位数分别为 0.365 和 0.374；分类分数差的中位数为 +0.0076，回归为 −0.0091。稠密机制分类有 12/16、回归有 14/16 达到这个时间要求，同时分数中位损失分别为 0.310 个 BA 绝对分值、0.849 个 R² 绝对分值。放宽精度门槛后应保留这些损失的原值。

运行新确认（使用三个新的输出名）：

```powershell
.\thesis-python.ps1 -m research.feature_game_eval.confirmation results/minpi-study/conf1 --pass-id 1
.\thesis-python.ps1 -m research.feature_game_eval.confirmation results/minpi-study/conf2 --pass-id 2 --reference results/minpi-study/conf1
.\thesis-python.ps1 -m research.feature_game_eval.confirmation_summary results/minpi-study/conf1 results/minpi-study/conf2 results/minpi-study/conf-summary
```

**复算更正和最终核验。** 保留三份新的输出目录，将下面示例的 `q`、`c1`、`c2` 替换为同一初始 126 组质量和两个计时归档。更正版在 [arrays.csv](../../results/minpi-study/residual-126-corrected/quality/arrays.csv) 中逐配置记录实际采用的数组及其 SHA-256。

```powershell
.\thesis-python.ps1 -m research.feature_game_eval.secom_correction measure results/minpi-study/f1 results/minpi-study/q --pass-id 1
.\thesis-python.ps1 -m research.feature_game_eval.secom_correction measure results/minpi-study/f2 results/minpi-study/q --pass-id 2 --reference results/minpi-study/f1
.\thesis-python.ps1 -m research.feature_game_eval.secom_correction integrate results/minpi-study/corrected results/minpi-study/q --original-times results/minpi-study/c1 results/minpi-study/c2 --corrected results/minpi-study/f1 results/minpi-study/f2
.\thesis-python.ps1 -m research.feature_game_eval.audit_controls results/minpi-study/corrected/quality results/minpi-study/corrected/summary/f-check.csv
.\thesis-python.ps1 -m research.feature_game_eval.audit_confirmation results/minpi-study/conf1
```

最终独立检查在 126 个历史配置及 64 个新模拟配置中均确认，按稳定的直接方差比或回归相关平方，选中的 F 支持在预算边界的最大效应量逆序为 0。同宽随机在历史面板逐配置核对固定支持和五个额外抽样。新模拟的 448 个外层分数和 224 个潜在因子覆盖数均逐项从原始数组重算。[126 组 F 的逐配置误差](../../results/minpi-study/residual-126-corrected/summary/independent-f-audit.csv) · [模拟核验程序](audit_confirmation.py)
