# minPI 自选宽度、选列耗时与适用范围

## 最终实现与原 69 组

在原 69 组上，把同一套 minPI 配对投票改成编译累加、在数值近并列处复核后，固定保留约 75% 时，69 组选列和预测均与原 NumPy 实现相同。单组选列耗时的中位数由 0.471 毫秒降至 0.346 毫秒，65/69 组选列更快。同场旧实现联合达标 48 组，新实现 50 组，同宽随机 51 组。上一轮归档的旧实现为 43 组，本次同场测得的新旧实现差距为两组。固定宽度的 4 组、自选宽度的 2 组触发参考重算，均已计时。

| 原 69 组，同场运行 | 实际删列 | 分数达标 | 三项达标 | 达标且至少快 5% | 达标且至少快 10% |
|---|---:|---:|---:|---:|---:|
| 旧 minPI，固定约 75% | 69 | 57 | 48 | 40 | 35 |
| 改进 minPI，固定约 75% | 69 | 57 | 50 | 46 | 38 |
| 同宽随机，固定约 75% | 69 | 53 | 51 | 48 | 44 |
| 改进 minPI，训练投票自选宽度 | 69 | 66 | 43 | 33 | 26 |
| 同宽随机，逐组匹配自选宽度 | 69 | 61 | 54 | 44 | 34 |

自选宽度平均保留 85.6%，在随机独胜的 14 组中，12 组是 minPI 的分数已达标、总耗时超标。相同输入和种子的 414 份选列及预测在另一进程全部复现。改变种子时，所选列的 Jaccard 相似度中位数为固定宽度 0.740、自选宽度 0.844。[逐组结果](../../results/minpi-study/stable-native-69-20261117a/cases.csv) · [数值边界复核](../../results/minpi-study/stable-native-69-20261117a/boundary-audit.csv) · [种子间相似度](../../results/minpi-study/stable-native-69-20261117a/same-outer-seed-overlap.csv)

## 最终实现的外部验证

在含真实分类、时间顺序回归和模拟分类及回归的 57 个外部配置中，自选宽度保留了 50.0%–98.71% 的特征。若要求测试分数完全不降且加速，minPI 达标 11/57，同宽随机 8/57。沿用既定的 0.02 容差，两者均为 14/57。以全特征总耗时为 1，各配置耗时比的中位数为 minPI 1.105、随机 0.998。“高度相关”预设类别内，自选 minPI 守住 19/21 组的精度，6/21 组三项达标，随机为 12/21，计入类别识别耗时后 minPI 为 3/21。高相关不足以作为推荐使用的条件。[外部逐组结果](../../results/minpi-study/stable-native-transfer-20261117a/cases.csv) · [类别检验](../../results/minpi-study/stable-native-transfer-20261117a/class-by-task.csv)

| 外部 57 组，同场运行 | 实际删列 | 分数达标 | 三项达标 | 达标且至少快 5% | 达标且至少快 10% |
|---|---:|---:|---:|---:|---:|
| 改进 minPI，训练投票自选宽度 | 57 | 36 | 14 | 8 | 5 |
| 同宽随机 | 57 | 28 | 14 | 9 | 6 |
| 同宽单变量 F 检验 | 57 | 37 | 4 | 4 | 2 |

为了与旧实验对照，在其中四个面板的 12 个配置另外复测固定 75%。minPI 有 7 组保住分数、0 组三项达标，同宽随机为 5 组、3 组。在 57 个自选与 12 个固定配置里，所选列和预测均与 NumPy 参考实现一致，包括触发近并列回退的 3 个固定宽度配置。全部 333 份方法输出在独立进程复现。稀疏输入专项的 9/9 来源、实际保留比例、单遍零列过滤的同场结果见下文“大量训练常量”。[后端核查](../../results/minpi-study/stable-native-transfer-20261117a/guard-cases.csv) · [独立进程核查](../../results/minpi-study/stable-native-transfer-20261117a/fresh-process-audit.csv) · [十轮实验按原始计时复算的总表](../../results/minpi-study/adaptive-comparison-20261117a/summary.csv)

## 判据和对照

每个配置只记一次成功。要求实际删列，测试分数相对全特征岭模型的差值至少为 −0.02，同时从选列到预测的总耗时中位数短于直接使用全特征。全留记为失败。另报分数完全不降、至少快 5% 和 10% 的情况。不同候选在每个配置的得分分别计算，不能按测试成绩临时换宽度或拼接方案。

历史开发集有 13 个数据源、29 个不同外层划分、69 个配置。沿用上一轮所有外层样本、分组、列顺序及评价指标。每组预热后轮换计时七次，每次重新选列、取列、按原方案补缺、训练、预测。F 是分类单因素方差分析或回归单变量 F 检验。随机是用同一种子无放回均匀抽取列。自适应 minPI 在计时内重新决定宽度，两项对照在计时前免费获得同一个宽度。所有方法最后使用同样的外层训练行。每轮重测全特征与对照，使用各自同场计时解释胜负。

上一轮固定 75% 的 69 组中，随机独胜 15 组。在其中 11 组，minPI 保住了精度但总耗时超标。15 组中，minPI 与随机子集的选列时间差中位数为 0.340 毫秒，下游的差为 0.013 毫秒。小问题的速度瓶颈主要在选列。

## 从训练数据决定删多少列

- **tournament** 在抽样训练行上进行成对淘汰。用一侧拟合、另一侧评价特征与目标的关系，取没有同伴和有同伴两种情况下较小的贡献。自动模式将淘汰者按分数排序，删除非正票并最多再花去正票总量的 2%。固定 75% 时只检查足以构成淘汰对的列。
- **pilot** 面向存在可相互替代测量的表格。两轮符号指纹给出疑似同组的列，两个训练子样本中相关系数绝对值都至少为 0.90 且方向一致才合组。组的价值取活跃成员中最大的非负训练投票强度，列按零同伴和全同伴的较小边际贡献排序。自动模式每组留一列。找不到可确认的同组列时，仅删最低一列。
- **exchange_vote** 从与随机基线相同的排列出发，最多检查 12 对可交换列及其上下文列，每一侧训练投票都支持才换列。以所抽列中在各评估侧及两个端点都缺乏正票的比例估计删除量，最多删一半。对至少 16 列的表格，如发现至少四分之一列为可确认的零票常量，自动模式保留其他列。稠密输入会完整核实有限常量，CSR 输入可直接核实全零列。缺失值不能充当全零证据。
- **native** 将充分统计量配对投票的累计运算提前编译为本机代码，固定 75% 用于和旧 moment 的选列及预测逐项核对。自适应模式先以零同伴、全同伴的较差票决定删除次序，再以各评估侧及端点中的最大正票限制删除代价，预算为总代价的 2%。这些训练票阈值都是代理准则，测试指标仍以真实岭模型重训的成绩计。

对于训练中严格恒定的列，标准化后其训练向量为零。带截距、正岭惩罚的线性模型给该列的最优系数为零。删除这样的列不改变其余系数或任意测试行的预测，即使该列在测试时取新值。此结论在精确算术下成立，浮点迭代求解的实际差异另行核对。普通的训练常量过滤也具有这个性质，因此其成本和成功数组成独立的实用对照。耗时改善取决于最终训练节省的时间是否覆盖认证与取列的开销。

## 外部面板与适用类别

四个上一轮额外面板为 Arcene、Dexter、冗余信号回归和稠密信号回归，各三个选择器种子，其余面板在取得本轮测试成绩前确定。新增 Wisconsin Diagnostic Breast Cancer 与 Wine 的分层划分，36 个独立生成的分类及回归实例（冗余代理、独立稀疏信号、稠密信号，各两种宽度），以及美国季度宏观数据的三个前推预测划分。共 57 个配置。外部评测每组五次单线程轮换计时并另启进程重放。

预先定义可从训练集识别的冗余类别：训练行数至少为列数的四倍，16–512 列，前 256 个训练样本中至少四分之一的列各有一个绝对相关系数不低于 0.90 的同伴。检测结果只用于分层报告。预定经验判据要求在符合条件的新增真实数据、模拟分类和模拟回归三层中分别过半联合达标，且每层至少追平同宽随机和 F。另列计入诊断耗时的结果，以及各层覆盖的真实来源数。

零票常量分支另作独立检验。提前确定两个新的 UCI 数据源、两个训练规模及对照方法。Farm Ads 采用原生 CSR 词袋和稀疏标准化岭分类器。Internet Advertisements 使用发布方的 URL、文本和图像特征，缺失值按训练均值补全。每个来源以相同特征向量首次出现的记录去重，固定三个互斥外层测试池，各抽 256 行测试，两个嵌套的训练集分别含 256 行和 768 行。另测 16 个可控分类和回归配置，训练常量列比例为 0%、25%、50%、75%，部分训练常量在测试时发生变化。普通常量过滤、同宽 F、同宽随机以及使用全特征均各自完整计时。真实数据不添加任何常量列。[UCI Farm Ads](https://archive.ics.uci.edu/dataset/218/farm+ads) · [UCI Internet Advertisements](https://archive.ics.uci.edu/dataset/51/internet+advertisements)

## 适用范围的实验结果

### 相关代理和有限票数

自动配对淘汰在原 69 组联合达标 41 组，同宽随机 45 组、F 为 40 组。它在 57 个外部配置达标 21 组，同宽随机与 F 各 17 组。外部精度只在 30/57 组达标，季度前推回归有一组 $R^2$ 相对全特征降低了 3.315。自动代理分组原 69 组为 17/69，额外为 7/57，额外有 54/57 组守住分数。它在 36 个额外配置仅删一列，这些配置均未加速。自动换列在原 69 组为 34/69，同宽随机为 47/69，两个固定 75% 方案分别为 42/69、49/69。在额外 57 组自动换列为 14/57，对应随机 18/57。恒定列认证在原 69 组没有触发。

相关代理类别共有 21 个配置，其中临床数据和宏观预测各三个、旧冗余回归三个、新增冗余模拟十二个。代理分组守住全部 21 组的精度，联合成功为 7/21，对应随机 15/21。配对淘汰为 12/21，对应随机 15/21。换列为 8/21，对应随机 14/21。高度相关的输入列本身不足以保证这三个方法在真实部署中节省总时间。按年份前推时，宏观数据用于确认相关性的最新训练组最多只有四行，代理确认尤其保守。[原 69 组](../../results/minpi-study/adaptive-69-20261117a/sensitivity.csv) · [额外数据分层](../../results/minpi-study/adaptive-transfer-20261117a/class-by-task.csv) · [换列复测](../../results/minpi-study/exchange-69-20261117a/sensitivity.csv)

### 大量训练常量

在新增 UCI 实验的 12 个配置中，9 个满足训练常量分支的输入条件。换列 minPI 在这 9 组全部守住精度，8 组也加速，8 组均至少快 10%。同宽随机为 4/9，通用常量过滤为 9/9，两种过滤在这 9 组选择了相同的列。在 Farm Ads 的六组，54,877 列缩减为 14,595–26,539 列，六组测试标签预测逐个与全特征相同。Internet Advertisements 的三个 256 行训练集均达到常量门槛，其中一组浮点 LSQR 重训后平衡准确率提高了 0.0122，另两组与全特征相同。三个 768 行训练集均低于常量门槛，转用换列投票，三组联合成功，同宽随机也三组成功。

模拟中确实有至少四分之一训练常量的 12 组，换列 minPI 守住全部精度，11 组加速，常量过滤也为 11/12，同宽随机为 0/12。另四组输入都没有常量，换列与随机均为 0/4。原 57 个额外配置中的 Dexter 亦出现自动换列与普通常量删除各 3/3 的情况。恒定列这个输入条件可以在训练时验证，能解释保分，不构成对墙钟加速的保证。[UCI 与模拟的逐组分支](../../results/minpi-study/constant-transfer-20261117a/panels.csv) · [同宽与常量基线](../../results/minpi-study/constant-transfer-20261117a/by-branch.csv)

进一步检查发现，原来的 CSR 分支已经知道保留哪些列，仍会生成全列随机排列、再将保留列排序。现在公共入口在这个分支直接读取已出现的列号。显式零、重复坐标、缺失值和全零训练集分别处理。稠密数据仍走原换列规则。采用原生 CSR 和同一稀疏岭分类器，重新测量 Farm Ads 的六个配置和 Dexter 官方验证划分的三个种子。这九组的所选列、测试预测都与旧换列实现逐项一致，测试分类结果也全部与全特征一致。选列中位数从 1.478 毫秒降到 0.225 毫秒。新版 9/9 联合达标，均至少快 10%，总耗时比中位数为 0.865，范围为 0.600–0.893。旧换列 8/9，同宽随机 2/9，同宽 F 1/9，通用常量过滤 6/9。专门的一遍 CSR 全零扫描同样 9/9，所选列相同。Dexter 选择 7,751/20,000 列，Farm Ads 保留约 27%–48%。这给出了稀疏固定特征空间的一种适用条件，优势可以由零列剔除本身解释。九个配置覆盖两个真实数据源，Farm Ads 两种训练规模共用每折测试行，Dexter 三个种子共用一个官方测试集。[同支持重测](../../results/minpi-study/sparse-exchange-20261117a/summary.csv) · [逐组耗时与分数](../../results/minpi-study/sparse-exchange-20261117a/cases.csv)

在 Internet Advertisements 出现分数变化的一折，所有被删列的拟合系数实测为零。原协议使用 float32 和 LSQR 容差 0.001，全列及删列解分别迭代 113、109 次，一个测试样本跨过判别边界。以 float64 和容差 $10^{-10}$ 复算，该折两种模型的最大决策分数差约为 $3.7\times10^{-10}$，三个划分的预测标签全部一致。[数值复核](../../results/minpi-study/constant-transfer-20261117a/zero-vote-numerics.csv)

### 数值复现

原型代理分组在某些含缺失的数据中会把常量 0.1 的舍入误差归一化为非零向量。公共实现现按列先减首个有限观测，再计算均值。修复后重新核对原 69 组的两个宽度以及额外 57 组自动、12 组固定宽度，共 207 个比较，所选列、代理分组及预测变化数均为零。保留原型及其耗时归档，修复版没有另作速度声明。[修复对照](../../results/minpi-study/proxy-normalization-20261117a/summary.csv) · [公共代理算法](../../src/minppi_mofs/proxy_minpi.py)

未处理近并列时，固定 75% 的 C 内核在原 69 组中的三个 CNAE 划分和额外的一个 Dexter 种子与 NumPy 选出了不同的边界列。CNAE 的票分差异约 $10^{-16}$，测试预测确有变化。公共接口因而加入参考实现复核：入选与落选的边界过近，或者累计删除代价接近训练预算门槛时，以相同随机种子、相同行和配对重新计算。回退开销包含在最终端到端计时中。[边界处理实现](../../src/minppi_mofs/stable_native_minpi.py)

## 调用与复算

```python
from minppi_mofs.support_selector import PivotalSupportSelector

selector = PivotalSupportSelector(rule="exchange_vote", fraction="auto",
                                  task="classification", random_state=1129)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
```

上例对 CSR 稀疏矩阵启用已验证的全零列分支。一般稠密输入的编译投票使用 `rule="native", fraction="auto"`。回归传入 `task="regression"`。有部署组时，向 `fit` 传入训练行对应的 `groups`，有明确先后顺序再传 `ordered_groups=True`。固定预算诊断用数值 `fraction`，例如 0.75。实验计时直接调用选择函数并在下游管线中取列，公共 scikit-learn 接口还会执行标准的输入校验。相关实现通过 246 项定向测试。[选择器入口](../../src/minppi_mofs/support_selector.py) · [稀疏实现](../../src/minppi_mofs/sparse_exchange_minpi.py) · [测试记录](../../results/minpi-study/stable-native-transfer-20261117a/minpi-tests.xml) · [原始计时复算脚本](../../experiments/audit_adaptive_success.py)

在 `MastersThesis/history` 依次运行。以下输出目录应是新目录，命令均使用本工作区原有数据与归档。每条评测命令第一次结束后，在同一命令末尾加 `--verify` 可在新进程核对支持和预测。正式计时运行时各轮互相错开。

```powershell
.\thesis-python.ps1 -m experiments.adaptive_minpi_benchmark --output results/minpi-study/replay-adaptive-69
.\thesis-python.ps1 -m experiments.adaptive_minpi_transfer --output results/minpi-study/replay-adaptive-transfer
.\thesis-python.ps1 -m experiments.exchange_vote_benchmark --output results/minpi-study/replay-exchange-69
.\thesis-python.ps1 -m experiments.exchange_vote_transfer --output results/minpi-study/replay-exchange-transfer
.\thesis-python.ps1 -m experiments.minpi_constant_transfer --output results/minpi-study/replay-constant-transfer
.\thesis-python.ps1 -m experiments.minpi_sparse_exchange_replay --output results/minpi-study/replay-sparse-exchange
.\thesis-python.ps1 -m experiments.build_native_pivotal
.\thesis-python.ps1 -m experiments.native_minpi_benchmark --output results/minpi-study/replay-native-69
.\thesis-python.ps1 -m experiments.native_pivotal_transfer --output results/minpi-study/replay-native-transfer
.\thesis-python.ps1 -m experiments.stable_native_benchmark --output results/minpi-study/replay-stable-native-69
.\thesis-python.ps1 -m experiments.stable_native_transfer --output results/minpi-study/replay-stable-native-transfer
.\thesis-python.ps1 -m experiments.check_proxy_normalization --output results/minpi-study/replay-proxy-normalization
```

原生计时使用显式编译好的本地动态库。源码、构建命令、编译器、库摘要及一次性编译时间记录在构建清单中。同一新测量及其复核应使用同一份库和清单。无本地库时，`rule="native"` 的日常调用会使用 NumPy。UCI 专项还需本地两个原始 ZIP，脚本在读取时检查各自 SHA-256。[构建脚本](../../experiments/build_native_pivotal.py) · [真实数据加载及校验](../../experiments/minpi_constant_transfer.py)
