# 自适应 minPI 能否从变化的列中选出更好的列

## 原 69 组与额外 57 组

“胜随机中位数”比较测试分数，每组独立抽取或用预定种子计入共 20 个相同宽度的随机支持。“随机联合”使用计时的预定随机子集，分别在每组计算实际删列、相对全特征最多掉 0.02 分和端到端加速。有一个额外配置保留全部列，联合计零。

| 面板 | 实际删列 | minPI 保分 | minPI 联合 | 随机联合 | F 联合 | 零同伴联合 | 胜随机中位数 | 联合且胜随机中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原 69 组 | 69 | 45 | 39 | 38 | 45 | 45 | 48 | 30 |
| 额外 57 组 | 56 | 33 | 20 | 16 | 23 | 22 | 38 | 19 |

原 69 组和额外 57 组保留比例的中位数分别为 60% 和 50%，选列耗时中位数分别为 0.746 和 0.371 毫秒。与上一轮的稳定版自适应投票在同场、同一划分比较，原 69 组联合数由 40 变为 39，保分数由 66 变为 45；额外 57 组联合数由 9 变为 20，保分数由 36 变为 33。新方法删列更积极，密集分散信号可能丢失。在额外面板的 12 个独立稀疏信号模拟上，minPI 的 12 个分数均优于各自全部 20 个同宽随机支持，联合成功 9/12，随机 0/12。早段、晚段两组计时均加速的只有 5/12。额外面板的稠密信号模拟为 0/12。将所有面板放在一起的总次数不能替代分布迁移证据。[原 69 逐组归因](../../results/minpi-study/signal-vote-69-20261117a/vote-comparisons.csv) · [额外 57 逐组归因](../../results/minpi-study/signal-vote-transfer-20261117a/vote-comparisons.csv)

## 哪种输入从投票中获益

**独立变化的干扰列。** 预定类别为至少 128 列，训练行数至少为列数的两倍，按数据设计已知至少四分之三为独立于响应和原始测量的干扰列。分类和回归模拟各 4/4、ANES96 分类和 Fair 回归两个调查各自补入干扰列后各 3/3，合计 14/14 达到强化判据。各组实际保留 11–71 列，均胜过至少 30/32 个同宽随机子集且分数超过全特征。11/14 组至少快 10%，ANES96 的三个配置加速幅度均小于 10%。同场单次随机为 2/14 联合，同宽 F 和零同伴消融均为 14/14。模拟数据的 12 个真实预测因子中，新方法每组找回 10–12 个，同宽单次随机只找到 0–4 个，F 每组全找回。未经扩列的两份原始调查六组均保分，六组均未加速。[专项逐组结果](../../results/minpi-study/signal-vote-confirmation-20261117a/contest.csv) · [独立随机对照](../../results/minpi-study/signal-vote-confirmation-20261117a/random-draws.csv)

**一个强信号被多列重复测量，同时还有独立的较弱信号。** 另用两个预定的重复测量数检验同伴投票的作用。128 个连续、变化的特征中，分别有 12 或 48 个相关系数约 0.98 的特征测量同一潜变量，另有 4 个独立预测变量，其余为独立噪声。每一设置独立生成 5 个分类、5 个回归问题。源代码先冻结，得到同宽预测成绩后再对两个设置的全部 20 组追加七轮计时及每组 32 个随机支持。

| 重复测量数 | minPI 联合 | 随机联合 | 零同伴联合 | F 联合 |
|---|---:|---:|---:|---:|
| 48/128 | 9/10 | 0/10 | 0/10 | 1/10 |
| 12/128 | 9/10 | 0/10 | 9/10 | 10/10 |

48 列重复时，新版自选 27–52 列，9/10 组留下全部 4 个独立预测变量，零同伴为 0/10，F 为 1/10。对零同伴，分类平衡准确率的配对差中位数为 +0.119，回归决定系数差中位数为 +0.260。成功的 9 组均至少加速 10%，余下一组回归的分数下降 0.0261。把重复测量数降到 12 后，同伴步骤的优势消失。两个水平的 640 份随机支持中，minPI 的测试分数严格高于 639 份。这是对两档模拟机制的检验，实际未知分布中的重复列比例和互补信号未必可知。[同伴机制](../../results/minpi-study/signal-vote-peer-mechanism-20261117a/by-mechanism.csv) · [同宽计时和随机抽样](../../results/minpi-study/signal-vote-peer-timing-20261117a/vote-comparisons.csv)

## 真实来源复核

将重复测量机制迁移到两份新的真实目标上时，使用原始的钞票真伪、风洞声压级作为标签，在每个训练折只用训练数据识别最强原始测量，为它添加 47 个带独立误差的重复通道，最后用独立变化的干扰列补至 128 列。两来源共六折全部保分、加速且严格胜过至少 27/32 个同宽随机支持。三项达标为 minPI 6/6、同宽单随机 1/6、F 和零同伴各 3/6。Airfoil 三折中两折同时胜过 F 及零同伴，达到事先设置的归因条件；在这两折中，minPI 保留了其余四项原始测量，两个对照均遗漏全部四项，minPI 的测试决定系数比 F 分别高 0.398、0.319。Banknote 仅一折同时胜过这两项对照。两来源总耗时比分别为 0.926、0.942（各自三折中位数），六折均未达到至少快 10% 的标准；原始低维版本均为 0/3 联合达标。人工扩列的半合成结果检验了重复通道造成的竞争。Airfoil 行级随机折只检验同一测量表内的泛化。[独立预案](signal-vote-real-proxy-protocol.md) · [逐折测量与消融](../../results/minpi-study/signal-vote-real-proxies-20261117a/contest.csv) · [UCI Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) · [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

在挑战发布方的探针数据上，所有方法都从同一批训练中实际变化的列开始。Dexter 官方验证从 7,751 个变化列中选择 720 个，平衡准确率 0.870，比全特征低 0.0033，胜过 32/32 份同宽随机支持；总耗时缩短约 35%。Gisette 三个互斥样本池的结果均胜过 32/32 份同宽随机支持，2/3 组强化达标；剩余一组下降 0.02148，超过 0.02 容差。Arcene 只赢 16/32 个随机支持。三个来源未达到预先设定的跨来源条件。手写数字 Multiple Features 的 2,000 个原始描述向量中有完全相同的记录。首次核对发现两个测试折各有四行与训练行重合，因此在原始折分配上排除所有重复向量的全部副本，共剔除 12 行，再将全部来源重跑。最终八个配置均无跨训练/测试重复向量；Multiple Features 三个配置依次胜过 12、17、21 个随机支持，0/3 强化达标。[去重后的逐组结果](../../results/minpi-study/signal-vote-sources-20261117b/vote-comparisons.csv) · [原始数据与六种图像描述](https://archive.ics.uci.edu/dataset/72/multiple+features)

### 可从训练集判断的条件

在得到测试结果前另定了一个容易计算的标记：训练集至少 512 行、至少 32 列，且 99% 正投票量集中在不超过三分之一的列。原 69 组有 21 组命中，minPI 三项达标 16/21，同宽随机 12/21，F 与零同伴各 19/21；加上胜过至少四分之三随机支持的要求后，minPI 为 8/21。额外 57 组中命中 12 组，均来自稀疏信号模拟，minPI 强化达标 9/12。来源专项只有 Gisette 的三组命中，其中 2/3 强化达标。这个条件可帮助判断训练票是否集中，单凭它不足以预言测试质量或同伴步骤的额外收益。[跨面板核对](../../results/minpi-study/signal-vote-comparison-20261117b/training-condition.csv)

在 48 列重复测量实验之后，还做了探索性检验，考察能否在已有数据找到类似结构。要求至少 512 个训练样本、至少 32 列，且在每个参与投票的拟合子样本中，最强预测列均有至少总列数三分之一的其他列与其绝对相关达到 0.90。该标记识别了 48 列重复模拟的全部 10 组，12 列重复模拟中命中 0/10；原 69 组和额外 57 组均未命中。该标记只用于描述，不是部署路由，表中联合耗时不含这项额外筛查。[训练条件及逐组覆盖](../../results/minpi-study/signal-vote-peer-transport-20261117a/membership.csv)

### 自选宽度对投票量阈值的敏感性

对 20 个重复测量模拟和 14 个独立干扰配置，固定各组的投票排名，将累计正票量阈值依次设为 95%、97.5%、99%、99.5%。四种设置均有 32/34 组实际删列且保分，34/34 组分数胜过该宽度下预定种子的单次随机。48 列重复时，四档保留列数的中位数从 31 增至 43.5，胜过同宽零同伴的配置数分别为 10、10、9、8。该分析只复算分数和宽度，各设置的联合速度结论限于正式计时的 99%。[阈值分层结果](../../results/minpi-study/signal-vote-mass-20261117a/by-group.csv)

## 投票和宽度

选择器在外层训练数据中取两个不相交的子样本，各至多 256 行。在拟合侧先找单列岭模型最强的预测列。对每一列，计算将它加入截距模型时留出平方损失降低多少，再计算已有最强列时，加入它并重拟合两列的系数，能降低多少留出损失。最强列自身以次强列为同伴。分类使用独热目标；有顺序的训练组只沿过去到最近一组的方向评估，普通数据交换拟合侧和留出侧。算法对两个方向取平均，再取两种同伴数中较低的增益。这里的同伴是按训练信号选出的一个具体同伴，其分数代表这两个上下文中的最差值。

按这个分数从高到低，取累积达到全部正分总和 99% 的最短前缀。99% 在本轮测量前固定，表示投票量覆盖率。实际保留列数由该训练划分的分数分布决定。如果全部分数非正，就保留全部特征并在联合评价中记作失败。两次增益都由同一批充分统计量计算，每列只需单列和双列的岭闭式解。[实现](../../src/minppi_mofs/signal_vote_minpi.py) · [用显式岭重拟合核对投票的测试](../../tests/test_signal_vote_minpi.py)

## 实验如何归因

每组都有使用全部特征的岭模型、与自适应 minPI 列数逐组完全相同的均匀随机子集、同宽 F 检验、同宽单独一列的留出投票。最后一项只删除有同伴时的分数，是排名消融，计时实现复用完整评分器。F、随机免费获得对应的宽度；所有下游模型均使用完整外层训练集拟合。测试均不参与选列。用预热后轮换运行的总耗时中位数，统计实际删列、测试分数最多下降 0.02 且加速的配置。另在每组固定下游模型重复抽同宽随机支持，强化达标还要求测试分数严格胜过至少四分之三的抽样。多个选择器种子共用同一个外层测试集时，不能将其解释为多个独立数据集。

三项外部检验各有不同含义。原额外 57 组沿用既有划分，验证能否迁移到此前比较过的任务。独立的生成与调查面板预定检验“大量独立变化干扰列”这一有生成背景的类别；八个新模拟实例和两个加噪调查数据源的三个外层折均在本轮冻结规则后检验，未经扩列的调查样本另报。公开来源专项先从所有方法的共同输入中去掉训练常量，然后在实际变化的列上比较，另含完整的 UCI Multiple Features 自然测量。三个试验的达标分母分别保留。[类别与判据](signal-vote-confirmation-protocol.md) · [来源与训练条件](signal-vote-source-protocol.md)

## 调用与复算

```python
from minppi_mofs.support_selector import PivotalSupportSelector

selector = PivotalSupportSelector(rule="signal_vote", fraction="auto",
                                  task="classification", random_state=1129)
selector.fit(X_train, y_train)
columns = selector.get_support(indices=True)
X_reduced = selector.transform(X_test)
```

在 `MastersThesis/history` 依次运行以下命令，使用尚不存在的输出目录。计时完成后，在相同命令后加 `--verify` 进行新进程复验。前两条再以 `--random-sensitivity` 生成每组 20 次同宽随机成绩。额外两项直接生成每组 32 个随机对照。基准直接测量选择函数、取列、预处理、岭模型拟合和预测；公开估计器还包含 scikit-learn 的常规输入校验。

```powershell
.\thesis-python.ps1 -m experiments.signal_vote_benchmark --panel original --output results/minpi-study/replay-signal-69
.\thesis-python.ps1 -m experiments.signal_vote_benchmark --panel transfer --output results/minpi-study/replay-signal-57
.\thesis-python.ps1 -m experiments.signal_vote_confirmation --output results/minpi-study/replay-signal-confirmation
.\thesis-python.ps1 -m experiments.signal_vote_sources --output results/minpi-study/replay-signal-sources
.\thesis-python.ps1 -m experiments.signal_vote_real_proxies --output results/minpi-study/replay-real-proxies
.\thesis-python.ps1 -m experiments.signal_vote_peer_mechanism --output results/minpi-study/replay-peer-scores
.\thesis-python.ps1 -m experiments.time_signal_vote_peer_mechanism --prior results/minpi-study/replay-peer-scores --output results/minpi-study/replay-peer-times
```

公开来源专项需要工作区中注册的 Gisette 快照和三个 UCI 压缩包，文件摘要由脚本验证。原 69 与额外 57 组的 756 份方法输出均在独立进程重放；新算法与历史 minPI 的定向回归测试共通过 258 项。[选择器入口](../../src/minppi_mofs/support_selector.py) · [定向测试记录](../../results/minpi-study/signal-vote-comparison-20261117b/minpi-tests.xml) · [独立重算的结果表](../../results/minpi-study/signal-vote-comparison-20261117b/summary.csv)
