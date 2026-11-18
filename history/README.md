# minPI 理论与算法研究

本项目研究 pivotal minPI 怎样选出更好的特征组合，以及选列的开销能否由训练和预测节省的时间抵消。算法从训练样本构造近邻投票，按同伴数计算边际贡献并取最低分。评价时记录所选列数、重拟合后的测试分数和从选列到预测的总时间。

| 要解决的问题 | 工作入口 |
| --- | --- |
| 单列排名何时错过好组合？哪些性质可以证明？ | [理论、反例和两篇参考论文的阅读笔记](research/theory/README.md) |
| 怎样从原 minPI 得到新的组合，如何调用与测试？ | [算法入口](research/algorithms/README.md) |
| ANOVA F、随机对照、岭模型和自定义对照分别怎样计算？ | [基线的定义、适用范围和独立核验](research/baselines/README.md) |
| 出处论文定义的离散 mRMR–MID 怎样实现、复核和比较？ | [论文基线实现与配对实验](research/comparators/README.md) |
| 怎样判定一个配置达标，怎样解释原 69 组和额外 57 组？ | [评价协议与证据入口](research/evaluation/README.md) |
| 所有数据放在一起，新版 minPI 相对旧版、全特征、同宽 F 和随机选列，用多少分数换了多少时间？ | [新版方法与全体结果](research/lowdim/README.md) · [全部数据对照表](research/lowdim/evidence/all-results.md) |
| 低维条件岭收益的算法、配置和实证结论如何写成论文？ | [论文 PDF](paper/minpi-lowdim/main.pdf) · [源稿与配套数据](paper/minpi-lowdim/README.md) |
| 原固定预算稿、历史版本、数据和复现包在哪里？ | [归档导航及清理记录](research/archive/README.md) |

**从这里运行。** 在 PowerShell 中切换到本 README 所在的 `MastersThesis/history`。下列入口调用共享环境，并从这里的 `src` 导入源码。第一条检查理论反例，第二条检查组合修正，第三条核验既有基线。

```powershell
.\thesis-python.ps1 research/theory/check_vote_exchange.py
.\thesis-python.ps1 -m pytest research/algorithms -q
.\thesis-python.ps1 research/baselines/verify_existing.py
```

**判断方法有没有用。** 在相同列数、样本划分和下游学习器下与有出处的监督筛选器及均匀随机选列比较测试分数。对每种实际部署方案报告列数、测试分数和含选列的完整耗时。当前先统计实际删列且更快的配置，公开对应分数损失，并单独统计得分相对同宽基线也提高的配置。原实验中最多下降 0.02 分的联合成功数保留作历史比较。跨来源结果按数据集和划分报告，共用测试集的种子用于考察可重复性。

**当前进展。** 给定同伴后重拟合岭模型的 minPI 在 HAR、ISOLET 与新增阿拉伯语数字的 Ridge 实验中，8 个配置都比使用全部训练行的同宽 ANOVA F 更准、更快。把 F 的样本预算也匹配到同样 768 行，minPI 仍有 8/8 更准，2/8 的总耗时在两个进程中均更短。换四种最终学习器后，HAR 的 12 个配对中有 10 个比独立半宽 F 少用列、更准、更快。Gas 漂移和小矩阵 Digits 各自暴露了排序质量和成本问题。[本轮算法与四阶段实验](research/four_stage/README.md) · [同训练行 F 对照](research/conditional_dense/sketch_f/evidence/cases.csv) · [四种最终学习器的迁移复核](research/conditional_dense/transfer/README.md)

放宽损失门槛后的新选宽规则按 minPI 得票的集中程度进一步删列。在未参与开发的英语口述数字数据上，三个互不重叠的说话人测试折全部从 208 列压到 117–127 列，完整时间比全特征少 10.5%–18.6%，两折同时比全行 F 和同样本 F 更准、更快。三折 BA 平均比上一版低 1.23 个百分点，比 Full 高 0.90 个百分点。[新选宽与六来源试验](research/conditional_dense/vote_width/README.md) · [新来源逐折验证](research/conditional_dense/fsdd/README.md)

最初稿的四个二分类和两个回归来源也已用集中度版重测。在相同列数下，Diabetes 单列及 Communities and Crime 的 11 列支持均比原始近邻 pivotal minPI 得到更高的 Ridge R²。对后来的 69 组，全部二分类与回归子集均重放，21 个二分类配置都加速，24 个回归配置中有 12 个加速。SECOM 的三个种子均漏掉全部测试缺陷。[原稿 30 个配置与等宽对照](research/conditional_dense/original_tasks/paper-results.md) · [原 69 组中 45 个二分类及回归配置](research/conditional_dense/original_tasks/README.md)

先前的单次等宽换列能在构造反例上将冻结票表多数收益从 2/3 提高到 1。Digits 三折中该版本相对全特征的三项联合成功为 0/3。[上一轮逐折结果及 mRMR–MID 复核](research/comparators/README.md)

旧稿、旧实验及归档复核程序继续使用原路径。后续理论、算法和基线的共同入口为 `research/`，历史稿件的结论及逐轮数据见[固定预算研究索引](docs/minpi-study/README.md)。
