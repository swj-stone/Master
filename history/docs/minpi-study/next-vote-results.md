# 自适应 minPI 的删列、预测分数与总耗时

成功以单个外层配置为单位。筛选器须删除至少一列，测试分数较全特征最多低 0.02，且选列、切片、标准化、重训、预测的总耗时中位数更短。下文与筛选基线的联合比较另外要求同一组中，minPI 的特征数不超过基线、测试分数较它最多低 0.02、总耗时也比它短。全留计为失败。二分类文本采用平衡准确率，即两类召回率的平均值；回归使用 R²。耗时为本机单线程实测，包含选择器开销。[预定标准](next-vote-protocol.md) · [逐组及逐基线汇总](../../results/minpi-study/next-vote-comparison-20261117a/summary.csv)

## 通用数值表与有序光谱

| 训练内改法 | 面板 | 相对全特征三项同时达标 | 同时通过全特征、半宽 F 和半宽随机 |
|---|---|---:|---:|
| 按投票一致性和有效维数给宽度设下界 | 原 69 / 额外 57 | 43/69 · 13/57 | 3/69 · 0/57 |
| 联合岭模型中按同伴数投票，验证候选前缀 | 原 69 / 额外 57 | 20/69 · 9/57 | 10/69 · 0/57 |
| 在另一组列中选强同伴，同时限制最多保留半数 | 原 69 / 额外 57 | 29/69 · 13/57 | 11/69 · 0/57 |
| 高相关相邻波段中投票选代表 | 7 个近红外来源，各 3 个种子 | 13/21 | 6/21 |
| 在独立训练行上校准波段宽度 | 10 个近红外任务，各 3 个种子 | 7/30 | 1/30 |
| 按波段分配不同名额并在训练内校准 | 12 个近红外任务，各 3 个种子 | 14/36 | 3/36 |

原 69 组来自 13 个来源，含共用同一外层划分的选择器种子。额外 57 组包含多种数据和压力测试，两个面板都参与过先前的开发。同场重测上一版自适应 minPI（signal_vote），相对全特征达标 38/69 和 19/57 组。不同光谱行也有共同任务。以上计数分别保留，不能相加作为独立数据集数。半宽指按原列数向上取整保留一半，F 使用分类的单变量方差分析或回归的单变量 F 排名，随机在训练列中均匀无放回抽样。同宽对照则在每个配置免费获得候选算法所用的确切列数。

限制半宽的通用方案在原面板有 52/69 组、额外面板有 43/57 组的分数超过各自 20 个同宽随机子集的中位数。把同宽 F、同宽随机、原自适应投票与两项独立半宽基线都纳入，这一版在原 69 组只有 Gas 的一个种子达标，额外 57 组为零。该组保留 64/128 列，minPI 分数 0.819、总耗时 81.21 ms；同宽 F 为 0.837、87.55 ms，随机为 0.687、91.60 ms，全特征为 0.667、310.80 ms。初始光谱实验事前指定的四个大样本来源中，9/12 个种子相对全特征达标，同时通过两个半宽对照只有梨的 3/12。该方向没有达到各来源都稳定胜出的目标。[通用逐组实测](../../results/minpi-study/capped-vote-69-20261117a/cases.csv) · [光谱基准](next-vote-protocol.md) · [扩大光谱来源的实验](allocation-vote-protocol.md)

## 稀疏二分类文本

以每份文档是否出现某一词作为一列，仅在当前训练折拟合词表，所有参与筛选的词项都在训练中实际变化。对于某个词，用相互隔离的训练子样本计算添加它带来的预测损失下降。分别比较仅有截距时和已经包含一个强预测词时的贡献，以两种贡献中的较小值排名。自动选择覆盖 99% 正贡献总量的最短前缀，并设词表宽度平方根的下界和半个词表的上界。各折自行确定列数。稀疏实现直接从词项、类别和同伴的共现计数计算留出损失。[选择公式及实现](../../src/minppi_mofs/document_vote_minpi.py) · [预定文本实验](document-vote-protocol.md)

短信、Amazon、IMDb、Yelp 使用各自三个不同的外层文档划分。YouTube 每次留出一整个视频。相同 17 个测试划分在两个新进程分别计时，每轮内各方法轮换顺序。参考实现相对全特征的联合计数为 15/17 和 14/17，其中 14 组在两轮都至少快 10%。同时通过全特征、半宽 F、半宽卡方和半宽随机的为 13/17 个相同划分，包括 SMS 2/3、YouTube 5/5、Amazon 3/3、IMDb 1/3、Yelp 2/3。每折实际抽取 20 个同宽随机子集，17 折共 340 个，全部低于 minPI 的分数。保留比例的中位数为 8.83%。计时从数值词袋开始；分词和建词表是共享的准备成本，把它加回，端到端相对加速比例就会减小。[参考实现首轮](../../results/minpi-study/document-vote-20261117b/vote-comparisons.csv) · [第二轮](../../results/minpi-study/document-vote-20261117c/vote-comparisons.csv)

以已选列数为共同预算，单独去掉强同伴投票、保留相同的跨折估计和并列规则，完整投票在 17 折中 12 胜 5 负。SMS 为 3 胜，YouTube 为 2 胜 3 负。进一步与全训练样本的单词边际排名比较，完整投票为 6 胜 1 平 10 负。三种监督排名都在每折胜过所有 20 个随机支持。实验支持监督选词优于同宽随机；在所有来源均须保留强同伴这一步的结论尚缺少支持。[同伴消融与原始支持集](../../results/minpi-study/document-peer-audit-20261117a/by-source.csv)

作为批次转移测试，另取 Apache SpamAssassin 两批邮件。3,088 封训练、2,420 封测试，词表有 52,260 个变化词项。正文中去掉空文档和重复词袋，使用官方勘误，按发布批次隔离。三个种子共用同一份独立测试集。参考实现保留 2,180 至 2,827 词；全词表平衡准确率为 0.824，三种种子分别得到 0.851、0.825、0.739。参考实现三次总耗时均长于全词表。这一结果区分了短消息和数万词的跨批次长邮件。[批次划分与排除规则](mail-vote-protocol.md) · [逐种子结果](../../results/minpi-study/mail-vote-20261117a/vote-comparisons.csv) · [发布方语料说明](https://spamassassin.apache.org/old/publiccorpus/readme.html)

## 加速后的结果与换种子检验

原生内核直接遍历压缩稀疏行矩阵，在同一内核中累计类别、词项及强同伴的共现统计量。支持集选择、两种同伴条件以及投票量预算与参考算法相同。在 17 个 UCI 配置和 3 个邮件配置，两种实现的全部逐列增益、排名依据、所选列和测试预测相同。编译一次约 0.43 秒；下表计时包括每次重新筛选、取列、标准化、拟合与预测。每个进程对 13 种方法各计时 13 次。表中成功是两个独立进程在同一个配置上都成功。[计时协议](fast-document-timing.md) · [两轮及两轮交集](../../results/minpi-study/fast-document-comparison-20261117a/summary.csv)

| 两轮共同满足的条件 | SMS 与 YouTube 垃圾消息 | 三种短评情感 | 独立邮件批次，3 个种子 |
|---|---:|---:|---:|
| 删列、保住全特征分数且总耗时至少少 10% | 8/8 | 7/9 | 2/3 |
| 同时通过全特征及四种半宽对照（F、卡方、随机、边际岭） | 8/8 | 7/9 | 2/3 |
| 同时通过全特征及同宽的全训练单词边际岭排名 | 7/8 | 0/9 | 0/3 |
| 同时通过全特征及同宽均匀随机 | 0/8 | 0/9 | 0/3 |

17 个 UCI 划分保留词表宽度的中位数为 8.83%，垃圾短消息的 8 个划分为 5.42%。全部 17 折中，15 折在两轮同时过四种半宽对照且每项至少快 10%，其中 13 折相对全部这些对照都没有掉分。与同宽 F 和卡方两者同时比较为 12/17；进一步加入同宽全训练单词边际岭为 6/17。与参考实现相比，二分类稀疏加速将 UCI 选词耗时降到原来的约 26%，完整模型链耗时降到原来的约 69%–70%。增加相同的词表准备时间后，各配置的时间比例可在两轮逐组记录中复算。跨批次邮件仍有一个种子的平衡准确率比全词表低 0.085，数值等价加速只能解决其耗时。[来源与基线](../../results/minpi-study/fast-document-comparison-20261117a/by-source.csv) · [首轮逐组含建词表耗时](../../results/minpi-study/fast-document-20261117a/vote-comparisons.csv)

在 8 个垃圾短消息划分上固定另外三个选择种子 6173、8111 和 9923，使用同一算法和规则。SMS 为 9/9，YouTube 为 15/15，全部 24 个配置同时通过全特征、半宽 F、半宽卡方及半宽随机。相对全词表，SMS 总耗时减少 36%–55%，YouTube 减少 69%–74%；保留数分别为 269–365 词和 164–279 词。相对同宽 F 通过 23/24，相对同宽全训练单词边际岭通过 19/24。相对同宽随机，24 个配置全部提高平衡准确率，同时全部耗时更长；分数提高中位数在 SMS 为 0.236，在 YouTube 为 0.244。跨种子的单折分数最大跨度分别为 0.0193、0.0378。这里检验的是相同真实测试划分上的选列种子敏感性，24 个配置仍只涉及原来的 8 个测试划分。[新种子逐折结果](../../results/minpi-study/seed-document-spam-20261117a/by-fold.csv) · [192 份方法输出复验](../../results/minpi-study/seed-document-spam-20261117a/fresh-process-audit.csv)

为与原论文的矩阵直接衔接，另在原 69 组中的 CNAE 词频矩阵、额外 57 组中的 Dexter 文档矩阵重测相同规则。保留原始输入、训练划分、分数和下游模型；筛选时才转换稀疏矩阵并计时，同宽随机只在实际变化的词中抽取。CNAE 三折均通过相对全特征及只去常量的三项标准，保留 157–171/856 列；Dexter 有 2/3 个种子达标，保留 959–1,074/20,000 列。进一步要求比半宽 F 和随机均达标，分别为 2/3 和 2/3，这里的半宽按变化词项数计算。CNAE 使用多分类参考实现，Dexter 使用二分类内核。[六组与历史全特征预测逐项核对](../../results/minpi-study/document-archived-20261117a/cases.csv)

## 接口与复现

在 `MastersThesis/history` 中通过本目录的 PowerShell 启动器运行。词袋可由稀疏的非负词频或 0/1 矩阵提供。筛选器在计票时使用词是否出现，输出时保留所选列的原始值；重现实验的下游分类器应使用 0/1 词袋。换外层划分须重新拟合词表和筛选器。

```python
from minppi_mofs.support_selector import PivotalSupportSelector

selector = PivotalSupportSelector(rule="document_vote", fraction="auto", random_state=1129)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
kept_columns = selector.get_support(indices=True)
```

```powershell
.\thesis-python.ps1 -m experiments.fetch_document_vote_data
.\thesis-python.ps1 -m experiments.build_native_document
.\thesis-python.ps1 -m experiments.document_vote_benchmark --output results/minpi-study/document-vote-20261117b --verify
.\thesis-python.ps1 -m experiments.mail_vote_transfer --output results/minpi-study/mail-vote-20261117a --verify
.\thesis-python.ps1 -m experiments.fast_document_vote_benchmark --pass-id 1 --output results/minpi-study/my-fast-pass-1
.\thesis-python.ps1 -m experiments.fast_document_vote_benchmark --pass-id 2 --output results/minpi-study/my-fast-pass-2
```

每份计时档案记录单次耗时、列索引、逐条预测、文件摘要和各基线的相对结果。编译是一次性操作，训练时直接装载。缺少原生库、使用多分类或不符合原生索引要求时，公共接口使用参考算法。通用候选接口另外提供 `budget_vote`、`ridge_vote`、`capped_vote`；三项光谱规则需要波长已按采集顺序排列。[公共接口](../../src/minppi_mofs/support_selector.py) · [加速实现](../../src/minppi_mofs/fast_document_vote_minpi.py)
