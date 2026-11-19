# minPI 固定预算特征筛选研究

[最新：组合选择、自适应宽度与强求解器成本审计](combination-exploration-results.md)

[方向一致性投票的实现、完整对照及失败边界](reliability-document-results.md)

[新一轮通用、近红外及稀疏文本的投票改进和基线比较](next-vote-results.md)

[自适应投票选列及独立随机支持对照](signal-vote-results.md) · [上一轮自选宽度与稀疏实验](adaptive-transfer.md) · [固定 50% 和 75% 的联合达标结果](compression-speed.md)

[Python 筛选器 PivotalSupportSelector](../../src/minppi_mofs/support_selector.py)。下文历史记录中的 `paper/minpi/` 路径对应复现包内的目录。工作区版本可在 Git 提交 `3167108` 中查阅。

[本地复现包](../../packages/pivotal-minpi-study.zip)附有本研究使用的输入、程序、分区、预测、论文和逐文件 SHA-256 摘要。验证压缩文件本身可比对[整包 SHA-256](../../packages/pivotal-minpi-study.zip.sha256)。

本研究从 `polish1` 的 hit/miss 三元组出发。每一列就同类邻居和异类邻居谁更接近查询样本投票。算法求出该票在指定同伴数下改变多数表决结果的平均边际贡献，按几个同伴数中最低的贡献排列各列。特征预算由调用者给定。用于计算排名的行全部取自训练数据。

### 当前主稿与复审

[当前独立审稿及历次退修理由](current-review.md) · [参数选择及敏感性](parameter-design-protocol.md) · [独立计算复核](independent-audit.md) · [本轮完整测试记录（1,538 项通过、4 项跳过）](../../results/minpi-study/round8-full-tests.xml)

主稿研究给定投票矩阵、同伴数集合和预算时，如何返回与数学定义相同的列集合，包括边界并列时的原索引顺序。训练数据、三元组和列顺序都影响问题本身。`PivotalSupportSelector(arithmetic="rational")` 启用原生整数或定点包围与有理数回退；`rational_top_from_votes(..., bits=b)` 单独运行定点后端。显式 `tiers` 接受合法同伴数，示例为 `(0, 1, 4, 16)`。直接处理外部三元投票矩阵可使用 `minppi_mofs.word_minpi.exact_top_from_votes`。

| 审稿中的问题 | 核查和复现材料 |
|---|---|
| 三种精确算法与未经验证的浮点算法差多少时间 | [配对计时协议](arithmetic-price-protocol.md) · [三种预算的全部结果](../../results/minpi-study/arithmetic-price) |
| 真实换列和唯一胜者各需多少纠错工作 | [固定矩阵的配对计时](boundary-cost-protocol.md) · [对称反例在所有允许精度上的记录](../../results/minpi-study/boundary-cost) |
| 有多少回退可以从逐列弃权信息直接消去 | [新旧实现的重放协议](zero-bounds-protocol.md) · [同一组目标、四种精度](../../results/minpi-study/zero-bounds) |
| Gas 索引换列的预测影响及机器字边界 | [固定模型、批次及算术界协议](arithmetic-consequence-protocol.md) · [逐批准确率与逐面板整数界](../../results/minpi-study/arithmetic-consequence) |
| 上一轮三种精确算法之间的计时 | [2%、10%、50% 三档预算协议](budget-comparison-protocol.md) · [逐次计时](../../results/minpi-study/budget-comparison/trials.csv) · [按机器字范围汇总](../../results/minpi-study/budget-comparison/summary.csv) |
| Gas 边界究竟如何换列 | [真实输入的整数、双精度结果和摘要](../../results/minpi-study/gas-replay/replay.json) · [单例重放脚本](../../experiments/minpi_study_gas_replay.py) |
| 数据准备成本、旧大整数对照与原生整数对照 | [原始六种算法计时](../../results/minpi-study/raw-cost) · [原生整数择路协议](word-design-protocol.md) · [对同一面板计时的配对比值](../../results/minpi-study/word-design/paired-ratios.csv) |
| 同伴数、预算、精度与边界并列 | [六种同伴数配置与精度的完整实验](../../results/minpi-study/precision-design/findings.md) · [改变起始层](../../results/minpi-study/anchor-design/candidates.csv) |
| 训练见证是否稳定 | [每个配置的 Jaccard 与数据摘要](../../results/minpi-study/witness-design/cases.csv) · [见证参数实验](witness-design-protocol.md) |
| 无弃权时的排序充分条件是否适用（原命题 3） | [逐矩阵的精确充分条件和实际排序](../../results/minpi-study/order-diagnostics/findings.md) |
| 模拟与真实迁移差距 | [票的方向、绝对枢轴权重与两设备类轮廓](../../results/minpi-study/transfer-gap/findings.md) |

### 第八轮意见逐项核实

[第八轮审稿原文](../../paper/minpi/README.md)对应本次修改前 11 页、362,869 字节的 PDF（SHA-256 前缀 `9ef53aac`）。以下页码指该审稿输入。重新计时和抽样所用程序、原始数据及适用范围见各行链接。

| 原审稿位置 | 对照 PDF 的核实和本轮处理 |
|---|---|
| 第 2、8 页 · 组合数、标题与见证敏感性 | 式（3）原倒数作用于整个组合数，归一化正确。现在写为 `1/D_k`。第 5 节原题已与范围相符；23 张固定票矩阵上的算法一致性与重新抽取见证后的不同选集均有原始结果支持。 |
| 第 4、5 页 · 命题 2、3，真并列与全回退 | 40/42 是大于充分间隙的面板数，39/42 是对称边界带只有一列的面板数。严格赢家的两个同负列为 4094、4095；另一构造精算 2,048 列，含 2,047 个并列赢家。互反双行原始共同尺度走区间路径，每列区间都为 [-1,0]。已将后端选路和逐列回退的证明分别叙述。新审稿员又构造了两行三列的混合边界：有一列是非静默的点区间，现实现仍精算它。修订稿明确全点边界直接使用端点、混合边界只复用静默列零值。[构造与原始计数](../../results/minpi-study/figure-witnesses/audit.json) |
| 第 2、4、7 页 · 机器字安全和表示 | 默认实现检查原始共同尺度。原始逐层路径可在另外 3 个面板上以机器字求和、用大整数交叉比较。新增 GCD 方案检查实际出现的枢轴分子。128 列互反例的共同上界由 70 位降到 40 位；原第 4 页的 67 位是单层分母，其两倍为 68 位。严格赢家及并列 4,096 列例约分后仍为 119 位。独立有理数核对了 42 个面板及四个构造或边界个案上所有层的顺序。[算法与计时边界](reduced-scale-protocol.md) · [逐面板和构造](../../results/minpi-study/reduced-scale) |
| 第 6、7 页 · 最强同规格对照与计时范围 | 静态与动态截点同属命题 2 的精确锚层实验。半宽 Gisette 节省 646 次列评分；两种冻结实现的运行时比接近 1。表 2 的三组主要比值和五法的计时位置经原始记录复算。正文将结论限定于单次预算，并注明 IQR 为面板间差异。同一矩阵、同一层集合的多预算查询可复用完整排名。用户已有名单时可审计或纠正；原 126 次供给有效名单的记录不支持一般性的提速。[逐预算记录](../../results/minpi-study/training-verifier/five-arm) · [原动态截点实验](dynamic-anchor-protocol.md) |
| 第 7 页 · 未来预测损失 | 在既有 Gas 批次 8、9、10，浮点方案相对精确方案的等批平均 BA 损失是 0.011885；第 9 批单批损失为 0.080655。新版要求预先规定部署单位、权重、容忍度和单侧不确定性判据，并在独立单位检验。[按预测记录复算](../../results/minpi-study/predictive-controls/audit.json) |
| 第 8、9 页 · 近邻并列和总体条件 | 原通用实现先分区，再在留下的最多四个参考样本中以参考库内位置打破距离并列。跨分区边界没有全局次键。41 个面板共 10,496 个 hit/miss 池，13 个在边界跨界并列。改用全池双键排序改变两处参考选择，所有票和原预算选集相同。MHEALTH 用递增窗口索引上首次取到的最近参考。0.912 是条件独立查询模型下的总体间隙门槛，实测样本间隙没有验证该条件。[逐池与逐面板记录](../../results/minpi-study/neighbour-ties) · [等距反例与协议](neighbour-ties-protocol.md) |
| 第 10 页 · 表 3 随机基线与材料评估量 | 原 Random 为每个来源等概率、不放回抽一份等宽名单，所有留出单位共用。固定训练、测试和学习器另抽每源 256 次，超过 minPI 的次数为 255、241、130。材料训练见证种子 3469 的两份名单相差三列。用种子 13469 在同一标签留出集建票，计算所选各列最低层分的均值，minPI 相对最大层的差为 −0.001053，测试 R² 差为 +0.059510。两个读数共享留出数据。[预测协议](predictive-controls-protocol.md) · [所有名单与预测](../../results/minpi-study/predictive-controls) |
| 第 9、10 页 · 排序命题及对新来源的解释 | 对每行绝对票差至少为二的矩阵，加入全体同伴使每列最低层分至多为零；单行 (+,+,+,−) 在层 0、3 下的分数为 (0,0,0,−1)。命题 5 的间隙条件足以保序，失败时仍可直接求原目标。原预算下，min 与最大层有 36/42 个面板名单一致。配对训练和预测过程固定时，一致名单的收益差为零；估计新来源平均收益还需要目标人群中名单变化的频率。[面板范围和支持核对](../../results/minpi-study/predictive-controls/audit.json) |

### 第七轮意见逐项核实

审阅对象为 361,987 字节、SHA-256 前缀 `68c07639` 的 11 页 PDF。下表的原页码均指该文件，意见原文保存在[审稿 README](../../paper/minpi/README.md)。

| 审稿疑点 | 原稿核实与修订位置 |
|---|---|
| 截图 1 · 预算及索引 | 原第 5 页的预算为 1。0 和 1 是两列的零基索引。第 4.1 节逐行给出严格赢家矩阵，图 1d 的结果段说明另一矩阵只更换第 2 行。差值仍为 `2/908054843662451242401`。 |
| 截图 1 · 19 个并列与 1 个回退 | 原图计数一致。19 是有并列的目标数，1 是这些目标中在 40 位仍需精算的个数。新版图注列出 0/83 与 1/19，唯一后者为 Gas，有三列精算。[逐档重算](../../results/minpi-study/figure-witnesses/findings.md) |
| 截图 1、4 · 第 5 节标题与见证抽样 | 标题改为见证构建、选集稳定性与预测。模拟、跨设备比较与留出预测有各自标题。23 个给定矩阵上精确结果一致，见证翻倍后保留列数仍可分别降至 158/500、9/13、0/9。[逐次支持](../../results/minpi-study/witness-design/cases.csv) |
| 截图 2 · 层分母 | 原第 2 页印的是组合数。现在先定义每个层分母，再对这些整数取最小公倍数。 |
| 截图 3 · 退化与整数位宽 | 原第 4、10 页已证明互反双行可令 128/128 列回退。复算全部 61 种允许精度；128 列时的层分母为 67 位，乘 2 后为 68 位，180 位属于 16,384 列的例子。正间隙免精算的 40/42 和边界带仅一列的 39/42 也有不同含义。[逐精度证据](../../results/minpi-study/boundary-cost/complement-bits.csv) |
| 独立发现 1 · 图 1d 的构造 | 原文未指定前两行同负列的位置。归档严格赢家矩阵中这两列为 4094、4095。与末七列的交集大小改为 0、1、2 时，零位精度分别精算 4,087、4,088、4,089 列，同一列始终以同一精确分差胜出。另一矩阵的前两行相同，有 2,047 个精确赢家和 2,048 个回退列。记录给出原始列数、不同列型和完整字节指纹。[两矩阵的重放记录](../../results/minpi-study/figure-witnesses/audit.json) |
| 截图 4、独立发现 2 · 精确比较方法 | 表 2 从同一训练输入、同一计时批次对比直接精确选择、精确锚点、纯浮点和两种提议验证，全部比值用[最终五法批次](../../results/minpi-study/training-verifier/five-arm)重算。此前四法计时单独保留。每次包含投票构造，验证方法包含名单生成。另在固定投票上比较静态与动态锚点，计入锚层与排序。Fagin 等研究的是已有顺序访问接口的模型，Shewchuk 给出自适应精度谓词。[端到端计时](../../results/minpi-study/training-verifier) · [动态截点实验](../../results/minpi-study/dynamic-anchor) |
| 独立发现 3 · 统计结论的范围 | Gas 的 0.02 容忍度只说明既有批次上平均与逐批判断可以不同；未来规则要在独立批次检验。图 1c 的 1/102 来自有意选定的压力测试。0.912 是独立查询模型下的总体间隙充分条件，Gisette 的样本间隙没有验证该总体条件。[Gas 逐批重算](../../results/minpi-study/one-peer/audit.json) |

### 第六轮意见逐项核实

本轮意见对应原稿 `8ec39cfa`（360,143 字节，11 页）。各条按该 PDF 核查。

| 意见 | 核查与处理 |
|---|---|
| 1 · 定理 1 的归属 | Bilbao 等命题 1 将排除焦点后的联盟人数与权重和同时编码，并归功于 Cantor。§3 现在写出三种固定票值对应的生成函数，式（3）对联合多元超几何分布求和。每行同号共享计数，再跨层取最小值以认证预算边界，是本文的方法。[原论文命题 1](https://citeseerx.ist.psu.edu/document?doi=b7084c94b091bcec1cebf2563731d30dffd15b68&repid=rep1&type=pdf) |
| 2 · 剪枝与舍入 | 命题 2 的比较要求局外列上界严格低于全体候选分数的一个下界。正文给出不等式链和相等时的处理。式（6）之后另从零分子推出全弃权列的点区间。原稿 Discussion 已说明间隙影响工作量，当前实现最坏要精算全部列。 |
| 3 · 两个截图中的公式 | 原稿互反两行的预算为 1，三行例中 0、1 为列号，差值分母为三倍组合数。现将差值单列排版。运行全部 61 种支持精度，互反两行例始终精算 128/128 列。[整数与区间复核](../../results/minpi-study/rounding-witnesses/audit.json) |
| 4 · 唯一性及 binary64 的两种失效 | 逐一核对三行构造的全部六种列型，给出其余 4,094 列的统一上界和前两名的绑定层。两名精确分数在同一舍入中点之下，仅保存最终 binary64 分数也无法区分。在 Gas 上三列的精确分子相同，单次除法得到同一位型，原先的 16、24 ulp 差异可定位到所测求值路径。[精确分数和位型](../../results/minpi-study/rounding-witnesses/findings.md) |
| 5 · 已有名单的核验开销 | 单独传入已归档投票矩阵和特征名单，配对计时验证与直接精确重算。42 个面板的三档预算共 126 个正确名单全部通过，另单列两个需要纠错的实例。§4.2 与表 2 区分是否计入生成名单的费用。[实验协议](supplied-support-protocol.md) · [逐目标结果](../../results/minpi-study/supplied-support/targets.csv) · [原始时间](../../results/minpi-study/supplied-support/trials.csv) |

表 2 的 `Float + strict check` 与第五轮的 `Verified float` 是同一对照。本轮按原始计时复算了所有单元格，并保留了[旧表的生成程序](../../results/minpi-study/supplied-support/historical-table-audit.py.txt)。

### 第五轮意见逐项核实

审查基稿为 358,707 字节、SHA-256 以 `06815998` 开头的 11 页 PDF。稿号以 PDF 散列为准，表号沿用正式稿。

| 意见 | 核查与修订 |
|---|---|
| 表 2 是否有数字冲突 | 原稿右组 50% 的 1.62 是时间比，表 1 Diabetes 的 1.62 是毫秒。左组的 0.86–0.91 为面板比值的四分位范围。新版在同表补列全部 42 个面板的中位数和四分位点，原始比值来自每面板各方法五次调用的中位时间。[逐数复核](../../results/minpi-study/table-audit/findings.md) |
| 如何选择精确名单或预测容忍度 | 表 2 附近集中说明保证、分支和时间起止，以及保留同一投票、预算和预测器后的配对损失判定。Gas 的事后 64 列案例区别于表 3 的默认 13 列案例。0.02 是说明平均与逐批判定差异的例值。[逐批预测重算](../../results/minpi-study/one-peer/audit.json) |
| 近期精确对照、间隙与最坏开销 | 摘要并报精确锚点及严格验证的浮点提议。Discussion 交代点区间、大间隙和全列回退的成本，并说明互补行可由专用实现直接消去。[精算最坏情况](../../results/minpi-study/boundary-cost/complement.json) |
| 用索引打破等端点能否提前接受 | 逐个测试 126 个名义预算目标和两个错误提议。旧规则因并列未能直接接受的 19 个目标全部通过新规则；Gas 和唯一胜者的错误提议都需纠正。[实验协议](tie-keys-protocol.md) · [逐目标决定](../../results/minpi-study/tie-keys/targets.csv) |
| Gas 单个同伴的代数关系 | 枚举宽度 2–8 所有二元行的 3,584 个焦点，并核对三个归档 Gas 矩阵的每一列。文章给出推导、三个并列列的精确单票及单同伴得分。[核验脚本](../../experiments/audit_minpi_study_one_peer.py) · [Lean 证明](../../paper/minpi/formal/Pivotal.lean) |
| 逐层使用机器字是否更快 | 增加可调用的 `minppi_mofs.layer_word_minpi.layer_word_top_from_votes`，检查每层最坏部分和范围，并实际使用逐层原生整数累加和安全交叉积比较。配对考察 42 个面板、三档预算、两种计时起点。另有 18 个面板的最坏单层界超限，走同一后备证书。三个新增材料面板共用训练划分。[协议](layer-word-protocol.md) · [各面板耗时](../../results/minpi-study/layer-word/paired.csv) · [汇总](../../results/minpi-study/layer-word/findings.md) |

从独立解包根目录运行 `python -m experiments.minpi_study_layer_word --output fresh-layer-word` 可复跑逐层计时。运行 `python -m experiments.audit_minpi_study_one_peer` 可复核 Gas 恒等式、各层有理分数和不同批次权重的影响。工作树内请将 `python` 替换为 `.\thesis-python.ps1`。

### 第四轮意见逐项核实

本轮审查以 358,188 字节、PDF 摘要以 `b79406ed` 开头的 11 页稿为起点。第 1 页原已列出 Gas 并列与构造的唯一胜者，第 2 页原已给出固定输入的范围。修订将两个错误提前，并补充了下列核验。

| 原意见 | 正文和复现结果 |
|---|---|
| 购买精确性的实际代价 | 表 1 与图 1b 保留原实现的默认预算计时。表 2 为更新实现、同一批面板新配对的四种路线，各格给出中位数与四分位范围。正文区分每次重建投票的端到端时间和打分阶段。 |
| 命题 3、间隙与最坏成本 | 命题 3 后将“仍未决”限定为经过两端严格比较的列。40/42 个间隙超过 40 位网格的保守阈值，39/42 个面板的对称带内只有一列。Crime/2027 在已入选一侧并列。另一个构造需要精算 2,048/4,096 列；两行互补反例在 61 种允许精度均需精算 128/128 列。[原始间隙](../../results/minpi-study/raw-cost/cases.csv) · [互补反例](../../results/minpi-study/boundary-cost/complement.json) |
| Gas 投票、输入索引与动机 | 第 4.1 节用四个同伴数在 Gas 上核对三列争两名，区分改目标后的十列变化与固定目标后的单列误选。唯一胜者构造给出前两行各自的正负票数。Discussion 对固定 Gas 替换报告三个留出批次的模型结果。[逐批评分](../../results/minpi-study/arithmetic-consequence/gas-accuracy.csv) |
| 错误提议需要多少纠正工作 | 在两个错误实例上各运行 32 对固定矩阵计时并记录列号及精算列数。第 4.1 节列出中位耗时；各次调用的记录在复现包内。[纠错实例](../../results/minpi-study/boundary-cost/corrections.json) |
| 行级舍入余量人为放大的 Gisette 边界 | 输入扫描标记全程弃权列并保留其在联盟中的位置，直接使用零分。三个目标各自对同一批 1,016 列的回退全部消失。更新图 1c/d，保留旧计时及可执行的旧内核。[逐目标对照](../../results/minpi-study/zero-bounds/counts.csv) |
| 机器字分支边界是否过强 | Discussion 给出材料数据上的单层整数界。42 个面板中有 21 个满足统一分母判据，另三个材料面板只满足按层求和判据。逐层机器字累加再安全比较异分母分数的路线未经计时。[完整整数界](../../results/minpi-study/arithmetic-consequence/native-bounds.csv) |

在本目录运行 `.\thesis-python.ps1 -m experiments.minpi_study_gas_replay` 可直接重放 Gas 三列并列。独立解包后，在装有本项目依赖且 `src` 已加入 `PYTHONPATH` 的环境运行 `python -m experiments.minpi_study_gas_replay`。两种方式都读取包内 `gas-1129.npz` 的 `train_votes`，按零基列号报告精确分数、双精度分数及两种选集。

上一轮三档预算保留了逐次计时、配对比值和边界比较列数。复跑该历史实现时，在独立解包副本内先按原路径还原[当时的内核、提议验证器和绘图脚本](rowwide-source-snapshot.zip)。用 `.\thesis-python.ps1 -m experiments.minpi_study_budget_comparison --output tmp/replay-budget` 可从训练数据复跑，输出目录须为新目录。直接解包可用 `python -m experiments.minpi_study_budget_comparison --output fresh-run`。

下面各阶段均保留当时的数据、源文件摘要及实验口径。更早的浮点筛选器类名含有 Exact，它精确枚举同伴数条件下的期望公式，但输出的概率是双精度近似。早期 `raw-cost` 计时对应的固定同伴数源码保存在[存档](fixed-tiers-source-snapshot.zip)。复跑时把结果写到新目录。

### 上一轮审稿修订

1. 对相同三元组逐步改变打分法，追踪了所有五次回归换列及两种后端的测试误差。四次来自 Communities and Crime，一次来自 Diabetes。固定同一训练集只重抽见证，Gisette 和 ISOLET 仍有明显换列。[六臂对照、列名和耗时](../../results/minpi-study/matched/findings.md)
2. 独立生成的比较能区分两类机制。分歧发生在可靠比较时，枢轴加权有利；发生在错误比较时，收益反转。在专门构造的跨仪器设置中，min 可以优于各单层。用真实手腕与脚踝传感器复核时，min 与最大层同选，五人平均较差设备的平衡准确率为 0.421，符号票为 0.463。[机制实验](../../results/minpi-study/mechanism/findings.md) · [真实设备转移](../../results/minpi-study/device/findings.md)
3. 在两个新来源预先固定预算与划分。气体漂移上，枢轴加权相对符号票的平均平衡准确率提高 0.022；材料临界温度预测的测试 R² 降低 0.160。主种子上两种来源的最小值与最大层均同选。[预定协议](new-cohorts-protocol.md) · [逐种子配对结果](../../results/minpi-study/new-cohorts/comparisons.csv)
4. 实现了可返回数学上准确支持集的整数证书。4,096 列构造中，两个实际分数相差约 2.20×10⁻²¹，浮点实现选错，整数核查找回唯一最优列。在单个 16,384 列矩阵上，整数证书用时 15.67 ms，逐列整数计算为 108.64 ms。预算扩大至 50% 时，浮点剪枝的耗时可以超过同内核穷举。[真实与构造数据实测](../../results/minpi-study/integer-final/findings.md) · [预算试验协议](frontier-protocol.md)

[原审稿意见](../../paper/minpi/README.md) · [修订与复跑说明](revision.md) · [上一轮独立复审](revision-review.md)

### 初稿的实验与历史结论

1. 第一轮把训练见证上的多数投票写成有限博弈，用三类票的组合计数求指定联盟规模下的精确期望。它消除了给定见证后的联盟抽样误差，训练样本的选择仍然影响最终结果。
2. 第二轮比较了跨层取最小值、跨层平均、单票和单独使用最大层。在 55 个正式划分中，最小值与最大层单独选出的前 10% 在 50 个划分上完全一样。UCI HAR 的 15 个种子与预算组合也全部相同。固定见证上的可重复计算不能推出跨训练样本的特征集稳定性。[消融与重合度](../../results/minpi-study/summary/ablation_overlap.csv)
3. 第三轮利用单层分数提供上界，先判断哪些列可能入选，再精算这些列的跨层最小值。55 个训练划分的结果与完整计算逐列一致，48 个划分只需细算最初的候选列。在 16,384 列的固定投票矩阵上，认证版耗时为同一计算内核全算的 72%；在 64 列上，两者接近。在 Gisette 上，从原始训练矩阵开始，认证版比同内核全算约快 6%。[分项计时](../../results/minpi-study/certificate/timings.csv) · [训练到返回支持集的计时](../../results/minpi-study/certificate/training.csv)
4. 将森林选择器的输入缩短到原宽度的 40%，在 Gisette 和 ISOLET 上改善了相对直接森林的耗时。改用同宽的 F 筛选或随机前筛也能产生这一效果。Spambase 的重复记录审查进一步缩小了准确率主张：按相同特征向量分组后，minPI 相对同宽 F 筛选的 AUC 中位差，在树模型上为 +0.021，在岭模型上为 −0.011。[同宽前筛比较](../../results/minpi-study/chain/findings.md) · [分组敏感性](../../results/minpi-study/summary/spambase_grouped_refit.json)

| 研究步骤 | 入口与保留的证据 |
|---|---|
| 主实验和消融 | [实验协议](../../configs/minpi-study.json)、[复现说明](main-experiment.md)、[每次拟合的指标](../../results/minpi-study/summary/split_metrics.csv)、[逐对比较](../../results/minpi-study/summary/paired_comparisons.csv) |
| 只求前 h 列 | [上界筛除与复现](certificate.md)、[支持集回放](../../results/minpi-study/certificate/replay.csv)、[同矩阵计时](../../results/minpi-study/certificate/timings.csv) |
| 近似误差与速度 | [固定投票矩阵的实验设计](kernel-protocol.md)、[精确与抽样计算](../../experiments/minpi_study_kernel.py)、[完整测量](../../results/minpi-study/kernel/curves.csv) |
| 独立高维数据 | [Arcene、Dexter 官方划分](external-protocol.md)、[条件区间](../../results/minpi-study/external-official/summary.json) |
| 跨受试者与预算 | [UCI HAR 协议](transfer-protocol.md)、[各预算和学习器结果](../../results/minpi-study/subject-transfer/summary.csv) |
| 同一森林前筛 | [按相同前筛宽度检验时间收益的来源](chain-protocol.md) |
| 理论与定位 | [Lean 证明](../../paper/minpi/formal/Pivotal.lean)、[Wolfram Language 枚举](../../paper/minpi/formal/verify.wl)、[论文定位与文献](literature.md) |
| 独立检查 | [上一轮独立复审](revision-review.md)、[上一轮完整测试](../../results/minpi-study/revision-final-tests.log)、[旧稿审查记录](referee.md) |

### 在其他学习器前使用

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from minppi_mofs.support_selector import PivotalSupportSelector

model = make_pipeline(
    PivotalSupportSelector(task="classification", fraction=0.1, random_state=1129,
                           arithmetic="rational"),
    ExtraTreesClassifier(n_estimators=48, random_state=1129, n_jobs=1),
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
selected = model.steps[0][1].get_support(indices=True)
```

输入为二维数值型稠密数组，或列名唯一、训练和预测列顺序一致的 DataFrame。回归使用 `task="regression"`，三元组的组别由参考训练样本的响应分位数确定。交叉验证时应将筛选器放入训练流水线，使每次划分重新拟合。输出按原始列顺序排列。该接口按预算求成员集合，集合内部的重要性排序不属于其输出。

### 复核与重跑

以下命令以本目录树的根 `MastersThesis/history` 为工作目录。Windows 启动器选择现有共享环境，优先导入本目录的源码。主实验按数据集与种子保存结果，接续时核对代码摘要。外部验证和计时曲线接收新的输出路径，可与已归档的运行逐项比较。

```powershell
.\thesis-python.ps1 -m pytest tests/test_exact_minpi.py -q
.\thesis-python.ps1 -m experiments.minpi_study --pilot --datasets sparse_signal gisette
.\thesis-python.ps1 -m experiments.minpi_study
.\thesis-python.ps1 -m experiments.summarize_minpi_study
.\thesis-python.ps1 -m experiments.minpi_study_external --output results/minpi-study/replay-external
.\thesis-python.ps1 -m experiments.minpi_study_kernel --output results/minpi-study/replay-kernel/curves.csv
.\thesis-python.ps1 -m experiments.minpi_study_subjects --check-data
.\thesis-python.ps1 -m experiments.minpi_study_subjects
.\thesis-python.ps1 -m experiments.minpi_study_chain --check-inputs
.\thesis-python.ps1 -m experiments.minpi_study_chain
.\thesis-python.ps1 -m experiments.audit_minpi_study
.\thesis-python.ps1 -m experiments.plot_minpi_study
.\thesis-python.ps1 -m experiments.audit_minpi_study_exact_targets
.\thesis-python.ps1 -m experiments.minpi_study_anchor_design --output tmp/replay-anchor
.\thesis-python.ps1 -m experiments.minpi_study_word_design --output tmp/replay-word
.\thesis-python.ps1 -m experiments.minpi_study_witness_design --output tmp/replay-witness
.\thesis-python.ps1 -m experiments.minpi_study_precision_design --output tmp/replay-precision
.\thesis-python.ps1 -m experiments.review_minpi_study --prepare
.\thesis-python.ps1 -m experiments.review_minpi_study
lean paper/minpi/formal/Pivotal.lean
latexmk -cd -pdf -interaction=nonstopmode paper/minpi/main.tex
```

主试验每个数据集、种子目录的 `record.json` 记录了输入身份、特征顺序、选列、软件版本及运行时间；同目录的 `predictions.npz` 保留真实标签、逐样本预测、连续分数和划分索引。[历史 PIv1 归档](../thesis/archives/piv1/README.md) 记录前一项研究的设置及结果。
