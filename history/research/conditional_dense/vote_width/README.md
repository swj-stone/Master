# 以两层投票集中度选择 minPI 宽度

在已有六个稠密分类数据来源的 16 个配置中，新规则有 12 个配置比原自动宽度进一步删列，其中 11 个在两个独立进程中都比原算法更快。新版和原版各有 14/16 个配置实际删列且快于 Full。若还要求相对 Full 的平衡准确率（BA）下降至多两个百分点，原版达到 13/16，新版达到 10/16，少的三个配置全部来自 HAR。本轮属于已知数据上的事后探索。[冻结协议](protocol.md) · [完整配对结果](evidence/cases.csv)

| 数据来源 | 原宽度 → 新宽度 / 输入宽度 | 新版平均 BA | 相对原版 | 相对 Full | 相对同宽全样本 F | 删列并快于 Full | 再要求 Full 损失 ≤2pp |
|---|---|---:|---:|---:|---:|---:|---:|
| ISOLET | 355–360 → 311–355 / 617 | 91.49% | −0.08pp | −0.36pp | +2.38pp | 2/2 | 2/2 |
| HAR | 267–274 → 27–89 / 561 | 87.41% | −7.59pp | −8.49pp | +0.63pp | 3/3 | 0/3 |
| Gas | 51–56 → 51–56 / 128 | 75.03% | 0.00pp | −0.09pp | −8.77pp | 3/3 | 2/3 |
| Multiple Features | 349–352 → 237–273 / 649 | 98.64% | +0.15pp | +0.20pp | +0.20pp | 3/3 | 3/3 |
| Digits | 52–53 → 41 / 64 | 94.22% | +0.02pp | +0.18pp | +0.83pp | 0/2 | 0/2 |
| Spoken Arabic Digit | 154–155 → 126–131 / 208 | 93.82% | +0.09pp | −0.27pp | +1.20pp | 3/3 | 3/3 |

分差为新算法减对照，pp 为百分点。各来源内对所列配置等权。HAR 与 Gas 先按测试受试者、测试批次分别计算 BA，再对这些单位等权。HAR、Gas 和 Arabic 的三个种子各自共用训练/测试划分。双进程分别取七轮实际耗时中位数，每个进程内都快才算加速。[各来源汇总](evidence/sources.csv) · [每进程分阶段毫秒数](evidence/process-medians.csv)

| 对照 | 新版严格更准 | 新版双进程更快 | 同时更准且更快 | 同组同时更准更快且删列快于 Full | 左项再约束 Full 损失 ≤2pp |
|---|---:|---:|---:|---:|---:|
| 原自动宽度 minPI | 4/16 | 13/16 | 3/16 | 3/16 | 3/16 |
| Full | 6/16 | 14/16 | 4/16 | 4/16 | 4/16 |
| 同宽全训练集 ANOVA F | 10/16 | 11/16 | 7/16 | 7/16 | 5/16 |
| 独立半宽全训练集 ANOVA F | 8/16 | 13/16 | 5/16 | 5/16 | 5/16 |
| 同宽、相同至多 768 行 ANOVA F | 12/16 | 6/16 | 4/16 | 3/16 | 3/16 |
| 同宽固定种子随机抽列 | 14/16 | 0/16 | 0/16 | 0/16 | 0/16 |

ISOLET 两折和 Arabic 三种子均实际删列，在原 Full 容差内，同时更准且快于同宽全训练集 F。更便宜的同训练样本预算 F 把这个交集缩为 ISOLET 第 3 折、Arabic 种子 2027 和 3469。在所有配置中，两次总时间都比廉价随机选择短的次数为零。同宽对照免费得知新宽度。独立半宽 F 自行取 ceil(p/2)，输入尽量相同的强对照与可单独部署的固定宽度对照回答不同的问题。[逐配置交集及两种得分标准](evidence/paired.csv) · [汇总计数](evidence/comparator-counts.csv)

Arabic 三种子的新流水线总时间中位数在两个进程中为 19.44–22.30 ms。采用每组对 Full 较保守的时间比，总耗时减少 18.3%–22.3%。同样的集中度规则在 HAR 将原宽度 274 压至 27 的那组，BA 比 Full 下降 15.57 个百分点。Gas 的旧宽度在三组中均已压住新预算，分数原样继承，三个同宽 F 都优于 minPI。Digits 虽然保住分数，切片后重新拟合带来的节省仍抵不过选列开销。投票集中度刻画训练子样本上收益分布，以上结果显示以它单独决定宽度可能产生显著的预测代价。[原始进程中位数及逐组分差](evidence/cases.csv)

## 原理和入口

令 s 为同一训练子样本中两种同伴规模的最小岭重拟合收益，令 k₀ 为旧自动宽度。对非零票计算 D₂=(Σs)²/Σs²，取 k=min(k₀,ceil(D₂))。D₂ 的碰撞概率解释和边界可直接检验。当 m 列平分全部票时 D₂=m；票越集中，有效列数越少。全零票回到 k₀。沿用原 minPI 分数排序，平分时优先入选列号较小者，对保留集按原列次序返回。整个计算只调用原条件 minPI 一次，计时包含这次投票、原宽度计算以及新的重新截断。D₂ 是二阶 Hill 有效数，特征组合预测精度仍需外层数据检验。[实现](method.py) · [Hill, 1973](https://doi.org/10.2307/1934352)

调用 `cols, info = vote_support(X_train, y_train, seed=1129)`。实际新宽度为 `len(cols)`。`info` 保留原条件 minPI 的诊断字典，`info["width"]` 表示原宽度 k₀，`info["rank"]` 表示按原列编号存放的两层最低收益。

## 复现

在 history 下运行。两次计时依次运行，并为每个进程指定新的输出目录。

```powershell
.\thesis-python.ps1 -m pytest research/conditional_dense/vote_width/test_vote_width.py -q
.\thesis-python.ps1 -m research.conditional_dense.vote_width.study tmp/vote-width-1 --pass-id 1
.\thesis-python.ps1 -m research.conditional_dense.vote_width.study tmp/vote-width-2 --pass-id 2
.\thesis-python.ps1 -m research.conditional_dense.vote_width.audit tmp/vote-width-1 tmp/vote-width-2 tmp/vote-width-audit
```

已保存两次进程的原始输出 `results/minpi-study/vote-width-timing1-20261118a` 和 `results/minpi-study/vote-width-timing2-20261118a`。审计独立重算了 16 个配置的选宽、排序、各方法预测分数，交叉比对了两个进程保存的 374 个数组，核实 1,568 次流水线计时和所有轮换顺序。独立组间平方和核对全部 F 的截断位置，最大效应量边界超差为零。[审计程序](audit.py) · [核验记录](evidence/verification.json)
