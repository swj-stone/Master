# 论文基线与同宽特征组合比较

**mRMR–MID** 是 Peng、Long 和 Ding 的最大相关、最小冗余特征选择法中，按互信息之差逐列选取的版本。它同时考虑候选列对标签的解释力和它与已选列的冗余，适合作为检验 minPI 能否选好一组列的对照。[作者论文，IEEE TPAMI 27(8), 1226–1238 (2005)](https://home.penglab.com/papersall/docpdf/2005_TPAMI_FeaSel.pdf) · [作者的软件说明及论文目录](https://home.penglab.com/proj/mRMR/)

## 来源、公式与实现范围

原文第 2.2 节公式 (7) 给出逐步选择准则。第一列最大化与分类标签的互信息 `I(Xj; Y)`。已选集合为 `S` 时，下一列最大化 `I(Xj; Y) − sum(I(Xj; Xi) for i in S) / |S|`。第 4.1 节的离散估计通过训练样本列联表得到联合和边际频率。本实现把互信息计为自然对数单位。切换对数底数仅给所有判据乘同一正数。[公式 (7) 及第 4.1 节](https://doi.org/10.1109/TPAMI.2005.159)

输入为**事先具有离散含义的特征和分类标签**，以完整的整数类别编码表示。每列按训练中实际出现的类别重新编号，保留全部观察等级及所有训练行。一次性批量统计候选列的列联表，逐步维护与已选列有关的互信息总和。两个判据相差至多 `32 × eps × max(log(n), 1)` nats 时取原始索引较小者，这是本项目的浮点并列约定。接口返回选取顺序和逐步分数。传递给公共学习器前按原列索引排序。连续输入需要另行指定、验证训练期互信息估计或离散化方案。本条目覆盖固定宽度的离散筛选阶段。论文另外研究的二阶段包装法、分类器和计时不在本实验的复现范围内。[当前实现](mrmr_mid.py)

仓库中早期的 mRMR 路径默认至多抽 4,096 行，以最多八个分位箱编码。其离散 MID 核心在输入完全相同的条件下可以作为另一份实现进行复核。早期实验的分箱与抽样应与相应历史结果一同记录。[历史实现](../../experiments/test/piv2_conventional_filters.py)

## 正确性证据

- 在四类、等概率的手算例中，有两列相同并携带标签的第一位，另一列携带第二位，第四列恒定。逐步胜者分数应为 `ln 2, ln 2, (ln 2)/2, 0`，前两列选择互补的两个比特。测试亦覆盖类别重编码、并列、无效输入和大字母表分支。[可执行检查](test_mrmr_mid.py)
- 对三个真实训练折的每一步，使用独立的 `sklearn.metrics.mutual_info_score` 重算候选列与标签及已选列的互信息。96 次入选的列均距该步最高判据不超过 `2e−12` nats，完整顺序也逐折匹配早期实现的同输入 MID 核心。未知方法名会直接报错，对此亦有专门的回归测试。共 15 项测试通过。[官方互信息定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html) · [测试结果](evidence/digits-20261118b/tests.xml) · [逐折核查代码](test_mrmr_mid.py)

## Digits 的三折结果

复用已归档的 3 个分层划分，每折有 1,198 个训练和 599 个测试样本。完整矩阵包含 1,797 行、64 个像素特征，每个像素是 0–16 的整数等级，保留原始等级适合计数法。各删列方法的宽度均预定为 32，算法种子均为 1129。Random 按这个固定种子从 64 列均匀无放回取 32 列，三折共享同一份随机列集合。各方法共用训练折的标准化和 `RidgeClassifier(alpha=1, solver="cholesky")`。此表只描述一个既有的手写数字数据集。[Digits 官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) · [实验编排](digits_comparison.py)

| 方法 | 列数 | 第 1 / 2 / 3 折平衡准确率 | 三折等权平均 | 各折选列至预测总时间范围 |
|---|---:|---|---:|---:|
| 全特征 Ridge | 64 | 91.11 / 93.44 / 94.64% | 93.06% | 2.27–2.31 ms |
| ANOVA F | 32 | 88.10 / 92.29 / 92.47% | 90.95% | 2.77–2.95 ms |
| 均匀随机 | 32 | 86.92 / 88.46 / 88.63% | 88.00% | 2.08–2.29 ms |
| mRMR–MID | 32 | 88.10 / 91.30 / 92.61% | 90.67% | 8.85–9.63 ms |
| 原分组 minPI | 32 | 89.10 / 91.62 / 92.44% | 91.06% | 7.04–7.65 ms |
| 加入一次组合交换的 minPI | 32 | 89.28 / 91.46 / 92.27% | 91.00% | 7.49–7.96 ms |

六种方法在单线程下各预热一次，每个独立进程每折轮转六轮，每种方法在每个执行位置各出现一次。每次计入重新选列、切片、标准化、岭模型拟合和测试预测。数据读取、评分和一致性核查在计时外。两进程共有 216 次计时。每折表中时间取两个进程各自中位数的较大者，范围给出三折的最小值和最大值。两进程的 69 个数组逐项一致，包含训练/测试索引、真值、所选列、预测、分类决策值及用于归因的排序和交换前列集。[逐折分项及总时长](evidence/digits-20261118b/summary.csv) · [进程一原始计时](evidence/digits-20261118b/pass-1/trials.csv) · [进程二原始计时](evidence/digits-20261118b/pass-2/trials.csv)

同时与原分组 minPI 比较，可以单独检查交换的作用。三个初始列集与旧方法完全一致，每折均实际替换一列。按零起始的列号，三个交换依次为 11→31、56→29、15→60。对应平衡准确率变化为 +0.175、−0.167、−0.172 个百分点。新候选相对 mRMR–MID 有两折更准，在本机三折均更快。相对原分组 minPI 只有第一折更准，三折总时间均更长。相对 Full 的损失有两折在 2 个百分点内，三折的总时间均超过 Full。该次固定宽度实验所见的新 minPI 联合成功数为 **0/3**。速度比较限于此处给出的具体实现、单机和三个划分。[全部支持和预测](evidence/digits-20261118b/pass-1)

## 复跑

在 `history` 目录运行。后两次计时使用新的输出目录，保留原始记录。

```powershell
.\thesis-python.ps1 -m pytest research/comparators/test_mrmr_mid.py -q
.\thesis-python.ps1 -m research.comparators.digits_comparison run --output tmp/mrmr-digits-1 --pass-id 1
.\thesis-python.ps1 -m research.comparators.digits_comparison run --output tmp/mrmr-digits-2 --pass-id 2
.\thesis-python.ps1 -m research.comparators.digits_comparison compare tmp/mrmr-digits-1 tmp/mrmr-digits-2
```

实验记录包含源码、输入和划分的 SHA-256，以及运行使用的 NumPy 和 scikit-learn 版本。[运行协议](evidence/digits-20261118b/pass-1/protocol.json)

初版实验入口在收到拼错的方法名时会落入随机选列分支。修正版只接受列明的六种方法名。核对新旧两版时，每个进程的 69 个数组均逐项一致，包含三折的支持、预测和决策值。上表采用修正入口后的新计时。[分派审计与两个版本的程序摘要](evidence/digits-20261118b/dispatch-audit.json) · [历史运行与冻结程序](../../results/minpi-study/digits-paper-baseline-dispatch-v1-20261118)
