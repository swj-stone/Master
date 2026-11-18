# 从原始 pivotal minPI 改进特征组合

**当前主实验**使用给定同伴后重拟合的岭目标改善量投票。输入完整的稠密数值分类数据，返回原矩阵中从零开始的列号，自动估计保留宽度，也支持给定固定预算。

```python
from research.conditional_dense.shared_method import conditional_support

columns, details = conditional_support(X_train, y_train, seed=1129)
```

该版本以较小、较大两种同伴规模中的较低贡献排序，同时复用一次矩阵分解。[推导、支持的输入与逐来源证据](../conditional_dense/README.md) · [固定宽度及自适应预算的跨模型比较](../four_stage/README.md)

需要测试更积极的删列时，`research.conditional_dense.vote_width.method.vote_support(X_train, y_train, seed=1129)` 将票的有效列数作为预算上限。它保留同一排序，只调整截断位置。实际新宽度为 `len(columns)`，返回元数据内的 `width` 是调整前的宽度。[16 个配置的同场比较与使用说明](../conditional_dense/vote_width/README.md)

分类与回归的逐步选择候选接受 `select_features(X_train, y_train, task=..., random_state=...)`，按空集与已选集两种情境投票。稀疏二元词项另有包含词出现和缺席证据的版本。各入口对应独立的预设实验。[逐步选择](coalition_residual.py) · [文本选择器](../text_peer/README.md)

## 前期的等宽换列实验

起点是[分组近邻 minPI](../../src/minppi_mofs/grouped_minpi.py)。先在训练子样本上挑选代表列，以同类与异类邻居构成投票，再按不同同伴数中的最低边际贡献选出 k 列。[本轮实现](witness_exchange.py)直接沿用原代表池、见证、投票和精确前 k 列。

**一次等宽换列。** 在原支持上，先按删掉单列引起的多数收益损失和添加单列产生的收益，各提议最多 8 列。逐对比较至多 64 个组合。分别计算其余已选列全部在场、随机缺少其中一列时的平均收益变化，取较小值。只执行使该值严格为正的最佳一对，平局按原列索引决定。只有一列的支持只需要零同伴条件。初始宽度由调用者提供，规则最多换一列。8 的选择限制额外计算，当前作为固定开发参数。

在冻结的训练票表上，接受必然严格提高整个组合的多数收益。新组合应使用相同最终模型重拟合，在外层样本测量预测收益与成本。原码中允许训练近邻的票表与最终模型表达不同信息。等宽时只比较原支持与新支持，就能测出换列本身的贡献。[推导和反例](../theory/README.md) · [直接枚举与训练入口测试](test_witness_exchange.py)

在 `history` 下，使用当前环境导入：

```python
from research.algorithms.witness_exchange import exchange_top_from_training

columns, details = exchange_top_from_training(X_train, y_train, count=32, random_state=1129)
X_train_selected = X_train[:, columns]
X_test_selected = X_test[:, columns]
```

输入为稠密数值分类矩阵与训练标签，调用者提供合法的训练划分。沿用原算法的 256 行见证上限，构造分类近邻要求 2–128 类且每类至少 2 个训练样本。返回的列号为从 0 开始的原矩阵列号，按升序排列。每次调用返回初始支持和是否接受交换，存在候选时还返回整数增益分子与公分母。当前配对实验使用固定预算评价组合质量，历史自适应选宽的结果见[成本分析](../../docs/minpi-study/combination-exploration-results.md)。
