# ICOMP 2026 会议论文规格

## 工作标题

"Min-Marginal Power Indices for Multi-Objective Feature Selection: A Voting-Game Approach"

## 会议信息

- ICOMP 2026 (International Conference on Computational Optimization), 卡萨布兰卡，摩洛哥
- 时间: 2026-11-11 至 2026-11-13
- 提交平台: OpenReview
- 摘要截止: 2026-06-21
- 全文截止: 2026-07-21
- 页数: 9 页

## 方法合成

把以下三条线索合成为一个统一框架：

- **WVG 框架** (Weighted Voting Game): 二元 v(S) ∈ {0,1}
- **Min 算子** (来自 MinShap): 用 min over permutations 替代标准 Shapley 的 mean
- **MOFS-SHAP 的 NSGA-II**: 多目标进化搜索 + 重要性引导算子
- **ShapG** (advisor 的 EAAI 2025 工作): 既作为 baseline，也作为 Algorithm B 的 importance signal 来源

## 方法概述

1. **WVG formulation**:
   - v(S) = 𝟙[Acc(model_S) ≥ q]，二元 winning 函数
   - 配额 q = α · Acc_full

2. **Min Power Indices (MinPPI)**:
   - MinSS_i = min_π [v(S_π^i ∪ {i}) - v(S_π^i)]，min Shapley-Shubik
   - MinBz_i = min_{S ⊆ N\{i}} [v(S ∪ {i}) - v(S)]，min Banzhaf
   - 命题: MinSS_i = MinBz_i (两者枚举同一组子集)

3. **MOFS-MinPPI**: 三目标 NSGA-II (|F|, 1-AUC, training time) + MinPPI 引导变异/交叉

4. **ShapG 集成**: 作为 Algorithm B (ShapG-guided adversarial search) 的引导信号，也作为 importance baseline

## 为什么这个合成绕开了 WVG-FS 的加性近似缺陷

WVG-FS 独立使用时假设 Σ w_i ≥ q ≈ Acc(model_S)，这在非线性模型上结构性错误。

MOFS-MinPPI 中：
- 精度由 NSGA-II fitness 真训模型给出，**不用**加性近似
- Power index 只当**搜索算子的引导信号**，不是最终决策
- 即使 MinPPI 估计有偏差，Pareto-optimality 完全不受影响

## 数据集 (5 个)

| 数据集 | 特征数 | 样本数 | 选取理由 |
|--------|--------|--------|---------|
| Spambase | 57 | 4601 | 中等维度，文本分类，标准 baseline |
| Madelon | 500 | 2600 | 高维 + 大量无关特征，专为特征选择测试设计 |
| ISOLET | 617 | 7797 | 高维语音，多分类 |
| Gisette | 5000 | 7000 | 超高维 headline 实验 |
| Credit-Card-Fraud | 30 | 284807 | 工业不平衡数据，**同时是毕业论文延续使用的数据集** |

## Baselines (7 个)

1. **SHAP-Top-k** —— 直接对照
2. **ShapG** —— advisor 的方法，作为 baseline + 组件
3. **MinShap** —— Zheng & Raskutti 2026 预印本，作为后验 baseline
4. **Banzhaf for FS** —— Karavas 2017
5. **Powershap** —— Verhaeghe 2022
6. **MOFS-SHAP** —— mean SHAP 版本，作为消融对照
7. **NSGA-II-uniform** —— 无引导，纯多目标 FS

## 关键消融

- **min vs mean 算子** —— 验证 min 算子的离散三分类优势
- **binary v(S) vs continuous v(S)** —— 验证 voting-game framing
- **引导强度 α (mutation bias parameter β)** —— 验证 MinPPI-guided operators 的实际贡献
- **ShapG 作为 importance signal 与 SHAP 的对比** —— 验证 Phase-1 filter 的便宜替代效果

## 评估指标

- Hypervolume (HV) on the Pareto front
- 在 1% / 3% / 5% AUC loss 阈值下的压缩率
- 训练时间加速比
- Friedman + Nemenyi 检验

## 算力使用

- 主要 CPU-bound (XGBoost fitness)
- 3× H100 用于：
  - 数据集 × seed 并行扫描
  - FT-Transformer + DeepSHAP 作为 importance signal 的对比实验
  - 备用：跑 Faith-Shap 等需要梯度的 baseline

## 实验关键风险与对策

- **MinSS 高方差**: min over permutations of binary v(S) 多数时候是 0（少数差排列把 min 拉零）。**对策**: smart sampling（per-feature worst-opponent search via Algorithm B），不是均匀采样
- **MinShap 是近期预印本**: 引用为 inspiration 不依赖
- **7.5 周内做这么多 baseline**: W1 必须把 baseline 接口统一；ShapG 已有开源代码，Powershap 有 PyPI 包，MinShap 可能需要复现

## 与毕业论文的接口

会议论文 = single-layer MinPPI-MOFS engine
毕业论文 = 推广到 bipartite (feature × architecture) two-layer game + 加 cross-Shapley + 加 NAS

详见 `../thesis/overview.md` 和 `../thesis/cross-shapley.md`。
