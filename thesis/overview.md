# 硕士论文整体规划

> **范围说明：** 本文档是未来硕士论文的研究草稿，不属于 ICOMP 2026 正式协议或实验结果；其中的时间、算力和方法均未冻结。

## 论文方向

**Bipartite Game-Theoretic Joint Feature Selection and Neural Architecture Search for Tabular Data**

把会议论文的 single-layer MinPPI-MOFS 推广到 bipartite (feature × architecture) factored game，引入 **cross-Shapley** 作为核心理论对象，结合现代 DARTS 进行特征选择与神经架构联合搜索。

## 论文时间表 (10 个月: 2026-08 至 2027-05)

| 月份 | 阶段 | 主要工作 |
|------|------|---------|
| 2026-08 | **Phase A: 理论基础** | 完成 cross-Shapley 公理化, Lemma 1-4 证明, Theorem 1 (uniqueness) 尝试 |
| 2026-09 | Phase A 完成 + Phase B 启动 | 完成 §4.5 ANOVA decomposition, §4.6 spurious interactions 反例。开始 DARTS 实现 |
| 2026-10 | **Phase B: 算法实现** | 选定近似算法路线 (SHAP-IQ / ShapG-bipartite / Monte Carlo)；实现 bipartite MinPPI-MOFS |
| 2026-11 | Phase B 完成 | EL-DARTS 或 DASH 集成，跑通 small-scale 联合搜索 |
| 2026-12 | **Phase C: 主实验** | 在 4 个表格数据集上跑联合 NAS+FS 实验，与会议论文方法对比 |
| 2027-01 | Phase C 完成 | 消融实验，超参敏感性 |
| 2027-02 | **Phase D: 高级分析** | 协同效应 $\Delta_{\text{synergy}}$ 的 cross-Shapley 严格表达，理论分析章节 |
| 2027-03 | Phase D 完成 + 写作启动 | 起草章节 |
| 2027-04 | **Phase E: 写作** | 完整论文 draft |
| 2027-05 | Phase E 完成 + 提交 | 修订并提交 |

## 论文章节结构 (预估)

1. **Introduction** —— 动机、贡献清单
2. **Related Work** —— FS / NAS / 博弈论方法的三线对比
3. **Preliminaries** —— Shapley, NSGA-II, DARTS 基础
4. **Cross-Shapley for Factored Games** —— 论文的核心理论 (见 `cross-shapley.md`)
5. **Bipartite MinPPI-MOFS Algorithm** —— 联合 NAS+FS 算法
6. **Approximation Algorithms** —— ShapG-bipartite / KernelSHAP-IQ-adaptation
7. **Experiments** —— 4 数据集 + 消融 + 与会议方法对比
8. **Discussion** —— spurious interactions 讨论, 局限, future work
9. **Conclusion**

## 数据集

至少 4 个 (与会议论文有 1-2 个重叠以便基础设施复用):
- **IEEE-CIS Fraud Detection** —— 工业延迟敏感
- **Higgs (subset)** —— 规模实验
- **PBMC scRNA-seq** —— p ≫ n 极高维场景
- **Criteo CTR** —— 高基数类别特征 + 量化压力

## NAS 引擎选择

**推荐栈**: PC-DARTS + DARTS- + FairDARTS 的稳定化组合，或直接用 EL-DARTS (2026) / DASH。

**不要用 2019 vanilla DARTS** —— skip-connection collapse 风险太大。

**Fallback (Topic 4-lite)**: 如果 DARTS 联合优化不收敛，切换到 NSGA-II 搜架构超参 (n_layers, hidden_dim, op 选择等)，搜索空间更小、更稳，与会议论文的 NSGA-II 引擎统一。失去 cell-level 自动发现的卖点，但保住"双层博弈"的理论核心。

## 与会议论文的关系

| 维度 | 会议论文 | 硕士论文 |
|------|---------|---------|
| 玩家集 | 单一 (特征 N₁) | 双重 (特征 N₁ × 操作 N₂) |
| 价值函数 | $v: 2^{N_1} \to \{0, 1\}$ | $v: 2^{N_1} \times 2^{N_2} \to \mathbb{R}$ |
| 核心对象 | MinPPI | cross-Shapley $\phi_{i,o}^{\text{cross}}$ |
| 搜索引擎 | NSGA-II (single-layer) | NSGA-II + DARTS (joint) |
| 时间 | 2026-06 至 2026-07 | 2026-08 至 2027-05 |

会议论文的 NSGA-II 引擎、ShapG 集成、5 个表格数据集 baseline 全部可以**直接迁移**到论文 Phase B。

## 算力使用

- 3× H100 可用
- 论文 Phase A (理论) 不消耗 GPU
- Phase B-D (DARTS supernet 训练 + 联合搜索) GPU-bound，3 张 H100 用满
- 实验阶段可以并行：每张 H100 跑一个数据集 × seed × 配置

## 论文核心独立贡献 (区别于会议论文 + advisor 已有工作)

1. **Cross-Shapley 形式化**: factored game 上 Shapley interaction 的合法推广 (基于 Grabisch-Roubens 1999 + Shapley-Owen 2019 框架)
2. **Bipartite MinPPI-MOFS 算法**: 把会议论文的 NSGA-II 引导扩展到 (feature × architecture) 联合搜索
3. **ANOVA-Shapley 等价命题**: cross-Shapley 与 bipartite 2 阶 ANOVA 项的关系 (§4.5)
4. **Spurious interactions 在 NAS+FS 下的实证与修正**: §4.6 的应用层贡献
5. **数值近似算法**: 提供了 cross-Shapley 在 factored game 下的计算工具 (对应 Rabitti-Borgonovo §8 列出的开放问题)

## 写作硬约束 (避免常见错误)

1. 会议论文 (MinPPI-MOFS) 作为先期工作引用，论文在此基础上做 bipartite 推广 + cross-Shapley + NAS 集成
2. ShapG 是 advisor 工作，论文中作为先期方法引用并扩展为 ShapG-bipartite
3. Shapley-Owen Index (Rabitti & Borgonovo 2019) 是 cross-Shapley 的最近相关工作，必须显式定位差异 (factored vs single-set)
4. **cross-Shapley 的理论靠山是 Grabisch-Roubens / Shapley-Owen 的 factored 扩展** —— 不必假借其它不相关的"双层"工作
5. DARTS 用现代变体 (PC-DARTS / DARTS- / EL-DARTS / DASH)，不要用 vanilla 2019 版本

## 详细理论框架

见 `cross-shapley.md`。
