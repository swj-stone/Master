# 会议论文 §4 Method 数学骨架

LaTeX 起稿大纲。每个 Lemma / Theorem 给出 statement + proof sketch；正文写作时把 sketch 展开为完整证明。

**预算**: §4 在 9 页 paper 中占约 2.5 页。

---

## §3 Preliminaries — 假定已在 §3 形式化

记号约定（§3 已定义，§4 直接用）：

- $\mathcal{D} = \{(\mathbf{x}_k, y_k)\}_{k=1}^N$: 表格数据集
- $N_1 = \{f_1, \dots, f_n\}$: 特征集合
- $S \subseteq N_1$: 特征子集
- $\mathcal{M}_S$: 用特征 $S$ 训练得到的基学习器 (XGBoost 默认)
- $\text{Acc}(\mathcal{M}_S; \mathcal{D}_{\text{val}})$: 在验证集上的精度指标 (默认 AUC)
- $\text{Acc}_{\text{full}} = \text{Acc}(\mathcal{M}_{N_1}; \mathcal{D}_{\text{val}})$: 全特征精度

§3 复述的现有量：
- Shapley-Shubik index (Shapley & Shubik 1954): $\phi_i^{SS} = \frac{1}{n!} \sum_\pi [v(S_\pi^i \cup \{i\}) - v(S_\pi^i)]$
- Banzhaf index (Banzhaf 1965): $\phi_i^{Bz} = \frac{1}{2^{n-1}} \sum_{S \subseteq N_1 \setminus \{i\}} [v(S \cup \{i\}) - v(S)]$
- WVG (Owen 1972): 简单博弈，二元 $v: 2^{N_1} \to \{0,1\}$
- MinShap (Zheng & Raskutti 2026): $\phi_i^{MinShap} = \min_\pi [v_{\text{cont}}(\dots)]$，连续 AUC 形式
- NSGA-II (Deb et al. 2002): 多目标进化算法，非支配排序 + 拥挤距离

---

## §4.1 Voting Game Formulation for Feature Selection

**Definition 1 (Accuracy-Threshold Winning Function).** 给定配额 $\alpha \in (0, 1]$，定义二元 winning 函数 $v_\alpha: 2^{N_1} \to \{0, 1\}$：

$$v_\alpha(S) = \mathbb{1}\left[\text{Acc}(\mathcal{M}_S; \mathcal{D}_{\text{val}}) \geq \alpha \cdot \text{Acc}_{\text{full}}\right]$$

满足 $v_\alpha(S) = 1$ 的 $S$ 称为 **winning coalition**。元组 $(N_1, v_\alpha)$ 是一个加权投票博弈。

**Remark 1.** 与原始 WVG 文献 (Owen 1972) 一致：$v_\alpha$ 是 simple game (输出 0 或 1)。但**与 WVG-FS** (Sec X) **不同**，我们不假设 $v_\alpha(S) \approx \mathbb{1}[\sum_{i \in S} w_i \geq q]$ —— 我们的 $v_\alpha$ 直接通过训练 $\mathcal{M}_S$ 评估得到，**没有加性近似**。

**Remark 2 (Non-monotonicity).** 与经典 voting game 不同，我们的 $v_\alpha$ **不保证单调**：增加特征可能让精度下降（噪声 / 冗余 / 过拟合），即 $\exists S \subset T: v_\alpha(S) = 1, v_\alpha(T) = 0$。这是 §4.2 中 MinPPI 三分类的关键基础。

---

## §4.2 Min-Marginal Power Indices (MinPPI)

**Definition 2 (Min Shapley-Shubik Index, MinSS).**

$$\text{MinSS}_i = \min_\pi \left[ v_\alpha(S_\pi^i \cup \{i\}) - v_\alpha(S_\pi^i) \right]$$

其中 $\pi$ 跑遍 $N_1$ 的所有 $n!$ 个排列，$S_\pi^i$ 是排列 $\pi$ 中 $i$ 之前的特征集合。

**Definition 3 (Min Banzhaf Index, MinBz).**

$$\text{MinBz}_i = \min_{S \subseteq N_1 \setminus \{i\}} \left[ v_\alpha(S \cup \{i\}) - v_\alpha(S) \right]$$

**Lemma 1 (Boundedness, Trinary Range).** $\text{MinSS}_i, \text{MinBz}_i \in \{-1, 0, +1\}$。

*Proof.* $v_\alpha \in \{0, 1\}$，故 $v_\alpha(\cdot \cup \{i\}) - v_\alpha(\cdot) \in \{-1, 0, +1\}$。Min 保持该值域。∎

这条 Lemma 看似平凡，但是 §4.2 的关键观察。**它把连续的 mean Shapley-Shubik (range $\subseteq [-1, 1]$，通常无聚类结构) 替换为三分类**。

**Theorem 1 (Three-Way Decisiveness Classification).** 在 voting game $(N_1, v_\alpha)$ 下：

(a) $\text{MinSS}_i = +1 \iff i$ **is decisive**: $\forall S \subseteq N_1\setminus\{i\}$, $v_\alpha(S) = 0$ **AND** $v_\alpha(S \cup \{i\}) = 1$。
即 $i$ 加到**任何** $S$ 上都赢，且**任何**不含 $i$ 的 $S$ 都输。

(b) $\text{MinSS}_i = -1 \iff i$ **is harmful**: $\exists S \subseteq N_1\setminus\{i\}$ 使 $v_\alpha(S) = 1$ 且 $v_\alpha(S \cup \{i\}) = 0$。
即加入 $i$ 可能把赢的联盟变成输的。

(c) $\text{MinSS}_i = 0 \iff i$ **is neutral**: 既非 decisive 也非 harmful。

*Proof sketch.*
- (a) $\Rightarrow$: $\text{MinSS}_i = +1$ ⟹ ∀π, marginal at $\pi^{-1}(i)$ position = +1。由 $v_\alpha \in \{0,1\}$ ⟹ $v_\alpha(S_\pi^i) = 0$ 且 $v_\alpha(S_\pi^i \cup \{i\}) = 1$。$S_\pi^i$ 跑遍 $N_1 \setminus \{i\}$ 的所有子集 ⟹ ∀S, $v_\alpha(S) = 0$ AND $v_\alpha(S \cup \{i\}) = 1$。
- (a) $\Leftarrow$: 反推显然。
- (b), (c): 类似 / 直接由定义。∎

**Important caveat (must include in paper).** "i 在每个 winning coalition 中" (即 ∀S: $v(S)=0$ when $i \notin S$) **只是 (a) 的一半**，不足以推出 $\text{MinSS}_i = +1$。还需要 ∀S: $v(S \cup \{i\}) = 1$。Demo `minppi_demo.py` Example 2 给出反例。

**Corollary 1 (Decisive ⟹ "Dictator-like").** Decisive 的特征**独自决定 winning** —— 它的有无完全决定胜负，与其他特征无关。这是非常强的条件，**实践中很少见**。

**Practical use of Theorem 1.** 即便 +1 罕见，**MinSS 的符号**仍是干净的实用信号：
- $\text{MinSS}_i = -1$: **必删** (in some context, i is harmful)
- $\text{MinSS}_i = +1$: **必留** (i is fully decisive)
- $\text{MinSS}_i = 0$: 需要看 mean Shapley / ShapG 等其它信号

实用上 sign(MinSS) 给出二分类（harmful vs not-harmful），这是 FS 的核心 actionable 信号。

**Lemma 2 (MinSS vs MinBz Coincidence).** $\text{MinSS}_i = \text{MinBz}_i$。

*Proof.* MinSS 的 min 跑遍排列；对每个排列 π，marginal 值仅依赖于 prefix $S_\pi^i$。当 π 跑遍所有排列，prefix 跑遍 $N_1\setminus\{i\}$ 的所有子集。MinBz 的 min 直接跑遍这些子集。同一组数取 min ⟹ 相等。∎

**Remark 4 (计算意义).** Lemma 2 让我们之后只用 MinBz 这一个对象 ($2^{n-1}$ 个子集，比 $n!$ 个排列易处理)。**论文中将 MinSS 与 MinBz 统一记为 MinPPI**。

---

## §4.3 Estimation of MinPPI

精确计算 $\text{MinPPI}_i$ 需要 $2^{n-1}$ 次模型训练 (每个 $S$ 一次)。对 $n = 50$ 即不可行。本节给出三种近似策略。

### §4.3.1 Random Sampling (Baseline)

**Algorithm A (Random MC).** 采样 $K$ 个均匀随机子集 $S^{(1)}, \dots, S^{(K)} \subseteq N_1 \setminus \{i\}$，计算

$$\widehat{\text{MinPPI}}_i^{\text{MC}} = \min_{k=1,\dots,K} \left[ v_\alpha(S^{(k)} \cup \{i\}) - v_\alpha(S^{(k)}) \right]$$

**Lemma 3 (Upward Bias).** $\widehat{\text{MinPPI}}_i^{\text{MC}} \geq \text{MinPPI}_i$ 几乎必然。

*Proof.* Sample min ≥ population min。∎

**Proposition 1 (Concentration for harmful detection).** 设 $p_-^i := \Pr_{S \sim U(2^{N_1\setminus\{i\}})}[v_\alpha(S \cup \{i\}) - v_\alpha(S) = -1]$。如果 $\text{MinPPI}_i = -1$ 且 $p_-^i > 0$，则

$$\Pr\left[\widehat{\text{MinPPI}}_i^{\text{MC}} > -1\right] = (1 - p_-^i)^K$$

故 $K = \lceil \log(\delta) / \log(1 - p_-^i) \rceil$ 个样本足以以概率 ≥ $1-\delta$ 检测到 harmful。

**Remark 5 (问题).** 当 harmful event 稀疏 (small $p_-^i$)，$K$ 增长很快。这是 random MC 的根本局限：它检测不到"特定罕见上下文下才有害"的特征。Demo `minppi_demo.py` Example 4 实证 Proposition 1，`algorithm_b_demo.py` 展示这一缺陷。

### §4.3.2 Adversarial Search with ShapG Guidance (Proposed)

**关键观察**: harmful event 对应"$i$ 与某些已选特征产生冗余/冲突"。这些 vulnerable subsets 可以用 ShapG (Zhao et al. 2025) 的特征相关图来辨识。

**Algorithm B (ShapG-Guided Adversarial Search).**

```
Input: 目标特征 i, ShapG correlation graph G_corr, search budget K_adv
Output: MinPPI estimate

1. C ← ShapG-neighbors(i, G_corr, top-k)         # 与 i 高度相关的特征
2. S_seed ← {top-AUC features 含 C}              # i 最可能"画蛇添足"的 baseline 联盟
3. min_marg ← +1
4. For r = 1 to K_adv:
   a. Perturb S_seed (随机翻转，偏置保留高相关邻居)
   b. Evaluate marg ← v_α(S ∪ {i}) - v_α(S)
   c. min_marg ← min(min_marg, marg)
   d. If min_marg = -1: break (found harmful)
5. Return min_marg
```

**Intuition.** ShapG 找出与 $i$ 相关的特征集 $C$。从含 $C$ 的 winning baseline 联盟出发搜索，是 $i$ 最可能产生 harmful interaction 的上下文。

Demo `algorithm_b_demo.py` 实证：在 $p_-^i = 0.033$ 的稀疏 harmful 场景下，Algorithm B 在 K=1 即达到 100% 检测率（vs Algorithm A 的 ~3%）；σ 噪声下 Algorithm B 仍优于 Algorithm A。

### §4.3.3 Hybrid Scheme

实际使用：
- $n \leq 20$: 精确枚举所有 $2^{n-1}$ 个 $S$
- $20 < n \leq 200$: Algorithm B (ShapG-guided)
- $n > 200$: Algorithm A + Algorithm B 组合，两个的 min

---

## §4.4 MOFS-MinPPI Algorithm

把 MinPPI 当 NSGA-II 搜索的引导信号。

**Multi-objective formulation.** 特征子集 $S$ 由二元掩码 $\mathbf{m} \in \{0,1\}^n$ 表示。三目标：

$$\min_\mathbf{m} \quad \left( \|\mathbf{m}\|_0, \; 1 - \text{AUC}(\mathcal{M}_\mathbf{m}), \; T_{\text{train}}(\mathcal{M}_\mathbf{m}) \right)$$

### §4.4.1 MinPPI-Guided Mutation

**Baseline bit-flip 变异**: 每个 bit 以概率 $p_{\text{base}}$ 翻转。

**MinPPI-guided 变异**: 翻转概率取决于 MinPPI 分类：

$$p_{\text{flip}}(i) = \begin{cases}
p_{\text{base}} \cdot \exp(-\beta) & \text{if } \text{MinPPI}_i = +1 \text{ (decisive, 保护)} \\
p_{\text{base}} \cdot \exp(+\beta) & \text{if } \text{MinPPI}_i = -1 \text{ (harmful, 鼓励变化)} \\
p_{\text{base}} & \text{if } \text{MinPPI}_i = 0 \text{ (neutral)}
\end{cases}$$

并对方向加偏置：
- $m_i = 1, \text{MinPPI}_i = -1$: 优先翻为 0 (移除 harmful)
- $m_i = 0, \text{MinPPI}_i = +1$: 优先翻为 1 (加入 decisive)

参数: $\beta \geq 0$ 控制引导强度。$\beta = 0$ 退化为 baseline。

### §4.4.2 MinPPI-Guided Crossover

定义 "decisive block" 为相邻 decisive 特征序列。Crossover 时**优先保留 decisive block 的整段**：

```
def minppi_crossover(parent_1, parent_2, MinPPI):
    child = []
    for each block of consecutive decisive features:
        # 整段从父代之一继承（不打散）
        child[block] = random_choice([parent_1[block], parent_2[block]])
    for each non-decisive position:
        child[pos] = uniform_choice(parent_1[pos], parent_2[pos])
    return child
```

### §4.4.3 完整算法

**Algorithm: MOFS-MinPPI**

```
Input: 数据集 D, 基学习器 M, 配额 α, 最大代数 G_max, 种群规模 N_pop
Output: Pareto 前沿 P*

1. # MinPPI 估计阶段
   MinPPI ← estimate_MinPPI(D, M, α)        # 用 §4.3 的某个算法

2. # NSGA-II 初始化
   P_0 ← random_population(N_pop)
   评估每个 m ∈ P_0: (|m|, 1-AUC(M_m), T_train(M_m))

3. For g = 1 to G_max:
   a. P_offspring ← []
   b. While |P_offspring| < N_pop:
      i.   p_1, p_2 ← tournament_select(P_g)
      ii.  child ← minppi_crossover(p_1, p_2, MinPPI)         # §4.4.2
      iii. child ← minppi_mutation(child, MinPPI, β)           # §4.4.1
      iv.  评估 child, 加入 P_offspring
   c. P_g+1 ← NSGA-II_select(P_g ∪ P_offspring, N_pop)        # 非支配排序 + 拥挤距离
   d. If converged: break

4. Return P_g* (最终 Pareto 前沿)
```

**复杂度**:
- MinPPI 估计: $O(n \cdot K)$ 次模型训练 (K = sampling budget per feature)
- NSGA-II: $O(G_{\max} \cdot N_{\text{pop}} \cdot T_{\text{model}})$
- 总计: 主要 cost 在 NSGA-II 部分 (MinPPI 是一次性预处理)

---

## §4.5 Discussion: Avoiding the WVG-FS Additivity Flaw

**WVG-FS standalone** (Sec X, baseline) 用加性近似：

$$v_\alpha(S) \approx \mathbb{1}\left[\sum_{i \in S} w_i \geq q\right]$$

其中 $w_i$ 是单特征预测精度。这在非线性模型上**结构性错误**：
- 强协同特征组合 ($v(\{i,j\}) > v(\{i\}) + v(\{j\})$) 被低估
- 强冗余特征组合被高估
- 这个误差是系统性的，不能通过增加采样修正

**MOFS-MinPPI 完全规避此问题**：

1. $v_\alpha$ 在算法中由**真实模型训练**评估 ($\mathcal{M}_S$ 实际训练 + 验证)，不用加性近似
2. MinPPI 仅作为搜索算子的**引导信号**，不参与最终 Pareto 决策
3. NSGA-II 的 fitness 决定最终前沿，fitness 基于 actual AUC + actual training time

**形式化 (planned as Proposition 2 in §5)**: 设 $\widehat{\text{MinPPI}}$ 是任意 (哪怕完全错误的) MinPPI 估计。MOFS-MinPPI 输出的 Pareto 前沿 $P^*$ 满足：

$$P^* \subseteq \{\mathbf{m} : \mathbf{m} \text{ is Pareto-optimal under true } (|\mathbf{m}|, 1-\text{AUC}, T_{\text{train}})\}$$

即 MinPPI 估计的错误**最多影响收敛速度**，**不影响 Pareto-optimality**。

> 这一节直接回应审稿人最可能的攻击点：你们怎么处理 WVG-FS 的加性近似缺陷？ 答：我们根本不用加性近似，只用 MinPPI 当软先验，硬约束由真实模型给出。

---

## §4 Summary

| 子节 | 内容 | 论文页数 |
|------|------|---------|
| §4.1 | Voting game formulation, non-monotonic $v_\alpha$ | 0.3 |
| §4.2 | MinSS / MinBz 定义, Lemma 1 (boundedness), **Theorem 1 (3-way classification)** | 0.8 |
| §4.3 | 三种估计算法 (精确 / random MC / ShapG-guided adversarial) | 0.6 |
| §4.4 | MOFS-MinPPI: NSGA-II + 引导算子 + 算法伪代码 | 0.6 |
| §4.5 | 讨论: 规避 WVG-FS 加性近似缺陷 | 0.2 |
| **合计** | | **2.5 页** |

---

## §5 Theory 章节预留 (不在 §4 范围)

放到 §5 (Theoretical Analysis) 的内容：

- **Theorem 2 (Sampling complexity)**: §4.3.1 Algorithm A 的 sample complexity bound (Hoeffding-type)
- **Theorem 3 (ShapG-guided sample efficiency)**: §4.3.2 Algorithm B 比 Algorithm A 节省的样本数 (在 mild assumption 下，可用 demo 中观测的 σ-degradation 曲线作为 Lipschitz-style proposition 的实证)
- **Proposition 2 (Pareto preservation)**: §4.5 末尾的形式化命题严格陈述

§5 大概 1.5 页。

---

## 待解决的写作问题

会议 paper 写作时必须澄清的几个点：

- [ ] **配额 α 的选择**: 实验中 α ∈ {0.80, 0.85, 0.90, 0.95, 0.99} 做敏感性。论文中主结果固定一个 α (推荐 0.95) + 附录展示 α 扫描
- [ ] **harmful 特征在实际数据中常见吗?** §4.2 Theorem 1 假设 $v_\alpha$ 非单调 (即 -1 marginal 存在)。需要在实验部分用真实数据集统计 |{i: MinPPI_i = -1}| / n 的分布，证明这不是 vacuous 情况
- [ ] **Algorithm B 中 "ShapG-neighbors top-k" 的 k 怎么选?**: 这是引入的超参，需要 ablation
- [ ] **β (引导强度) 的合理范围**: 论文里需要 β ∈ {0, 1, 2, 5} 的 ablation
- [ ] **§4.4.2 的 decisive block 定义需要更精确**: 当前是"相邻 decisive 特征"，但特征顺序是任意的。可能要改为"以 ShapG 图上邻居关系定义 block"

---

## Demo 已经验证的内容

| 命题 | Demo 文件 | 检查项 |
|------|----------|--------|
| Lemma 1 (Boundedness) | `minppi_demo.py` Examples 1-3 | MinSS ∈ {-1, 0, +1} 全程成立 |
| Theorem 1 (3-way) | `minppi_demo.py` Examples 1-2 | Decisive / Harmful / Neutral 分类 |
| Theorem 1 caveat | `minppi_demo.py` Example 2 | "在每个 winning 中 ≠ MinSS=+1" 反例 |
| Proposition 1 | `minppi_demo.py` Example 4 | 理论与实测吻合 |
| Algorithm B advantage | `algorithm_b_demo.py` | 38× sample efficiency @ K=1 |
| Algorithm B robustness | `algorithm_b_demo.py` | σ-degradation 测试 |

这些 demo 的输出可以直接做成论文 Figure 1-3 + Tables 1-2。

---

## 引用

- Banzhaf, J.F. (1965). "Weighted voting doesn't work: a mathematical analysis." *Rutgers Law Review* 19.
- Shapley, L.S. & Shubik, M. (1954). "A method for evaluating the distribution of power in a committee system." *American Political Science Review* 48(3).
- Owen, G. (1972). "Multilinear extensions of games." *Management Science* 18(5).
- Deb, K. et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II." *IEEE TEC* 6(2).
- Zheng, C. & Raskutti, G. (2026). "MinShap: A modified Shapley value approach for feature selection." arXiv:2604.15107.
- Zhao, C., Liu, J., Parilina, E. (2025). "ShapG: new feature importance method based on the Shapley value." *EAAI* 148.
- Zhao, C., Liu, J., Parilina, E. (2025). "Strategic inputs: feature selection from game-theoretic perspective." arXiv:2510.24982.
- Karavas et al. (2017). "Feature importance scores and lossless feature pruning using Banzhaf power indices." arXiv:1711.04992.
- Verhaeghe et al. (2022). "Powershap: a power-full Shapley feature selection method." *ECML PKDD*.
