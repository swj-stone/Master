# Cross-Shapley Framework — 论文理论核心

把单层 Shapley 推广到 **factored game** $v: 2^{N_1} \times 2^{N_2} \to \mathbb{R}$ 上的 cross-Shapley interaction value。这是论文的核心理论对象。

---

## Definition

对特征 $i \in N_1$ 和架构操作 $o \in N_2$：

$$\phi_{i,o}^{\text{cross}} = \sum_{S \subseteq N_1\setminus\{i\}} \sum_{T \subseteq N_2\setminus\{o\}} w_1(|S|, n_1) \cdot w_2(|T|, n_2) \cdot \delta_{i,o}(S, T)$$

其中：
- $w_k(s, n) = s!(n-s-1)!/n!$ (单层 Shapley 权重)
- $\delta_{i,o}(S, T) = v(S\cup\{i\}, T\cup\{o\}) - v(S\cup\{i\}, T) - v(S, T\cup\{o\}) + v(S, T)$
- $v(F, A)$ = 用特征集 $F$ 和架构组件 $A$ 训练的模型的精度 (AUC/F1)

## Interpretation

- $\phi_{i,o} > 0$: 协同 —— $i$ 和 $o$ 互相需要
- $\phi_{i,o} \approx 0$: 独立
- $\phi_{i,o} < 0$: 冗余

---

## Theoretical Foundation

### 基础数学骨架

- **Shapley value** (Shapley 1953)
- **Owen value** (Owen 1972) —— games with a priori unions
- **Shapley Interaction Index** (Grabisch & Roubens 1999) —— 同一玩家集中成对玩家的交互
- **Shapley-Owen Index** (Rabitti & Borgonovo 2019, *SIAM/ASA JUQ* 7(3)) —— Grabisch-Roubens 的现代公理化形式

### Grabisch-Roubens 四公理 (Rabitti-Borgonovo §4 复述)

Shapley-Owen interaction index 由这 4 条公理**唯一刻画**：

| # | 公理 | 内容 |
|---|------|------|
| **A1** | Linearity | $\phi_S^{\nu+w} = \phi_S^\nu + \phi_S^w$ |
| **A2** | Dummy | $i$ 为 dummy ⟹ $\phi_i = \nu(\{i\})$ 且 $\phi_{S\cup\{i\}} = 0$ for $S\neq\emptyset$ |
| **A3** | Symmetry | 对 $N$ 的任意置换 $\pi$: $\phi_S^\nu = \phi_{\pi S}^{\pi\nu}$ |
| **A4** | Recursivity | $\phi_{ij}^\nu = \phi_{[ij]}^{\nu^{[ij]}} - \phi_j^{\nu^{N\setminus i}} - \phi_i^{\nu^{N\setminus j}}$，其中 $[ij]$ 是把 $i,j$ **合并成一个虚构玩家**的 reduced game |

**核心定理 (Grabisch-Roubens 1999)**: 这 4 条公理唯一决定 $\phi_S$ 的形式 (eq 4.1)。

### 与 Shapley-Owen Index 的精确关系

Rabitti & Borgonovo (2019) eq. (4.2) 给出**单一玩家集** $N$ 中 $i, j$ 对的交互值：

$$\phi_{i,j}^{SO} = \sum_{T \subseteq N\setminus\{i,j\}} \frac{(n-|T|-2)!|T|!}{(n-1)!} \cdot \delta_{i,j}(T)$$

| | Shapley-Owen Index | Cross-Shapley (本工作) |
|---|---|---|
| 玩家集 | 单集合 $N$ | 两集合 $N_1$ (特征), $N_2$ (操作) |
| 价值函数 | $v: 2^N \to \mathbb{R}$ | $v: 2^{N_1} \times 2^{N_2} \to \mathbb{R}$ (factored) |
| 求和范围 | $T \subseteq N\setminus\{i,j\}$ (混合联盟) | $S \subseteq N_1\setminus\{i\}$, $T \subseteq N_2\setminus\{o\}$ (**独立**) |
| 权重 | $(n-|T|-2)!|T|!/(n-1)!$ (统一) | $w_1(|S|,n_1) \times w_2(|T|,n_2)$ (**积式**) |

**结构关系**: 若把 $N = N_1 \sqcup N_2$ 并设 $v$ 同时定义在 $2^N$ 上，把 $T$ 分解为 $T = T_1 \sqcup T_2$，则 $\delta_{i,o}(T_1\sqcup T_2)$ 与我们的 $\delta$ 一致，**但权重不同**：Shapley-Owen 用统一 $|T|$ 权重；我们用 $N_1, N_2$ 上独立 Shapley 权重的积。

**为什么 factored 是 NAS+FS 的正确语义**：
- 特征 (输入数据) 和操作 (模型结构) 是**异质实体**
- "特征 $i$ 的边际贡献" 应在特征空间内排序定义
- "操作 $o$ 的边际贡献" 应在架构空间内独立定义
- 积式权重反映这种独立性

**cross-Shapley 是 Shapley-Owen / Grabisch-Roubens 交互指数在 factored game 上的合法推广**。

---

## §4 章节大纲 (论文 ~7 页)

### §4.1 Background (1.5 页)
- Shapley value review
- Shapley-Owen interaction index (eq 4.1 from Rabitti-Borgonovo)
- Grabisch-Roubens 4 公理 (review only — proof 在他们论文)

### §4.2 Factored Games (0.5 页)
- 形式化 $v: 2^{N_1} \times 2^{N_2} \to \mathbb{R}$
- 与 single-set games 的嵌入关系 ($N = N_1 \sqcup N_2$)
- 异质玩家集的语义动机 (NAS+FS specific)
- ANOVA 分解视角作引出

### §4.3 Cross-Shapley Definition & Basic Properties (1 页)
- Definition (我们的公式)
- **Lemma 1 (Linearity)**: $\phi^{u+v}_{i,o} = \phi^u_{i,o} + \phi^v_{i,o}$. *Proof*: 对 $\delta$ 的线性
- **Lemma 2 ($N_k$-dummy)**: $i \in N_1$ 满足 $v(S\cup\{i\}, T) = v(S, T)\ \forall S, T$ ⟹ $\phi_{i,o}^{\text{cross}} = 0\ \forall o$
- **Lemma 3 (Sub-Symmetry)**: 在 $N_1$ 内、$N_2$ 内分别对置换不变（**注意不要求跨层对称**）
- **Lemma 4 (Additive Game Collapse)**: $v(F,A) = v_1(F) + v_2(A)$ ⟹ $\phi_{i,o}^{\text{cross}} = 0\ \forall i, o$

### §4.4 Recursivity in Factored Games (1.5 页) ⭐ **核心理论贡献**

**问题**: Grabisch-Roubens A4 把 $i$ 和 $j$ 合并成虚构玩家 $[ij]$。在 factored 设置下，$i \in N_1$ 与 $o \in N_2$ 合并产生的 $[i,o]$ **属于哪个层？语义不明**。

**两种 reformulation**:

**A4'-strong (Intra-Layer Recursivity, 推荐):**
$$\phi_{ii'}^{\text{cross-1}} = \phi_{[ii']}^{\nu^{[ii']}} - \phi_{i'}^{\nu^{N_1\setminus i}} - \phi_{i}^{\nu^{N_1\setminus i'}} \quad \text{for } i, i' \in N_1$$
同理 $N_2$ 内。**跨层合并不引入新公理。**
- 优点: 语义清楚，与 Grabisch-Roubens 直接对接
- 缺点: 不直接刻画 cross-Shapley，间接通过同层 Shapley-Owen

**A4'-weak (Cross-Layer Hybrid Recursivity):**
定义"hybrid player" $[i,o]$ 在扩展玩家集 $\tilde{N} = N_1 \sqcup N_2 \sqcup \{[i,o]\}$ 中，给出 hybrid marginal。
- 优点: 直接给 cross-Shapley uniqueness
- 缺点: hybrid player 语义需要从头设计

**Theorem 1 (planned)**: Cross-Shapley 是同时满足 A1, A2, A3, A4'-strong 的唯一函数族 (待证)。

**Fallback (如果证不出 Theorem 1)**: 写"我们的 cross-Shapley 满足 A1-A3 + 修正后的 A4'-strong；公理唯一性留作 future work"。即便不证 uniqueness，cross-Shapley 仍然是 well-defined 的对象。

### §4.5 ANOVA Variance Decomposition (0.5 页)

类比 Rabitti-Borgonovo eq (4.10) ($\phi_N = \sigma^2_N$)，在 factored 场景下：

$$v(F, A) = v_\emptyset + \sum_i v_i(F_i) + \sum_o v_o(A_o) + \sum_{i,o} v_{i,o}(F_i, A_o) + R$$

**Proposition (planned)**: 在独立 + Sobol-like 价值函数下，$\phi_{i,o}^{\text{cross}} = \kappa \cdot \mathbb{V}[v_{i,o}(F_i, A_o)]$ 其中 $\kappa$ 是某个组合常数。

给 cross-Shapley 一个 variance-based 语义。

### §4.6 Spurious Interactions under Feature Correlation (1 页) ⭐ **实用警告**

Rabitti-Borgonovo §7.3 给出三变量 Gaussian + correlation 反例:

> "interactions may be carried over by correlation. This effect is a possibility only when the input is correlated, and is absent when the input is not correlated."

**对 NAS+FS 的具体警告**:
- 若 $f_i$ 和 $f_j$ 强相关 (数据上)，而 $f_j$ 与操作 $o$ 真正协同
- **cross-Shapley $\phi_{i,o}$ 将显示 spurious 协同信号** (来自 $f_j$，不来自 $f_i$)
- 审稿人**一定会问**：你们怎么处理？

**计划反例**: 2 个强相关 features + 1 个真协同 op。展示 spurious cross-Shapley 信号的形状。

**初步修正方案**: 用 ShapG correlation graph 先标出强相关特征对 → cross-Shapley 矩阵带一层 "correlation-corrected" 视图 (具体形式 W2 决定)。或者退一步：cross-Shapley 的**符号** (synergy/antagonism) 在相关下仍然有效，但**幅度**需要 caveat。

### §4.7 Numerical Computation (1 页)
- 精确: $O(2^{n_1+n_2})$ per $(i,o)$ pair, $n_1 \cdot n_2$ pairs 总
- 近似路线 (W2 决定), 候选见下文
- **重要事实**: Rabitti-Borgonovo §8 明确说 Shapley-Owen 数值估计"是开放问题"。**我们的近似算法可能同时贡献到这个开放问题** —— 这是论文之后冲击 AISTATS / ICML 应用赛道的潜在 hook

---

## 三个 Hard Challenges

### Challenge 1: A4 Recursivity 是 §4 最难的一块
如果学生能给严格的 A4'-strong + uniqueness theorem，**这一段就是 venue-strong 的理论贡献** (AISTATS / GAMES 水准)。如果证不出 uniqueness，退而求其次：写 "cross-Shapley well-defined, A1-A3 holds, A4' is future work"。这是论文级别的天花板和地板的关键区分。

### Challenge 2: Spurious Interactions
Rabitti-Borgonovo §7.3 把它列为核心警告。审稿人**一定**会问。**必须有答案** (哪怕是 caveat-style 的诚实回答)。

### Challenge 3: 数值估计本身开放
§8 明确说 Shapley-Owen 数值估计未解。**我们的近似算法 = 论文章节 + 潜在独立短文章** (关于 Shapley-Owen 通用近似 / factored 特例)。

---

## 实证 Demo 已验证

`../demos/cross_shapley_demo.py` —— 5 特征 × 3 操作玩具数据：
- "f_2, f_3 × poly_x2 强协同" 被矩阵正确捕获 (+0.064, +0.062)
- 噪声特征 f_4 的 cross-Shapley ≈ 0 ✓
- identity 操作列全 0 ✓ (sanity property)
- 二阶截断残差占 ~17%

**Demo 是 §4 Lemma 1-4 + §4.5 ANOVA 命题的实证插图**。

## 算法用途 (论文 Phase B/C)

1. **DARTS α 初始化先验**: $\alpha_o^{(i,j)}|_{t=0} = \beta \cdot \phi_o^{\text{arch}}$
2. **$\mathbf{m}$ 与 $\alpha$ 的搜索引导**:
   - $m_i$ 更新方向 $\propto \phi_i^{\text{feat}} + \sum_o \phi_{i,o}^{\text{cross}} \cdot \alpha_o$
   - $\alpha_o$ 更新方向 $\propto \phi_o^{\text{arch}} + \sum_i \phi_{i,o}^{\text{cross}} \cdot m_i$
3. **离散化后的事后验证**: 用从头训练的 $v(F,A)$ 估真实 cross-Shapley，对比 supernet 估计偏差

## 计算复杂度问题

精确: $O(2^{n_1+n_2})$ per $(i,o)$ pair × $n_1 \cdot n_2$ 对 = 对 demo ($n_1=5, n_2=3$) 完全可行；对真实 ($n_1=100, n_2=20$) **不可行**。

### 近似算法路线 (W2 必须选定)

| 路线 | 思路 | 优点 | 风险 |
|---|---|---|---|
| **SHAP-IQ / KernelSHAP-IQ 适配** | 直接用 [Muschalik 2024](https://arxiv.org/abs/2405.10852) 的任意阶 Shapley 交互近似，适配到 factored | 现成代码 | 需要 reframe 为 $N = N_1 \sqcup N_2$ |
| **ShapG-bipartite** | 把 ShapG 推广到 bipartite graph | 与 advisor 研究线连贯 | 需自研，工作量大 |
| **Monte Carlo 排列采样** | 在 $N_1, N_2$ 上各自随机排列，Hoeffding bound | 简单，理论清晰 | 收敛率可能不够好 |

W2 之前必须决定一个具体路线。

---

## 关键参考文献

- Owen, G. (1972). "Multilinear extensions of games." *Management Science* 18(5).
- Owen, G. (1977). "Values of games with a priori unions." In *Mathematical Economics and Game Theory*.
- **Grabisch, M. & Roubens, M. (1999). "An axiomatic approach to the concept of interaction among players in cooperative games." *Int. J. Game Theory* 28: 547-565.** ← 4 公理的原始来源
- **Rabitti, G. & Borgonovo, E. (2019). "A Shapley-Owen Index for Interaction Quantification." *SIAM/ASA JUQ* 7(3): 1060–1075.**
- Tsai, C-P. et al. (2023). "Faith-Shap: The Faithful Shapley Interaction Index." *JMLR* 24.
- Muschalik, M. et al. (2024). "KernelSHAP-IQ: Weighted Least-Square Optimization for Shapley Interactions." arXiv:2405.10852.
- Sundararajan, M. et al. (2020). "The Shapley Taylor Interaction Index." *ICML 2020*.
- Song, E., Nelson, B.L., Staum, J. (2016). "Shapley Effects for Global Sensitivity Analysis: Theory and Computation." *SIAM/ASA JUQ* 4 —— 排列采样策略

---

## W1 待办 (论文 Phase A 第 1 周)

**全 thesis 最重要的 1 周 (公理体系定下来，后面 9 个月都顺)**:

- [ ] **形式化 $N_k$-dummy 定义** → 证 Lemma 2 (基础)
- [ ] **形式化 sub-symmetry (intra-layer)** → 证 Lemma 3 (基础)
- [ ] **验证 Lemma 4 (additive game collapse)** —— demo 已经数值确认，给严格证明
- [ ] **决定 A4 reformulation**: A4'-strong (推荐) vs A4'-weak (高级但难)
- [ ] **尝试 Theorem 1 (uniqueness under A1-A3 + A4'-strong)** —— 模仿 Grabisch-Roubens proof 改写。**如果 1 周内证不出，标记为 "future work"，§4.4 改为 well-definedness only**
- [ ] **设计 §4.6 spurious interaction 反例**: 2 strongly correlated features + 1 op，用 cross-Shapley 显示 spurious 信号
- [ ] **数值验证 §4.5 ANOVA-cross-Shapley 等价**: 在独立 + Sobol-like 价值函数下用 toy data 验证 $\phi_{i,o}$ 与 $\mathbb{V}[v_{i,o}]$ 的关系

## W2-W3 待办

- [ ] 精读 Faith-Shap (JMLR 2023): faithfulness 公理是否能推广到 factored
- [ ] 精读 KernelSHAP-IQ (2024): 能否直接适配作为 cross-Shapley 近似
- [ ] 决定 §4.7 用哪条近似路线 (SHAP-IQ / ShapG-bipartite / Monte Carlo)
- [ ] **实现选定近似算法的最小原型**: 把 `../demos/cross_shapley_demo.py` 从 (n₁=5, n₂=3) 精确枚举推到中等规模 (n₁=15, n₂=5)，测量近似与精确的偏差（精确枚举此时仍可作为 ground truth，2^14 × 2^4 = 262144 子集，单机几小时可完成）
- [ ] 在中等规模 demo 上验证 §4.3 的三个性质：Lemma 1 (boundedness 等价物)、Lemma 4 (additive game collapse)、§4.5 ANOVA 等价

## W4 退出标准

§4 outline 全部 7 pages 落到 LaTeX 草稿；Lemma 1-4 完整证明；Theorem 1 至少有 proof sketch 或明确的 fallback 形式；近似算法原型在中等规模 demo 上跑通且与精确解的偏差有量化数据。
