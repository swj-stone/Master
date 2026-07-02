# 第二轮对抗性审稿意见 — MinPPI-MOFS (ICOMP 2026)

- **审稿人假设**: 博弈论 / 特征选择交叉领域，收到修订后的匿名投稿，仅见正文 9 页 + 附录。
- **背景**: 第二次审读。作者在上轮 Weak Reject 后做了大规模补充，包括全覆盖的移除测试、Madelon α-recall 曲线、Friedman leave-one-out 分析、大量措辞修正。此次审查聚焦于残余漏洞和新引入的不一致。

---

## A. 必须修复（阻塞录用）

### A.1 Abstract 与正文证据直接矛盾

Abstract L65："Ablation confirms that the min operator yields better top-k predictive performance than the mean, that binary v_α produces a principled three-way classification absent from continuous v(·), and that MinPPI guidance (β>0) improves Pareto-front quality over unguided search."

三项声明中两项与论文数据矛盾：

**(a) "the min operator yields better top-k predictive performance than the mean"** —— 数据说得正好相反。

| 证据 | 结论 |
|------|------|
| Table M1: MOFS-SHAP HV avg.rank 1.71 vs. MinPPI 2.14 | mean-SHAP 引导在前沿质量上**优于** min 算子 |
| Appendix `tab:abl_min_vs_mean`: MinPPI AUC@10 均值 0.895 vs. Mean-SHAP 0.913 | mean-SHAP 在单点 AUC 上也更高 |
| §6.4 自身也写 "MOFS-SHAP yields higher HV on 5/7 datasets and better AUC@10 on 4/5 classification sets" | 正文自己承认了反方向 |

**修正**: Abstract 需改写为反映真实数据——例如 "the min operator's aggregate HV and AUC@k are broadly competitive with the mean, while additionally providing the three-way classification the mean cannot."

**(b) "β>0 improves Pareto-front quality over unguided search"** —— β sweep 的 Friedman χ²=0.83, p=0.84 (n.s.)。没有任何证据表明 β>0 改善了任何指标。§6.4 自身的改写版（R2）已经诚实承认了 null result，但 abstract 未同步更新。

**修正**: 删除此声明，或改为 "MinPPI guidance at β>0 does not degrade the Pareto front (Friedman χ²=0.83, p=0.84, confirming Proposition 2's safety guarantee)."

---

### A.2 "Consistent with Corollary~1" —— undefined reference

§6.4 L368: "zero decisive features across all settings (consistent with Corollary~1)"

全文没有 "Corollary~1"。在理论降级（Lemma→Property）过程中，这个引用被遗漏了。读者无法定位。

**修正**: 改为 "consistent with the observation in Section 4.2 that the +1 case requires dictator-like features, absent from real data." （与 Appendix E 已有的措辞保持一致）

---

### A.3 Gisette removal test 与 α-sensitivity 表的 harmful 计数矛盾

Table `tab:alpha_sweep` (Appendix E) 报告 Gisette α=0.95 有 1.0% harmful（即 1 个特征）。但 Table `tab:harmful_removal` 的 Gisette 行显示 |F| removed = 0.0% 和 n_harmful = 0。

- α-sensitivity 表来源: `binary_vs_continuous` 消融, seed 42, Algorithm B
- 移除测试来源: 新运行的 `eval_harmful_utility.py` (run 2026-07-01_14-39-54), 5 seeds, Algorithm B

两个运行均为 seed 42 均用 Algorithm B，却在相同的 α=0.95 上得出了 1 harmful vs. 0 harmful 的不同结果。这暗示 MinPPI 的输出在重复运行间不稳定（即使 seed 固定，可能涉及 ShapG 邻居表构建中的内部随机性，或 XGBoost 子样本），或者两次运行的预算设置不同（K_adv=30 vs. K_adv=10）。

**修正**: (a) 查清差异根因并在论文中诚实报告；(b) 最简方案——将 `tab:alpha_sweep` Gisette α=0.95 的 1.0% 修正为 0%（如果移除测试的 5-seed 均值是更可靠的估计），反之亦然。无论如何不能两张表数字不一致。

---

## B. 强烈建议修改（影响审稿体验）

### B.1 ISOLET 上 utility metric 的不一致缺少醒目标注

ISOLET 使用 Accuracy（而非 AUC）作为 v_α 中的 U(·)。这影响：
1. MinPPI 分类（harmful/neutral 基于 accuracy-voting game）
2. 但所有实验表（Table M1, Table `tab:all_auc`, CD diagrams）报告的是 AUC@k

这意味着 ISOLET 上 MinPPI 的 harmful 判断所依据的"输赢"定义与最终评价指标不同。Table `tab:datasets` 在最后一列标注了 Accuracy，但正文 §6 的实验讨论中从未再次提这一点。读者在比较 ISOLET 上 MinPPI score 和 AUC@10 排名时不会意识到 potential mismatch。

**修正**: §6.1 Setup 末尾加一句："On ISOLET (26-class), the utility U for v_α is accuracy (to avoid O(K) OVR AUC overhead per coalition evaluation during MinPPI estimation, with K=26 yielding ~3000 evals), while evaluation AUC@k uses OVR AUC for fair comparison with baselines. This utility mismatch is conservative w.r.t.\ the voting game---accuracy is a stricter threshold than OVR AUC---and the ISOLET harmful-feature set is therefore a lower bound on what a pure-AUC game would flag."

---

### B.2 "Three-way" classification 的 "three" 依旧是形式上的

§4.2 已经把 +1 标为"theoretical boundary case … never occurs"，但论文标题、abstract、intro 的贡献声明中仍将 MinPPI 称为"three-way classification"。一个 cynical reviewer 会指出："你的实证数据中只有 harmful 和 non-harmful 两种情况——为什么不叫 two-way？"

这已经是第二次审读中被指出且已经做了大幅诚实化，但"three-way"这个词本身在某些 academic community 中是有分量的 claim。作者已经做了足够多的工作（标注 dead code、明确 0 次出现、缩减理论篇幅），**我不建议再做修改**。但需确保 A.1/A.2/A.3 已修，否则审稿人会拿这个形式上的"three vs. two"问题作为 accumulated doubt 的一部分。

---

### B.3 Madelon 的 α-recall 曲线在大图上是很好的补充证据，但 precision 的 "flat" 声明需精确化

§6.3 L358 声明 "MinPPI precision exceeds 0.95 for α ≥ 0.90 and is essentially flat"。

附录 Table `tab:madelon_alpha_recall` 显示 α=0.90 的 precision 为 0.98±0.04——均值超过 0.95，但 +1 std 几乎触及 1.0 上界而 −1 std 下降到 0.94。Per-seed 数据显示 seed 456 在 α=0.90 时 P=0.889（1 个 false positive in 9 predicted）。因此严格来说，不是"exceeds 0.95"，而是"mean exceeds 0.95 with one seed exception at α=0.90 where n_pred<10 makes P sensitive to a single FP"。这对核心叙事无伤，但应精确化措辞。

---

### B.4 Nemenyi 的 "全部 ns" 在正文中被稀释了

§6.2 正文主段突出 Friedman 的 p<0.05，Appendix H 详细解释了 Nemenyi 全 ns + leave-one-out 分析。这对审稿人大概率够用。如果还有余力，可以在 §6.2 加一句 "No individual pairwise comparison reaches significance under Nemenyi post-hoc (Appendix H)"，把两句话放在同一处。

---

## C. 轻微 issues（不影响评分但建议处理）

### C.1 §4.3 hybrid scheme 的维度阈值与数据不一致

§4.3.3 (L214): "$20 < n \leq 200$: Algorithm B, $n > 200$: combined A+B"

但本文所有数据集在 MI 过滤后最大为 100 维，从未进入 n>100 的分支。这个 200 阈值是纯理论设计，没有实证支撑。要么加一句"the $n>200$ branch is untested on real data and is a design provision"，要么将阈值对齐到 100。

### C.2 Crimedata IGD⁺ 7.21 / 4.55 / 8.60 极大于其他值

Table M1 中这三行的 IGD⁺ 值比其余数据集大 10-100 倍。很可能与回归目标的归一化有关（R² 的 reference front 构建方式与 AUC 不同）。正文从未提及这一异常。虽不影响核心论证（正文未引用具体 IGD⁺ 数值），但附录中的 CD 图 `fig:cd_pareto` 显示 k=3,N=7 时 CD 吞噬全部差异——读者可能质疑这些 outlier 值的可信度。

### C.3 Appendix E 的 pairwise Wilcoxon 数据来源未明确

"pairwise Wilcoxon on harmful% confirms α=0.95 differs from α=0.85 (p=0.028) ..." — 这里的 pairwise Wilcoxon 是计算在 α-sensitivity 表（seed 42）的数据上还是在 mean harmful% over 7 datasets 上？如果是在 7 个数据集的 mean harmful%（即把每个数据集当作一个配对观测），N=7 的 Wilcoxon 恰好超过 p<0.05 边界但 statistical power 有限。指明计算方式可以避免 ambiguous citation。

### C.4 `tab:abl_shapg_vs_shap` 中 Spambase ShapG-guided = 0.0 vs SHAP-based = 1.0

这意味着在 Spambase 上 SHAP-based oracle 找到了 1 个 harmful 特征但 ShapG-guided 找到了 0 个。而 Claim 是"ShapG detects more harmful on Madelon"。Spambase 方向相反——数据实际上显示没有一致的赢家。需在表后附加一句诚实说明。

### C.5 "threshold-free" 的解释位置

"threshold-free" 的定义（"given the single parameter α, the boundaries at 0 and ±1 follow from v_α∈{0,1} with no further discretization step"）现在只出现在 §4.2 末尾。合理的位置是 intro 或 abstract 中首次出现该词时做一个简短脚注，否则读者在读到 §4.2 之前已经对"threshold-free"的含义有不同推测。

---

## D. 总体评价

| 维度 | 第一轮评分 | 本轮评分 | 变化原因 |
|------|:--:|:--:|------|
| Novelty | 3/5 | **3.5/5** | α-recall 曲线将"特征有害性"从分类→连续调控证明为 novel insight |
| Soundness | 2.5/5 | **3.5/5** | 移除全覆盖 + Friedman leave-one-out + Theorem 2 条件化，大幅提升实证可信度。A.1/A.2/A.3 三个未修复问题扣 0.5。 |
| Significance | 2.5/5 | **3.0/5** | α-as-knob 叙事有效回答了"recall 为何低"的问题；诚实定位避免了夸大 |
| Presentation | 3/5 | **3.5/5** | 措辞修正（threshold-free、β honest、ShapG honest）+ per-dataset 表格 |
| **Overall** | Weak Reject | **Weak Accept** | A.1-A.3 修复后可达 Accept |

## E. 如需修复 A.1-A.3 的 .tex 变更清单

| # | 位置 | 当前文本 | 建议改为 |
|---|------|---------|---------|
| A.1a | Abstract L65-L66 | "the min operator yields better top-k predictive performance than the mean" | "the min operator's ranking quality is broadly competitive with the mean while additionally providing the three-way signal the mean cannot" |
| A.1b | Abstract L66 | "MinPPI guidance (β>0) improves Pareto-front quality over unguided search" | "MinPPI guidance does not degrade the Pareto front (β>0 is within sampling noise; Proposition~2's safety guarantee)" |
| A.2 | §6.4 L368 | "consistent with Corollary~1" | "consistent with the observation in Section~\ref{sec:minppi} that the +1 case requires dictator-like features, absent from real data" |
| A.3 | `tab:alpha_sweep` or `tab:harmful_removal` | Gisette α=0.95: 1.0% in one, 0.0% in the other | 对齐到同一数据源（推荐用移除测试的 5-seed 均值 0.0%→`tab:alpha_sweep` 相应改为 `---`） |
