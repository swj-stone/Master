# 对抗性审稿意见 — MinPPI-MOFS (ICOMP 2026)

- **审稿人身份假设**: 博弈论 / 特征选择交叉领域研究者，收到匿名投稿，仅见正文 9 页 + 附录，未接触代码、数据或实验产出。
- **审查方法**: 逐节审读 + 交叉验证所有声明与论文中实际呈现的证据（表、图、附录）。以下按严重程度降序排列。

---

## 1. 核心贡献的实证支撑不足（严重）

### 1.1 方法在全部聚合指标上均不领先

Table M1 显示 MOFS-MinPPI 在 HV、IGD⁺、CR₃% 的 21 个单元格中仅领先 3 个（Abalone HV、Credit-Card IGD⁺ 和 CR₃%），平均秩在 3 个 NSGA-II 方法中排末位（2.14 vs. MOFS-SHAP 1.71）。AUC@10 排名 6/8。论文称"not the best on any single aggregate ranking metric"（L370）并将其定位为差异化优势——这一论述在逻辑上成立，但前提是**有害检测的独特价值被实证充分证明**。审稿人将重点审查下述 1.2–1.4。

### 1.2 移除测试仅覆盖 3 个数据集，遗漏关键 case

Table `tab:harmful_removal` 仅报告 Spambase、Madelon、ISOLET。正文声称"MinPPI-guided removal never degrades test AUC"（L352），但以下数据集缺失：

- **Gisette**: α=0.95 时 harmful 为 0（附录 Table `tab:alpha_sweep`），移除测试将移除 0 个特征 → ΔAUC=0。结果虽 trivial 但应报告，否则读者不知道是漏了还是选择性汇报。
- **Credit-Card-Fraud**: α=0.95 时 harmful 10.3%，29 维中约 3 个。29 维的移除测试成本极低，为何不做？
- **crimedata / abalone**: 回归版本的移除测试未在正文出现。

**建议**: 补齐全部 7 个数据集的移除测试结果（含 trivial case），或明确解释缺失原因。

### 1.3 提出的"主要增量"从未与最直接的竞争对手正面比较

论文将 MinPPI 的三分类（harmful/neutral/decisive）定位为核心增量。但以下关键问题未回答：

- **如果我只用 mean-SHAP bottom-k 做硬删除（不经过 MinPPI），AUC 会怎样？** 即：用 SHAP 排名末位的特征做删除，与 MinPPI−1 做删除，谁的 ΔAUC 更好？当前的对照组是"same number of features by mean-SHAP rank"（Table `tab:harmful_removal`），但 SHAP 排名未产生+1/0/−1 三分类——这是公平对照吗？
- **如果 α 降低到 0.80，MinPPI 是否能检测更多 harmful 并仍保持 precision？** α→harmful% 只给了单 seed 表（Table `tab:alpha_sweep`），没有给出不同 α 下 harmful 检测的 precision 变化曲线。

### 1.4 Madelon ground truth 低召回未充分解释

MinPPI 对 Madelon 已知噪声特征的召回仅 0.214（97 个探针中只抓到 ~21 个），远低于 MinShap 的 0.668。论文将此归因为"voting-game conservatism"（L354），但这实际上意味着 **MinPPI 遗漏了约 78% 的已知噪声特征**。一个自然的质疑是：

- 既然你的目标是"识别 harmful 特征"，为什么你的方法检出率只有 1/5？
- 如果 α=0.95 的 quota 太严，为什么不降低 α？正文推荐的 α=0.95 看起来是武断的——是否有客观的 α 选择准则？
- α 降低后 precision 是否会崩？附录中的 pairwise Wilcoxon 显示 α=0.95 与 α=0.99 无显著差异（p=0.128），暗示 α=0.95→0.99 之间 harmful% 饱和。那 α=0.95→0.85 呢？

---

## 2. 三分类的"三"在实践中退化为"二"（中等严重）

Properties 1–3 和 Theorem 1 的一半（decisive case）占据了 §4.2 的核心篇幅。但附录 Table `tab:alpha_sweep` 明确报告：**Decisive 特征在所有 7 个数据集、全部 4 个 α 水平上均为零**。这使：

1. **§4.4.2 MinPPI-Guided Crossover 成为死代码**——论文承认这是"placeholder"（L241），但正文 §4.4.2 呈现方式仍像一个 active component。
2. **Theorem 1 的 (a) 方向在实证中永远不触发**——三分类本质上退化为 harmful / non-harmful 二分类。
3. **Table `tab:operator-comparison` 中"Output range {−1, 0, +1}"的 {+1} 在本工作的实证结论中无实例支撑**。

**建议**: 要么在正文中坦率说明 +1 是纯理论边界且当前实验中从未观测到（目前仅 §4.2 末尾提了一句），要么想办法在某个数据集上构造出 +1 的实证 case。当前的论述方式会让审稿人认为理论浮夸。

---

## 3. 统计检验的局限性未充分披露（中等严重）

### 3.1 Nemenyi post-hoc 全部不显著

附录 Table `tab:nemenyi_auc10` 全部为 "ns"——在 k=8, N=5 的条件下 CD=4.83 ranks，没有任何成对比较显著。论文在 L747 坦率承认了这个局限，但正文 §6.2 使用"Friedman confirms significant method differences"（L319）作为**积极的统计证据**，而 CD 图的所有 clique bar 显示全部方法均在同一显著组内。

一个 adversarial 解读是：**Friedman 的显著性是因为 ShapG 在 ISOLET 上的 AUC 极低（~0.91），拉低了整个秩分布的方差；实际上前 6 个方法的排序是随机的。**

### 3.2 缺少 AUC@10/20 的 per-dataset 方差信息

Table `tab:all_auc` 只报告均值，未报告标准差。读者无法评估 Top-k AUC 排名在 seed 间的稳定性（而这直接影响 Friedman 秩的可信度）。

---

## 4. 维度扫描图中的 200 与正文的 29–100 矛盾（轻微，但易被抓）

正文三次声明特征维度范围为 29–100（abstract L65, intro L87, §6 L278）。但 Figure M2 的 x 轴扩展到 200（Gisette 的 MI 截断范围）。正文 §6.3 相关段落也写了"from 22–27% at 20 features to 0–2% at 200"（L356），**明确出现 200**。审稿人会问：你的实验到底是用 100 维还是 200 维？

**建议**: 在正文维度声明处加一句"dimension scan extends to 200 features solely for Figure~M2"，将维度扫描划为**超出主实验范围**的额外分析。

---

## 5. 消融证据的强度不足（中等）

### 5.1 Min vs. mean 消融只给了两行表

Table `tab:abl_min_vs_mean` — 仅一行 MinPPI、一行 Mean-SHAP，每个 aggregated over 7 datasets。per-dataset 的分布（特别是 ISOLET/Madelon 上的差异方向）完全不可见。

### 5.2 β sweep 不显著

β sweep 的 Friedman χ²=0.83, p=0.84 (n.s.)，论文将其解释为"consistent with Proposition~2 that MinPPI affects convergence speed rather than optimality"（L366）。但另一解读是：**MinPPI 引导完全没有效果**——即使 β=5 的强偏置也没改变最终 Pareto 质量。如果引导是无效的，那么 MinPPI 嵌入 NSGA-II 的意义是什么？

### 5.3 ShapG vs. SHAP 的 Wilcoxon p=1.00

N=4（仅 binary 分类集）下的 Wilcoxon 检验 p=1.00，意味着两侧完全无差异。论文仍写了方向性结论（"ShapG-guided detects more harmful on Madelon"），这在统计上不成立。

---

## 6. 方法论层面 issue（轻微–中等）

### 6.1 "Parameter-free" 声称有歧义

正文多次强调 MinPPI 三分类是"parameter-free"（L62 abstract, L79 intro, L370 positioning）。但 α 和 β 是用户必须指定的参数。更准确的说法是：**给定 α 后**，三分类的阈值（harmful / neutral / decisive 的边界）+1/0/−1 是自动的（无需调 threshold）。当前措辞可能让读者误解为整体方法无需参数。

### 6.2 Theorem 2 的"structural condition"无法在真实数据上验证

Theorem 2 假设 harmful witness $S_i^*$ 包含在 ShapG 邻居中，这个条件在真实数据上无法检查（因为不知道 ground-truth harmful witness 是什么）。玩具验证（Appendix J）不能替代真实数据上的检验。算法 B 的效率保证在实践上依然是 heuristic。

### 6.3 Credit-Card-Fraud 的纳入合理性

284K 样本 × 29 特征——在 29 维上做多目标特征选择，NSGA-II 的搜索空间是 $2^{29} \approx 5\times 10^8$，但 Pareto 前沿在这么低的维度下可能非常稀疏。而且对于这个样本量的不平衡数据，简单保留全部 29 维（total ~0.5MB）的训练成本远低于跑一个 NSGA-II 搜索。读者可能质疑这个数据集对特征选择方法评估的代表性。

---

## 7. 表述与呈现 issue（轻微）

### 7.1 附录 G (Ablation Full Tables) 实际上表非常少

4 张表（min vs mean, binary vs continuous, beta, shapg vs shap），每张只有 1–2 行聚合数据。标题"Ablation Full Tables"给读者期望看到 per-dataset breakdown，但实际内容是高度聚合的。

### 7.2 回归实验的定位模糊

论文在 abstract 中强调 UT 可替换为 R²，但 §6 正文几乎没有讨论回归结果（全部推给 Appendix D），抽象中的"seven tabular classification and regression datasets"暗示回归与分类同等权重，但正文讨论极不对称。

### 7.3 部分 caption 可读性已改善（感谢修改），但仍有个别表缺少叙述段落

Appendix G 的四张消融表之间无任何过渡文字，读者需自行理解每张表对应正文的哪段论证。

---

## 总体评估

| 维度 | 评分 | 说明 |
|------|:--:|------|
| Novelty | 3/5 | min+binary 的配对有结构性新颖性，Algorithm B 的 ShapG 借力有新意 |
| Soundness | 2.5/5 | 理论框架自洽但决定性实证偏弱；移除测试覆盖面不足；统计显著性被 Nemenyi 吞噬 |
| Significance | 2.5/5 | harmful 检测的价值主张合理，但由 recall=0.214 和"从未领先"削弱 |
| Presentation | 3/5 | 结构清晰，诚实定位值得肯定，部分表述精度和实证覆盖有提升空间 |

**建议**: Weak Reject → 如果作者能补充 (a) Madelon α 扫描与 recall 曲线，(b) 移除测试在 Credit-Card/Gisette 上的完整结果，(c) 澄清 decisive 死代码地位并缩减对应理论篇幅，(d) 对统计结果的诚实讨论（特别是 Nemenyi 全军 ns 的含义），可考虑提升至 Weak Accept。
