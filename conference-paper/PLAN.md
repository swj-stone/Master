# ICOMP 2026 修订计划（单一工作文件）

> 取代并合并了原 `spec.md` / `method-skeleton.md` / `paper_correctness_review.md` / `revision_plan.md` / `experiment_plan.md`（内容已落地或折叠至此；历史见 git）。
> 论文：`conference-paper/paper/icomp2026_conference.tex`。最后更新：2026-06-24。

---

## 0. 硬约束（CFP 已确认）

- **正文 9 页 strict**；**citations 不计、appendix 不计**（均为免费区）。
- 全文截止 **2026-07-05（AoE）**（摘要 06-21 已过）；提交平台 OpenReview。
- 触发本轮修订：外部审稿 **Weak Reject**（理论过誉 / 结果不占优 / 实验不全 / 无高维 等 9 条）。

---

## 1. 已锁定决策（2026-06-24）

- **A 指标重构**：主表换 Pareto 质量（HV / IGD+ / Spacing / Runtime），AUC@k 降为辅助；诚实定位"单点 competitive，核心价值 = Pareto 前沿 + harmful 三分类 + 跨任务泛化"。
- **B 维度诚实**：撤"高维/5000"宣称，明写 MI 预处理 ≤200，定位"中维 + 大量无关特征"。诚实区间 **29–200**。
- **C 值函数抽象 UTVG**：`v_α(S)=𝟙[U(M_S)≥α·U_full]`，`U` 为任务相关效用（**分类 AUC**、回归验证集 R²）；要求 U≥0、U_full>0；不平衡用 AUC 勿用 accuracy；RMSE 须写技能分形式（≈R²），勿用 −NRMSE。
- **D 回归扩展**：加 2 个真实回归集（crimedata、abalone），**验证集 R²**，主进附录。
- **理论降级**：Lemma1/Thm1/Lemma2 → Property 1/2/3（construction 直接后果，非主贡献）；真 Theorem = 样本复杂度 + ShapG 效率；caveat（"every winning ≠ decisive" + f0 反例）前置为非平凡 insight。
- **数据集数量 5+2 = 7**（刻意非默认）：为 Friedman–Nemenyi/CD 图统计效力（N 太小检不出差异）。

---

## 2. 数据集（5 分类 + 2 回归）

| 数据集 | 任务 | 预处理后维度 | target | 效用 U | 下载/来源 |
|---|---|---|---|---|---|
| Spambase | 二分类 | 56 | `is_spam` | AUC | 已在 `data/processed/` |
| Madelon | 二分类 | 100 | `label` | AUC | 已有；**含已知噪声特征 → harmful ground truth** |
| ISOLET | 26 类 | 100 | `letter` | AUC(ovr) | 已有 |
| Gisette | 二分类 | **200** | `label` | AUC | 已有（论文现写 100 = bug，§6 改 200） |
| Credit-Card | 二分类 | 29 | `Class` | AUC（不平衡，勿 accuracy） | 已有 |
| **crimedata**（Communities & Crime） | **回归** | ~119 | **`nonViolPerPop`**（末列 `state_encoded` 非 target） | 验证集 R² | `https://raw.githubusercontent.com/vectorsss/shapG/master/datasets/crimedata_processing.csv` |
| **abalone** | **回归** | 8 | **`Rings`** | 验证集 R² | `https://raw.githubusercontent.com/Vector-s-Lab/LinearRegression/main/data/abalone_processing.csv` |

> 回归集**不入 git**（`data/` 被 .gitignore 忽略）；按上表链接下载到 `data/raw/`，再按 MI≤200 协议预处理。
> 维度口径 P0：MI 预处理规则 = p≤100 全留 / 100<p≤1000 取 top-100 / p>1000 取 top-200；摘要与 §6 按此对齐到 29–200，撤"5000/高维"。

---

## 3. Baselines（7）

SHAP-Top-k · ShapG · MinShap · Banzhaf-FS · Powershap · MOFS-SHAP · NSGA2-uniform。（回归任务投票值/目标换 R²。）

---

## 4. 指标体系（回应 #5/#6/#7）

- **Pareto 质量（新 headline）**：HV、**IGD/IGD+**、**Spacing**、**Runtime**。参考前沿 = 全方法×全 seed 非支配解之并集（真 PF 未知时标准做法）；目标归一化到 [0,1]。
- **单点排序（辅助）**：top-k AUC@{1,5,10,20,50}。
- **harmful 检测效用（核心增量，答"优势在哪"）**：
  - (c1) **移除测试**（真实数据）：移除 MinPPI 判 −1 的特征 vs 移除 mean/随机判负的同数特征，比 ΔAUC/ΔR² 与 |F| 缩减。
  - (c2) **ground-truth 精度**（Madelon + Friedman1 合成）：precision/recall，并作 MinShap 专项（#8）。
- **统计（#7）**：Friedman + **Nemenyi 后验** + **CD 临界差异图** + **effect size（Kendall's W）**。

---

## 5. 专项实验（逐审稿人）

- #2 harmful 占比 vs 维度（29→200）曲线 → 正文小图（平稳即反驳退化担忧）。
- #3 α 敏感性 {0.80–0.99} → 正文小图 + 全表附录。
- #8 MinShap vs MinPPI（连续 vs 二元 harmful 检测可靠性）。
- #4 真实数据 harmful 恢复（Madelon/crimedata，不再只 toy game）。
- C/D 回归泛化（crimedata + abalone 复现主结论）。

---

## 6. 正文（9 页）vs 免费附录 切分

**正文 §6（~1.5–2 页）**：Table M1（Pareto 主表 HV/IGD+/Spacing/Runtime + CR@3%）· Figure M1（CD 图）· Figure M2（harmful vs 维度小图）· 一段 harmful 效用 + MinShap · 一段消融 + 诚实定位。
**附录（不计页数）**：完整 per-dataset 表 · α 全扫描 · MinShap 专项 + ground-truth · 回归全表 · 消融全表 · Nemenyi 后验矩阵 + effect size · 协议细节。

---

## 7. 实验脚本与运行顺序

| 脚本 | 作用 | 运行 |
|---|---|---|
| `experiments/run_main_table.py` | 7 方法 × 7 数据集 × 5 seeds；top-k AUC、HV、CR；**已打补丁存盘 Pareto 前沿** `<method>/<prefix>_s<seed>_pareto.csv` | `python -u experiments/run_main_table.py` |
| `experiments/pareto_metrics.py` | 指标库：参考前沿 + IGD/IGD+/Spacing（归一化，兼容回归） | （被下条 import） |
| `experiments/compute_pareto_metrics.py` | 读存盘前沿 → 算 IGD/IGD+/Spacing + Runtime → `pareto_metrics*.csv` | `python experiments/compute_pareto_metrics.py results/logs/<run_id>` |
| `experiments/make_cd_diagram.py` | Friedman + Nemenyi + 平均秩 + CD 图(PDF) + Kendall's W | `python experiments/make_cd_diagram.py <metrics.csv> --metric hv --higher`（IGD+ 用 `--metric igd_plus --lower`） |
| `experiments/run_alpha_sweep.py` | α 敏感性（复用） | 既有 |
| `experiments/run_ablation.py` | 4 维消融（复用）+ 待加 MinShap harmful 专项 | 既有 |

> 待新增：回归预处理（crimedata/abalone，MI≤200，验证集 R²）；`eval_harmful_utility.py`（移除测试 + ground-truth precision/recall）；`run_main_table` 加 `task=regression`(R²) 与 ovr-AUC。

---

## 8. 进度

**已落地 `.tex`（编译通过）**：UTVG 改名 + 值函数抽象 U（认领非单调）；理论降级 Property 1/2/3 + 重编号 + 贡献 #1 重写 + caveat 前置；9 证明移 Appendix A；Table 2 + Figure 2 移 Appendix B；§1/§2/§3/§4.5 prose 压缩；em-dash 26→11。**主文现 ~10.5–11 页**。

**TODO**
- [ ] P0：跑 `run_main_table`（含前沿存盘）→ `compute_pareto_metrics` → `make_cd_diagram` → Table M1 + CD 图。
- [ ] P1：MinShap 专项 + α + harmful-vs-维度 + 真实 harmful 恢复 + 移除测试。
- [ ] P1.5：回归集预处理 + 跑 + 全表。
- [ ] 重写 §6 正文（Table M1 + CD 图 + 两段）+ 灌附录。
- [ ] §4 Method surgery + 全文 9 页定稿（移 Table 1/Table 3 等到附录）。
- [ ] 维度口径对齐（摘要/§6 与上方 MI 规则，Gisette 100）。
- [ ] 软化 robustness 措辞；Thm1(a) decisive 死代码处理。
