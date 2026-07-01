# 数据审计 & 代码修正 & 补救方案
**审计日期**: 2026-07-01  
**审计范围**: `results/logs/`(6-27+), `results/tables/`(6-27+), `experiments/`, `src/minppi_mofs/`  
**设计**: 基于对抗性审稿发现的数据/代码问题 → 归因 → 修正 → 优先级排序

---

## 第I部分: 数据质量审计

### I.1 主表 run_main_table (2026-06-28_07-44-10)

| 检查项 | 状态 | 备注 |
|--------|:--:|------|
| 8 method × 7 dataset × 5 seed | ✅ | 280 行, 0 行 FAILED |
| MOFS-MinPPI 回归正确输出 R² | ✅ | crimedata R²@10≈0.93 |
| HV 仅 NSGA-II 方法有值 | ✅ | 非 NSGA-II 方法 HV=NaN |
| AUC@10 表 5 分类集 | ✅ | Friedman χ²=14.13, p=0.049 |
| 8 张 CD 图生成 | ✅ | AUC@10/20 cr_1/3/5 hv igd_plus spacing |
| ShapG ISOLET AUC@10≈0.91 | ⚠️ | 比其余方法低 ~0.06; 原因是 multiclass 用 Accuracy 做 v_α、ShapG 的 Pearson 相关图在 26 类上性能衰退; README 已有文档 |
| crimedata IGD+ 7.21/4.55/8.60 | ⚠️ | 极大于其余数据集（0.1–1.7），回归目标归一化可能有问题；不影响正文结论（正文未报告 IGD+ 数值） |

### I.2 Alpha Sweep (2026-06-28_07-19-46)

| 检查项 | 状态 | 备注 |
|--------|:--:|------|
| 7 数据集, α∈{0.85,0.90,0.95,0.99} | ✅ | 单 seed(42) |
| α≡0.80 已删除 | ✅ | 与 PLAN 一致 |
| Algorithm A(shared-coalition) 统一使用 | ✅ | 成本合理 |
| 📛 **harmful 数量与 binary_vs_continuous 不一致** | ❌ | 见 I.3 |

### I.3 binary_vs_continuous 消融 (2026-06-27_19-51-48, 中断)

| 检查项 | 状态 | 备注 |
|--------|:--:|------|
| 4 dataset 有完整产出 | ✅ | spambase/madelon/isolet/gisette |
| credit_card_fraud 中断 | ⚠️ | 需补跑 |
| 连续 v(S) 所有 7 数据集有数据 | ✅ | |
| 📛 **与 alpha sweep 数据不一致 (均为 seed=42)** | ❌ | **重大发现** |

**不一致根源**: Alpha sweep 用 `estimate_minppi_random_mc(shared_coalitions=True)` (Algorithm A 共享联盟)，binary_vs_continuous 对 binary 集用 `estimate_minppi(..., method="random_mc")` (per-feature MC)。

| 数据集 | α=0.95 alpha_sweep | α=0.95 bvc | 差值 | 归因 |
|--------|:--:|:--:|:--:|------|
| spambase | 0 | 7 | +7 | A→B: 算法差异大 |
| credit_card_fraud | 3 | 15 | +12 | 同上 |
| madelon | 49 | 46 | −3 | 接近 |
| isolet | 14 | 8 | −6 | 多类共享联盟 vs 单类 |
| gisette | 0 | 1 | +1 | 接近 |

**这意味**: 论文 α-sensitivity 表 (Appendix E) 和消融 binary_vs_continuous 的 harmful% 数值来自不同估计算法，不能直接对比。需统一。

### I.4 c1 移除测试 (2026-06-28_15-28-58)

| 检查项 | 状态 | 备注 |
|--------|:--:|------|
| 分类 3 dataset × 5 seed | ✅ | spambase/madelon/isolet |
| 回归 2 dataset × 5 seed | ✅ | crimedata/abalone |
| 📛 **Gisette + Credit-Card 缺失** | ❌ | **对抗审稿 #1.2** |
| MinPPI ΔAUC 始终 ≥ 0 (3 dataset) | ✅ | 核心证据 |
| MinShap ΔAUC = −0.036 (Madelon) | ✅ | 对比强烈 |

### I.5 c2 Madelon Ground Truth (2026-06-28_15-28-58)

| 检查项 | 状态 | 备注 |
|--------|:--:|------|
| 5 seeds, 3 methods | ✅ | |
| madelon MI-100 中 97 probes, 3 real | ⚠️ | **极端不平衡**: 97% 是探针 |
| MinPPI precision=0.987, recall=0.214 | ⚠️ | 高 P 低 R: 仅抓到 21% 的探针 |
| 📛 seed 间 recall 差异大 | ⚠️ | s42: 0.144, s123: 0.464, s456: 0.144 |

### I.6 harmful_vs_dimension (2026-06-29_07-08-39)

| 检查项 | 状态 | 备注 |
|--------|:--:|------|
| 3 dataset × 3 seed × 5 cutoff × 3 α | ✅ | 135 行 |
| Gisette harmful%=0 | ⚠️ | 曲线几乎平线 |
| k=100, α=0.95 madelon harmful%=7.7% | ⚠️ | 远低于 bvc(46%) / alpha_sweep(49%) |

**维度扫描与主数据不一致**: 维度扫描全用 Algorithm A，与消融用的 Algorithm B 不同。但维度扫描目的是**趋势**而非绝对值，因此可接受。

### I.7 ablation 其余维度

| 维度 | 状态 | 备注 |
|------|:--:|------|
| min_vs_mean | ✅ | 7 datasets, 2 guidance |
| beta_sweep | ✅ | 7 datasets, 4 β |
| shapg_vs_shap | ⚠️ | 仅 4 binary datasets, p=1.00(n.s.) |

---

## 第II部分: 代码审查

### II.1 experiments/run_main_table.py

| 行/函数 | 问题 | 严重程度 |
|--------|------|:--:|
| `_stat_test` | ✅ 刚修复 Friedman/Wilcoxon dispatch | — |
| `friedman_tests.csv` 输出 | ✅ 修复后正常 | — |
| `_build_tables` | 使用 `aggfunc="first"` 做 pivot——(method,dataset) 在 mean_df 中已唯一聚合，无影响 | 低 |
| MOFS-MinPPI baseline runner | ✅ 回归用 R², 分类用 AUC, 多类用 Accuracy | — |

### II.2 experiments/run_ablation.py

| 行/函数 | 问题 | 严重程度 |
|--------|------|:--:|
| L143 `alpha_values` | 常量已修正为 {0.85,0.90,0.95,0.99} | ✅ |
| L560 `run_ablation_binary_vs_continuous` | 循环中实际使用 `[0.85, 0.90, 0.95, 0.99]` | ✅ |
| 📛 **per-alpha 算法不一致** | 该函数 alpha 循环内调用 `estimate_minppi(method="random_mc")`，但 alpha sweep 用 `shared_coalitions=True` 模式——两者 harmful 检出率不同 | ❌ 中 |
| `run_ablation_min_vs_mean` | ✅ 正确 | — |
| `run_ablation_shapg_vs_shap` | 仅 binary 数据集参与(跳过 multiclass/regression)，p=1.00(n.s.) at N=4——统计效力不足 | ⚠️ 低 |

### II.3 experiments/eval_harmful_utility.py

| 行/函数 | 问题 | 严重程度 |
|--------|------|:--:|
| Seed bug (all→1024) | ✅ 已修复 | — |
| c1 分类数据集列表 | `C1_CLS_DATASETS = [spambase, madelon, isolet]` —— **缺失 credit_card_fraud, gisette** | ❌ 中 |

### II.4 experiments/harmful_vs_dimension.py

| 行/函数 | 问题 | 严重程度 |
|--------|------|:--:|
| ShapG inside MI loop | ✅ 已修复(移入 cutoff 循环内) | — |
| 全用 Algorithm A | 设计原因(成本)。维度扫描只要趋势，接受 | 低 |
| DIM_DATASETS 中 Gisette 缺失 | madelon/isolet/gisette 三个均在代码中？检查: 代码确实只有 madelon, isolet, gisette。但 Gisette harmful%=0 全程 | 低 |

### II.5 src/minppi_mofs/

| 文件 | 问题 | 严重程度 |
|------|------|:--:|
| `voting_game.py` | ⚠️ 多类(26类)用 `accuracy_score` 而非 AUC(ovr): L165 `val = float(accuracy_score(...))`。这使 ISOLET 上的 v_α 与 AUC 不对应，但论文已在 Table B 中标注"Accuracy" | 低(已文档) |
| `min_ppi.py` | `estimate_minppi_random_mc` with `shared_coalitions=True`: 共享联盟模式每个样本评估 `per_coalition` 个不在联盟中的特征——当 `n < per_coalition*2` 时可能多次评估同一特征 | 低 |
| `mofs_minppi.py` | `compute_hypervolume` 3D 用 MC 采样 5000 点，精度有限但 HV 不用于正文核心论证 | 低 |
| `mofs_minppi_baseline.py` | 回归 R²<0 时 clamp 为 0: `if quality < 0.0: quality = 0.0` — 合理 | 低 |

---

## 第III部分: 关键发现汇总

### 严重 (影响论文可接受度)

| # | 发现 | 来源 | 影响 |
|---|------|------|------|
| **C1** | α 敏感性表(algo A)与消融(algo B)的 harmful 值来自不同算法，不可对比 | I.3 | 附录 E 与正文消融段数值不一致，审稿人可据此攻击 |
| **C2** | 移除测试缺少 Gisette 和 Credit-Card-Fraud | I.4 | 正文声称"never degrades"仅基于 3/5 分类集，选择偏差嫌疑 |
| **C3** | Madelon ground truth: 97/100 是探针，recall=0.214，seed 间不稳 | I.5 | 审稿#1.4: 你只抓到 1/5 已知噪声特征，方法真的有用吗？ |

### 中等 (需要补充证据或澄清)

| # | 发现 | 来源 | 影响 |
|---|------|------|------|
| **M1** | Nemenyi post-hoc 全部不显著，Friedman 显著可能由 ShapG-ISOLET outlier 驱动 | I.1 | 正文应做 dropping-one-dataset 敏感性 |
| **M2** | β sweep 不显著(χ²=0.83,p=0.84)，引导可能无真实效果 | I.2 | §6.4 解释应更诚实 |
| **M3** | ShapG vs SHAP N=4 统计效力太弱 | I.7 | 正文结论应弱化 |
| **M4** | 维度扫描图出现 200，与正文 "29–100" 矛盾 | I.6 | 正文应加一句解释 |

### 低 (可快速修复或可接受)

| # | 发现 | 来源 |
|---|------|------|
| L1 | vote_game.py 多类用 accuracy → ISOLET v_α 语义与 AUC 不一致(已文档) | II.5 |
| L2 | 附录 G 四张消融表无 per-dataset 分解 | I.7 |
| L3 | 回归原文定位太少 | II.1 |

---

## 第IV部分: 补救方案 (按优先级，可执行顺序)

### 优先级 1 — 阻塞提交 (必须在 07-05 前完成)

#### 1.1 统一 harmful% 算法 & 修复 alpha 表

**问题**: α-sensitivity table (Appendix E) 与消融/主表使用的 harmful% 来自不同算法，数值不一致。

**操作**:
- 修改 `experiments/run_alpha_sweep.py`: binary 数据集改用与 `run_ablation.py` 一致的 `estimate_minppi(method="random_mc", K=..., shared_coalitions=False)`。或者反过来，将 paper 中所有 harmful% 报告统一为 Algorithm B（更精确）
- **推荐方案**: 不改代码运行——直接更新 `tab:alpha_sweep` 的数据为 `run_ablation.py` binary_vs_continuous 中 α 扫描的实际数值（已有 4 α × 7 数据集）。Alpha sweep 的推荐结论（α=0.95 是 sweet spot）用 binary_vs_continuous 的数据重新验证
- 在 Appendix E 的 caption 中加注: "computed via Algorithm B (ShapG-guided) for binary datasets, Algorithm A (shared-coalition) for multiclass/regression"

**运行成本**: 零(已有完整 bvc 数据)

#### 1.2 补齐 Credit-Card-Fraud + Gisette 移除测试

**问题**: 对抗审稿 #1.2

**操作**:
- 修改 `experiments/eval_harmful_utility.py` 中 `C1_CLS_DATASETS` 加 `("credit_card_fraud", "Class", "binary")` 和 `("gisette", "label", "binary")`
- 重新运行: `python -u experiments/eval_harmful_utility.py`（预计 extra ~60min per dataset）

**预期结果**: Gisette @ α=0.95 仅 1 harmful，移除测试 trivial（ΔAUC≈0, cr≈1%），需诚实报告。Credit-Card 约 3–15 harmful，ΔAUC 预期接近 0 或轻微正。

#### 1.3 Madelon α–recall 曲线 (解决审稿 #1.4)

**问题**: recall=0.214 意味着你只抓了 1/5 探针。为什么 α=0.95？降低 α 能提高 recall 吗？precision 会崩吗？

**操作**:
- 新建脚本 `experiments/madelon_alpha_recall.py`:
  - 在 5 seeds × {0.80, 0.85, 0.90, 0.95, 0.99} 下计算 MinPPI/Minshap/SHAP 的 P/R/F₁ 对 Madelon 探针 ground truth
  - 输出: `madelon_alpha_recall.csv` + `madelon_alpha_recall.pdf`
- 约 20 分钟运行时间 (shapg_guided per alpha per seed)

### 优先级 2 — 高优先级补充 (提升录用概率)

#### 2.1 Friedman dropping-one-dataset 分析

**问题**: 对抗审稿 #3: "Friedman 显著是否由 ShapG-ISOLET 单点驱动？"

**操作**:
- 新建脚本 `experiments/sensitivity_friedman.py`:
  - 依次删掉 1 个分类集 → 重新算 AUC@10/AUC@20 的 Friedman χ² + p
  - 输出: `friedman_leave_one_out.csv`

**预期**: 删 ISOLET 后 χ² 可能下降但仍有趋势；删 madelon 影响小。

#### 2.2 补齐 per-dataset 消融表到 Appendix G

**操作**:
- 修改 `run_ablation.py`: 消融完成后，每个维度额外输出 per-dataset 的 CSV → 灌入 Appendix G（当前 G 只有聚合行）
- 或者直接从现有 logs 中提取 per-dataset 数据，手动制表放入 `.tex`

#### 2.3 正文维度声明 + Figure M2 200 解释

**操作**: 在 §6.3 Figure M2 段落中加一句: "The x-axis extends to 200 solely to confirm trend stability beyond the 100-feature operational regime"

### 优先级 3 — 加分项 (时间允许则做)

#### 3.1 强化 α=0.95 推荐的统计基础

**操作**: 从 binary_vs_continuous α 扫描数据中提取 pairwise Wilcoxon (α=0.95 vs 其他)，加入 Appendix E 的叙述中（现有: bvc 的 harmful% × α 的 Friedman 已有 p=0.019，但 pairwise 仅 run_alpha_sweep 有）

#### 3.2 β sweep 解释强化

**操作**: 修改 §6.4 相关段: "β>0 never degrades HV" → "HV differences across β are within sampling noise (Friedman χ²=0.83, p=0.84). At the population sizes used (30 individuals × 20 generations), MinPPI guidance does not measurably alter the final Pareto front, consistent with Proposition~2 that it accelerates convergence rather than improving the limit."

#### 3.3 ISOLET ShapG 异常

**操作**: 在 §6 或附录中加一句解释: "ShapG's Pearson-based correlation graph degrades on 26-class ISOLET (accuracy-based utility), explaining its lower top-k AUC relative to other methods on this dataset. This confirms the README observation (§2.1) and does not affect ShapG's use as an Algorithm B oracle."

---

## 执行顺序 (TODO, 按依赖关系)

```text
┌─ 第一天 (最高优先级, ~3h) ──────────────────────────────┐
│ [ ] T1.1 统一 α 表数据                                  │
│     从 binary_vs_continuous 提取 4α×7ds harmful%       │
│     → 更新 tab:alpha_sweep (.tex)                      │
│     → 在 caption 加算法标注                             │
│ [ ] T1.3 Madelon α–recall 曲线                          │
│     新建 experiments/madelon_alpha_recall.py            │
│     运行 → .csv + .pdf → 加入 Appendix F               │
│     → 正文 §6.3 加一段 "α 如何影响 recall/precision"    │
├─ 第二天 (~2h) ─────────────────────────────────────────┤
│ [ ] T1.2 补齐 Gisette + Credit-Card 移除测试            │
│     修改 C1_CLS_DATASETS → 运行 eval_harmful_utility   │
│     → 更新 tab:harmful_removal (5 行 → 7 行)           │
│ [ ] T2.1 Friedman dropping-one-dataset                  │
│     新建 experiments/sensitivity_friedman.py            │
│     → 正文 §6 加一句话: "χ² stable under leave-one-out" │
├─ 第三天 (~1h) ─────────────────────────────────────────┤
│ [ ] T2.2 per-dataset 消融表 → Appendix G                │
│ [ ] T2.3 维度声明补一句话                               │
│ [ ] T3.1–3.3 (可选加分项)                               │
└─────────────────────────────────────────────────────────┘
```

## .tex 需同步修改的位置 (摘要)

| 文件位置 | 修改内容 |
|---------|---------|
| Appendix E `tab:alpha_sweep` | 数据改为 bvc 实际值 + 算法标注 |
| Table `tab:harmful_removal` | 加 gisette + credit_card_fraud 行 |
| Figure M2 相关段落 | 加一句 x 轴 200 的解释 |
| §6.4 β sweep 段 | 强化解释 (T3.2) |
| Appendix F | 加 Madelon α-recall 曲线图及表 |
| Appendix G | 加 per-dataset 分解表 |
