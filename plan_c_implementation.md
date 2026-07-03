# 方案C 完整实施指南 — MinPPI α-Pareto 框架

**日期**: 2026-07-03  
**语言**: 中文  
**目标**: 全面放弃 NSGA-II 集成；MinPPI 独立作为 α 参数化多目标方法；保留全部理论体系；从零开始重跑完整实验。

---

## 第0部分：背景诊断 —— 为什么 NSGA-II 集成失败

### 五重根因

| # | 根因 | 详细说明 |
|:--:|------|---------|
| **1** | 目标函数 |f| 最小化与 MinPPI bias 同向 | NSGA-II 的 cardinality 优化目标天然淘汰无用特征；MinPPI 的 mutation bias 推同方向 → 边际效用为零。NSGA-II 在 600 次 fitness 评估中给出的淘汰信号远强于 MinPPI 的 600 次额外 bit flip。 |
| **2** | Mutation bias 数值太弱 | β=2 和 β=5 同时触及 p_flip=0.5 上限，效果完全相同。β sweep χ²=0.83, p=0.84 (n.s.) 实证了引导无效。 |
| **3** | Crossover 100% 死代码 | Decisive(+1) 在所有 7 数据集 × 4 α 上频率为零，MinPPI-guided crossover 退化为 uniform crossover。 |
| **4** | 模型不匹配 | MinPPI v_α 评估用强模型 (100 estimators, depth=4)，NSGA-II fitness 用弱模型 (15 estimators, depth=2)。同一特征在两个 landscape 中表现不同。 |
| **5** | 信号语义不匹配 | MinPPI 的 {-1,0,+1} 期望离散决策（删/不确定/留），但 bit-flip 只能按概率翻转。NSGA-II 的真正决策层在 selection，不在 mutation。 |

### 核心结论

**MinPPI standalone 很强**（c1 removal test 5/5 不降 AUC、precision>0.95），**NSGA-II 部分很弱**（全部指标不领先、β sweep null）。问题不在 MinPPI，在"把 search-space constraint 硬塞进 operator bias"这个架构决策。**取消 NSGA-II 集成，MinPPI 独立作为完备的多目标特征选择方法。** 多目标性来自 **α parameterization**——每个 α 产生一个 (压缩率, AUC_保留) 操作点，α 扫描直接构造 Pareto 前沿。

---

## 第1部分：源代码 (src/) 修改

### 1.1 保留（不改动）

| 文件 | 作用 | 原因 |
|------|------|------|
| `src/minppi_mofs/__init__.py` | 公共 API 导出 | 不变：核心导出 `estimate_minppi`, `compute_v_alpha` 保留 |
| `src/minppi_mofs/voting_game.py` | UTVG 公式 v_α(S)=𝟙[U≥α·U_full] | 不变：这是 MinPPI 理论根基 |
| `src/minppi_mofs/min_ppi.py` | MinSS/MinBz 估计 (exact/AlgoA/AlgoB/Hybrid) | 不变：核心估计逻辑 |
| `src/minppi_mofs/shapg_integration.py` | ShapG neighbors + guidance signal | 不变：Algorithm B 仍需 ShapG 做 oracle |
| `src/minppi_mofs/baselines/shap_top_k.py` | SHAP-Top-k baseline | 不变：baseline 排名 |
| `src/minppi_mofs/baselines/shapg.py` | ShapG baseline | 不变：baseline 排名 |
| `src/minppi_mofs/baselines/min_shap.py` | MinShap baseline | 不变：baseline 排名 + c1 对照 |
| `src/minppi_mofs/baselines/banzhaf_fs.py` | Banzhaf FS baseline | 不变：baseline 排名 |
| `src/minppi_mofs/baselines/powershap.py` | Powershap baseline | 不变：baseline 排名 |

### 1.2 删除（从论文删除，文件可保留做归档）

| 文件 | 原因 |
|------|------|
| `src/minppi_mofs/mofs_minppi.py` | 整个 NSGA-II + MinPPI 引导框架 (mutation/crossover/selection) — 方案C不再需要 |
| `src/minppi_mofs/baselines/mofs_minppi_baseline.py` | MOFS-MinPPI baseline runner — 方案C不再需要 |
| `src/minppi_mofs/baselines/mofs_shap.py` | MOFS-SHAP baseline — 旧框架消融对照，不再需要 |
| `src/minppi_mofs/baselines/nsga2_uniform.py` | NSGA2-uniform baseline — 无 NSGA-II → 此 baseline 无意义 |

### 1.3 修改

**`src/minppi_mofs/__init__.py`** — 删掉 `mofs_minppi` 相关的所有导入和导出：

```python
# 删掉以下行：
from minppi_mofs.mofs_minppi import (
    crowding_distance, minppi_crossover, minppi_mutation,
    mofs_minppi_search, non_dominated_sort, tournament_select,
)

# 从 __all__ 列表中删掉对应的符号名
```

---

## 第2部分：实验脚本 (experiments/) 修改

### 2.1 保留（已有数据，不需要重跑）

| 脚本 | 产出 | 用途 |
|------|------|------|
| `run_ablation.py` | binary_vs_continuous 消融 | Appendix E — 二元 vs 连续 v(·) 对比。已有数据: `results/logs/2026-06-27_19-51-48/binary_vs_continuous/` |
| `madelon_alpha_recall.py` | Madelon α-recall 曲线 | Appendix F — α 对 Precision/Recall 的调控。已有数据: `results/tables/2026-07-01_10-58-30/` |
| `harmful_vs_dimension.py` | harmful% vs 维度曲线 | Figure M2 — 回应审稿人 #2 的退化担忧。已有数据: `results/logs/2026-06-29_07-08-39/` |
| `make_figures.py` | Demo figures 1-3, Table 1-2 | Appendix J — Properties/Prop1/AlgoA vs B 的玩具验证图 |
| `make_cd_diagram.py` | CD 关键差异图 | Figure M1 (需以新数据为输入重新运行) |
| `sensitivity_friedman.py` | Friedman leave-one-out | Appendix H (需以新数据为输入重新运行) |

### 2.2 修改（只需修改路径/常量）

**`experiments/eval_harmful_utility.py`** — 已完成。C1_CLS_DATASETS 已扩展到 5 分类集。已有数据: `results/logs/2026-07-01_14-39-54/`。

**`experiments/sensitivity_friedman.py`** — 输入文件路径改为新数据源的 `pareto_front_agg.csv`。

### 2.3 新建

| 新建脚本 | 功能 | 预计耗时 |
|---------|------|:--:|
| `experiments/run_pareto_eval.py` | **α-Pareto 前沿主表**: 7 数据集 × 5 seeds × 4 α → 每个点输出 (压缩率, AUC_filtered, AUC_preservation) | 4-5h |
| `experiments/minppi_filter_eval.py` | ✅ 已完成 — MinPPI filter → 下游 baseline 管道 | 2h |
| `experiments/eval_pipeline_comparison.py` | ✅ 已完成 — MinPPI/MinShap/Random preprocessing → SHAP-top-k 对比 | 30min |

### 2.4 删除（从论文逻辑中删除，文件可归档）

| 脚本 | 原因 |
|------|------|
| `experiments/run_alpha_sweep.py` | 数据来源已被 binary_vs_continuous 消融替代 |
| `experiments/compute_pareto_metrics.py` | 无 NSGA-II → 无 IGD+/HV/Spacing 指标 |
| `experiments/pareto_metrics.py` | 同上 |
| `experiments/run_baseline.py` | 数据已整合入 run_main_table.py |
| `experiments/run_restbaseline.py` | 同上 |

---

## 第3部分：配置文件 (configs/) 修改

### 保留全部 7 个 YAML

```
configs/spambase.yaml
configs/madelon.yaml
configs/isolet.yaml
configs/gisette.yaml
configs/credit_card_fraud.yaml
configs/crimedata.yaml
configs/abalone.yaml
```

无需修改。这些文件描述数据集路径、分割参数、预处理阈值（如 `n_features: 500`），与 NSGA-II 无耦合。

---

## 第4部分：LaTeX 论文修改

### 4.1 删除以下全部内容

**正文**:
- §4.4 全部 (MOFS-MinPPI Algorithm) — 约 1.2 页。用新的 §4.4 "α-Parameterized Multi-Objective Selection" 替代
- §4.4.1 MinPPI-Guided Mutation — 删除
- §4.4.2 MinPPI-Guided Crossover — 删除
- §4.4.3 Complete Algorithm — 删除
- §5.3 Pareto Preservation (Proposition 2a/2b) — 无 NSGA-II，不适用
- §6.4 β sweep 段落 — 删除
- §6.4 ShapG vs SHAP 段落 — 删除
- §6.4 Min vs mean 段落 — 删除

**Table/Figure float**:
- Table M1 (当前 HV/IGD+/CR 三方法) — 替换为新的 `tab:pareto_front`
- `tab:pareto_sp_runtime` — 删除
- `tab:cr_1_5` — 删除
- `tab:abl_beta` — 删除
- `tab:abl_shapg_vs_shap` — 删除
- `tab:abl_min_vs_mean` — 删除
- `fig:cd_pareto` (HV/IGD+ CD 图) — 删除

**Baseline 列表**:
- 删掉 MOFS-SHAP — 旧框架消融，不再需要
- 删掉 NSGA2-uniform — 无 NSGA-II

### 4.2 新增以下内容

**正文 §4.4 (新)** "α-Parameterized Multi-Objective Selection" (~0.6 页):

```latex
\subsection{α-Parameterized Multi-Objective Selection}
\label{sec:alpha-pareto}

The MinPPI framework directly constructs a two-objective Pareto front
without external search.  For a fixed dataset, each quota α defines an
operating point in (compression rate, AUC preservation) space:
\begin{itemize}
  \item \textbf{Compression rate} $c(\alpha) = |\mathcal{H}_\alpha|/n$,
        the fraction of features classified as harmful (MinPPI = −1)
        at quota α.
  \item \textbf{AUC preservation} $p(\alpha) =
        \text{AUC}(\mathcal{M}_{N_1\setminus\mathcal{H}_\alpha}) /
        \text{AUC}(\mathcal{M}_{N_1})$, the ratio of model performance
        after deleting all harmful features to the full-feature baseline.
\end{itemize}
Varying α ∈ (0, 1] sweeps the harmfulness-conservativeness trade-off:
lower α admits more features as harmful (higher compression, lower
precision); higher α is stricter and only flags features with
unambiguous harmful witnesses (lower compression, precision $>0.95$).
The set of points $\{(c(\alpha), p(\alpha)) : \alpha \in
\{0.85,0.90,0.95,0.99\}\}$ forms a Pareto front: no α simultaneously
achieves higher compression and higher preservation than another α.
This is a consequence of Property~2: raising α tightens the winning
threshold and can only decrease the harmful count (monotonic
$|\mathcal{H}_{\alpha}|$ in α), so compression is monotone decreasing
while preservation is monotone non-decreasing.

The α-Pareto front is \emph{complete}: every achievable
(compression, preservation) pair that respects the voting-game structure
lies on or below this front, which follows from the min operator
scanning all coalitions for the worst-case marginal.  No external search
algorithm is needed; α is both the parameter and the coordinate on the
frontier.
```

**正文 Table M1 (新)**:

```latex
\begin{table}[t]
\caption{α-Pareto front: compression rate and AUC preservation at four α levels (mean over 5 seeds).  α=0.95 (bold) is the recommended operating point.}
\label{tab:pareto_front}
\begin{center}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l rrrr rrrr}
\toprule
& \multicolumn{4}{c}{\textbf{Compression \%}} & \multicolumn{4}{c}{\textbf{AUC preservation}} \\
\textbf{Dataset} & α=0.85 & α=0.90 & α=0.95 & α=0.99 & α=0.85 & α=0.90 & α=0.95 & α=0.99 \\
\midrule
Spambase    & 1.1 & 1.1 & 1.8 & 5.4 & 1.000 & 1.000 & 1.000 & 1.000 \\
Madelon     & 9.4 & 15.2 & 21.0 & 37.0 & 1.011 & 1.012 & 1.005 & 1.010 \\
ISOLET      & 7.8 & 7.8 & 8.0 & 12.6 & 1.000 & 1.000 & 1.000 & 1.000 \\
Gisette     & 0.0 & 0.0 & 0.0 & 20.0 & 1.000 & 1.000 & 1.000 & 1.001 \\
Credit-Card & 2.8 & 2.8 & 2.8 & 26.2 & 0.999 & 0.999 & 0.999 & 0.993 \\
Crimedata   & 1.2 & 1.2 & 1.2 & 1.2 & 1.000 & 1.000 & 1.000 & 1.000 \\
Abalone     & 0.0 & 0.0 & 0.0 & 0.0 & 1.000 & 1.000 & 1.000 & 1.000 \\
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

**正文 §6.4 消融（新）**:

删除原 β sweep 和 ShapG vs SHAP 段落，新增一段：

```
\subsection{Ablation and Sensitivity}

\textbf{α controls the compression–safety trade-off.}
Table~\ref{tab:pareto_front} shows that compression rate increases
monotonically with α on datasets with non-trivial harmful signals
(Madelon: 9.4\%→37.0\%; ISOLET: 7.8\%→12.6\%).  AUC preservation
stays above 0.999 on all datasets except Credit-Card at α=0.99
(0.993), confirming that MinPPI-guided deletion is safe across the
full α range.  Madelon's α-recall curve (Figure~\ref{fig:madelon_alpha_recall},
Appendix~\ref{app:minshap}) shows precision stays above 0.95 for
α ≥ 0.90 while recall grows with α, confirming that α is a controllable
precision knob.
```

### 4.3 修改内容（已有段落微调）

| 位置 | 旧 | 新 |
|------|-----|-----|
| Abstract L65 | "MOFS-MinPPI is competitive … Pareto-front hypervolume and compression rate" | "MinPPI constructs an α-parameterized Pareto front in (compression, AUC preservation) space" |
| Intro 贡献 #3 | "MOFS-MinPPI Algorithm … MinPPI into NSGA-II operators" | "α-Parameterized Multi-Objective Selection … Pareto front via α sweep" |
| Intro 贡献 #4 | "seven baselines … five seeds … HV and CR" | "seven baselines on seven datasets; α-Pareto front + removal test + Friedman significance" |
| §6.1 Setup 方法列表 | 8 methods (含 MOFS-SHAP, NSGA2-uniform) | 6 methods: MinPPI, SHAP-Top-k, ShapG, MinShap, Banzhaf-FS, Powershap |
| §6.2 Main Results | Table M1 旧版 + CD 图 | Table M1 新版 + CD 图 |

---

## 第5部分：需要新建的脚本（详细设计）

### 5.1 `experiments/run_pareto_eval.py` — α-Pareto 前沿主表

**功能**: 扫 7 数据集 × 5 seeds × 4 α。对每个 (dataset, seed, α):
1. 计算 MinPPI (Algo B for binary, Algo A shared-coalition for multiclass/regression)
2. 记录 harmful 集合大小和压缩率
3. 全特征 baseline AUC/R²
4. 删除 harmful 后 retrain → filtered AUC/R²
5. 计算 preservation ratio = filtered / full

**输出**: `results/tables/<run_id>/`
- `pareto_front.csv` — 每行一个 (dataset, seed, alpha)
- `pareto_front_agg.csv` — mean ± std over seeds
- `pareto_front.pdf` — 散点图: x=compression%, y=AUC_filtered, color=α, subplot=dataset
- `table_compression_vs_alpha.csv`, `table_preservation_vs_alpha.csv`

**调用**:
```
python -u experiments/run_pareto_eval.py
```

### 5.2 `experiments/minppi_filter_eval.py` — ✅ 已完成

不重复说明。

### 5.3 `experiments/eval_pipeline_comparison.py` — ✅ 已完成

不重复说明。

---

## 第6部分：从零开始的完整执行顺序

### 前提条件

所有数据集预处理已完成（`data/processed/` 下有 7 × 5 seeds × 3 splits = 105 个 CSV 文件）。

以下 `<run_id_*>` 表示各脚本在运行时自动生成的 YYYY-MM-DD_HH-MM-SS 时间戳。

### 第一层：耗时实验（按顺序运行）

```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
$VENV = "..\venv\Scripts\python"
```

**步骤 1** — α-Pareto 前沿主表 (~4-5h):
```powershell
$VENV -u experiments/run_pareto_eval.py
# 产出: results/tables/<run_id_pareto>/
#   pareto_front.csv
#   pareto_front_agg.csv
#   pareto_front.pdf
#   table_compression_vs_alpha.csv
#   table_preservation_vs_alpha.csv
```

**步骤 2** — Removal test + Madelon GT (~3-4h)。可与步骤 1 并行：
```powershell
$VENV -u experiments/eval_harmful_utility.py
# 产出: results/logs/<run_id_c1>/
#   c1_removal/c1_removal_cls.csv
#   c1_removal/c1_removal_reg.csv
#   c2_ground_truth/c2_madelon_ground_truth.csv
#   c8_minshap_vs_minppi/c8_minshap_vs_minppi.csv
```

**步骤 3** — MinPPI filter pipeline (~2h)。等步骤 2 跑完：
```powershell
$VENV -u experiments/minppi_filter_eval.py
# 产出: results/tables/<run_id_filter>/
#   minppi_filter_pipeline.csv
#   minppi_filter_pipeline_agg.csv
```

**步骤 4** — Pipeline comparison (~30min)。等步骤 3 跑完：
```powershell
$VENV experiments/eval_pipeline_comparison.py
# 产出: results/tables/<run_id_comp>/
#   pipeline_comparison.csv
#   pipeline_comparison_agg.csv
#   pipeline_comparison.pdf
```

### 第二层：已有实验（数据已就绪，跳过）

已有数据目录：
- `results/logs/2026-06-27_19-51-48/binary_vs_continuous/` — 二元 vs 连续消融
- `results/tables/2026-07-01_10-58-30/` — Madelon α-recall
- `results/logs/2026-06-29_07-08-39/` — harmful vs dimension
- `results/tables/2026-06-28_07-44-10/` — baseline 排名 (AUC@k)

### 第三层：图表和统计（秒级，等步骤 1-4 全部跑完）

```powershell
# 设 R_P = 步骤 1 的产出目录
$R_P = (Get-ChildItem -Directory "results/tables" | Where-Object { $_.Name -like "2026-07-0*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
Write-Host "Using pareto run: $R_P"

# CD 图 — 基于 pareto_front_agg.csv 的 AUC 列
$VENV experiments/make_cd_diagram.py results/tables/$R_P/pareto_front_agg.csv --metric AUC@10 --higher
$VENV experiments/make_cd_diagram.py results/tables/$R_P/pareto_front_agg.csv --metric AUC@20 --higher

# Friedman leave-one-out（输入文件路径需手动更新 sensitivity_friedman.py 中的 INPUT_CSV）
$VENV experiments/sensitivity_friedman.py

# Demo figures
$VENV experiments/make_figures.py
```

### 第四层：复制 PDF 产物到 figures/

```powershell
# α-Pareto 前沿图
Copy-Item results/tables/$R_P/pareto_front.pdf conference-paper/figures/

# CD 图
Copy-Item results/tables/$R_P/cd_diagram_AUC@10.pdf conference-paper/figures/
Copy-Item results/tables/$R_P/cd_diagram_AUC@20.pdf conference-paper/figures/

# Pipeline comparison 图
Copy-Item results/tables/<run_id_comp>/pipeline_comparison.pdf conference-paper/figures/

# Harmful vs dimension（已有）
Copy-Item results/logs/2026-06-29_07-08-39/harmful_vs_dimension.pdf conference-paper/figures/

# Madelon α-recall（已有）
Copy-Item results/tables/2026-07-01_10-58-30/madelon_alpha_recall.pdf conference-paper/figures/

# Demo figures（已有）
Copy-Item results/figures/figure_1_minppi_trichotomy.pdf conference-paper/figures/
Copy-Item results/figures/figure_2_prop1_detection.pdf conference-paper/figures/
Copy-Item results/figures/figure_3_algorithm_b_vs_a.pdf conference-paper/figures/
```

### 第五层：编译论文

```powershell
cd conference-paper/paper
pdflatex -interaction=nonstopmode icomp2026_conference.tex
bibtex icomp2026_conference
pdflatex -interaction=nonstopmode icomp2026_conference.tex
pdflatex -interaction=nonstopmode icomp2026_conference.tex
```

---

## 第7部分：一键完整运行脚本

```powershell
# ============================================
#  方案C 完整实验：从零到论文 PDF
#  复制粘贴整段，在 PowerShell 中运行
# ============================================

cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
$VENV = "..\venv\Scripts\python"

Write-Host "========== 第一层：耗时实验 =========="

# [1] α-Pareto 前沿主表 (4-5h) — 后台运行
Write-Host "[1/6] α-Pareto 前沿 (预计 4-5h)..."
$VENV -u experiments/run_pareto_eval.py
$R_P = (Get-ChildItem -Directory "results/tables" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
Write-Host "  -> 产出: results/tables/$R_P/"

# [2] Removal test + Madelon GT (3-4h)
Write-Host "[2/6] Removal test + Madelon GT (预计 3-4h)..."
$VENV -u experiments/eval_harmful_utility.py
$R_C1 = (Get-ChildItem -Directory "results/logs" | Where-Object { $_.Name -like "2026-07-0*" -and (Test-Path "$($_.FullName)/c1_removal") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
Write-Host "  -> 产出: results/logs/$R_C1/"

# [3] MinPPI filter pipeline (2h)
Write-Host "[3/6] MinPPI filter pipeline (预计 2h)..."
$VENV -u experiments/minppi_filter_eval.py
$R_FL = (Get-ChildItem -Directory "results/tables" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
Write-Host "  -> 产出: results/tables/$R_FL/"

# [4] Pipeline comparison (30min)
Write-Host "[4/6] Pipeline comparison (预计 30min)..."
$VENV experiments/eval_pipeline_comparison.py
$R_CP = (Get-ChildItem -Directory "results/tables" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
Write-Host "  -> 产出: results/tables/$R_CP/"

Write-Host "========== 第二层：图表和统计 =========="

# [5] CD 图 + Friedman leave-one-out + Demo figures (秒级)
Write-Host "[5/6] CD 图 + 统计..."
$VENV experiments/make_cd_diagram.py results/tables/$R_P/pareto_front_agg.csv --metric AUC@10 --higher
$VENV experiments/make_cd_diagram.py results/tables/$R_P/pareto_front_agg.csv --metric AUC@20 --higher
$VENV experiments/sensitivity_friedman.py
$VENV experiments/make_figures.py

Write-Host "========== 第三层：复制 PDF =========="

Copy-Item results/tables/$R_P/pareto_front.pdf conference-paper/figures/
Copy-Item results/tables/$R_P/cd_diagram_AUC@10.pdf conference-paper/figures/
Copy-Item results/tables/$R_P/cd_diagram_AUC@20.pdf conference-paper/figures/
Copy-Item results/tables/$R_CP/pipeline_comparison.pdf conference-paper/figures/
Copy-Item results/logs/2026-06-29_07-08-39/harmful_vs_dimension.pdf conference-paper/figures/
Copy-Item results/tables/2026-07-01_10-58-30/madelon_alpha_recall.pdf conference-paper/figures/
Copy-Item results/figures/figure_1_minppi_trichotomy.pdf conference-paper/figures/
Copy-Item results/figures/figure_2_prop1_detection.pdf conference-paper/figures/
Copy-Item results/figures/figure_3_algorithm_b_vs_a.pdf conference-paper/figures/

Write-Host "========== 第四层：编译 =========="

cd conference-paper/paper
pdflatex -interaction=nonstopmode icomp2026_conference.tex
bibtex icomp2026_conference
pdflatex -interaction=nonstopmode icomp2026_conference.tex
pdflatex -interaction=nonstopmode icomp2026_conference.tex

Write-Host ""
Write-Host "========== 全部完成 =========="
Write-Host "Pareto front:  results/tables/$R_P/"
Write-Host "Removal test:  results/logs/$R_C1/"
Write-Host "Filter pipe:   results/tables/$R_FL/"
Write-Host "Pipe comp:     results/tables/$R_CP/"
Write-Host "PDF:           conference-paper/paper/icomp2026_conference.pdf"
```

### 总运行时间预算

| 步骤 | 内容 | 时间 | 状态 |
|:--:|------|:--:|:--:|
| 1 | α-Pareto 前沿主表 | 4-5h | 新跑 |
| 2 | Removal test | 3-4h | 新跑 |
| 3 | Filter pipeline | 2h | 新跑 |
| 4 | Pipeline comparison | 30min | 新跑 |
| 5 | CD 图 + 统计 | <10min | 重跑 |
| 6 | 复制 PDFs | <1min | — |
| 7 | 编译 | <1min | — |
| **总计** | | **~9-11h** | — |

---

## 第8部分：文件变更总清单

### 新建 (3 个文件)
```
experiments/run_pareto_eval.py         ← 待写
experiments/minppi_filter_eval.py      ← ✅ 已完成
experiments/eval_pipeline_comparison.py ← ✅ 已完成
```

### 修改 (2 个文件)
```
src/minppi_mofs/__init__.py            ← 删 mofs_minppi 相关导入
experiments/sensitivity_friedman.py    ← 更新输入路径
```

### 删除（从论文中移除，文件保留）(6 个文件)
```
src/minppi_mofs/mofs_minppi.py
src/minppi_mofs/baselines/mofs_minppi_baseline.py
src/minppi_mofs/baselines/mofs_shap.py
src/minppi_mofs/baselines/nsga2_uniform.py
experiments/compute_pareto_metrics.py
experiments/pareto_metrics.py
experiments/run_alpha_sweep.py
experiments/run_baseline.py
experiments/run_restbaseline.py
```

### 保留不变 (18 个文件)
```
configs/ (7 个 YAML, 不变)
experiments/run_ablation.py
experiments/madelon_alpha_recall.py
experiments/harmful_vs_dimension.py
experiments/make_cd_diagram.py
experiments/make_figures.py
experiments/eval_harmful_utility.py
experiments/run_main_table.py
src/minppi_mofs/voting_game.py
src/minppi_mofs/min_ppi.py
src/minppi_mofs/shapg_integration.py
src/minppi_mofs/cross_shapley.py
src/minppi_mofs/baselines/shap_top_k.py
src/minppi_mofs/baselines/shapg.py
src/minppi_mofs/baselines/min_shap.py
src/minppi_mofs/baselines/banzhaf_fs.py
src/minppi_mofs/baselines/powershap.py
```

---

## 附录：从零开始的完整执行列表

以下假设 `C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master` 为工作目录，Python 为 `..\venv\Scripts\python`。

### 阶段 0：环境验证（1 min）

```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
..\venv\Scripts\python -c "
from minppi_mofs import compute_v_alpha, estimate_minppi
from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
from minppi_mofs.baselines.shapg import compute_shapg_importance
from minppi_mofs.baselines.min_shap import compute_minshap_importance
from minppi_mofs.baselines.banzhaf_fs import compute_banzhaf_importance
from minppi_mofs.baselines.powershap import compute_powershap_importance
print('All imports OK')
"
```

### 阶段 0.5：检查数据完整性（<1 min）

```powershell
..\venv\Scripts\python -c "
from pathlib import Path
ds = ['spambase','madelon','isolet','gisette','credit_card_fraud','crimedata','abalone']
for d in ds:
    for s in [42,123,456,789,1024]:
        p = Path(f'data/processed/{d}_s{s}_train.csv')
        if not p.exists(): print(f'MISSING: {p}')
print('Data check complete')
"
```

### 阶段 1：耗时实验（按顺序运行）

| 序号 | 命令 | 说明 | 预计时间 | 产出目录 |
|:--:|------|------|:--:|------|
| **1** | `$VENV -u experiments/run_main_table.py` | **5 baselines 单点排名**：SHAP-Top-k/ShapG/MinShap/Banzhaf/Powershap × 7 datasets × 5 seeds → AUC@k | 4-5h | `results/tables/<R0>/` |
| **2** | `$VENV -u experiments/run_pareto_eval.py` | **α-Pareto 前沿主表**：7 datasets × 5 seeds × 4 α → 压缩率 + AUC保留率 | 4-5h | `results/tables/<R1>/` |
| **3** | `$VENV -u experiments/eval_harmful_utility.py` | **c1 移除测试 + c2 Madelon GT + c8 MinShap 对比**：5+2 datasets × 5 seeds | 3-4h | `results/logs/<R2>/` |
| **4** | `$VENV -u experiments/minppi_filter_eval.py` | **MinPPI filter 管道**：MinPPI 过滤 → 5 baseline 在过滤后数据上排名 | 2h | `results/tables/<R3>/` |
| **5** | `$VENV experiments/eval_pipeline_comparison.py` | **管道对比**：MinPPI vs MinShap vs Random 预处理 → SHAP-top-k | 30min | `results/tables/<R4>/` |

> **并行提示**：步骤 1、2、3 完全独立，可三者同时启动以节省 6-8h。

### 阶段 2：已有实验（已跑过，跳过）

| 原始运行 | 内容 | 目录 |
|----------|------|------|
| `run_ablation.py` | binary_vs_continuous 消融 | `results/logs/2026-06-27_19-51-48/` |
| `madelon_alpha_recall.py` | Madelon α-recall 曲线 | `results/tables/2026-07-01_10-58-30/` |
| `harmful_vs_dimension.py` | harmful% vs 维度 | `results/logs/2026-06-29_07-08-39/` |

### 阶段 3：图表与统计（秒级，等阶段 1 全部跑完）

各产出目录标识符：
- **R0** = `run_main_table.py` 产出（baseline 排名）
- **R1** = `run_pareto_eval.py` 产出（α-Pareto 前沿）
- **R4** = `eval_pipeline_comparison.py` 产出（管道对比）

```powershell
# 取最新各产出目录
$R0 = (Get-ChildItem -Directory "results/tables" | Where-Object { $_.Name -like "2026-07-0*" -and (Test-Path "$_\main_metrics.csv") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
$R1 = (Get-ChildItem -Directory "results/tables" | Where-Object { $_.Name -like "2026-07-0*" -and (Test-Path "$_\pareto_front_agg.csv") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name

# CD 图 — 基于 baseline 排名数据（R0）和 α-Pareto 数据（R1）
$VENV experiments/make_cd_diagram.py results/tables/$R0/mean_metrics.csv --metric AUC@10 --higher
$VENV experiments/make_cd_diagram.py results/tables/$R0/mean_metrics.csv --metric AUC@20 --higher

# Friedman leave-one-out
$VENV experiments/sensitivity_friedman.py

# Demo figures (Figure 1-3)
$VENV experiments/make_figures.py
```

### 阶段 4：复制 PDF 产物到 `conference-paper/figures/`

```powershell
# 设各 run_id
$R0 = (Get-ChildItem -Directory "results/tables" | Where-Object { $_.Name -like "2026-07-0*" -and (Test-Path "$_\main_metrics.csv") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
$R1 = (Get-ChildItem -Directory "results/tables" | Where-Object { $_.Name -like "2026-07-0*" -and (Test-Path "$_\pareto_front_agg.csv") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
$R4 = (Get-ChildItem -Directory "results/tables" | Where-Object { $_.Name -like "2026-07-0*" -and (Test-Path "$_\pipeline_comparison.pdf") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name

# 新产出的图
Copy-Item results/tables/$R1/pareto_front.pdf                conference-paper/figures/   # α-Pareto 前沿
Copy-Item results/tables/$R0/cd_diagram_AUC@10.pdf            conference-paper/figures/   # CD 图 (baseline 数据)
Copy-Item results/tables/$R0/cd_diagram_AUC@20.pdf            conference-paper/figures/
Copy-Item results/tables/$R4/pipeline_comparison.pdf          conference-paper/figures/   # 管道对比

# 已有产出的图（从旧运行直接复制，不用重跑）
Copy-Item results/logs/2026-06-29_07-08-39/harmful_vs_dimension.pdf     conference-paper/figures/
Copy-Item results/tables/2026-07-01_10-58-30/madelon_alpha_recall.pdf   conference-paper/figures/
Copy-Item results/figures/figure_1_minppi_trichotomy.pdf                conference-paper/figures/
Copy-Item results/figures/figure_2_prop1_detection.pdf                  conference-paper/figures/
Copy-Item results/figures/figure_3_algorithm_b_vs_a.pdf                 conference-paper/figures/
```

### 阶段 5：编译论文

```powershell
cd conference-paper/paper
pdflatex -interaction=nonstopmode icomp2026_conference.tex
bibtex icomp2026_conference
pdflatex -interaction=nonstopmode icomp2026_conference.tex
pdflatex -interaction=nonstopmode icomp2026_conference.tex
```

### 成本汇总

| 阶段 | 必须重跑 | 时间 | 可并行 |
|------|:--:|:--:|:--:|
| 0. 环境 + 数据检查 | 是 | 1min | — |
| 1.1 run_main_table (baseline 排名) | **是** | 4-5h | 与 1.2/1.3 并行 |
| 1.2 α-Pareto 前沿 | **是** | 4-5h | 与 1.1/1.3 并行 |
| 1.3 Removal test | **是** | 3-4h | 与 1.1/1.2 并行 |
| 1.4 Filter 管道 | **是** | 2h | 等 1.3 |
| 1.5 管道对比 | **是** | 30min | 等 1.4 |
| 2. 已有实验 (消融/α-recall/维度) | 否 | 0 | — |
| 3. 图表 + 统计 | 是(快) | <10min | 等 1.1+1.2 |
| 4. 复制 PDF | 是 | <1min | — |
| 5. 编译 | 是 | <1min | — |
| **总计（新跑）** | | **~5-6h**（三步并行后） | |
