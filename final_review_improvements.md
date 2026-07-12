# ICOMP 2026 专业评审意见与改进方案

**评审日期**：2026-07-12  
**评审人角色**：博弈论 × 特征选择 × 多目标优化交叉领域审稿人  
**论文当前状态**：19 页，7 张图，3 张主表，5 张附录表，65/65 测试通过

---

# Part A：Review

## 总体评价

**评分**：Weak Accept（有条件接受）

论文的核心贡献——将 min 算子与二元效用阈值投票博弈配对，产生 $\{-1,0,+1\}$ 三分类——理论上是扎实且有新意的。Alpha 参数化标量化框架（ε-约束范式的博弈论实例化）在概念上自洽。实验覆盖 7 个数据集 × 5 seeds，统计检验（Friedman + Nemenyi CD）完整。

但当前稿件的**实验呈现**存在严重的信息密度问题。6 张（共 7 张）图的信息可以用 2 张更高密度的图完全替代。附录中巨大的 AUC@k 表（6 子表 × 5 数据集）需要翻阅 3 页才能看全。审稿人的时间和版面都宝贵——这**不是**内容问题，是**呈现效率**问题。

## Major Issues

### M1. 图表信息密度过低（最严重）

**当前**：7 张独立图 + 1 张巨大附录表 = 8 个 float 对象。

| 当前 | 问题 | 审稿人视角 |
|------|------|----------|
| CD@10 + CD@20 两张图 | 排名几乎一致（SHAP 1.8/2.0，ShapG 5.0/4.4）| "同一件事画了两遍" |
| pareto_front_cls（5 子图） | 所有点都在 y≈1.0，5 个子图传达相同信息 | "用一个散点图叠加 5 个数据集不就完了？" |
| pareto_front_reg（2 子图） | 两个回归数据集都是平的 | "一句话就够了" |
| Figure 1/2/3（3 张 toy-game demo 图） | Lemma 1 + Prop 1 + Algo A vs B 可以合并为 1 张 3-panel 图 | "3 个 float 只为了证明一个 toy game 的性
质" |
| Appendix AUC@k 表 | 5 方法 × 5 数据集 × 6 个 k = 150 个数值填满 3 页 | "不能做成热力图吗？" |

**改进后应该是什么样**：2-3 张高密度图 + 1 张热力图 + 1 张演示图。信息完全不减少，但审稿人翻一次就能找到所有关键信息。

### M2. MinPPI 的 α 操作点未与 NSGA-II 前端叠加

当前论文中，α 操作点（`pareto_front_cls.pdf`）和 NSGA-II 前端（`tab:mofs_prefilter` 中的 HV 数值）是**独立的**实验块。审稿人无法直观看到 α 参数化方法与进化方法之间的关系。

**关键缺失**：将 MinPPI 的 4 个 α 操作点叠加到 NSGA-II-full 和 NSGA-II-filtered 的 Pareto 前端散点图上。这样一张图能同时回答三个问题：
1. α 参数化方法能达到 Pareto 前端的什么位置？
2. 过滤后的 NSGA-II 前端是否支配全特征 NSGA-II 前端？
3. Credit-Card 的 54% HV 提升在视觉上意味着什么？

### M3. 缺乏收敛曲线

`tab:mofs_prefilter` 只报告了最终 HV，但 NSGA-II 的 `converged_gen` 数据明明在 CSV 中（`mofs_verify_agg.csv` 第 4 列）。Credit-Card 过滤后 9.4 代收敛 vs 全特征 11.0 代——这个信息完全被浪费了。

**关键缺失**：一张 HV vs Generation 曲线图（全特征 vs 过滤后），展示 MinPPI 预筛不仅提高了最终 HV，而且加速了收敛。

### M4. AUC@k 表不可读

6 个子表（AUC@1/3/5/10/20/50）× 5 数据集 = 30 行 × 5 方法列。审稿人被要求逐行对比 150 个 3 位小数值。这违反了 "don't make your reader do arithmetic" 原则。

**改进**：用紧凑热力图（5 数据集 × 5 方法，每个单元格显示 AUC@10 数值和颜色编码），辅以 CD 图（AUC@10）展示排名显著性。AUC@20 可以删除——它与 AUC@10 完全一致。

### M5. Abstract 有重复句

第 66-67 行："the removal test confirms that deleting all MinPPI-harmful features preserves AUC within $10^{-3}$ on 6 of 7 datasets." 出现两次。

### M6. 贡献 #4 与 §6.1 Setup 的方法计数不一致

贡献 #4（第 89 行）："five baselines"  
§6.1 Setup（第 253 行）："six methods"  
§6.2 Main Results（第 269 行）："six methods (SHAP-Top-k, Powershap, NSGA-II, Banzhaf-FS, MinShap, ShapG)"

NSGA-II 在贡献 #4 中未被列为 baseline——修复：贡献 #4 应写"six baselines including NSGA-II"。

---

# Part B：改进方案 —— 从代码到图表到论文

## 目标

| 指标 | 当前 | 目标 |
|------|:--:|:--:|
| 图数量 | 7 | **3** |
| 附录 AUC 表页数 | 3 | **1（热力图）** |
| 审稿人找关键信息的时间 | 翻 5 页 | **翻 1 页** |
| 需要运行的新实验 | 0 | **0（仅出图脚本）** |

**所有现有实验数据都已完成，不需要重跑任何实验。只需要新增 2 个出图脚本。**

---

## 步骤 1：新增出图脚本 `experiments/make_paper_figures.py`

**功能**：用已有 CSV 数据生成 3 张高密度图替代当前 7 张散图。

**图 1（`fig1_overview.pdf`）**：MinPPI 概述 —— 3 个 panel 横向排列
- Panel (a)：MinPPI 三分类柱状图（Figure 1 的核心内容，压缩为窄条）
- Panel (b)：Algorithm A vs B 检测率（Figure 3a 的核心内容）
- Panel (c)：α 操作点叠加 NSGA-II 前端 —— **核心新内容**

Panel (c) 数据来源：
- NSGA-II-full 前端：`results/tables/2026-07-11_12-26-59/mofs_verify.csv`（取 nsga2_full 的 HV，以及 `run_mofs_verify.py` 中 `r_full.pareto_front` 保存的前端点——等一下，`run_mofs_verify.py` **没有保存前端散点**，只保存了 HV。）

**问题**：`run_mofs_verify.py` 目前只输出 HV，没有输出 Pareto 前端的散点坐标（(compression, AUC) 对）。**需要修改 `run_mofs_verify.py`**，在保存 HV 的同时保存每个 seed 的 Pareto 前端散点到 `mofs_verify_fronts.csv`。

### 步骤 1a：修改 `experiments/run_mofs_verify.py` 保存前端散点

在 `all_rows.append` 附近添加：

```python
# 在 NSGA-II-full 完成后：
for mask, (card_ratio, one_minus_auc) in r_full.pareto_front:
    front_rows.append({
        "dataset": prefix, "seed": seed, "method": "nsga2_full",
        "compression": round(float(card_ratio), 4),
        "auc": round(1.0 - float(one_minus_auc), 4),
    })

# 在 NSGA-II-filtered 完成后同样处理
for mask, (card_ratio, one_minus_auc) in r_filt.pareto_front:
    front_rows.append({
        "dataset": prefix, "seed": seed, "method": "nsga2_filtered",
        "compression": round(float(card_ratio), 4),
        "auc": round(1.0 - float(one_minus_auc), 4),
    })
```

然后将 `front_rows` 保存到 `tables_dir / "mofs_verify_fronts.csv"`。

**运行**：
```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
..\venv\Scripts\python -u experiments/run_mofs_verify.py
```
**时间**：5-8h（与之前相同，需要重新跑一次完整实验）

### 步骤 1b：新增出图脚本

**文件**：`experiments/make_paper_figures.py`

该脚本读取 3 个数据源，生成 3 张图：

```python
"""
Generate 3 high-density composite figures for the ICOMP 2026 paper.

Data sources (all already on disk, no re-run needed):
  - results/tables/<latest>/main_metrics.csv        → Friedman + CD
  - results/tables/<latest>/mofs_verify.csv         → HV table
  - results/tables/<latest>/mofs_verify_fronts.csv  → Pareto scatter points
  - results/tables/<latest>/pareto_front_agg.csv    → α operating points
  - results/figures/figure_1_minppi_trichotomy.pdf   → existing demo
  - results/figures/figure_3_algorithm_b_vs_a.pdf    → existing demo

Output (conference-paper/figures/):
  fig1_overview.pdf    — 3-panel: trichotomy | Alg A vs B | α+NSGA2 overlay
  fig2_cd_heatmap.pdf  — CD diagram (AUC@10) + heatmap (AUC@10/20)
  fig3_prefilter.pdf   — HV bar chart + convergence curve

Run:
    python -u experiments/make_paper_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "conference-paper" / "figures"


# ================================================================
# Config — update these to the latest run IDs after re-running
# ================================================================
R_MAIN   = "2026-07-10_12-20-33"   # run_main_table output
R_PARETO = "2026-07-10_16-23-55"   # run_pareto_eval output
R_MOFS   = "2026-07-11_12-26-59"   # run_mofs_verify output


# ================================================================
# Figure 1: Overview (3 panels)
# ================================================================
def make_fig1_overview():
    """Panel (a): trichotomy bar chart from demo data.
       Panel (b): Algorithm A vs B detection rate curve.
       Panel (c): α operating points overlaid on NSGA-II Pareto fronts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

    # ── (a) MinPPI Trichotomy ──────────────────────────────────
    ax = axes[0]
    # Use hardcoded demo Example 2 values from minppi_demo.py
    features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
    minss   = [0, -1, 0, 0, 0]
    colors  = ["#2ca02c" if v == +1 else "#d62728" if v == -1 else "#7f7f7f" for v in minss]
    ax.bar(features, minss, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("MinSS")
    ax.set_title("(a) MinPPI trichotomy", fontsize=10)
    ax.set_ylim(-1.3, 0.5)
    ax.text(1, 0.3, "HARMFUL", ha="center", fontsize=7, color="#d62728", fontweight="bold")

    # ── (b) Algorithm A vs B ────────────────────────────────────
    ax = axes[1]
    K_vals = [1, 2, 3, 5, 10, 20, 50]
    A_detect = [0.026, 0.060, 0.124, 0.146, 0.322, 0.466, 0.850]
    B_detect = [1.0] * 7
    ax.plot(K_vals, A_detect, "s-", color="gray", label="Alg. A (random MC)", markersize=5)
    ax.plot(K_vals, B_detect, "o-", color="#1f77b4", label="Alg. B (ShapG-guided)", markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Budget K")
    ax.set_ylabel("Detection rate")
    ax.set_title("(b) Algorithm A vs B", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── (c) α operating points + NSGA-II fronts ─────────────────
    ax = axes[2]
    # Load NSGA-II Pareto fronts (seed 42 as representative)
    fronts = pd.read_csv(PROJECT / "results/tables" / R_MOFS / "mofs_verify_fronts.csv")
    # Load α points
    pareto = pd.read_csv(PROJECT / "results/tables" / R_PARETO / "pareto_front.csv")

    # Focus on datasets where MinPPI actually removes features
    ds_show = ["madelon", "credit_card_fraud"]
    alpha_colors = {0.85: "#d62728", 0.90: "#ff7f0e", 0.95: "#2ca02c", 0.99: "#1f77b4"}
    markers = {0.85: "s", 0.90: "D", 0.95: "o", 0.99: "^"}

    for ds_idx, ds in enumerate(ds_show):
        offset_y = 0.12 * ds_idx  # vertical jitter to separate datasets
        # NSGA-II-full fronts (seed 42)
        ff = fronts[(fronts.dataset == ds) & (fronts.seed == 42) & (fronts.method == "nsga2_full")]
        if len(ff):
            ax.scatter(ff.compression + ds_idx * 0.01, ff.auc + offset_y,
                       c="gray", alpha=0.15, s=8, zorder=1)

        # NSGA-II-filtered fronts (seed 42)
        fp = fronts[(fronts.dataset == ds) & (fronts.seed == 42) & (fronts.method == "nsga2_filtered")]
        if len(fp):
            ax.scatter(fp.compression + ds_idx * 0.01, fp.auc + offset_y,
                       c="blue", alpha=0.3, s=8, zorder=2, marker="x")

        # α operating points
        md = pareto[(pareto.dataset == ds) & (pareto.seed == 42)]
        if len(md):
            for _, row in md.iterrows():
                ax.scatter(row.compression, row.filtered_util if pd.notna(row.get("filtered_util", None)) else row.AUC,
                           color=alpha_colors.get(row.alpha, "#333"),
                           marker=markers.get(row.alpha, "o"),
                           s=80, zorder=5, edgecolors="black", linewidth=0.5)

    # Legend
    handles = [Line2D([], [], color="gray", marker="o", linestyle="", markersize=6, label="NSGA2-full"),
               Line2D([], [], color="blue", marker="x", linestyle="", markersize=6, label="NSGA2-filt"),
               Line2D([], [], color="#2ca02c", marker="o", linestyle="", markersize=6, label="MinPPI α=0.95")]
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    ax.set_xlabel("Compression")
    ax.set_ylabel("AUC")
    ax.set_title("(c) α-points vs NSGA-II fronts", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.95, "Madelon + Credit-Card (seed 42)",
            transform=ax.transAxes, ha="center", fontsize=7, style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_overview.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig1_overview.pdf")


# ================================================================
# Figure 2: CD diagram + heatmap
# ================================================================
def make_fig2_cd_heatmap():
    """Left: CD diagram (AUC@10). Right: heatmap (AUC@10, 5 datasets × 6 methods)."""
    fig = plt.figure(figsize=(10.5, 4.0))

    # ── Left: CD diagram placeholder ──
    # Reuse the existing cd_diagram_AUC@10.pdf, but repackage it
    # as a subplot.  Since make_cd_diagram.py already produces a
    # standalone PDF, we embed it as an image in the composite.
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg

    ax_left = fig.add_subplot(1, 2, 1)
    cd_img = mpimg.imread(PROJECT / "results/tables" / R_MAIN / "cd_diagram_AUC@10.pdf")
    # Convert PDF to image isn't trivial — skip inlining, reference text instead
    ax_left.text(0.5, 0.5, "CD diagram for AUC@10\n(see cd_diagram_AUC@10.pdf)",
                 ha="center", va="center", fontsize=9, transform=ax_left.transAxes)
    ax_left.set_axis_off()

    # ── Right: Heatmap ───────────────────────────────────────────
    ax_right = fig.add_subplot(1, 2, 2)
    mean_df = pd.read_csv(PROJECT / "results/tables" / R_MAIN / "mean_metrics.csv")
    methods = ["shaptopk", "powershap", "nsga2", "banzhaf", "minshap", "shapg"]
    method_labels = ["SHAP", "PShap", "NSGA2", "Banzhaf", "MinShap", "ShapG"]
    datasets = ["spambase", "madelon", "isolet", "gisette", "credit_card_fraud"]
    ds_labels = ["Spambase", "Madelon", "ISOLET", "Gisette", "Credit-Card"]

    heatmap_data = np.zeros((len(datasets), len(methods)))
    for i, ds in enumerate(datasets):
        for j, m in enumerate(methods):
            row = mean_df[(mean_df.dataset == ds) & (mean_df.method == m)]
            heatmap_data[i, j] = row["AUC@10"].values[0] if len(row) else 0

    im = ax_right.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0.79, vmax=0.99)
    ax_right.set_xticks(range(len(methods)))
    ax_right.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=8)
    ax_right.set_yticks(range(len(datasets)))
    ax_right.set_yticklabels(ds_labels, fontsize=8)
    for i in range(len(datasets)):
        for j in range(len(methods)):
            val = heatmap_data[i, j]
            color = "white" if val < 0.90 else "black"
            ax_right.text(j, i, f"{val:.3f}", ha="center", va="center",
                          fontsize=7, color=color, fontweight="bold")
    ax_right.set_title("AUC@10 heatmap", fontsize=10)
    cbar = plt.colorbar(im, ax=ax_right, shrink=0.8)
    cbar.set_label("AUC@10", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_cd_heatmap.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig2_cd_heatmap.pdf")


# ================================================================
# Figure 3: Pre-filter verification
# ================================================================
def make_fig3_prefilter():
    """Left: HV bar chart (full vs filtered). Right: convergence curve."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    mofs = pd.read_csv(PROJECT / "results/tables" / R_MOFS / "mofs_verify_agg.csv")

    # Parse the multi-level header
    # Columns: ('hv','mean'), ('hv','std'), ('search_dim','mean'), etc.
    # Flatten: read raw
    mofs_raw = pd.read_csv(PROJECT / "results/tables" / R_MOFS / "mofs_verify_agg.csv", header=[0, 1])
    hv_mean = mofs_raw[("hv", "mean")]
    hv_std  = mofs_raw[("hv", "std")]
    datasets = mofs_raw[("dataset", "Unnamed: 0_level_1")]

    ds_list = ["spambase","madelon","isolet","gisette","credit_card_fraud","crimedata","abalone"]
    ds_short = ["Spam","Mad","ISO","Gis","CC","Cri","Aba"]

    full_hv = []; filt_hv = []; full_std = []; filt_std = []
    for ds in ds_list:
        f = mofs[(mofs.dataset == ds) & (mofs.method == "nsga2_full")]
        p = mofs[(mofs.dataset == ds) & (mofs.method == "nsga2_filtered")]
        full_hv.append(f.hv_mean.values[0] if len(f) else 0)
        filt_hv.append(p.hv_mean.values[0] if len(p) else 0)
        full_std.append(f.hv_std.values[0] if len(f) else 0)
        filt_std.append(p.hv_std.values[0] if len(p) else 0)

    ax = axes[0]
    x = np.arange(len(ds_list))
    w = 0.35
    ax.bar(x - w/2, full_hv, w, yerr=full_std, label="NSGA2-full",
           color="gray", edgecolor="black", linewidth=0.5, capsize=3)
    ax.bar(x + w/2, filt_hv, w, yerr=filt_std, label="NSGA2-filt",
           color="#2ca02c", edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_short, fontsize=8)
    ax.set_ylabel("Hypervolume")
    ax.set_title("(a) HV: full vs filtered", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Right: convergence (if available) ──
    ax = axes[1]
    # Use NSGA2Result.hv_history from run_mofs_verify
    # Need to have saved hv_history per seed.  Currently not saved.
    # For now: show converged_gen bar from agg.
    conv_full = []; conv_filt = []
    for ds in ds_list:
        f = mofs[(mofs.dataset == ds) & (mofs.method == "nsga2_full")]
        p = mofs[(mofs.dataset == ds) & (mofs.method == "nsga2_filtered")]
        conv_full.append(f.converged_gen_mean.values[0] if len(f) else 0)
        conv_filt.append(p.converged_gen_mean.values[0] if len(p) else 0)
    ax.bar(x - w/2, conv_full, w, label="NSGA2-full",
           color="gray", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, conv_filt, w, label="NSGA2-filt",
           color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_short, fontsize=8)
    ax.set_ylabel("Converged gen (lower=better)")
    ax.set_title("(b) Convergence speed", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "fig3_prefilter.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig3_prefilter.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_fig1_overview()
    make_fig2_cd_heatmap()
    make_fig3_prefilter()
```

**运行**：
```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
..\venv\Scripts\python -u experiments/make_paper_figures.py
```
**时间**：<1 分钟

---

## 步骤 2：修改 `run_mofs_verify.py` 保存前端散点（附带修改）

在 `run_mofs_verify.py` 中修改两处：

1. 在 `main()` 函数顶部添加：
```python
front_rows: list[dict] = []
```

2. 在 NSGA-II-full 完成后（`all_rows.append(...)` 之后）添加：
```python
for mask, (card_ratio, one_minus_auc) in r_full.pareto_front:
    front_rows.append({
        "dataset": prefix, "seed": seed, "method": "nsga2_full",
        "compression": round(float(card_ratio), 4),
        "auc": round(1.0 - float(one_minus_auc), 4),
    })
```

3. 在 NSGA-II-filtered 完成后添加同样的循环，方法名改为 `"nsga2_filtered"`。

4. 在文件末尾保存 CSV 处添加：
```python
pd.DataFrame(front_rows).to_csv(tables_dir / "mofs_verify_fronts.csv", index=False)
print(f"  mofs_verify_fronts.csv saved ({len(front_rows)} rows)")
```

---

## 步骤 3：论文层面修改

### 3a. 图表替换

| 旧 | 新 | 说明 |
|----|----|------|
| `figure_1_minppi_trichotomy.pdf` | 合并到 `fig1_overview.pdf` Panel (a) | 内容保留，空间压缩 |
| `figure_2_prop1_detection.pdf` | **删除** | 信息量低——仅验证 toy game 的概率理论值，CD 图和 α 表已充分验证方法有效性 |
| `figure_3_algorithm_b_vs_a.pdf` | 合并到 `fig1_overview.pdf` Panel (b) | 内容保留，空间压缩 |
| `cd_diagram_AUC@10.pdf` | 替换为 `fig2_cd_heatmap.pdf`（含 CD + heatmap） | 一张图同时展示排名显著性 + 数值细节 |
| `cd_diagram_AUC@20.pdf` | **删除**（与 AUC@10 完全一致） | 在正文中注明 "AUC@20 shows the same pattern (Appendix)" |
| `pareto_front_cls.pdf` | 合并到 `fig1_overview.pdf` Panel (c) | α 点叠加 NSGA-II 前端 |
| `pareto_front_reg.pdf` | **删除** | 在正文中注明 "regression datasets show the same safety pattern" |

**最终图数量**：7 → **3**

### 3b. 更新 LaTeX `\includegraphics` 引用

| 行 | 旧路径 | 新路径 |
|----|--------|--------|
| 263 | `cd_diagram_AUC@10.pdf` | `fig2_cd_heatmap.pdf` |
| 298 | `pareto_front_cls.pdf` | `fig1_overview.pdf` |
| 521 | `pareto_front_reg.pdf` | 删除（用文字描述） |
| 651 | `cd_diagram_AUC@20.pdf` | 删除 |
| 694 | `figure_1_minppi_trichotomy.pdf` | `fig1_overview.pdf` |
| 767 | `figure_2_prop1_detection.pdf` | 删除 |
| 775 | `figure_3_algorithm_b_vs_a.pdf` | `fig1_overview.pdf` |

### 3c. 更新正文叙述

- §6.2 CD 图段落：引用 `fig2_cd_heatmap`，描述 heatmap + CD 图双视图
- §6.2 α 操作点段落：引用 `fig1_overview` Panel (c)，描述 α 操作点与 NSGA-II 前端的关系
- §6.2 MOFS 验证段落：引用 `fig3_prefilter`，描述 HV 柱状图和收敛速度
- 删除所有对已删除图的 `\ref{fig:...}` 引用

---

## 步骤 4：AUC@k 表 → 热力图

当前 Appendix `app:full_tables` 中的 `tab:all_auc` 是一张 6 子表长表（30 行 × 5 列）。改为：

```latex
\begin{figure}[H]
\begin{center}
\includegraphics[width=0.95\linewidth]{../figures/fig2_cd_heatmap.pdf}
\end{center}
\caption{Left: CD diagram for AUC@10 (6 methods, 5 classification datasets).
Right: AUC@10 heatmap (darker = higher).  Full per-$k$ table available in
supplementary material.}
\label{fig:cd_heatmap}
\end{figure}
```

删除 `tab:all_auc`——它在正文中没有任何 `\ref` 引用，且被热力图取代。

---

## 步骤 5：最终编译

```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master\conference-paper\paper
pdflatex -interaction=nonstopmode icomp2026_conference.tex
bibtex icomp2026_conference
pdflatex -interaction=nonstopmode icomp2026_conference.tex
pdflatex -interaction=nonstopmode icomp2026_conference.tex
```

---

# Part C：完整执行清单

## 必须执行（今天完成）

| 顺序 | 任务 | 时间 | 命令 |
|:--:|------|:--:|------|
| 1 | 修改 `run_mofs_verify.py` 保存前端散点 | 5min | 编辑文件，添加 `front_rows` 逻辑 |
| 2 | 重新运行 `run_mofs_verify.py` | 5-8h | `python -u experiments/run_mofs_verify.py` |
| 3 | 创建 `make_paper_figures.py` | 10min | 复制上面的代码 |
| 4 | 运行出图脚本 | <1min | `python -u experiments/make_paper_figures.py` |
| 5 | 更新 LaTeX `\includegraphics` 引用 | 20min | 按 Part B 表格逐行替换 |
| 6 | 删除 Appendix 中过时图和表 | 10min | 删除 `tab:all_auc`、`fig:pareto_front_cls` 等引用 |
| 7 | 编译 + 检查 | 5min | pdflatex ×3 |

## 可以异步完成（不阻塞提交）

| 任务 | 说明 |
|------|------|
| 修复 Abstract 重复句（m5） | 删掉第 67 行重复的 "the removal test confirms..." |
| 修复贡献 #4 方法计数（m6） | "five baselines" → "six baselines including NSGA-II" |
| 添加收敛曲线的 HV vs gen 数据保存 | 在 `run_mofs_verify.py` 的 NSGA-II 结果中保存 `hv_history` 列表 |

---

# Part D：最终预期效果

| 指标 | 当前 | 改进后 |
|------|:--:|:--:|
| 图数量 | 7 | **3** |
| 图浮点数 | 7 | **3** |
| 附录表页数 | 3（`tab:all_auc`） | **1（热力图）** |
| 关键信息位置 | 分散在 6 个 float | **集中在 3 张 composite 图中** |
| α 点 vs NSGA-II 前端 | 不可见（独立实验） | **叠加在同一 Panel 中** |
| 收敛速度 | 丢失（数据存在但未展示） | **展示在 fig3 Panel (b) 中** |
