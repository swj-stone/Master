"""
Generate 3 high-density composite figures replacing 7 scattered figures.

Data sources (all on disk, no re-runs needed):
  - results/tables/2026-07-10_12-20-33/main_metrics.csv  → CD + heatmap
  - results/tables/2026-07-10_16-23-55/pareto_front.csv   → α operating points
  - results/tables/2026-07-12_08-02-42/mofs_verify.csv    → HV table
  - results/tables/2026-07-12_08-02-42/mofs_verify_fronts.csv → Pareto scatter points
  - results/tables/2026-07-12_08-02-42/mofs_verify_history.csv → HV per generation

Run:
    python -u experiments/make_paper_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "conference-paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

R_MAIN = "2026-07-10_12-20-33"
R_PARETO = "2026-07-10_16-23-55"
R_MOFS = "2026-07-12_08-02-42"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
})


# ================================================================
# Figure 1: Overview (3 panels)
# ================================================================
def make_fig1_overview():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    # ── (a) MinPPI Trichotomy — Example 2 from minppi_demo ────
    ax = axes[0]
    features = ["f_0", "f_1", "f_2", "f_3", "f_4"]
    minss = [0, -1, 0, 0, 0]
    colors = ["#2ca02c" if v == +1 else "#d62728" if v == -1 else "#7f7f7f"
              for v in minss]
    ax.bar(features, minss, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("MinSS")
    ax.set_title("(a) MinPPI trichotomy", fontsize=10)
    ax.set_ylim(-1.3, 0.5)
    ax.text(1, 0.3, "HARMFUL", ha="center", fontsize=7,
            color="#d62728", fontweight="bold")

    # ── (b) Algorithm A vs B detection rate ───────────────────
    ax = axes[1]
    K_vals = [1, 2, 3, 5, 10, 20, 50]
    A_detect = [0.026, 0.060, 0.124, 0.146, 0.322, 0.466, 0.850]
    B_detect = [1.0] * 7
    ax.plot(K_vals, A_detect, "s-", color="gray", label="Alg. A (random MC)",
            markersize=5)
    ax.plot(K_vals, B_detect, "o-", color="#1f77b4",
            label="Alg. B (MI-correlation-guided)", markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Budget K")
    ax.set_ylabel("Detection rate")
    ax.set_title("(b) Algorithm A vs B", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── (c) α operating points + NSGA-II fronts ───────────────
    ax = axes[2]
    fronts = pd.read_csv(PROJECT / "results/tables" / R_MOFS / "mofs_verify_fronts.csv")
    pareto = pd.read_csv(PROJECT / "results/tables" / R_PARETO / "pareto_front.csv")

    ds_show = ["madelon", "credit_card_fraud"]
    alpha_colors = {0.85: "#d62728", 0.90: "#ff7f0e", 0.95: "#2ca02c",
                    0.99: "#1f77b4"}
    markers = {0.85: "s", 0.90: "D", 0.95: "o", 0.99: "^"}
    for ds_idx, ds in enumerate(ds_show):
        offset_y = 0.12 * ds_idx
        # NSGA-II-full fronts (seed 42 as representative)
        ff = fronts[(fronts.dataset == ds) & (fronts.seed == 42) &
                     (fronts.method == "nsga2_full")]
        if len(ff):
            ax.scatter(ff.compression + ds_idx * 0.01,
                       ff.auc + offset_y,
                       c="gray", alpha=0.20, s=8, zorder=1)

        # NSGA-II-filtered fronts (seed 42)
        fp = fronts[(fronts.dataset == ds) & (fronts.seed == 42) &
                     (fronts.method == "nsga2_filtered")]
        if len(fp):
            ax.scatter(fp.compression + ds_idx * 0.01,
                       fp.auc + offset_y,
                       c="#1f77b4", alpha=0.35, s=10, zorder=2, marker="x")

        # α operating points (seed 42)
        md = pareto[(pareto.dataset == ds) & (pareto.seed == 42)]
        if len(md):
            for _, row in md.iterrows():
                y_val = (row.filtered_util
                         if pd.notna(row.get("filtered_util"))
                         else (row["AUC"] if pd.notna(row.get("AUC"))
                               else row.get("R2", 0)))
                ax.scatter(row.compression + ds_idx * 0.01,
                           y_val + offset_y,
                           color=alpha_colors.get(row.alpha, "#333"),
                           marker=markers.get(row.alpha, "o"),
                           s=80, zorder=5, edgecolors="black", linewidth=0.5)

    handles = [
        Line2D([], [], color="gray", marker="o", linestyle="", markersize=6,
               label="NSGA2-full"),
        Line2D([], [], color="#1f77b4", marker="x", linestyle="", markersize=6,
               label="NSGA2-filt"),
        Line2D([], [], color="#2ca02c", marker="o", linestyle="", markersize=6,
               label="MinPPI α=0.95"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    ax.set_xlabel("Compression")
    ax.set_ylabel("AUC")
    ax.set_title("(c) α-points vs NSGA-II fronts", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.95, "Madelon (top)  Credit-Card (bottom) — seed 42",
            transform=ax.transAxes, ha="center", fontsize=7, style="italic")

    fig.tight_layout()
    fp_out = OUT / "fig1_overview.pdf"
    fig.savefig(fp_out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp_out.name}")


# ================================================================
# Figure 2: CD diagram placeholder + AUC@10 heatmap
# ================================================================
def make_fig2_cd_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # ── Left: CD diagram info ──────────────────────────────────
    ax = axes[0]
    avg_ranks = pd.read_csv(
        PROJECT / "results/tables" / R_MAIN / "avg_ranks_AUC@10.csv")
    methods = avg_ranks["method"].tolist()
    ranks = avg_ranks["avg_rank"].tolist()

    method_labels = {"shaptopk": "SHAP", "powershap": "PShap", "nsga2": "NSGA2",
                     "banzhaf": "Banzhaf", "minshap": "MinShap", "shapg": "ShapG"}
    labels = [method_labels.get(m, m) for m in methods]

    # Simple rank bar chart + significance annotation
    colors_bar = ["#2ca02c" if r <= 2.5 else "#ff7f0e" if r <= 4.0 else "#d62728"
                  for r in ranks]
    y_pos = range(len(methods))
    ax.barh(y_pos, ranks, color=colors_bar, edgecolor="black", linewidth=0.5,
            height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Avg. rank (1=best)")
    ax.set_title("(a) AUC@10 ranks  (Friedman p=0.017)", fontsize=10)
    for i, r in enumerate(ranks):
        ax.text(r + 0.05, i, f"{r:.1f}", va="center", fontsize=7)
    ax.set_xlim(0, max(ranks) + 1.2)
    ax.grid(True, alpha=0.3, axis="x")

    # ── Right: Heatmap ─────────────────────────────────────────
    ax = axes[1]
    mean_df = pd.read_csv(
        PROJECT / "results/tables" / R_MAIN / "mean_metrics.csv")
    m_order = ["shaptopk", "powershap", "nsga2", "banzhaf", "minshap", "shapg"]
    m_labels = ["SHAP", "PShap", "NSGA2", "Banzhaf", "MinShap", "ShapG"]
    ds_order = ["spambase", "madelon", "isolet", "gisette", "credit_card_fraud"]
    ds_labels = ["Spambase", "Madelon", "ISOLET", "Gisette", "Credit-Card"]

    heatmap_data = np.zeros((len(ds_order), len(m_order)))
    for i, ds in enumerate(ds_order):
        for j, m in enumerate(m_order):
            row = mean_df[(mean_df.dataset == ds) & (mean_df.method == m)]
            heatmap_data[i, j] = (row["AUC@10"].values[0]
                                  if len(row) else 0)

    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto",
                   vmin=0.79, vmax=0.99)
    ax.set_xticks(range(len(m_order)))
    ax.set_xticklabels(m_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ds_order)))
    ax.set_yticklabels(ds_labels, fontsize=8)
    for i in range(len(ds_order)):
        for j in range(len(m_order)):
            val = heatmap_data[i, j]
            color = "white" if val < 0.90 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")
    ax.set_title("(b) AUC@10 heatmap", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("AUC@10", fontsize=8)

    fig.tight_layout()
    fp_out = OUT / "fig2_cd_heatmap.pdf"
    fig.savefig(fp_out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp_out.name}")


# ================================================================
# Figure 3: Pre-filter verification
# ================================================================
def make_fig3_prefilter():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    mofs_agg = pd.read_csv(
        PROJECT / "results/tables" / R_MOFS / "mofs_verify_agg.csv",
        skiprows=3,
        names=["dataset", "method", "hv_mean", "hv_std",
               "search_dim_mean", "search_dim_std",
               "runtime_s_mean", "runtime_s_std",
               "converged_gen_mean", "converged_gen_std"])

    ds_list = ["spambase", "madelon", "isolet", "gisette",
               "credit_card_fraud", "crimedata", "abalone"]
    ds_short = ["Spam", "Mad", "ISO", "Gis", "CC", "Cri", "Aba"]

    full_hv, filt_hv = [], []
    full_std, filt_std = [], []
    full_conv, filt_conv = [], []

    for ds in ds_list:
        fr = mofs_agg[(mofs_agg.dataset == ds) &
                      (mofs_agg.method == "nsga2_full")]
        pr = mofs_agg[(mofs_agg.dataset == ds) &
                      (mofs_agg.method == "nsga2_filtered")]
        full_hv.append(float(fr["hv_mean"].iloc[0]) if len(fr) else 0)
        full_std.append(float(fr["hv_std"].iloc[0]) if len(fr) else 0)
        filt_hv.append(float(pr["hv_mean"].iloc[0]) if len(pr) else 0)
        filt_std.append(float(pr["hv_std"].iloc[0]) if len(pr) else 0)
        full_conv.append(float(fr["converged_gen_mean"].iloc[0]) if len(fr) else 0)
        filt_conv.append(float(pr["converged_gen_mean"].iloc[0]) if len(pr) else 0)

    # ── (a) HV bar chart ───────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(ds_list))
    w = 0.35
    ax.bar(x - w / 2, full_hv, w, yerr=full_std,
           label="NSGA2-full", color="gray", edgecolor="black",
           linewidth=0.5, capsize=3)
    ax.bar(x + w / 2, filt_hv, w, yerr=filt_std,
           label="NSGA2-filt", color="#2ca02c", edgecolor="black",
           linewidth=0.5, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_short, fontsize=8)
    ax.set_ylabel("Hypervolume")
    ax.set_title("(a) HV: full vs filtered", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # ── (b) Convergence speed ──────────────────────────────────
    ax = axes[1]
    ax.bar(x - w / 2, full_conv, w, label="NSGA2-full",
           color="gray", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, filt_conv, w, label="NSGA2-filt",
           color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_short, fontsize=8)
    ax.set_ylabel("Converged gen (lower = faster)")
    ax.set_title("(b) Convergence speed", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fp_out = OUT / "fig3_prefilter.pdf"
    fig.savefig(fp_out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp_out.name}")


if __name__ == "__main__":
    make_fig1_overview()
    make_fig2_cd_heatmap()
    make_fig3_prefilter()
    print(f"\nAll figures saved to: {OUT}")
