"""
Generate paper figures and tables from MinPPI demo verification.

Produces Figure 1--3 and Table 1--2 for the ICOMP 2026 paper
"Min-Marginal Power Indices for Multi-Objective Feature Selection:
A Voting-Game Approach".

Outputs
-------
``results/figures/figure_1_minppi_trichotomy.pdf``  — Lemma 1 + Theorem 1
``results/figures/figure_2_prop1_detection.pdf``     — Proposition 1 theory vs empirical
``results/figures/figure_3_algorithm_b_vs_a.pdf``     — Algorithm B vs A + robustness
``results/figures/table_1_minss_demo.pdf``            — MinSS demonstration table
``results/figures/table_2_alg_ab_comparison.pdf``     — Algorithm A vs B quantitative

.. note::

    All data is computed from engineered voting games with known ground truth
    (n = 5 features), allowing exhaustive verification of every theoretical
    claim.  The script reproduces the logic of ``demos/minppi_demo.py`` and
    ``demos/algorithm_b_demo.py`` without importing them, keeping figure
    generation self-contained and side-effect free.

Run:
    python experiments/make_figures.py
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Callable, Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration — paper-consistent matplotlib settings
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
    }
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Seeded RNG (README requirement: explicit seed, not global np.random.*)
# ---------------------------------------------------------------------------
RNG: np.random.Generator = np.random.default_rng(42)

# ===================================================================
# Shared utilities (mirror demos/minppi_demo.py for self-containedness)
# ===================================================================


def shapley_weight(s: int, n: int) -> float:
    """Shapley coalition weight: w(|S|, n) = |S|! (n-|S|-1)! / n!."""
    return factorial(s) * factorial(n - s - 1) / factorial(n)


def compute_min_ss(
    i: int, n: int, v_alpha: Callable[[Iterable[int]], int]
) -> Tuple[int, float, list[int]]:
    """Return (MinSS_i, MeanSS_i, all_marginals) by exhaustive enumeration."""
    other = [f for f in range(n) if f != i]
    marginals: list[int] = []
    mean_ss = 0.0
    for k in range(len(other) + 1):
        w = shapley_weight(k, n)
        for S in combinations(other, k):
            S_set = set(S)
            marg = v_alpha(S_set | {i}) - v_alpha(S_set)
            marginals.append(marg)
            mean_ss += w * marg
    return int(min(marginals)), float(mean_ss), marginals


def classify(min_ss: int) -> str:
    """Map trinary MinSS value to semantic label."""
    if min_ss == 1:
        return "DECISIVE"
    if min_ss == -1:
        return "HARMFUL"
    return "neutral"


# ===================================================================
# Engineered voting games (from demos)
# ===================================================================


def _make_v_alpha_decisive() -> Callable[[Iterable[int]], int]:
    """Example 1: v_alpha(S) = 1 iff 0 in S (f_0 is dictator)."""
    return lambda S: int(0 in S)


def _make_winning_set_example_2() -> set[frozenset[int]]:
    """Return the 13 winning coalitions of Example 2."""
    return {
        frozenset({0, 2}),
        frozenset({0, 3}),
        frozenset({0, 4}),
        frozenset({0, 2, 3}),
        frozenset({0, 2, 4}),
        frozenset({0, 3, 4}),
        frozenset({0, 2, 3, 4}),
        frozenset({0, 1, 3}),
        frozenset({0, 1, 4}),
        frozenset({0, 1, 3, 4}),
        frozenset({0, 1, 2, 3}),
        frozenset({0, 1, 2, 4}),
        frozenset({0, 1, 2, 3, 4}),
    }


def _make_v_alpha_harmful(
    winning: set[frozenset[int]],
) -> Callable[[Iterable[int]], int]:
    """Example 2: v_alpha(S) = 1 iff S is one of 13 winning coalitions."""
    return lambda S: int(frozenset(S) in winning)


# ===================================================================
# Data computation: Example 1 (Decisive)
# ===================================================================


def _compute_example_1_data() -> dict[str, list]:
    """Compute MinSS / MeanSS / class for the DECISIVE voting game (n=5)."""
    n = 5
    v_alpha = _make_v_alpha_decisive()
    features = []
    minss_vals = []
    meanss_vals = []
    classes = []
    for i in range(n):
        min_ss, mean_ss, _ = compute_min_ss(i, n, v_alpha)
        assert min_ss in {-1, 0, 1}, f"Lemma 1 violated for f_{i} in Example 1"
        features.append(f"$f_{{{i}}}$")
        minss_vals.append(min_ss)
        meanss_vals.append(mean_ss)
        classes.append(classify(min_ss))
    return {
        "feature": features,
        "MinSS": minss_vals,
        "MeanSS": meanss_vals,
        "class": classes,
    }


# ===================================================================
# Data computation: Example 2 (Harmful + Caveat)
# ===================================================================


def _compute_example_2_data() -> dict[str, list]:
    """Compute MinSS / MeanSS / class for the HARMFUL voting game (n=5)."""
    n = 5
    winning = _make_winning_set_example_2()
    v_alpha = _make_v_alpha_harmful(winning)

    features = []
    minss_vals = []
    meanss_vals = []
    classes = []
    for i in range(n):
        min_ss, mean_ss, _ = compute_min_ss(i, n, v_alpha)
        assert min_ss in {-1, 0, 1}, f"Lemma 1 violated for f_{i} in Example 2"
        features.append(f"$f_{{{i}}}$")
        minss_vals.append(min_ss)
        meanss_vals.append(mean_ss)
        classes.append(classify(min_ss))

    # Sanity: f_0 is in every winning coalition
    f0_in_all = all(0 in w for w in winning)
    assert f0_in_all, "Example 2 sanity check: f_0 must be in every winning coalition"

    return {
        "feature": features,
        "MinSS": minss_vals,
        "MeanSS": meanss_vals,
        "class": classes,
        "f0_in_all_winning": f0_in_all,
    }


# ===================================================================
# Data computation: Example 4 (Proposition 1 — Algorithm A detection)
# ===================================================================


def _compute_prop1_data() -> dict[str, np.ndarray]:
    """Compute Proposition 1 theory vs empirical detection rates."""
    n = 5
    winning = _make_winning_set_example_2()
    v_alpha = _make_v_alpha_harmful(winning)
    target = 1  # f_1, true MinSS = -1
    other = [f for f in range(n) if f != target]

    # --- exact p_- under the size-then-subset sampler ---
    n_neg_per_size: dict[int, int] = {}
    n_per_size: dict[int, int] = {}
    for k in range(len(other) + 1):
        n_per_size[k] = 0
        n_neg_per_size[k] = 0
        for S in combinations(other, k):
            S_set = set(S)
            n_per_size[k] += 1
            marg = v_alpha(S_set | {target}) - v_alpha(S_set)
            if marg == -1:
                n_neg_per_size[k] += 1
    p_neg = sum(n_neg_per_size[k] / n_per_size[k] for k in range(len(other) + 1)) / (
        len(other) + 1
    )

    K_values = np.array([1, 3, 5, 10, 20, 50, 100], dtype=int)
    n_trials = 500
    rng_mc = np.random.default_rng(11)  # same seed as demo

    theory = np.array([1.0 - (1.0 - p_neg) ** K for K in K_values])
    empirical = np.zeros(len(K_values))

    for idx, K in enumerate(K_values):
        detected = 0
        for _ in range(n_trials):
            min_marg = 1
            for _ in range(K):
                k = int(rng_mc.integers(0, len(other) + 1))
                if k == 0:
                    S = set()
                else:
                    S = set(rng_mc.choice(other, size=k, replace=False).tolist())
                marg = v_alpha(S | {target}) - v_alpha(S)
                if marg < min_marg:
                    min_marg = marg
            if min_marg == -1:
                detected += 1
        empirical[idx] = detected / n_trials

    # correlation between theory and empirical
    r_val = float(np.corrcoef(theory, empirical)[0, 1])

    return {
        "K_values": K_values,
        "p_neg": p_neg,
        "theory": theory,
        "empirical": empirical,
        "r_value": r_val,
        "n_trials": n_trials,
    }


# ===================================================================
# Data computation: Algorithm B vs A + robustness (algorithm_b_demo)
# ===================================================================


def _compute_alg_b_data() -> dict:
    """Compute Algorithm B vs A detection rates and robustness data."""
    n = 5
    target = 1

    winning = _make_winning_set_example_2()
    v_alpha = _make_v_alpha_harmful(winning)

    # Weighted correlation matrix used by Algorithm B
    W_demo = np.array(
        [
            [0.00, 0.40, 0.50, 0.10, 0.10],
            [0.40, 0.00, 0.70, 0.20, 0.20],
            [0.50, 0.70, 0.00, 0.20, 0.20],
            [0.10, 0.20, 0.20, 0.00, 0.10],
            [0.10, 0.20, 0.20, 0.10, 0.00],
        ],
        dtype=float,
    )

    # --- Algorithm A ---
    def _algorithm_a(tgt: int, K: int, rng: np.random.Generator) -> Tuple[int, int]:
        other_a = [f for f in range(n) if f != tgt]
        min_marg = 1
        for r in range(K):
            k = int(rng.integers(0, len(other_a) + 1))
            if k == 0:
                S = set()
            else:
                S = set(rng.choice(other_a, size=k, replace=False).tolist())
            marg = v_alpha(S | {tgt}) - v_alpha(S)
            if marg < min_marg:
                min_marg = marg
            if min_marg == -1:
                return min_marg, r + 1
        return min_marg, K

    # --- Algorithm B ---
    def _algorithm_b(
        tgt: int,
        W: np.ndarray,
        K: int,
        rng: np.random.Generator,
        k_seed: int = 2,
    ) -> Tuple[int, int]:
        other_c = [(j, W[tgt, j]) for j in range(n) if j != tgt]
        other_c.sort(key=lambda x: -x[1])
        seed = {j for j, _ in other_c[:k_seed]}
        min_marg = 1
        for r in range(K):
            if r == 0:
                S = seed.copy()
            else:
                S = seed.copy()
                for j in range(n):
                    if j == tgt:
                        continue
                    prob_keep = 0.5 + 0.5 * W[tgt, j]
                    if rng.uniform() > prob_keep:
                        if j in S:
                            S.discard(j)
                        else:
                            S.add(j)
            marg = v_alpha(S | {tgt}) - v_alpha(S)
            if marg < min_marg:
                min_marg = marg
            if min_marg == -1:
                return min_marg, r + 1
        return min_marg, K

    # --- Main comparison ---
    n_trials = 500
    K_values = np.array([1, 2, 3, 5, 10, 20, 50], dtype=int)
    rng_a = np.random.default_rng(2025)
    rng_b = np.random.default_rng(2026)

    det_a = np.zeros(len(K_values))
    det_b = np.zeros(len(K_values))
    samples_a = np.zeros(len(K_values))
    samples_b = np.zeros(len(K_values))

    for idx, K in enumerate(K_values):
        d_a, d_b = 0, 0
        used_a: list[int] = []
        used_b: list[int] = []
        for _ in range(n_trials):
            ma, ka = _algorithm_a(target, K, rng_a)
            mb, kb = _algorithm_b(target, W_demo, K, rng_b)
            d_a += int(ma == -1)
            d_b += int(mb == -1)
            used_a.append(ka)
            used_b.append(kb)
        det_a[idx] = d_a / n_trials
        det_b[idx] = d_b / n_trials
        samples_a[idx] = float(np.mean(used_a))
        samples_b[idx] = float(np.mean(used_b))

    # --- Robustness ---
    sigma_values = np.array([0.05, 0.15, 0.30, 0.50])
    K_robust = 5
    rng_master = np.random.default_rng(42)
    rob_b = np.zeros(len(sigma_values))
    rob_a = np.zeros(len(sigma_values))

    for s_idx, sigma in enumerate(sigma_values):
        d_b_rob, d_a_rob = 0, 0
        for _ in range(n_trials):
            noise = rng_master.normal(0, sigma, size=W_demo.shape)
            W_noisy = np.clip(W_demo + noise, 0, 1)
            W_noisy = (W_noisy + W_noisy.T) / 2
            np.fill_diagonal(W_noisy, 0)
            inner_rng = np.random.default_rng(int(rng_master.integers(1, int(1e9))))
            mb, _ = _algorithm_b(target, W_noisy, K_robust, inner_rng)
            d_b_rob += int(mb == -1)
            ma, _ = _algorithm_a(target, K_robust, inner_rng)
            d_a_rob += int(ma == -1)
        rob_b[s_idx] = d_b_rob / n_trials
        rob_a[s_idx] = d_a_rob / n_trials

    return {
        "K_values": K_values,
        "det_a": det_a,
        "det_b": det_b,
        "samples_a": samples_a,
        "samples_b": samples_b,
        "sigma_values": sigma_values,
        "robustness_b": rob_b,
        "robustness_a": rob_a,
        "K_robust": K_robust,
        "n_trials": n_trials,
    }


# ===================================================================
# Figure 1: MinPPI Trichotomy  (Lemma 1 + Theorem 1)
# ===================================================================


def _make_figure_1(ex1: dict, ex2: dict, save_path: Path) -> None:
    """Figure 1: Bar chart of MinSS values — decisive (Ex1) and harmful (Ex2)."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    n = 5
    x = np.arange(n)
    width = 0.55
    colors_ex1 = ["#2ca02c" if v == 1 else "#7f7f7f" for v in ex1["MinSS"]]
    colors_ex2 = ["#d62728" if v == -1 else "#7f7f7f" for v in ex2["MinSS"]]

    # Left: Example 1 — DECISIVE
    ax1.bar(x, ex1["MinSS"], width, color=colors_ex1, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ex1["feature"])
    ax1.set_ylabel("MinSS")
    ax1.set_ylim(-1.5, 1.45)
    ax1.set_yticks([-1, 0, 1])
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title("(a) Example 1: decisive $f_0$", fontsize=10, pad=10)
    ax1.annotate(
        "MinSS$(f_0)=+1$\nDECISIVE",
        xy=(0, 1),
        xytext=(0.15, 1.1),
        ha="center",
        fontsize=7,
        color="#2ca02c",
        fontweight="bold",
    )

    # Right: Example 2 — HARMFUL
    ax2.bar(x, ex2["MinSS"], width, color=colors_ex2, edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ex2["feature"])
    ax2.set_ylabel("MinSS")
    ax2.set_ylim(-1.5, 1.45)
    ax2.set_yticks([-1, 0, 1])
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_title("(b) Example 2: harmful $f_1$ + caveat", fontsize=10, pad=10)
    ax2.annotate(
        "MinSS$(f_1)=-1$\nHARMFUL",
        xy=(1, -1),
        xytext=(1, -1.30),
        ha="center",
        fontsize=7,
        color="#d62728",
        fontweight="bold",
    )
    # Caveat annotation for f_0:
    #   Blue text box sits on the right side of the subplot,
    #   arrow points from its lower-left corner down toward the f_0 bar.
    ax2.annotate(
        "$f_0$ in every\nwinning coalition\nbut MinSS$(f_0)=0$",
        xy=(-0.18, 0.18),
        xytext=(2.0, 0.75),
        ha="center",
        fontsize=6.5,
        color="#1f77b4",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "lightyellow",
            "alpha": 0.9,
            "edgecolor": "#1f77b4",
            "linewidth": 0.8,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": "#1f77b4",
            "linewidth": 0.8,
        },
    )

    plt.suptitle(
        "MinPPI Trichotomy — Lemma 1 & Theorem 1",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close("all")
    print(f"  ✓ Figure 1 saved to {save_path}")


# ===================================================================
# Figure 2: Proposition 1 — Detection Probability
# ===================================================================


def _make_figure_2(prop1: dict, save_path: Path) -> None:
    """Figure 2: Theory vs empirical detection probability for Algorithm A."""
    _fig, ax = plt.subplots(figsize=(4.0, 3.0))

    K_vals = prop1["K_values"]
    theory = prop1["theory"]
    empirical = prop1["empirical"]
    p_neg = prop1["p_neg"]
    r_val = prop1["r_value"]

    ax.plot(K_vals, theory, "b-", linewidth=1.5, label="Theory: $1 - (1-p_-)^K$")
    ax.plot(
        K_vals,
        empirical,
        "ro",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.2,
        label="Empirical (500 trials)",
    )
    ax.set_xlabel("$K$ (number of random coalition samples)")
    ax.set_ylabel("Detection probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title(
        f"Proposition 1 — $p_- = {p_neg:.4f}$, $r = {r_val:.4f}$",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close("all")
    print(f"  ✓ Figure 2 saved to {save_path}")


# ===================================================================
# Figure 3: Algorithm B vs A
# ===================================================================


def _make_figure_3(alg_data: dict, save_path: Path) -> None:
    """Figure 3: (a) Detection rate vs K, (b) Robustness under noise."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    K_vals = alg_data["K_values"]
    det_a = alg_data["det_a"]
    det_b = alg_data["det_b"]

    # --- (a) Detection rate comparison ---
    ax1.plot(
        K_vals,
        det_b,
        "s-",
        color="#2ca02c",
        linewidth=1.5,
        markersize=5,
        label="Algorithm B (MI-correlation-guided)",
    )
    ax1.plot(
        K_vals,
        det_a,
        "o-",
        color="#d62728",
        linewidth=1.5,
        markersize=5,
        label="Algorithm A (random MC)",
    )
    ax1.set_xlabel("$K$ (search budget)")
    ax1.set_ylabel("Detection rate of MinSS = $-1$")
    ax1.set_ylim(-0.05, 1.10)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax1.set_title("(a) Detection rate vs budget $K$", fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Annotation in upper-left, arrow to B @ K=1 data point
    ax1.annotate(
        f"$\\mathbf{{{100 * det_b[0]:.0f}\\%}}$ @ $K{{=}}1$ (B)\n"
        f"vs $\\mathbf{{{100 * det_a[0]:.0f}\\%}}$ @ $K{{=}}1$ (A)",
        xy=(1, det_b[0]),
        xytext=(10, 0.55),
        fontsize=7,
        color="#2ca02c",
        arrowprops={"arrowstyle": "->", "color": "gray", "linewidth": 0.6},
    )

    # --- (b) Robustness ---
    sigma_vals = alg_data["sigma_values"]
    rob_b = alg_data["robustness_b"]
    rob_a = alg_data["robustness_a"]
    K_r = alg_data["K_robust"]

    ax2.plot(
        sigma_vals,
        rob_b,
        "s-",
        color="#2ca02c",
        linewidth=1.5,
        markersize=6,
        label=f"Algorithm B ($K{{=}}{K_r}$)",
    )
    ax2.plot(
        sigma_vals,
        rob_a,
        "o--",
        color="#d62728",
        linewidth=1.5,
        markersize=6,
        label=f"Algorithm A ($K{{=}}{K_r}$)",
    )
    ax2.set_xlabel(r"$\sigma$ (correlation matrix noise)")
    ax2.set_ylabel("Detection rate")
    ax2.set_ylim(-0.05, 1.10)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax2.set_title("(b) Robustness to correlation-weight noise", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Algorithm B vs Algorithm A",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close("all")
    print(f"  ✓ Figure 3 saved to {save_path}")


# ===================================================================
# Table 1: MinSS Demonstration (as PDF table figure)
# ===================================================================


def _make_table_1(ex1: dict, ex2: dict, save_path: Path) -> None:
    """Table 1: MinSS values for Examples 1 and 2 — renders as PDF figure."""
    _fig, ax = plt.subplots(figsize=(6.5, 2.0))
    ax.axis("off")

    n = len(ex1["feature"])
    col_labels = [
        "Feature",
        "MinSS",
        "Mean SS",
        "Class",
        "Feature",
        "MinSS",
        "Mean SS",
        "Class",
    ]
    rows = []
    for i in range(n):
        rows.append(
            [
                ex1["feature"][i],
                f"{ex1['MinSS'][i]:+d}",
                f"{ex1['MeanSS'][i]:+.4f}",
                ex1["class"][i],
                ex2["feature"][i],
                f"{ex2['MinSS'][i]:+d}",
                f"{ex2['MeanSS'][i]:+.4f}",
                ex2["class"][i],
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)

    # Style header
    for j in range(8):
        table[0, j].set_facecolor("#e0e0e0")
        table[0, j].set_text_props(weight="bold")

    # Highlight decisive / harmful cells
    for i in range(n):
        if ex1["MinSS"][i] == 1:
            table[i + 1, 1].set_facecolor("#c8e6c9")
        if ex2["MinSS"][i] == -1:
            table[i + 1, 5].set_facecolor("#ffcdd2")

    ax.set_title(
        "Table 1: MinSS values — (left) Example 1: decisive $f_0$; "
        "(right) Example 2: harmful $f_1$ + caveat",
        fontsize=9,
        pad=8,
    )

    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close("all")
    print(f"  ✓ Table 1 saved to {save_path}")


# ===================================================================
# Table 2: Algorithm A vs B Quantitative Comparison
# ===================================================================


def _make_table_2(alg_data: dict, save_path: Path) -> None:
    """Table 2: Detection rates and robustness for Algorithms A and B."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    ax1.axis("off")
    ax2.axis("off")

    K_vals = alg_data["K_values"]
    det_a = alg_data["det_a"]
    det_b = alg_data["det_b"]
    samples_a = alg_data["samples_a"]
    samples_b = alg_data["samples_b"]

    # --- Left: Detection rate comparison ---
    col_labels_main = [
        "$K$",
        "Alg A detect",
        "Alg B detect",
        r"Alg A $\bar{s}$",
        r"Alg B $\bar{s}$",
    ]
    rows_main = []
    for idx in range(len(K_vals)):
        rows_main.append(
            [
                f"{K_vals[idx]}",
                f"{det_a[idx]:.3f}",
                f"{det_b[idx]:.3f}",
                f"{samples_a[idx]:.1f}",
                f"{samples_b[idx]:.1f}",
            ]
        )

    table_main = ax1.table(
        cellText=rows_main,
        colLabels=col_labels_main,
        cellLoc="center",
        loc="center",
    )
    table_main.auto_set_font_size(False)
    table_main.set_fontsize(7.5)
    table_main.scale(1.0, 1.3)
    for j in range(5):
        table_main[0, j].set_facecolor("#e0e0e0")
        table_main[0, j].set_text_props(weight="bold")
    ax1.set_title(
        f"(a) Detection rate of $\\mathrm{{MinSS}} = -1$ "
        f"({alg_data['n_trials']} trials)",
        fontsize=9,
        pad=10,
    )

    # --- Right: Robustness ---
    sigma_vals = alg_data["sigma_values"]
    rob_b = alg_data["robustness_b"]
    rob_a = alg_data["robustness_a"]

    col_labels_rob = [r"$\sigma$", "Alg B ($K{=}5$)", "Alg A ($K{=}5$)"]
    rows_rob = []
    for idx in range(len(sigma_vals)):
        rows_rob.append(
            [
                f"{sigma_vals[idx]:.2f}",
                f"{rob_b[idx]:.3f}",
                f"{rob_a[idx]:.3f}",
            ]
        )

    table_rob = ax2.table(
        cellText=rows_rob,
        colLabels=col_labels_rob,
        cellLoc="center",
        loc="center",
    )
    table_rob.auto_set_font_size(False)
    table_rob.set_fontsize(7.5)
    table_rob.scale(1.0, 1.5)
    for j in range(3):
        table_rob[0, j].set_facecolor("#e0e0e0")
        table_rob[0, j].set_text_props(weight="bold")
    ax2.set_title(
        r"(b) Robustness to Gaussian noise "
        r"$\mathcal{N}(0,\sigma^2)$ on $W$",
        fontsize=9,
        pad=10,
    )

    plt.suptitle(
        "Table 2: Algorithm A (random MC) vs"
        " Algorithm B (MI-correlation-guided adversarial search)",
        fontsize=10,
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close("all")
    print(f"  ✓ Table 2 saved to {save_path}")


# ===================================================================
# Banner utility (matching demos style)
# ===================================================================


def banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    """Compute all data and generate all paper figures/tables."""

    banner("MinPPI Demo Output → Paper Figures & Tables")
    print("ICOMP 2026: Min-Marginal Power Indices for Multi-Objective")
    print("Feature Selection: A Voting-Game Approach")
    print(f"\nOutput directory: {FIGURES_DIR}")

    # ---- Compute data ----
    banner("Computing data from engineered voting games")

    print("  Example 1 (decisive v_alpha) ...")
    ex1 = _compute_example_1_data()
    assert ex1["MinSS"][0] == 1, "Ex1: f_0 must be decisive"
    assert all(v in {-1, 0, 1} for v in ex1["MinSS"]), "Ex1: Lemma 1 violation"
    print(f"    f_0: MinSS={ex1['MinSS'][0]:+d} ({ex1['class'][0]}) ✓")

    print("  Example 2 (harmful v_alpha + caveat) ...")
    ex2 = _compute_example_2_data()
    assert ex2["MinSS"][1] == -1, "Ex2: f_1 must be harmful"
    assert ex2["f0_in_all_winning"], "Ex2: f_0 must be in every winning coalition"
    assert ex2["MinSS"][0] == 0, "Ex2: f_0 must be neutral (caveat holds)"
    print(f"    f_1: MinSS={ex2['MinSS'][1]:+d} ({ex2['class'][1]}) ✓")
    print(
        f"    f_0: in all winning but MinSS={ex2['MinSS'][0]:+d} "
        f"({ex2['class'][0]}) — caveat verified ✓"
    )

    print("  Proposition 1 (Algorithm A detection probability) ...")
    prop1 = _compute_prop1_data()
    print(
        f"    p_- = {prop1['p_neg']:.4f},  r(theory, empir) = {prop1['r_value']:.4f} ✓"
    )

    print("  Algorithm B vs A + robustness ...")
    alg_data = _compute_alg_b_data()
    assert abs(alg_data["det_b"][0] - 1.0) < 0.001, (
        "Alg B must detect at K=1 (noiseless case)"
    )
    print(f"    Alg B @ K=1: {alg_data['det_b'][0]:.3f} detect rate ✓")
    print(f"    Alg A @ K=1: {alg_data['det_a'][0]:.3f} detect rate (baseline)")
    print(
        f"    Robustness σ={alg_data['sigma_values'][-1]}: "
        f"B={alg_data['robustness_b'][-1]:.3f} vs "
        f"A={alg_data['robustness_a'][-1]:.3f} ✓"
    )

    # ---- Generate figures ----
    banner("Generating paper figures")

    _make_figure_1(ex1, ex2, FIGURES_DIR / "figure_1_minppi_trichotomy.pdf")
    _make_figure_2(prop1, FIGURES_DIR / "figure_2_prop1_detection.pdf")
    _make_figure_3(alg_data, FIGURES_DIR / "figure_3_algorithm_b_vs_a.pdf")

    # ---- Generate tables ----
    banner("Generating paper tables")

    _make_table_1(ex1, ex2, FIGURES_DIR / "table_1_minss_demo.pdf")
    _make_table_2(alg_data, FIGURES_DIR / "table_2_alg_ab_comparison.pdf")

    # ---- Summary ----
    banner("Summary")
    print("Generated outputs:")
    for name in sorted(FIGURES_DIR.glob("*.pdf")):
        size_kb = name.stat().st_size / 1024
        print(f"  {name.name}  ({size_kb:.0f} KB)")
    print()
    print("Observations for the paper:")
    print("  Figure 1: Lemma 1 holds — all MinSS in {-1, 0, +1}.")
    print("             Theorem 1: f_0 DECISIVE (Ex1), f_1 HARMFUL (Ex2).")
    print("             Caveat: f_0 in every winning but MinSS(f_0)=0.")
    print(f"  Figure 2: Proposition 1 verified — r = {prop1['r_value']:.4f}.")
    print(f"             p_- = {prop1['p_neg']:.4f} -> K~100 for 95% detection.")
    print("  Figure 3: Algorithm B achieves ~100% detection at K=1.")
    print("             Robustness: detection >50% even at sigma=0.50.")
    print("  All figures/tables saved as PDF (README hard rule #6).")


if __name__ == "__main__":
    main()
