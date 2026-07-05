"""
Madelon alpha–recall curve — how α affects MinPPI's ability to recover
known noise (probe) features on the Madelon dataset (§6, reviewer #8/#4).

Madelon has 500 original features: f1–f20 are real (informative),
f21–f500 are probes (pure noise).  After MI filtering to 100 features,
97 are probes and 3 are real.  We compute MinPPI at multiple α levels
per seed against this ground-truth probe set, and compare with MinShap
and SHAP (which are α-independent).

Data reuse
----------
- c2 ground truth (``results/logs/2026-06-28_15-28-58/c2_ground_truth/``):
  MinShap and SHAP per-seed predictions → α-independent, reused as-is.
- binary_vs_continuous (``results/logs/2026-06-27_19-51-48/binary_vs_continuous/``):
  MinPPI at α ∈ {0.85, 0.90, 0.95, 0.99} for seed 42 → reused.
- **Computed fresh**: MinPPI at α ∈ {0.85, 0.90, 0.99} for seeds 123, 456,
  789, 1024 (α=0.95 already covered by c2).

Output (``results/tables/<run_id>/``)::

    madelon_alpha_recall.csv          # per-seed, per-method, per-α P/R/F₁
    madelon_alpha_recall_agg.csv      # mean ± std
    madelon_alpha_recall.pdf          # recall vs α line plot

Run::

    python -u experiments/madelon_alpha_recall.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minppi_mofs.min_ppi import estimate_minppi
from minppi_mofs.shapg_integration import build_guidance_from_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA_VALUES: list[float] = [0.85, 0.90, 0.95, 0.99]
SEEDS: list[int] = [42, 123, 456, 789, 1024]

# Madelon ground truth: f1–f20 = real, f21–f500 = probes
MADELON_PROBES = {f"f{i}" for i in range(21, 501)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _prf(predicted: set[str], ground_truth: set[str]) -> tuple[float, float, float]:
    """Precision, recall, F₁ of *predicted* against *ground_truth*."""
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0
    tp = len(predicted & ground_truth)
    prec = tp / len(predicted)
    rec = tp / len(ground_truth)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Madelon alpha–recall curve for MinPPI/MinShap/SHAP"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: results/tables/<run_id>)",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds to use")
    ap.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=ALPHA_VALUES,
        help="Alpha values to evaluate MinPPI at",
    )
    ap.add_argument(
        "--skip-minppi",
        action="store_true",
        help="Skip fresh MinPPI computation (use only cached data)",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    git_root = project_root.parent
    processed_dir = project_root / "data" / "processed"
    bvc_dir = (
        project_root
        / "results"
        / "logs"
        / "2026-06-27_19-51-48"
        / "binary_vs_continuous"
    )
    c2_dir = (
        project_root / "results" / "logs" / "2026-06-28_15-28-58" / "c2_ground_truth"
    )

    run_id = _now_stamp()
    out_dir = args.out or (project_root / "results" / "tables" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_commit(git_root)

    def _header(msg: str) -> None:
        print("\n" + "=" * 72, flush=True)
        print(f"  {msg}", flush=True)
        print("=" * 72, flush=True)

    _header(f"Madelon Alpha–Recall Curve — {run_id}")
    print(f"Commit: {commit}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Alphas: {args.alpha}", flush=True)

    # Metadata
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "alpha_values": args.alpha,
        "data_sources": {
            "c2_ground_truth": str(c2_dir),
            "bvc_per_alpha": str(bvc_dir),
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Ground truth labels for the MI-100 features
    train_s42 = pd.read_csv(processed_dir / "madelon_s42_train.csv")
    all_features = [c for c in train_s42.columns if c != "label"]
    gt_probes = set(all_features) & MADELON_PROBES
    print(
        f"\n  Ground truth: {len(gt_probes)} probes in MI-100 ({len(all_features)} features)"
    )
    print(f"  Real features: {sorted(set(all_features) - gt_probes)}")

    # ==================================================================
    # Step 1: MinShap & SHAP from c2 (α-independent, reuse existing)
    # ==================================================================
    _header("Step 1: Loading MinShap & SHAP from c2 (α-independent)")

    c2_csv = c2_dir / "c2_madelon_ground_truth.csv"
    if not c2_csv.exists():
        print(f"  ERROR: {c2_csv} not found — run eval_harmful_utility.py first")
        sys.exit(1)

    c2_df = pd.read_csv(c2_csv)

    # Per-seed harmful sets for MinShap/SHAP (α-independent, reuse from α=0.95 runs)
    minshap_per_seed: dict[int, set[str]] = {}
    shap_per_seed: dict[int, set[str]] = {}
    for seed in args.seeds:
        sub = c2_df[(c2_df["method"] == "minshap") & (c2_df["seed"] == seed)]
        if len(sub) == 0:
            print(f"  WARNING: no c2 data for MinShap seed {seed}, skipping")
            continue
        # We need the actual predicted feature sets, not just counts.
        # The c2 CSV only stores aggregate counts, not per-feature labels.
        # Fallback: re-read the minshap importance scores from the original
        # eval_harmful_utility run, or recompute.
        pass

    # The c2 CSV only stores n_predicted, precision, recall, f1 — not the actual
    # feature sets.  We recompute MinShap/SHAP once per seed (cheap, ~30s each).
    print(
        "  c2 CSV has aggregate stats only; recomputing MinShap/SHAP per seed (~2 min)"
    )

    from minppi_mofs.baselines.min_shap import compute_minshap_importance
    from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance

    for seed in args.seeds:
        train_df = pd.read_csv(processed_dir / f"madelon_s{seed}_train.csv")
        X_train = train_df.drop(columns=["label"])
        y_train = train_df["label"].values

        ms_scores = compute_minshap_importance(
            X_train,
            y_train,
            n_coalitions=100,
            per_coalition=5,
            seed=seed,
            task="binary",
        )
        # MinShap: use all negative-scored features as harmful set
        minshap_per_seed[seed] = {f for f, v in ms_scores.items() if v < 0}
        if not minshap_per_seed[seed]:
            ranked = sorted(ms_scores.items(), key=lambda kv: kv[1])
            k = max(1, len(minshap_per_seed.get(42, set())) // 5)
            minshap_per_seed[seed] = {f for f, _ in ranked[:k]}

        sh_scores = compute_shap_topk_importance(
            X_train, y_train, seed=seed, task="binary"
        )
        # SHAP: use bottom-15 (≈MinPPI mean harmful count across α)
        ranked_sh = sorted(sh_scores.items(), key=lambda kv: kv[1])
        k_shap = max(1, 15)
        shap_per_seed[seed] = {f for f, _ in ranked_sh[:k_shap]}

        p_ms, r_ms, f1_ms = _prf(minshap_per_seed[seed], gt_probes)
        p_sh, r_sh, f1_sh = _prf(shap_per_seed[seed], gt_probes)
        print(
            f"  s{seed}: MinShap={len(minshap_per_seed[seed]):3d} P={p_ms:.3f} R={r_ms:.3f}  "
            f"SHAP={len(shap_per_seed[seed]):3d} P={p_sh:.3f} R={r_sh:.3f}"
        )

    # ==================================================================
    # Step 2: MinPPI at each α per seed
    # ==================================================================
    _header("Step 2: MinPPI per α per seed")

    minppi_per_seed_alpha: dict[tuple[int, float], set[str]] = {}

    for seed in args.seeds:
        print(f"\n  -- Seed {seed} --", flush=True)
        train_df = pd.read_csv(processed_dir / f"madelon_s{seed}_train.csv")
        val_df = pd.read_csv(processed_dir / f"madelon_s{seed}_val.csv")
        X_train = train_df.drop(columns=["label"])
        y_train = train_df["label"].values
        X_val = val_df.drop(columns=["label"])
        y_val = val_df["label"].values

        # ShapG neighbors (same for all α, computed once per seed)
        sg_scores, sg_neighbors = build_guidance_from_data(
            X_train,
            y_train,
            depth=1,
            m=5,
            seed=seed,
        )

        for alpha in args.alpha:
            key = (seed, alpha)

            # Check if already cached from bvc (seed 42 at all α)
            if seed == 42:
                bvc_file = bvc_dir / f"madelon_alpha{int(alpha * 100):02d}_minppi.csv"
                if bvc_file.exists():
                    bvc_minppi = pd.read_csv(bvc_file)
                    harmful_set = set(
                        bvc_minppi[bvc_minppi["minppi"] == -1]["feature"].values
                    )
                    minppi_per_seed_alpha[key] = harmful_set
                    p, r, f1 = _prf(harmful_set, gt_probes)
                    print(
                        f"    α={alpha:.2f} [bvc]: {len(harmful_set):3d} harmful  "
                        f"P={p:.3f} R={r:.3f} F1={f1:.3f}",
                        flush=True,
                    )
                    continue

            if args.skip_minppi:
                continue

            t0 = time.perf_counter()
            minppi = estimate_minppi(
                X_train,
                X_val,
                y_train,
                y_val,
                alpha=alpha,
                method="shapg_guided",
                K_adv=20,
                shapg_neighbors=sg_neighbors,
                shapg_scores=sg_scores,
                seed=seed,
                verbose=False,
                task="binary",
            )
            elapsed = time.perf_counter() - t0
            harmful_set = {f for f, v in minppi.items() if v == -1}
            minppi_per_seed_alpha[key] = harmful_set
            p, r, f1 = _prf(harmful_set, gt_probes)
            print(
                f"    α={alpha:.2f}: {len(harmful_set):3d} harmful  "
                f"P={p:.3f} R={r:.3f} F1={f1:.3f}  ({elapsed:.0f}s)",
                flush=True,
            )

    # ==================================================================
    # Step 3: Aggregate and save
    # ==================================================================
    _header("Step 3: Aggregating results")

    rows: list[dict] = []
    for seed in args.seeds:
        # MinShap & SHAP (α-independent)
        if seed in minshap_per_seed:
            p, r, f1 = _prf(minshap_per_seed[seed], gt_probes)
            rows.append(
                {
                    "dataset": "madelon",
                    "seed": seed,
                    "alpha": None,
                    "method": "minshap",
                    "n_predicted": len(minshap_per_seed[seed]),
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                }
            )
        if seed in shap_per_seed:
            p, r, f1 = _prf(shap_per_seed[seed], gt_probes)
            rows.append(
                {
                    "dataset": "madelon",
                    "seed": seed,
                    "alpha": None,
                    "method": "shap",
                    "n_predicted": len(shap_per_seed[seed]),
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                }
            )
        # MinPPI per α
        for alpha in args.alpha:
            key = (seed, alpha)
            if key in minppi_per_seed_alpha:
                p, r, f1 = _prf(minppi_per_seed_alpha[key], gt_probes)
                rows.append(
                    {
                        "dataset": "madelon",
                        "seed": seed,
                        "alpha": alpha,
                        "method": "minppi",
                        "n_predicted": len(minppi_per_seed_alpha[key]),
                        "precision": p,
                        "recall": r,
                        "f1": f1,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "madelon_alpha_recall.csv", index=False)
    print(f"\n  madelon_alpha_recall.csv saved ({len(df)} rows)", flush=True)

    # Aggregate over seeds
    agg = (
        df.groupby(["method", "alpha"])[["n_predicted", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(out_dir / "madelon_alpha_recall_agg.csv")
    print("  madelon_alpha_recall_agg.csv saved", flush=True)
    print(agg.to_string(), flush=True)

    # ==================================================================
    # Step 4: Plot
    # ==================================================================
    _header("Step 4: Plotting")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Left: recall vs α (MinPPI only, with error bars over seeds)
    ppi = df[df["method"] == "minppi"].copy()
    if len(ppi) > 0:
        grp = ppi.groupby("alpha")[["precision", "recall", "f1"]]
        mean_r = grp.mean()
        std_r = grp.std()
        alphas = mean_r.index.values
        ax1.errorbar(
            alphas,
            mean_r["recall"],
            yerr=std_r["recall"],
            fmt="s-",
            color="#2ca02c",
            capsize=4,
            lw=1.5,
            ms=6,
            label="MinPPI recall",
        )
        ax1.errorbar(
            alphas,
            mean_r["precision"],
            yerr=std_r["precision"],
            fmt="o--",
            color="#1f77b4",
            capsize=4,
            lw=1.2,
            ms=5,
            label="MinPPI precision",
        )
        ax1.set_xlabel(r"$\alpha$")
        ax1.set_ylabel("Score")
        ax1.set_ylim(-0.05, 1.15)
        ax1.legend(fontsize=8, loc="upper left", ncol=2)
        ax1.set_title(r"(a) MinPPI vs. $\alpha$", fontsize=10)
        ax1.grid(True, alpha=0.3)

    # Right: F₁ bar chart — MinPPI at each α + MinShap + SHAP
    methods_order = [("minppi", a) for a in args.alpha] + [
        ("minshap", None),
        ("shap", None),
    ]
    labels = [
        f"PPI\n{a:.2f}" if m == "minppi" else m.capitalize() for m, a in methods_order
    ]
    f1_means = []
    f1_stds = []
    for method, alpha in methods_order:
        sub = df[(df["method"] == method)]
        if alpha is not None:
            sub = sub[sub["alpha"] == alpha]
        f1_means.append(sub["f1"].mean())
        f1_stds.append(sub["f1"].std())

    x = np.arange(len(labels))
    ax2.bar(
        x,
        f1_means,
        yerr=f1_stds,
        capsize=3,
        color=["#2ca02c"] * len(args.alpha) + ["#d62728", "#7f7f7f"],
        edgecolor="black",
        linewidth=0.4,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("F$_1$ score")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) F$_1$ comparison", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    # Highlight MinShap bar
    ms_idx = len(args.alpha)
    ax2.annotate(
        f"F$_1$={f1_means[ms_idx]:.2f}",
        xy=(ms_idx, f1_means[ms_idx]),
        xytext=(ms_idx + 0.4, f1_means[ms_idx] + 0.08),
        fontsize=7,
        color="#d62728",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "gray", "linewidth": 0.6},
    )

    fig.tight_layout()
    plot_path = out_dir / "madelon_alpha_recall.pdf"
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path.name}", flush=True)

    # ==================================================================
    # Done
    # ==================================================================
    _header("Done")
    print(f"Output directory: {out_dir}", flush=True)
    print("\nLaunch examples:", flush=True)
    print("  # Full run (all seeds, all alpha):", flush=True)
    print("  python -u experiments/madelon_alpha_recall.py", flush=True)
    print(
        "\n  # Quick test (seed 42 only, skip fresh MinPPI → use bvc cache):",
        flush=True,
    )
    print(
        "  python experiments/madelon_alpha_recall.py --seeds 42 --skip-minppi",
        flush=True,
    )
    print("\n  # Single alpha (seed 42):", flush=True)
    print(
        "  python experiments/madelon_alpha_recall.py --seeds 42 --alpha 0.90",
        flush=True,
    )


if __name__ == "__main__":
    main()
