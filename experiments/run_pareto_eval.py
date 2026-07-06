"""
Alpha-Pareto front evaluation — Plan C main table.

For each (dataset, seed, alpha), this script:
  1. Computes MinPPI (Algorithm B for binary, Algorithm A for
     multiclass/regression).
  2. Records the harmful-feature set and compression rate.
  3. Trains an XGBoost baseline on all features → baseline AUC or R².
  4. Deletes harmful features, retrains on the remaining set →
     filtered AUC or R².
  5. Computes AUC preservation = AUC_filtered / AUC_full.

Scanning alpha in {0.85, 0.90, 0.95, 0.99} over 7 datasets x 5 seeds
produces the alpha-Pareto front in (compression, AUC preservation) space.

Output (results/tables/<run_id>/)::

    pareto_front.csv       # per-(dataset, seed, alpha)
    pareto_front_agg.csv   # mean over seeds
    pareto_front.pdf       # scatter + connected front per dataset

Run::

    python -u experiments/run_pareto_eval.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm

from minppi_mofs.min_ppi import estimate_minppi, estimate_minppi_random_mc
from minppi_mofs.shapg_integration import build_guidance_from_data

# ============================================================
# Constants
# ============================================================

ALPHA_VALUES: list[float] = [0.85, 0.90, 0.95, 0.99]
SEEDS: list[int] = [42, 123, 456, 789, 1024]

DATASETS: list[tuple[str, str, str]] = [
    ("spambase", "is_spam", "binary"),
    ("madelon", "label", "binary"),
    ("isolet", "letter", "multiclass"),
    ("gisette", "label", "binary"),
    ("credit_card_fraud", "Class", "binary"),
    ("crimedata", "nonViolPerPop", "regression"),
    ("abalone", "Rings", "regression"),
]

DATASET_BUDGET: dict[str, dict] = {
    "spambase": {"K_adv": 30},
    "madelon": {"K_adv": 20},
    "isolet": {"K": 30, "per_coalition": 2},  # multiclass × 26 → ultra-light
    "gisette": {"K_adv": 10},
    "credit_card_fraud": {"K_adv": 30},
    "crimedata": {"K": 40, "per_coalition": 2},  # regression
    "abalone": {"K": 40, "per_coalition": 2},  # regression, 8 feats
}

# ============================================================
# Helpers
# ============================================================


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


class _Tee:
    def __init__(self, log_path: Path) -> None:
        self._console = sys.stdout
        self._file = log_path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        self._console.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._console
        sys.stderr = self._console
        self._file.close()


def load_split(
    prefix: str,
    target: str,
    processed_dir: Path,
    *,
    seed: int,
) -> tuple[
    pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray
]:
    train = pd.read_csv(processed_dir / f"{prefix}_s{seed}_train.csv")
    val = pd.read_csv(processed_dir / f"{prefix}_s{seed}_val.csv")
    test = pd.read_csv(processed_dir / f"{prefix}_s{seed}_test.csv")
    return (
        train.drop(columns=[target]),
        train[target].values,
        val.drop(columns=[target]),
        val[target].values,
        test.drop(columns=[target]),
        test[target].values,
    )


def _budget_for(prefix: str) -> dict:
    return DATASET_BUDGET.get(prefix, {"K_adv": 20})


def train_and_eval(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    *,
    task: str = "binary",
    seed: int = 42,
) -> tuple[float, float]:
    t0 = time.perf_counter()
    n_classes = len(np.unique(y_te))

    if task == "regression":
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=seed,
        )
        model.fit(X_tr.values, y_tr)
        y_pred = model.predict(X_te.values)
        val = float(r2_score(y_te, y_pred))
        if val < 0.0:
            val = 0.0
    else:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=seed,
        )
        model.fit(X_tr.values, y_tr)
        y_prob = model.predict_proba(X_te.values)
        if n_classes == 2:
            val = float(roc_auc_score(y_te, y_prob[:, 1]))
        else:
            val = float(roc_auc_score(y_te, y_prob, multi_class="ovr"))
    return val, time.perf_counter() - t0


# ============================================================
# Main
# ============================================================


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    git_root = project_root.parent
    processed_dir = project_root / "data" / "processed"

    run_id = _now_stamp()
    run_dir = project_root / "results" / "logs" / run_id
    tables_dir = project_root / "results" / "tables" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tee = _Tee(run_dir / "output.log")
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    commit = _git_commit(git_root)

    def _header(msg: str) -> None:
        print("\n" + "=" * 72, flush=True)
        print(f"  {msg}", flush=True)
        print("=" * 72, flush=True)

    _header(f"alpha-Pareto Front — {run_id}")
    print(f"Commit: {commit}")

    metadata = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alpha_values": ALPHA_VALUES,
        "seeds": SEEDS,
        "datasets": [d[0] for d in DATASETS],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (tables_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    all_rows: list[dict] = []

    for prefix, target, task in tqdm(
        DATASETS, desc="alpha-Pareto", unit="ds", file=sys.stdout
    ):
        for seed in tqdm(
            SEEDS, desc=f"  seeds ({prefix})", unit="seed", file=sys.stdout, leave=False
        ):
            try:
                X_tr, y_tr, X_val, y_val, X_te, y_te = load_split(
                    prefix, target, processed_dir, seed=seed
                )
            except FileNotFoundError:
                print(f"\n  SKIP {prefix} s{seed}: missing split")
                continue

            n_full = X_tr.shape[1]
            n_classes = len(np.unique(y_val))
            metric_label = "R2" if task == "regression" else "AUC"

            base_util, base_time = train_and_eval(
                X_tr, y_tr, X_te, y_te, task=task, seed=seed
            )
            print(f"\n  {prefix} s{seed}: base {metric_label}={base_util:.4f}")

            sg_scores: dict = {}
            sg_neighbors = None
            if n_classes == 2 and task != "regression":
                sg_scores, sg_neighbors = build_guidance_from_data(
                    X_tr, y_tr, depth=1, m=5, seed=seed
                )

            for alpha in ALPHA_VALUES:
                t0 = time.perf_counter()
                budget = _budget_for(prefix)

                if n_classes > 2 or task == "regression":
                    minppi = estimate_minppi_random_mc(
                        X_tr,
                        X_val,
                        y_tr,
                        y_val,
                        alpha=alpha,
                        K=budget.get("K", 200),
                        shared_coalitions=True,
                        per_coalition=budget.get("per_coalition", 5),
                        seed=seed,
                        verbose=False,
                        task=task,
                    )
                else:
                    minppi = estimate_minppi(
                        X_tr,
                        X_val,
                        y_tr,
                        y_val,
                        alpha=alpha,
                        method="shapg_guided",
                        K_adv=budget["K_adv"],
                        shapg_neighbors=sg_neighbors,
                        shapg_scores=sg_scores,
                        seed=seed,
                        verbose=False,
                        task=task,
                    )

                harmful = {f for f, v in minppi.items() if v == -1}
                n_harmful = len(harmful)
                cr = n_harmful / n_full if n_full > 0 else 0.0
                t_minppi = time.perf_counter() - t0

                if n_harmful > 0:
                    keep = [c for c in X_tr.columns if c not in harmful]
                    filt_util, _ = train_and_eval(
                        X_tr[keep], y_tr, X_te[keep], y_te, task=task, seed=seed
                    )
                else:
                    filt_util, _ = base_util, base_time

                preserv = filt_util / base_util if base_util > 0 else 1.0

                row = {
                    "dataset": prefix,
                    "seed": seed,
                    "alpha": alpha,
                    "n_full": n_full,
                    "n_harmful": n_harmful,
                    "compression": round(cr, 4),
                    "base_util": round(base_util, 4),
                    "filtered_util": round(filt_util, 4),
                    "AUC" if task != "regression" else "R2": round(filt_util, 4),
                    "preservation": round(preserv, 4),
                    "time_minppi_s": round(t_minppi, 1),
                    "metric": metric_label,
                }
                all_rows.append(row)
                print(
                    f"    a={alpha:.2f}: harmful={n_harmful} ({cr * 100:.1f}%), "
                    f"{metric_label}={filt_util:.4f} (preserv={preserv:.4f}) "
                    f"({t_minppi:.1f}s)",
                    flush=True,
                )

    # ---- Save CSVs ----
    df = pd.DataFrame(all_rows)
    df.to_csv(tables_dir / "pareto_front.csv", index=False)
    print(f"\n  pareto_front.csv saved ({len(df)} rows)")

    num_cols = [
        c
        for c in df.columns
        if c
        in (
            "compression",
            "base_util",
            "filtered_util",
            "AUC",
            "R2",
            "preservation",
            "time_minppi_s",
        )
    ]
    agg = df.groupby(["dataset", "alpha"])[num_cols].agg(["mean", "std"]).round(4)
    agg.to_csv(tables_dir / "pareto_front_agg.csv")
    print("  pareto_front_agg.csv saved")

    for col_key in ["compression", "preservation"]:
        pivot = df.pivot_table(
            index="dataset", columns="alpha", values=col_key, aggfunc="mean"
        ).round(4)
        pivot.to_csv(tables_dir / f"table_{col_key}_vs_alpha.csv")

    # ---- Plot ----
    _header("Plotting")
    ds_names = sorted(df["dataset"].unique())
    alpha_colors = {0.85: "#d62728", 0.90: "#ff7f0e", 0.95: "#2ca02c", 0.99: "#1f77b4"}
    alpha_markers = {0.85: "s", 0.90: "D", 0.95: "o", 0.99: "^"}

    # Classification: 5 datasets (4 AUC + 1 multiclass/Accuracy)
    cls_ds = [d for d in ds_names if df[df["dataset"] == d]["metric"].iloc[0] == "AUC"]
    reg_ds = [d for d in ds_names if df[df["dataset"] == d]["metric"].iloc[0] == "R2"]

    # Build compression-count summary table (all 7 datasets × 4 alpha)
    # for embedding into the CLS figure at position (1,1).
    _harmful_pivot = (
        df.groupby(["dataset", "alpha"])["n_harmful"]
        .mean()
        .round(0)
        .astype(int)
        .unstack("alpha")
    )
    _n_full_per_ds = df.groupby("dataset")["n_full"].first().astype(int)
    _alpha_cols = sorted(ALPHA_VALUES)

    # --- Classification panel ---
    if cls_ds:
        isolet_ds = [d for d in cls_ds if d == "isolet"]
        auc_ds = sorted(d for d in cls_ds if d != "isolet")
        n_cols, n_rows = 3, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5, 6.0), squeeze=False)
        fig.suptitle(
            "Alpha-Pareto Front -- Classification"
            + chr(10)
            + "X = compression (% features removed),  "
            "Y = preservation (1.0 = full-model baseline)",
            fontsize=11,
            y=1.02,
        )

        grid_map = {}
        auc_positions = [(0, 0), (0, 1), (0, 2), (1, 0)]
        for ds, (r, c) in zip(auc_ds, auc_positions):
            grid_map[ds] = (r, c)
        if isolet_ds:
            grid_map[isolet_ds[0]] = (1, 2)

        for ds, (rr, cc) in grid_map.items():
            ax = axes[rr, cc]
            sub = (
                df[df["dataset"] == ds]
                .groupby("alpha")[["compression", "preservation"]]
                .mean()
            )
            n_full = int(df[df["dataset"] == ds]["n_full"].iloc[0])
            metric_label = df[df["dataset"] == ds]["metric"].iloc[0]
            label_suffix = " (Accuracy)" if ds == "isolet" else f" ({metric_label})"

            yv = sub["preservation"].values
            span = max(float(yv.max()) - float(yv.min()), 0.001)
            pad = max(span * 0.3, 0.003)
            y_lo = round(min(float(yv.min()) - pad, 0.999), 4)
            y_hi = round(max(float(yv.max()) + pad, 1.001), 4)

            xv = sub["compression"].values * 100
            x_span = max(float(xv.max()) - float(xv.min()), 0.01)
            x_lo = max(float(xv.min()) - x_span * 0.4, -0.5)
            x_hi = float(xv.max()) + x_span * 0.4

            for a in ALPHA_VALUES:
                if a not in sub.index:
                    continue
                x_val = sub.loc[a, "compression"] * 100
                y_val = sub.loc[a, "preservation"]
                ax.scatter(
                    x_val,
                    y_val,
                    color=alpha_colors[a],
                    marker=alpha_markers[a],
                    s=80,
                    zorder=5,
                    edgecolors="black",
                    linewidth=0.3,
                )

            ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Compression %")
            ax.set_title(f"{ds}{label_suffix}  (n={n_full})", fontsize=10)
            ax.grid(True, alpha=0.3)

        # --- Compression-count table at position (1,1) ---
        table_ax = axes[1, 1]
        table_ax.set_axis_off()
        all_ds_ordered = [d[0] for d in DATASETS]
        row_labels: list[str] = []
        cell_text: list[list[str]] = []
        for ds_name in all_ds_ordered:
            nf = _n_full_per_ds.get(ds_name, 0)
            row_labels.append(ds_name)
            row_cells: list[str] = []
            for a in _alpha_cols:
                nh = (
                    int(_harmful_pivot.loc[ds_name, a])
                    if ds_name in _harmful_pivot.index and a in _harmful_pivot.columns
                    else 0
                )
                row_cells.append(f"{nh}/{nf}")
            cell_text.append(row_cells)
        col_labels = [f"α={a:.2f}" for a in _alpha_cols]
        tbl = table_ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.25)
        for key, cell in tbl.get_celld().items():
            cell.set_linewidth(0.4)
            if key[0] == 0:
                cell.set_fontsize(6.5)
        table_ax.set_title(
            "Harmful features removed  (n_harmful / n_full)", fontsize=8, pad=2
        )

        for rr in range(n_rows):
            if axes[rr, 0].get_visible():
                axes[rr, 0].set_ylabel("Preservation ratio")
                break

        handles = [
            Line2D(
                [],
                [],
                color=alpha_colors[a],
                marker=alpha_markers[a],
                linestyle="",
                markersize=7,
                label=f"α={a:.2f}",
            )
            for a in ALPHA_VALUES
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=4, fontsize=8, framealpha=0.9
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        plot_path = tables_dir / "pareto_front_cls.pdf"
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {plot_path.name}")

    # --- Regression panel ---
    if reg_ds:
        n_ds = len(reg_ds)
        n_cols = min(2, n_ds)
        n_rows = (n_ds + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows), squeeze=False
        )
        fig.suptitle(
            "Alpha-Pareto Front -- Regression"
            + chr(10)
            + "X = compression (% features removed),  "
            "Y = R2 preservation (1.0 = full-model baseline)",
            fontsize=11,
            y=1.02,
        )

        for ax_i, ds in enumerate(reg_ds):
            r, c = ax_i // n_cols, ax_i % n_cols
            ax = axes[r, c]
            sub = (
                df[df["dataset"] == ds]
                .groupby("alpha")[["compression", "preservation"]]
                .mean()
            )
            n_full = int(df[df["dataset"] == ds]["n_full"].iloc[0])

            yv = sub["preservation"].values
            span = max(float(yv.max()) - float(yv.min()), 0.001)
            pad = max(span * 0.3, 0.003)
            y_lo = round(min(float(yv.min()) - pad, 0.999), 4)
            y_hi = round(max(float(yv.max()) + pad, 1.001), 4)

            xv = sub["compression"].values * 100
            x_span = max(float(xv.max()) - float(xv.min()), 0.01)
            x_lo = max(float(xv.min()) - x_span * 0.4, -0.5)
            x_hi = float(xv.max()) + x_span * 0.4

            for a in ALPHA_VALUES:
                if a not in sub.index:
                    continue
                x_val = sub.loc[a, "compression"] * 100
                y_val = sub.loc[a, "preservation"]
                ax.scatter(
                    x_val,
                    y_val,
                    color=alpha_colors[a],
                    marker=alpha_markers[a],
                    s=80,
                    zorder=5,
                    edgecolors="black",
                    linewidth=0.3,
                )

            ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Compression %")
            if c == 0:
                ax.set_ylabel("Preservation ratio")
            ax.set_title(f"{ds} (R2)  (n={n_full})", fontsize=10)
            ax.grid(True, alpha=0.3)

        for ax_i in range(len(reg_ds), n_rows * n_cols):
            axes[ax_i // n_cols, ax_i % n_cols].set_visible(False)

        handles = [
            Line2D(
                [],
                [],
                color=alpha_colors[a],
                marker=alpha_markers[a],
                linestyle="",
                markersize=7,
                label=f"α={a:.2f}",
            )
            for a in ALPHA_VALUES
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=4, fontsize=8, framealpha=0.9
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        plot_path = tables_dir / "pareto_front_reg.pdf"
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {plot_path.name}")

    # ---- Per-dataset recommendation ----
    _header("Dataset-level alpha recommendation")
    for ds in ds_names:
        sub = (
            df[df["dataset"] == ds]
            .groupby("alpha")[["compression", "preservation"]]
            .mean()
        )
        if len(sub) == 0:
            continue
        best_alpha = sub["compression"].idxmax()
        print(
            f"  {ds:20s}: best a={best_alpha:.2f} "
            f"(cr={sub.loc[best_alpha, 'compression'] * 100:.1f}%, "
            f"preserv={sub.loc[best_alpha, 'preservation']:.4f})"
        )

    _header("Done")
    print(f"Run directory: {run_dir}")
    print(f"Tables:       {tables_dir}")
    tee.close()


if __name__ == "__main__":
    main()
