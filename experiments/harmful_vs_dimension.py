"""
Harmful% vs feature-dimension curve (§6, reviewer #2).

For datasets with high native dimensionality, we vary the MI-filter cutoff
and measure the fraction of features classified as harmful (MinPPI = −1) at
each dimensionality level.  The resulting curve shows whether the harmful
signal grows, shrinks, or stays flat as more features are included — a
direct response to the reviewer's concern that MinPPI's harmful detection
might degrade as dimensionality increases.

Datasets
--------
- madelon (native 500, MI → {20, 50, 100})
- isolet  (native 617, MI → {20, 50, 100})
- gisette (native 5000, MI → {20, 50, 100, 200})

Method
------
At each cutoff *k*:
1. Load raw data + train/test split (seed = 42, consistent).
2. MI-select top-*k* features on train, propagate to val.
3. Compute MinPPI at α ∈ {0.90, 0.95, 0.99}.
4. Record per-class count and harmful%.

Output (``results/logs/<run_id>/``)::

    metadata.json
    config.yaml
    output.log
    harmful_vs_dimension.csv         # per-(dataset, k, alpha) rows
    harmful_vs_dimension_agg.csv     # mean ± std over seeds
    harmful_vs_dimension.pdf         # line plot

Run::

    python -u experiments/harmful_vs_dimension.py
"""

from __future__ import annotations

import json
import os
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
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from minppi_mofs.min_ppi import estimate_minppi_random_mc, minppi_stats
from minppi_mofs.voting_game import make_acc_full_cache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM_CUTOFFS: list[int] = [20, 50, 100, 150, 200]
ALPHA_VALUES: list[float] = [0.90, 0.95, 0.99]
SEEDS: list[int] = [42, 123, 456]

# (prefix, target, task, raw_csv, n_native)
DIM_DATASETS: list[tuple[str, str, str, str, int]] = [
    ("madelon", "label", "binary", "madelon.csv", 500),
    ("isolet", "letter", "multiclass", "isolet.csv", 617),
    ("gisette", "label", "binary", "gisette.csv", 5000),
]


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


def _mi_select(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    k: int,
    *,
    seed: int = 42,
    task: str = "binary",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select top-*k* features by mutual information.

    Args:
        X_train: Training feature matrix.
        y_train: Training targets.
        X_val: Validation feature matrix (columns subset to match).
        k: Number of features to retain.
        seed: Random seed.
        task: ``"binary"`` or ``"multiclass"``.

    Returns:
        ``(X_train_k, X_val_k, selected_features)``.
    """
    n = X_train.shape[1]
    if k >= n:
        cols = list(X_train.columns)
        return X_train, X_val, cols

    mi = mutual_info_classif(X_train, y_train, random_state=seed)
    scores = pd.Series(mi, index=X_train.columns)
    top_cols = scores.abs().sort_values(ascending=False).head(k).index.tolist()
    return X_train[top_cols], X_val[top_cols], top_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    git_root = project_root.parent
    raw_dir = project_root / "data" / "raw"

    run_id = _now_stamp()
    run_dir = project_root / "results" / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "output.log"
    tee = _Tee(log_path)
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    _orig_cwd = os.getcwd()
    os.chdir(str(run_dir))

    commit = _git_commit(git_root)

    def _header(msg: str) -> None:
        print("\n" + "=" * 72, flush=True)
        print(f"  {msg}", flush=True)
        print("=" * 72, flush=True)

    _header(f"Harmful% vs Dimension — {run_id}")
    print(f"Commit: {commit}", flush=True)
    print(f"Cutoffs: {DIM_CUTOFFS}", flush=True)
    print(f"Alphas:  {ALPHA_VALUES}", flush=True)
    print(f"Seeds:   {SEEDS}", flush=True)

    # Metadata
    metadata: dict = {
        "run_id": run_id,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dim_cutoffs": DIM_CUTOFFS,
        "alpha_values": ALPHA_VALUES,
        "seeds": SEEDS,
        "datasets": [d[0] for d in DIM_DATASETS],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    run_config: dict = {
        "description": "Harmful% vs feature-dimension curve (§6, reviewer #2)",
        "dim_cutoffs": DIM_CUTOFFS,
        "alpha_values": ALPHA_VALUES,
        "seeds": SEEDS,
        "datasets": [
            {"prefix": d[0], "target": d[1], "task": d[2], "n_native": d[4]}
            for d in DIM_DATASETS
        ],
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Per-dataset, per-seed, per-dimension, per-alpha
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    for prefix, target, task, raw_csv, n_native in DIM_DATASETS:
        _header(f"Dimension scan — {prefix} (native {n_native})")

        raw_path = raw_dir / raw_csv
        if not raw_path.exists():
            print(f"  SKIP: raw data not found at {raw_path}", flush=True)
            continue

        raw_df = pd.read_csv(raw_path)
        raw_X = raw_df.drop(columns=[target])
        raw_y = raw_df[target].values
        if raw_y.dtype == np.float64 and np.array_equal(raw_y, raw_y.astype(int)):
            raw_y = raw_y.astype(int)

        n_full = raw_X.shape[1]
        # Cap cutoffs to the available feature count
        valid_cutoffs = [k for k in DIM_CUTOFFS if k <= n_full]
        if n_full not in valid_cutoffs and n_full <= max(DIM_CUTOFFS):
            valid_cutoffs.append(n_full)
        valid_cutoffs = sorted(set(valid_cutoffs))
        print(f"  Valid cutoffs: {valid_cutoffs}", flush=True)

        # ShapG guidance (binary only — cached per seed)
        for seed in tqdm(SEEDS, desc=f"  seeds ({prefix})", unit="seed", file=sys.stdout):
            # Split once per seed: fixed train/val for all cutoffs
            X_tr_full, X_val_full, y_tr, y_val = train_test_split(
                raw_X, raw_y, test_size=0.2, random_state=seed,
                stratify=raw_y if task != "regression" else None,
            )
            y_tr_vals = y_tr.values if isinstance(y_tr, pd.DataFrame) else y_tr
            y_val_vals = y_val.values if isinstance(y_val, pd.DataFrame) else y_val

            for k in valid_cutoffs:
                # MI-select top-k
                X_tr_k, X_val_k, _ = _mi_select(
                    pd.DataFrame(X_tr_full, columns=raw_X.columns),
                    y_tr_vals,
                    pd.DataFrame(X_val_full, columns=raw_X.columns),
                    k, seed=42, task=task,
                )
                n_actual = X_tr_k.shape[1]

                # Use Algorithm A (shared-coalition random MC) for all datasets.
                # O(K · per_coalition) instead of O(p · K_adv) per alpha.
                # The dimension scan only needs the sign of MinPPI, not
                # precise identification of sparse harmful features.
                cache = make_acc_full_cache(
                    X_tr_k, X_val_k, y_tr_vals, y_val_vals, seed=seed, task=task,
                )
                for alpha in ALPHA_VALUES:
                    cache.set_threshold(alpha)
                    t0 = time.perf_counter()

                    minppi = estimate_minppi_random_mc(
                        X_tr_k, X_val_k, y_tr_vals, y_val_vals,
                        alpha=alpha, K=300, shared_coalitions=True,
                        per_coalition=5, acc_cache=cache, seed=seed,
                        verbose=False, task=task,
                    )

                    t_elapsed = time.perf_counter() - t0
                    stats = minppi_stats(minppi)

                    all_rows.append({
                        "dataset": prefix,
                        "seed": seed,
                        "alpha": alpha,
                        "cutoff": k,
                        "n_actual": n_actual,
                        "n_harmful": stats["n_harmful"],
                        "n_neutral": stats["n_neutral"],
                        "n_decisive": stats["n_decisive"],
                        "pct_harmful": round(stats["pct_harmful"], 1),
                        "time_s": round(t_elapsed, 1),
                    })
                    print(
                        f"    k={k:>3d} α={alpha:.2f}: "
                        f"harmful={stats['n_harmful']:>3d} "
                        f"({stats['pct_harmful']:5.1f}%)  "
                        f"neutral={stats['n_neutral']:>3d}  "
                        f"({t_elapsed:.1f}s)",
                        flush=True,
                    )

    # ------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        _header("No data — aborting")
        os.chdir(_orig_cwd)
        tee.close()
        return

    df.to_csv(run_dir / "harmful_vs_dimension.csv", index=False)
    print(f"\n  harmful_vs_dimension.csv saved ({len(df)} rows)", flush=True)

    # Aggregate over seeds
    agg = df.groupby(["dataset", "alpha", "cutoff", "n_actual"])[
        ["n_harmful", "pct_harmful", "time_s"]
    ].agg(["mean", "std"]).round(2)
    agg.to_csv(run_dir / "harmful_vs_dimension_agg.csv")
    print("  harmful_vs_dimension_agg.csv saved", flush=True)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    _header("Plotting")
    n_datasets = len(df["dataset"].unique())
    fig, axes = plt.subplots(1, n_datasets, figsize=(5.5 * n_datasets, 4.2),
                             squeeze=False)

    for ax_i, (ds_name, ax) in enumerate(zip(sorted(df["dataset"].unique()),
                                              axes[0])):
        for alpha in ALPHA_VALUES:
            sub = df[(df["dataset"] == ds_name) & (df["alpha"] == alpha)]
            if len(sub) == 0:
                continue
            # Mean over seeds
            pts = sub.groupby("n_actual")["pct_harmful"].mean()
            ax.plot(pts.index, pts.values, "o-", lw=1.5, ms=5,
                    label=f"α={alpha:.2f}")
        ax.set_title(ds_name)
        ax.set_xlabel("Number of features (MI top-k)")
        ax.set_ylabel("Harmful %")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = run_dir / "harmful_vs_dimension.pdf"
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path.name}", flush=True)

    _header("Done")
    print(f"Run directory: {run_dir}", flush=True)

    os.chdir(_orig_cwd)
    tee.close()


if __name__ == "__main__":
    main()
