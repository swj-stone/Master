"""
MOFS Verification — MinPPI Pre-Filter + NSGA-II vs NSGA-II-full.

**Fixed (Task 2 + Task 3):** Uses independent train / val / test splits
from ``data/processed_v2/``:

  - **train**: fit coalition models (MinPPI) and NSGA-II individuals
  - **val**:   compute coalition utility → MinPPI estimation *and*
               NSGA-II objective evaluation
  - **test**:  one-time evaluation of the frozen validation Pareto set
               (never seen by MinPPI or NSGA-II during search)

For each (dataset, seed):
  1. Compute MinPPI at α=0.95 on (train, val) → harmful set H.
  2. Run NSGA-II on ALL features → validate on val → report val HV.
  3. Delete H, run NSGA-II on remaining features → val HV.
  4. For both, retrain each Pareto-optimal mask on train+val,
     evaluate on test → report **test HV**.
  5. Total filtered cost includes MinPPI time.

Output (results/tables/<run_id>/)::

    mofs_verify.csv       — per-(dataset, seed, method): hv_val, hv_test, ...
    mofs_verify_agg.csv   — mean ± std over seeds
    mofs_verify_fronts.csv — Pareto scatter points (test evaluation)
    mofs_verify_history.csv — HV per generation (val)

Run::

    python -u experiments/run_mofs_verify.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from minppi_mofs.min_ppi import estimate_minppi, estimate_minppi_random_mc
from minppi_mofs.shapg_integration import build_mi_correlation_guidance
from minppi_mofs.baselines.nsga2_fs import (
    _compute_hv,
    _evaluate_individual,
    nsga2_feature_selection,
)

# ============================================================
# Constants
# ============================================================

ALPHA: float = 0.95
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
    "isolet": {"K": 30, "per_coalition": 2},
    "gisette": {"K_adv": 10},
    "credit_card_fraud": {"K_adv": 30},
    "crimedata": {"K": 40, "per_coalition": 2},
    "abalone": {"K": 40, "per_coalition": 2},
}

NSGA2_POP: int = 20
NSGA2_GEN: int = 20

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root), text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class _Tee:
    def __init__(self, log_path: Path) -> None:
        self._console, self._file = sys.stdout, log_path.open("w", encoding="utf-8")

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


def _load_splits(
    prefix: str, target: str, data_dir: Path, *, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_tr, X_va, X_te, y_tr, y_va, y_te) from processed_v2."""
    tr = pd.read_csv(data_dir / f"{prefix}_s{seed}_train.csv")
    va = pd.read_csv(data_dir / f"{prefix}_s{seed}_val.csv")
    te = pd.read_csv(data_dir / f"{prefix}_s{seed}_test.csv")
    return (
        tr.drop(columns=[target]), va.drop(columns=[target]), te.drop(columns=[target]),
        tr[target].values, va[target].values, te[target].values,
    )


def _evaluate_front_on_test(
    pareto_front: list,          # list[tuple[np.ndarray, tuple[float,float]]]
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    *,
    task: str,
    seed: int,
) -> tuple[list[tuple[float, float]], float]:
    """Retrain each Pareto-optimal mask on train+val, evaluate on test.

    Returns:
        test_points: list of (card_ratio, 1 - utility) on test.
        test_hv: hypervolume of the test-evaluated front.
    """
    X_trv = pd.concat([X_tr, X_va], axis=0)
    y_trv = np.concatenate([y_tr, y_va])

    test_points: list[tuple[float, float]] = []
    for mask, _val_point in pareto_front:
        f1, f2 = _evaluate_individual(
            mask.astype(np.float64), X_trv, y_trv, X_te, y_te,
            task=task, seed=seed,
        )
        test_points.append((f1, f2))

    if not test_points:
        return [], 0.0

    pts = np.array(test_points)
    hv = _compute_hv(pts, np.array([1.0, 1.0]))
    return test_points, hv


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    git_root = project_root.parent
    data_dir = project_root / "data" / "processed_v2"

    run_id = _now_stamp()
    run_dir = project_root / "results" / "logs" / run_id
    tables_dir = project_root / "results" / "tables" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tee = _Tee(run_dir / "output.log")
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    commit = _git_commit(git_root)
    print("=" * 72)
    print(f"  MOFS Verify — {run_id}  (commit: {commit})")
    print("  Data: data/processed_v2/  (train/val/test split)")
    print("  Protocol: train→fit, val→search, test→final eval")
    print("=" * 72)

    metadata = {
        "run_id": run_id,
        "git_commit": commit,
        "alpha": ALPHA,
        "seeds": SEEDS,
        "nsga2_pop": NSGA2_POP,
        "nsga2_gen": NSGA2_GEN,
        "datasets": [d[0] for d in DATASETS],
        "data_source": "processed_v2 (train/val/test)",
        "protocol": "train→fit, val→utility+NSGA2-obj, test→one-time eval",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (tables_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    all_rows: list[dict] = []
    front_rows: list[dict] = []
    history_rows: list[dict] = []

    for prefix, target, task in tqdm(DATASETS, desc="MOFS-verify", unit="ds", file=sys.stdout):
        for seed in tqdm(SEEDS, desc=f"  seeds ({prefix})", unit="sd", file=sys.stdout, leave=False):
            try:
                X_tr, X_va, X_te, y_tr, y_va, y_te = _load_splits(
                    prefix, target, data_dir, seed=seed,
                )
            except FileNotFoundError:
                print(f"\n  SKIP {prefix} s{seed}: missing split in processed_v2/")
                continue

            n_full = X_tr.shape[1]
            budget = DATASET_BUDGET.get(prefix, {"K_adv": 30})

            # ── Phase 1: MinPPI on (train, val) ──────────────────────
            t_p = time.perf_counter()
            n_cls = len(np.unique(y_tr))
            if n_cls > 2 or task == "regression":
                minppi = estimate_minppi_random_mc(
                    X_tr, X_va, y_tr, y_va,
                    alpha=ALPHA,
                    K=budget.get("K", 200),
                    shared_coalitions=True,
                    per_coalition=budget.get("per_coalition", 5),
                    seed=seed, verbose=False, task=task,
                )
            else:
                mi_scores, correlation_neighbors = build_mi_correlation_guidance(
                    X_tr, y_tr, task=task, seed=seed
                )
                minppi = estimate_minppi(
                    X_tr, X_va, y_tr, y_va,
                    alpha=ALPHA,
                    method="mi_correlation_guided",
                    K_adv=budget.get("K_adv", 30),
                    correlation_neighbors=correlation_neighbors,
                    mi_scores=mi_scores,
                    seed=seed, verbose=False, task=task,
                )
            harmful = {f for f, v in minppi.items() if v == -1}
            t_minppi = time.perf_counter() - t_p
            keep = [c for c in X_tr.columns if c not in harmful]
            n_h = len(harmful)

            print(
                f"\n  {prefix} s{seed}: harmful={n_h}/{n_full}  "
                f"keep={len(keep)}  [MinPPI {t_minppi:.0f}s]",
            )

            # ── NSGA-II on ALL features (val = objective) ────────────
            print("    NSGA-II-full ...", end="", flush=True)
            r_full = nsga2_feature_selection(
                X_tr, y_tr, X_va, y_va,
                task=task, seed=seed,
                pop_size=NSGA2_POP, max_gen=NSGA2_GEN,
                verbose=False,
            )
            # One-time test evaluation of the frozen Pareto set
            _, test_hv_full = _evaluate_front_on_test(
                r_full.pareto_front, X_tr, y_tr, X_va, y_va, X_te, y_te,
                task=task, seed=seed,
            )
            print(
                f" HV_val={r_full.hv:.4f} HV_test={test_hv_full:.4f}  "
                f"dim={n_full}  conv_gen={r_full.converged_gen}  "
                f"[{r_full.runtime_s:.0f}s]",
            )
            all_rows.append({
                "dataset": prefix, "seed": seed, "method": "nsga2_full",
                "search_dim": n_full,
                "hv_val": round(r_full.hv, 4),
                "hv_test": round(test_hv_full, 4),
                "runtime_s": round(r_full.runtime_s, 1),
                "converged_gen": r_full.converged_gen,
            })
            for mask, (card_ratio, one_minus_auc) in r_full.pareto_front:
                front_rows.append({
                    "dataset": prefix, "seed": seed, "method": "nsga2_full",
                    "compression": round(float(card_ratio), 4),
                    "auc": round(1.0 - float(one_minus_auc), 4),
                })
            for gen, hv_val in enumerate(r_full.hv_history):
                history_rows.append({
                    "dataset": prefix, "seed": seed, "method": "nsga2_full",
                    "generation": gen, "hv": round(hv_val, 4),
                })

            # ── NSGA-II on FILTERED features ─────────────────────────
            if len(keep) < 2:
                print("    NSGA-II-filtered SKIP (<2 features)")
                continue
            X_tr_f, X_va_f, X_te_f = X_tr[keep], X_va[keep], X_te[keep]
            t_nsga2_filt = time.perf_counter()
            print("    NSGA-II-filtered ...", end="", flush=True)
            r_filt = nsga2_feature_selection(
                X_tr_f, y_tr, X_va_f, y_va,
                task=task, seed=seed,
                pop_size=NSGA2_POP, max_gen=NSGA2_GEN,
                verbose=False,
            )
            runtime_filt = time.perf_counter() - t_nsga2_filt
            _, test_hv_filt = _evaluate_front_on_test(
                r_filt.pareto_front, X_tr_f, y_tr, X_va_f, y_va, X_te_f, y_te,
                task=task, seed=seed,
            )
            print(
                f" HV_val={r_filt.hv:.4f} HV_test={test_hv_filt:.4f}  "
                f"dim={len(keep)}  conv_gen={r_filt.converged_gen}  "
                f"[NSGA2 {runtime_filt:.0f}s, total w/ MinPPI {runtime_filt + t_minppi:.0f}s]",
            )
            all_rows.append({
                "dataset": prefix, "seed": seed, "method": "nsga2_filtered",
                "search_dim": len(keep),
                "hv_val": round(r_filt.hv, 4),
                "hv_test": round(test_hv_filt, 4),
                "runtime_s": round(runtime_filt + t_minppi, 1),
                "converged_gen": r_filt.converged_gen,
            })
            for mask, (card_ratio, one_minus_auc) in r_filt.pareto_front:
                front_rows.append({
                    "dataset": prefix, "seed": seed, "method": "nsga2_filtered",
                    "compression": round(float(card_ratio), 4),
                    "auc": round(1.0 - float(one_minus_auc), 4),
                })
            for gen, hv_val in enumerate(r_filt.hv_history):
                history_rows.append({
                    "dataset": prefix, "seed": seed, "method": "nsga2_filtered",
                    "generation": gen, "hv": round(hv_val, 4),
                })

    # ── Save ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(tables_dir / "mofs_verify.csv", index=False)
    print(f"\n  mofs_verify.csv saved ({len(df)} rows)")

    pd.DataFrame(front_rows).to_csv(tables_dir / "mofs_verify_fronts.csv", index=False)
    print(f"  mofs_verify_fronts.csv saved ({len(front_rows)} rows)")

    pd.DataFrame(history_rows).to_csv(tables_dir / "mofs_verify_history.csv", index=False)
    print(f"  mofs_verify_history.csv saved ({len(history_rows)} rows)")

    agg = (
        df.groupby(["dataset", "method"])[
            ["hv_val", "hv_test", "search_dim", "runtime_s", "converged_gen"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(tables_dir / "mofs_verify_agg.csv")
    print("  mofs_verify_agg.csv saved")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Summary: mean HV_test (over seeds)")
    print("=" * 72)
    for ds in sorted(df["dataset"].unique()):
        f_hv = df[(df["dataset"] == ds) & (df["method"] == "nsga2_full")]["hv_test"].mean()
        p_rows = df[(df["dataset"] == ds) & (df["method"] == "nsga2_filtered")]
        if len(p_rows) and p_rows["hv_test"].notna().any():
            p_hv = p_rows["hv_test"].mean()
            delta = p_hv - f_hv
            better = "+" if delta > 0 else ""
            print(f"  {ds:20s}: full={f_hv:.4f}  filtered={p_hv:.4f}  ({better}{delta:+.4f})")
        else:
            print(f"  {ds:20s}: full={f_hv:.4f}  filtered=(skipped)")

    print(f"\nDone.  Tables: {tables_dir}")
    tee.close()


if __name__ == "__main__":
    main()
