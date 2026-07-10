"""
MOFS Verification — MinPPI Pre-Filter + NSGA-II vs NSGA-II-full.

For each (dataset, seed):
  1. Compute MinPPI at α=0.95 → harmful set H.
  2. Run NSGA-II on ALL features (NSGA-II-full).
  3. Delete H from data, run NSGA-II on REMAINING features (NSGA-II-filtered).
  4. Compare: HV, search space size, runtime.

MinPPI and NSGA-II are **independent** — MinPPI only provides the
pre-filter; NSGA-II operates on whatever features it receives.  This
experiment validates whether MinPPI's voting-game pre-filter helps
downstream multi-objective search.

Output (results/tables/<run_id>/)::

    mofs_verify.csv     # per-(dataset, seed, method): hv, search_dim, runtime
    mofs_verify_agg.csv # mean ± std over seeds

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
from minppi_mofs.shapg_integration import build_guidance_from_data
from minppi_mofs.baselines.nsga2_fs import nsga2_feature_selection

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


def load_split(
    prefix: str,
    target: str,
    processed_dir: Path,
    *,
    seed: int,
):
    train = pd.read_csv(processed_dir / f"{prefix}_s{seed}_train.csv")
    test = pd.read_csv(processed_dir / f"{prefix}_s{seed}_test.csv")
    return (
        train.drop(columns=[target]),
        train[target].values,
        test.drop(columns=[target]),
        test[target].values,
    )


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
    print("=" * 72)
    print(f"  MOFS Verify — {run_id}  (commit: {commit})")
    print("=" * 72)

    metadata = {
        "run_id": run_id,
        "git_commit": commit,
        "alpha": ALPHA,
        "seeds": SEEDS,
        "nsga2_pop": NSGA2_POP,
        "nsga2_gen": NSGA2_GEN,
        "datasets": [d[0] for d in DATASETS],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (tables_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    all_rows: list[dict] = []

    for prefix, target, task in tqdm(
        DATASETS,
        desc="MOFS-verify",
        unit="ds",
        file=sys.stdout,
    ):
        for seed in tqdm(
            SEEDS,
            desc=f"  seeds ({prefix})",
            unit="sd",
            file=sys.stdout,
            leave=False,
        ):
            try:
                X_tr, y_tr, X_te, y_te = load_split(
                    prefix,
                    target,
                    processed_dir,
                    seed=seed,
                )
            except FileNotFoundError:
                print(f"\n  SKIP {prefix} s{seed}: missing split")
                continue

            n_full = X_tr.shape[1]
            budget = DATASET_BUDGET.get(prefix, {"K_adv": 30})

            # ── Phase 1: MinPPI ────────────────────────────────────
            t_p = time.perf_counter()
            n_cls = len(np.unique(y_tr))
            if n_cls > 2 or task == "regression":
                minppi = estimate_minppi_random_mc(
                    X_tr,
                    X_tr,
                    y_tr,
                    y_tr,
                    alpha=ALPHA,
                    K=budget.get("K", 200),
                    shared_coalitions=True,
                    per_coalition=budget.get("per_coalition", 5),
                    seed=seed,
                    verbose=False,
                    task=task,
                )
            else:
                sg_s, sg_n = build_guidance_from_data(X_tr, y_tr, seed=seed)
                minppi = estimate_minppi(
                    X_tr,
                    X_tr,
                    y_tr,
                    y_tr,
                    alpha=ALPHA,
                    method="shapg_guided",
                    K_adv=budget.get("K_adv", 30),
                    shapg_neighbors=sg_n,
                    shapg_scores=sg_s,
                    seed=seed,
                    verbose=False,
                    task=task,
                )
            harmful = {f for f, v in minppi.items() if v == -1}
            t_minppi = time.perf_counter() - t_p
            keep = [c for c in X_tr.columns if c not in harmful]
            n_h = len(harmful)

            print(
                f"\n  {prefix} s{seed}: harmful={n_h}/{n_full}  "
                f"keep={len(keep)}  [{t_minppi:.0f}s]",
            )

            # ── NSGA-II on ALL features ────────────────────────────
            print("    NSGA-II-full ...", end="", flush=True)
            r_full = nsga2_feature_selection(
                X_tr,
                y_tr,
                X_te,
                y_te,
                task=task,
                seed=seed,
                pop_size=NSGA2_POP,
                max_gen=NSGA2_GEN,
                verbose=False,
            )
            print(
                f" HV={r_full.hv:.4f}  "
                f"dim={n_full}  "
                f"conv_gen={r_full.converged_gen}  "
                f"[{r_full.runtime_s:.0f}s]",
            )
            all_rows.append(
                {
                    "dataset": prefix,
                    "seed": seed,
                    "method": "nsga2_full",
                    "search_dim": n_full,
                    "hv": round(r_full.hv, 4),
                    "runtime_s": round(r_full.runtime_s, 1),
                    "converged_gen": r_full.converged_gen,
                }
            )

            # ── NSGA-II on FILTERED features ───────────────────────
            if len(keep) < 2:
                print("    NSGA-II-filtered SKIP (<2 features)")
                continue
            X_tr_f, X_te_f = X_tr[keep], X_te[keep]
            print("    NSGA-II-filtered ...", end="", flush=True)
            r_filt = nsga2_feature_selection(
                X_tr_f,
                y_tr,
                X_te_f,
                y_te,
                task=task,
                seed=seed,
                pop_size=NSGA2_POP,
                max_gen=NSGA2_GEN,
                verbose=False,
            )
            print(
                f" HV={r_filt.hv:.4f}  "
                f"dim={len(keep)}  "
                f"conv_gen={r_filt.converged_gen}  "
                f"[{r_filt.runtime_s:.0f}s]",
            )
            all_rows.append(
                {
                    "dataset": prefix,
                    "seed": seed,
                    "method": "nsga2_filtered",
                    "search_dim": len(keep),
                    "hv": round(r_filt.hv, 4),
                    "runtime_s": round(r_filt.runtime_s, 1),
                    "converged_gen": r_filt.converged_gen,
                }
            )

    # ── Save ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(tables_dir / "mofs_verify.csv", index=False)
    print(f"\n  mofs_verify.csv saved ({len(df)} rows)")

    agg = (
        df.groupby(["dataset", "method"])[
            ["hv", "search_dim", "runtime_s", "converged_gen"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(tables_dir / "mofs_verify_agg.csv")
    print("  mofs_verify_agg.csv saved")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Summary: mean HV (over seeds)")
    print("=" * 72)
    for ds in sorted(df["dataset"].unique()):
        f_hv = df[(df["dataset"] == ds) & (df["method"] == "nsga2_full")]["hv"].mean()
        p_hv = df[(df["dataset"] == ds) & (df["method"] == "nsga2_filtered")][
            "hv"
        ].mean()
        if pd.notna(p_hv):
            delta = p_hv - f_hv
            better = "+" if delta > 0 else ""
            print(
                f"  {ds:20s}: full={f_hv:.4f}  filtered={p_hv:.4f}  ({better}{delta:+.4f})"
            )
        else:
            print(f"  {ds:20s}: full={f_hv:.4f}  filtered=(skipped)")

    print(f"\nDone.  Tables: {tables_dir}")
    tee.close()


if __name__ == "__main__":
    main()
