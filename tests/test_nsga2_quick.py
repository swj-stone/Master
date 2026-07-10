"""
Quick-test: NSGA-II baseline only — single dataset, single seed.

Use before launching the full ``run_main_table.py`` (which takes 4-5 h)
to verify that the NSGA-II baseline works correctly on real data.

Run::

    python -u tests/test_nsga2_quick.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from minppi_mofs.baselines.nsga2_fs import nsga2_feature_selection, nsga2_ranking

# ── config ──────────────────────────────────────────────────────────
DATASET: tuple[str, str, str] = ("spambase", "is_spam", "binary")
SEED: int = 42
POP_SIZE: int = 10
MAX_GEN: int = 10
# ─────────────────────────────────────────────────────────────────────

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    prefix, target, task = DATASET

    train = pd.read_csv(PROCESSED / f"{prefix}_s{SEED}_train.csv")
    test = pd.read_csv(PROCESSED / f"{prefix}_s{SEED}_test.csv")
    X_tr = train.drop(columns=[target])
    y_tr = train[target].values
    X_te = test.drop(columns=[target])
    y_te = test[target].values

    print(f"Dataset: {prefix}  ({X_tr.shape[1]} features, {X_tr.shape[0]} samples)")
    print(f"Task: {task}")
    print(f"NSGA-II: pop={POP_SIZE}, gen={MAX_GEN}, seed={SEED}")
    print()

    # ── Full NSGA-II ────────────────────────────────────────────────
    t0 = time.perf_counter()
    result = nsga2_feature_selection(
        X_tr, y_tr, X_te, y_te,
        task=task, seed=SEED,
        pop_size=POP_SIZE, max_gen=MAX_GEN,
        verbose=True,
    )
    t1 = time.perf_counter() - t0
    print(f"\nDone: {t1:.1f}s")
    print(f"  HV       = {result.hv:.4f}")
    print(f"  Front    = {len(result.pareto_front)} non-dominated solutions")
    print(f"  Converged at gen {result.converged_gen}")
    print(f"  HV curve = {[round(h, 4) for h in result.hv_history]}")

    # ── Ranking adapter ─────────────────────────────────────────────
    t0 = time.perf_counter()
    ranking = nsga2_ranking(
        X_tr, y_tr, task=task, seed=SEED,
        pop_size=POP_SIZE, max_gen=MAX_GEN, verbose=False,
    )
    t2 = time.perf_counter() - t0
    print(f"\nRanking adapter: {t2:.1f}s")
    print("  Top-10 features:")
    for i, (feat, val) in enumerate(ranking[:10], 1):
        print(f"    {i:2d}. {feat:30s}  {val:.6f}")

    print("\nAll OK — ready for run_main_table.py.")


if __name__ == "__main__":
    main()
