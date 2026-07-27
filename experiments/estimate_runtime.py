"""Estimate full formal run time using actual schedule construction logic.

Run: python experiments/estimate_runtime.py
"""

import sys, time, json, hashlib
from pathlib import Path
from itertools import combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minppi_mofs.nested_estimation import (
    make_nested_random_schedule,
    make_nested_guided_schedule,
)

DATASETS = {
    "abalone": 8,
    "spambase": 57,
    "madelon": 100,
    "isolet": 100,
    "gisette": 100,
    "credit_card_fraud": 29,
    "crimedata": 100,
}

# XGBoost CUDA fit time (seconds, median), benchmarked on actual train split sizes
PER_FIT_S = {
    "abalone": 0.15,
    "spambase": 0.16,
    "madelon": 0.17,
    "isolet": 0.18,
    "gisette": 0.18,
    "credit_card_fraud": 0.58,
    "crimedata": 0.16,
}

BUDGETS = [10, 25, 50, 100]
N_SEEDS = 5
NSGA2_POP = 40
NSGA2_EVALS = 800
NSGA2_VARIANTS = 2


def count_coalitions(feature_names, budgets, seed):
    """Count unique coalitions for random + guided schedules."""
    features = list(feature_names)
    n = len(features)

    random_sched = make_nested_random_schedule(features, budgets, seed)
    random_set = set()
    for ft in random_sched.trials:
        for c in ft:
            random_set.add(c)
    random_naive = sum(len(ft) for ft in random_sched.trials)

    guided_sched = make_nested_guided_schedule(features, budgets, seed)
    guided_count = sum(len(ft) for ft in guided_sched.trials)

    # Exact: all 2^(n-1) coalitions, shared across all n features
    if n <= 16:
        all_feats = set(features)
        exact_set = set()
        for feat in features:
            others = all_feats - {feat}
            for k in range(len(others) + 1):
                for combo in combinations(sorted(others), k):
                    exact_set.add(frozenset(combo))
        exact_unique = len(exact_set)
        exact_naive = n * exact_unique
    else:
        exact_unique = 0
        exact_naive = 0

    # Guided coalitions: virtually all unique (perturbation-based, per-feature)
    # Intersection between random and guided: empirically ~1-2%
    overlap = len(random_set & set())  # always 0 since guided set is empty here
    total_unique = len(random_set) + guided_count + exact_unique

    return {
        "n_features": n,
        "random_naive": random_naive,
        "guided_naive": guided_count,
        "exact_unique": exact_unique,
        "exact_naive": exact_naive,
        "random_unique": len(random_set),
        "guided_unique": guided_count,
        "total_naive": random_naive + guided_count + exact_naive,
        "total_unique": total_unique,
    }


def main():
    print("=" * 75)
    print(" ICOMP 2026 Formal Run — Runtime Estimation")
    print(f" Budgets={BUDGETS}  Seeds={N_SEEDS}  GPU=RTX 5070 Laptop")
    print("=" * 75)
    print()

    # ---- Estimator stage ----
    print("━" * 75)
    print(" STAGE: estimator  (per-seed unique coalitions)")
    print("━" * 75)
    header = (
        f"{'Dataset':<22} {'n':>4}  {'Random':>8} {'Guided':>8} "
        f"{'Exact':>8} {'TotalUniq':>10}  {'s/fit':>6} {'Seed-h':>8}"
    )
    print(header)
    print("-" * 75)

    per_seed_fits = {}
    per_seed_hours = {}
    for ds_name, n_features in DATASETS.items():
        feature_names = tuple(f"f{i}" for i in range(n_features))
        c = count_coalitions(feature_names, BUDGETS, 42)
        fits = c["total_unique"]
        fit_s = PER_FIT_S[ds_name]
        seed_h = fits * fit_s / 3600
        per_seed_fits[ds_name] = fits
        per_seed_hours[ds_name] = seed_h
        print(
            f"{ds_name:<22} {c['n_features']:>4}  {c['random_naive']:>8} "
            f"{c['guided_naive']:>8} {c['exact_unique']:>8} {fits:>10}  "
            f"{fit_s:>6.2f} {seed_h:>8.2f}"
        )

    total_estimator_seed_h = sum(per_seed_hours.values())
    total_estimator_h = total_estimator_seed_h * N_SEEDS
    print("-" * 75)
    print(
        f"{'Estimator (1 seed)':<22} {'':>4}  {'':>8} {'':>8} {'':>8} "
        f"{'':>10}  {'':>6} {total_estimator_seed_h:>8.2f}"
    )
    print(
        f"{'Estimator (5 seeds)':<22} {'':>4}  {'':>8} {'':>8} {'':>8} "
        f"{'':>10}  {'':>6} {total_estimator_h:>8.2f}"
    )
    print()

    # ---- NSGA-II stage ----
    print("━" * 75)
    print(" STAGE: nsga2")
    print("━" * 75)
    evals_per_ds_seed = NSGA2_VARIANTS * NSGA2_EVALS
    nsga2_h = 0
    for ds_name in DATASETS:
        physical = evals_per_ds_seed * 0.5  # 50% memoization hit rate
        ds_h = physical * PER_FIT_S[ds_name] * N_SEEDS / 3600
        nsga2_h += ds_h
    print(
        f"  {NSGA2_VARIANTS} variants × {NSGA2_EVALS} evals = {evals_per_ds_seed} per ds/seed"
    )
    print(
        f"  50% memoization hit → {evals_per_ds_seed * 0.5:.0f} physical fits per ds/seed"
    )
    print(f"  7 datasets × {N_SEEDS} seeds: {nsga2_h:.1f} hours")
    print()

    # ---- Removal stage ----
    print("━" * 75)
    print(" STAGE: removal")
    print("━" * 75)
    # MinPPI: 0 new fits (cached)
    # SHAP: TreeSHAP (fast, in-memory)
    # MinShap + Banzhaf: shared evaluator, ~80% cache hits from estimator
    #    → each needs n_features * K_shap/bz coalition evaluations
    # Random matched: 30 repeats × ~8 harmful features avg
    # Final models: ~6 controls × 1 fit each
    removal_h = 0
    for ds_name, n_features in DATASETS.items():
        # MinShap + Banzhaf: assume K_shap ≈ 100 per feature, 20% cold
        ms_bz_fits = n_features * 100 * 2 * 0.20
        # Random matched + final models
        rm_final = 30 * 8 + 6
        total_fits = ms_bz_fits + rm_final
        ds_h = total_fits * PER_FIT_S[ds_name] * N_SEEDS / 3600
        removal_h += ds_h
    print(f"  MinShap+Banzhaf (20% cold) + random matched + final models")
    print(f"  7 datasets × {N_SEEDS} seeds: {removal_h:.1f} hours")
    print()

    # ---- Alpha stage ----
    print("━" * 75)
    print(" STAGE: alpha")
    print("━" * 75)
    # 4 alpha values, each needs retained-subset evaluation
    # Plus full-model evaluation for alpha threshold
    # All coalition utilities from cache; just ~5 retained-model fits per alpha
    alpha_fits_per_ds_seed = 4 * 5  # 20 fits
    alpha_h = 0
    for ds_name in DATASETS.items():
        alpha_h += alpha_fits_per_ds_seed * PER_FIT_S[ds_name[0]] * N_SEEDS / 3600
    print(f"  4 α values × 5 retained-model fits per ds/seed")
    print(f"  7 datasets × {N_SEEDS} seeds: {alpha_h:.2f} hours")
    print()

    # ---- Ablation stage ----
    print("━" * 75)
    print(" STAGE: ablation")
    print("━" * 75)
    # Utility ablation: ISOLET only, shared prediction digest → 0 extra fits
    # Guidance ablation: 4 variants with cached coalitions
    # Dispatcher ablation: projection only, 0 extra fits
    # Very light: ~500 fits total
    ablation_h = 500 * 0.18 / 3600
    print(f"  ~500 fits total: {ablation_h:.2f} hours")
    print()

    # ---- Overhead ----
    print("━" * 75)
    print(" STAGES: preprocess + summarize + figures")
    print("━" * 75)
    overhead_h = 0.5
    print(f"  ~{overhead_h:.1f} hours (CPU-bound, parallel with GPU stages)")
    print()

    # ---- Total ----
    total_h = (
        total_estimator_h + nsga2_h + removal_h + alpha_h + ablation_h + overhead_h
    )
    print("=" * 75)
    print(f" ESTIMATED TOTAL: {total_h:.1f} hours ({total_h / 24:.1f} days)")
    print("=" * 75)
    print()
    print("Breakdown:")
    print(
        f"  estimator:   {total_estimator_h:>6.1f}h  ({total_estimator_h / total_h * 100:.0f}%)"
    )
    print(f"  nsga2:       {nsga2_h:>6.1f}h  ({nsga2_h / total_h * 100:.0f}%)")
    print(f"  removal:     {removal_h:>6.1f}h  ({removal_h / total_h * 100:.0f}%)")
    print(f"  alpha:       {alpha_h:>6.1f}h  ({alpha_h / total_h * 100:.0f}%)")
    print(f"  ablation:    {ablation_h:>6.1f}h  ({ablation_h / total_h * 100:.0f}%)")
    print(f"  overhead:    {overhead_h:>6.1f}h  ({overhead_h / total_h * 100:.0f}%)")
    print()

    # ---- Sensitivity to Credit-Card ----
    print("━" * 75)
    print(" SENSITIVITY ANALYSIS")
    print("━" * 75)

    # What if Credit-Card is slower?
    for credit_s in [0.60, 0.80, 1.00]:
        delta = credit_s - PER_FIT_S["credit_card_fraud"]
        extra = delta * per_seed_fits["credit_card_fraud"] * N_SEEDS / 3600
        print(f"  Credit-Card @ {credit_s:.2f}s/fit: total = {total_h + extra:.1f}h")

    # What if Gisette goes to 200 features (per plan MI rule)?
    gisette_200 = count_coalitions(tuple(f"f{i}" for i in range(200)), BUDGETS, 42)
    g200_fits = gisette_200["total_unique"]
    g200_h = g200_fits * 0.20 * N_SEEDS / 3600
    current_gisette_h = per_seed_hours["gisette"] * N_SEEDS
    delta_g = g200_h - current_gisette_h
    print(
        f"  Gisette 100→200feats: adds {delta_g:.1f}h → total = {total_h + delta_g:.1f}h"
    )

    # What if Crimedata is 119 features (per plan)?
    crimedata_119 = count_coalitions(tuple(f"f{i}" for i in range(119)), BUDGETS, 42)
    c119_fits = crimedata_119["total_unique"]
    c119_h = c119_fits * 0.17 * N_SEEDS / 3600
    current_crimedata_h = per_seed_hours["crimedata"] * N_SEEDS
    delta_c = c119_h - current_crimedata_h
    print(
        f"  Crimedata 100→119feats: adds {delta_c:.1f}h → total = {total_h + delta_c:.1f}h"
    )

    print()
    print("━" * 75)
    print(" TIMELINE FEASIBILITY (Jul 27 → Aug 5)")
    print("━" * 75)
    print(f"  Run 1 full:         {total_h:.0f}h (~{total_h / 24:.1f} days)")
    print(f"  Analysis + revise:  1-2 days")
    print(f"  Run 2 (if needed):  {total_h:.0f}h (~{total_h / 24:.1f} days)")
    print(f"  Final revise:       1 day")
    print(f"  Submit buffer:      0.5 day")
    print()
    total_days = total_h / 24 * 2 + 4.5  # 2 runs + fixed work
    print(f"  Best case (1 run):  {total_h / 24 + 2.5:.1f} days — feasible ✓")
    print(
        f"  Worst case (2 runs): {total_days:.1f} days — "
        f"{'feasible ✓' if total_days <= 9 else 'TIGHT ⚠'}"
    )


if __name__ == "__main__":
    main()
