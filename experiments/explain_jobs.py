"""Explain estimator job count and structure."""

import json
from pathlib import Path

# Smoke v3 verification
smoke_status = Path("results/icomp2026_rerun/smoke-full-20260727-v3/status.json")
if smoke_status.exists():
    st = json.loads(smoke_status.read_text())
    est = st["stages"]["estimator"]
    print("=== Smoke v3 estimator ===")
    for k, v in est.items():
        print(f"  {k}: {v}")
    print(f"  Total smoke estimator jobs: {sum(est.values())}")

# Formal job count
datasets = {
    "abalone": 8,
    "spambase": 56,
    "madelon": 100,
    "isolet": 100,
    "gisette": 100,
    "credit_card_fraud": 29,
    "crimedata": 100,
}
SEEDS = 5
BUDGETS = [10, 25, 50, 100]
N_BUDGETS = len(BUDGETS)

print("\n=== Formal estimator job breakdown ===")
total = 0
for ds, nf in datasets.items():
    n_methods = 4 if nf <= 16 else 3  # exact only for n<=16
    jobs = n_methods * N_BUDGETS * SEEDS
    total += jobs
    methods_str = "exact+" if nf <= 16 else ""
    print(
        f"  {ds:22s} ({nf:3d}f): {n_methods} methods ({methods_str}random+guided+hybrid) x {N_BUDGETS} budgets x {SEEDS} seeds = {jobs}"
    )

print(f"  {'Total':>38} = {total} estimator jobs")

print("\n=== But most jobs are projections (0 new fits) ===")
print("Bundle architecture:")
print("  1 bundle per dataset x seed = 7 x 5 = 35 bundles")
print("  Each bundle contains:")
print("    - random_mc schedule: all 5 budgets as nested prefixes of max=100")
print("    - guided schedule:    all 5 budgets as nested prefixes of max=100")
print("    - exact schedule:     only for Abalone (8 features)")
print("  Each bundle is a SINGLE physical run:")
print("    - Cold fits at max budget K=100, cached to SQLite")
print("    - All 5 budget checkpoints are deterministic prefixes of the same schedule")
print("    - Smaller budgets (K=10,25,50,100) are just projections = 0 new fits")

for ds, nf in datasets.items():
    # Estimate physical fits per seed
    tc = min(200, 2 ** (nf - 1))
    random_fits = nf * tc  # ~one per feature-trial, mostly unique
    guided_fits = nf * tc  # same
    if nf <= 16:
        exact_fits = nf * (2 ** (nf - 1))
    else:
        exact_fits = 0
    # Hybrid: zero extra, uses subset of max-budget checkpoint
    physical = random_fits + guided_fits + exact_fits
    print(
        f"  {ds:22s}: {physical:>6d} physical fits per seed x 5 seeds = {physical * 5:>8d} fits"
    )

print("\n  These are the *actual* model.fit() calls.")
print("  549 is the manifest job count (budget projections).")
print("  Physical work = 35 bundles, each running ~200 coalition evals per feature.")

# Per-seed time estimate
per_fit = {
    "abalone": 0.15,
    "spambase": 0.16,
    "madelon": 0.17,
    "isolet": 0.18,
    "gisette": 0.18,
    "credit_card_fraud": 0.58,
    "crimedata": 0.16,
}
print("\n=== Per-seed time estimate ===")
total_h = 0
for ds, nf in datasets.items():
    tc = min(200, 2 ** (nf - 1))
    fits = nf * tc * 2  # random + guided
    if nf <= 16:
        fits += nf * (2 ** (nf - 1))
    h = fits * per_fit[ds] / 3600
    total_h += h * SEEDS
    print(
        f"  {ds:22s}: {fits:>6d} fits x {per_fit[ds]:.2f}s = {fits * per_fit[ds]:.0f}s = {h:.1f}h per seed"
    )

print(f"\n  Estimator total: {total_h:.1f}h across {SEEDS} seeds")
print(f"  ≈ {total_h / 24:.1f} days")
