"""Runtime estimate (K=100 budgets) based on measured K=200 timing, halved."""

per_fit = {
    "abalone": 0.15,
    "spambase": 0.17,
    "madelon": 0.17,
    "isolet": 0.18,
    "gisette": 0.18,
    "credit_card_fraud": 0.58,
    "crimedata": 0.16,
}
datasets = {
    "abalone": 8,
    "spambase": 56,
    "madelon": 100,
    "isolet": 100,
    "gisette": 100,
    "credit_card_fraud": 29,
    "crimedata": 100,
}

# MEASURED at K=200: spambase random_mc 20.7min / 21 features ≈ 1.0 min/feature
# With K=100: 2 fits × 100 coalitions = 200 fits/feature (half of K=200)
# Estimated random_mc ≈ 0.5 min/feature for spambase
# Guided remains ~5× random (adversarial search per trial)

measured_random_min_per_feat_K200 = 20.7 / 21  # 1.0 min/feat @ K=200
scaling = 100 / 200  # K=100 → half the coalitions → half the time for random_mc
measured_random_min_per_feat = (
    measured_random_min_per_feat_K200 * scaling
)  # 0.5 min/feat @ K=100

# guided is ~5× random due to multi-step perturbation per trial
guided_factor = 5.0

print(
    f"Measured K=200: {measured_random_min_per_feat_K200:.1f} min/feat (spambase random_mc)"
)
print(
    f"Scaled K=100:   {measured_random_min_per_feat:.2f} min/feat (spambase random_mc)"
)
print(
    f"Guided factor:  {guided_factor:.0f}× (measured from K=200 run: 3 min/feat guided vs 0.6 min/feat random)"
)
print()

# Cost per feature = per_fit × 200 (2 fits × 100 coalitions)
spambase_ref = 0.17 * 200  # reference cost per feature @ K=100
total_h = 0
print("=== Formal run estimate (K=100) ===")
print(f"{'Dataset':<22} {'feat':>5} {'min/feat':>9} {'h/seed':>8} {'h/5seeds':>9}")
print("-" * 58)
for ds, nf in datasets.items():
    cost_per_feature = per_fit[ds] * 200  # 2 fits × 100 coalitions
    random_min_per_feat = measured_random_min_per_feat * (
        cost_per_feature / spambase_ref
    )
    guided_min_per_feat = random_min_per_feat * guided_factor
    h_per_seed = (random_min_per_feat + guided_min_per_feat) * nf / 60
    abalone_factor = 1.3 if ds == "abalone" else 1.0  # + exact (small overhead)
    h_per_seed *= abalone_factor
    h_5seeds = h_per_seed * 5
    total_h += h_5seeds
    blended = (random_min_per_feat + guided_min_per_feat) / 2
    print(f"{ds:<22} {nf:>5} {blended:>7.1f}m {h_per_seed:>8.1f} {h_5seeds:>8.1f}")

print("-" * 58)
print(f"{'Estimator total':<22} {'':>5} {'':>9} {'':>8} {total_h:>8.1f}")
other = total_h * 0.15
print(f"\n  + nsga2 + removal + alpha + ablation + summarize + figures ≈ {other:.0f}h")
print(f"  Grand total ≈ {total_h + other:.0f}h ({(total_h + other) / 24:.1f} days)")
