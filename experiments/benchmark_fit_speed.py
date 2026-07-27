"""Benchmark XGBoost CUDA fit times for run-time estimation."""
import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split


def benchmark(label, n_samples, n_features, n_trials=30):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, n_samples)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        tree_method="hist", device="cuda", n_jobs=1, verbosity=0,
    )

    # warmup
    clf.fit(X_train[:10, :5], y_train[:10])
    _ = clf.predict_proba(X_val[:10, :5])[:, 1]

    times = []
    for _ in range(n_trials):
        subset_size = rng.integers(5, min(n_features + 1, n_features))
        subset = rng.choice(n_features, size=subset_size, replace=False)
        t0 = time.perf_counter()
        clf.fit(X_train[:, subset], y_train)
        _ = clf.predict_proba(X_val[:, subset])[:, 1]
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(
        f"{label}: median={np.median(times):.4f}s  mean={np.mean(times):.4f}s  "
        f"p95={np.percentile(times, 95):.4f}s  min={np.min(times):.4f}s"
    )
    return np.median(times)


def main():
    print("=== XGBoost CUDA fit-time benchmark ===")
    print(f"XGBoost version: {xgb.__version__}")
    print()

    # Abalone-like: 8 features, ~4k rows
    benchmark("Abalone-like     (8f, 4k rows)", 4177, 8)
    # Spambase-like: 56 features, ~4.6k rows
    benchmark("Spambase-like   (56f, 4.6k rows)", 4601, 56)
    # Madelon-like: 100 features, ~2.6k rows
    benchmark("Madelon-like   (100f, 2.6k rows)", 2600, 100)
    # ISOLET-like: 100 features, ~7.8k rows
    benchmark("ISOLET-like    (100f, 7.8k rows)", 7797, 100)
    # Gisette-like: 200 features, ~7k rows
    benchmark("Gisette-like   (200f, 7k rows)", 7000, 200)
    # Crimedata-like: ~119 features, ~2k rows
    benchmark("Crimedata-like (119f, 2k rows)", 1994, 119)
    # Credit-Card scaling test (small sample to estimate large)
    med_5k = benchmark("Credit-5k       (29f, 5k rows)", 5000, 29, n_trials=15)
    med_20k = benchmark("Credit-20k      (29f, 20k rows)", 20000, 29, n_trials=10)
    med_50k = benchmark("Credit-50k      (29f, 50k rows)", 50000, 29, n_trials=10)

    # Credit-Card-Fraud has ~285k rows with ~29 features after MI
    # Fit time scales roughly O(n_samples * n_estimators * n_features * max_depth)
    # but GPU parallelizes well. Let's estimate from 50k benchmark
    scaling_285k = 284807 / 50000
    est_credit = med_50k * scaling_285k
    print()
    print(f"Credit-Card est (284807 rows, 29f): ~{est_credit:.2f}s per fit")
    print(f"  (based on 50k benchmark × {scaling_285k:.1f}x)")

    # Also check if scaling is sub-linear due to GPU parallelism
    ratio_5k_20k = med_20k / med_5k if med_5k > 0 else 0
    ratio_20k_50k = med_50k / med_20k if med_20k > 0 else 0
    print(f"  Scaling: 5k->20k: {ratio_5k_20k:.2f}x (ideal 4x), "
          f"20k->50k: {ratio_20k_50k:.2f}x (ideal 2.5x)")


if __name__ == "__main__":
    main()
