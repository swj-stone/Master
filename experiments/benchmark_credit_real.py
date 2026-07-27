"""Benchmark XGBoost on actual Credit-Card-Fraud train split size."""
import numpy as np, pandas as pd, time, xgboost as xgb
from pathlib import Path

# Load actual Credit-Card training split to match real data size
train_path = Path("data/processed/credit_card_fraud_s42_train.csv")
df = pd.read_csv(train_path)
X = df.drop(columns=["Class"]).values
y = df["Class"].values.astype(int)
n_samples, n_features = X.shape
print(f"Credit-Card train: {n_samples} rows × {n_features} features")

# XGBoost with same params as pipeline
clf = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    tree_method="hist", device="cuda", n_jobs=1, verbosity=0,
)

# Warmup on small subset
clf.fit(X[:50, :5], y[:50])

rng = np.random.default_rng(42)
times = []
for i in range(15):
    subset_size = rng.integers(5, min(n_features + 1, n_features))
    subset = rng.choice(n_features, size=subset_size, replace=False)
    t0 = time.perf_counter()
    clf.fit(X[:, subset], y)
    times.append(time.perf_counter() - t0)

times = np.array(times)
print(f"Credit-Card actual ({n_samples} rows, {n_features}f): "
      f"median={np.median(times):.4f}s mean={np.mean(times):.4f}s "
      f"p95={np.percentile(times, 95):.4f}s")
