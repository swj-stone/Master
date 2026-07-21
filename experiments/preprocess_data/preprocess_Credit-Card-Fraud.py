"""
Preprocess the Credit-Card-Fraud dataset (leak-free, split-first protocol).

Dataset: Credit Card Fraud Detection — 29 features, 284807 samples, binary.
Source: OpenML name="creditcard".

Output: ``data/processed_v2/``.

Run:
    python experiments/preprocess_data/preprocess_Credit-Card-Fraud.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from minppi_mofs.data_protocol import prepare_split

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
TASK = "binary"
SEEDS = (42, 123, 456, 789, 1024)
MAX_FEATURES = 100  # will keep all 29 since 29 < 100
PREFIX = "credit_card_fraud"


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def _save(split, dest_dir: Path, seed: int) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for tag, X_df, y_arr in [
        ("train", split.X_train, split.y_train),
        ("val", split.X_val, split.y_val),
        ("test", split.X_test, split.y_test),
    ]:
        out = X_df.copy()
        out[TARGET] = y_arr
        out.to_csv(dest_dir / f"{PREFIX}_s{seed}_{tag}.csv", index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed_v2"
    raw_path = raw_dir / "credit_card_fraud.csv"

    banner("Credit-Card-Fraud — download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Loading cached {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print("Fetching from OpenML name='creditcard' ...")
        bunch = fetch_openml(name="creditcard", as_frame=True, parser="auto")
        df = bunch.frame
        y = df["Class"].map({"0": 0, "1": 1}).astype(int)
        X = df[[c for c in df.columns if c != "Class"]].copy()
        X.columns = FEATURE_NAMES
        df = pd.concat([X, pd.Series(y.values, name=TARGET, index=X.index)], axis=1)
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}")
    fraud_pct = df[TARGET].mean() * 100
    print(f"  Fraud: {fraud_pct:.4f}%")

    banner("Credit-Card-Fraud — drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  After dedup: {df.shape}")

    X, y = df.drop(columns=[TARGET]), df[TARGET].values.astype(int)

    for seed in SEEDS:
        banner(f"Credit-Card-Fraud — seed {seed}")
        split = prepare_split(
            X, y, task=TASK, seed=seed, max_features=MAX_FEATURES,
        )
        _save(split, processed_dir, seed)
        print(f"  train={len(split.y_train)}  val={len(split.y_val)}  test={len(split.y_test)}")
        print(f"  features={len(split.selected_features)}")

    banner("Done")
    print(f"Output: {processed_dir}/")


if __name__ == "__main__":
    main()
