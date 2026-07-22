"""
Preprocess the Credit-Card-Fraud dataset (leak-free, split-first protocol).

Dataset: Credit Card Fraud Detection — 29 features, 284807 samples, binary.
Source: OpenML name="creditcard".

Output: ``data/raw/credit_card_fraud.csv`` only.

Run:
    python experiments/preprocess_data/preprocess_Credit-Card-Fraud.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
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
    df.to_csv(raw_path, index=False)
    print(f"  After dedup: {df.shape}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
