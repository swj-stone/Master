"""
Preprocess the Credit-Card-Fraud dataset (leak-free, split-first protocol).

Dataset: Credit Card Fraud Detection — 29 features, 284807 samples, binary.
Source: OpenML name="creditcard".

Output: ``data/raw/credit_card_fraud.csv`` only.

Run:
    ../venv/Scripts/python.exe experiments/preprocess_data/preprocess_Credit-Card-Fraud.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

try:
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )
except ModuleNotFoundError:  # Direct script execution from this directory.
    from raw_snapshot import load_registered_snapshot, materialize_registered_snapshot

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(*, project_root: Path | None = None) -> None:
    project_root = (
        Path(__file__).resolve().parent.parent.parent
        if project_root is None
        else Path(project_root).resolve()
    )
    snapshot = load_registered_snapshot(
        project_root / "configs" / "credit_card_fraud.yaml",
        expected_dataset_id="credit_card_fraud",
    )

    def produce(destination: Path) -> None:
        print("Fetching from OpenML data_id=1597 ...")
        bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto", cache=False)
        df = bunch.frame
        y = df["Class"].map({"0": 0, "1": 1}).astype(int)
        X = df[[c for c in df.columns if c != "Class"]].copy()
        X.columns = FEATURE_NAMES
        df = pd.concat([X, pd.Series(y.values, name=TARGET, index=X.index)], axis=1)
        df.to_csv(destination, index=False, lineterminator="\r\n")

    banner("Credit-Card-Fraud — registered raw snapshot")
    raw_path = materialize_registered_snapshot(snapshot, produce)
    df = pd.read_csv(raw_path)
    print(f"  Shape: {df.shape}")
    fraud_pct = df[TARGET].mean() * 100
    print(f"  Fraud: {fraud_pct:.4f}%")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
