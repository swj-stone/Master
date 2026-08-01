"""
Preprocess the Madelon dataset (leak-free, split-first protocol).

Dataset: Madelon — 500 features, 2600 samples, binary.
Source: OpenML ID 1485.

Processing: download, clean, and cache the raw CSV only.

Run:
    ../venv/Scripts/python.exe experiments/preprocess_data/preprocess_Madelon.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

FEATURE_NAMES = [f"f{i}" for i in range(1, 501)]
TARGET = "label"


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_path = raw_dir / "madelon.csv"

    banner("Madelon — download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Loading cached {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print("Fetching from OpenML ID 1485 ...")
        bunch = fetch_openml(data_id=1485, as_frame=True, parser="auto")
        X = bunch.data.astype(float)
        y = bunch.target
        labels = sorted(y.unique())
        y_mapped = y.map({labels[0]: 0, labels[1]: 1})
        X.columns = FEATURE_NAMES
        df = pd.concat([X, pd.Series(y_mapped.values, name=TARGET, index=X.index)], axis=1)
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}")

    banner("Madelon — drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(raw_path, index=False)
    print(f"  After dedup: {df.shape}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
