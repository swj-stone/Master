"""
Preprocess the ISOLET dataset (leak-free, split-first protocol).

Dataset: ISOLET — 617 features, 7797 samples, 26-class.
Source: OpenML name="isolet", version=1.

Output: ``data/raw/isolet.csv`` only.

Run:
    ../venv/Scripts/python.exe experiments/preprocess_data/preprocess_ISOLET.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

FEATURE_NAMES = [f"f{i}" for i in range(1, 618)]
TARGET = "letter"
N_CLASSES = 26


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_path = raw_dir / "isolet.csv"

    banner("ISOLET — download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Loading cached {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print("Fetching from OpenML name='isolet' ...")
        bunch = fetch_openml(name="isolet", version=1, as_frame=True, parser="auto")
        X = bunch.data.astype(float)
        y = bunch.target
        label_map = {lbl: i for i, lbl in enumerate(sorted(y.unique()))}
        y_mapped = y.map(label_map)
        X.columns = FEATURE_NAMES
        df = pd.concat([X, pd.Series(y_mapped.values, name=TARGET, index=X.index)], axis=1)
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}  Classes: {df[TARGET].nunique()}")

    banner("ISOLET — drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(raw_path, index=False)
    print(f"  After dedup: {df.shape}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
