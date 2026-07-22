"""
Preprocess the Abalone dataset (leak-free, split-first protocol).

Dataset: Abalone — 8 features, 4177 samples, regression.
Source: Vector-s-Lab mirror (GitHub).

Output: ``data/raw/abalone.csv`` only.

Run:
    python experiments/preprocess_data/preprocess_abalone.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Rings"


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_path = raw_dir / "abalone.csv"

    banner("Abalone — download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Loading cached {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        url = "https://raw.githubusercontent.com/Vector-s-Lab/LinearRegression/main/data/abalone_processing.csv"
        print(f"Downloading from {url}")
        df = pd.read_csv(url)
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}")
    print(f"  Target: min={df[TARGET].min():.2f}  mean={df[TARGET].mean():.2f}  max={df[TARGET].max():.2f}")

    banner("Abalone — drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(raw_path, index=False)
    print(f"  After dedup: {df.shape}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
