"""
Preprocess the Communities & Crime dataset (leak-free, split-first protocol).

Dataset: Communities & Crime — ~121 features, 2215 samples, regression.
Source: shapG mirror (GitHub).

Output: ``data/raw/crimedata.csv`` only.

Run:
    python experiments/preprocess_data/preprocess_crimedata.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "nonViolPerPop"


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_path = raw_dir / "crimedata.csv"

    banner("Crimedata — download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Loading cached {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        url = "https://raw.githubusercontent.com/vectorsss/shapG/master/datasets/crimedata_processing.csv"
        print(f"Downloading from {url}")
        df = pd.read_csv(url)
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}")
    print(f"  Target: min={df[TARGET].min():.3f}  mean={df[TARGET].mean():.3f}  max={df[TARGET].max():.3f}")

    banner("Crimedata — drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(raw_path, index=False)
    print(f"  After dedup: {df.shape}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
