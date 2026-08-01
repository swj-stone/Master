"""
Preprocess the Gisette dataset (leak-free, split-first protocol).

Dataset: Gisette — 5000 features, 7000 samples, binary.
Source: OpenML name="gisette", version=1.

Output: ``data/raw/gisette.csv`` only.

Run:
    ../venv/Scripts/python.exe experiments/preprocess_data/preprocess_Gisette.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.datasets import fetch_openml

TARGET = "label"


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_path = raw_dir / "gisette.csv"

    banner("Gisette — download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Loading cached {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print("Fetching from OpenML name='gisette' ...")
        bunch = fetch_openml(name="gisette", version=1, as_frame=False, parser="auto")
        X_data = bunch.data
        if issparse(X_data):
            X = X_data.toarray().astype(float)
        else:
            X = np.asarray(X_data, dtype=float)
        y_raw = np.asarray(bunch.target)
        labels = sorted(np.unique(y_raw))
        y_mapped = np.array([0 if v == labels[0] else 1 for v in y_raw])
        cols = [f"f{i}" for i in range(1, X.shape[1] + 1)]
        full = np.column_stack([X, y_mapped.reshape(-1, 1)])
        df = pd.DataFrame(full, columns=cols + [TARGET])
        df[TARGET] = df[TARGET].astype(int)
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}")

    banner("Gisette — drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(raw_path, index=False)
    print(f"  After dedup: {df.shape}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
