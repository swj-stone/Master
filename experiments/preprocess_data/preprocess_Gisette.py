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

try:
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )
except ModuleNotFoundError:  # Direct script execution from this directory.
    from raw_snapshot import load_registered_snapshot, materialize_registered_snapshot

TARGET = "label"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(*, project_root: Path | None = None) -> None:
    project_root = (
        Path(__file__).resolve().parent.parent.parent
        if project_root is None
        else Path(project_root).resolve()
    )
    snapshot = load_registered_snapshot(
        project_root / "configs" / "gisette.yaml",
        expected_dataset_id="gisette",
    )

    def produce(destination: Path) -> None:
        print("Fetching from OpenML name='gisette' ...")
        bunch = fetch_openml(
            name="gisette",
            version=1,
            as_frame=False,
            parser="auto",
            cache=False,
        )
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
        df.to_csv(destination, index=False, lineterminator="\r\n")

    banner("Gisette — registered raw snapshot")
    raw_path = materialize_registered_snapshot(snapshot, produce)
    df = pd.read_csv(raw_path)
    print(f"  Shape: {df.shape}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
