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

try:
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )
except ModuleNotFoundError:  # Direct script execution from this directory.
    from raw_snapshot import load_registered_snapshot, materialize_registered_snapshot

FEATURE_NAMES = [f"f{i}" for i in range(1, 618)]
TARGET = "letter"
N_CLASSES = 26


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(*, project_root: Path | None = None) -> None:
    project_root = (
        Path(__file__).resolve().parent.parent.parent
        if project_root is None
        else Path(project_root).resolve()
    )
    snapshot = load_registered_snapshot(
        project_root / "configs" / "isolet.yaml",
        expected_dataset_id="isolet",
    )

    def produce(destination: Path) -> None:
        print("Fetching from OpenML name='isolet' ...")
        bunch = fetch_openml(
            name="isolet",
            version=1,
            as_frame=True,
            parser="auto",
            cache=False,
        )
        X = bunch.data.astype(float)
        y = bunch.target
        label_map = {lbl: i for i, lbl in enumerate(sorted(y.unique()))}
        y_mapped = y.map(label_map)
        X.columns = FEATURE_NAMES
        df = pd.concat(
            [X, pd.Series(y_mapped.values, name=TARGET, index=X.index)], axis=1
        )
        df.to_csv(destination, index=False, lineterminator="\r\n")

    banner("ISOLET — registered raw snapshot")
    raw_path = materialize_registered_snapshot(snapshot, produce)
    df = pd.read_csv(raw_path)
    print(f"  Shape: {df.shape}  Classes: {df[TARGET].nunique()}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
