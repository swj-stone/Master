"""
Preprocess the Abalone dataset (leak-free, split-first protocol).

Dataset: Abalone — 8 features, 4177 samples, regression.
Source: Vector-s-Lab mirror (GitHub).

Output: ``data/raw/abalone.csv`` only.

Run:
    ../venv/Scripts/python.exe experiments/preprocess_data/preprocess_abalone.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from experiments.preprocess_data.raw_snapshot import (
        load_registered_snapshot,
        materialize_registered_snapshot,
    )
except ModuleNotFoundError:  # Direct script execution from this directory.
    from raw_snapshot import load_registered_snapshot, materialize_registered_snapshot

TARGET = "Rings"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(*, project_root: Path | None = None) -> None:
    project_root = (
        Path(__file__).resolve().parent.parent.parent
        if project_root is None
        else Path(project_root).resolve()
    )
    snapshot = load_registered_snapshot(
        project_root / "configs" / "abalone.yaml",
        expected_dataset_id="abalone",
    )

    def produce(destination: Path) -> None:
        url = "https://raw.githubusercontent.com/Vector-s-Lab/LinearRegression/main/data/abalone_processing.csv"
        print(f"Downloading from {url}")
        df = pd.read_csv(url)
        df.to_csv(destination, index=False, lineterminator="\r\n")

    banner("Abalone — registered raw snapshot")
    raw_path = materialize_registered_snapshot(snapshot, produce)
    df = pd.read_csv(raw_path)
    print(f"  Shape: {df.shape}")
    print(
        f"  Target: min={df[TARGET].min():.2f}  mean={df[TARGET].mean():.2f}  max={df[TARGET].max():.2f}"
    )

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
