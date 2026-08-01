"""
Preprocess the Communities & Crime dataset (leak-free, split-first protocol).

Dataset: Communities & Crime — ~121 features, 2215 samples, regression.
Source: shapG mirror (GitHub).

Output: ``data/raw/crimedata.csv`` only.

Run:
    ../venv/Scripts/python.exe experiments/preprocess_data/preprocess_crimedata.py
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

TARGET = "nonViolPerPop"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(*, project_root: Path | None = None) -> None:
    project_root = (
        Path(__file__).resolve().parent.parent.parent
        if project_root is None
        else Path(project_root).resolve()
    )
    snapshot = load_registered_snapshot(
        project_root / "configs" / "crimedata.yaml",
        expected_dataset_id="crimedata",
    )

    def produce(destination: Path) -> None:
        url = "https://raw.githubusercontent.com/vectorsss/shapG/master/datasets/crimedata_processing.csv"
        print(f"Downloading from {url}")
        df = pd.read_csv(url)
        df.to_csv(destination, index=False, lineterminator="\r\n")

    banner("Crimedata — registered raw snapshot")
    raw_path = materialize_registered_snapshot(snapshot, produce)
    df = pd.read_csv(raw_path)
    print(f"  Shape: {df.shape}")
    print(
        f"  Target: min={df[TARGET].min():.3f}  mean={df[TARGET].mean():.3f}  max={df[TARGET].max():.3f}"
    )

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
