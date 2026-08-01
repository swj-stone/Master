"""
Preprocess the Spambase dataset (leak-free, split-first protocol).

Dataset: Spambase — 57 features, 4601 samples, binary.
Source: UCI ML Repository.

Output: ``data/raw/spambase.csv`` only.

Run:
    ../venv/Scripts/python.exe experiments/preprocess_data/preprocess_Spambase.py
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

TARGET = "is_spam"

WORD_FREQ = [
    "make",
    "address",
    "all",
    "3d",
    "our",
    "over",
    "remove",
    "internet",
    "order",
    "mail",
    "receive",
    "will",
    "people",
    "report",
    "addresses",
    "free",
    "business",
    "email",
    "you",
    "credit",
    "your",
    "font",
    "000",
    "money",
    "hp",
    "hpl",
    "george",
    "650",
    "lab",
    "labs",
    "telnet",
    "857",
    "data",
    "415",
    "85",
    "technology",
    "1999",
    "parts",
    "pm",
    "direct",
    "cs",
    "meeting",
    "original",
    "project",
    "re",
    "edu",
    "table",
    "conference",
]
CHAR_FREQ = [";", "(", "[", "!", "$", "#"]
CAP_RUN = [
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]
ALL_COLS = (
    [f"word_freq_{w}" for w in WORD_FREQ]
    + [f"char_freq_{c}" for c in CHAR_FREQ]
    + CAP_RUN
)


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(*, project_root: Path | None = None) -> None:
    project_root = (
        Path(__file__).resolve().parent.parent.parent
        if project_root is None
        else Path(project_root).resolve()
    )
    snapshot = load_registered_snapshot(
        project_root / "configs" / "spambase.yaml",
        expected_dataset_id="spambase",
    )

    def produce(destination: Path) -> None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        print(f"Downloading from {url}")
        df = pd.read_csv(url, header=None)
        df.columns = ALL_COLS + [TARGET]
        df.to_csv(destination, index=False, lineterminator="\r\n")

    banner("Spambase — registered raw snapshot")
    raw_path = materialize_registered_snapshot(snapshot, produce)
    df = pd.read_csv(raw_path)
    print(f"  Shape: {df.shape}")

    banner("Done")
    print(f"Raw cache: {raw_path}")


if __name__ == "__main__":
    main()
