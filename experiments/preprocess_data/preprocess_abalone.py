"""
Preprocess the Abalone dataset (leak-free, split-first protocol).

Dataset: Abalone — 8 features, 4177 samples, regression.
Source: Vector-s-Lab mirror (GitHub).

Output: ``data/processed_v2/``.

Run:
    python experiments/preprocess_data/preprocess_abalone.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from minppi_mofs.data_protocol import prepare_split

TARGET = "Rings"
TASK = "regression"
SEEDS = (42, 123, 456, 789, 1024)
MAX_FEATURES = 100  # will keep all 8 since 8 < 100
PREFIX = "abalone"


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def _save(split, dest_dir: Path, seed: int) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for tag, X_df, y_arr in [
        ("train", split.X_train, split.y_train),
        ("val", split.X_val, split.y_val),
        ("test", split.X_test, split.y_test),
    ]:
        out = X_df.copy()
        out[TARGET] = y_arr
        out.to_csv(dest_dir / f"{PREFIX}_s{seed}_{tag}.csv", index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed_v2"
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
    print(f"  After dedup: {df.shape}")

    X, y = df.drop(columns=[TARGET]), df[TARGET].values.astype(float)

    for seed in SEEDS:
        banner(f"Abalone — seed {seed}")
        split = prepare_split(
            X, y, task=TASK, seed=seed, max_features=MAX_FEATURES,
        )
        _save(split, processed_dir, seed)
        print(f"  train={len(split.y_train)}  val={len(split.y_val)}  test={len(split.y_test)}")
        print(f"  features={len(split.selected_features)}")

    banner("Done")
    print(f"Output: {processed_dir}/")


if __name__ == "__main__":
    main()
