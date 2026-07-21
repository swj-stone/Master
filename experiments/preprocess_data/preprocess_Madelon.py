"""
Preprocess the Madelon dataset (leak-free, split-first protocol).

Dataset: Madelon — 500 features, 2600 samples, binary.
Source: OpenML ID 1485.

Processing: download → drop duplicates → prepare_split → save.

Output: ``data/processed_v2/`` (does NOT overwrite existing ``data/processed/``).

Run:
    python experiments/preprocess_data/preprocess_Madelon.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from minppi_mofs.data_protocol import prepare_split

FEATURE_NAMES = [f"f{i}" for i in range(1, 501)]
TARGET = "label"
TASK = "binary"
SEEDS = (42, 123, 456, 789, 1024)
MAX_FEATURES = 100
PREFIX = "madelon"


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
    raw_path = raw_dir / "madelon.csv"

    banner("Madelon — download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Loading cached {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print("Fetching from OpenML ID 1485 ...")
        bunch = fetch_openml(data_id=1485, as_frame=True, parser="auto")
        X = bunch.data.astype(float)
        y = bunch.target
        labels = sorted(y.unique())
        y_mapped = y.map({labels[0]: 0, labels[1]: 1})
        X.columns = FEATURE_NAMES
        df = pd.concat([X, pd.Series(y_mapped.values, name=TARGET, index=X.index)], axis=1)
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}")

    banner("Madelon — drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  After dedup: {df.shape}")

    X, y = df.drop(columns=[TARGET]), df[TARGET].values

    for seed in SEEDS:
        banner(f"Madelon — seed {seed}")
        split = prepare_split(
            X, y, task=TASK, seed=seed, max_features=MAX_FEATURES,
        )
        _save(split, processed_dir, seed)
        n_harmful = sum(1 for v in split.y_train if v == 1)
        print(f"  train={len(split.y_train)}  val={len(split.y_val)}  test={len(split.y_test)}")
        print(f"  features={len(split.selected_features)}  pos_pct={n_harmful/len(split.y_train)*100:.1f}%")

    banner("Done")
    print(f"Output: {processed_dir}/")


if __name__ == "__main__":
    main()
