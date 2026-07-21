"""
Preprocess the Gisette dataset (leak-free, split-first protocol).

Dataset: Gisette — 5000 features, 7000 samples, binary.
Source: OpenML name="gisette", version=1.

Output: ``data/processed_v2/``.

Run:
    python experiments/preprocess_data/preprocess_Gisette.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.datasets import fetch_openml

from minppi_mofs.data_protocol import prepare_split

TARGET = "label"
TASK = "binary"
SEEDS = (42, 123, 456, 789, 1024)
MAX_FEATURES = 100
PREFIX = "gisette"


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
    print(f"  After dedup: {df.shape}")

    X, y = df.drop(columns=[TARGET]), df[TARGET].values.astype(int)

    for seed in SEEDS:
        banner(f"Gisette — seed {seed}")
        split = prepare_split(
            X, y, task=TASK, seed=seed, max_features=MAX_FEATURES,
        )
        _save(split, processed_dir, seed)
        pos_pct = y.mean() * 100
        print(f"  train={len(split.y_train)}  val={len(split.y_val)}  test={len(split.y_test)}")
        print(f"  features={len(split.selected_features)}  pos_pct={pos_pct:.1f}%")

    banner("Done")
    print(f"Output: {processed_dir}/")


if __name__ == "__main__":
    main()
