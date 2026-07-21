"""
Preprocess the Spambase dataset (leak-free, split-first protocol).

Dataset: Spambase — 57 features, 4601 samples, binary.
Source: UCI ML Repository.

Output: ``data/processed_v2/``.

Run:
    python experiments/preprocess_data/preprocess_Spambase.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from minppi_mofs.data_protocol import prepare_split

TARGET = "is_spam"
TASK = "binary"
SEEDS = (42, 123, 456, 789, 1024)
MAX_FEATURES = 100
PREFIX = "spambase"

WORD_FREQ = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference",
]
CHAR_FREQ = [";", "(", "[", "!", "$", "#"]
CAP_RUN = ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
ALL_COLS = [f"word_freq_{w}" for w in WORD_FREQ] + [f"char_freq_{c}" for c in CHAR_FREQ] + CAP_RUN


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
    raw_path = raw_dir / "spambase.csv"

    banner("Spambase — download")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Loading cached {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        print(f"Downloading from {url}")
        df = pd.read_csv(url, header=None)
        df.columns = ALL_COLS + [TARGET]
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}")

    banner("Spambase — drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  After dedup: {df.shape}")

    X, y = df.drop(columns=[TARGET]), df[TARGET].values.astype(int)

    for seed in SEEDS:
        banner(f"Spambase — seed {seed}")
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
