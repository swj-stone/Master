"""
Download and preprocess the Spambase dataset from UCI ML Repository.

Dataset: Spambase
  - Features: 57 (word/char frequencies + run-lengths)
  - Samples: 4601
  - Classes: spam (1) / non-spam (0)
  - Source: https://archive.ics.uci.edu/dataset/94/spambase

Processing steps:
  1. Download raw data from UCI.
  2. Add column names, save to data/raw/spambase.csv.
  3. Feature engineering (drop duplicates, low-variance filter, cap features).
  4. Standardize features (zero mean, unit variance).
  5. Stratified split: 60% train / 20% val / 20% test.
  6. Save train/val/test CSV files to data/processed/.

Run:
    python experiments/preprocess_data/preprocess_Spambase.py --config configs/spambase.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Spambase column names (57 features + 1 label)
# ---------------------------------------------------------------------------
WORD_FREQ_WORDS = [
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

CHAR_FREQ_CHARS = [";", "(", "[", "!", "$", "#"]

CAPITAL_RUN_FEATURES = [
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]

FEATURE_NAMES = (
    [f"word_freq_{w}" for w in WORD_FREQ_WORDS]
    + [f"char_freq_{c}" for c in CHAR_FREQ_CHARS]
    + CAPITAL_RUN_FEATURES
)


def banner(title: str) -> None:
    """Print a section banner."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_config(config_path: Path) -> dict:
    """Load preprocessing configuration from YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_spambase(url: str, raw_path: Path, target_name: str) -> pd.DataFrame:
    """Download Spambase from UCI and return as a DataFrame with column names.

    Args:
        url: UCI data URL.
        raw_path: Where to save the raw CSV.
        target_name: Name of the target column.

    Returns:
        DataFrame with 57 feature columns + label.
    """
    print("Downloading Spambase from UCI...")
    df = pd.read_csv(url, header=None)
    df.columns = FEATURE_NAMES + [target_name]
    return df


def engineer_features(
    df: pd.DataFrame,
    target_name: str,
    *,
    drop_duplicates: bool,
    variance_threshold: float,
    max_features: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Clean and select features.

    Applies in order: drop duplicates, drop low-variance features, cap to
    max_features (select top-N by variance).

    Args:
        df: Raw DataFrame with features + label.
        target_name: Name of the target column.
        drop_duplicates: Whether to drop duplicate rows.
        variance_threshold: Minimum variance threshold (columns below are dropped).
        max_features: Maximum number of feature columns to keep.
        seed: Random seed for tie-breaking.

    Returns:
        Tuple of (cleaned DataFrame, list of kept feature names).
    """
    n_before = len(df)

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        n_dup = n_before - len(df)
        if n_dup > 0:
            print(f"  Dropped {n_dup} duplicate rows ({n_before} -> {len(df)})")

    X_cols = [c for c in df.columns if c != target_name]
    X = df[X_cols]

    # Variance-based filtering
    sel = VarianceThreshold(threshold=variance_threshold)
    X_sel = sel.fit_transform(X)
    kept_mask = sel.get_support()
    kept_cols = [X_cols[i] for i in range(len(X_cols)) if kept_mask[i]]
    n_dropped = len(X_cols) - len(kept_cols)
    if n_dropped > 0:
        print(
            f"  Dropped {n_dropped} low-variance features (threshold={variance_threshold})"
        )

    # Cap to max_features, selecting top-N by variance
    X_kept = pd.DataFrame(X_sel, columns=kept_cols, index=df.index)
    p_kept = X_kept.shape[1]
    if p_kept > max_features:
        rng = np.random.default_rng(seed)
        variances = X_kept.var(ddof=0).values
        top_idx = rng.choice(
            p_kept,
            size=max_features,
            replace=False,
            p=variances / variances.sum(),
        )
        kept_cols = [kept_cols[j] for j in np.sort(top_idx)]
        print(f"  Capped features from {p_kept} to {max_features} (top by variance)")
    else:
        # Rebuild kept_cols from kept array
        pass  # kept_cols already correct from VarianceThreshold

    df_out = pd.concat(
        [X[kept_cols], df[[target_name]]],
        axis=1,
    )
    print(f"  After feature engineering: {df_out.shape[0]} x {len(kept_cols) + 1}")
    return df_out, kept_cols


def select_top_features(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Select top-100 features by mutual information with the target.

    Args:
        X: Feature DataFrame.
        y: Target array.
        seed: Random seed for mutual information computation.

    Returns:
        Tuple of (DataFrame with selected features, list of selected columns).
    """
    p = X.shape[1]
    k = 100

    mi = mutual_info_classif(X, y, random_state=seed)
    scores = pd.Series(mi, index=X.columns)
    top_cols = scores.abs().sort_values(ascending=False).head(k).index.tolist()

    print(
        f"  Feature selection (mutual info): {p} -> {len(top_cols)} "
        f"(rule: top{k})",
    )
    return X[top_cols], top_cols


def standardize_and_split(
    df: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    *,
    seed: int,
    test_size: float,
    val_size: float,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize features and split into train/val/test.

    Args:
        df: Cleaned DataFrame with features + label column.
        feature_names: List of feature column names.
        target_name: Name of the target column.
        seed: Random seed for reproducibility.
        test_size: Fraction of data for test split.
        val_size: Fraction of remaining data for validation split.
        stratify: Whether to use stratified split.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    X = df[feature_names].values
    y = df[target_name].values

    rng = np.random.default_rng(seed)
    stratify_y = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=rng.integers(0, 2**31),
        stratify=stratify_y,
    )
    stratify_y_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=rng.integers(0, 2**31),
        stratify=stratify_y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    def to_df(X_arr: np.ndarray, y_arr: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame(X_arr, columns=feature_names)
        out[target_name] = y_arr.astype(int)
        return out

    return (
        to_df(X_train_scaled, y_train),
        to_df(X_val_scaled, y_val),
        to_df(X_test_scaled, y_test),
    )


def save_processed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Path,
    train_filename: str,
    val_filename: str,
    test_filename: str,
) -> None:
    """Save train/val/test splits to CSV."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / train_filename, index=False)
    val_df.to_csv(processed_dir / val_filename, index=False)
    test_df.to_csv(processed_dir / test_filename, index=False)
    print(f"  Processed splits saved to {processed_dir}/")


def verify(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_name: str,
    n_expected: int,
) -> None:
    """Run sanity checks on the processed data."""
    n_total = len(train_df) + len(val_df) + len(test_df)
    assert n_total == n_expected, f"Row count mismatch: {n_total} vs {n_expected}"
    assert train_df[target_name].nunique() == 2, "Not binary classification"

    feature_cols = [c for c in train_df.columns if c != target_name]
    train_arr = train_df[feature_cols].values
    train_means = train_arr.mean(axis=0)
    train_stds = train_arr.std(axis=0, ddof=0)
    assert np.allclose(train_means, 0.0, atol=1e-6), (
        f"Train mean not zero: max abs = {np.abs(train_means).max():.2e}"
    )
    assert np.allclose(train_stds, 1.0, atol=1e-6), (
        f"Train std not one: max diff = {np.abs(train_stds - 1.0).max():.2e}"
    )

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        spam_pct = d[target_name].mean() * 100
        print(f"  {name}: {len(d):5d} rows,  spam={spam_pct:.1f}%")

    print("  All sanity checks passed.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess Spambase dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "spambase.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full download-engineer-preprocess-save-verify pipeline."""
    args = parse_args()
    cfg = load_config(args.config)

    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / cfg["paths"]["raw_dir"]
    processed_dir = project_root / cfg["paths"]["processed_dir"]
    raw_path = raw_dir / cfg["paths"]["raw_filename"]
    target_name = cfg["dataset"]["target_name"]
    split_cfg = cfg["split"]
    fe_cfg = cfg["feature_engineering"]

    banner(f"{cfg['dataset']['name']} Preprocessing")

    # Step 1: Download
    banner("Step 1: Download raw data")
    if raw_path.exists():
        print(f"Loading cached raw data from {raw_path}...")
        df = pd.read_csv(raw_path)
    else:
        df = download_spambase(cfg["dataset"]["url"], raw_path, target_name)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution: {df[target_name].value_counts().to_dict()}")

    # Step 2: Feature engineering (seed-independent)
    banner("Step 2: Feature engineering")
    df, feature_names = engineer_features(
        df,
        target_name,
        drop_duplicates=fe_cfg["drop_duplicates"],
        variance_threshold=fe_cfg["variance_threshold"],
        max_features=fe_cfg["max_features"],
        seed=split_cfg["feature_seed"],
    )

    # Step 2.5: Feature selection by mutual information
    y_arr = df[target_name].values
    df_X = df[feature_names]
    df_X_sel, feature_names = select_top_features(
        df_X, y_arr, seed=split_cfg["feature_seed"]
    )
    df = pd.concat([df_X_sel, df[[target_name]]], axis=1)

    prefix = cfg["paths"]["file_prefix"]
    seeds = split_cfg["seeds"]

    for seed in seeds:
        banner(f"Seed {seed}: Standardize, split, save, verify")

        train_df, val_df, test_df = standardize_and_split(
            df,
            feature_names,
            target_name,
            seed=seed,
            test_size=split_cfg["test_size"],
            val_size=split_cfg["val_size"],
            stratify=split_cfg["stratify"],
        )

        save_processed(
            train_df,
            val_df,
            test_df,
            processed_dir,
            f"{prefix}_s{seed}_train.csv",
            f"{prefix}_s{seed}_val.csv",
            f"{prefix}_s{seed}_test.csv",
        )

        verify(train_df, val_df, test_df, target_name, len(df))
        print()

    banner("Done")
    print(f"Final features: {len(feature_names)}")
    print(f"Splits: {len(seeds)} seeds x train/val/test = {len(seeds) * 3} files")
    print(f"Raw:     {raw_path}")
    print(f"Processed: {processed_dir}/")
    print("Next: run preprocess_Madelon.py")


if __name__ == "__main__":
    main()
