"""
Friedman leave-one-out sensitivity — answers the reviewer concern that
Friedman significance on AUC@10/20 might be driven by a single outlier
dataset (e.g., ShapG's unusually low AUC on ISOLET).

For each classification metric (AUC@10, AUC@20), the script computes
the Friedman chi-squared and p-value (i) on the full set of 5 classification
datasets, and (ii) on each leave-one-out subset (drop one dataset, recompute).

Data reuse
----------
- ``results/tables/2026-06-28_07-44-10/main_metrics.csv``:
  8 methods × 7 datasets × 5 seeds → filtered to 5 classification datasets.

Output (``results/tables/<run_id>/``)::

    friedman_leave_one_out.csv    # per-metric per-dataset chi2/p

Run::

    python experiments/sensitivity_friedman.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_CSV = "results/tables/2026-06-28_07-44-10/main_metrics.csv"
METRICS = ["AUC@10", "AUC@20"]
CLASSIFICATION_DATASETS = [
    "credit_card_fraud", "gisette", "isolet", "madelon", "spambase",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(project_root / INPUT_CSV)
    cls_df = df[df["dataset"].isin(CLASSIFICATION_DATASETS)]

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = project_root / "results" / "tables" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    (out_dir / "metadata.json").write_text(
        json.dumps({
            "script": "sensitivity_friedman.py",
            "input": INPUT_CSV,
            "datasets": CLASSIFICATION_DATASETS,
            "metrics": METRICS,
        }, indent=2),
        encoding="utf-8",
    )

    rows: list[dict] = []

    for metric in METRICS:
        # Full
        agg_full = (
            cls_df.groupby(["method", "dataset"])[metric].mean().reset_index()
        )
        pivot_full = (
            agg_full.pivot(index="dataset", columns="method", values=metric)
            .dropna()
        )
        k_full, n_full = pivot_full.shape[1], pivot_full.shape[0]
        stat_f, p_f = friedmanchisquare(
            *[pivot_full[c].values for c in pivot_full.columns]
        )

        print(f"\n{'=' * 60}")
        print(f"{metric}  (FULL: N={n_full}, k={k_full})")
        print(f"  chi2 = {stat_f:.3f},  p = {p_f:.4f}")
        rows.append({
            "metric": metric, "dropped": "(full)",
            "n_datasets": n_full, "chi2": round(stat_f, 3),
            "p_value": round(p_f, 5),
        })

        # Leave-one-out
        for drop_ds in CLASSIFICATION_DATASETS:
            sub = cls_df[cls_df["dataset"] != drop_ds]
            agg = sub.groupby(["method", "dataset"])[metric].mean().reset_index()
            piv = agg.pivot(index="dataset", columns="method", values=metric).dropna()
            stat, p = friedmanchisquare(
                *[piv[c].values for c in piv.columns]
            )
            sig_mark = " *" if p < 0.05 else ""
            print(
                f"  - {drop_ds:25s}: chi2 = {stat:.3f},  p = {p:.4f}{sig_mark}"
            )
            rows.append({
                "metric": metric, "dropped": drop_ds,
                "n_datasets": piv.shape[0],
                "chi2": round(stat, 3), "p_value": round(p, 5),
            })

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "friedman_leave_one_out.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Inverse question: if we drop madelon, does significance increase?
    print("\n=== Key finding ===")
    for metric in METRICS:
        full_row = out_df[(out_df["metric"] == metric) & (out_df["dropped"] == "(full)")].iloc[0]
        mad_row = out_df[(out_df["metric"] == metric) & (out_df["dropped"] == "madelon")].iloc[0]
        iso_row = out_df[(out_df["metric"] == metric) & (out_df["dropped"] == "isolet")].iloc[0]
        print(f"{metric}:")
        print(f"  Full chi2={full_row['chi2']:.1f} (p={full_row['p_value']:.4f})")
        print(f"  Drop Madelon:  chi2={mad_row['chi2']:.1f} (p={mad_row['p_value']:.4f})  ← {'significance INCREASES' if mad_row['chi2'] > full_row['chi2'] else 'significance decreases'}")
        print(f"  Drop ISOLET:   chi2={iso_row['chi2']:.1f} (p={iso_row['p_value']:.4f})  ← {'significance DECREASES (ShapG outlier?)' if iso_row['chi2'] < full_row['chi2'] else 'significance increases'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
