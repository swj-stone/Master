"""Rebuild the six-source comparison of frozen forests and archived refitted gain."""
import pandas as pd

from experiments.minpi_compression.benchmark import ROOT


def main():
    evidence = ROOT / "thesis-paper/evidence/minpi_rework"
    output = evidence / "predictive-prepared"
    forest = pd.read_csv(output / "scores.csv")
    old = pd.read_csv(evidence / "quota-tree/historical-42-123-456-789-1024.csv")
    refit = old.loc[(old.method == "refit_gain") & (old.part == "primary")]
    choices = {"Full RF": forest[forest.method == "full"],
               "RF MDI": forest[forest.method == "mdi_min"],
               "RF MinPI": forest[forest.method == "mdi_plus_min_min"],
               "XGB Gain refit": refit}
    reference = choices["Full RF"].set_index(["dataset", "seed"]).p.sort_index()
    for rows in choices.values():
        inputs = rows.set_index(["dataset", "seed"]).p.sort_index()
        pd.testing.assert_series_equal(inputs, reference, check_dtype=False)
    fields = ["p", "k", "auc", "accuracy", "r2", "mae", "pipeline_s"]
    mean = (pd.concat(choices, names=["algorithm"]).reset_index(level="algorithm")
            .groupby(["dataset", "algorithm"])[fields].mean())
    mean.to_csv(output / "refit-comparison.csv")
    text = [
        "# Original six prepared datasets, five seeds",
        "",
        "The first three columns use the same fitted depth-four random forest and original test rows. "
        "MDI and MinPI share the acceptance quota and the intact-metric check. MinPI can choose from "
        "the MDI and minimum-vote paths. The archival XGBoost pipeline ranks by gain and refits on "
        "the selected columns using the full development sample. It answers the practical question "
        "of what can be achieved if replacing the original model is allowed.",
        "",
        "Each cell gives mean retained columns · AUC / accuracy for classification or "
        "R² / MAE for Abalone · fit-inclusive seconds. Means use seeds 42, 123, 456, 789 and 1024. "
        "The two learners were timed in different study runs; treat differences in seconds accordingly. "
        "Historical preprocessing retains different widths on some Spambase splits.",
        "",
        "| Source | Full RF | RF MDI | RF MinPI | XGB Gain refit |",
        "|---|---|---|---|---|",
    ]
    for name in reference.index.unique("dataset"):
        cells = []
        for label in choices:
            row = mean.loc[name, label]
            a, b = (row.auc, row.accuracy) if pd.notna(row.auc) else (row.r2, row.mae)
            cells.append(f"{row.k:.1f} · {a:.4f} / {b:.4f} · {row.pipeline_s:.2f}s")
        text.append("| " + " | ".join([name.replace("_", " "), *cells]) + " |")
    text.extend(["", "The archived refit uses fewer columns on all six sources and has higher mean "
                 "AUC on all five classification sources and lower MAE on Abalone. In the "
                 "matched frozen-forest comparison, Madelon accounts for the largest width reduction.",
                 "", "[Forest scores](scores.csv) · [Archived refit on the primary test rows]"
                 "(../quota-tree/historical-42-123-456-789-1024.csv) · [Numeric table](refit-comparison.csv)"])
    (output / "comparison.md").write_text("\n".join(text) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
