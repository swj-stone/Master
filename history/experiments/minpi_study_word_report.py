"""Tabulate paired training-to-support costs for the native-aware controls."""
import argparse
from pathlib import Path

import pandas as pd

from experiments.minpi_study import OUTPUT, ROOT


def report(folder, fit_file, table):
    trials = pd.read_csv(folder/"trials.csv")
    keys = ["source", "seed", "native", "method"]
    groups = trials.groupby(keys)
    if trials.duplicated(keys + ["repeat"]).any() or not groups.size().eq(5).all():
        raise ValueError("Expected five distinct timed rounds per panel and policy")
    panel = groups.select_ms.median().unstack("method")
    ratios = panel.apply(lambda column: panel.dispatch / column)
    ratios.to_csv(folder/"paired-ratios.csv")
    by_source = panel.groupby(level="source").median()
    fit = pd.read_csv(fit_file).groupby(["source", "seed"]).downstream_ms.median()
    by_source["fit"] = fit.groupby("source").median()
    sizes = pd.read_csv(folder/"supports.csv")[["source", "native", "h", "p"]].drop_duplicates()
    if sizes.source.duplicated().any():
        raise ValueError("Expected one arithmetic range and feature budget per source")
    names = {"breast_cancer": "Breast cancer", "diabetes": "Diabetes",
             "digits": "Digits", "mhealth": "MHEALTH", "spambase": "Spambase",
             "crimedata": "Crime", "gas": "Gas", "gisette": "Gisette",
             "isolet": "ISOLET", "supercon": "Materials"}
    rows = []
    sources = sizes.set_index("source").loc[list(names)]
    for native, heading in ((True, r"Machine-word range ($wD\le 2^{63}-1$)"),
                            (False, r"Above the machine-word range ($wD>2^{63}-1$)")):
        subset = sources[sources.native == native].sort_values("p")
        panels = panel.loc[panel.index.get_level_values("source").isin(subset.index)]
        if rows:
            rows.append(r"\addlinespace[4pt]")
        rows.append(r"\multicolumn{7}{@{}l}{\itshape " + heading
                    + f", {len(panels)} panels" + r"}\\[1pt]")
        for source, size in subset.iterrows():
            times = by_source.loc[source, ["float", "full", "anchor", "dispatch", "fit"]]
            budget = f"{size.h}/{size.p:,}"
            fields = [names[source], budget, *(f"{v:.2f}" for v in times)]
            rows.append(" & ".join(fields) + r"\\")
    # The table supplies the last row break immediately before its bottom rule.
    rows[-1] = rows[-1].removesuffix(r"\\")
    table.write_text("\n".join([r"\newcommand{\minpiTimingRows}{%", *rows, "}"]) + "\n",
                     encoding="utf-8")
    observed = ratios[ratios.index.get_level_values("source") != "madelon"]
    print(observed.groupby(level="native").median().round(3))
    print(observed.median().round(3))
    print(observed.groupby(level="source").median().median().round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT/"word-design")
    parser.add_argument("--fit", type=Path, default=OUTPUT/"raw-cost/fit.csv")
    parser.add_argument("--table", type=Path, default=ROOT/"paper/minpi/word-design-rows.tex")
    args = parser.parse_args()
    report(args.input, args.fit, args.table)

