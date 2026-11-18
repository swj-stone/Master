"""Summarize the timing replay at the training-panel level and render the manuscript table."""
import argparse
from pathlib import Path

import pandas as pd

from experiments.minpi_study import OUTPUT, ROOT


def report(folder, table):
    cases = pd.read_csv(folder/"cases.csv").set_index(["source", "seed"])
    trials = pd.read_csv(folder/"trials.csv")
    assert not trials.duplicated(["source", "seed", "method", "repeat"]).any()
    groups = trials.groupby(["source", "seed", "method"])
    assert groups.size().eq(5).all()
    medians = groups[["construct_ms", "score_ms", "select_ms"]].median().unstack()
    select = medians.select_ms
    panels = select.copy()
    panels["construct"] = medians.construct_ms.integer_cert
    panels["integer_score"] = medians.score_ms.integer_cert
    panels["fit_predict"] = pd.read_csv(folder/"fit.csv").groupby(["source", "seed"]).downstream_ms.median()
    for other in ["integer_anchor", "integer_full", "float_cert"]:
        panels[f"ratio_{other}"] = panels.integer_cert / panels[other]
    panels = panels.join(cases)
    panels.to_csv(folder/"panel-summary.csv")
    sources = panels.groupby(level="source").median(numeric_only=True)
    sources.insert(0, "number_of_panels", panels.groupby(level="source").size())
    sources.to_csv(folder/"source-summary.csv")
    natural = panels[panels.index.get_level_values("source") != "madelon"]
    assert len(natural) == 42 and natural.float_changes.sum() == 0 and natural.exact_columns.sum() == 0
    assert (natural.gap == 0).sum() == 2
    assert (natural.within_twice_eta == 1).sum() == 39
    names = {"gisette": "Gisette", "isolet": "ISOLET", "gas": "Gas",
             "supercon": "Materials", "mhealth": "MHEALTH"}
    rows = []
    for key, display in names.items():
        s = sources.loc[key]
        values = [s.float_cert, s.integer_full, s.integer_anchor, s.integer_cert, s.fit_predict]
        rows.append(display + " & " + " & ".join(f"{x:.2f}" for x in values))
    table.write_text((r"\\" + "\n").join(rows) + "\n", encoding="utf-8")
    print(natural[["ratio_integer_anchor", "ratio_integer_full", "ratio_float_cert"]].agg(["min","median","max"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT/"raw-cost")
    parser.add_argument("--table", type=Path, default=ROOT/"paper/minpi/raw-cost-rows.tex")
    args = parser.parse_args()
    report(args.input, args.table)
