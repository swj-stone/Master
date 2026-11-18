"""Rebuild Table 2 from the original timed calls, including the pooled panel row."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from math import isclose
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/minpi-study"
NEW = STUDY / "arithmetic-price/trials.csv"
OLD = STUDY / "word-design/trials.csv"
ARCHIVED = STUDY / "arithmetic-price/summary.csv"
BUDGETS = (2, 10, 50)
GROUPS = (("All", "all"), ("Fits", "True"), ("Exceeds", "False"))
CONTROLS = {"float": "Pure float", "anchor": "Exact anchor", "proposal": "Float + strict check"}


def panel_ratios(path, exact_method, elapsed_column, controls):
    calls = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            method = row["method"]
            if row["source"] == "madelon" or method not in (*controls, exact_method):
                continue
            panel = (row["source"], int(row["seed"]), int(row.get("budget_pct", 10)), row["native"])
            repeat = int(row["repeat"])
            group = calls[panel, method]
            if repeat in group:
                raise ValueError(f"Duplicate call {panel} {method} {repeat}")
            group[repeat] = Fraction(row[elapsed_column])
    medians = {}
    for key, observations in calls.items():
        if set(observations) != set(range(5)) or min(observations.values()) <= 0:
            raise ValueError(f"Expected five positive call times in {key}")
        medians[key] = median(observations.values())
    panels = {panel for panel, _ in calls}
    return {(panel, control): medians[panel, exact_method] / medians[panel, control]
            for panel in panels for control in controls}


def summarize():
    ratios = panel_ratios(NEW, "minpi", "total_ns", CONTROLS)
    prior = panel_ratios(OLD, "dispatch", "select_ms", ["float"])
    if {key for key in ratios if key[0][2] == 10 and key[1] == "float"} != set(prior):
        raise ValueError("The two 10% panel inventories differ")
    with ARCHIVED.open(encoding="utf-8", newline="") as stream:
        archived = {(int(row["budget_pct"]), row["native"]): row for row in csv.DictReader(stream)}
    summary = {}
    for _, group in GROUPS:
        for control in (["float"] if group == "all" else CONTROLS):
            for pct in BUDGETS:
                values = [value for ((_, _, budget, native), method), value in ratios.items()
                          if method == control and budget == pct and (group == "all" or group == native)]
                count = 42 if group == "all" else 21
                if len(values) != count:
                    raise ValueError(f"Expected {count} panel ratios in {group}/{control}/{pct}")
                q1, mid, q3 = quantiles(values, n=4, method="inclusive")
                stats = dict(panels=count, median=float(mid), q1=float(q1), q3=float(q3))
                if group != "all":
                    for stat in ("median", "q1", "q3"):
                        saved = archived[pct, group][f"total_vs_{control}_{stat}"]
                        if not isclose(stats[stat], float(saved), rel_tol=1e-13):
                            raise ValueError(f"Archived subgroup disagrees: {group}/{control}/{pct}/{stat}")
                summary[group, control, pct] = stats
    return summary, float(median(prior.values()))


def table_text(summary):
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Training-to-selection time ratios. For each panel we divide minPI's "
        r"five-call median by the comparator's. Entries give the median [25th, 75th percentiles] "
        r"of the individual ratios, pooled or split at $wD=2^{63}-1$. "
        r"Within this bound minPI and the exact anchor share each call.}",
        r"\label{tab:crossbudget}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule",
        r"Panels & Comparator & 2\% budget & 10\% budget & 50\% budget\\\midrule",
    ]
    for label, group in GROUPS:
        if group != "all":
            lines.append(r"\addlinespace[2pt]")
        controls = ["float"] if group == "all" else CONTROLS
        for i, control in enumerate(controls):
            items = [summary[group, control, pct] for pct in BUDGETS]
            cells = [f"{s['median']:.2f} [{s['q1']:.2f}, {s['q3']:.2f}]" for s in items]
            left = f"{label} ({items[0]['panels']})" if i == 0 else ""
            lines.append(" & ".join([left, CONTROLS[control], *cells]) + r" \\")
    lines.extend([r"\bottomrule\end{tabular*}", r"\end{table}", ""])
    return "\n".join(lines)


def run(output, table):
    summary, old_median = summarize()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "statistics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["common_scale_fits", "comparator", "budget_pct",
                                                     "panels", "median", "q1", "q3"])
        writer.writeheader()
        for (group, control, pct), stats in summary.items():
            writer.writerow(dict(common_scale_fits=group, comparator=control, budget_pct=pct, **stats))
    provenance = {}
    protocol = ROOT / "docs/minpi-study/table-audit-protocol.md"
    for path in (NEW, OLD, ARCHIVED, protocol, Path(__file__).resolve()):
        with path.open("rb") as stream:
            provenance[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    comparison = dict(word_design_10pct=old_median, arithmetic_price_10pct=summary["all", "float", 10]["median"],
                      source_sha256=provenance)
    (output / "audit.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    if table:
        table.write_text(table_text(summary), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    run(args.output, args.table)
