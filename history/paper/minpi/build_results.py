"""Regenerate manuscript tables and timing figure from the complete formal split records."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parents[1] / "results" / "minpi-study" / "summary"
DATASETS = {
    "breast_cancer": "Breast cancer", "digits": "Digits", "madelon": "Madelon",
    "spambase": "Spambase", "isolet": "ISOLET", "gisette": "Gisette",
    "diabetes": "Diabetes", "crimedata": "Communities and Crime",
}
LEARNERS = {"ridge": "Ridge", "trees": "Trees"}


def rows_tex(filename, rows):
    """Only table bodies are generated; the paper supplies captions and column headings."""
    (HERE / filename).write_text(" \\\\\n".join(" & ".join(row) for row in rows) + "\n",
                                encoding="utf-8")


def load_results():
    frame = pd.read_csv(SUMMARY / "split_metrics.csv", keep_default_na=False,
                        na_values=[""]).set_index(["dataset", "learner", "method", "seed"])
    if frame.index.has_duplicates or len(frame) != 11 * 2 * 12 * 5:
        raise ValueError("The formal evidence must contain exactly one record per prescribed case.")
    return frame.sort_index()


def make_tables(frame):
    methods = ["full", "random", "f", "forest", "exact", "exact_forest"]
    primary, effect, ablation = [], [], []
    overlap = pd.read_csv(SUMMARY / "ablation_overlap.csv", keep_default_na=False,
                          na_values=[""]).groupby("dataset").median()
    for dataset, label in DATASETS.items():
        for learner, name in LEARNERS.items():
            source = frame.loc[dataset, learner]
            medians = [source.loc[m, "primary"].median() for m in methods]
            primary.append([label if learner == "ridge" else "", name, *(f"{m:.3f}" for m in medians)])
            new, base = source.loc["exact_forest"], source.loc["forest"]
            dq = new.primary - base.primary
            ratio = new.total_seconds / base.total_seconds
            passing = (ratio <= .8) & (dq >= (-.03 if dataset in ("diabetes", "crimedata") else -.02))
            if not new.accuracy.isna().all():
                passing &= (new.accuracy - base.accuracy >= -.03)
            full = source.loc["full"]
            against_full = new.primary - full.primary >= (-.03 if dataset in ("diabetes", "crimedata") else -.02)
            if not new.accuracy.isna().all():
                against_full &= new.accuracy - full.accuracy >= -.03
            def show_delta(x):
                return "0.000" if abs(x) < .0005 else f"{x:+.3f}"
            effect.append([label if learner == "ridge" else "", name, show_delta(dq.median()),
                           f"{ratio.median():.2f}", str(passing.sum()), str((passing & against_full).sum()),
                           show_delta(dq.min()), show_delta(dq.max())])
        ol = overlap.loc[dataset]
        ablation.append([label, *(f"{ol[f'jaccard_{m}']:.2f}" for m in ("mean","vote","k16","mc32","margin")),
                         f"{ol.maximum_mc_abs_error:.3f}"])
    rows_tex("primary-rows.tex", primary)
    rows_tex("prefilter-rows.tex", effect)
    rows_tex("ablation-rows.tex", ablation)
    synth = []
    for dataset, label in [("sparse_signal", "Sparse signal"), ("xor", "XOR"), ("null", "Null")]:
        for method, name in [("random","Uniform"),("f","Univariate $F$"),("forest","Tree impurity"),
                             ("vote","Signed ballots"),("mean","Mean layers"),("exact","Pivotal minPI")]:
            first, second = frame.loc[dataset, "ridge", method], frame.loc[dataset, "trees", method]
            recall = first.known_signal_recall.median()
            synth.append([label if method == "random" else "", name, f"{first.primary.median():.3f}",
                          f"{second.primary.median():.3f}", "--" if pd.isna(recall) else f"{recall:.2f}"])
    rows_tex("synthetic-rows.tex", synth)


def timing_figure(frame):
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.1), sharey=True, layout="constrained")
    comparisons = [("exact", "full", "Pivotal / full"), ("exact", "f", "Pivotal / F"),
                   ("exact_forest", "forest", "Screen + trees / trees")]
    for ax, (method, base, title) in zip(axes, comparisons):
        for i, dataset in enumerate(DATASETS):
            for learner, offset, color, marker in [("ridge", -.12, "#21618c", "o"),
                                                  ("trees", .12, "#a14724", "D")]:
                source = frame.loc[dataset, learner]
                ratio = source.loc[method, "total_seconds"] / source.loc[base, "total_seconds"]
                y = i + offset
                ax.plot([ratio.min(), ratio.max()], [y,y], lw=1, color=color, alpha=.62)
                ax.plot(ratio.median(), y, marker=marker, ms=4, color=color, linestyle="none",
                        label=LEARNERS[learner] if i == 0 else "_")
        ax.set_xscale("log")
        ax.axvline(1, color=".3", lw=.8, ls="--")
        ax.set_title(title, fontsize=8.5, weight="bold")
        ticks = {"full": [.05, .2, 1, 3], "f": [.75, 1, 1.5, 2], "forest": [.5, .75, 1]}
        ax.xaxis.set_major_locator(FixedLocator(ticks[base]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format(x, "g")))
        ax.tick_params(labelsize=7.5)
        ax.minorticks_off()
    axes[0].set_yticks(range(len(DATASETS)), list(DATASETS.values()))
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(.5,-.28), fontsize=8)
    fig.supxlabel("Paired end-to-end time ratio (log scale)", fontsize=8.5)
    fig.savefig(HERE / "time-ratios.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    formal = load_results()
    make_tables(formal)
    timing_figure(formal)
