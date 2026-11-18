"""Build the two numerical paper figures directly from the study outputs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/minpi-study"
FIGURES = ROOT / "paper/minpi/figures"
PALETTE = ("#116461", "#bc4937", "#396293", "#8c6531", "#87528d", "#60656b")
MARKERS = ("o", "s", "^", "D", "v", "P")
LINES = ("-", "--", "-.", ":", "--", ":")


def paired_axes():
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.15))
    fig.subplots_adjust(left=.102, right=.98, bottom=.30, top=.87, wspace=.38)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.21, linewidth=.55)
        ax.tick_params(length=3)
    return fig, axes


def finish(fig, handles, output):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.008), ncols=3,
               frameon=False, columnspacing=1.5, handletextpad=.55)
    fig.savefig(FIGURES / output, metadata={"Creator": "Matplotlib"})
    plt.close(fig)


def kernel():
    frame = pd.read_csv(DATA / "kernel/curves.csv")
    if len(frame) != 144 or frame[["width", "abstain_prob", "seed", "method", "draws"]].duplicated().any():
        raise ValueError("Incomplete kernel benchmark")
    summary = frame.groupby(["method", "draws", "width"]).median(numeric_only=True)
    fig, (time_axis, error_axis) = paired_axes()
    time_axis.set(title="Scoring time", ylabel="Time (ms)", xlabel="Number of features")
    error_axis.set(title="Error against exact scores", ylabel="Mean absolute error", xlabel="Number of features")
    specs = [("exact", 0, "Exact"), ("replacement", 0, "Replacement"),
             *(("mc", b, f"MC {b}") for b in (4, 16, 64, 256))]
    handles = []
    for i, (method, draws, label) in enumerate(specs):
        group = summary.loc[(method, draws)]
        style = dict(color=PALETTE[i], marker=MARKERS[i], ls=LINES[i], lw=1.3, ms=3.5)
        line, = time_axis.plot(group.index, 1000*group.median_s, label=label, **style)
        handles.append(line)
        if method != "exact":
            error_axis.plot(group.index, group.mean_abs_error, **style)
    for ax in (time_axis, error_axis):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([64, 512, 4096, 16384], ["64", "512", "4096", "16384"])
    finish(fig, handles, "kernel.pdf")


def subject_transfer():
    frame = pd.read_csv(DATA / "subject-transfer/summary.csv")
    fig, axes = paired_axes()
    handles = []
    for panel, (ax, learner, title) in enumerate(zip(axes, ("ridge", "knn5"), ("Ridge classifier", "Five-neighbour classifier"))):
        part = frame[frame.learner == learner]
        specs = (("exact", "Pivotal minPI"), ("mean", "Size mean"), ("f", "Univariate F"),
                 ("forest", "ExtraTrees"), ("random", "Uniform"))
        for i, (method, label) in enumerate(specs):
            group = part[part.method == method].sort_values("percent")
            if list(group.percent) != [1, 5, 10, 25, 50]:
                raise ValueError(f"Incomplete budgets for {method}")
            line, = ax.plot(group.percent, group.subject_balanced_accuracy, label=label,
                            color=PALETTE[i], marker=MARKERS[i], ls=LINES[i], lw=1.3, ms=3.5)
            if panel == 0:
                handles.append(line)
        full = part.loc[part.method == "full", "subject_balanced_accuracy"].item()
        line = ax.axhline(full, color="#292929", ls=(0, (4, 3)), lw=.95, label="All features")
        if panel == 0:
            handles.append(line)
        ax.set(title=title, xlabel="Feature budget (%)")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 25, 50])
        ax.xaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel("Subject-balanced accuracy")
    bounds = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(min(b[0] for b in bounds), max(b[1] for b in bounds))
    finish(fig, handles, "subject-transfer.pdf")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42})
    FIGURES.mkdir(parents=True, exist_ok=True)
    kernel()
    subject_transfer()
