"""Recreate the score–time curves from the archived wide-block experiment."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVIDENCE = Path(__file__).with_name("evidence")
METHODS = [
    ("wide", "minPI (block selection)", "#006a67", "-"),
    ("old", "Earlier minPI", "#b04c16", "--"),
    ("F", "Matched F", "#435e9a", "-."),
    ("random", "Matched random", "#695c70", ":"),
]


def main():
    rows = pd.read_csv(EVIDENCE / "all-paired.csv", keep_default_na=False)
    assert len(rows) == 405 and not rows.duplicated(["cohort", "dataset", "fold", "seed"]).any()
    assert rows.k_F.equals(rows.k_wide) and rows.k_random.equals(rows.k_wide)
    archive = pd.read_csv(EVIDENCE / "pipelines-vs-full.csv").query("scope == 'all'").set_index("method")
    bands = pd.cut(rows.p, [0, 32, 128, 512, 2048, 10000, np.inf], labels=False)
    sizes = bands.value_counts().sort_index()
    labels = ["4–32", "33–128", "129–512", "513–2,048", "2,049–10,000", ">10,000"]
    fig, (curve, bars) = plt.subplots(1, 2, figsize=(11.3, 4.6), layout="constrained")
    for j, (arm, name, color, style) in enumerate(METHODS):
        eligible = rows[f"k_{arm}"] < rows.p
        for i in (1, 2):
            eligible &= rows[f"total_ms{i}_{arm}"] < rows[f"total_ms{i}_full"]
        loss = rows.score_full - rows[f"score_{arm}"]
        qualified = np.sort(loss[eligible].to_numpy())
        points = np.unique(np.r_[0., .02, qualified[(qualified > 0) & (qualified < .1)], .1])
        counts = np.searchsorted(qualified, points + 1e-12, side="right")
        cut = eligible & (loss <= .02 + 1e-12)
        assert (len(qualified), counts[0], cut.sum()) == tuple(
            archive.loc[arm, ["cut_faster", "no_loss", "within_002"]])
        curve.step(points, counts, where="post", label=name, color=color,
                   lw=2.4 if arm == "wide" else 1.9, ls=style)
        curve.plot(.02, cut.sum(), marker="o", ms=5, mec="white", mew=.7, color=color)
        bars.bar(np.arange(len(sizes))+(j-1.5)*.19, cut.groupby(bands).mean()*100,
                 width=.18, color=color)
    curve.axvline(.02, color="#888888", lw=.8, zorder=0)
    curve.set(title="All 405 configurations", xlabel="Allowed test-score loss",
              ylabel="Qualifying configurations", xlim=(0, .1), ylim=(0, None))
    bars.set(title="By input width at a score loss ≤ 0.02", xlabel="Input columns (number of configurations)",
             ylabel="Qualifying configurations (%)", ylim=(0, 100))
    bars.set_xticks(np.arange(len(sizes)), [f"{lab}\n({n})" for lab,n in zip(labels, sizes, strict=True)], fontsize=8)
    curve.legend(frameon=False, fontsize=9, loc="lower right")
    for ax in (curve, bars):
        ax.grid(axis="y", lw=.65, color="#ddd")
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.supxlabel("Each qualifying run selects fewer columns and finishes sooner than the full-feature model in both timing passes.",
                  fontsize=10, color="#39424a")
    for ext in ("png", "pdf"):
        fig.savefig(EVIDENCE / f"block-selection-tradeoff.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
