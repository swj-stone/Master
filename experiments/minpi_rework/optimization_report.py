"""Aggregate paired held-source optimization experiments at the dataset level."""
import numpy as np
import pandas as pd

from experiments.minpi_compression.benchmark import ROOT


BASE = ROOT / "thesis-paper/evidence/minpi_rework"
METRICS = ("auc", "accuracy", "r2", "mae", "scaled_mae")


def contrast(frame, focal, target):
    """Ratios on positive costs, paired score differences on their native scales."""
    rows = []
    dimensions = ["panel", "dataset", "task", "seed"]
    a = frame[frame.method == focal].set_index(dimensions).sort_index()
    b = frame[frame.method == target].set_index(dimensions).sort_index()
    if not a.index.equals(b.index) or not a.index.is_unique:
        raise ValueError(f"Unpaired {focal}/{target}")
    if not a.groupby(level="dataset").size().eq(5).all():
        raise ValueError("Expected five stored splits per dataset")
    effects = pd.DataFrame(index=a.index)
    for metric in METRICS:
        effects[metric] = a[metric] - b[metric]
    for cost in ("k", "selection_s", "deployment_s", "search_s"):
        if cost in a and (a[cost] > 0).all() and (b[cost] > 0).all():
            effects[cost + "_ratio"] = np.log(a[cost] / b[cost])
    effects["columns"] = a.k - b.k
    if "certified" in a and a.certified.notna().all() and b.certified.notna().all():
        effects["certification_difference"] = a.certified.astype(float) - b.certified.astype(float)
    sources = effects.groupby(level=["panel","dataset"]).mean()
    for label, subset in [("all",sources), *sources.groupby(level="panel")]:
        for metric in effects:
            values = subset[metric].dropna().to_numpy()
            if not len(values):
                continue
            rng = np.random.default_rng(20261117)
            boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
            estimate, low, high = values.mean(), *np.quantile(boot, [.025,.975])
            if metric.endswith("_ratio"):
                estimate, low, high = np.exp([estimate, low, high])
            rows.append(dict(panel=label,method=focal,reference=target,metric=metric,
                             sources=len(values),estimate=estimate,lower=low,upper=high))
    return rows


def main():
    for study, method, refs in [
        ("directional-confirmation","shared",["independent","direct","cone","local","energy","gain","full"]),
        ("exchange-confirmation","width",["unrestricted","bnb","greedy","gain","energy","full"]),
    ]:
        folder = BASE / study
        frame = pd.read_csv(folder / "scores.csv")
        identifiers = ["panel","dataset","task","method"]
        if "mode" in frame:
            identifiers.append("mode")
        numeric = ["p","k","deployment_s","selection_s",*METRICS]
        numeric += [x for x in ("certified","lb","forced_in","forced_out","search_s") if x in frame]
        frame.groupby(identifiers,as_index=False)[numeric].mean().to_csv(folder / "source-means.csv",index=False)
        data = []
        groups = frame.groupby("mode") if "mode" in frame else [("faithful",frame)]
        for mode, sub in groups:
            pairs = [(method,ref) for ref in refs]
            if study == "exchange-confirmation":
                pairs.append(("sector_both","sector_plane"))
            for candidate, comparison in pairs:
                data.extend(dict(mode=mode,**r) for r in contrast(sub,candidate,comparison))
        pd.DataFrame(data).to_csv(folder / "paired-contrasts.csv",index=False)


if __name__ == "__main__":
    main()
