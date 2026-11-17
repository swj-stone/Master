"""Evaluate the locked point criteria and resample sources as paired clusters."""
import argparse
from itertools import product
import json

import numpy as np
import pandas as pd

from experiments.minpi_compression.benchmark import EVIDENCE, RUN, digest


def compare(phase):
    dest = EVIDENCE / phase
    raw = pd.read_csv(dest / "scores.csv")
    rows, decisions = [], []
    for task, group in raw.groupby("task"):
        paired = group.set_index(["dataset","seed","method"])
        selected = paired.xs("minpi",level="method")
        for baseline in sorted(set(group.method) - {"minpi"}):
            other = paired.xs(baseline,level="method").reindex(selected.index)
            diffs = pd.DataFrame(index=selected.index)
            metrics = ["auc","accuracy"] if task == "classification" else ["r2","scaled_mae","mae"]
            # Only native-unit MAE should be reported by source, never averaged across units.
            for m in [m for m in metrics if m != "mae"]:
                diffs[m] = selected[m] - other[m]
            for m in ("retained","rank_s","selection_s","deployment_s"):
                if baseline != "full" or m in {"retained","deployment_s"}:
                    diffs[m] = np.log(selected[m]/other[m])
            by_source = diffs.groupby(level="dataset").mean().sort_index()
            n = len(by_source)
            draws = np.array(list(product(range(n), repeat=n)))
            effects = {}
            for field in by_source:
                sample = by_source[field].to_numpy()
                dist = sample[draws].mean(axis=1)
                effect, lo, hi = sample.mean(), *np.quantile(dist,[.025,.975])
                if field in {"retained","rank_s","selection_s","deployment_s"}:
                    effect, lo, hi = np.exp([effect, lo, hi])
                rows.append(dict(task=task,baseline=baseline,field=field,sources=n,estimate=effect,
                                 lo=lo,hi=hi))
                effects[field] = effect
            good = ((effects["auc"] >= -.01 and effects["accuracy"] >= -.01) if task=="classification"
                    else (effects["r2"] >= -.02 and effects["scaled_mae"] <= .02))
            if baseline in {"rfe","full"}:
                width = effects["retained"]
                decisions.append(dict(task=task,baseline=baseline,quality=good,
                    compression=good and (width <= .85 if baseline=="full" else width < 1),
                    time_and_width=good and width <= 1 and effects["selection_s"]<=.9 if baseline=="rfe" else None))
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(dest/"contrasts.csv",index=False)
    verdict = pd.DataFrame(decisions)
    verdict.to_csv(dest/"criteria.csv",index=False)
    print(verdict.to_string(index=False))
    print(contrasts[contrasts.baseline.isin(["full","rfe"])].round(4).to_string(index=False))

    # A support-level ablation can distinguish a new rank from an identical selected set.
    directory = RUN / phase
    diagnostics = []
    for key, receipt in json.loads((directory/"lock.json").read_bytes()).items():
        path = directory/key/"selection.json"
        assert digest(path) == receipt
        job = json.loads(path.read_bytes())
        records = {r["method"]:r for r in job["rows"] if r["repeat"] == 0}
        a, b = records["minpi"], records["coefficient"]
        diagnostics.append(dict(dataset=job["dataset"],seed=job["seed"],p=a["p"],
            negative=a["minimum"].count(-1), neutral=a["minimum"].count(0), positive=a["minimum"].count(1),
            same_order_as_coefficient=a["order"] == b["order"],
            same_support_as_coefficient=set(a["order"][:a["k"]]) == set(b["order"][:b["k"]]),
            **a["work"]))
    pd.DataFrame(diagnostics).to_csv(dest/"vote_diagnostics.csv",index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=["pilot","confirmation"])
    compare(p.parse_args().phase)
