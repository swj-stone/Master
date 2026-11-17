"""Seed-to-seed agreement of the actual deployed input names on a fixed source."""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from experiments.minpi_compression.benchmark import ROOT, digest


def main():
    out = ROOT/"thesis-paper/evidence/minpi_rework/predictive-reachable"
    receipts = json.loads((out/"lock.json").read_bytes())["receipts"]
    rows = []
    for job, expected in receipts.items():
        path = ROOT/"results/minpi_rework/predictive-reachable"/job/"selection.json"
        if digest(path) != expected:
            raise ValueError(f"Incorrect archived support: {job}")
        jobdata = json.loads(path.read_bytes())
        for arm in ("mdi_canonical", "portfolio_min_canonical"):
            rows.append(dict(panel=jobdata["panel"], dataset=jobdata["name"], seed=jobdata["seed"],
                             method=arm, features=set(jobdata["records"][arm]["canonical"])))
    summary = []
    for key, group in pd.DataFrame(rows).groupby(["panel","dataset","method"]):
        if group.seed.nunique() != 5:
            raise ValueError(f"Expected five fitted supports for {key}")
        supports = list(group.features)
        overlap = [len(a & b)/len(a | b) if a | b else 1. for a,b in combinations(supports,2)]
        widths = [len(s) for s in supports]
        summary.append(dict(panel=key[0], dataset=key[1], method=key[2], mean_jaccard=np.mean(overlap),
                            min_jaccard=min(overlap), max_jaccard=max(overlap), min_columns=min(widths),
                            max_columns=max(widths), shared_by_all=len(set.intersection(*supports))))
    pd.DataFrame(summary).to_csv(out/"stability.csv",index=False)


if __name__ == "__main__":
    main()
