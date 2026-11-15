"""Replay the matched-history endpoints independently from saved per-row probabilities."""
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/context_review/matched-history"


def metrics(y, p):
    hit = y == p.argmax(axis=1)
    recalls, aucs = [], []
    for c in np.unique(y):
        group = y == c
        pos = group.sum()
        neg = len(y) - pos
        recalls.append(hit[group].mean())
        if neg:
            ranks = rankdata(p[:, c])
            aucs.append((ranks[group].sum() - pos * (pos+1) / 2) / (pos * neg))
    return [np.mean(recalls), hit.mean(), np.mean(aucs)]


def labels(dataset):
    root = ROOT / "data/context_study"
    if dataset == "gas":
        with ZipFile(root / "uci-gas-drift/uci224.zip") as z:
            groups = [np.array([int(row.split()[0])-1 for row in z.open(f"Dataset/batch{b}.dat")])
                      for b in (8, 9, 10)]
        return np.concatenate(groups), np.repeat([8, 9, 10], [len(v) for v in groups])
    archive = root / "uci-har/uci-har-240.zip"
    with ZipFile(archive) as z, ZipFile(BytesIO(z.read("UCI HAR Dataset.zip"))) as inner:
        prefix = "UCI HAR Dataset/test/"
        y = np.loadtxt(BytesIO(inner.read(prefix+"y_test.txt")), dtype=int) - 1
        unit = np.loadtxt(BytesIO(inner.read(prefix+"subject_test.txt")), dtype=int)
    return y, unit


def main():
    audit = []
    for dataset in ("gas", "har"):
        y, unit = labels(dataset)
        unit_scores = pd.read_csv(OUT / f"{dataset}-by-unit.csv")
        table = pd.read_csv(OUT / f"{dataset}-summary.csv")
        for row in table.itertuples():
            rec = f"{dataset}_{row.history}_{row.method}_{row.k}"
            with np.load(OUT / f"{rec}.npz") as saved:
                p = saved["probabilities"]
                np.testing.assert_array_equal(unit, saved["units"])
            assert p.shape == (len(y), 6) and np.isfinite(p).all()
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
            selected = json.loads((OUT / f"{rec}.json").read_bytes())["selected"]
            assert len(selected) == len(set(selected)) == row.k
            distinct = np.unique(unit)
            scores = np.array([metrics(y[unit == u], p[unit == u]) for u in distinct])
            group = unit_scores.query("history == @row.history and method == @row.method and k == @row.k").sort_values("unit")
            np.testing.assert_array_equal(distinct, group.unit)
            errors = np.abs(scores - group[["recall", "accuracy", "auc"]].to_numpy())
            means = scores.mean(axis=0)
            if dataset == "har":
                primary = means[1]
                _, acc, auc = metrics(y, p)
                means = [primary, acc, auc]
            error = max(errors.max(), np.max(np.abs(np.array(means)-[row.primary,row.accuracy,row.auc])))
            assert error < 1e-12
            audit.append(dict(record=rec, units=len(distinct), evaluated_rows=len(y), max_abs_error=error))
    frame = pd.DataFrame(audit)
    assert len(frame) == 14
    frame.to_csv(OUT / "audit.csv", index=False)
    print(f"{len(frame)} controls, {frame.units.sum()} unit endpoints; max error {frame.max_abs_error.max():.2g}")


if __name__ == "__main__":
    main()
