"""Enumerate single transpositions of oppositely labelled outer inputs, holding both fits fixed."""
from io import BytesIO
from itertools import combinations
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import transcript_permutations as tp
from experiments.context_review.transcript_information import paired_gap


DEST = tp.ROOT / "thesis-paper/evidence/transcript_swaps"


def count_swaps(y, candidate, anchor, classes, base, contributions):
    total = reversed_count = 0
    witness = None
    factual = np.all(base >= 0)
    for g in np.split(np.argsort(classes, kind="stable"), np.cumsum(np.bincount(classes))[:-1]):
        if len(g) < 2:
            continue
        for a, b in combinations(np.unique(y[g]), 2):
            left, right = g[y[g] == a], g[y[g] == b]
            total += len(left)*len(right)
            for start in range(0, len(left), 96):
                rows = left[start:start+96]
                auc, acc = (value + (c[rows,b] - c[rows,a])[:,None] + (c[right,a] - c[right,b])
                            for value, c in zip(base, contributions))
                opposite = ((auc >= 0) & (acc >= 0)) != factual
                # Resolve near-floor floating comparisons with the original scorer.
                for i, j in np.argwhere((abs(auc)<1e-14) | (abs(acc)<1e-14)):
                    p = np.arange(len(y))
                    u, v = rows[i], right[j]
                    p[[u,v]] = p[[v,u]]
                    check, _ = paired_gap(y, candidate[p], anchor[p])
                    opposite[i,j] = np.all(check >= 0) != factual
                reversed_count += int(np.count_nonzero(opposite))
                if witness is None and np.any(opposite):
                    i, j = np.argwhere(opposite)[0]
                    u, v = int(rows[i]), int(right[j])
                    p = np.arange(len(y))
                    p[[u,v]] = p[[v,u]]
                    check, _ = paired_gap(y, candidate[p], anchor[p])
                    assert classes[u] == classes[v] and np.all(check >= 0) != factual
                    witness = (u, v)
    return total, reversed_count, witness


def replay():
    manifest = tp.rules.read(tp.DEST / "manifest.json")
    source = tp.DEST / "row-partitions.zip"
    assert tp.digest(source) == manifest["archive"]
    rows = []
    with ZipFile(source) as archive:
        for key, rec in manifest["folds"].items():
            with np.load(BytesIO(archive.read(key+".npz"))) as arrays:
                y, anchor, codes = arrays["y"], arrays["anchor"], arrays["codes"].T
                reference = tp.score_columns(y, anchor)
                for i, proposal in enumerate(rec["proposals"]):
                    candidate = arrays[f"s{i}"]
                    base, _ = paired_gap(y, candidate, anchor)
                    gaps = tuple(z-.97*a for z,a in zip(tp.score_columns(y,candidate),reference))
                    for obs in ("proposal_only", "outer_point"):
                        groups = (np.zeros(len(y),dtype=np.int32) if obs=="proposal_only"
                                  else tp.partition(codes[:,proposal["point"]]))
                        n, m, pair = count_swaps(y, candidate, anchor, groups, base, gaps)
                        rows.append(dict(dataset=rec["dataset"], fold=rec["fold"], proposal=i,
                                         observer=obs, original_pass=bool(np.all(base>=0)), allowed=n,
                                         reversals=m, fraction=m/n if n else np.nan,
                                         left=pair[0] if pair else None, right=pair[1] if pair else None))
    frame = pd.DataFrame(rows)
    assert len(frame)==1408
    DEST.mkdir(parents=True,exist_ok=True)
    frame.to_csv(DEST/"proposals.csv", index=False)
    tasks = frame.assign(vulnerable=frame.reversals.gt(0), mixed=frame.allowed.gt(0)).groupby(
        ["observer","dataset"]).agg(proposals=("proposal","size"), mixed=("mixed","sum"),
            reversible=("vulnerable","sum"), mean_fraction=("fraction","mean"),
            allowed=("allowed","sum"), reversals=("reversals","sum")).reset_index()
    tasks.to_csv(DEST/"tasks.csv", index=False)
    summary=tasks.groupby("observer").agg(proposals=("proposals","sum"), with_pairs=("mixed","sum"),
        vulnerable=("reversible","sum"), sources_contributing=("mean_fraction","count"),
        source_mean_fraction=("mean_fraction","mean"), pair_total=("allowed","sum"),
        reversing_pairs=("reversals","sum")).reset_index()
    summary.to_csv(DEST/"summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__=="__main__":
    replay()
