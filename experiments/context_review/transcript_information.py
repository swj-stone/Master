"""Search for opposing refit outcomes with three different amounts of observed outer information."""
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import transcript_permutations as tp
from experiments.context_study.data import ROOT, digest


DEST = ROOT / "thesis-paper/evidence/transcript_information"
MAX_BLOCK = 48


def paired_gap(y, candidate, anchor):
    cm, am = [tp.metrics(y, p, False) for p in (candidate, anchor)]
    return (np.array([cm[k] - .97 * am[k] + 1e-12 for k in ("auc", "accuracy")]),
            dict(candidate=cm, anchor=am))


def exchange_classes(y, contributions, codes, block_limit):
    groups, restricted = [], False
    order = np.argsort(codes, kind="stable")
    for group in np.split(order, np.cumsum(np.bincount(codes))[:-1]):
        if len(group) < 2 or np.all(y[group] == y[group[0]]):
            continue
        # If all label-specific contrasts between rows vanish, no permutation of
        # this class changes either objective even if the underlying scores vary.
        if all(np.all(a[group] == a[group[0]]) for a in contributions):
            continue
        step = block_limit or len(group)
        restricted |= len(group) > step
        for start in range(0, len(group), step):
            b = group[start:start+step]
            if len(b) > 1 and np.any(y[b] != y[b[0]]):
                groups.append(b)
    return groups, restricted


def audit(y, candidate, anchor, observed, base, contributions, block_limit=MAX_BLOCK):
    codes = tp.partition(observed)
    groups, restricted = exchange_classes(y, contributions, codes, block_limit)
    passed = bool(np.all(base >= 0))
    low, high = base.copy(), base.copy()
    witness = None
    identity = np.arange(len(y))
    for j, a in enumerate(contributions):
        for maximize in (False, True):
            p = tp.extremizer(y, a, groups, maximize)
            assert np.array_equal(observed[p], observed) and np.array_equal(np.sort(p), identity)
            delta = np.array([(cost[p,y] - cost[identity,y]).sum() for cost in contributions])
            v = base + delta
            (high if maximize else low)[j] = v[j]
            if witness is None and bool(np.all(v >= 0)) != passed:
                exact, scores = paired_gap(y, candidate[p], anchor[p])
                np.testing.assert_allclose(exact, v, atol=1e-13, rtol=0)
                if bool(np.all(exact >= 0)) != passed:
                    changed = np.flatnonzero(p != identity)
                    witness = dict(positions=changed.tolist(), donors=p[changed].tolist(), **scores)
    separated = np.any(high < -1e-13) or np.all(low > 1e-13) or not groups
    invariant = not restricted and separated
    if np.array_equal(candidate, anchor):
        invariant = True
    label = "witness" if witness is not None else "invariant" if invariant else "unresolved"
    return dict(disposition=label, original_pass=passed, distinct_row_signatures=len(np.unique(codes)),
                restricted_blocks=restricted, min_tested_auc_gap=low[0], max_tested_auc_gap=high[0],
                min_tested_accuracy_gap=low[1], max_tested_accuracy_gap=high[1]), witness


def replay():
    path = tp.DEST / "row-partitions.zip"
    manifest = tp.rules.read(tp.DEST / "manifest.json")
    assert digest(path) == manifest["archive"]
    results, witnesses = [], []
    with ZipFile(path) as archive:
        for key, meta in manifest["folds"].items():
            with np.load(BytesIO(archive.read(key+".npz"))) as data:
                y, anchor, masks = data["y"], data["anchor"], data["codes"].T
                base_cols = tp.score_columns(y, anchor)
                initial = data["initial"]
                for i, s in enumerate(meta["proposals"]):
                    candidate = data[f"s{i}"]
                    base, factual = paired_gap(y, candidate, anchor)
                    cc = tp.score_columns(y, candidate)
                    contributions = tuple(c - .97*a for c, a in zip(cc, base_cols))
                    for observer in ("proposal_only", "outer_point", "outer_unmasked",
                                     "outer_point_unmasked", "outer_family"):
                        if observer == "proposal_only":
                            observed = np.zeros((len(y), 1), dtype=np.uint8)
                        elif observer == "outer_unmasked":
                            observed = initial[:, None]
                        elif observer == "outer_family":
                            observed = masks[:, s["recorded"]]
                        else:
                            observed = masks[:, s["point"]]
                            if observer == "outer_point_unmasked":
                                observed = np.column_stack((observed, initial))
                        ident = dict(dataset=meta["dataset"], fold=meta["fold"], proposal=i, observer=observer)
                        record, example = audit(y, candidate, anchor, observed, base, contributions)
                        first_pass = record["disposition"]
                        if first_pass == "unresolved" and record["restricted_blocks"]:
                            record, example = audit(y, candidate, anchor, observed, base, contributions,
                                                    block_limit=None)
                        record["first_pass"] = first_pass
                        results.append(dict(**ident, **record))
                        if example is not None:
                            witnesses.append(dict(**ident, factual=factual, alternative=example))
    df = pd.DataFrame(results)
    assert len(df) == 5*704
    DEST.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST / "proposals.csv", index=False)
    counts = df.groupby(["observer", "disposition"]).agg(
        proposals=("proposal","size"), tasks=("dataset","nunique"), original_pass=("original_pass","sum")
    ).reset_index()
    counts.to_csv(DEST / "summary.csv", index=False)
    (DEST / "witnesses.json").write_text(json.dumps(witnesses, separators=(",", ":")), encoding="utf8")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    replay()
