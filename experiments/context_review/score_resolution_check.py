"""Independent checks of resolution summaries and selected small permutation orbits."""
from collections import Counter
from io import BytesIO
from itertools import product
import json
from math import comb
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import score_resolution as sr
from experiments.context_review import transcript_information as info
from experiments.context_review import transcript_permutations as tp


def audit_nesting(frame):
    keys = ["dataset", "fold", "proposal", "draw", "observer"]
    levels = frame.pivot(index=keys, columns="precision", values="disposition")
    for i, coarse in enumerate(["2", "4", "8"]):
        for fine in ["4", "8", "full"][i:]:
            assert not ((levels[coarse]=="invariant") & (levels[fine]=="witness")).any()
    joint = frame.pivot(index=keys[:-1]+["precision"], columns="observer", values="disposition")
    assert not ((joint.code=="invariant") & (joint.point_plus_code=="witness")).any()
    assert frame.permutation_sha.str.len().eq(64).equals(frame.disposition.eq("witness"))


def allocations(counts):
    if not sum(counts.values()):
        yield []
        return
    for label in counts:
        if counts[label]:
            counts[label] -= 1
            for rest in allocations(counts):
                yield [label, *rest]
            counts[label] += 1


def enumerate_decisions(y, anchor, candidate, cells):
    # Vary labels against a fixed multiset of paired scores. A simultaneous
    # inverse row permutation gives the same two pairs of metric values.
    orbit = 1
    pools = []
    for cell in cells:
        if np.all(anchor[cell] == anchor[cell[0]]) and np.all(candidate[cell] == candidate[cell[0]]):
            continue
        counts = Counter(y[cell])
        size = len(cell)
        for c in counts.values():
            orbit *= comb(size, c)
            size -= c
            if orbit > 32:
                return None
        pools.append((cell, counts))
    if orbit == 1:
        return None
    decisions = set()
    n = 0
    for labels in product(*(allocations(c) for _,c in pools)):
        changed = y.copy()
        for (cell,_), target in zip(pools, labels):
            changed[cell] = target
        gaps, _ = info.paired_gap(changed, candidate, anchor)
        decisions.add(bool(np.all(gaps >= 0)))
        n += 1
    assert n == orbit
    return n, "witness" if len(decisions)>1 else "invariant"


def cells_for(codes, y):
    buckets = {}
    for i, value in enumerate(codes):
        buckets.setdefault(tuple(value), []).append(i)
    return [np.asarray(b) for b in buckets.values() if len(set(y[b]))>1]


def check():
    saved = pd.read_csv(sr.DEST/"proposals.csv", keep_default_na=False, dtype={"precision":str})
    audit_nesting(saved)
    assert len(saved)==33792
    rows = saved.set_index(["dataset","fold","draw","precision","observer","proposal"]).disposition
    totals = Counter()
    examples = []
    folds = tp.rules.read(tp.DEST/"manifest.json")["folds"]
    with ZipFile(sr.DEST/"unmasked.zip") as tokens, ZipFile(tp.DEST/"row-partitions.zip") as stored:
        for index,(key, meta) in enumerate(sorted(folds.items())):
            name = key+".npz"
            with np.load(BytesIO(tokens.read(name))) as src, np.load(BytesIO(stored.read(name))) as ds:
                x, t = src["probabilities"], src["types"]
                y, anchor, masks = ds["y"], ds["anchor"], ds["codes"]
                vocab, first, count = np.unique(t,return_index=True,return_counts=True)
                perm = vocab.copy()
                rng = np.random.default_rng(np.random.SeedSequence([153611,index,1]))
                for qty in np.unique(count):
                    g = sorted(vocab[count==qty])
                    perm[g] = rng.permutation(g)
                for draw, xmap in enumerate((x,x[first[perm[t]]])):
                    for precision, code in (("full", xmap),("8",np.floor(xmap.astype(float)*256).clip(max=255))):
                        for mode in ("code","point_plus_code"):
                            for i, prop in enumerate(meta["proposals"]):
                                case = (meta["dataset"],meta["fold"],draw,precision,mode,i)
                                kind = rows[case]
                                category = (draw, mode, kind)
                                if totals[category]>=2 or kind=="unresolved":
                                    continue
                                provided = code if mode=="code" else np.column_stack((code, masks[prop["point"]].T))
                                result = enumerate_decisions(y, anchor, ds[f"s{i}"], cells_for(provided, y))
                                if result is not None:
                                    n, answer = result
                                    assert answer==kind, case
                                    examples.append(dict(zip(rows.index.names,case),checked=answer,allocations=n))
                                    totals[category] += 1
    out = dict(proposal_conditions=len(saved), exhaustive_nontrivial=examples)
    (sr.DEST/"audit.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print(f"{len(examples)} nontrivial finite orbits and {len(saved)} nested dispositions checked")


if __name__=="__main__":
    check()
