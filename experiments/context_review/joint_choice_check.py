"""Verify witness permutations and independently enumerate small joint information classes."""
from collections import Counter
from io import BytesIO
from itertools import product
from math import prod
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from experiments.context_review import transcript_permutations as tp
from experiments.context_review.joint_choice import (
    DEST, Decision, assigned_permutation, code, exact_point, label_assignments, score_basis,
)


def direct(y, p):
    auc = (roc_auc_score(y, p[:, 1]) if p.shape[1] == 2
           else roc_auc_score(y, p, average="macro", multi_class="ovr"))
    return np.array([auc, accuracy_score(y, p.argmax(axis=1))])


def groups_of(codes, y):
    return [g for g in np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
            if len(np.unique(y[g])) > 1]


def candidate_codes(meta, data, ids):
    columns = sorted({q for i in ids for q in meta["proposals"][i]["point"]})
    return tp.partition(data["codes"][columns].T)


def small_orbit(y, anchor, candidates, codes, skip_constants):
    groups = groups_of(codes, y)
    if skip_constants:
        groups = [g for g in groups if any(np.any(p[g] != p[g[0]]) for p in [anchor, *candidates])]
    assignments = [list(label_assignments(g, y)) for g in groups]
    assert prod(map(len, assignments)) <= 1000
    outcomes = Counter()
    for choices in product(*assignments):
        perm = assigned_permutation(y, codes, zip(groups, choices, strict=True))
        floor = .97 * direct(y, anchor[perm]) - 1e-12
        state = tuple(bool(np.all(direct(y, p[perm]) >= floor)) for p in candidates)
        outcomes[code(state)] += 1
    return outcomes


def case_study():
    key = "spectf/fold-2"
    manifest = tp.rules.read(tp.DEST/"manifest.json")["folds"][key]
    subset = [4, 7, 8]
    width = len(manifest["proposals"][subset[0]]["columns"])
    entire_width = [i for i, s in enumerate(manifest["proposals"]) if len(s["columns"]) == width]
    settings = {"three_candidates": subset, "all_nine_candidates": entire_width}
    rows = []
    with ZipFile(tp.DEST/"row-partitions.zip") as worlds, np.load(BytesIO(worlds.read(key+".npz"))) as data:
        for label, ids in settings.items():
            codes = candidate_codes(manifest, data, ids)
            counts = small_orbit(data["y"], data["anchor"], [data[f"s{i}"] for i in ids], codes, False)
            for state, count in sorted(counts.items()):
                rows.append(dict(dataset="spectf", fold=2, observer=label, ids=" ".join(map(str, ids)),
                                 state=state, assignments=count))
    pd.DataFrame(rows).to_csv(DEST/"case-study.csv", index=False)


def audit():
    manifest = tp.rules.read(tp.DEST/"manifest.json")["folds"]
    pairs = pd.read_csv(DEST/"pairs.csv", dtype={"joint":str})
    single = pd.read_csv(DEST.parent/"transcript_information/proposals.csv").query("observer=='outer_point'")
    states = ("00", "01", "10", "11")
    checks = dict(pairs=0, verified_common_worlds=0, rescored_model_worlds=0, independent_small_orbits=0)
    with ZipFile(DEST/"witnesses.zip") as ws, ZipFile(tp.DEST/"row-partitions.zip") as archive:
        for key, meta in sorted(manifest.items()):
            dataset, fold = meta["dataset"], meta["fold"]
            pp = pairs[(pairs.dataset == dataset) & (pairs.fold == fold)]
            status = single[(single.dataset == dataset) & (single.fold == fold)].set_index("proposal")
            worlds = np.load(BytesIO(ws.read(key+".npy")))
            with np.load(BytesIO(archive.read(key+".npz"))) as ds:
                preds = [ds[f"s{i}"] for i in range(len(meta["proposals"]))]
                ref, y, masks = ds["anchor"], ds["y"], ds["codes"]
            assert np.all(np.sort(worlds, axis=1) == np.arange(len(y)))
            anchors = [direct(y, ref[perm]) for perm in worlds]
            checked = {}
            for row in pp.itertuples():
                pair = (row.a, row.b)
                mask_ids = sorted({q for i in pair for q in meta["proposals"][i]["point"]})
                codes = tp.partition(masks[mask_ids].T) if row.observer == "both_points" else None
                attained = set()
                for flag in states:
                    ix = getattr(row, "world_"+flag)
                    if ix < 0:
                        continue
                    perm = worlds[ix]
                    if codes is not None:
                        np.testing.assert_array_equal(codes[perm], codes)
                    for i, digit in zip(pair, flag, strict=True):
                        if (i, ix) not in checked:
                            measured = direct(y, preds[i][perm])
                            checked[i,ix] = bool(np.all(measured >= .97 * anchors[ix] - 1e-12))
                        assert checked[i,ix] == bool(int(digit))
                    attained.add(flag)
                    checks["verified_common_worlds"] += 1
                assert set(row.joint.split()) == attained
                if codes is None:
                    assert attained == set(states)
                else:
                    allowable = [[int(status.loc[i, "original_pass"])] if status.loc[i, "disposition"] == "invariant"
                                 else [0, 1] for i in pair]
                    envelope = {code(r) for r in product(*allowable)}
                    assert attained <= envelope
                    if attained != envelope:
                        assert attained == small_orbit(y, ref, [preds[i] for i in pair], codes, True).keys()
                        checks["independent_small_orbits"] += 1
                checks["pairs"] += 1
            checks["rescored_model_worlds"] += len(checked)
    assert checks["pairs"] == 2 * 1283
    y = np.array([0, 0, 1, 1])
    codes = np.zeros(4, dtype=int)
    anchor = np.full((4, 2), .5)
    # Different magnitudes with the same ranks and class predictions. Both passage
    # bits move, but the two models tie in every one of the six label allocations.
    probabilities = [np.column_stack((1-scores, scores)) for scores in
                     (np.array([.1, .2, .8, .9]), np.array([.3, .4, .6, .7]))]
    toy = small_orbit(y, anchor, probabilities, codes, False)
    ref = score_basis(y, anchor, 4)
    models = tuple(Decision(y, p, ref, 4) for p in probabilities)
    paired = exact_point(y, codes[:, None], models, (False, False))
    assert set(map(code, paired)) == set(toy) == {"00", "11"} and sum(toy.values()) == 6
    checks["four_row_example"] = dict(sorted(toy.items()))
    (DEST/"audit.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf8")
    print(checks)


if __name__ == "__main__":
    audit()
