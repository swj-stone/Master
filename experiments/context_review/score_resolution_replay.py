"""Compare exact and coarsened unmasked predictions with occupancy-matched random codes."""
from hashlib import sha256
from io import BytesIO
from itertools import permutations
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import transcript_information as info
from experiments.context_review import transcript_permutations as tp
from experiments.context_review.score_resolution import DESIGN, DEST, point_digest, text_digest
from experiments.context_study.data import digest


RESOLUTIONS = (2, 4, 8, "full")
N_DRAWS = 5
BLOCK = 48


def remappings(types, index):
    values, first, inverse, counts = np.unique(
        types, return_index=True, return_inverse=True, return_counts=True
    )
    assert np.array_equal(values, np.arange(len(values)))
    bins = [np.flatnonzero(counts == m) for m in np.unique(counts)]
    maps = [np.arange(len(first))]
    for draw in range(1, N_DRAWS+1):
        rng = np.random.default_rng(np.random.SeedSequence([153611, index, draw]))
        p = np.arange(len(first))
        for g in bins:
            p[g] = rng.permutation(g)
        np.testing.assert_array_equal(counts, counts[p])
        maps.append(p)
    movable = sum(len(g) for g in bins if len(g) > 1)
    return first, inverse.reshape(-1), maps, len(bins), movable


def discretize(probabilities, precision):
    if precision == "full":
        values = probabilities
    else:
        bins = 2**precision
        values = np.minimum(np.floor(probabilities.astype(np.float64)*bins), bins-1).astype(np.int32)
    return tp.partition(values)


def matching_groups(codes, y):
    counts = np.bincount(codes)
    start = np.r_[0, np.cumsum(counts)[:-1]]
    order = np.argsort(codes, kind="stable")
    groups = []
    for i in np.flatnonzero(counts > 1):
        g = order[start[i]:start[i]+counts[i]]
        if np.any(y[g] != y[g[0]]):
            groups.append(g)
    return groups


def decide(y, candidate, anchor, codes, groups, original, costs):
    passed = np.all(original >= 0)
    if not groups:
        return "invariant", "homogeneous_labels", ""
    idx = np.arange(len(y))
    pieces = []
    for g in groups:
        for start in range(0, len(g), BLOCK):
            b = g[start:start+BLOCK]
            if len(b) > 1 and np.any(y[b] != y[b[0]]):
                pieces.append(b)
    restricted = any(len(g) > BLOCK for g in groups)
    runs = [(pieces, False), (groups, True)] if restricted else [(groups, True)]
    for cells, unrestricted in runs:
        ends = []
        for j, cost in enumerate(costs):
            p = tp.extremizer(y, cost, cells, maximize=not passed)
            delta = np.array([(c[p, y] - c[idx,y]).sum() for c in costs])
            endpoint = original + delta
            ends.append(endpoint[j])
            # Check every witness against the two library-scored models on the same
            # assignment, independent of the midrank/assignment objective.
            if np.all(endpoint >= 0) != passed:
                np.testing.assert_array_equal(codes[p], codes)
                np.testing.assert_array_equal(np.sort(p), idx)
                exact, _ = info.paired_gap(y, candidate[p], anchor[p])
                np.testing.assert_allclose(exact, endpoint, rtol=0, atol=1e-13)
                if np.all(exact >= 0) != passed:
                    return "witness", "auc" if j == 0 else "accuracy", sha256(p.astype("<i8").tobytes()).hexdigest()
            if unrestricted and not passed and endpoint[j] < -1e-13:
                return "invariant", "coordinate_ceiling", ""
        if unrestricted and passed and min(ends) >= 1e-13:
            return "invariant", "coordinate_floor", ""
    return "unresolved", "joint_feasibility", ""


def sanity_check():
    # This small multiclass group has movable labels and genuine score ties.
    y = np.array([0, 1, 2, 0, 1])
    p = np.array([[2,5,3], [2,5,3], [1,2,7], [6,3,1], [3,3,4]]) / 10
    a = p[[2,0,1,4,3]]
    order = np.arange(len(y))
    costs = [x-.97*b for x,b in zip(tp.score_columns(y,p), tp.score_columns(y,a))]
    base, _ = info.paired_gap(y,p,a)
    exact = []
    for perm in permutations(order):
        perm = np.asarray(perm)
        actual, _ = info.paired_gap(y, p[perm], a[perm])
        calculated = base + np.array([(c[perm,y]-c[order,y]).sum() for c in costs])
        np.testing.assert_allclose(calculated, actual, rtol=0, atol=1e-13)
        exact.append(actual)
    for j, c in enumerate(costs):
        for maximize, reduce in ((True,np.max), (False,np.min)):
            sol = tp.extremizer(y, c, [order], maximize)
            optimum = base[j] + (c[sol,y]-c[order,y]).sum()
            np.testing.assert_allclose(optimum, reduce(np.array(exact)[:,j]), rtol=0, atol=1e-13)


def run_fold(index, key, meta, fp, archive, prior_point, rows, occupancy):
    suff = key+".npz"
    with np.load(BytesIO(archive.read(suff))) as data, np.load(BytesIO(fp.read(suff))) as small:
        y, anchor, masks = data["y"], data["anchor"], data["codes"].T
        raw_codes, probs = small["types"], small["probabilities"]
        first, inverse, maps, strata, movable = remappings(raw_codes, index)
        np.testing.assert_array_equal(probs, probs[first][inverse])
        anchors = tp.score_columns(y, anchor)
        records = []
        for i, proposal in enumerate(meta["proposals"]):
            c = data[f"s{i}"]
            base, _ = info.paired_gap(y,c,anchor)
            cols = [x-.97*b for x,b in zip(tp.score_columns(y,c),anchors)]
            point = tp.partition(masks[:, proposal["point"]])
            records.append((c, base, cols, point))
        ident = dict(dataset=meta["dataset"], fold=meta["fold"])
        for draw, mapping in enumerate(maps):
            invariants = set()
            for precision in RESOLUTIONS:
                codes = discretize(probs, precision)
                if precision=="full":
                    np.testing.assert_array_equal(data["initial"], codes)
                assigned = codes[first][mapping][inverse]
                occ = np.bincount(codes)
                np.testing.assert_array_equal(occ, np.bincount(assigned))
                source = "unmasked" if not draw else "random_code"
                tags = dict(**ident, source=source, draw=draw, precision=str(precision))
                occupancy.append(dict(**tags, rows=len(y), input_types=len(first), strata=strata,
                                      permutable_types=movable, signatures=len(occ), singleton_rows=int(sum(occ==1))))
                groups = matching_groups(assigned, y)
                for i, (c, base, costs, point) in enumerate(records):
                    for mode in ("code","point_plus_code"):
                        if (i,mode) in invariants or (mode=="point_plus_code" and (i,"code") in invariants):
                            status, method, certificate = "invariant","nested_code",""
                        elif mode=="point_plus_code" and prior_point.loc[meta["dataset"],meta["fold"],i]=="invariant":
                            status, method, certificate = "invariant","point_certificate",""
                        else:
                            obs, cells = assigned, groups
                            if mode=="point_plus_code":
                                obs = tp.partition(np.column_stack((point, assigned)))
                                cells = matching_groups(obs, y)
                            status, method, certificate = decide(y,c,anchor,obs,cells,base,costs)
                        if status=="invariant":
                            invariants.add((i,mode))
                        rows.append(dict(**tags, proposal=i, observer=mode, disposition=status,
                                         check=method, permutation_sha=certificate))


def replay():
    sanity_check()
    rec = tp.rules.read(DEST/"inputs.json")
    fp = DEST/"unmasked.zip"
    original = tp.DEST/"row-partitions.zip"
    assert text_digest(DESIGN)==rec["design"] and digest(fp)==rec["predictions"] and digest(original)==rec["source"]
    point_path = info.DEST/"proposals.csv"
    assert point_digest(point_path)==rec["point_certificates"]
    old = pd.read_csv(point_path, keep_default_na=False)
    prior = old[old.observer=="outer_point"].set_index(["dataset","fold","proposal"]).disposition
    assert len(prior)==704 and prior.index.is_unique
    rows, occupancy = [], []
    with ZipFile(fp) as codes, ZipFile(original) as partitions:
        folds = tp.rules.read(tp.DEST/"manifest.json")["folds"]
        for index,(key, meta) in enumerate(sorted(folds.items())):
            before = perf_counter()
            run_fold(index,key,meta,codes,partitions,prior,rows,occupancy)
            elapsed = perf_counter()-before
            if elapsed>3600:
                raise TimeoutError(key)
            print(f"[{index+1}/42] {key} {elapsed:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    assert len(df)==704*2*len(RESOLUTIONS)*(N_DRAWS+1)
    occ = pd.DataFrame(occupancy)
    tables = {"occupancy":occ, "proposals":df}
    facets = ["observer","source","draw","precision"]
    for name, keys in (("summary",facets),("tasks",["dataset",*facets])):
        counts = df.groupby([*keys,"disposition"]).size().unstack(fill_value=0).reset_index()
        for label in ("witness","invariant","unresolved"):
            if label not in counts:
                counts[label] = 0
        tables[name] = counts
    for name, table in tables.items():
        table.to_csv(DEST/f"{name}.csv", index=False)
    print(tables["summary"].groupby(["observer","source","precision"]).witness.agg(["min","max"]).to_string())


if __name__=="__main__":
    replay()
