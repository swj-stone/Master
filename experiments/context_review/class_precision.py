"""Retain predicted class and input multiplicity while permuting four-bit score codes."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import score_resolution as sr
from experiments.context_review import score_resolution_replay as resolution
from experiments.context_review import transcript_information as info
from experiments.context_review import transcript_permutations as tp
from experiments.context_study.data import ROOT, digest


DEST = ROOT / "thesis-paper/evidence/class_precision"
DESIGN = ROOT / "configs/context_review/class_precision.json"
DRAW_COUNT = 5
KEYS = ["dataset", "fold", "source", "draw", "proposal", "observer"]
STATES = ["witness", "unresolved", "invariant"]


def assign_codes(probabilities, types, fold_index):
    """Generate the codebooks without target labels, proposal supports or reduced-model outputs."""
    values, first, inverse, count = np.unique(types, return_index=True, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    np.testing.assert_array_equal(values, np.arange(len(values)))
    np.testing.assert_array_equal(probabilities, probabilities[first][inverse])
    classes = probabilities[first].argmax(axis=1)
    code = resolution.discretize(probabilities, 4)[first]
    keys = np.column_stack((count, classes))
    unique_keys, group_id = np.unique(keys, axis=0, return_inverse=True)
    groups = [np.flatnonzero(group_id.reshape(-1) == i) for i in range(len(unique_keys))]
    nonsingleton = np.zeros(len(first), dtype=bool)
    recodable = np.zeros(len(first), dtype=bool)
    for group in groups:
        nonsingleton[group] = len(group) > 1
        recodable[group] = len(np.unique(code[group])) > 1
    code_class_pairs = np.unique(np.column_stack((code, classes)), axis=0)
    multiclass_codes = np.flatnonzero(np.bincount(code_class_pairs[:, 0]) > 1)
    shared_class = np.isin(code, multiclass_codes)
    common = dict(rows=len(types), types=len(first), strata=len(groups),
                  permutable_types=int(nonsingleton.sum()), permutable_rows=int(count[nonsingleton].sum()),
                  recodable_types=int(recodable.sum()), recodable_rows=int(count[recodable].sum()),
                  cross_class_codes=len(multiclass_codes), rows_with_cross_class_code=int(count[shared_class].sum()))
    for draw in range(1, DRAW_COUNT + 1):
        rng = np.random.default_rng(np.random.SeedSequence([153611, fold_index, draw]))
        donor = np.arange(len(first))
        for group in groups:
            donor[group] = rng.permutation(group)
        np.testing.assert_array_equal(np.sort(donor), values)
        np.testing.assert_array_equal(keys, keys[donor])
        # This preserves the conditional code counts over types and, since the
        # multiplicities are fixed in each group, over observed rows as well.
        assigned = code[donor]
        for group in groups:
            np.testing.assert_array_equal(np.sort(code[group]), np.sort(assigned[group]))
        row_codes = assigned[inverse]
        np.testing.assert_array_equal(np.bincount(code[inverse]), np.bincount(row_codes))
        changed = assigned != code
        donor_changed = donor != values
        assert not np.any(changed & ~recodable)
        occupancy = dict(**common, donor_changed_types=int(donor_changed.sum()),
                         donor_changed_rows=int(count[donor_changed].sum()), changed_types=int(changed.sum()),
                         changed_rows=int(count[changed].sum()),
                         codebook_sha256=sha256(assigned.astype("<i8").tobytes()).hexdigest())
        yield draw, row_codes, occupancy


def reference_tables():
    records = pd.read_csv(sr.DEST / "proposals.csv", dtype={"precision": str}, keep_default_na=False)
    ref = records.loc[records.precision == "4"].drop(columns="precision").copy()
    assert len(ref) == 704 * 2 * (DRAW_COUNT + 1) and not ref.duplicated(KEYS).any()
    old = pd.read_csv(info.DEST / "proposals.csv")
    point = old[old.observer == "outer_point"].set_index(["dataset", "fold", "proposal"]).disposition
    assert len(point) == 704 and point.index.is_unique
    return ref, point


def run_fold(meta, codebooks, models, point_status):
    y, anchor, masks = models["y"], models["anchor"], models["codes"].T
    ident = dict(dataset=meta["dataset"], fold=meta["fold"])
    anchor_columns = tp.score_columns(y, anchor)
    candidates = []
    for i, description in enumerate(meta["proposals"]):
        candidate = models[f"s{i}"]
        original, _ = info.paired_gap(y, candidate, anchor)
        costs = [x - .97 * b for x, b in zip(tp.score_columns(y, candidate), anchor_columns, strict=True)]
        point_code = tp.partition(masks[:, description["point"]])
        candidates.append((candidate, original, costs, point_code))
    proposals, occupancy = [], []
    for draw, code, measures in codebooks:
        occupancy.append(dict(**ident, draw=draw, **measures))
        groups = resolution.matching_groups(code, y)
        for i, (candidate, original, costs, point_code) in enumerate(candidates):
            labels = dict(**ident, source="class_matched", draw=draw, proposal=i)
            status, reason, witness = resolution.decide(y, candidate, anchor, code, groups, original, costs)
            proposals.append(dict(**labels, observer="code", disposition=status, check=reason, permutation_sha=witness))
            if status == "invariant":
                status, reason, witness = "invariant", "code_certificate", ""
            elif point_status.loc[meta["dataset"], meta["fold"], i] == "invariant":
                status, reason, witness = "invariant", "point_certificate", ""
            else:
                joint = tp.partition(np.column_stack((code, point_code)))
                cells = resolution.matching_groups(joint, y)
                status, reason, witness = resolution.decide(y, candidate, anchor, joint, cells, original, costs)
            proposals.append(dict(**labels, observer="point_plus_code", disposition=status, check=reason,
                                  permutation_sha=witness))
    return proposals, occupancy


def counts(frame, fields):
    return (frame.groupby(fields + ["disposition"]).size().unstack(fill_value=0)
            .reindex(columns=STATES, fill_value=0).reset_index())


def replay(record=False):
    reference, point_status = reference_tables()
    labels = tp.rules.read(tp.DEST / "manifest.json")
    probability_file = sr.DEST / "unmasked.zip"
    partition_file = tp.DEST / "row-partitions.zip"
    previous = tp.rules.read(sr.DEST / "inputs.json")
    assert digest(probability_file) == previous["predictions"]
    assert digest(partition_file) == previous["source"] == labels["archive"]
    provenance = dict(protocol=sr.text_digest(DESIGN), probabilities=previous["predictions"],
                      partitions=previous["source"], points=sr.point_digest(info.DEST / "proposals.csv"),
                      reference=sha256(reference.to_csv(index=False, lineterminator="\n").encode()).hexdigest())
    DEST.mkdir(parents=True, exist_ok=True)
    receipt = DEST / "receipt.json"
    if not record:
        assert tp.rules.read(receipt) == provenance
    all_rows, all_occupancy = [], []
    with ZipFile(probability_file) as preds, ZipFile(partition_file) as data:
        for index, (key, meta) in enumerate(sorted(labels["folds"].items())):
            member = key + ".npz"
            with np.load(BytesIO(preds.read(member))) as frozen, np.load(BytesIO(data.read(member))) as models:
                plans = list(assign_codes(frozen["probabilities"], frozen["types"], index))
                rows, occupancy = run_fold(meta, plans, models, point_status)
            all_rows.extend(rows)
            all_occupancy.extend(occupancy)
            print(f"[{index+1}/42] {key}", flush=True)
    frame = pd.concat((reference, pd.DataFrame(all_rows)), ignore_index=True)
    assert not frame.duplicated(KEYS).any() and len(frame) == 704 * 2 * (2 * DRAW_COUNT + 1)
    assert frame.permutation_sha.str.len().eq(64).equals(frame.disposition.eq("witness"))
    merge_key = ["dataset", "fold", "source", "draw", "proposal"]
    nested = frame.pivot(index=merge_key, columns="observer", values="disposition")
    assert not ((nested.code == "invariant") & (nested.point_plus_code == "witness")).any()
    facets = ["observer", "source", "draw"]
    summary, tasks = counts(frame, facets), counts(frame, ["dataset", *facets])
    assert (summary[STATES].sum(axis=1) == 704).all()
    mobility = pd.DataFrame(all_occupancy)
    for name, df in (("proposals", frame), ("summary", summary), ("tasks", tasks), ("mobility", mobility)):
        df.to_csv(DEST / f"{name}.csv", index=False)
    if record:
        receipt.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Write the recipe and input receipt on first collection")
    replay(parser.parse_args().record)
