"""Independently reconstruct the conditional codebooks and score selected small orbits."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import class_precision as study
from experiments.context_review import score_resolution as sr
from experiments.context_review import score_resolution_check as exhaustive
from experiments.context_review import transcript_permutations as tp


def reconstruct(types, probabilities, index, draw):
    counts = Counter(types.tolist())
    raw = {}
    for kind, probability in zip(types, probabilities, strict=True):
        quantized = tuple(min(15, int(16 * float(x))) for x in probability)
        pair = (int(np.argmax(probability)), quantized)
        if kind in raw:
            assert raw[kind] == pair
        raw[kind] = pair
    strata = defaultdict(list)
    for kind in sorted(raw):
        strata[counts[kind], raw[kind][0]].append(kind)
    rng = np.random.default_rng(np.random.SeedSequence([153611, index, draw]))
    permuted = {}
    for key, ids in sorted(strata.items()):
        donors = rng.permutation(ids)
        original = Counter(raw[t][1] for t in ids)
        assigned = Counter(raw[t][1] for t in donors)
        assert original == assigned
        for t, donor in zip(ids, donors, strict=True):
            assert counts[t] == counts[donor] and raw[t][0] == raw[donor][0]
            permuted[t] = raw[donor][1]
    codebook = {code: i for i, code in enumerate(sorted(set(permuted.values())))}
    vector = np.array([codebook[permuted[t]] for t in sorted(raw)], dtype="<i8")
    changed = [t for t in raw if raw[t][1] != permuted[t]]
    return (vector[types], sha256(vector.tobytes()).hexdigest(),
            len(changed), sum(counts[t] for t in changed))


def check():
    mobility = pd.read_csv(study.DEST / "mobility.csv").set_index(["dataset", "fold", "draw"])
    saved = pd.read_csv(study.DEST / "proposals.csv", keep_default_na=False)
    observed = saved[saved.source == "class_matched"].set_index(["dataset", "fold", "draw", "proposal", "observer"])
    assert len(mobility) == 42 * 5 and len(observed) == 704 * 2 * 5
    cases = []
    totals = Counter()
    folds = tp.rules.read(tp.DEST / "manifest.json")["folds"]
    with ZipFile(sr.DEST / "unmasked.zip") as store, ZipFile(tp.DEST / "row-partitions.zip") as partitions:
        for index, (key, meta) in enumerate(sorted(folds.items())):
            with np.load(BytesIO(store.read(key + ".npz"))) as data, np.load(BytesIO(partitions.read(key + ".npz"))) as ds:
                x, t = data["probabilities"], data["types"]
                for draw in range(1, 6):
                    codes, checksum, changed, positions = reconstruct(t, x, index, draw)
                    identity = meta["dataset"], meta["fold"], draw
                    archived = mobility.loc[identity]
                    assert (archived.codebook_sha256, archived.changed_types, archived.changed_rows) == (
                        checksum, changed, positions)
                    if draw != 1:
                        continue
                    y, anchor = ds["y"], ds["anchor"]
                    base = codes[:, None]
                    for i, proposal in enumerate(meta["proposals"]):
                        for observer in ("code", "point_plus_code"):
                            answer = observed.loc[(*identity, i, observer)].disposition
                            if answer == "unresolved" or totals[observer, answer] >= 3:
                                continue
                            provided = base if observer == "code" else np.column_stack((base, ds["codes"][proposal["point"]].T))
                            cells = exhaustive.cells_for(provided, y)
                            result = exhaustive.enumerate_decisions(y, anchor, ds[f"s{i}"], cells)
                            if result is not None:
                                n, exact_answer = result
                                assert exact_answer == answer
                                totals[observer, answer] += 1
                                cases.append(dict(dataset=identity[0], fold=identity[1], draw=draw, proposal=i,
                                                  observer=observer, disposition=answer, allocations=n))
    assert cases and {c["disposition"] for c in cases} == {"invariant", "witness"}
    registered = study.counts(saved, ["observer", "source", "draw"])
    pd.testing.assert_frame_equal(pd.read_csv(study.DEST / "summary.csv"), registered.rename_axis(None, axis=1))
    result = dict(independently_rebuilt_codebooks=len(mobility), class_matched_proposal_conditions=len(observed),
                  exhaustive_cases=cases)
    (study.DEST / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(f"{len(mobility)} row codebooks matched and {len(cases)} nontrivial finite orbits enumerated")


if __name__ == "__main__":
    check()
