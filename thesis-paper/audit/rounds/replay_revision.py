"""Compare the reusable veto scheduler with the archived path and the two NIPS executions."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from experiments.context_study.benchmark import fit
from experiments.context_study.data import DATA, OUT as PRIMARY, ROOT, config, digest
from experiments.context_study.working_set import oracle
from minppi_mofs.context_design import separating_contexts
from minppi_mofs.piv3_compact_witness import _sample
from minppi_mofs.veto_schedule import search_with_recent_veto


HERE = Path(__file__).resolve().parent
RUNS = ROOT / "results/context_study"


def schedule(name, fold):
    original = PRIMARY / name / f"fold-{fold}"
    tested = RUNS / "schedules" / name / f"fold-{fold}"
    lock = json.loads((original / "completed.json").read_bytes())
    source = DATA / name / f"{name}.tsv.gz"
    assert digest(source) == lock["data_sha256"]
    assert digest(original / "selection.json") == lock["selection_sha256"]
    record = json.loads((original / "selection.json").read_bytes())
    expected = json.loads((tested / "selection.json").read_bytes())
    timing = json.loads((tested / "completed.json").read_bytes())
    frame = pd.read_csv(source, sep="\t")
    y = frame.pop("target").map({v:i for i,v in enumerate(record["class_order"])}).to_numpy()
    x = frame.to_numpy(dtype=np.float32)
    train, rank = map(np.asarray, (record["train"], record["rank"]))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
    xt, xr = imputer.transform(x[train]), imputer.transform(x[rank])
    seed = 20421 + fold
    predictor = fit(xt, y[train], seed)
    task = "binary" if predictor.n_classes_ == 2 else "multiclass"
    ids, weights = _sample(y[rank], task, np.random.SeedSequence([seed, 659]))
    bg = np.random.default_rng(seed).choice(len(train), min(512, len(train)), replace=False)
    rare = task == "binary" and min(np.bincount(y[rank])) <= .05 * len(rank)
    order, k = record["order"], record["anchor_k"]
    anchor = frozenset(order[:k])
    ballot, cache = oracle(predictor, xr[ids], np.median(xt[bg], axis=0),
                           y[rank][ids], weights, anchor, rare)
    support, result = search_with_recent_veto(order[:k], separating_contexts(order[k:]), ballot)
    current = anchor
    observed = []
    for decision in result["trace"]:
        candidate = current.difference(decision["block"])
        observed.append([sorted(candidate), decision["accepted"]])
        if decision["accepted"]:
            current = candidate
    assert observed == expected["traces"]["recent"] == expected["traces"]["direct"]
    assert sorted(support) == expected["selected"]["recent"] == record["selected"]["antichain"]
    assert len(cache) == next(row["queries"] for row in timing["rows"] if row["arm"] == "recent")
    return dict(dataset=name, fold=fold, proposals=len(observed), distinct_masks=len(cache), retained=len(support))


def guarded_challenge():
    reference = RUNS / "nips2003"
    guarded = RUNS / "nips2003-guarded"
    config = json.loads((ROOT / "configs/context_study/nips2003.json").read_bytes())
    for name in config["datasets"]:
        for seed in config["seeds"]:
            a, b = [d / name / str(seed) for d in (reference, guarded)]
            old, new = [json.loads((d / "frozen.json").read_bytes()) for d in (a, b)]
            for field in ("supports", "train", "rank"):
                assert old[field] == new[field]
            with np.load(a/"predictions.npz") as pa, np.load(b/"predictions.npz") as pb:
                assert set(pa.files) == set(pb.files)
                for key in pa.files:
                    np.testing.assert_array_equal(pa[key], pb[key])
    a, b = [pd.read_csv(d / "scores.csv").sort_values(["dataset", "seed", "arm"]).reset_index(drop=True)
            for d in (reference, guarded)]
    assert len(a) == len(config["datasets"]) * len(config["seeds"]) * len(config["methods"])
    columns = [field for field in a if not field.endswith("_s")]
    pd.testing.assert_frame_equal(a[columns], b[columns], check_exact=True)
    return len(a)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        rows = [schedule(dataset, fold) for dataset in config()["datasets"] for fold in range(3)]
    with (HERE / "veto-schedule-replay.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} complete veto-schedule paths, {sum(r['distinct_masks'] for r in rows)} masks; "
          f"{guarded_challenge()} guarded challenge results match")
