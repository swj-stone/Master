"""Query-matched random prefixes and an exhaustive audit for the two small new sources."""
import argparse
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import chain_validation as panel, oracle_audit
from minppi_mofs import dictator_minpi


OUT = panel.OUT / "matched-control"


def select():
    panel.bind_cpu()
    OUT.mkdir(exist_ok=True)
    lock = json.loads((panel.OUT / "lock.json").read_bytes())
    assert len(lock) == 20
    results = {}
    for key, digest in lock.items():
        primary = panel.OUT / key / "selection.json"
        assert panel.b.digest(primary) == digest
        record = json.loads(primary.read_bytes())
        chain = next(r for r in record["rows"] if r["method"] == "chain")
        name, seed = record["dataset"], record["seed"]
        x, y, domain, idx = panel.data.split(name, seed)
        y = y.copy()
        y[idx["test"]] = -1
        budget = chain["work"]["refinement_queries"]
        random_rank = partial(panel.chain_minpi.rank_chain, random_queries=budget)
        with patch.object(panel.chain_minpi, "rank_chain", random_rank), patch.object(panel.b, "ranked", panel.rank):
            rec, pred = panel.b.select_one("chain", x, y, domain, idx, seed)
        assert rec["work"]["distinct_votes"] == chain["work"]["distinct_votes"]
        assert rec["work"]["anchors"] == chain["work"]["anchors"]
        path = OUT / f"{key}.npz"
        np.savez_compressed(path, prediction=pred)
        results[key] = dict(primary=digest, random=rec, prediction=panel.b.digest(path))
    results = dict(hashes={**panel.signatures(), "dictator_minpi.py":panel.b.digest(Path(dictator_minpi.__file__)),
                           "control":panel.b.digest(Path(__file__))}, jobs=results)
    (OUT / "lock.json").write_text(json.dumps(results),encoding="utf8")
    print(f"Sealed all {len(lock)} matched-budget forecasts")


def audit():
    records = json.loads((OUT / "lock.json").read_bytes())
    assert len(records["jobs"]) == 20
    rows, oracle = [], []
    with ZipFile(panel.EVIDENCE / "matched-forecasts.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(OUT / "lock.json", "lock.json")
        for key, record in records["jobs"].items():
            primary = panel.OUT / key / "selection.json"
            assert record["primary"] == panel.b.digest(primary)
            state = json.loads(primary.read_bytes())
            name, seed = state["dataset"], state["seed"]
            x, y, task, idx = panel.data.split(name, seed)
            chain = next(r for r in state["rows"] if r["method"] == "chain")
            old = next(r for r in state["rows"] if r["method"] == "minpi")
            random = record["random"]
            path = OUT / f"{key}.npz"
            assert record["prediction"] == panel.b.digest(path)
            archive.write(path, path.name)
            with np.load(path) as data:
                pred = data["prediction"]
            panel.verify_predictions(x, y, task, idx, random, pred)
            dev = np.r_[idx["fit"],idx["vote"],idx["width"]]
            metrics = panel.b.metrics(y[idx["test"]], pred, task, y[dev].std())
            rows.append(dict(dataset=name,seed=seed,task=task,
                             queries=chain["work"]["distinct_votes"],refine=chain["work"]["refinement_queries"],
                             chain_neg=chain["minimum"].count(-1),random_neg=random["minimum"].count(-1),
                             original_neg=old["minimum"].count(-1),k_random=random["k"],k_chain=chain["k"],
                             same_support=set(chain["order"][:chain["k"]])==set(random["order"][:random["k"]]),
                             **metrics))
            if x.shape[1] <= 9:
                t, v = idx["fit"], idx["vote"]
                prep = panel.b.transform(x,t)
                votes = oracle_audit.complete_votes(prep.transform(x[t]),y[t],prep.transform(x[v]),y[v],task)
                for j in range(x.shape[1]):
                    base = np.array([s for s in range(len(votes)) if not s & (1 << j)])
                    exact = (votes[base | (1 << j)]-votes[base]).min()
                    estimates = dict(original=old["minimum"][j], chain=chain["minimum"][j], random=random["minimum"][j])
                    assert min(estimates.values()) >= exact
                    oracle.append(dict(dataset=name,seed=seed,feature=j,exact=exact,**estimates))
    scores = pd.DataFrame(rows)
    scores.to_csv(panel.EVIDENCE / "matched-control.csv",index=False)
    exact = pd.DataFrame(oracle)
    exact.to_csv(panel.EVIDENCE / "exact-minima.csv",index=False)
    print(scores.groupby("dataset")[["queries","refine","chain_neg","random_neg","original_neg","same_support"]].sum())
    print(exact[exact.exact==-1].groupby("dataset")[["exact","original","chain","random"]].agg(lambda s:(s==-1).sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["select","audit"])
    with threadpool_limits(limits=1):
        (select if parser.parse_args().action=="select" else audit)()
