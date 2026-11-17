"""Apply the frozen six-method selector to archived source-specific development and test folds."""
import argparse
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.context_review.timing_process import bind_logical_cpu
from experiments.minpi_compression import benchmark as bench


DONOR = bench.ROOT / "results/context_review/wide-uci"
OUT = bench.RUN / "wide-transfer"
EVIDENCE = bench.EVIDENCE / "wide-transfer"


def folds():
    folders = sorted(DONOR.glob("*/fold-*/inputs.npz"))
    assert len(folders) == 16 and len({p.parent.parent for p in folders}) == 6
    for path in folders:
        yield path.parent, f"{path.parent.parent.name}-{path.parent.name}"


def select():
    cpu = bind_logical_cpu()
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for directory, key in folds():
        config = json.loads((directory / "layout.json").read_bytes())
        assert bench.digest(directory / "inputs.npz") == config["inputs_sha256"]
        with np.load(directory / "inputs.npz") as data:
            train, query = data["x"][:, :100], data["x_outer"][:, :100]
            x = np.concatenate([train, query]).astype(float)
            y = np.r_[data["y"], np.full(len(query), -1, dtype=int)]
            idx = dict(fit=data["train"], vote=data["proposal"], width=data["guard"],
                       test=np.arange(len(train), len(x)))
            test_row = data["test_row"]
        pools = np.concatenate(list(idx.values()))
        assert len(np.unique(pools)) == len(pools)
        seed = 42 + int(directory.name.split("-")[-1])
        methods = np.random.default_rng(seed).permutation(bench.METHODS)
        records, predictions = [], {}
        for repeat in range(3):
            for method in np.roll(methods, repeat * 2):
                record, prediction = bench.select_one(method, x, y, "classification", idx, seed)
                if repeat:
                    np.testing.assert_array_equal(prediction, predictions[method])
                predictions[method] = prediction
                records.append(dict(**record, repeat=repeat))
        dest = OUT / key
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest / "predictions.npz", test_row=test_row, **predictions)
        state = dict(inputs_sha256=config["inputs_sha256"], seed=seed, cpu=cpu, rows=records,
                     predictions=bench.digest(dest / "predictions.npz"), sources=bench.stamp())
        (dest / "selection.json").write_text(json.dumps(state), encoding="utf8")
        receipts[key] = bench.digest(dest / "selection.json")
        print(f"[{len(receipts)}/16] sealed {key}", flush=True)
    (OUT / "lock.json").write_text(json.dumps(receipts, indent=2), encoding="utf8")


def score():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    receipts = json.loads((OUT / "lock.json").read_bytes())
    assert set(receipts) == {key for _, key in folds()}
    rows = []
    with ZipFile(EVIDENCE / "forecasts.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(OUT / "lock.json", "lock.json")
        for directory, key in folds():
            dest = OUT / key
            state = json.loads((dest / "selection.json").read_bytes())
            config = json.loads((directory / "layout.json").read_bytes())
            assert bench.digest(dest / "selection.json") == receipts[key]
            assert bench.digest(dest / "predictions.npz") == state["predictions"]
            assert bench.digest(directory / "targets.npz") == config["targets_sha256"]
            for f in ("selection.json", "predictions.npz"):
                archive.write(dest / f, f"{key}/{f}")
            archive.write(directory / "targets.npz", f"{key}/targets.npz")
            archive.write(directory / "layout.json", f"{key}/layout.json")
            with np.load(dest / "predictions.npz") as pred, np.load(directory / "targets.npz") as target:
                np.testing.assert_array_equal(target["test_row"], pred["test_row"])
                for method in bench.METHODS:
                    runs = [r for r in state["rows"] if r["method"] == method]
                    rec = runs[0]
                    assert all(r["order"] == rec["order"] and r["k"] == rec["k"] for r in runs)
                    clocks = {k: np.median([r[k] for r in runs]) for k in ("rank_s","selection_s","deployment_s")}
                    metric = bench.metrics(target["y"], pred[method], "classification", 1.)
                    rows.append(dict(dataset=directory.parent.name, fold=directory.name, method=method,
                        p=rec["p"], k=rec["k"], retained=rec["k"]/rec["p"],
                        selected=" ".join(map(str,sorted(rec["order"][:rec["k"]]))),
                        negative=rec.get("minimum", []).count(-1), **metric, **clocks))
    frame = pd.DataFrame(rows)
    frame.to_csv(EVIDENCE / "scores.csv", index=False)
    frame.groupby(["dataset","method"]).mean(numeric_only=True).to_csv(EVIDENCE / "sources.csv")
    minpi = frame[frame.method == "minpi"].set_index(["dataset","fold"])
    contrasts = []
    for method in bench.METHODS:
        control = frame[frame.method == method].set_index(["dataset","fold"])
        d = pd.DataFrame({k: minpi[k] - control[k] for k in ("auc","accuracy")})
        for key in ("k","selection_s","deployment_s"):
            d[key] = np.log(minpi[key] / control[key])
        row = d.groupby(level="dataset").mean().mean()
        row[["k","selection_s","deployment_s"]] = np.exp(row[["k","selection_s","deployment_s"]])
        contrasts.append(dict(method=method, **row.to_dict()))
    result = pd.DataFrame(contrasts)
    result.to_csv(EVIDENCE / "contrasts.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "score"])
    with threadpool_limits(limits=1):
        (select if parser.parse_args().action == "select" else score)()
