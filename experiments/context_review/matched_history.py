"""Retrospective check of how much labeled history native selectors receive."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.context_review import drift, external_har
from experiments.context_study.benchmark import fit, store
from experiments.context_study.data import digest


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/context_review/matched_history.json"
OUT = ROOT / "results/context_review/matched-history"


def read(path):
    return json.loads(path.read_bytes())


def say(message):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    with (OUT / "run.log").open("a", encoding="utf8") as handle:
        print(line, file=handle)
    print(line, flush=True)


def inputs(dataset):
    if dataset == "har":
        with external_har.original_archive() as archive:
            x = external_har.load_member(archive, "train", "X")
            y = external_har.load_member(archive, "train", "y") - 1
            subject = external_har.load_member(archive, "train", "subject")
            test = external_har.load_member(archive, "test", "X")
            unit = external_har.load_member(archive, "test", "subject")
        split = read(external_har.OUT / "split.json")
        history = {15: np.array(split["fit"]), 21: np.arange(len(x))}
        assert set(subject[history[15]]) == set(split["train_subjects"])
        assert len(set(subject[history[15]])) == 15 and not set(subject) & set(unit)
    else:
        with ZipFile(drift.DATA) as archive:
            features = [drift.load_features(archive, b) for b in range(1, 11)]
            y = np.concatenate([drift.load_labels(archive, b) for b in range(1, 8)])
        x = np.vstack(features[:7])
        test = np.vstack(features[7:])
        unit = np.repeat([8, 9, 10], [len(f) for f in features[7:]])
        history = {k: np.arange(sum(len(f) for f in features[:k])) for k in (5, 6, 7)}
    assert np.isfinite(x).all() and np.isfinite(test).all()
    return x, y, test, unit, history


def fit_grid(dataset, grid):
    x, y, test, unit, histories = inputs(dataset)
    for h in grid["histories"]:
        rows = histories[h]
        for method in ("rfe", "anova"):
            for width in grid[f"{method}_widths"]:
                name = f"{dataset}_{h}_{method}_{width}"
                say(f"Selecting {name} using {len(rows)} rows")
                begin = perf_counter()
                train = x[rows]
                if dataset == "gas":
                    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                    train = imputer.fit_transform(train)
                selector = (RFE(external_har.estimator(), n_features_to_select=width, step=.2)
                            if method == "rfe" else SelectKBest(f_classif, k=width))
                selector.fit(train, y[rows])
                cols = selector.get_support(indices=True)
                selection_s = perf_counter() - begin
                begin = perf_counter()
                xa, xe = x, test
                if dataset == "gas":
                    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
                    xa, xe = imputer.transform(x), imputer.transform(test)
                pred = fit(xa[:, cols], y, external_har.SEED).predict_proba(xe[:, cols])
                refit_s = perf_counter() - begin
                assert len(cols) == width and pred.shape == (len(test), 6)
                np.savez_compressed(OUT / f"{name}.npz", probabilities=pred, units=unit)
                store(OUT / f"{name}.json", dict(name=name, dataset=dataset, method=method,
                      history=h, rows=len(rows), k=width, selected=cols.tolist(),
                      selection_s=selection_s, refit_predict_s=refit_s,
                      prediction_sha256=digest(OUT / f"{name}.npz")))
                say(f"Saved {name} predictions, {selection_s:.2f}s selection, {refit_s:.2f}s refit/predict")


def verify_original(dataset, record, probabilities):
    h, width, method = record["history"], record["k"], record["method"]
    if dataset == "gas" and h == 7:
        arm = (("rfe_at_challenger_width" if width == 5 else "rfe_at_vote_width")
               if method == "rfe" else ("anova_at_10pct" if width == 13 else "anova_at_25pct"))
        expected = read(drift.OUT / "supports.json")["supports"][arm]
        with np.load(drift.OUT / "predictions.npz") as saved:
            np.testing.assert_array_equal(probabilities, saved[arm])
    elif dataset == "har" and h == 21:
        folder = external_har.OUT / "rfe_donor"
        expected = read(folder / "support.json")["selected"]
        with np.load(folder / "predictions.npz") as saved:
            np.testing.assert_array_equal(probabilities, saved["probabilities"])
    else:
        return False
    assert expected == record["selected"]
    return True


def evaluate(dataset):
    if dataset == "gas":
        with ZipFile(drift.DATA) as archive:
            y = np.concatenate([drift.load_labels(archive, b) for b in (8, 9, 10)])
    else:
        with external_har.original_archive() as archive:
            y = external_har.load_member(archive, "test", "y") - 1
    summary, cases = [], []
    for path in sorted(OUT.glob(f"{dataset}_*.json")):
        rec = read(path)
        assert digest(path.with_suffix(".npz")) == rec["prediction_sha256"]
        with np.load(path.with_suffix(".npz")) as saved:
            p, units = saved["probabilities"], saved["units"]
        match = verify_original(dataset, rec, p)
        take = ["dataset", "history", "method", "k", "rows", "selection_s", "refit_predict_s"]
        line = {k: rec[k] for k in take}
        series = []
        for unit in np.unique(units):
            use = units == unit
            series.append(dict(unit=unit, n=use.sum(), **drift.score(y[use], p[use])))
        cases.extend([{**line, **s} for s in series])
        if dataset == "gas":
            primary = np.mean([s["recall"] for s in series])
            accuracy = np.mean([s["accuracy"] for s in series])
            auc = np.mean([s["auc"] for s in series])
        else:
            primary = np.mean([s["accuracy"] for s in series])
            accuracy = accuracy_score(y, p.argmax(axis=1))
            auc = roc_auc_score(y, p, multi_class="ovr", average="macro")
        summary.append(dict(**line, primary=primary, accuracy=accuracy, auc=auc,
                       total_s=rec["selection_s"] + rec["refit_predict_s"],
                       reproduced_original=match))
    table = pd.DataFrame(summary)
    table.to_csv(OUT / f"{dataset}-summary.csv", index=False)
    pd.DataFrame(cases).to_csv(OUT / f"{dataset}-by-unit.csv", index=False)
    print(table.round(5).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["har", "gas"])
    args = parser.parse_args()
    grid = read(SPEC)
    if args.worker:
        with threadpool_limits(limits=1):
            return fit_grid(args.worker, grid[args.worker])
    OUT.mkdir(parents=True, exist_ok=True)
    drift.acquire()
    external_har.acquire()
    sources = [SPEC, Path(__file__), Path(drift.__file__), Path(external_har.__file__),
               ROOT / "experiments/context_study/benchmark.py", external_har.OUT / "split.json"]
    lock = {str(p.relative_to(ROOT).as_posix()): digest(p) for p in sources}
    lock.update(gas_archive=digest(drift.DATA), har_archive=digest(external_har.DATA))
    if (OUT / "protocol.json").exists() and read(OUT / "protocol.json") != lock:
        raise ValueError("Existing protocol has different inputs")
    store(OUT / "protocol.json", lock)
    for dataset in ("gas", "har"):
        if not (OUT / f"{dataset}-frozen.json").exists():
            say(f"Starting fixed retrospective {dataset} grid")
            subprocess.run([sys.executable, "-m", "experiments.context_review.matched_history",
                            "--worker", dataset], check=True, timeout=3600)
            artifacts = sorted(OUT.glob(f"{dataset}_*.json"))
            expected = len(grid[dataset]["histories"]) * (
                len(grid[dataset]["rfe_widths"]) + len(grid[dataset]["anova_widths"]))
            assert len(artifacts) == expected
            store(OUT / f"{dataset}-frozen.json", {p.name: digest(p) for p in artifacts})
        for filename, sha in read(OUT / f"{dataset}-frozen.json").items():
            assert digest(OUT / filename) == sha
        say(f"Scoring frozen {dataset} controls")
        evaluate(dataset)


if __name__ == "__main__":
    main()
