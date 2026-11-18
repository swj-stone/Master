"""Pair the frozen concentration budget with the original binary and continuous panels."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.conditional_dense.original_tasks.regression import regression_geometric_support, regression_support
from research.conditional_dense.shared_method import conditional_support
from research.conditional_dense.vote_width.method import vote_support
from research.feature_game_eval import panels as bench
from research.feature_game_eval.audit_controls import effect_size


HERE = Path(__file__).resolve().parent
BINARY = ("spambase", "madelon", "secom")
REGRESSION = ("abalone", "crimedata", "diabetes", "supercon")
METHODS = ("vote_width", "geometric", "full", "F", "random")
KEY = ["dataset", "fold", "seed"]


def train_features(panel):
    if panel.name == "secom":
        return bench._impute_validation(panel.X, panel.X[:0])[0]
    return panel.X


def select(panel, method, width):
    if method in ("full", "random"):
        return bench.select(method, panel, width)
    X = train_features(panel)
    if method == "F":
        centered = X.astype(np.float64, copy=True)
        centered -= centered.mean(axis=0)
        return bench.choose("f", centered, panel.y, panel.task, panel.seed, width)
    if panel.task == "regression":
        selector = regression_support if method == "vote_width" else regression_geometric_support
    else:
        selector = vote_support if method == "vote_width" else conditional_support
    return selector(X, panel.y, seed=panel.seed)[0]


def deploy(panel, method, width):
    start = perf_counter_ns()
    cols = select(panel, method, width)
    selected = perf_counter_ns()
    prediction, decision = bench.fit(panel, cols)
    end = perf_counter_ns()
    return cols, prediction, decision, (selected-start)/1e6, (end-start)/1e6


def benchmark(out, reference):
    pass_id = 2 if reference is not None else 1
    registry = pd.read_csv(bench.ARCHIVE/"adaptive-69-20261117a"/"inputs.csv")
    registry = registry[registry.dataset.isin(BINARY+REGRESSION)].set_index(KEY)
    quality = pd.read_csv(bench.ARCHIVE/"residual-126-corrected"/"quality"/"arrays.csv").set_index("panel")
    assert len(registry) == 45
    out.mkdir(parents=True, exist_ok=False)
    seen, checks = set(), []
    sources = [HERE/"protocol.md", HERE/"study.py", HERE/"regression.py", HERE.parent/"shared_method.py",
               HERE.parent/"vote_width"/"method.py", Path(bench.__file__), Path(bench.original.__file__),
               Path(bench.original.previous.__file__), Path(bench.deployment.__file__),
               bench.ROOT/"experiments/compare_predictive_minpi.py", bench.ROOT/"src/minppi_mofs/stable_minpi.py"]
    state = {"process":pass_id, "threads":1, "methods":METHODS, "repeats":len(METHODS),
             "sklearn":bench.sklearn.__version__, "numpy":np.__version__,
             "sha256":{f.relative_to(bench.ROOT).as_posix():bench.checksum(f) for f in sources}}
    if reference is not None:
        previous = json.loads((reference/"run.json").read_text(encoding="utf-8"))
        assert previous["sha256"] == state["sha256"] and previous["process"] == 1
    fields = [*KEY,"task","p","k","method","repeat","position","score","average_precision","select_ms","total_ms"]
    with (out/"times.csv").open("x",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for panel in bench.original.previous.panels():
            key = panel.name, panel.fold, panel.seed
            if panel.name not in BINARY + REGRESSION:
                assert panel.task == "classification" and np.unique(panel.y).size > 2
                continue
            assert (np.unique(panel.y).size == 2) == (panel.name in BINARY)
            assert (panel.task == "regression") == (panel.name in REGRESSION)
            assert key not in seen
            seen.add(key)
            cfg = dict(zip(KEY, key))
            expected = cfg | registry.loc[key].to_dict()
            if bench.original.input_hashes(panel) != expected:
                raise ValueError(f"Original data or split changed for {key}")
            identifier = f"{panel.name}-{panel.fold}-{panel.seed}"
            prior = quality.loc["original-"+identifier]
            historical = bench.ROOT/prior.source
            assert bench.checksum(historical) == prior.sha256
            p = panel.X.shape[1]
            width = None
            arrays = {"truth":panel.gold}
            scores = {}
            for method in METHODS:
                cols, pred, decision, _, _ = deploy(panel, method, width)
                if method == "vote_width":
                    width = len(cols)
                if method in ("F","random"):
                    assert len(cols) == width
                if method == "full":
                    with np.load(historical) as archived:
                        np.testing.assert_array_equal(pred, archived["full_pred"])
                if method == "F":
                    values = effect_size(train_features(panel), panel.y, panel.task)
                    outside = np.setdiff1d(np.arange(p), cols)
                    gap = max(0., float(np.max(values[outside]) - np.min(values[cols]))) if outside.size else 0.
                    exact = np.array_equal(cols, np.sort(np.argsort(-values, kind="stable")[:width]))
                    checks.append(dict(**cfg, width=width, independent_F_gap=gap, exact_F_set=exact))
                    assert gap <= 1e-9
                scores[method] = bench.deployment.score_predictions(
                    panel.name, panel.gold, pred, panel.units, panel.task, decision)[::2]
                arrays[method+"_cols"], arrays[method+"_pred"] = cols, pred
                if panel.name == "secom":
                    arrays[method+"_decision"] = decision
            if reference is not None:
                with np.load(reference/(identifier+".npz")) as recorded:
                    assert set(arrays) == set(recorded.files)
                    for name, value in arrays.items():
                        np.testing.assert_array_equal(recorded[name], value)
            np.savez_compressed(out/(identifier+".npz"), **arrays)
            schedule = np.random.default_rng([64631, panel.fold, panel.seed, pass_id]).permutation(METHODS)
            for repeat in range(len(METHODS)):
                for position, method in enumerate(np.roll(schedule, repeat)):
                    cols, pred, decision, select_ms, total_ms = deploy(panel, method, width)
                    np.testing.assert_array_equal(cols, arrays[method+"_cols"])
                    np.testing.assert_array_equal(pred, arrays[method+"_pred"])
                    if panel.name == "secom":
                        np.testing.assert_array_equal(decision, arrays[method+"_decision"])
                    score, ap = scores[method]
                    writer.writerow(dict(**cfg, task=panel.task, p=p, k=len(cols), method=method,
                                         repeat=repeat, position=position, score=score, average_precision=ap,
                                         select_ms=select_ms, total_ms=total_ms))
            stream.flush()
            print(f"{len(seen)}/45 {identifier}: {width}/{p}", flush=True)
    assert seen == set(registry.index)
    pd.DataFrame(checks).to_csv(out/"f-audit.csv",index=False)
    state["configurations"] = len(seen)
    (out/"run.json").write_text(json.dumps(state,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        benchmark(args.output, args.reference)
