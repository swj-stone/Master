"""Time the frozen two-tier rank with a concentration-based support budget."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.grouped_solver_cost_audit import fit
from experiments.minpi_backward_confirmation import evaluate
from research.conditional_dense.confirmation import Panel, ROOT, SEEDS, panels as dense_panels
from research.conditional_dense.shared_method import conditional_support
from research.conditional_dense.sketch_f.study import sample_rows, sketch_f
from research.conditional_dense.speech import ARCHIVE, SHA256, utterances
from research.conditional_dense.vote_width.method import vote_support
from research.feature_game_eval.audit_controls import effect_size


METHODS = ("vote_width", "conditional", "full", "F", "F_half", "random", "sketch_F")
REPEATS = len(METHODS)
HERE = Path(__file__).resolve().parent


def panels():
    yield from dense_panels()
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == SHA256
    with ZipFile(ARCHIVE) as archive:
        X, y = utterances(archive.read("Train_Arabic_Digit.txt"), 660)
        Xt, truth = utterances(archive.read("Test_Arabic_Digit.txt"), 220)
    for seed in SEEDS:
        yield Panel("spoken_arabic_digit", 1, seed, X, y, Xt, truth, None)


def deploy(panel, method, width):
    start = perf_counter_ns()
    if method == "vote_width":
        cols, _ = vote_support(panel.X, panel.y, seed=panel.seed)
    elif method == "conditional":
        cols, _ = conditional_support(panel.X, panel.y, seed=panel.seed)
    elif method == "sketch_F":
        cols = sketch_f(panel.X, panel.y, width, panel.seed)
    else:
        budget = (panel.X.shape[1] + 1) // 2 if method == "F_half" else width
        cols = choose(method.removesuffix("_half").lower(), panel.X, panel.y,
                      "classification", panel.seed, budget)
    selected = perf_counter_ns()
    pred, decision, times = fit(panel, cols, "cholesky")
    end = perf_counter_ns()
    return cols, pred, decision, dict(select_ms=(selected-start)/1e6,
                                     total_ms=(end-start)/1e6, **times)


def run(out, pass_id):
    out.mkdir(parents=True, exist_ok=False)
    paths = [HERE/"study.py", HERE/"method.py", HERE/"protocol.md",
             HERE.parent/"shared_method.py", HERE.parent/"sketch_f/study.py",
             HERE.parent/"confirmation.py", HERE.parent/"speech.py",
             ROOT/"experiments/grouped_solver_cost_audit.py",
             ROOT/"experiments/compare_predictive_minpi.py",
             ROOT/"experiments/minpi_backward_confirmation.py",
             ROOT/"research/feature_game_eval/audit_controls.py"]
    protocol = dict(process=pass_id, threads=1, repeats=REPEATS, methods=METHODS,
                    numpy=np.__version__, sklearn=sklearn.__version__, panels=[],
                    source_sha256={p.relative_to(ROOT).as_posix():
                                   hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    fields = ("panel", "source", "seed", "fold", "p", "k", "repeat", "position", "method",
              "score", "select_ms", "slice_ms", "fit_ms", "predict_ms", "total_ms")
    with (out/"trials.csv").open("x", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for panel in panels():
            key = f"{panel.name}-{panel.fold}-{panel.seed}"
            proposed, info = vote_support(panel.X, panel.y, seed=panel.seed)
            width = len(proposed)
            cache = {"truth": panel.gold, "votes": info["rank"]}
            if panel.units is not None:
                cache["units"] = panel.units
            historical = ("conditional-speech" if panel.name=="spoken_arabic_digit"
                          else "conditional-dense") + "-quality-20261118a"
            effects = effect_size(panel.X, panel.y, "classification")
            rows = sample_rows(panel.y, panel.seed)
            sample_effects = effect_size(panel.X[rows], panel.y[rows], "classification")
            validation = {}
            with np.load(ROOT/"results/minpi-study"/historical/(key+".npz")) as old:
                for arm in METHODS:
                    cols, pred, decision, _ = deploy(panel, arm, width)
                    for suffix, value in (("cols",cols),("pred",pred),("decision",decision)):
                        cache[arm+"_"+suffix] = value
                        if arm in ("conditional","full","F_half"):
                            np.testing.assert_array_equal(value,old[arm+"_"+suffix])
                    if arm in ("F","F_half","sketch_F"):
                        eta = sample_effects if arm == "sketch_F" else effects
                        outside = np.setdiff1d(np.arange(len(eta)), cols)
                        gap = float(max(0., eta[outside].max() - eta[cols].min())) if len(outside) else 0.
                        validation[arm] = gap
                        assert gap <= 1e-5, (key, arm, gap)
            np.savez_compressed(out/(key+".npz"), **cache)
            protocol["panels"].append(dict(panel=key, p=panel.X.shape[1],
                 k=width, original_width=info["width"], F_boundary_gaps=validation))
            order = np.random.default_rng([panel.seed,panel.fold,pass_id,21187]).permutation(METHODS)
            for repeat in range(REPEATS):
                for position, arm in enumerate(np.roll(order,repeat)):
                    cols,pred,decision,timings = deploy(panel,arm,width)
                    for suffix, value in (("cols",cols),("pred",pred),("decision",decision)):
                        np.testing.assert_array_equal(value, cache[arm+"_"+suffix])
                    score, _ = evaluate(panel.name, panel.gold, pred, panel.units, "classification")
                    writer.writerow(dict(panel=key,source=panel.name,seed=panel.seed,fold=panel.fold,
                            p=panel.X.shape[1],k=len(cols),repeat=repeat,position=position,
                            method=arm,score=score,**timings))
            fp.flush()
            print(key, info["width"], width, flush=True)
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--pass-id",required=True,type=int,choices=(1,2))
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output,args.pass_id)
