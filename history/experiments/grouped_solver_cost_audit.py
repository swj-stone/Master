"""Two-case diagnostic: price both original LSQR and dense Cholesky baselines.

This does not tune any minPI parameter or establish cross-source performance.
Every timed arm selects afresh, including the scaled pilot's calibration cost.
"""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments import budget_minpi_deployments as deployment
from experiments.compare_predictive_minpi import choose
from experiments.minpi_study import P
from experiments.shared_grouped_development import panels
from experiments.stable_minpi_stress import ROOT, digest, file_hash
from minppi_mofs.grouped_minpi import grouped_top_from_training
from minppi_mofs.scaled_pilot_grouped_minpi import scaled_pilot_grouped_top_from_training
from minppi_mofs.prefix_factor_minpi import factorized_grouped_top_from_training


SELECTORS = ("full", "grouped50", "F50", "random50", "shared_c",
             "pivotal_bestcal", "F_bestcal", "representatives_bestcal")
SOLVERS = ("lsqr", "cholesky")
ARMS = tuple((selector, solver) for selector in SELECTORS for solver in SOLVERS)
REFERENCE = ROOT / "results/minpi-study/shared-grouped-development-20261117c"
BEST_REFERENCE = ROOT / "results/minpi-study/shared-grouped-bestcal-development-20261117a"


def select(selector, panel):
    p = panel.X.shape[1]
    if selector == "full":
        return np.arange(p)
    if selector == "shared_c":
        return scaled_pilot_grouped_top_from_training(panel.X, panel.y, random_state=panel.seed)[0]
    if selector.endswith("_bestcal"):
        rank = selector.removesuffix("_bestcal")
        return factorized_grouped_top_from_training(panel.X, panel.y, random_state=panel.seed,
                                                    ranking=rank, budget_rule="bestcal")[0]
    if selector == "grouped50":
        return grouped_top_from_training(panel.X, panel.y, ceil(p/2), random_state=panel.seed)[0]
    return choose("f" if selector == "F50" else "random", panel.X, panel.y,
                  panel.task, panel.seed, ceil(p/2))


def fit(panel, columns, solver):
    start = perf_counter_ns()
    A, B = ((panel.X, panel.Xt) if len(columns) == panel.X.shape[1]
            else (panel.X[:, columns], panel.Xt[:, columns]))
    sliced = perf_counter_ns()
    model = make_pipeline(StandardScaler(), RidgeClassifier(
        alpha=1., solver=solver, tol=P["ridge_tol"], max_iter=P["ridge_max_iter"]))
    model.fit(A, panel.y)
    fitted = perf_counter_ns()
    decision = model.decision_function(B)
    prediction = model.classes_[(decision > 0).astype(int)] if decision.ndim == 1 else model.classes_[decision.argmax(axis=1)]
    end = perf_counter_ns()
    return prediction, decision, dict(slice_ms=(sliced-start)/1e6,
                                      fit_ms=(fitted-sliced)/1e6,
                                      predict_ms=(end-fitted)/1e6)


def protocol(pass_id):
    paths = [Path(__file__), ROOT/"configs/minpi-study.yaml"]
    paths += [ROOT/"src/minppi_mofs"/(name+".py") for name in (
        "shared_grouped_minpi", "scaled_pilot_grouped_minpi", "bestcal_grouped_minpi",
        "prefix_factor_minpi", "grouped_minpi",
        "certified_minpi", "exact_minpi", "word_minpi", "integer_minpi", "piv2")]
    paths += [ROOT/"experiments"/(name+".py") for name in (
        "shared_grouped_development", "budget_minpi_deployments", "fast_minpi_five_source",
        "compare_predictive_minpi", "minpi_study", "stable_minpi_stress")]
    # Use the actual configured path; installations can keep it under another name.
    from experiments.minpi_study import CONFIG
    paths[1] = CONFIG
    return dict(pass_id=pass_id, repeats=5, stage="two_development_cases_solver_cost_diagnostic",
                cases=["isolet-1-1129", "digits-1-1129"], selectors=SELECTORS, solvers=SOLVERS,
                alpha=1., threads=1, numpy=np.__version__, sklearn=sklearn.__version__,
                original_lsqr_tol=P["ridge_tol"], original_lsqr_max_iter=P["ridge_max_iter"],
                timing="Fresh selection/calibration + slicing + scaler/model fit + prediction; one warmup then five shuffled rotations per process",
                warning="Same supports across solvers; Cholesky predictions may differ and are evaluated independently. No cross-source claim.",
                reference_sha256={f.relative_to(ROOT).as_posix(): file_hash(f)
                                  for base in (REFERENCE, BEST_REFERENCE)
                                  for f in (base/"isolet-1-1129.npz", base/"digits-1-1129.npz")},
                source_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths})


def run(output, pass_id, verify):
    state = protocol(pass_id)
    assert P["linear_alpha"] == 1.
    if verify:
        assert json.loads((output/"protocol.json").read_text(encoding="utf8")) == state
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/"protocol.json").write_text(json.dumps(state, indent=2), encoding="utf8")
    rows, replays, inputs, comparison = [], [], [], []
    for panel in panels():
        p = panel.X.shape[1]
        assert panel.task == "classification" and len(panel.X) > p
        cache, quality = {}, {}
        with np.load(REFERENCE/f"{panel.name}-1-1129.npz") as historical:
            for selector in SELECTORS:
                columns = select(selector, panel)
                for solver in SOLVERS:
                    key = selector+"_"+solver
                    pred, decision, _ = fit(panel, columns, solver)
                    score, _, _ = deployment.score_predictions(panel.name, panel.gold, pred,
                                                               panel.units, panel.task, decision)
                    cache[key+"_columns"], cache[key+"_pred"], cache[key+"_decision"] = columns, pred, decision
                    quality[key] = score
                    if solver == "lsqr" and selector in ("full", "grouped50", "shared_c"):
                        reference_name = "shared" if selector == "shared_c" else selector
                        np.testing.assert_array_equal(columns, historical[reference_name+"_cols"])
                        np.testing.assert_array_equal(pred, historical[reference_name+"_pred"])
                    if selector == "full" and solver == "lsqr":
                        np.testing.assert_array_equal(pred, panel.historical_full)
                    if solver == "lsqr" and selector.endswith("_bestcal"):
                        old_name = {"pivotal_bestcal": "shared", "F_bestcal": "F_auto",
                                    "representatives_bestcal": "representatives_auto"}[selector]
                        with np.load(BEST_REFERENCE/f"{panel.name}-1-1129.npz") as best:
                            np.testing.assert_array_equal(columns, best[old_name+"_cols"])
                            np.testing.assert_array_equal(pred, best[old_name+"_pred"])
                    replays.append(dict(dataset=panel.name, selector=selector, solver=solver,
                                        width=len(columns), score=score,
                                        support_sha256=digest(columns), prediction_sha256=digest(pred)))
                comparison.append(dict(dataset=panel.name, selector=selector, width=len(columns),
                                       lsqr_score=quality[selector+"_lsqr"],
                                       cholesky_score=quality[selector+"_cholesky"],
                                       score_delta=quality[selector+"_cholesky"]-quality[selector+"_lsqr"],
                                       changed_predictions=int(np.count_nonzero(cache[selector+"_lsqr_pred"] != cache[selector+"_cholesky_pred"])),
                                       test_rows=len(panel.gold)))
        archive = output/f"{panel.name}-1-1129.npz"
        if verify:
            with np.load(archive) as prior:
                assert set(prior.files) == set(cache)
                for key, value in cache.items():
                    np.testing.assert_array_equal(value, prior[key], err_msg=key)
            continue
        np.savez_compressed(archive, **cache)
        schedule = np.random.default_rng([1129, pass_id, p]).permutation(len(ARMS))
        for repeat in range(5):
            for position, index in enumerate(np.roll(schedule, repeat)):
                selector, solver = ARMS[index]
                key = selector+"_"+solver
                start = perf_counter_ns()
                columns = select(selector, panel)
                selected = perf_counter_ns()
                pred, decision, components = fit(panel, columns, solver)
                end = perf_counter_ns()
                np.testing.assert_array_equal(columns, cache[key+"_columns"])
                np.testing.assert_array_equal(pred, cache[key+"_pred"])
                np.testing.assert_array_equal(decision, cache[key+"_decision"])
                rows.append(dict(dataset=panel.name, fold=1, seed=panel.seed, selector=selector,
                                 solver=solver, repeat=repeat, position=position, width=len(columns),
                                 p=p, score=quality[key], select_ms=(selected-start)/1e6,
                                 total_ms=(end-start)/1e6, **components))
        inputs.append(dict(dataset=panel.name, n_train=len(panel.X), p=p, n_test=len(panel.Xt),
                           X=digest(panel.X), y=digest(panel.y), Xt=digest(panel.Xt), gold=digest(panel.gold)))
        print(f"Finished solver cost diagnosis pass {pass_id}: {panel.name}", flush=True)
    assert state == protocol(pass_id)
    if verify:
        pd.DataFrame(replays).to_csv(output/"fresh-process-audit.csv", index=False)
        return
    raw = pd.DataFrame(rows)
    raw.to_csv(output/"trials.csv", index=False)
    cases = raw.groupby(["dataset", "selector", "solver"], as_index=False).agg(
        width=("width", "first"), p=("p", "first"), score=("score", "first"),
        select_ms=("select_ms", "median"), fit_ms=("fit_ms", "median"),
        predict_ms=("predict_ms", "median"), slice_ms=("slice_ms", "median"), total_ms=("total_ms", "median"))
    cases.to_csv(output/"cases.csv", index=False)
    pd.DataFrame(inputs).to_csv(output/"inputs.csv", index=False)
    pd.DataFrame(comparison).to_csv(output/"solver-score-differences.csv", index=False)
    print(cases.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass-id", type=int, choices=(1, 2), required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.pass_id, args.verify)
