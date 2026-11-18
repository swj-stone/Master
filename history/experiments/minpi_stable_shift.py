"""Evaluate minPI on frozen temporal, composition, and subject holdouts."""
import argparse
import hashlib
import json
from itertools import combinations
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import f_classif, f_regression
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.minpi_har_confirmation import full_fit_pfi as subject_pfi
from experiments.minpi_iteration_benchmark import full_fit_pfi
from experiments.minpi_study import CONFIG, ROOT, file_hash, train
from experiments.minpi_study_new_cohorts import DIGESTS, batch_scores, load_cohort
from experiments.minpi_study_subjects import DIGEST, load_har
from minppi_mofs.environment_minpi import InsufficientEnvironmentsError, environment_top_from_training
from minppi_mofs.local_refit_fast import local_refit_fast_top_from_training
from minppi_mofs.stable_minpi import StableMinPISelector, _calibrate_supports, _partition


SEEDS = (1129, 2027, 3469)
SOURCES = ("gas", "supercon", "har")
CONTROLS = ("full", "environment50", "local50", "grouped50", "f50", "pfi50", "calibrated_f",
            "f_matched", "pfi_matched", "random_matched")


def get_shift(source):
    if source == "har":
        part, _ = load_har(False)
        X, y, groups = part["train"]
        Xt, truth, units = part["test"]
        task = "classification"
    else:
        X, y, groups, fit, held, task = load_cohort(source)
        Xt, truth, units = X[held], y[held], groups[held]
        X, y, groups = X[fit], y[fit], groups[fit]
    assert not np.intersect1d(groups, units).size
    if source == "gas":
        assert groups.max() < units.min()
    return X, y, groups, Xt, truth, units, task


def digest(vector):
    return hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def stable_fit(X, y, groups, task, source, seed):
    estimator = StableMinPISelector(task=task, random_state=seed)
    estimator.fit(X, y, groups=groups, ordered_groups=source == "gas")
    if estimator.diagnostics_["validation_rows"]:
        discovered = np.asarray(estimator.diagnostics_["discovery_rows"])
        assessments = [np.asarray(v) for v in estimator.diagnostics_["validation_rows"]]
        for held in assessments:
            assert not np.intersect1d(groups[discovered], groups[held]).size
            if source == "gas":
                assert groups[discovered].max() < groups[held].min()
    return estimator.get_support(indices=True), estimator.diagnostics_


def baseline(name, X, y, groups, task, source, seed, count):
    half = ceil(X.shape[1] / 2)
    if name == "full" or (name.endswith("_matched") and count == X.shape[1]):
        return np.arange(X.shape[1])
    if name == "local50":
        return local_refit_fast_top_from_training(X, y, half, task=task, random_state=seed)[0]
    if name == "environment50":
        return environment_top_from_training(X, y, half, task=task, random_state=seed,
                                             groups=groups, ordered_groups=source == "gas")[0]
    if name == "grouped50":
        return choose("grouped", X, y, task, seed, half)
    if name == "calibrated_f":
        try:
            discovery, held = _partition(y, groups, source == "gas", task, seed)
            discovery_x, discovery_y = X[discovery], y[discovery]
            # Compute F statistics once per calibration, selecting prefixes of the same order.
            stat = (f_classif if task == "classification" else f_regression)(discovery_x, discovery_y)[0]
            order = np.argsort(-np.nan_to_num(stat, nan=-np.inf), kind="stable")
            widths = {ceil(X.shape[1] * f) for f in (.1, .5, .75)}
            supports = {w: np.sort(order[:w]) for w in widths}
            return _calibrate_supports(X, y, discovery, held, supports, task, seed, .02, rule="F")[0]
        except InsufficientEnvironmentsError:
            return np.arange(X.shape[1])
    if name in ("f50", "f_matched", "random_matched"):
        method = "random" if name == "random_matched" else "f"
        return choose(method, X, y, task, seed, half if name == "f50" else count)
    if source == "har":
        order, _ = subject_pfi(X, y, groups, seed)
    else:
        order = full_fit_pfi(X, y, task, seed, groups)
    return np.sort(order[:half if name == "pfi50" else count])


def evaluate_source(source, output, verify, rows, unit_rows):
    X, y, groups, Xt, truth, units, task = get_shift(source)
    arrays = {"truth": truth, "units": units}
    decisions = {}
    for seed in SEEDS:
        methods = [] if verify else list(np.random.default_rng(seed).permutation(CONTROLS))
        for method in ("stable", *methods):
            start = perf_counter_ns()
            if method == "stable":
                support, details = stable_fit(X, y, groups, task, source, seed)
                decisions[str(seed)] = details
                width = len(support)
            else:
                support = baseline(method, X, y, groups, task, source, seed, width)
            selected = perf_counter_ns()
            if len(support) == X.shape[1]:
                A, B = X, Xt
            else:
                A, B = X[:, support], Xt[:, support]
            pred, _ = train("ridge", A, y, B, task, seed)
            end = perf_counter_ns()
            assert len(support) and np.array_equal(np.unique(support), support)
            assert support.min() >= 0 and support.max() < X.shape[1]
            key = f"{method}-{seed}"
            arrays[key + "-cols"] = support
            arrays[key + "-pred"] = pred
            if task == "regression":
                score = r2_score(truth, pred)
            else:
                scores = batch_scores(truth, pred, units)
                score = np.mean(list(scores.values()))
                unit_rows.extend(dict(source=source, method=method, seed=seed, unit=unit,
                                      score=value) for unit, value in scores.items())
            rule = details["rule"] if method == "stable" else method
            rows.append(dict(source=source, method=method, rule=rule, seed=seed, width=len(support),
                             inputs=X.shape[1], score=score, select_ms=(selected-start)/1e6,
                             model_ms=(end-selected)/1e6, total_ms=(end-start)/1e6,
                             support_sha256=digest(support), prediction_sha256=digest(pred)))
        print(f"{source} seed={seed} width={width}/{X.shape[1]}", flush=True)
    path = output / f"{source}-predictions.npz"
    decisions_path = output / f"{source}-decisions.json"
    decisions_text = json.dumps(decisions, default=serializable, sort_keys=True, indent=2)
    if verify:
        with np.load(path) as old:
            for name, value in arrays.items():
                np.testing.assert_array_equal(old[name], value)
            trials = pd.read_csv(output / "trials.csv")
            for trial in trials[trials.source == source].itertuples():
                key = f"{trial.method}-{trial.seed}"
                support, pred = old[key + "-cols"], old[key + "-pred"]
                assert trial.width == len(support) and trial.prediction_sha256 == digest(pred)
                assert trial.support_sha256 == digest(support)
                if task == "regression":
                    score = 1 - np.square(truth - pred).sum() / np.square(truth - truth.mean()).sum()
                else:
                    recalls = []
                    for group in np.unique(units):
                        mask = units == group
                        recalls.append(np.mean([np.mean(pred[mask & (truth == c)] == c)
                                                for c in np.unique(truth[mask])]))
                    score = np.mean(recalls)
                np.testing.assert_allclose(score, trial.score, atol=1e-12, rtol=0)
        if decisions_path.read_text(encoding="utf-8") != decisions_text:
            raise AssertionError(f"{source} training decisions differed in the second process")
    else:
        np.savez_compressed(path, **arrays)
        decisions_path.write_text(decisions_text, encoding="utf-8")


def seed_sensitivity(output):
    """Rerun only the frozen selector at ten additional seeds; exclude timing claims."""
    manifest = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in manifest["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError(f"Source changed since paired benchmark: {name}")
    records, overlaps = [], []
    for source in SOURCES:
        X, y, groups, Xt, truth, units, task = get_shift(source)
        with np.load(output / f"{source}-predictions.npz") as saved:
            full = saved[f"full-{SEEDS[0]}-pred"]
        def score(pred):
            if task == "regression":
                return r2_score(truth, pred)
            return np.mean(list(batch_scores(truth, pred, units).values()))
        full_score = score(full)
        supports = {name: {} for name in ("stable", "environment50", "local50")}
        for seed in range(10):
            for name, chosen in supports.items():
                if name == "stable":
                    support, details = stable_fit(X, y, groups, task, source, seed)
                    rule = details["rule"]
                else:
                    support = baseline(name, X, y, groups, task, source, seed, ceil(X.shape[1]/2))
                    rule = name
                chosen[seed] = support
                if len(support) == X.shape[1]:
                    pred = full
                else:
                    pred, _ = train("ridge", X[:, support], y, Xt[:, support], task, seed)
                value = score(pred)
                records.append(dict(source=source, method=name, seed=seed, rule=rule, width=len(support),
                                    score=value, delta_full=value-full_score, pass_tolerance=value>=full_score-.02,
                                    support_sha256=digest(support), prediction_sha256=digest(pred)))
        for method, chosen in supports.items():
            for left, right in combinations(chosen, 2):
                a, b = chosen[left], chosen[right]
                common = len(np.intersect1d(a,b))
                overlaps.append(dict(source=source, method=method, left=left, right=right,
                                     jaccard=common / (len(a)+len(b)-common)))
    pd.DataFrame(records).to_csv(output / "additional-seeds.csv", index=False)
    pd.DataFrame(overlaps).to_csv(output / "additional-seed-overlap.csv", index=False)


def run(output, verify):
    if not verify:
        output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), CONFIG, *(ROOT / "experiments" / name for name in (
        "minpi_study.py", "minpi_study_subjects.py", "minpi_study_new_cohorts.py",
        "minpi_har_confirmation.py", "minpi_iteration_benchmark.py", "compare_predictive_minpi.py")),
        *(ROOT / "src/minppi_mofs" / name for name in (
            "stable_minpi.py", "environment_minpi.py", "local_refit_fast.py", "grouped_minpi.py"))]
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in paths}
    manifest = dict(sources=SOURCES, primary_seed=SEEDS[0], seed_sensitivity=SEEDS, threads=1,
                    numpy=np.__version__, sklearn=sklearn.__version__, data_sha256={**DIGESTS, "har":DIGEST},
                    source_sha256=hashes, quality_tolerance=.02,
                    splits="Gas fit 1-7 and test 8-10. Supercon grouped holdout from seed 61429. HAR official split (21/9 distinct people).",
                    scoring="Gas and HAR equal-weight held group BA, materials pooled held R2",
                    timing="Fresh selector (including adaptive calibration) plus slicing and full outer ridge train/predict. Once per selector seed. Medians are descriptive over three seeds.",
                    controls="Fixed new environment minPI, local refit, grouped, F and full-fit training-held PFI at half width. Calibrated F uses identical discovery and calibration rows, widths and tolerance. F, random, full-fit PFI at actual adaptive width. Full width matched controls return identity.",
                    reproduction="Repeat stable from raw input in a separate Python process for all 9 source/seed cases, require exact support/prediction and diagnostics equality.")
    manifest_file = output / "protocol.json"
    if verify:
        if json.loads(manifest_file.read_text(encoding="utf-8")) != json.loads(json.dumps(manifest)):
            raise RuntimeError("Verification configuration or code differs")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, units = [], []
    for source in SOURCES:
        evaluate_source(source, output, verify, rows, units)
    frame = pd.DataFrame(rows)
    if verify:
        frame[["source","method","seed","width","score","support_sha256","prediction_sha256"]].to_csv(
            output / "fresh-process-audit.csv", index=False)
        return
    frame.to_csv(output / "trials.csv", index=False)
    pd.DataFrame(units).to_csv(output / "unit_scores.csv", index=False)
    summary = frame.groupby(["source","method"], as_index=False).agg(
        score=("score","mean"), min_seed_score=("score","min"), max_seed_score=("score","max"),
        min_width=("width","min"), max_width=("width","max"), input_width=("inputs","first"),
        select_ms=("select_ms","median"), model_ms=("model_ms","median"), total_ms=("total_ms","median"))
    summary.to_csv(output / "summary.csv", index=False)
    overlaps = []
    for source in SOURCES:
        with np.load(output / f"{source}-predictions.npz") as saved:
            for method in ("stable", "environment50", "local50"):
                for left, right in combinations(SEEDS, 2):
                    a, b = (saved[f"{method}-{seed}-cols"] for seed in (left, right))
                    common = len(np.intersect1d(a, b))
                    overlaps.append(dict(source=source, method=method, left=left, right=right,
                                         width_left=len(a), width_right=len(b),
                                         jaccard=common/(len(a)+len(b)-common)))
    pd.DataFrame(overlaps).to_csv(output / "seed_overlap.csv", index=False)
    pairs = frame[frame.method == "stable"].merge(frame[frame.method == "full"], on=["source","seed"],
                                                  suffixes=("_stable","_full"))
    pairs["delta_vs_full"] = pairs.score_stable - pairs.score_full
    pairs["quality_pass"] = pairs.delta_vs_full >= -.02
    pairs["cold_speedup"] = pairs.total_ms_full / pairs.total_ms_stable
    pairs["break_even_reuses"] = [
        ceil(row.select_ms_stable/(row.total_ms_full - row.model_ms_stable))
        if row.width_stable < row.inputs_stable and row.total_ms_full > row.model_ms_stable else None
        for row in pairs.itertuples()]
    pairs.to_csv(output / "paired.csv", index=False)
    print(summary.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.sensitivity:
            seed_sensitivity(args.output)
        else:
            run(args.output, args.verify)
