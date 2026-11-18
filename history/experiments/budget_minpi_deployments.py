"""Replay the 24 archived deployment configurations under strict compressed-width timing."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.fast_minpi_five_source import CANDIDATES, METHODS, check_columns, summarize_timings
from experiments.minpi_backward_confirmation import evaluate
from experiments.minpi_consensus_confirmation import SOURCES as CONFIRMATION, benchmarks
from experiments.minpi_stable_shift import SEEDS, digest
from experiments.minpi_study import CONFIG, ROOT, file_hash, train
from experiments.minpi_study_chain import dataset, split
from experiments.stable_minpi_stress import SPLIT_SEED
from minppi_mofs.consensus_minpi import ConsensusMinPISelector
from minppi_mofs.pair_vote_minpi import pair_vote_top_from_training
from minppi_mofs.sketch_minpi import reduced_width, sketch_top_from_training
from minppi_mofs.stable_minpi import _impute_validation


RUNS = ROOT / "results/minpi-study"
CONFIRMATION_RUN = RUNS / "consensus-confirmation-20261117a"
UNSEEN_RUN = RUNS / "consensus-unseen-20261117a"
ADDITIONAL = ("secom", "cnae")
SOURCES = (*CONFIRMATION, *ADDITIONAL)
REPEATED_OUTER = ("gas", "supercon", "har", "mhealth", "secom")


def configurations(name):
    """Use the old loaders, column order and outer split, including the two-device MHEALTH test."""
    if name in CONFIRMATION:
        for seed, (X, y, groups, Xt, yt, units, task) in benchmarks(name):
            if groups is not None:
                assert not np.intersect1d(groups, units).size
            yield seed, X, y, groups, Xt, yt, units, task, {}
        return
    X, y, task, order, _, times = dataset(name)
    for seed in SEEDS:
        fit, test = split(name, y, task, times, seed)
        groups = None
        if times is not None:
            assert times[fit].max() < times[test].min()
            groups = (times[fit] - times[fit].min()) // np.timedelta64(7, "D")
        yield seed, X[fit], y[fit], groups, X[test], y[test], test, task, dict(
            tr=fit, te=test, column_order=order)


def prior_run(name):
    return UNSEEN_RUN if name in ADDITIONAL else CONFIRMATION_RUN


def prior_arrays(name, seed, truth, units, extra):
    with np.load(prior_run(name) / f"{name}-{seed}.npz") as archive:
        np.testing.assert_array_equal(archive["truth"], truth)
        if name in ADDITIONAL:
            for key in ("tr", "te"):
                np.testing.assert_array_equal(archive[key], extra[key])
            return archive["full_pred"], archive["auto_pred"], archive["auto_cols"]
        np.testing.assert_array_equal(archive["units"], units)
        return archive["full-pred"], archive["auto-pred"], archive["auto-support"]


def historical_controls():
    rows = []
    for parent in (CONFIRMATION_RUN, UNSEEN_RUN):
        old = pd.read_csv(parent / "cases.csv")
        keys = ["source", "seed"]
        auto = old[old.method == "auto"].merge(
            old[old.method == "full"][[*keys, "score", "total_ms"]].rename(
                columns={"score": "full_score", "total_ms": "full_total_ms"}),
            on=keys, validate="one_to_one")
        auto["reduced"] = auto.width < auto.p
        auto["within"] = auto.score - auto.full_score >= -.02 - 1e-12
        auto["historical_joint"] = auto.reduced & auto.within & (auto.total_ms < auto.full_total_ms)
        auto["method"] = "consensus_policy"
        auto["archive"] = parent.name
        rows.append(auto)
    return pd.concat(rows, ignore_index=True).rename(columns={"source": "dataset"})


def score_predictions(name, truth, pred, units, task, decision):
    score, per_unit = evaluate(name, truth, pred, units, task)
    ap = average_precision_score(truth, decision) if name == "secom" else np.nan
    return score, per_unit, ap


def check_folds(info, groups, ordered, n):
    for pair in info["folds"]:
        fit, held = pair["fit_rows"], pair["held_rows"]
        assert min(fit.min(), held.min()) >= 0 and max(fit.max(), held.max()) < n
        assert not np.intersect1d(fit, held).size
        if groups is not None:
            assert not np.intersect1d(groups[fit], groups[held]).size
            if ordered:
                assert groups[fit].max() < groups[held].min()


def screen(method, name, X, y, task, groups, seed):
    ordered = name in ("gas", "secom")
    params = dict(task=task, random_state=seed)
    env = dict(groups=groups, ordered_groups=ordered)
    p = X.shape[1]
    if method == "full":
        return np.arange(p), {}
    if method == "consensus_policy":
        model = ConsensusMinPISelector(**params).fit(X, y, **env)
        return model.get_support(indices=True), {}
    width = reduced_width(p)
    if method in CANDIDATES:
        chooser = sketch_top_from_training if method == "sketch" else pair_vote_top_from_training
        return chooser(X, y, width, **params, **env)
    kind, suffix = method.split("_", maxsplit=1)
    assert suffix == "matched"
    if method == "F_matched" and name in ADDITIONAL:
        X, _ = _impute_validation(X, X[:0])
    return choose(kind.lower(), X, y, task, seed, width), {}


def fit_predict(name, X, y, Xt, columns, task, seed):
    A, B = (X, Xt) if len(columns) == X.shape[1] else (X[:, columns], Xt[:, columns])
    if name in ADDITIONAL:
        A, B = _impute_validation(A, B)
    return train("ridge", A, y, B, task, seed)


def protocol(repeats):
    src = ROOT / "src/minppi_mofs"
    exp = ROOT / "experiments"
    files = [Path(__file__), CONFIG]
    files += [src / (name + ".py") for name in (
        "sketch_minpi", "pair_vote_minpi", "consensus_minpi", "stable_minpi", "environment_minpi",
        "masked_loss_minpi", "restoration_minpi", "utility_minpi")]
    files += [exp / (name + ".py") for name in (
        "fast_minpi_five_source", "minpi_study", "minpi_study_chain", "minpi_stable_shift",
        "minpi_backward_confirmation", "minpi_consensus_confirmation", "compare_predictive_minpi",
        "minpi_study_new_cohorts", "minpi_study_subjects", "consensus_minpi_unseen")]
    archived = [run / f for run in (CONFIRMATION_RUN, UNSEEN_RUN) for f in ("protocol.json", "cases.csv")]
    archived += [prior_run(name) / f"{name}-{seed}.npz" for name in SOURCES for seed in SEEDS]
    return dict(sources=list(SOURCES), seeds=list(SEEDS), methods=list(METHODS), repeats=repeats,
                source_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
                archived_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in archived},
                sklearn=sklearn.__version__, numpy=np.__version__, python=platform.python_version(),
                platform=platform.platform(), cpu=platform.processor(), threads=1, tolerance=.02,
                repeated_outer=list(REPEATED_OUTER),
                split="Reuse the six confirmation and two unseen archives. In repeated_outer, seeds vary "
                      "on one frozen outer split. Diabetes/ISOLET/CNAE have three distinct paired splits.",
                score="Official HAR subject and Gas batch mean balanced accuracy; MHEALTH mean of "
                      "per-person lower wrist/ankle BA; regression pooled R2; remaining classification "
                      "pooled BA. Report SECOM average precision as a secondary metric.",
                timing="One warm-up for every arm, then independently select, slice, apply training-only "
                       "imputation under the historical SECOM/CNAE protocol, refit standardized ridge "
                       "and predict. Randomized initial order rotated each round. Separate test scoring, "
                       "input loading, assertions and serialization from all clocks.",
                success="Strictly fewer columns, score - full score >= -0.02, ratio of median total times "
                        "< 1. Retain raw times and paired speed counts; secondary ratio thresholds .95/.90.")


def case(name, fold, data, repeats, verifying=None):
    seed, X, y, groups, Xt, truth, units, task, extra = data
    old_full, old_auto, old_columns = prior_arrays(name, seed, truth, units, extra)
    saved = dict(truth=truth, units=units, **extra)
    records, quality, units_rows, audits = [], {}, [], []
    timed_order = np.random.default_rng([SPLIT_SEED, fold, seed]).permutation(METHODS)
    methods = METHODS if verifying is None else CANDIDATES
    for method in methods:
        cols, info = screen(method, name, X, y, task, groups, seed)
        check_columns(cols, X.shape[1])
        if method not in ("full", "consensus_policy"):
            assert len(cols) == reduced_width(X.shape[1]) < X.shape[1]
        pred, decision = fit_predict(name, X, y, Xt, cols, task, seed)
        if method in CANDIDATES:
            check_folds(info, groups, name in ("gas", "secom"), len(y))
            saved[f"{method}_votes"] = info["scores"]
            for i, part in enumerate(info["folds"]):
                for key, value in part.items():
                    saved[f"{method}_fold{i}_{key}"] = np.asarray(value)
        saved[f"{method}_cols"], saved[f"{method}_pred"] = cols, pred
        if name == "secom":
            saved[f"{method}_decision"] = decision
        score, per_unit, ap = score_predictions(name, truth, pred, units, task, decision)
        quality[method] = score, ap
        units_rows.extend(dict(dataset=name, fold=fold, seed=seed, method=method, unit=u, score=s)
                          for u, s in per_unit.items())
        if method == "full":
            np.testing.assert_array_equal(pred, old_full)
        if method == "consensus_policy":
            np.testing.assert_array_equal(pred, old_auto)
            np.testing.assert_array_equal(cols, old_columns)
        if verifying is not None:
            audits.append(dict(dataset=name, fold=fold, seed=seed, method=method, width=len(cols),
                               score=score, support_sha256=digest(cols), prediction_sha256=digest(pred)))
    if verifying is not None:
        for key, value in saved.items():
            np.testing.assert_array_equal(value, verifying[key], err_msg=f"{name}-{seed}-{key}")
        return audits
    for repeat in range(repeats):
        for position, method in enumerate(np.roll(timed_order, repeat)):
            start = perf_counter_ns()
            cols, _ = screen(method, name, X, y, task, groups, seed)
            middle = perf_counter_ns()
            pred, decision = fit_predict(name, X, y, Xt, cols, task, seed)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, saved[f"{method}_cols"])
            np.testing.assert_array_equal(pred, saved[f"{method}_pred"])
            if name == "secom":
                np.testing.assert_array_equal(decision, saved[f"{method}_decision"])
            score, ap = quality[method]
            records.append(dict(dataset=name, fold=fold, seed=seed, method=method, repeat=repeat,
                                position=position, width=len(cols), p=X.shape[1], score=score, ap=ap,
                                select_ms=(middle-start)/1e6, refit_ms=(end-middle)/1e6,
                                total_ms=(end-start)/1e6))
    return records, saved, units_rows


def run(out, repeats, verify):
    current = protocol(repeats)
    manifest = out / "protocol.json"
    if verify:
        assert current == json.loads(manifest.read_text(encoding="utf-8"))
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps(current, indent=2), encoding="utf-8")
    trials, unit_scores, audited, input_hashes, stability = [], [], [], [], []
    for name in SOURCES:
        banks = {}
        for i, data in enumerate(configurations(name), 1):
            seed, X, y, groups, Xt, truth, units, task, _ = data
            fold = 1 if name in REPEATED_OUTER else i
            path = out / f"{name}-{seed}.npz"
            input_hashes.append(dict(dataset=name, fold=fold, seed=seed, n_train=len(X), n_test=len(Xt),
                                     p=X.shape[1], train_sha256=digest(X), y_sha256=digest(y),
                                     test_sha256=digest(Xt), truth_sha256=digest(truth),
                                     fit_groups_sha256=digest(groups) if groups is not None else "none",
                                     units_sha256=digest(units)))
            if verify:
                with np.load(path) as prev:
                    audited.extend(case(name, fold, data, repeats, verifying=prev))
            else:
                rows, saved, units_rows = case(name, fold, data, repeats)
                trials.extend(rows)
                unit_scores.extend(units_rows)
                banks[seed] = {m: saved[f"{m}_cols"] for m in CANDIDATES}
                np.savez_compressed(path, **saved)
            print(f"{'Verified' if verify else 'Measured'} {name} {seed}", flush=True)
        if not verify:
            for a, b in combinations(SEEDS, 2):
                for method in CANDIDATES:
                    ca, cb = banks[a][method], banks[b][method]
                    stability.append(dict(dataset=name, same_outer=name in REPEATED_OUTER, method=method,
                                          a=a, b=b, jaccard=len(np.intersect1d(ca, cb))/len(np.union1d(ca, cb))))
    inputs = pd.DataFrame(input_hashes)
    assert len(inputs) == 24 and not inputs.duplicated(["dataset", "seed"]).any()
    if verify:
        pd.testing.assert_frame_equal(inputs, pd.read_csv(out/"inputs.csv"), check_dtype=False)
        pd.DataFrame(audited).to_csv(out/"fresh-process-audit.csv", index=False)
        return
    assert current == protocol(repeats)
    inputs.to_csv(out/"inputs.csv", index=False)
    pd.DataFrame(unit_scores).to_csv(out/"unit_scores.csv", index=False)
    pd.DataFrame(stability).to_csv(out/"seed-overlap.csv", index=False)
    cases = summarize_timings(out, trials)
    raw = pd.DataFrame(trials)
    means = raw.groupby(["dataset", "fold", "seed", "method"], as_index=False).agg(
        mean_total_ms=("total_ms", "mean"), mean_select_ms=("select_ms", "mean"),
        mean_refit_ms=("refit_ms", "mean"))
    means.to_csv(out/"mean-timing.csv", index=False)
    secom = raw[raw.dataset == "secom"].drop_duplicates(["seed","method"])
    secom[["dataset","seed","method","score","ap"]].to_csv(out/"secom-ap.csv", index=False)
    old = historical_controls()
    old.to_csv(out/"historical-consensus.csv", index=False)
    matched = cases[cases.method == "consensus_policy"].merge(
        old[["dataset","seed","width","score","historical_joint"]], on=["dataset","seed"],
        suffixes=("", "_archived"), validate="one_to_one")
    np.testing.assert_array_equal(matched.width, matched.width_archived)
    np.testing.assert_allclose(matched.score, matched.score_archived, rtol=0, atol=1e-14)
    matched.to_csv(out/"consensus-rematch.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--verify", action="store_true")
    cli.add_argument("--repeats", type=int, default=7)
    args = cli.parse_args()
    if args.repeats < 3:
        cli.error("At least three repeated timings are required")
    with threadpool_limits(limits=1):
        run(args.output, args.repeats, args.verify)
