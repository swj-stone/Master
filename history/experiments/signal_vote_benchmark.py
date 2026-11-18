"""Replay adaptive minimum-over-context votes and width-matched controls on frozen outer splits."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_benchmark as original
from experiments import adaptive_minpi_transfer as transfer
from experiments import budget_minpi_deployments as deployment
from experiments import fast_minpi_five_source as five
from experiments import moment_minpi_benchmark as moment
from experiments.compare_predictive_minpi import choose
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction, write_json
from minppi_mofs.native_pivotal_minpi import library_path
from minppi_mofs.signal_vote_minpi import signal_vote_top_from_training
from minppi_mofs.stable_minpi import _impute_validation
from minppi_mofs.stable_native_minpi import stable_native_top_from_training


METHODS = ("signal_auto", "no_peer", "random_signal", "F_signal", "stable_auto", "full")
MATCHED = ("signal_auto", "no_peer", "random_signal", "F_signal")
CASE = ["dataset", "fold", "seed"]
REFERENCE = {
    "original": ROOT / "results/minpi-study/stable-native-69-20261117a",
    "transfer": ROOT / "results/minpi-study/stable-native-transfer-20261117a",
}
N_CASES = {"original": 69, "transfer": 57}
REPEATS = {"original": 7, "transfer": 5}
N_RANDOM = 20


def protocol(suite):
    old = REFERENCE[suite]
    inherited = moment.protocol(7) if suite == "original" else transfer.protocol()
    files = [Path(__file__), library_path(), ROOT/"src/minppi_mofs/native_pivotal_kernel.c"]
    files.extend(ROOT/"src/minppi_mofs"/(name+".py") for name in (
        "native_pivotal_minpi", "stable_native_minpi", "signal_vote_minpi", "sketch_minpi", "utility_minpi"))
    files.extend(ROOT/"experiments"/(name+".py") for name in (
        "adaptive_minpi_benchmark", "adaptive_minpi_transfer", "moment_minpi_benchmark",
        "fast_minpi_five_source", "compare_predictive_minpi", "budget_minpi_deployments"))
    parents = inherited.get("source_sha256", inherited.get("hashes")).copy()
    parents.update({p.relative_to(ROOT).as_posix(): file_hash(p) for p in files})
    return dict(panel=suite, repeats=REPEATS[suite], threads=1, methods=list(METHODS),
                expected=N_CASES[suite], source_sha256=parents,
                historical_protocol=file_hash(old/"protocol.json"), input_archives={
                    p.name: file_hash(p) for p in sorted(old.glob("*.npz"))},
                versions={k: inherited[k] for k in ("numpy","sklearn","python","platform")},
                success="width < p, test score - full score >= -0.02, median total elapsed < full median.",
                protocol="Preserve archived outer splits, tasks, missing-value processing, "
                         "downstream ridge, original scoring units and test data. Warm each arm once, "
                         "measure all arms in seeded rotated order. Include fresh selection, width "
                         "choice, slicing, preprocessing, training and predicting in every timed call. "
                         "Random, F and no_peer share the actual vote width; random/F get it for free. "
                         "no_peer ranks by the empty-context scores at that width and computes the "
                         "full vote engine, so its timing is not a stand-alone marginal optimization. "
                         "All scores, diagnostics, assertions, storage and extra random draws are untimed.",
                random_sensitivity="Original seeded permutation plus 19 independent extra draws with "
                                   "SeedSequence([891703, selector_seed, draw_id]), draw_id 1..19. "
                                   "Keep downstream model seed fixed. Run separately from clocked jobs.")


def panels(suite):
    return moment.panels() if suite == "original" else transfer.panels()


def select(method, panel, width, ordered):
    X, y = panel.X, panel.y
    if method == "full":
        return np.arange(X.shape[1]), {}
    if method in ("signal_auto", "no_peer", "stable_auto"):
        function = stable_native_top_from_training if method == "stable_auto" else signal_vote_top_from_training
        cols, info = function(X, y, task=panel.task, random_state=panel.seed, groups=panel.groups,
                              ordered_groups=ordered)
        if method == "no_peer":
            assert len(cols) == width
            zero_scores = np.asarray(info["zero_peer_scores"])
            assert zero_scores.shape == (X.shape[1],) and np.isfinite(zero_scores).all()
            cols = np.sort(np.argsort(-zero_scores, kind="stable")[:width])
        return cols, info
    if method == "F_signal" and panel.name in deployment.ADDITIONAL:
        X, _ = _impute_validation(X, X[:0])
    return choose(method.split("_")[0].lower(), X, y, panel.task, panel.seed, width), {}


def fit(panel, cols, suite):
    if suite == "original":
        return deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
    return prediction(panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed), None


def score(panel, pred, decision, suite):
    if suite == "original":
        return deployment.score_predictions(panel.name, panel.gold, pred, panel.units, panel.task, decision)
    return metric(panel.truth, pred, panel.task), {}, np.nan


def case_id(panel):
    return dict(dataset=panel.name, fold=panel.fold, seed=panel.seed)


def run_case(panel, suite, output, verify, index):
    key = case_id(panel)
    name = f"{panel.name}-{panel.fold}-{panel.seed}.npz"
    ordered = panel.name in ("gas", "secom") if suite == "original" else panel.ordered
    truth = panel.gold if suite == "original" else panel.truth
    arrays = {"input_hashes": np.asarray([digest(a) for a in (panel.X,panel.y,panel.Xt,truth)])}
    metadata = dict(**key, task=panel.task, ordered=ordered, n_train=len(panel.y), n_test=len(truth),
                    p=panel.X.shape[1], family=getattr(panel,"family","original"),
                    mechanism=getattr(panel,"mechanism",panel.name))
    scores, units, audit, trials = {}, [], [], []
    width = None
    for method in METHODS:
        cols, info = select(method, panel, width, ordered)
        five.check_columns(cols, panel.X.shape[1])
        if method == "signal_auto":
            width = len(cols)
            if info.get("folds"):
                deployment.check_folds(info, panel.groups, ordered, len(panel.y))
            metadata.update({k:v for k,v in info.items() if np.isscalar(v)})
            arrays["zero_peer_scores"] = np.asarray(info["zero_peer_scores"])
        if method in MATCHED:
            assert len(cols) == width
        pred, decision = fit(panel, cols, suite)
        value, by_unit, ap = score(panel, pred, decision, suite)
        scores[method] = value, ap
        arrays[method+"_cols"], arrays[method+"_pred"] = cols, pred
        if panel.name == "secom":
            arrays[method+"_decision"] = decision
        units.extend(dict(**key, method=method, unit=u, score=s) for u,s in by_unit.items())
        audit.append(dict(**key, method=method, width=len(cols), score=value,
                          support_sha256=digest(cols), prediction_sha256=digest(pred)))
    with np.load(REFERENCE[suite]/name) as previous:
        for method in ("stable_auto", "full"):
            for attr in ("cols", "pred"):
                field = method+"_"+attr
                np.testing.assert_array_equal(arrays[field], previous[field], err_msg=name+" "+field)
        if suite == "transfer":
            np.testing.assert_array_equal(arrays["input_hashes"], previous["input_hashes"])
    if verify:
        with np.load(output/name) as saved:
            assert set(saved.files) == set(arrays)
            for k, value in arrays.items():
                np.testing.assert_array_equal(value, saved[k], err_msg=name+" "+k)
        return trials, audit, units, metadata
    order = np.random.default_rng([60761, index, panel.seed]).permutation(METHODS)
    for repeat in range(REPEATS[suite]):
        for pos, method in enumerate(np.roll(order,repeat)):
            start = perf_counter_ns()
            cols, _ = select(method, panel, width, ordered)
            chosen = perf_counter_ns()
            pred, decision = fit(panel, cols, suite)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, arrays[method+"_cols"])
            np.testing.assert_array_equal(pred, arrays[method+"_pred"])
            if panel.name == "secom":
                np.testing.assert_array_equal(decision, arrays[method+"_decision"])
            value, ap = scores[method]
            trials.append(dict(**key, method=method, repeat=repeat, position=pos, width=len(cols),
                               p=panel.X.shape[1], score=value, ap=ap, select_ms=(chosen-start)/1e6,
                               refit_ms=(end-chosen)/1e6, total_ms=(end-start)/1e6))
    np.savez_compressed(output/name, **arrays)
    return trials, audit, units, metadata


def summarize(output, trials, metadata):
    cases = five.summarize_timings(output, trials)
    vote = cases.loc[cases.method=="signal_auto"].merge(metadata,on=CASE,validate="one_to_one",
                                                      suffixes=("","_meta"))
    assert vote.p.eq(vote.p_meta).all()
    for method in ("random_signal","F_signal","no_peer","stable_auto"):
        keep = CASE + ["score","width","joint","ratio"]
        control = cases.loc[cases.method==method,keep].rename(columns={k:k+"_"+method for k in keep[3:]})
        vote = vote.merge(control,on=CASE,validate="one_to_one")
        if method in MATCHED:
            np.testing.assert_array_equal(vote.width,vote["width_"+method])
        diff = vote.score-vote["score_"+method]
        vote["delta_"+method] = diff
        vote["outcome_"+method] = np.select([diff>1e-12, diff < -1e-12],["win","loss"],default="tie")
    vote.to_csv(output/"vote-comparisons.csv",index=False)
    return cases


def random_sensitivity(output, suite):
    """Extra same-width supports with fixed downstream seed, separate from wall-clock timing."""
    cases = pd.read_csv(output/"cases.csv").set_index(CASE+["method"])
    rows = []
    for panel in panels(suite):
        key = case_id(panel)
        ix = tuple(key.values())
        width = int(cases.loc[(*ix,"signal_auto"),"width"])
        vote = cases.loc[(*ix,"signal_auto"),"score"]
        for draw in range(N_RANDOM):
            if draw == 0:
                cols = choose("random",panel.X,panel.y,panel.task,panel.seed,width)
            else:
                rng = np.random.default_rng([891703,panel.seed,draw])
                cols = np.sort(rng.permutation(panel.X.shape[1])[:width])
            pred, dec = fit(panel,cols,suite)
            value, _, _ = score(panel,pred,dec,suite)
            if draw == 0:
                np.testing.assert_allclose(value,cases.loc[(*ix,"random_signal"),"score"],atol=1e-12,rtol=0)
            rows.append(dict(**key,draw=draw,width=width,random_score=value,vote_score=vote,
                             vote_minus_random=vote-value,support_sha256=digest(cols),prediction_sha256=digest(pred)))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"random-draws.csv",index=False)
    frame["win"] = frame.vote_minus_random > 1e-12
    frame["tie"] = frame.vote_minus_random.abs() <= 1e-12
    result = frame.groupby(CASE).agg(vote_score=("vote_score","first"),random_median=("random_score","median"),
                                     avg_delta=("vote_minus_random","mean"),wins=("win","sum"),ties=("tie","sum"),
                                     draws=("draw","size"))
    assert len(result) == N_CASES[suite] and result.draws.eq(N_RANDOM).all()
    result["median_delta"] = result.vote_score-result.random_median
    result.to_csv(output/"random-sensitivity.csv")


def run(output, suite, verify, extra_random):
    state = protocol(suite)
    manifest = output/"protocol.json"
    if verify or extra_random:
        assert json.loads(manifest.read_text(encoding="utf-8")) == state
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(manifest,state)
    if extra_random:
        random_sensitivity(output,suite)
        return
    trials, audit, units, metadata, inputs = [], [], [], [], []
    for index, panel in enumerate(panels(suite)):
        timed, checked, per_unit, meta = run_case(panel,suite,output,verify,index)
        trials.extend(timed)
        audit.extend(checked)
        units.extend(per_unit)
        metadata.append(meta)
        if suite == "original":
            inputs.append(original.input_hashes(panel))
        print("verified" if verify else "measured",index+1,panel.name,panel.fold,panel.seed,flush=True)
    assert len(metadata) == N_CASES[suite] and state == protocol(suite)
    if inputs:
        frame = pd.DataFrame(inputs)
        pd.testing.assert_frame_equal(frame,pd.read_csv(REFERENCE[suite]/"inputs.csv"),check_dtype=False)
        if not verify:
            frame.to_csv(output/"inputs.csv",index=False)
    metadata = pd.DataFrame(metadata)
    assert len(audit) == len(METHODS)*N_CASES[suite]
    if verify:
        pd.testing.assert_frame_equal(metadata,pd.read_csv(output/"panels.csv",keep_default_na=False),
                                     check_dtype=False)
        pd.DataFrame(audit).to_csv(output/"fresh-process-audit.csv",index=False)
    else:
        assert len(trials) == len(audit)*REPEATS[suite]
        metadata.to_csv(output/"panels.csv",index=False)
        pd.DataFrame(units).to_csv(output/"unit-scores.csv",index=False)
        summarize(output,trials,metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=tuple(N_CASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify",action="store_true")
    mode.add_argument("--random-sensitivity",action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.panel,args.verify,args.random_sensitivity)
