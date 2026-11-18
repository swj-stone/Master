"""Compare training-only adaptive votes and their width-matched controls on frozen panels."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import signal_vote_benchmark as prior
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from experiments.stable_minpi_stress import ROOT, digest, file_hash, write_json
from minppi_mofs.budget_vote_minpi import budget_vote_top_from_training
from minppi_mofs.ridge_vote_minpi import ridge_vote_top_from_training
from minppi_mofs.sketch_minpi import reduced_width


NEW = {"budget": budget_vote_top_from_training, "ridge": ridge_vote_top_from_training}
ARMS = ("budget", "ridge", "random_budget", "F_budget", "random_ridge", "F_ridge",
        "F50", "random50", "signal_auto", "full")
KEY = prior.CASE
PREVIOUS = {k: ROOT / "results/minpi-study" / ("signal-vote-" + v + "-20261117a")
            for k, v in (("original", "69"), ("transfer", "transfer"))}


def protocol(suite):
    archive = PREVIOUS[suite]
    old = json.loads((archive / "protocol.json").read_text())
    files = [Path(__file__), Path(prior.__file__)]
    files.extend(ROOT / "src/minppi_mofs" / (name+"_vote_minpi.py") for name in NEW)
    files.extend(ROOT / "src/minppi_mofs" / name for name in
                 ("masked_loss_minpi.py", "moment_minpi.py", "sketch_minpi.py", "utility_minpi.py"))
    return dict(previous_manifest_sha256=file_hash(archive/"protocol.json"),
                source_sha256={**old["source_sha256"],
                               **{f.relative_to(ROOT).as_posix(): file_hash(f) for f in files}},
                saved_input_outputs={f.name:file_hash(f) for f in sorted(archive.glob("*.npz"))},
                versions=old["versions"], panel=suite, arms=ARMS, repeats=prior.REPEATS[suite],
                n_cases=prior.N_CASES[suite], threads=1, random_draws=20,
                protocol="Use the preceding outer rows, columns, deployed estimator, preprocessing and scores. "
                         "Warm up all arms, then rotate freshly computed selection, column gathering, "
                         "training and prediction. Each random/F arm receives its candidate's width for free. "
                         "Zero-peer rankings and 19 additional seeded random supports are quality-only. "
                         "Strict reduction, score minus full >= -0.02 and median total wall time < full. "
                         "Report zero score loss, 5% and 10% time margins and disjoint early/late rounds.")


def select(arm, panel, widths, ordered):
    if arm in NEW:
        return NEW[arm](panel.X, panel.y, task=panel.task, random_state=panel.seed,
                        groups=panel.groups, ordered_groups=ordered)
    if arm in ("full","signal_auto"):
        return prior.select(arm, panel, None, ordered)
    if arm in ("F50","random50"):
        return prior.select(arm[:-2]+"_signal", panel, reduced_width(panel.X.shape[1],.5), ordered)
    base, candidate = arm.split("_")
    return prior.select(base + "_signal", panel, widths[candidate], ordered)


def evaluate(panel, suite, out, index, verify):
    key = prior.case_id(panel)
    name = f"{panel.name}-{panel.fold}-{panel.seed}.npz"
    ordered = panel.name in ("gas","secom") if suite == "original" else panel.ordered
    truth = panel.gold if suite=="original" else panel.truth
    arrays = {"input_hashes": np.array([digest(x) for x in (panel.X,panel.y,panel.Xt,truth)])}
    widths, scores, audit, trials, metadata = {}, {}, [], [], []
    for arm in ARMS:
        cols, info = select(arm, panel, widths, ordered)
        check_columns(cols, panel.X.shape[1])
        if arm in NEW:
            widths[arm] = len(cols)
            if info["folds"]:
                prior.deployment.check_folds(info, panel.groups, ordered, len(panel.y))
            scalar_info = {k:v for k,v in info.items() if np.isscalar(v)}
            metadata.append(dict(**key, candidate=arm, task=panel.task, n_train=len(panel.y),
                                 n_test=len(truth), p=panel.X.shape[1], width=len(cols),
                                 details=json.dumps(scalar_info,sort_keys=True,default=prior.original.json_value)))
            zero = np.asarray(info["zero_peer_scores"])
            assert zero.shape == (panel.X.shape[1],) and np.isfinite(zero).all()
            zero_cols = np.sort(np.argsort(-zero,kind="stable")[:len(cols)])
            zero_pred, zero_decision = prior.fit(panel, zero_cols, suite)
            value = prior.score(panel, zero_pred, zero_decision, suite)[0]
            arrays[arm+"_zero_cols"] = zero_cols
            arrays[arm+"_zero_pred"] = zero_pred
            arrays[arm+"_zero_score"] = np.asarray(value)
        elif arm in ("F50","random50"):
            assert len(cols) == reduced_width(panel.X.shape[1],.5)
        elif arm not in ("full","signal_auto"):
            assert len(cols) == widths[arm.split("_")[1]]
        pred, decision = prior.fit(panel, cols, suite)
        value, _, ap = prior.score(panel, pred, decision, suite)
        scores[arm] = value, ap
        arrays[arm+"_cols"], arrays[arm+"_pred"] = cols, pred
        if panel.name == "secom":
            arrays[arm+"_decision"] = decision
        audit.append(dict(**key,method=arm,width=len(cols),score=value,support_sha256=digest(cols),
                          prediction_sha256=digest(pred)))
    with np.load(PREVIOUS[suite]/name) as reference:
        for arm in ("signal_auto","full"):
            for field in ("cols","pred"):
                np.testing.assert_array_equal(arrays[arm+"_"+field], reference[arm+"_"+field])
        np.testing.assert_array_equal(arrays["input_hashes"],reference["input_hashes"])
    if verify:
        with np.load(out/name) as saved:
            assert set(arrays) == set(saved.files)
            for field, value in arrays.items():
                np.testing.assert_array_equal(value,saved[field],err_msg=f"{name} {field}")
        return trials, audit, metadata
    order = np.random.default_rng([113827,index,panel.seed]).permutation(ARMS).tolist()
    for repeat in range(prior.REPEATS[suite]):
        offset = repeat % len(ARMS)
        for position, arm in enumerate(order[offset:] + order[:offset]):
            start = perf_counter_ns()
            cols, _ = select(arm,panel,widths,ordered)
            selected = perf_counter_ns()
            pred, decision = prior.fit(panel,cols,suite)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols,arrays[arm+"_cols"])
            np.testing.assert_array_equal(pred,arrays[arm+"_pred"])
            if panel.name == "secom":
                np.testing.assert_array_equal(decision,arrays[arm+"_decision"])
            value, ap = scores[arm]
            trials.append(dict(**key, method=arm, repeat=repeat, position=position, width=len(cols),
                               p=panel.X.shape[1], score=value, ap=ap, select_ms=(selected-start)/1e6,
                               refit_ms=(end-selected)/1e6, total_ms=(end-start)/1e6))
    np.savez_compressed(out/name, **arrays)
    return trials, audit, metadata


def attribution(out):
    cases = pd.read_csv(out/"cases.csv").set_index(KEY+["method"]).sort_index()
    raw = pd.read_csv(out/"trials.csv")
    n = raw.repeat.max()+1
    records, relative = [], []
    for candidate in NEW:
        vote = cases.xs(candidate,level="method").copy()
        for arm in ["F_"+candidate,"random_"+candidate,"signal_auto"]:
            baseline = cases.xs(arm,level="method")
            if arm != "signal_auto":
                np.testing.assert_array_equal(vote.width, baseline.width)
            vote["delta_"+arm] = vote.score - baseline.score
        vote["kept_fraction"] = vote.width/vote.p
        vote["joint_no_loss"] = vote.joint & (vote.gap>=0)
        vote["both_windows"] = vote.joint
        for samples in [raw.repeat < n//2, raw.repeat >= n-n//2]:
            times = raw[samples].groupby(KEY+["method"]).total_ms.median().unstack()
            vote["both_windows"] &= times[candidate] < times.full
        for ix in vote.index:
            with np.load(out/("-".join(str(k) for k in ix)+".npz")) as trace:
                vote.loc[ix,"zero_score"] = trace[candidate+"_zero_score"].item()
        vote["candidate"] = candidate
        vote["delta_zero"] = vote.score-vote.zero_score
        records.append(vote.reset_index())
        for arm in ["full","random_"+candidate,"F_"+candidate,"signal_auto","F50","random50"]:
            baseline = cases.xs(arm,level="method")
            comparison = vote[["width","p","score","total_ms","joint"]].copy()
            comparison["candidate"] = candidate
            comparison["baseline"] = arm
            comparison["baseline_width"] = baseline.width
            comparison["baseline_score"] = baseline.score
            comparison["baseline_total_ms"] = baseline.total_ms
            comparison["score_gap"] = vote.score-baseline.score
            comparison["time_ratio"] = vote.total_ms/baseline.total_ms
            comparison["width_ratio"] = vote.width/baseline.width
            safe = vote.reduced & (comparison.score_gap >= -.02) & (comparison.time_ratio < 1)
            comparison["relative_joint"] = safe & (vote.width<=baseline.width)
            comparison["also_full_joint"] = vote.joint & comparison.relative_joint
            comparison["relative_joint_width_10pct"] = safe & (10*vote.width<=11*baseline.width)
            comparison["exact_quality_joint"] = comparison.relative_joint & (comparison.score_gap>=0.)
            relative.append(comparison.reset_index())
    pd.concat(records,ignore_index=True).to_csv(out/"attribution.csv",index=False)
    joined = pd.concat(relative,ignore_index=True)
    joined.to_csv(out/"baseline-comparisons.csv",index=False)
    joined.groupby(["candidate","baseline"]).agg(cases=("width","size"),success=("relative_joint","sum"),
        plus_full=("also_full_joint","sum"),width_10pct=("relative_joint_width_10pct","sum"),
        no_loss=("exact_quality_joint","sum"),median_score_gap=("score_gap","median"),
        median_time_ratio=("time_ratio","median"),median_width_ratio=("width_ratio","median")).to_csv(
        out/"baseline-summary.csv")
    for field in ("relative_joint","also_full_joint","relative_joint_width_10pct","exact_quality_joint"):
        flags = joined.pivot(index=KEY+["candidate"],columns="baseline",values=field)
        matched = flags.groupby(level="candidate").apply(
            lambda f: f[["F_"+f.name,"random_"+f.name,"full","signal_auto","F50","random50"]].all(axis=1))
        matched.rename(field).to_csv(out/(field+"-all-baselines.csv"))


def random_comparison(out,suite):
    records = []
    cases = pd.read_csv(out/"cases.csv").set_index(KEY+["method"])
    for panel in prior.panels(suite):
        key = prior.case_id(panel)
        ix = (panel.name,panel.fold,panel.seed)
        by_width = {}
        for candidate in NEW:
            vote = cases.loc[(*ix,candidate)]
            width = int(vote.width)
            if width not in by_width:
                draws = []
                for draw in range(prior.N_RANDOM):
                    seed = panel.seed if draw==0 else [891703,panel.seed,draw]
                    cols = np.sort(np.random.default_rng(seed).permutation(panel.X.shape[1])[:width])
                    pred, dec = prior.fit(panel,cols,suite)
                    value = prior.score(panel,pred,dec,suite)[0]
                    draws.append((value,digest(cols),digest(pred)))
                by_width[width] = draws
            for draw,(value,support,prediction) in enumerate(by_width[width]):
                if draw == 0:
                    np.testing.assert_allclose(value,cases.loc[(*ix,"random_"+candidate)].score,rtol=0,atol=1e-12)
                records.append(dict(**key,candidate=candidate,draw=draw,width=width,vote_score=vote.score,
                                    random_score=value,support_sha256=support,prediction_sha256=prediction))
    frame = pd.DataFrame(records)
    assert len(frame) == prior.N_CASES[suite]*len(NEW)*prior.N_RANDOM
    frame.to_csv(out/"random-draws.csv",index=False)
    frame["win"] = frame.vote_score-frame.random_score > 1e-12
    group = frame.groupby(KEY+["candidate"],as_index=False)
    median = group.agg(random_median=("random_score","median"),wins=("win","sum"))
    merged = pd.read_csv(out/"attribution.csv").merge(median,on=KEY+["candidate"],validate="one_to_one")
    merged["above_random_median"] = merged.score - merged.random_median > 1e-12
    merged["strong_joint"] = merged.joint & (merged.wins >= 15)
    merged.to_csv(out/"random-comparisons.csv",index=False)
    for levels,name in [(["candidate"],"overall"),(["dataset","candidate"],"by-source")]:
        merged.groupby(levels).agg(configs=("width","size"),reduced=("reduced","sum"),quality=("within","sum"),
            joint=("joint","sum"),no_loss=("joint_no_loss","sum"),strict=("strong_joint","sum"),
            both_windows=("both_windows","sum"),joint_10pct=("joint_10pct","sum"),
            above_random_median=("above_random_median","sum"),median_ratio=("ratio","median"),
            median_keep=("kept_fraction","median"),median_select_ms=("select_ms","median")).to_csv(out/(name+".csv"))


def run(out,suite,mode):
    state = json.loads(json.dumps(protocol(suite)))
    if mode == "measure":
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/"protocol.json",state)
    else:
        assert state == json.loads((out/"protocol.json").read_text(encoding="utf8"))
    if mode == "random":
        random_comparison(out,suite)
        return
    all_trials, all_audit, all_meta = [], [], []
    for index,panel in enumerate(prior.panels(suite)):
        trials,audit,metadata = evaluate(panel,suite,out,index,mode=="verify")
        all_trials.extend(trials)
        all_audit.extend(audit)
        all_meta.extend(metadata)
        print(mode,index+1,panel.name,panel.fold,panel.seed,flush=True)
    assert len(all_audit) == len(ARMS)*prior.N_CASES[suite]
    assert json.loads(json.dumps(protocol(suite))) == state
    meta = pd.DataFrame(all_meta)
    if mode == "verify":
        pd.testing.assert_frame_equal(meta,pd.read_csv(out/"panels.csv"),check_dtype=False)
        pd.DataFrame(all_audit).to_csv(out/"fresh-process-audit.csv",index=False)
    else:
        meta.to_csv(out/"panels.csv",index=False)
        assert len(all_trials) == len(all_audit)*prior.REPEATS[suite]
        summarize_timings(out,all_trials)
        attribution(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--panel",choices=tuple(prior.N_CASES),required=True)
    parser.add_argument("--mode",choices=("measure","verify","random"),default="measure")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.panel,args.mode)
