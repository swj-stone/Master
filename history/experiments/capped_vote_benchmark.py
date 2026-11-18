"""Clock a frozen capped adaptive vote against matched and independent budgets."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import next_vote_benchmark as previous
from experiments import signal_vote_benchmark as base
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from experiments.stable_minpi_stress import ROOT, digest, file_hash, write_json
from minppi_mofs.capped_vote_minpi import capped_vote_top_from_training


METHODS = ("capped", "F_capped", "random_capped", "F50", "random50", "signal_auto", "full")
KEY = base.CASE


def snapshot(suite):
    state = previous.protocol(suite)
    state["arms"] = list(METHODS)
    files = [Path(__file__), ROOT/"src/minppi_mofs/capped_vote_minpi.py"]
    state["source_sha256"].update({f.relative_to(ROOT).as_posix():file_hash(f) for f in files})
    return json.loads(json.dumps(state))


def support(panel, method, width, ordered):
    if method == "capped":
        return capped_vote_top_from_training(panel.X,panel.y,task=panel.task,random_state=panel.seed,
                                              groups=panel.groups,ordered_groups=ordered)
    return previous.select(method,panel,{"capped":width},ordered)


def measure_case(panel,suite,out,index,verify):
    key = base.case_id(panel)
    archive = f"{panel.name}-{panel.fold}-{panel.seed}.npz"
    ordered = panel.name in ("gas","secom") if suite=="original" else panel.ordered
    truth = panel.gold if suite=="original" else panel.truth
    trace = {"input_hashes": np.array([digest(x) for x in (panel.X,panel.y,panel.Xt,truth)])}
    audits, observations, scores = [], [], {}
    width = None
    for arm in METHODS:
        columns, info = support(panel,arm,width,ordered)
        check_columns(columns,panel.X.shape[1])
        if arm == "capped":
            width = len(columns)
            if info["folds"]:
                base.deployment.check_folds(info,panel.groups,ordered,len(panel.y))
            zero = np.asarray(info["zero_peer_scores"])
            assert zero.shape == (panel.X.shape[1],) and np.isfinite(zero).all()
            trace["zero_cols"] = np.sort(np.argsort(-zero,kind="stable")[:width])
            trace["zero_pred"], zero_dec = base.fit(panel,trace["zero_cols"],suite)
            trace["zero_score"] = np.asarray(base.score(panel,trace["zero_pred"],zero_dec,suite)[0])
        elif arm.endswith("_capped"):
            assert len(columns) == width
        pred, decision = base.fit(panel,columns,suite)
        score, _, ap = base.score(panel,pred,decision,suite)
        scores[arm] = score, ap
        trace[arm+"_cols"], trace[arm+"_pred"] = columns,pred
        if panel.name == "secom":
            trace[arm+"_decision"] = decision
        audits.append(dict(**key,method=arm,width=len(columns),score=score,
                           support_sha256=digest(columns),prediction_sha256=digest(pred)))
    with np.load(previous.PREVIOUS[suite]/archive) as old:
        for field in ("input_hashes","full_pred","full_cols","signal_auto_pred","signal_auto_cols"):
            np.testing.assert_array_equal(trace[field],old[field],err_msg=archive+field)
    if verify:
        with np.load(out/archive) as saved:
            assert set(trace) == set(saved.files)
            for field,value in trace.items():
                np.testing.assert_array_equal(value,saved[field],err_msg=archive+field)
        return observations,audits
    initial = np.random.default_rng([319877,index,panel.seed]).permutation(METHODS)
    for repeat in range(base.REPEATS[suite]):
        for position,arm in enumerate(np.roll(initial,repeat)):
            start = perf_counter_ns()
            columns, _ = support(panel,arm,width,ordered)
            selected = perf_counter_ns()
            pred,decision = base.fit(panel,columns,suite)
            end = perf_counter_ns()
            np.testing.assert_array_equal(columns,trace[arm+"_cols"])
            np.testing.assert_array_equal(pred,trace[arm+"_pred"])
            if panel.name=="secom":
                np.testing.assert_array_equal(decision,trace[arm+"_decision"])
            score,ap = scores[arm]
            observations.append(dict(**key,method=arm,repeat=repeat,position=position,width=len(columns),
                                     p=panel.X.shape[1],score=score,ap=ap,select_ms=(selected-start)/1e6,
                                     refit_ms=(end-selected)/1e6,total_ms=(end-start)/1e6))
    np.savez_compressed(out/archive,**trace)
    return observations,audits


def compare(out):
    cases = pd.read_csv(out/"cases.csv").set_index(KEY+["method"]).sort_index()
    a = cases.xs("capped",level="method").copy()
    rows = []
    for arm in METHODS[1:]:
        b = cases.xs(arm,level="method")
        if arm.endswith("_capped"):
            np.testing.assert_array_equal(a.width,b.width)
        frame = pd.DataFrame(dict(width=a.width,p=a.p,baseline_width=b.width,score=a.score,
                                  baseline_score=b.score,total_ms=a.total_ms,baseline_total_ms=b.total_ms))
        frame["candidate"] = "capped"
        frame["baseline"] = arm
        frame["score_gap"] = a.score-b.score
        frame["time_ratio"] = a.total_ms/b.total_ms
        frame["width_ratio"] = a.width/b.width
        valid = a.reduced & (frame.score_gap >= -.02) & (frame.time_ratio < 1.)
        frame["relative_joint"] = valid & (a.width<=b.width)
        frame["also_full_joint"] = frame.relative_joint & a.joint
        frame["relative_joint_width_10pct"] = valid & (10*a.width<=11*b.width)
        frame["exact_quality_joint"] = frame.relative_joint & (frame.score_gap>=0.)
        rows.append(frame.reset_index())
    pairs = pd.concat(rows,ignore_index=True)
    pairs.to_csv(out/"baseline-comparisons.csv",index=False)
    pairs.groupby("baseline").agg(cases=("width","size"),success=("relative_joint","sum"),
        full_too=("also_full_joint","sum"),margin=("relative_joint_width_10pct","sum"),
        no_loss=("exact_quality_joint","sum"),median_score_gap=("score_gap","median"),
        median_ratio=("time_ratio","median"),median_width_ratio=("width_ratio","median")).to_csv(
        out/"baseline-summary.csv")
    for ix in a.index:
        with np.load(out/("-".join(str(k) for k in ix)+".npz")) as trace:
            a.loc[ix,"zero_score"] = trace["zero_score"].item()
    a.to_csv(out/"attribution.csv")


def random_comparison(out,suite):
    saved = pd.read_csv(out/"cases.csv").set_index(KEY+["method"])
    rows = []
    for panel in base.panels(suite):
        key = base.case_id(panel)
        ix = (panel.name,panel.fold,panel.seed)
        a = saved.loc[(*ix,"capped")]
        for draw in range(20):
            seed = panel.seed if draw==0 else [891703,panel.seed,draw]
            cols = np.sort(np.random.default_rng(seed).permutation(panel.X.shape[1])[:int(a.width)])
            pred,dec = base.fit(panel,cols,suite)
            value = base.score(panel,pred,dec,suite)[0]
            if draw==0:
                np.testing.assert_allclose(value,saved.loc[(*ix,"random_capped")].score,rtol=0,atol=1e-12)
            rows.append(dict(**key,candidate="capped",draw=draw,width=int(a.width),vote_score=a.score,
                             random_score=value,support_sha256=digest(cols),prediction_sha256=digest(pred)))
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"random-draws.csv",index=False)
    frame["win"] = frame.vote_score-frame.random_score>1e-12
    g = frame.groupby(KEY).agg(wins=("win","sum"),draws=("draw","size"),random_median=("random_score","median"))
    a = pd.read_csv(out/"attribution.csv").set_index(KEY)
    g = a.join(g,validate="one_to_one")
    assert g.draws.eq(20).all() and len(g)==base.N_CASES[suite]
    g["strong_joint"] = g.joint & (g.wins>=15)
    g["above_random_median"] = g.score-g.random_median>1e-12
    g.to_csv(out/"random-comparisons.csv")


def run(out,suite,mode):
    manifest = snapshot(suite)
    if mode=="measure":
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/"protocol.json",manifest)
    else:
        assert manifest==json.loads((out/"protocol.json").read_text(encoding="utf8"))
    if mode=="random":
        random_comparison(out,suite)
        return
    trials,audits = [], []
    for i,panel in enumerate(base.panels(suite)):
        records,log = measure_case(panel,suite,out,i,mode=="verify")
        trials.extend(records)
        audits.extend(log)
        print(mode,i+1,panel.name,panel.fold,panel.seed,flush=True)
    assert manifest==snapshot(suite) and len(audits)==base.N_CASES[suite]*len(METHODS)
    if mode=="verify":
        pd.DataFrame(audits).to_csv(out/"fresh-process-audit.csv",index=False)
    else:
        assert len(trials)==len(audits)*base.REPEATS[suite]
        summarize_timings(out,trials)
        compare(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",choices=tuple(base.N_CASES),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=("measure","verify","random"),default="measure")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.panel,args.mode)
