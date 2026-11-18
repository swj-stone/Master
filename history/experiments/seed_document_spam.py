"""Frozen three-additional-seed stress test on eight unchanged spam-detection partitions."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import document_vote_benchmark as old
from experiments.fast_document_vote_benchmark import select
from experiments.fast_minpi_five_source import check_columns, summarize_timings
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training, library_path

SEEDS = (6173, 8111, 9923)
ARMS = ("fast_vote", "full", "F_50", "chi_50", "random_50", "F_match", "random_match", "pooled_match")
CASE = old.CASE


def measure(X,y,Xt,truth,meta,out,verify):
    key = {name:meta[name] for name in CASE}
    seed = key["seed"]
    trace = out / f"{key['dataset']}-{key['fold']}-{seed}.npz"
    ref, ref_info = old.document_vote_top_from_training(X, y, random_state=seed)
    candidate, native = fast_document_vote_top_from_training(X, y, random_state=seed, backend="native")
    np.testing.assert_array_equal(candidate, ref)
    for field in ("scores","zero_peer_scores","one_peer_scores"):
        np.testing.assert_array_equal(native[field], ref_info[field])
    width = len(candidate)
    arrays, audit, scores, trials = {}, [], {}, []
    for method in ARMS:
        cols = select(method, X, y, seed, width)
        check_columns(cols, X.shape[1])
        if method.endswith("match"):
            assert len(cols) == width
        elif method.endswith("50"):
            assert len(cols) == (X.shape[1]+1)//2
        pred = old.fit_predict(X, y, Xt, cols, "classification", seed, False)
        arrays[method+"_cols"], arrays[method+"_pred"] = cols, pred
        scores[method] = old.balanced_accuracy_score(truth, pred)
        audit.append(dict(**key, method=method, score=scores[method], width=len(cols),
                          support_sha256=old.digest(cols), prediction_sha256=old.digest(pred)))
    if verify:
        with np.load(trace) as saved:
            for k,v in arrays.items():
                np.testing.assert_array_equal(v, saved[k])
        return [],audit
    np.savez_compressed(trace, **arrays)
    order = np.random.default_rng([427051,key["fold"],seed]).permutation(ARMS)
    for r in range(len(ARMS)):
        for pos, method in enumerate(np.roll(order,r)):
            start = perf_counter_ns()
            cols = select(method, X, y, seed, width)
            selected = perf_counter_ns()
            pred = old.fit_predict(X, y, Xt, cols, "classification", seed, False)
            end = perf_counter_ns()
            np.testing.assert_array_equal(cols, arrays[method+"_cols"])
            np.testing.assert_array_equal(pred, arrays[method+"_pred"])
            trials.append(dict(**key,method=method,repeat=r,position=pos,width=len(cols),p=X.shape[1],
                               score=scores[method],select_ms=(selected-start)/1e6,refit_ms=(end-selected)/1e6,
                               total_ms=(end-start)/1e6))
    return trials,audit


def summarize(out, trials):
    cases = summarize_timings(out, trials)
    vote = cases[cases.method=="fast_vote"].set_index(CASE).copy()
    for method in ARMS[1:]:
        baseline = cases[cases.method==method].set_index(CASE)
        vote[method+"_score_delta"] = vote.score-baseline.score
        vote[method+"_time_ratio"] = vote.total_ms/baseline.total_ms
        vote[method+"_joint"] = (vote.joint & (vote.width<=baseline.width) &
                                 (vote[method+"_score_delta"]>=-.02) & (vote[method+"_time_ratio"]<1))
    vote["independent_baselines"] = vote.F_50_joint & vote.chi_50_joint & vote.random_50_joint
    vote["matched_baselines"] = vote.F_match_joint & vote.random_match_joint & vote.pooled_match_joint
    vote.to_csv(out/"comparisons.csv")
    aggregation = dict(cases=("score","size"), min_width=("width","min"), max_width=("width","max"),
                       min_score=("score","min"), max_score=("score","max"), quality=("within","sum"),
                       joint=("joint","sum"), joint10=("joint_10pct","sum"),
                       independent=("independent_baselines","sum"), matched=("matched_baselines","sum"),
                       F_half=("F_50_joint","sum"), F_match=("F_match_joint","sum"),
                       random_match=("random_match_joint","sum"), pooled_match=("pooled_match_joint","sum"),
                       median_random_delta=("random_match_score_delta","median"),
                       median_random_time_ratio=("random_match_time_ratio","median"))
    for keys, name in [(["dataset"],"by-source"), (["dataset","fold"],"by-fold"), (["dataset","seed"],"by-seed")]:
        vote.groupby(keys).agg(**aggregation).to_csv(out/(name+".csv"))


def run(out, verify):
    base = old.ROOT/"results/minpi-study/document-vote-20261117b"
    hashes = [Path(__file__), Path(old.__file__), base/"panels.csv", old.ROOT/"experiments/fast_document_vote_benchmark.py",
              old.ROOT/"experiments/minpi_constant_transfer.py", library_path(),
              old.ROOT/"src/minppi_mofs/fast_document_vote_minpi.py",old.ROOT/"src/minppi_mofs/document_vote_kernel.c",
              old.ROOT/"src/minppi_mofs/document_vote_minpi.py", old.ROOT/"configs/minpi-study.json"]
    protocol = dict(seeds=list(SEEDS), methods=list(ARMS), repeats=len(ARMS), sources=old.ASSETS,
                    file_sha256={f.relative_to(old.ROOT).as_posix():old.file_hash(f) for f in hashes},
                    test="All three new seeds on all fixed outer splits, no selection by results; "
                         "same check against full, plus simultaneous relative comparisons to independent "
                         "half-width F, chi-square, random. Equal-width methods reported separately.")
    if verify:
        assert json.loads((out/"protocol.json").read_text()) == protocol
    else:
        out.mkdir(parents=True, exist_ok=False)
        old.write_json(out/"protocol.json",protocol)
    frozen = pd.read_csv(base/"panels.csv").set_index(["dataset","fold"])
    trials, audits = [], []
    for X,y,Xt,truth,meta in old.panels():
        if meta["task"] != "spam":
            continue
        original = frozen.loc[(meta["dataset"],meta["fold"])]
        assert original.seed not in SEEDS
        for name in meta:
            if name.endswith("_sha256"):
                assert meta[name] == original[name]
        for seed in SEEDS:
            rows, replay = measure(X,y,Xt,truth,dict(meta,seed=seed),out,verify)
            trials.extend(rows)
            audits.extend(replay)
        print("Verified" if verify else "Measured",meta["dataset"],meta["fold"],flush=True)
    assert len(audits) == 24*len(ARMS)
    if verify:
        pd.DataFrame(audits).to_csv(out/"fresh-process-audit.csv",index=False)
    else:
        assert len(trials) == 24*len(ARMS)**2
        summarize(out, trials)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.output.resolve(), args.verify)
