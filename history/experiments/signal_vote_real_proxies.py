"""Test the preregistered strong-peer mechanism on two measured targets with repeated channels."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from experiments.adaptive_minpi_transfer import Panel, run_case
from experiments.fast_minpi_five_source import summarize_timings
from experiments.signal_vote_confirmation import ARMS, SELECT_SEED, compare_draws, select
from experiments.stable_minpi_stress import ROOT, digest, file_hash


SEED = 2026111711
PROTOCOL = ROOT/"docs/minpi-study/signal-vote-real-proxy-protocol.md"
SOURCES = {
    "banknote": ("data_banknote_authentication.txt", ",", "classification",
                 "1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227"),
    "airfoil": ("airfoil_self_noise.dat", None, "regression",
                "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"),
}


def panels():
    for idx, (name, (member, delimiter, task, expected)) in enumerate(SOURCES.items()):
        archive = ROOT/f"data/raw/uci-{name}.zip"
        assert file_hash(archive) == expected
        with ZipFile(archive) as z, z.open(member) as f:
            values = np.loadtxt(f, delimiter=delimiter)
        X, y = values[:, :-1], values[:, -1]
        # Record identifiers are source rows; de-duplication never consults y.
        _, first = np.unique(X, axis=0, return_index=True)
        first.sort()
        X, y = X[first], y[first]
        if task == "classification":
            y = y.astype(int)
        seed = int(np.random.default_rng([SEED, 0, idx]).integers(2**31))
        folds = (StratifiedKFold if task == "classification" else KFold)(3, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(folds.split(X, y), start=1):
            train_centered = X[tr] - X[tr].mean(axis=0)
            norm = np.linalg.norm(train_centered, axis=0)
            rhs = np.abs(train_centered.T @ (y[tr] - y[tr].mean()))
            source = int(np.divide(rhs, norm, out=np.zeros_like(rhs), where=norm > 0).argmax())
            for expanded in (False, True):
                data = X
                types = np.full(X.shape[1], "other", dtype="<U8")
                types[source] = "source"
                if expanded:
                    rng = np.random.default_rng([SEED, 1, idx, fold])
                    scale = X[tr,source].std() * np.sqrt(.98**-2 - 1.)
                    proxies = X[:, source, None] + scale * rng.normal(size=(len(X), 47))
                    nuisance = rng.normal(size=(len(X), 128 - X.shape[1] - 47))
                    perm = rng.permutation(128)
                    data = np.column_stack((X, proxies, nuisance))[:, perm]
                    types = np.concatenate([types, np.repeat("copy", 47), np.repeat("noise", nuisance.shape[1])])[perm]
                family = "semisynthetic" if expanded else "raw"
                split = {"train_sha256": digest(first[tr]), "test_sha256": digest(first[te]),
                         "source_col": source, "types": types, "distinct_rows": len(first),
                         "input_rows": len(values)}
                yield Panel(name+"_"+family, family, name, task, fold, SELECT_SEED,
                            np.ascontiguousarray(data[tr],dtype=np.float32), y[tr],
                            np.ascontiguousarray(data[te],dtype=np.float32), y[te], split)


def summaries(out, rows, meta, draws):
    cases = summarize_timings(out, rows)
    key = ["dataset", "fold", "seed"]
    meta.to_csv(out/"panels.csv", index=False)
    draws.to_csv(out/"random-draws.csv", index=False)
    random = draws.groupby(key).agg(random_wins=("vote_wins","sum"), draws=("draw","size"),
                                    random_median=("score","median"),
                                    random_q25=("score", lambda x: x.quantile(.25)),
                                    random_q75=("score", lambda x: x.quantile(.75)))
    controls = cases.pivot(index=key, columns="method", values="score")[["F_vote","random_vote","zero_peer"]]
    results = cases[cases.method == "vote"].merge(meta,on=key,validate="one_to_one")
    results = results.join(random,on=key,validate="one_to_one").join(controls,on=key,validate="one_to_one")
    assert len(results) == 12 and results.draws.eq(32).all()
    results.rename(columns={"random_wins":"vote_beats_draws"}, inplace=True)
    results["strict_success"] = results.joint & (results.vote_beats_draws >= 24)
    results["gap_random_median"] = results.score - results.random_median
    results["gap_F"] = results.score - results.F_vote
    results["gap_zero_peer"] = results.score - results.zero_peer
    results["better_than_both"] = (results.gap_F > 0) & (results.gap_zero_peer > 0)
    results.to_csv(out/"contest.csv",index=False)
    results.groupby(["dataset","family"]).agg(
        cases=("fold","size"), reduce=("reduced","sum"), within=("within","sum"), joint=("joint","sum"),
        joint_5pct=("joint_5pct","sum"), joint_10pct=("joint_10pct","sum"),
        strict_success=("strict_success","sum"), better_than_both=("better_than_both","sum"),
        median_ratio=("ratio","median"), median_F_gap=("gap_F","median"),
        median_zero_gap=("gap_zero_peer","median"), median_random_gap=("gap_random_median","median")
    ).to_csv(out/"by-source.csv")


def run(out, verify):
    paths = [PROTOCOL, Path(__file__), ROOT/"experiments/signal_vote_confirmation.py",
             ROOT/"experiments/adaptive_minpi_transfer.py", ROOT/"experiments/stable_minpi_stress.py",
             ROOT/"experiments/minpi_study.py", ROOT/"experiments/compare_predictive_minpi.py",
             ROOT/"experiments/fast_minpi_five_source.py", ROOT/"configs/minpi-study.json",
             ROOT/"src/minppi_mofs/signal_vote_minpi.py"]
    state = {"source_sha256": {k:v[-1] for k,v in SOURCES.items()},
             "implementation_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths},
             "seed": SEED, "candidate_seed": SELECT_SEED, "methods": list(ARMS), "single_thread": True,
             "repeats": 7, "random_draws": 32}
    if verify:
        assert json.loads((out/"protocol.json").read_text(encoding="utf8")) == state
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/"protocol.json").write_text(json.dumps(state, indent=2),encoding="utf8")
    rows, meta, draws, audits = [], [], [], []
    for i, panel in enumerate(panels()):
        trace = out/f"{panel.name}-{panel.fold}-{panel.seed}.npz"
        assert (np.ptp(panel.X, axis=0) > 0).all()
        trials, audit = run_case(panel,[SEED, 2, i],7,trace,verify,ARMS,select,{"vote":"vote"})
        rows.extend(trials)
        audits.extend(audit)
        draws.extend(compare_draws(panel,trace,100+i))
        split = dict(panel.split_info)
        types = split.pop("types")
        with np.load(trace) as stored:
            widths = [len(stored[m+"_cols"]) for m in ARMS if m != "full"]
            assert len(set(widths)) == 1
            for method in ARMS:
                kept = types[stored[method+"_cols"]]
                for kind in ("source","other","copy","noise"):
                    split[f"{method}_{kind}_count"] = np.count_nonzero(kept == kind)
        meta.append(dict(dataset=panel.name,fold=panel.fold,seed=panel.seed,family=panel.family,
                         n_train=len(panel.y),n_test=len(panel.truth), **split))
        print(f"{i+1}/12 {panel.name} split {panel.fold}",flush=True)
    draws, meta = pd.DataFrame(draws), pd.DataFrame(meta)
    if verify:
        pd.testing.assert_frame_equal(meta,pd.read_csv(out/"panels.csv"))
        pd.testing.assert_frame_equal(draws,pd.read_csv(out/"random-draws.csv"),rtol=0,atol=1e-14)
        pd.DataFrame(audits).to_csv(out/"fresh-process-audit.csv",index=False)
        draws.to_csv(out/"fresh-process-random-draws.csv",index=False)
    else:
        summaries(out,rows,meta,draws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.verify)
