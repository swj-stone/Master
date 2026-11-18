"""Predeclared signal-vote replication on varying noise columns and two original surveys."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from statsmodels.datasets import anes96, fair
from threadpoolctl import threadpool_limits

from experiments.adaptive_minpi_transfer import Panel, run_case
from experiments.compare_predictive_minpi import choose
from experiments.fast_minpi_five_source import summarize_timings
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction
from minppi_mofs.signal_vote_minpi import signal_vote_top_from_training


ROOT_SEED = 2026111709
SELECT_SEED = 1129
ARMS = ("vote", "random_vote", "F_vote", "zero_peer", "full")
PROTOCOL = ROOT / "docs/minpi-study/signal-vote-confirmation-protocol.md"


def random_generator(*key):
    return np.random.default_rng([ROOT_SEED, *key])


def simulation(task, fold):
    rng = random_generator(0, int(task == "regression"), fold)
    n, p, active = 2304, 384, 12
    X = rng.standard_normal((n, p))
    weights = np.linspace(.5, 1.5, active) * rng.choice([-1., 1.], active)
    target = X[:, :active] @ (weights / np.linalg.norm(weights)) + rng.normal(0, .45, n)
    if task == "classification":
        target = (target > 0).astype(int)
    perm = rng.permutation(p)
    X = np.ascontiguousarray(X[:, perm], dtype=np.float32)
    return Panel("independent_" + task, "controlled", "independent_noise", task,
                 fold, SELECT_SEED, X[:1536], target[:1536], X[1536:], target[1536:],
                 {"source_cols": np.flatnonzero(perm < active).tolist()})


def survey_panels(module, task, target, index):
    frame = module.load_pandas().data
    X = frame.drop(columns=[target, "logpopul"], errors="ignore")
    keep = np.flatnonzero(~X.duplicated())
    y = frame[target].to_numpy()[keep]
    X = X.to_numpy(dtype=np.float32)[keep]
    split_seed = int(random_generator(1, index).integers(2**31))
    splitter = (StratifiedKFold if task == "classification" else KFold)(
        n_splits=3, shuffle=True, random_state=split_seed)
    source = module.__name__.split(".")[-1]
    for fold, (tr, te) in enumerate(splitter.split(X, y), start=1):
        split = {"train_digest": digest(keep[tr]), "test_digest": digest(keep[te]),
                 "input_rows": len(frame), "distinct_rows": len(y)}
        for expanded in (False, True):
            data = X
            original_cols = np.arange(X.shape[1])
            if expanded:
                rng = random_generator(2, index, fold)
                noise = rng.standard_normal((len(X), 192-X.shape[1]), dtype=np.float32)
                perm = rng.permutation(192)
                data = np.column_stack((X, noise))[:, perm]
                original_cols = np.flatnonzero(perm < X.shape[1])
            family = "semisynthetic" if expanded else "raw"
            yield Panel(source + "_" + family, family, source, task, fold, SELECT_SEED,
                        np.ascontiguousarray(data[tr]), y[tr], np.ascontiguousarray(data[te]), y[te],
                        {**split, "source_cols": original_cols.tolist()})


def panels():
    for task in ("classification", "regression"):
        for fold in range(1, 5):
            yield simulation(task, fold)
    yield from survey_panels(anes96, "classification", "vote", 0)
    yield from survey_panels(fair, "regression", "affairs", 1)


def select(method, panel, widths):
    if method in ("vote", "zero_peer"):
        columns, info = signal_vote_top_from_training(panel.X, panel.y, task=panel.task,
                                                      random_state=panel.seed)
        if method == "vote":
            return columns
        rank = np.argsort(-info["zero_peer_scores"], kind="stable")
        return np.sort(rank[:widths["vote"]])
    return choose(method.split("_")[0].lower(), panel.X, panel.y, panel.task,
                  panel.seed, widths["vote"])


def compare_draws(panel, trace, case_index):
    rows = []
    with np.load(trace) as saved:
        vote_cols = saved["vote_cols"]
        score = metric(panel.truth, saved["vote_pred"], panel.task)
    for draw in range(32):
        rng = random_generator(4, case_index, draw)
        cols = np.sort(rng.permutation(panel.X.shape[1])[:len(vote_cols)])
        pred = prediction(panel.X, panel.y, panel.Xt, cols, panel.task, panel.seed)
        random_score = metric(panel.truth, pred, panel.task)
        rows.append(dict(dataset=panel.name, fold=panel.fold, seed=panel.seed, draw=draw, width=len(cols),
                         score=random_score, vote_score=score, vote_wins=score > random_score,
                         support_sha256=digest(cols), prediction_sha256=digest(pred)))
    return rows


def summarize(out, rows, metadata, draws):
    cases = summarize_timings(out, rows)
    key = ["dataset", "fold", "seed"]
    meta = pd.DataFrame(metadata)
    meta.to_csv(out/"panels.csv", index=False)
    draws.to_csv(out/"random-draws.csv", index=False)
    contest = draws.groupby(key).agg(draws=("draw", "size"), vote_wins=("vote_wins", "sum"),
                                     random_median=("score", "median"),
                                     random_q25=("score", lambda s: s.quantile(.25)),
                                     random_q75=("score", lambda s: s.quantile(.75)))
    compares = cases.pivot(index=key, columns="method", values="score")[["F_vote", "zero_peer", "random_vote"]]
    results = cases[cases.method == "vote"].merge(meta, on=key, validate="one_to_one")
    results = results.join(contest, on=key, validate="one_to_one").join(compares, on=key, validate="one_to_one")
    assert (results.draws == 32).all() and len(results) == 20
    results["strict_success"] = results.joint & (results.vote_wins >= 24)
    results["gap_to_random_median"] = results.score - results.random_median
    results["joint_no_loss"] = results.reduced & (results.gap >= 0) & (results.ratio < 1)
    results["gap_to_F"] = results.score - results.F_vote
    results["gap_to_zero_peer"] = results.score - results.zero_peer
    results.to_csv(out/"contest.csv", index=False)
    groups = results.groupby(["dataset", "family", "task"]).agg(
        cases=("fold", "size"), within=("within", "sum"), joint=("joint", "sum"),
        strict_success=("strict_success", "sum"), median_random_gap=("gap_to_random_median", "median"),
        min_random_wins=("vote_wins", "min"), median_F_gap=("gap_to_F", "median"),
        median_zero_gap=("gap_to_zero_peer", "median"), joint_5pct=("joint_5pct", "sum"),
        joint_10pct=("joint_10pct", "sum"), median_total_ratio=("ratio","median"))
    groups.to_csv(out/"by-source.csv")


def run(out, verify):
    files = [PROTOCOL, Path(__file__), ROOT/"src/minppi_mofs/signal_vote_minpi.py",
             ROOT/"experiments/adaptive_minpi_transfer.py", ROOT/"experiments/stable_minpi_stress.py",
             ROOT/"experiments/minpi_study.py", ROOT/"experiments/compare_predictive_minpi.py",
             ROOT/"experiments/fast_minpi_five_source.py", ROOT/"configs/minpi-study.json"]
    local_hashes = {f.relative_to(ROOT).as_posix(): file_hash(f) for f in files}
    state = {"local_sha256": local_hashes, "source_sha256": {
        name: file_hash(Path(module.__file__).with_name(name + ".csv"))
        for name, module in (("anes96", anes96), ("fair", fair))},
             "root_seed": ROOT_SEED, "selection_seed": SELECT_SEED, "arms": list(ARMS),
             "rotated_repeats": 7, "independent_random_draws": 32, "threads": 1}
    if verify:
        assert json.loads((out/"protocol.json").read_text(encoding="utf8")) == state
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out/"protocol.json").write_text(json.dumps(state, indent=2), encoding="utf8")
    rows, metadata, audits, draws = [], [], [], []
    for idx, panel in enumerate(panels()):
        trace = out/f"{panel.name}-{panel.fold}-{panel.seed}.npz"
        train_n, p = panel.X.shape
        known = np.asarray(panel.split_info["source_cols"])
        target_class = p >= 128 and train_n >= 2*p and 1 - len(known)/p >= .75
        assert target_class == (panel.family != "raw")
        assert (np.ptp(panel.X, axis=0) > 0).all()
        record, audit = run_case(panel, [ROOT_SEED, 3, idx], 7, trace, verify, ARMS, select,
                                 adaptive_names={"vote": "vote"})
        rows.extend(record)
        audits.extend(audit)
        draws.extend(compare_draws(panel, trace, idx))
        with np.load(trace) as saved:
            lengths = {len(saved[m+"_cols"]) for m in ARMS if m != "full"}
            assert len(lengths) == 1
            meta = {m+"_source_kept": len(np.intersect1d(known, saved[m+"_cols"])) for m in ARMS}
            meta["vote_zero_peer_same"] = np.array_equal(saved["vote_cols"], saved["zero_peer_cols"])
            meta["vote_F_same"] = np.array_equal(saved["vote_cols"], saved["F_vote_cols"])
        metadata.append(dict(dataset=panel.name, fold=panel.fold, seed=panel.seed, family=panel.family,
                             task=panel.task, n_train=train_n, n_test=len(panel.truth), source_p=len(known),
                             target_class=target_class, split=json.dumps(panel.split_info), **meta))
        print(f"{idx+1}/20 {panel.name} split {panel.fold}", flush=True)
    draws = pd.DataFrame(draws)
    if verify:
        reference = pd.read_csv(out/"random-draws.csv")
        pd.testing.assert_frame_equal(draws, reference, check_exact=False, rtol=0, atol=1e-14)
        pd.DataFrame(audits).to_csv(out/"fresh-process-audit.csv", index=False)
        draws.to_csv(out/"fresh-process-random-draws.csv", index=False)
    else:
        summarize(out, rows, metadata, draws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
