"""Vary positive vote mass at frozen ranks on the two focused confirmation panels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import signal_vote_confirmation as confirm
from experiments import signal_vote_peer_mechanism as mechanism
from experiments.adaptive_minpi_transfer import Panel
from experiments.compare_predictive_minpi import choose
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction, write_json
from minppi_mofs.signal_vote_minpi import signal_vote_top_from_training


MASS = (.95, .975, .99, .995)
RESULTS = ROOT / "results/minpi-study"
NOISE = RESULTS / "signal-vote-confirmation-20261117a"
PEERS = RESULTS / "signal-vote-peer-timing-20261117a"


def panels():
    names = dict(vote="vote", F="F_vote", random="random_vote", no_peer="zero_peer", full="full")
    for panel in confirm.panels():
        if panel.family != "raw":
            yield "varying-noise", panel, NOISE, names
    names = dict(vote="signal_auto", F="F_signal", random="random_signal", no_peer="no_peer", full="full")
    for task in mechanism.TASKS:
        for q in mechanism.PROXIES:
            for fold, seed in enumerate(mechanism.SEEDS, 1):
                X, y, Xt, truth, _ = mechanism.generated(task, q, seed)
                panel = Panel(f"{task}-q{q}", "synthetic", "proxy", task, fold, mechanism.SELECTOR_SEED,
                              X, y, Xt, truth, {})
                yield f"q{q}", panel, PEERS, names


def evaluate(group, panel, trace, names):
    X, y, Xt, truth = panel.X, panel.y, panel.Xt, panel.truth
    cols, info = signal_vote_top_from_training(X, y, task=panel.task, random_state=panel.seed)
    scores = info["scores"]
    rank = np.argsort(-scores, kind="stable")
    zero_rank = np.argsort(-info["zero_peer_scores"], kind="stable")
    mass = np.cumsum(np.maximum(scores[rank], 0.))
    full = np.arange(X.shape[1])
    rows = []
    with np.load(trace) as saved:
        full_prediction = prediction(X, y, Xt, full, panel.task, panel.seed)
        np.testing.assert_array_equal(full, saved["full_cols"])
        np.testing.assert_array_equal(full_prediction, saved["full_pred"])
        full_score = metric(truth, full_prediction, panel.task)
        for limit in MASS:
            k = int(np.searchsorted(mass, limit * mass[-1]) + 1) if mass[-1] else len(rank)
            choices = dict(vote=np.sort(rank[:k]), no_peer=np.sort(zero_rank[:k]),
                           random=choose("random", X, y, panel.task, panel.seed, k),
                           F=choose("f", X, y, panel.task, panel.seed, k))
            if limit == .99:
                np.testing.assert_array_equal(cols, choices["vote"])
            for method, selected in choices.items():
                assert len(selected) == k
                pred = prediction(X, y, Xt, selected, panel.task, panel.seed)
                if limit == .99:
                    np.testing.assert_array_equal(selected, saved[names[method]+"_cols"])
                    np.testing.assert_array_equal(pred, saved[names[method]+"_pred"])
                score = metric(truth, pred, panel.task)
                rows.append(dict(group=group, dataset=panel.name, task=panel.task, fold=panel.fold,
                                 mass=limit, method=method, width=k, p=len(rank), retained=k/len(rank),
                                 score=score, full_score=full_score, gap=score-full_score,
                                 within=score >= full_score-.02, reduced=k<len(rank),
                                 safe_compression=k<len(rank) and score >= full_score-.02,
                                 support_sha256=digest(selected), prediction_sha256=digest(pred)))
    return rows


def main(out, verify):
    spec = dict(mass=list(MASS), chosen_mass=.99, intent="Score and width sensitivity; no timing and no tuning. "
                "Use identical rank on every threshold, same-width F, random, and zero-peer, on all 20 original "
                "correlated measurement configurations and 14 predeclared varying-noise configurations.",
                sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in (Path(__file__), Path(confirm.__file__),
                         Path(mechanism.__file__), ROOT/"src/minppi_mofs/signal_vote_minpi.py",
                         NOISE/"protocol.json", PEERS/"protocol.json")})
    if verify:
        assert json.loads((out/"protocol.json").read_text()) == spec
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/"protocol.json", spec)
    cases = pd.DataFrame(row for group, panel, source, names in panels()
                         for row in evaluate(group, panel, source/f"{panel.name}-{panel.fold}-{panel.seed}.npz", names))
    assert len(cases) == len(MASS) * 34 * 4
    if verify:
        pd.testing.assert_frame_equal(cases, pd.read_csv(out/"cases.csv"), rtol=0, atol=1e-14)
        cases[["group", "dataset", "fold", "mass", "method", "support_sha256", "prediction_sha256"]].to_csv(
            out/"fresh-process-audit.csv", index=False)
        return
    cases.to_csv(out/"cases.csv", index=False)
    keys = ["group", "dataset", "task", "fold", "mass"]
    joined = cases.merge(cases[cases.method=="vote"][[*keys, "score"]].rename(columns={"score":"vote_score"}), on=keys)
    joined["vote_delta"] = joined.vote_score - joined.score
    joined["vote_win"] = joined.vote_delta > 1e-12
    joined["vote_loss"] = joined.vote_delta < -1e-12
    for keys, name in ((["group","dataset","mass","method"],"by-source"), (["group","mass","method"],"by-group")):
        joined.groupby(keys).agg(cases=("width","size"), width_min=("width","min"), width_med=("width","median"),
                                 width_max=("width","max"), retained_median=("retained","median"),
                                 safe_compression=("safe_compression","sum"), vote_wins=("vote_win","sum"),
                                 vote_losses=("vote_loss","sum"), median_score_delta=("vote_delta","median")
                                 ).to_csv(out/(name+".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        main(args.output, args.verify)
