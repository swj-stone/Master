"""Add seven timed repeats and 32 same-width random supports to the frozen mechanism panel."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_transfer as transfer
from experiments import signal_vote_benchmark as benchmark
from experiments import signal_vote_peer_mechanism as mechanism
from experiments.fast_minpi_five_source import summarize_timings
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric, prediction, write_json


ARMS = {"signal_auto": "vote", "no_peer": "no_peer", "F_signal": "F", "random_signal": "random", "full": "full"}
REPEATS = 7
N_DRAWS = 32


def select(method, panel, widths):
    return benchmark.select(method, panel, widths.get("signal"), ordered=False)[0]


def main(out, prior, verify):
    files = (Path(__file__), Path(benchmark.__file__), Path(transfer.__file__), Path(mechanism.__file__),
             ROOT / "src/minppi_mofs/signal_vote_minpi.py", prior / "protocol.json", prior / "cases.csv")
    protocol = dict(extension="Chosen after outcome of score-only mechanism experiment. Same 20 input matrices, "
                              "seeds, all five methods and truth. Report counts for q=12 and q=48. The frequent-proxy "
                              "subgroup result is exploratory. Strict width<p, test score gap>=-.02, median total "
                              "time<full. Include selector and width computation for vote; F and random know width. "
                              "32 additional uniform same-width draws, downstream seed held fixed, outside timer.",
                    repeats=REPEATS, draws=N_DRAWS, arms=list(ARMS),
                    files={str(p.resolve().relative_to(ROOT)): file_hash(p) for p in files})
    if verify:
        previous = json.loads((out/"protocol.json").read_text())
        assert previous.pop("started_utc") and previous == protocol
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/"protocol.json", dict(**protocol, started_utc=datetime.now(timezone.utc).isoformat()))
    saved = pd.read_csv(prior/"cases.csv").set_index(["task", "proxies_available", "seed", "method"])
    traces, audits, random_draws = [], [], []
    for task in mechanism.TASKS:
        for q in mechanism.PROXIES:
            for fold, seed in enumerate(mechanism.SEEDS, 1):
                X, y, Xt, truth, _ = mechanism.generated(task, q, seed)
                panel = transfer.Panel(f"{task}-q{q}", "synthetic", "proxy", task, fold, mechanism.SELECTOR_SEED,
                                       X, y, Xt, truth, {})
                key = benchmark.case_id(panel)
                trace = out / f"{panel.name}-{fold}-{panel.seed}.npz"
                rows, checked = transfer.run_case(panel, [898412, q, fold, mechanism.TASKS.index(task)],
                                                  REPEATS, trace, verify, tuple(ARMS), select, {"signal_auto": "signal"})
                traces.extend(rows)
                audits.extend(checked)
                with np.load(trace) as arrays:
                    for method, previous in ARMS.items():
                        record = saved.loc[task, q, seed, previous]
                        assert record.support_sha256 == digest(arrays[method+"_cols"])
                        assert record.prediction_sha256 == digest(arrays[method+"_pred"])
                    width = len(arrays["signal_auto_cols"])
                for draw in range(N_DRAWS):
                    rng = np.random.default_rng([638819, mechanism.TASKS.index(task), q, seed, draw])
                    cols = np.sort(rng.permutation(X.shape[1])[:width])
                    pred = prediction(X, y, Xt, cols, task, panel.seed)
                    random_draws.append(dict(**key, draw=draw, width=width, score=metric(truth, pred, task),
                                             support_sha256=digest(cols), prediction_sha256=digest(pred)))
    random_frame = pd.DataFrame(random_draws)
    assert len(random_frame) == 20 * N_DRAWS
    if verify:
        assert len(audits) == 20 * len(ARMS)
        pd.testing.assert_frame_equal(random_frame, pd.read_csv(out/"random-draws.csv"), rtol=0, atol=1e-14)
        pd.DataFrame(audits).to_csv(out/"fresh-process-audit.csv", index=False)
        return
    assert len(traces) == 20 * REPEATS * len(ARMS)
    cases = summarize_timings(out, traces)
    random_frame.to_csv(out/"random-draws.csv", index=False)
    votes = cases[cases.method == "signal_auto"].copy()
    by_draw = random_frame.merge(votes[[*benchmark.CASE, "score"]], on=benchmark.CASE, suffixes=("_draw", "_vote"))
    by_draw["win"] = by_draw.score_vote > by_draw.score_draw + 1e-12
    counts = by_draw.groupby(benchmark.CASE, as_index=False).win.sum().rename(columns={"win": "wins_out_of_32"})
    votes = votes.merge(counts, on=benchmark.CASE)
    votes["strict"] = votes.joint & (votes.wins_out_of_32 >= 24)
    votes.to_csv(out/"vote-comparisons.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=ROOT/"results/minpi-study/signal-vote-peer-mechanism-20261117a")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        main(args.output, args.prior, args.verify)
