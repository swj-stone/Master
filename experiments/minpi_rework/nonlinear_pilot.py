"""Development-only comparison of triplet margin, vote and stratum minimum.

Run with thesis-python.ps1 -m experiments.minpi_rework.nonlinear_pilot.
The archived width split supplies the pilot scores. Outer labels are never read.
"""
import argparse
import json
import math
from types import SimpleNamespace
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark, legacy_gain, legacy_gain_receiver
from minppi_mofs.nonlinear_vote import rank_triplets
from experiments.minpi_rework.nonlinear_vote_tree import rank_trees


ROOT = benchmark.ROOT / "results/minpi_rework/nonlinear"


def run(name, seed, directory, row, panels, game):
    x, y, task, idx = legacy_gain.read_split(directory, row, seed)
    started = perf_counter()
    fit, vote, width = (idx[k] for k in ("fit", "vote", "width"))
    ranker, settings = ((rank_triplets, dict(panels=panels)) if game == "triplet" else
                        (rank_trees, dict(trees=panels, conditional=game == "conditional")))
    ranks = ranker(x[fit], y[fit], x[vote], y[vote], seed=seed, task=task, **settings)
    rank_s = perf_counter() - started
    pool = np.sort(np.r_[fit, vote])
    p = x.shape[1]
    ks = sorted({1, *(math.ceil(p*f) for f in benchmark.DESIGN["width_fractions"])})
    archive = legacy_gain_receiver.OUT / f"{name}-{seed}" / "selection.json"
    prior = json.loads(archive.read_bytes())["records"]
    res = dict(dataset=name, seed=seed, rank_s=rank_s, triplets=getattr(ranks, "n_triplets", None), panels=panels, game=game,
               stratum=ranks.strata.tolist(), arms={})
    for method in ("gain", "coefficient"):
        old = prior["guard_"+method]
        res["arms"][method] = dict(k=old["k"], trace=old["trace"], selection_s=old["selection_s"])
    cache = {}
    methods = ("minimum", "pooled", "margin") if game == "triplet" else ("minimum", "pooled", "margin", "impurity")
    if game == "screen":
        ranks_state = json.loads((legacy_gain.RUN / f"{name}-{seed}" / "selection.json").read_bytes())
        gain_order = next(r for r in ranks_state["rows"] if r["method"]=="gain" and r["repeat"]==0)["order"]
        gain_pos = np.argsort(gain_order)
        strata = ranks.strata
        scores = dict(min_screen=strata.min(axis=0), pooled_screen=strata.mean(axis=0),
                      deletion_min_screen=strata.max(axis=0))
        scores["min_priority"] = np.sign(strata).min(axis=0)
        scores["pooled_priority"] = np.sign(strata).mean(axis=0)
        orders = {m:np.lexsort((gain_pos, -v if m.endswith("priority") else (v < 0)))
                  for m, v in scores.items()}
        res["negative"] = {m: int((v < 0).sum()) for m, v in scores.items()}
        ranks = SimpleNamespace(**orders)
        methods = tuple(orders)
    for method in methods:
        order = getattr(ranks, method)
        trace = {}
        times = []
        for k in ks:
            columns = tuple(sorted(order[:k]))
            if columns not in cache:
                t0 = perf_counter()
                pred = legacy_gain_receiver.predict(x, y, pool, width, np.asarray(columns), row["task"], seed)
                metric = benchmark.metrics(y[width], pred, task, max(np.std(y[pool]), 1e-12))
                cache[columns] = metric, perf_counter() - t0
            metric, taken = cache[columns]
            trace[k] = metric
            times.append(taken)
        k = next(k for k in ks if benchmark.eligible(trace[k], trace[p], task))
        gain_full = res["arms"]["gain"]["trace"][str(p)]
        np.testing.assert_allclose(list(gain_full.values()), list(trace[p].values()), atol=1e-10)
        res["arms"][method] = dict(order=order.tolist(), k=k, trace=trace, guard_s=sum(times))
    res["elapsed_s"] = perf_counter() - started
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["madelon"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--panels", type=int, default=8)
    parser.add_argument("--game", choices=["triplet", "tree", "conditional", "screen"], default="triplet")
    args = parser.parse_args()
    jobs = legacy_gain.jobs()
    dest = ROOT / (f"pilot-p{args.panels}" if args.game == "triplet" else f"pilot-{args.game}{args.panels}")
    dest.mkdir(exist_ok=True, parents=True)
    with threadpool_limits(1):
        for n in args.datasets:
            for s in args.seeds:
                rec = run(n, s, *jobs[n, s], args.panels, args.game)
                (dest / f"{n}-{s}.json").write_text(json.dumps(rec), encoding="utf-8")
                print(n, s, f"{rec['elapsed_s']:.2f}s, ranking {rec['rank_s']:.3f}s",
                      {m: t["k"] for m, t in rec["arms"].items()}, flush=True)


if __name__ == "__main__":
    main()
