"""Retrospective mechanism comparison on the four already released wide sources."""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark
from experiments.minpi_rework import predictive_reachable as reach
from experiments.minpi_rework import quota_coalition_rf as forest
from experiments.minpi_rework import wide_veto_data as data, wide_veto_run as wide
from experiments.minpi_rework.predictive_guard import margin_path
from experiments.minpi_rework.predictive_mask_study import centered, curve
from minppi_mofs.predictive_quota import PredictiveGame

RUN = benchmark.ROOT / "results/minpi_rework/wide-veto-ablation"
OUT = benchmark.ROOT / "thesis-paper/evidence/minpi_rework/wide-veto-ablation"


def rankings(voting, game, target, model, min_order):
    args = wide.args_for(game)
    weights, tolerance = args[-2:]
    orders = {"mdi": np.argsort(model.model.feature_importances_, kind="stable"),
              "min": np.array(min_order)}
    for name, values in (("pooled", weights.mean(axis=0, keepdims=True)), ("intact", weights[-1:])):
        orders[name], _ = wide.rank("coalesced_straight", (*args[:-2], values, tolerance))
    marginal = (voting.a @ voting.incidence).toarray()
    ablated = 2 * ((target - voting.full) @ marginal) + np.einsum("ij,ij->j", marginal, marginal)
    orders["static"] = np.argsort(ablated, kind="stable")
    return orders


def paths(model, calibration, game, y, task, scale, orders):
    """Inspect deepest feasible support and all depths that tie its physical width."""
    selected = {}
    budget = None
    p = calibration.p
    size_factor = p/np.arange(1, p+1)
    for name, order in orders.items():
        preds = margin_path(model, calibration, order)
        full = benchmark.metrics(y, preds[0], task, scale)
        risk = (game.path_curves(order)[:, 1:] * size_factor).max(axis=1)
        quota = .02 * game.null + 1e-12 * max(1., game.null)
        tau = reach.retirement(model.sets, order)
        width = p - np.cumsum(np.bincount(tau, minlength=p+1))
        options = []
        for d in np.flatnonzero(risk <= quota)[::-1]:
            if options and width[d] > width[options[0]]:
                break
            if benchmark.eligible(benchmark.metrics(y, preds[d], task, scale), full, task):
                options.append(d)
        nominal_depth = options[0]
        canonical_depth = min(options, key=lambda d: (risk[d], -d))
        if name == "mdi":
            budget = p-nominal_depth
        for role, d in (("nominal", nominal_depth), ("canonical", canonical_depth), ("fixed", p-budget)):
            selected[f"{name}_{role}"] = dict(path=name, columns=np.sort(order[d:]).tolist(),
                physical=np.flatnonzero(tau > d).tolist(), guard=float(risk[d]/max(game.null, 1e-100)))
    for name in ("min", "pooled", "intact", "static"):
        for role in ("nominal", "canonical"):
            pair = [selected[f"{method}_{role}"] for method in ("mdi", name)]
            if role == "canonical":
                best = min(pair, key=lambda a: (len(a["physical"]), a["guard"]))
            else:
                best = min(pair, key=lambda a: len(a["columns"]))
            selected[f"portfolio_{name}_{role}"] = best
    return selected


def collect_job(name, seed, parent, parent_reach):
    x, y, task, idx = data.split(name, seed)
    held = y[idx["test"]]
    assert (held == -1).all() if task == "classification" else np.isnan(held).all()
    model = forest.fit(x[idx["fit"]], y[idx["fit"]], task, seed)
    vote = model.game(x[idx["vote"]])
    vote_y = centered(model, y[idx["vote"]], task)
    vg = PredictiveGame.from_paths(vote.a, vote.incidence, vote_y)
    orders = rankings(vote, vg, vote_y, model, parent["path"])
    cal = model.game(x[idx["width"]])
    cy = y[idx["width"]]
    cg = PredictiveGame.from_paths(cal.a, cal.incidence, centered(model, cy, task))
    arms = paths(model, cal, cg, cy, task, max(np.std(y[idx["fit"]]), 1e-12), orders)
    for which in ("mdi", "min", "portfolio_min"):
        assert arms[which+"_nominal"]["columns"] == parent["sets"]["portfolio" if which=="portfolio_min" else which]
        assert arms[which+"_canonical"]["physical"] == parent_reach["records"][which+"_canonical"]["canonical"]
    return dict(dataset=name, seed=seed, task=task, p=x.shape[1], splits=parent["splits"], arms=arms)


def collect():
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    locks = [json.loads((folder / "lock.json").read_bytes())["receipts"] for folder in (wide.OUT, reach.OUT)]
    completed = {}
    start = perf_counter()
    for key, digest in locks[0].items():
        source = wide.ROOT/key/"selection.json"
        other = reach.RUN/key/"selection.json"
        assert benchmark.digest(source) == digest and benchmark.digest(other) == locks[1][key]
        name, raw_seed = key.rsplit("-", 1)
        receipt = collect_job(name, int(raw_seed), json.loads(source.read_bytes()), json.loads(other.read_bytes()))
        path = RUN/(key+".json")
        path.write_text(json.dumps(receipt), encoding="utf8")
        completed[key] = benchmark.digest(path)
        print(f"{len(completed)}/20 {key} elapsed {perf_counter()-start:.1f}s", flush=True)
    assert len(completed) == 20
    (OUT/"lock.json").write_text(json.dumps(completed, indent=2), encoding="utf8")


def score():
    completed = json.loads((OUT/"lock.json").read_bytes())
    assert len(completed) == 20
    rows = []
    for key, digest in completed.items():
        path = RUN/(key+".json")
        assert benchmark.digest(path) == digest
        receipt = json.loads(path.read_bytes())
        name, seed = receipt["dataset"], receipt["seed"]
        x, y, task, idx = data.partition(name, seed)
        assert {k: v.tolist() for k, v in idx.items()} == receipt["splits"]
        model = forest.fit(x[idx["fit"]], y[idx["fit"]], task, seed)
        test_x, test_y = x[idx["test"]], y[idx["test"]]
        raw = model.game(test_x)
        g = PredictiveGame.from_paths(raw.a, raw.incidence, centered(model, test_y, task))
        scale = max(np.std(y[np.r_[idx["fit"], idx["vote"], idx["width"]]]), 1e-12)
        scores = {}
        for arm, record in receipt["arms"].items():
            cols = tuple(record["physical"])
            if cols not in scores:
                pred = model.predict(test_x[:, list(cols)], list(cols))
                risk = curve(g, list(cols))
                scores[cols] = dict(worst=risk.max(), mean_size=risk.mean(), intact=risk[-1],
                                     **benchmark.metrics(test_y, pred, task, scale))
            rows.append(dict(dataset=name, seed=seed, arm=arm, path=record["path"],
                p=receipt["p"], width=len(record["columns"]), physical=len(cols), **scores[cols]))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"scores.csv", index=False)
    frame.groupby(["dataset", "arm"]).mean(numeric_only=True).to_csv(OUT/"source-means.csv")
    paired = []
    for scope, prefix in (("fixed", ""), ("canonical", "portfolio_"), ("nominal", "portfolio_")):
        primary = frame[frame.arm == prefix+"min_"+scope].set_index(["dataset", "seed"])
        for name in ("pooled", "intact", "static", "mdi"):
            arm = ("" if name=="mdi" else prefix) + name + "_" + scope
            other = frame[frame.arm == arm].set_index(["dataset", "seed"]).loc[primary.index]
            for column in ("width", "physical", "worst", "auc", "accuracy", "r2", "scaled_mae"):
                delta = (primary[column]-other[column]).dropna().groupby(level=0).mean()
                draws = np.random.default_rng(673).choice(delta, (20000, len(delta)), replace=True).mean(axis=1)
                paired.append(dict(scope=scope, reference=name, metric=column, sources=len(delta),
                    mean_delta=delta.mean(), lo=np.quantile(draws, .025), hi=np.quantile(draws, .975),
                    **delta.to_dict()))
    pd.DataFrame(paired).to_csv(OUT/"contrasts.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["collect", "score"])
    with threadpool_limits(1):
        (collect if parser.parse_args().operation=="collect" else score)()
