"""Measure native XGBoost refits along Gain and compensated deletion paths."""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b, legacy_gain
from experiments.minpi_design.game import Design, Game
from experiments.minpi_design.study import OUT, load, metrics, target
from minppi_mofs.modeling import make_model


def dyadic(p):
    return sorted({p} | {2**i for i in range(p.bit_length())})


def forecasts(model, x, task):
    if task == "classification":
        return model.predict_proba(x)
    return model.predict(x)[:, None]


def run(name, seed):
    x, y, task, idx, labels = load(name, seed)
    fit = np.r_[idx["fit"], idx["vote"]]
    val, test = idx["width"], idx["test"]
    c = len(np.unique(y[fit])) if task == "classification" else 1
    problem = "multiclass" if c > 2 else "binary" if c == 2 else "regression"
    scale = max(np.std(y[fit]), 1e-12)
    grid = dyadic(x.shape[1])
    full = np.arange(x.shape[1])

    def fit_model(cols):
        start = perf_counter()
        model = make_model(task=problem, seed=seed, backend="cpu",
                           model_params=legacy_gain.CONFIG["model_params"])
        model.fit(x[np.ix_(fit, cols)], y[fit])
        pred = forecasts(model, x[np.ix_(val, cols)], task)
        score = metrics(y[val], pred, task, scale)
        return model, score, perf_counter()-start

    model, base, base_s = fit_model(full)
    device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    if device != "cpu":
        raise RuntimeError(device)
    tick = perf_counter()
    gains = model.get_booster().get_score(importance_type="gain")
    order = sorted(full, key=lambda j: (gains.get(f"f{j}", 0.), j), reverse=True)
    gain_path = {k: np.sort(order[:k]) for k in grid}
    gain_s = perf_counter()-tick

    tick = perf_counter()
    design, z = Design.fit(x[fit], "linear")
    ys = target(y[fit], task, c)
    bias = ys.mean(axis=0)
    game = Game(z, ys-bias, 1)
    design_s = perf_counter()-tick
    tick = perf_counter()
    zv = design.transform(x[val])
    proxy_base = metrics(y[val], zv @ game.head(full, False)+bias, task, scale)
    proxy_base_s = perf_counter()-tick

    # Lock predictions before calling the outer-label reader. Prefix-related
    # arms share identical native calculations, including each measured charge.
    cache = {tuple(full): (model, base, base_s)}
    locked = []
    for rank in np.random.default_rng(seed).permutation(["gain", "min", "pooled"]):
        if rank == "gain":
            path, rank_s = gain_path, gain_s
        else:
            tick = perf_counter()
            nested, _ = game.deletion_path(rank)
            path = {len(cols): cols for cols in nested if len(cols) in grid}
            rank_s = perf_counter()-tick + design_s
        tick = perf_counter()
        proxy_width = grid[-1]
        for k in grid[:-1]:
            cols = path[k]
            proxy_pred = zv[:, cols] @ game.head(cols, False)+bias
            if b.eligible(metrics(y[val], proxy_pred, task, scale), proxy_base, task):
                proxy_width = k
                break
        proxy_s = perf_counter()-tick + proxy_base_s + (design_s if rank == "gain" else 0.)
        for gate in ("direct", "proxy"):
            seconds = base_s + rank_s + (proxy_s if gate == "proxy" else 0.)
            fits = 1
            widths = []
            for k in grid:
                if gate == "proxy" and k < proxy_width:
                    continue
                cols = path[k]
                key = tuple(cols)
                if key not in cache:
                    cache[key] = fit_model(cols)
                candidate, check, spent = cache[key]
                widths.append(k)
                if k < len(full):
                    seconds += spent
                    fits += 1
                if b.eligible(check, base, task):
                    break
            tick = perf_counter()
            pred = forecasts(candidate, x[np.ix_(test, cols)], task)
            seconds += perf_counter()-tick
            rec = dict(dataset=name, seed=seed, task=task, rank=rank, search=gate,
                       p=len(full), k=k, fits=fits, elapsed_s=seconds, rank_s=rank_s, base_s=base_s,
                       proxy_width=proxy_width if gate=="proxy" else 0,
                       tried=",".join(map(str,widths)), cols=",".join(map(str,cols)))
            locked.append((rec,pred))
    tick = perf_counter()
    pred_full = forecasts(model, x[np.ix_(test, full)], task)
    locked.append((dict(dataset=name,seed=seed,task=task,rank="full",search="none",
                        p=len(full),k=len(full),fits=1,elapsed_s=base_s+perf_counter()-tick,
                        rank_s=0.,base_s=base_s,proxy_width=0,tried="",cols=",".join(map(str,full))),pred_full))
    held = labels()
    return [dict(**rec, **metrics(held, pred, task, scale)) for rec, pred in locked]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", default=legacy_gain.CONFIG["datasets"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    rows = []
    for seed in a.seeds:
        for name in a.names:
            start = perf_counter()
            rows.extend(run(name, seed))
            pd.DataFrame(rows).to_csv(OUT/(a.tag+".csv"), index=False)
            print(f"{name}/{seed} complete in {perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
