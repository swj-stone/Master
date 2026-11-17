"""Development test of native-margin compensation with actual downstream refits."""
import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import DMatrix

from experiments.minpi_compression import benchmark as b, legacy_gain
from experiments.minpi_design.game import Game
from experiments.minpi_design.native import dyadic, forecasts
from experiments.minpi_design.study import OUT, load, metrics
from minppi_mofs.modeling import make_model


def teacher_game(model, sample):
    start = perf_counter()
    dm = DMatrix(sample, nthread=1)
    booster = model.get_booster()
    a = booster.predict(dm, pred_contribs=True, approx_contribs=True,
                        strict_shape=True).astype(float)
    teacher = booster.predict(dm, output_margin=True, strict_shape=True)
    np.testing.assert_allclose(a.sum(axis=-1),teacher,rtol=2e-5,atol=1e-4)
    if a.shape[1] > 1:
        a -= a.mean(axis=1, keepdims=True)
    z = a[:,:,:-1]
    score = np.mean(np.abs(z), axis=(0,1))
    contribution_s = perf_counter()-start
    z = (z-z.mean(axis=0)).reshape(-1,z.shape[-1])
    y = z.sum(axis=1,keepdims=True)
    weights = np.sqrt(np.mean(z**2,axis=0) + 1/len(z))
    return Game(z/weights,y,1), score, contribution_s


def bisect(grid, accepted):
    lo, hi = -1, len(grid)-1
    while hi-lo > 1:
        mid = (lo+hi)//2
        if accepted(grid[mid]):
            hi = mid
        else:
            lo = mid
    return grid[hi]


def run(name,seed):
    x,y,task,idx,read_labels = load(name,seed)
    fit = np.r_[idx["fit"],idx["vote"]]
    val,test = idx["width"],idx["test"]
    c = len(np.unique(y[fit])) if task=="classification" else 1
    problem = "multiclass" if c>2 else "binary" if c==2 else "regression"
    scale = max(np.std(y[fit]),1e-12)
    full = np.arange(x.shape[1])
    grid = dyadic(len(full))

    def fit_model(cols):
        tick = perf_counter()
        model = make_model(task=problem,seed=seed,backend="cpu",
                           model_params=legacy_gain.CONFIG["model_params"])
        model.fit(x[np.ix_(fit,cols)],y[fit])
        pred = forecasts(model,x[np.ix_(val,cols)],task)
        score = metrics(y[val],pred,task,scale)
        return model,score,perf_counter()-tick

    model,base,base_s = fit_model(full)
    tick = perf_counter()
    gains = model.get_booster().get_score(importance_type="gain")
    gain_order = sorted(full,key=lambda j:(gains.get(f"f{j}",0.),j),reverse=True)
    gain_s = perf_counter()-tick
    tick = perf_counter()
    sample = np.random.default_rng(seed).choice(fit,min(512,len(fit)),replace=False)
    sample_rows = x[sample]
    sample_s = perf_counter()-tick
    game,size,contribution_s = teacher_game(model,sample_rows)
    game_s = perf_counter()-tick
    cache = {tuple(full):(model,base,base_s)}
    locked = []
    for rank in np.random.default_rng(seed).permutation(["gain","min","pooled","magnitude"]):
        tick = perf_counter()
        if rank in ("min","pooled"):
            nested, details = game.deletion_path(rank)
            path = {len(cols):cols for cols in nested if len(cols) in grid}
        else:
            order = gain_order if rank=="gain" else np.argsort(-size,kind="stable")
            path = {k:np.sort(order[:k]) for k in grid}
            details = dict(exact=0,possible=0)
        method_s = {"gain":gain_s, "magnitude":sample_s+contribution_s,
                    "min":game_s, "pooled":game_s}
        rank_s = perf_counter()-tick + method_s[rank]
        for search in ("direct","bisect"):
            charge = base_s+rank_s
            measured = []
            def accepted(k):
                nonlocal charge
                cols = path[k]
                key = tuple(cols)
                if key not in cache:
                    cache[key] = fit_model(cols)
                mdl,sc,elapsed = cache[key]
                measured.append(k)
                if k<len(full):
                    charge += elapsed
                return b.eligible(sc,base,task)
            k = (next(w for w in grid if accepted(w)) if search=="direct"
                 else bisect(grid,accepted))
            cols = path[k]
            selected = cache[tuple(cols)][0]
            tick = perf_counter()
            pred = forecasts(selected,x[np.ix_(test,cols)],task)
            charge += perf_counter()-tick
            rec = dict(dataset=name,seed=seed,task=task,rank=rank,search=search,k=k,p=len(full),
                       fits=1+sum(i<len(full) for i in measured),elapsed_s=charge,rank_s=rank_s,
                       base_s=base_s,tried=",".join(map(str,measured)),cols=",".join(map(str,cols)),**details)
            locked.append((rec,pred))
    tick = perf_counter()
    pred = forecasts(model,x[np.ix_(test,full)],task)
    locked.append((dict(dataset=name,seed=seed,task=task,rank="full",search="none",k=len(full),
                        p=len(full),fits=1,elapsed_s=base_s+perf_counter()-tick,rank_s=0.,base_s=base_s,
                        tried="",cols=",".join(map(str,full)),exact=0,possible=0),pred))
    held = read_labels()
    return [dict(**rec,**metrics(held,pred,task,scale)) for rec,pred in locked]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--names",nargs="+",default=legacy_gain.CONFIG["datasets"])
    p.add_argument("--seeds",nargs="+",type=int,default=[42])
    p.add_argument("--tag",required=True)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in args.seeds:
        for name in args.names:
            tick=perf_counter()
            rows.extend(run(name,seed))
            pd.DataFrame(rows).to_csv(OUT/(args.tag+".csv"),index=False)
            print(f"{name}/{seed} complete in {perf_counter()-tick:.1f}s",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
