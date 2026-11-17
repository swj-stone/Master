"""Measure full supervised selection while both paths share the same short-circuit acceptance."""
import argparse
import ctypes
from itertools import product
import json
import os
from threading import Timer
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_rework.predictive_polynomial_timing import OUT, rank_interval, warm, wide
from minppi_mofs.size_extrema import rank_arrays


def pipeline(parts, task, seed, method, production=False):
    wall = perf_counter()
    (xf,yf),(xv,yv),(xa,ya),test = parts
    model = wide.rf.fit(xf,yf,task,seed)
    fit = perf_counter()
    mdi = np.argsort(model.model.feature_importances_,kind="stable")
    if method!="mdi":
        raw = model.game(xv)
        pg = wide.pq.PredictiveGame.from_paths(raw.a,raw.incidence,wide.timing.centered(model,yv,task))
        args = wide.args_for(pg)
    ready = perf_counter()
    interval = rank_arrays if production else rank_interval
    order = mdi if method=="mdi" else (interval(*args) if method=="interval" else wide.rank(method,args))[0]
    ranked = perf_counter()
    cal = model.game(xa)
    cg = wide.pq.PredictiveGame.from_paths(cal.a,cal.incidence,wide.timing.centered(model,ya,task))
    p = len(order)
    choices = []
    for ranking in ((mdi,) if method=="mdi" else (mdi,order)):
        scores = wide.guard.margin_path(model,cal,ranking)
        risk = cg.path_curves(ranking)[:,1:]*(p/np.arange(1,p+1))
        pass_loss = risk.max(axis=1)<=.02*cg.null+1e-12*max(1.,cg.null)
        choices.append((ranking,scores,pass_loss))
    scale = max(np.std(yf),1e-12)
    def native(row):
        return wide.b.metrics(ya,row if task=="classification" else row[:,0],task,scale)
    full = native(choices[0][1][0])
    count = 1
    for width in range(p+1):
        for ranking,scores,good in choices:
            drop = p-width
            if good[drop]:
                if drop:
                    count += 1
                    if not wide.b.eligible(native(scores[drop]),full,task):
                        continue
                cols = np.sort(ranking[drop:])
                pred = model.predict(test[:,cols],cols)
                end = perf_counter()
                return order,cols,pred,dict(total_s=end-wall,fit_s=fit-wall,prepare_s=ready-fit,
                                             rank_s=ranked-ready,accept_predict_s=end-ranked,native_calls=count)
    raise AssertionError("Full predictor failed acceptance")


def measure(production, prefix, mdi_only=False):
    warm()
    rng = np.random.default_rng(488)
    tiny = rng.normal(size=(35,6))
    parts = (tiny[:15],tiny[:15,0]),(tiny[15:23],tiny[15:23,0]),(tiny[23:29],tiny[23:29,0]),tiny[29:]
    pipeline(parts,"regression",92,"interval",production)
    strongest = pd.read_csv(OUT/"paired.csv").set_index(["dataset","seed"]).baseline
    rows = []
    lock = json.loads((wide.OUT/"lock.json").read_bytes())["receipts"]
    for (name,seed),baseline in strongest.items():
        alarm = Timer(3600, lambda:os._exit(124))
        alarm.daemon = True
        alarm.start()
        try:
            folder = wide.ROOT/f"{name}-{seed}"
            receipt = folder/"selection.json"
            assert wide.b.digest(receipt)==lock[folder.name]
            rec = json.loads(receipt.read_bytes())
            assert wide.b.digest(folder/"forecasts.npz")==rec["forecast"]
            with np.load(folder/"forecasts.npz",allow_pickle=False) as arch:
                targets = {key:arch[key] for key in ("portfolio","mdi")}
            x,y,task,idx = wide.data.split(name,seed)
            parts = (*[(x[idx[k]],y[idx[k]]) for k in ("fit","vote","width")],x[idx["test"]])
            rng = np.random.default_rng(6118+seed)
            methods = list(rng.permutation(["interval","mdi" if mdi_only else baseline]))
            for repeat in range(3):
                for method in methods:
                    order,cols,forecast,time = pipeline(parts,task,seed,method,production)
                    key = "mdi" if method=="mdi" else "portfolio"
                    if method!="mdi":
                        np.testing.assert_array_equal(order,rec["path"])
                    np.testing.assert_array_equal(cols,rec["sets"][key])
                    np.testing.assert_array_equal(forecast,targets[key])
                    rows.append(dict(dataset=name,seed=seed,method=method,repeat=repeat,k=len(cols),**time))
                methods.reverse()
            pd.DataFrame(rows).to_csv(OUT/(prefix+"-repeats.csv"),index=False)
            print(name,seed,flush=True)
        finally:
            alarm.cancel()


def summarize(prefix):
    data = pd.read_csv(OUT/(prefix+"-repeats.csv"))
    assert len(data)==120 and data.groupby(["dataset","seed","method"]).size().eq(3).all()
    med = data.groupby(["dataset","seed","method"]).median(numeric_only=True).reset_index()
    left = med[med.method=="interval"].set_index(["dataset","seed"])
    right = med[med.method!="interval"].set_index(["dataset","seed"])
    paired = pd.DataFrame(dict(baseline=right.method,interval_s=left.total_s,baseline_s=right.total_s,
                               ratio=left.total_s/right.total_s,native_calls=left.native_calls))
    if prefix!="mdi":
        paired["rank_ratio"] = left.rank_s/right.rank_s
    else:
        paired["k_portfolio"],paired["k_mdi"] = left.k,right.k
    paired.to_csv(OUT/(prefix+"-paired.csv"))
    means = np.log(paired.ratio).groupby("dataset").mean()
    boot = np.array([means.to_numpy()[list(ix)].mean() for ix in product(range(len(means)),repeat=len(means))])
    print(paired.groupby("dataset")[["interval_s","baseline_s"]].median().join(np.exp(means)))
    print("whole-source summary",np.exp(means.mean()),np.exp(np.quantile(boot,[.025,.975])))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--production",action="store_true",help="Use the stricter floating-point dormant-column check")
    parser.add_argument("--mdi",action="store_true",help="Compare production portfolio to fresh MDI-only fits")
    args = parser.parse_args()
    prefix = "mdi" if args.mdi else "production" if args.production else "pipeline"
    if os.name=="nt" and not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<10)):
        raise OSError("Affinity failed")
    with threadpool_limits(1):
        measure(args.production or args.mdi,prefix,args.mdi)
    summarize(prefix)
