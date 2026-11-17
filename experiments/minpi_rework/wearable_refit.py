"""Explore genuine one-fit and path-nominated refits after the original device study."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.minpi_rework import wearable_run as base
from experiments.minpi_rework import wearable_data as data


RUN = base.RUN.parent/"wearable-refits"
ARMS = ("mdi","min","pooled","mdi_min","mdi_pooled","exhaustive","one_mdi","one_min","one_pooled")


def method(arm, parts, unit):
    (xf,yf),(xv,yv),(xa,ya),xt = parts
    p = unit.max().item()+1
    full = (1<<p)-1
    start = perf_counter()
    teacher = base.rf.fit(xf,yf,"classification",42)
    prior = np.bincount(yf,minlength=len(teacher.empty))/len(yf)
    nominations = set()
    orders = {}
    if arm=="exhaustive":
        nominations.update(range(full+1))
    else:
        if "mdi" in arm:
            totals = np.bincount(unit,weights=teacher.model.feature_importances_,minlength=p)
            orders["mdi"] = np.argsort(totals,kind="stable")
        if "min" in arm or "pooled" in arm:
            raw = teacher.game(xv)
            a,inc = base.collapse_path_groups(raw.a,raw.incidence,unit)
            game = base.PredictiveGame.from_paths(a,inc,(np.eye(len(prior))[yv]-teacher.empty).ravel())
            if "min" in arm:
                orders["min"] = base.rank_minimum(game)[0]
            if "pooled" in arm:
                orders["pooled"] = game.ranked_path("pooled")[0]
        for order in orders.values():
            path = base.path_masks(order)
            nominations.update(path[:2] if arm.startswith("one_") else path)
    reference_grid = base.predict_grid(teacher,xa,unit,full,prior)
    reference = {s:base.brier(ya,pred) for s,pred in reference_grid.items()}
    full_scores = base.metrics(ya,reference_grid[full])
    null = reference[0]*len(ya)
    slack = 1e-12*max(1.,null)/null
    models,records = {full:teacher},{}
    fits = 1
    for s in sorted(nominations):
        cols = base.columns(unit,s)
        if s==full:
            probs = reference_grid
        else:
            models[s] = base.rf.fit(xf[:,cols],yf,"classification",42) if s else None
            fits += int(s!=0)
            probs = base.predict_grid(models[s],xa[:,cols],unit[cols],s,prior)
        records[s] = base.direct_record(ya,probs,s,reference,reference[0],p,full_scores,slack)
    selected = min((r for r in records.values() if r["eligible"]),key=lambda r:(r["k"],r["worst"],r["mask"]))["mask"]
    cols = base.columns(unit,selected)
    model = models[selected]
    intact = model.predict(xt[:,cols]) if model else np.tile(prior,(len(xt),1))
    elapsed = perf_counter()-start
    grid = base.predict_grid(model,xt[:,cols],unit[cols],selected,prior)
    np.testing.assert_array_equal(grid[selected],intact)
    return dict(k=selected.bit_count(),mask=selected,fits=fits,elapsed_s=elapsed,
                nominated=sorted(nominations),eligible=[s for s,r in records.items() if r["eligible"]]),records,grid


def worker(shard):
    if os.name=="nt":
        if not ctypes.WinDLL("kernel32").SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(1<<(2+4*shard))):
            raise OSError("Affinity failed")
    jobs = [(name,i) for name in data.SOURCES for i in range(data.SOURCES[name][2])]
    with threadpool_limits(1):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(100,6))
        y = np.arange(len(x))%3
        demo = ((x[:40],y[:40]),(x[40:60],y[40:60]),(x[60:85],y[60:85]),x[85:])
        for label in ("one_min","one_pooled"):
            method(label,demo,np.repeat(range(3),2))
        for name,pos in jobs[shard::3]:
            alarm = Timer(3600,lambda:os._exit(124))
            alarm.daemon = True
            alarm.start()
            try:
                start = perf_counter()
                key = f"{name}-{pos}"
                oldfolder = base.RUN/key
                oldlock = json.loads((data.OUT/f"lock-{name}.json").read_bytes())["receipts"]
                assert data.digest(oldfolder/"selection.json")==oldlock[key]
                orig = json.loads((oldfolder/"selection.json").read_bytes())
                x,y,unit,idx,_ = data.fold(name,pos)
                assert (y[idx["test"]]==-1).all()
                parts = (*[(x[idx[k]],y[idx[k]]) for k in ("fit","vote","width")],x[idx["test"]])
                folder = RUN/key
                folder.mkdir(parents=True,exist_ok=True)
                rec,pred = {},{}
                for arm in np.random.default_rng(4001+pos).permutation(ARMS):
                    r,book,grid = method(str(arm),parts,unit)
                    for s,v in book.items():
                        reference = orig["refit"][str(s)]
                        assert v["eligible"]==reference["eligible"]
                        np.testing.assert_allclose([v["auc"],v["acc"],*v["risk"]],
                                                   [reference["auc"],reference["acc"],*reference["risk"]],atol=1e-12)
                    if arm=="exhaustive":
                        assert r["mask"]==orig["arms"]["refit_all"]
                    rec[str(arm)] = r
                    for s,out in grid.items():
                        keyarray = f"m{r['mask']}_{s}"
                        if keyarray in pred:
                            np.testing.assert_array_equal(pred[keyarray],out)
                        pred[keyarray] = out
                np.savez_compressed(folder/"forecasts.npz",**pred)
                (folder/"selection.json").write_text(json.dumps(dict(arms=rec,initial=oldlock[key],
                    forecast=data.digest(folder/"forecasts.npz")),indent=1),encoding="utf8")
                print(key,f"{perf_counter()-start:.1f}s",flush=True)
            finally:
                alarm.cancel()


def collect():
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker,range(3)))
    locks = {f"{name}-{i}":data.digest(RUN/f"{name}-{i}"/"selection.json")
             for name in data.SOURCES for i in range(data.SOURCES[name][2])}
    (data.OUT/"refit-lock.json").write_text(json.dumps(dict(receipts=locks,source=data.digest(__file__),
        protocol=data.digest(Path(__file__).with_name("wearable_refit_protocol.md"))),indent=2),encoding="utf8")


def score():
    from experiments.minpi_rework.wearable_score import lock_all
    initial = lock_all()
    locks = json.loads((data.OUT/"refit-lock.json").read_bytes())["receipts"]
    assert set(locks)==set(initial)
    rows = []
    for key,lock in sorted(locks.items()):
        name,pos = key.rsplit("-",1)
        folder,old = RUN/key,base.RUN/key
        assert data.digest(folder/"selection.json")==lock
        rec = json.loads((folder/"selection.json").read_bytes())
        assert rec["initial"]==initial[key] and data.digest(folder/"forecasts.npz")==rec["forecast"]
        orig = json.loads((old/"selection.json").read_bytes())
        _,truth,subject,unit = data.read(name)
        ix = np.flatnonzero(np.isin(subject,orig["people"]["test"]))
        active = truth[ix]>=0
        y = truth[ix][active]
        p = int(unit.max()+1)
        with np.load(folder/"forecasts.npz",allow_pickle=False) as f,np.load(old/"forecasts.npz",allow_pickle=False) as oldf:
            ref = {s:base.brier(y,oldf[f"frozen_{s}"][active]) for s in range(1<<p)}
            full_auc,full_acc = base.metrics(y,oldf[f"frozen_{(1<<p)-1}"][active])
            for arm,r in rec["arms"].items():
                m = r["mask"]
                aset = {s:f[f"m{m}_{s}"][active] for s in base.submasks(m)}
                auc,acc = base.metrics(y,aset[m])
                worst = max(base.curve({s:base.brier(y,a) for s,a in aset.items()},m,ref,ref[0],p))
                rows.append(dict(dataset=name,fold=int(pos),method=arm,k=r["k"],mask=m,fits=r["fits"],
                     runtime_s=r["elapsed_s"],auc=auc,acc=acc,delta_auc=auc-full_auc,delta_acc=acc-full_acc,
                     held_worst=worst,held_joint=bool(worst<=.02+1e-12 and auc>=full_auc-.01 and acc>=full_acc-.01)))
    return pd.DataFrame(rows)


if __name__=="__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("action",choices=["collect","score"])
    if cmd.parse_args().action=="collect":
        collect()
    else:
        h = score()
        h.to_csv(data.OUT/"refit-held.csv",index=False)
        q = h.groupby(["dataset","method"]).agg(n=("k","size"),mean_devices=("k","mean"),fits=("fits","mean"),
            median_s=("runtime_s","median"),auc=("auc","mean"),acc=("acc","mean"),held_joint=("held_joint","sum"))
        q.to_csv(data.OUT/"refit-summary.csv")
        print(q.to_string())
