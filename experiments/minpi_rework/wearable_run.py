"""Sealed subject-level acquisition experiment. Collection never reads held targets."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Timer
from time import perf_counter

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from experiments.minpi_rework import quota_coalition_rf as rf
from experiments.minpi_rework import wearable_data as data
from minppi_mofs.grouped_paths import collapse_path_groups
from minppi_mofs.predictive_quota import PredictiveGame
from minppi_mofs.size_extrema import rank_minimum


RUN = data.ROOT/"results/minpi_rework/wearable-groups"


def submasks(mask):
    return [s for s in range(mask+1) if s & mask == s]


def columns(unit, mask):
    return np.flatnonzero(((1<<unit)&mask)!=0)


def predict_grid(model, x, unit, mask, prior):
    if not mask:
        return {0:np.tile(prior,(len(x),1))}
    out = {}
    for s in submasks(mask):
        cols = columns(unit,s)
        out[s] = model.predict(x[:,cols],cols)
    return out


def metrics(y,pred):
    used = np.unique(y)
    auc = [roc_auc_score(y==i,pred[:,i]) for i in used if (y!=i).any()]
    return float(np.mean(auc)),float(np.mean(y==np.argmax(pred,axis=1)))


def brier(y,probs):
    # Multiclass Brier sum over all classes per row.
    return float((np.einsum("ij,ij->i",probs,probs)-2*probs[np.arange(len(y)),y]+1).mean())


def curve(losses, mask, reference, scale, p):
    return [np.mean([losses[t&mask]-reference[t] for t in range(1<<p) if t.bit_count()==s])*p/(s*scale)
            for s in range(1,p+1)]


def direct_record(y, probs, mask, reference, scale, p, full_scores, slack):
    losses = {s:brier(y,v) for s,v in probs.items()}
    risks = curve(losses,mask,reference,scale,p)
    auc,acc = metrics(y,probs[mask])
    clean = bool(auc>=full_scores[0]-.01 and acc>=full_scores[1]-.01)
    return dict(mask=mask,k=mask.bit_count(),risk=risks,worst=float(max(risks)),auc=auc,acc=acc,
                clean=clean,eligible=bool(clean and max(risks)<=.02+slack))


def grouped(model,x,y,unit,p):
    raw = model.game(x)
    a,inc = collapse_path_groups(raw.a,raw.incidence,unit)
    target = (np.eye(len(model.empty))[y]-model.empty).ravel()
    pg = PredictiveGame.from_paths(a,inc,target)
    probs = predict_grid(model,x,unit,(1<<p)-1,None)
    ref = {s:brier(y,v) for s,v in probs.items()}
    scale = ref[0]
    assert scale>0
    for s,pred in probs.items():
        active = np.array([(s & (1<<i))!=0 for i in range(p)])
        enabled = np.asarray(inc@(~active).astype(int)).ravel()==0
        from_group = (a@enabled.astype(float)).reshape(pred.shape)+model.empty
        np.testing.assert_allclose(from_group,pred,rtol=1e-10,atol=1e-12)
        actual = pg.conditional_risk(active)[1:]*p/(np.arange(1,p+1)*pg.null)
        np.testing.assert_allclose(actual,curve(ref,s,ref,scale,p),rtol=1e-8,atol=1e-11)
    return pg,probs,ref


def path_masks(order):
    mask = (1<<len(order))-1
    path = [mask]
    for f in order:
        mask ^= 1<<int(f)
        path.append(mask)
    return path


def select_arms(mdi,minimum,frozen,refit,p):
    arms = {}
    for name,candidates,book in (("mdi",mdi,frozen),("min",minimum,frozen),
             ("portfolio",sorted(set(mdi+minimum)),frozen),("frozen_all",list(frozen),frozen),
             ("refit_all",list(refit),refit)):
        arms[name] = min((book[s] for s in candidates if book[s]["eligible"]),
                         key=lambda r:(r["k"],r["worst"],r["mask"]))["mask"]
        arms[name+"_fixed"] = min((book[s] for s in candidates if s.bit_count()==p-1),
                                  key=lambda r:(r["worst"],r["mask"]))["mask"]
    arms["refit_clean"] = min((r for r in refit.values() if r["clean"]),
                              key=lambda r:(r["k"],-r["auc"],-r["acc"],r["mask"]))["mask"]
    arms["full"] = (1<<p)-1
    return arms


def job(name, pos):
    x,y,unit,idx,people = data.fold(name,pos)
    assert np.all(y[idx["test"]]==-1)
    p = int(unit.max())+1
    full = (1<<p)-1
    xf,yf = x[idx["fit"]],y[idx["fit"]]
    xv,yv = x[idx["vote"]],y[idx["vote"]]
    xa,ya = x[idx["width"]],y[idx["width"]]
    xt = x[idx["test"]]
    c = data.SOURCES[name][3]
    np.testing.assert_array_equal(np.unique(yf),np.arange(c))
    prior = np.bincount(yf,minlength=c)/len(yf)
    start = perf_counter()
    model = rf.fit(xf,yf,"classification",42)
    fit_s = perf_counter()-start
    start = perf_counter()
    voting,_,vloss = grouped(model,xv,yv,unit,p)
    order,_ = rank_minimum(voting)
    minimum = path_masks(order)
    # Verify the selected candidate's regret against every remaining device by direct traversal.
    for s,chosen in zip(minimum,minimum[1:]):
        vals = {s^(1<<j):max(curve(vloss,s^(1<<j),vloss,vloss[0],p)) for j in range(p) if s&(1<<j)}
        assert vals[chosen]<=min(vals.values())+1e-10
    vote_s = perf_counter()-start
    start = perf_counter()
    dev_mdi = np.bincount(unit,weights=model.model.feature_importances_,minlength=p)
    mdi = path_masks(np.argsort(dev_mdi,kind="stable"))
    acceptance,base,ref = grouped(model,xa,ya,unit,p)
    full_scores = metrics(ya,base[full])
    slack = 1e-12*max(1.,acceptance.null)/acceptance.null
    frozen = {s:direct_record(ya,{t:base[t] for t in submasks(s)},s,ref,ref[0],p,full_scores,slack)
              for s in range(full+1)}
    cal_s = perf_counter()-start
    refit = {full:frozen[full]}
    models = {full:model}
    start = perf_counter()
    for s in range(full):
        cols = columns(unit,s)
        models[s] = rf.fit(xf[:,cols],yf,"classification",42) if s else None
        grid = predict_grid(models[s],xa[:,cols],unit[cols],s,prior)
        refit[s] = direct_record(ya,grid,s,ref,ref[0],p,full_scores,slack)
    refit_s = perf_counter()-start
    arms = select_arms(mdi,minimum,frozen,refit,p)
    forecasts = {f"frozen_{s}":v for s,v in predict_grid(model,xt,unit,full,prior).items()}
    for s in {s for name,s in arms.items() if name.startswith("refit")}:
        cols = columns(unit,s)
        for t,v in predict_grid(models[s],xt[:,cols],unit[cols],s,prior).items():
            forecasts[f"refit_{s}_{t}"] = v
    folder = RUN/f"{name}-{pos}"
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"forecasts.npz",**forecasts)
    rec = dict(dataset=name,fold=pos,people=people,counts={k:len(v) for k,v in idx.items()},
        classes_vote=np.unique(yv).tolist(),classes_accept=np.unique(ya).tolist(),
        rows_sha256=sha256(np.asarray(idx["test"],dtype="<i8").tobytes()).hexdigest(),
        forecast_sha256=data.digest(folder/"forecasts.npz"),full=full,arms=arms,mdi=mdi,minimum=minimum,
        frozen=frozen,refit=refit,fit_s=fit_s,vote_audit_s=vote_s,accept_audit_s=cal_s,
        additional_refit_s=refit_s,n_additional_fits=full-1)
    (folder/"selection.json").write_text(json.dumps(rec,indent=1),encoding="utf8")
    return data.digest(folder/"selection.json")


def stamp():
    from minppi_mofs import grouped_paths, size_extrema, availability_minpi, predictive_quota
    files = [Path(__file__),Path(data.__file__),Path(rf.__file__),Path(grouped_paths.__file__),
             Path(size_extrema.__file__),Path(availability_minpi.__file__),Path(predictive_quota.__file__),
             data.PINS,Path(__file__).with_name("wearable_protocol.md")]
    return {f.name:data.digest(f) for f in files}


def collect(name):
    data.OUT.mkdir(parents=True,exist_ok=True)
    lock = {}
    for pos in range(data.SOURCES[name][2]):
        start = perf_counter()
        alarm = Timer(3600,lambda:os._exit(124))
        alarm.daemon = True
        alarm.start()
        try:
            lock[f"{name}-{pos}"] = job(name,pos)
        finally:
            alarm.cancel()
        elapsed = perf_counter()-start
        print(f"{name} {pos+1}/{data.SOURCES[name][2]} {elapsed:.1f}s, ETA {elapsed*(data.SOURCES[name][2]-pos-1):.0f}s",flush=True)
    (data.OUT/f"lock-{name}.json").write_text(json.dumps(dict(code=stamp(),receipts=lock),indent=2),encoding="utf8")


if __name__=="__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("name",choices=data.SOURCES)
    args = cli.parse_args()
    with threadpool_limits(1):
        collect(args.name)
