"""Structural input support and sealed path re-evaluation on frozen RF benchmarks."""
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import wide_veto_run as wide,wide_veto_data,validation_predictive
from experiments.minpi_rework import predictive_confirmation as seven,predictive_guard as guard
from experiments.minpi_rework import quota_coalition_rf as rf
from experiments.minpi_rework.predictive_mask_study import centered
from minppi_mofs.predictive_quota import PredictiveGame

RUN=b.ROOT/"results/minpi_rework/predictive-reachable"
OUT=b.ROOT/"thesis-paper/evidence/minpi_rework/predictive-reachable"


def retirement(groups,order):
    """Last deletion depth at which each feature ceases to be reachable."""
    ranks=np.argsort(order)+1
    last=np.zeros(len(order),int)
    for h in groups:
        cols=np.asarray(h,int)
        last[cols]=np.maximum(last[cols],ranks[cols].min())
    return last


def structural(groups,p,cols):
    keep=np.zeros(p,bool)
    keep[cols]=True
    referenced=np.zeros(p,bool)
    for h in groups:
        fields=np.asarray(h,int)
        if keep[fields].all():
            referenced[fields]=True
    return np.flatnonzero(referenced)


def confirm_paths(model,x,y,idx,task,voting):
    raw=model.game(x[idx["vote"]])
    target=centered(model,y[idx["vote"]],task)
    game=PredictiveGame.from_paths(raw.a,raw.incidence,target)
    initial=(raw.a@raw.incidence).toarray()
    residual=target-raw.full
    costs=2*(residual@initial)+np.einsum("ij,ij->j",initial,initial)
    return {**{rule:game.ranked_path(rule)[0] for rule in ("min","intact","pooled")},
            "static":np.argsort(costs,kind="stable"),"mdi":voting}


def job(panel,source,name,seed,base,lock):
    oldpath=base/f"{name}-{seed}"/"selection.json"
    assert b.digest(oldpath)==lock[f"{name}-{seed}"]
    old=json.loads(oldpath.read_bytes())
    x,y,task,idx=source.split(name,seed)
    assert (y[idx["test"]]==-1).all() if task=="classification" else np.isnan(y[idx["test"]]).all()
    model=rf.fit(x[idx["fit"]],y[idx["fit"]],task,seed)
    p=x.shape[1]
    mdi=np.argsort(model.model.feature_importances_,kind="stable")
    orders=({"min":np.asarray(old["path"],int),"mdi":mdi} if panel=="wide"
            else confirm_paths(model,x,y,idx,task,mdi))
    rows=idx["width"]
    val=model.game(x[rows])
    game=PredictiveGame.from_paths(val.a,val.incidence,centered(model,y[rows],task))
    truth_scale=max(float(np.std(y[idx["fit"]])),1e-12)
    radius=.02*game.null+1e-12*max(1.,game.null)
    rec={}
    pred={}
    tolerance=0.
    def save(arm,rawcols,cols,rank,risk,depth,feasible):
        nonlocal tolerance
        rawcols=np.asarray(rawcols,int)
        cols=np.asarray(cols,int)
        mask=np.isin(np.arange(p),rawcols)
        cmask=np.isin(np.arange(p),cols)
        np.testing.assert_array_equal(structural(model.sets,p,rawcols),cols)
        np.testing.assert_array_equal(structural(model.sets,p,cols),cols)
        # Combinatorial predictor/risk equality for all acquisition masks.
        for h in model.sets:
            ix=list(h)
            assert mask[ix].all()==cmask[ix].all()
        curves=game.conditional_risk(mask),game.conditional_risk(cmask)
        np.testing.assert_allclose(*curves,rtol=1e-9,atol=1e-9)
        for sample in (rows,idx["test"]):
            a=model.predict(x[np.ix_(sample,rawcols)],rawcols)
            c=model.predict(x[np.ix_(sample,cols)],cols)
            err=float(abs(a-c).max())
            tolerance=max(tolerance,err)
            np.testing.assert_array_equal(a,c)
        rec[arm]=dict(nominal=rawcols.tolist(),canonical=cols.tolist(),path=rank,
            risk=float(risk),depth=int(depth),feasible_depths=int(feasible))
        pred[arm]=c

    save("full",np.arange(p),structural(model.sets,p,np.arange(p)),"full",0,0,1)
    for rank,order in orders.items():
        allpred=guard.margin_path(model,val,order)
        passes=guard.passing_native(y[rows],allpred,task,truth_scale)
        risk=(game.path_curves(order)[:,1:]*(p/np.arange(1,p+1))).max(axis=1)
        allowed=np.flatnonzero(passes&(risk<=radius))
        d0=int(allowed[-1])
        oldkeep=(old["sets"][rank] if panel=="wide" else old["records"][rank+"_min"]["support"])
        np.testing.assert_array_equal(np.sort(order[d0:]),oldkeep)
        tau=retirement(model.sets,order)
        width=p-np.cumsum(np.bincount(tau,minlength=p+1))
        candidates=allowed[width[allowed]==width[allowed].min()]
        # All tied widths have the same canonical support by nestedness.
        d=int(min(candidates,key=lambda i:(risk[i],-i)))
        np.testing.assert_array_equal(tau>d,tau>d0)
        for suffix,depth in (("original",d0),("canonical",d)):
            rawcols=np.sort(order[depth:])
            save(rank+"_"+suffix,rawcols,np.flatnonzero(tau>depth),rank,risk[depth]/max(game.null,1e-100),
                 depth,len(allowed))
    for rank in orders:
        if rank=="mdi":
            continue
        named="portfolio_"+rank
        for suffix in ("original","canonical"):
            choices=["mdi_"+suffix,rank+"_"+suffix]
            if suffix=="canonical":
                winner=min(choices,key=lambda m:(len(rec[m]["canonical"]),rec[m]["risk"],int(not m.startswith("mdi_"))))
            else:
                if panel=="wide":
                    winner=min(choices,key=lambda m:len(rec[m]["nominal"]))
                else:
                    locked=old["records"][f"mdi_plus_{rank}_min"]["winner"].removesuffix("_min")
                    winner=locked+"_"+suffix
            rec[named+"_"+suffix]=dict(rec[winner],winner=winner)
            pred[named+"_"+suffix]=pred[winner]
    folder=RUN/f"{name}-{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"forecasts.npz",**pred)
    info=dict(panel=panel,name=name,seed=seed,task=task,p=p,base_digest=b.digest(oldpath),
        outer=idx["test"].tolist(),max_discrepancy=tolerance,records=rec,forecast=b.digest(folder/"forecasts.npz"))
    temp=folder/"selection.tmp"
    temp.write_text(json.dumps(info),encoding="utf8")
    temp.replace(folder/"selection.json")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock={}
    start=perf_counter()
    for panel,source,base,meta in (("seven",validation_predictive,seven.RUN,seven.OUT),
                                   ("wide",wide_veto_data,wide.ROOT,wide.OUT)):
        parent=json.loads((meta/"lock.json").read_bytes())["receipts"]
        for name in source.SOURCES:
            for seed in range(42,47):
                key=f"{name}-{seed}"
                job(panel,source,name,seed,base,parent)
                lock[key]=b.digest(RUN/key/"selection.json")
            print(panel,name,len(lock),f"{perf_counter()-start:.0f}s",flush=True)
    assert len(lock)==55
    stamp={p.name:b.digest(p) for p in (Path(__file__),Path(__file__).with_name("predictive_reachable_protocol.md"),
        wide.OUT/"lock.json",Path(guard.__file__),Path(rf.__file__))}
    stamp["seven_lock"]=b.digest(seven.OUT/"lock.json")
    (OUT/"lock.json").write_text(json.dumps(dict(code=stamp,receipts=lock),indent=2),encoding="utf8")
    bundle=b.ROOT/"packages/minpi-reachable-forecasts.zip"
    with ZipFile(bundle,"w",ZIP_DEFLATED) as z:
        z.write(OUT/"lock.json","lock.json")
        for key in lock:
            for fn in ("selection.json","forecasts.npz"):
                z.write(RUN/key/fn,key+"/"+fn)
    (OUT/"archive-sha256.txt").write_text(b.digest(bundle)+"  "+bundle.name+"\n",encoding="utf8")


if __name__=="__main__":
    with threadpool_limits(1):
        main()
