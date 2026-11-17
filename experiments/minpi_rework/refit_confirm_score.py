"""Score both sealed prediction families. This entry point is separate from blind training."""
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,mean_absolute_error,r2_score,roc_auc_score

from experiments.minpi_rework import refit_confirm as frozen


def score_arrays(y,pred,task,scale,w):
    if task=="classification":
        return dict(accuracy=accuracy_score(y,pred.argmax(axis=1),sample_weight=w),
            auc=np.mean([roc_auc_score(y==j,pred[:,j],sample_weight=w) for j in range(pred.shape[1])]))
    mae=mean_absolute_error(y,pred,sample_weight=w)
    return dict(mae=mae,scaled_mae=mae/scale,r2=r2_score(y,pred,sample_weight=w))


def max_loss(y,pred,parts,offset,cols,task,weights):
    target=np.eye(len(offset))[y] if task=="classification" else y[:,None]
    err=target-np.reshape(pred,target.shape)
    norm=max(np.finfo(float).eps,np.average(((target-offset)**2).sum(axis=1),weights=weights))
    intact=np.average((err*err).sum(axis=1),weights=weights)
    missing=np.array([np.average(((err+parts[:,j])**2).sum(axis=1),weights=weights) for j in cols])
    return dict(intact=intact/norm,worst=max(intact,np.max(missing))/norm)


def intervals(frame):
    all_effects=[]
    groupcols=["workflow","policy","weight","task"]
    for subname,subset in [*frame.groupby("panel"),("pooled",frame)]:
        for (workflow,policy,weight,task),group in subset.groupby(groupcols):
            arm="width" if workflow=="refit" else "exchange_min"
            refs=["greedy","energy","gain","full"] if workflow=="refit" else ["exchange_mean","exchange_point","energy","full"]
            for ref in refs:
                cols=["k","retained","selection_s","deployment_s","accuracy","auc","r2","mae","scaled_mae","intact","worst"]
                for metric in cols:
                    if metric not in group: continue
                    wide=group.pivot(index=["source","seed"],columns="method",values=metric)[[arm,ref]].dropna()
                    if len(wide)==0: continue
                    assert len(wide)==len(group[["source","seed"]].drop_duplicates())
                    ratio=metric in ["k","retained","selection_s","deployment_s"]
                    delta=np.log(wide[arm]/wide[ref]) if ratio else wide[arm]-wide[ref]
                    values=delta.groupby(level=0).mean().to_numpy()
                    rng=np.random.default_rng(377)
                    boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
                    stats=np.r_[values.mean(),np.quantile(boot,[.025,.975])]
                    if ratio: stats=np.exp(stats)
                    all_effects.append(dict(panel=subname,workflow=workflow,policy=policy,weight=weight,task=task,
                        candidate=arm,reference=ref,metric=metric,estimate=stats[0],lo=stats[1] if len(values)>1 else np.nan,
                        hi=stats[2] if len(values)>1 else np.nan,sources=len(values),splits=len(wide)))
    return pd.DataFrame(all_effects)


def score(expected_seal):
    dest=frozen.DEST
    sealed=dest/"sealed.json"
    if frozen.hash_file(sealed)!=expected_seal: raise ValueError("Wrong seal digest")
    meta=json.loads(sealed.read_bytes())
    assert meta["preflight"]==frozen.hash_file(frozen.CONFIG)
    bundle=dest/"forecasts.zip"
    assert meta["bundle"]==frozen.hash_file(bundle)
    keys={f"{n}-{s}" for n,s in frozen.JOBS}
    assert set(meta["files"])==keys
    rows=[]
    with ZipFile(bundle) as archive:
        expected={key+"/"+f for key in keys for f in [*frozen.FILES,"receipt.json"]}
        assert set(archive.namelist())==expected
        # Every member checked before accessing even the first held-out target.
        for key,files in meta["files"].items():
            assert set(files)=={*frozen.FILES,"receipt.json"}
            for f,digest in files.items():
                assert hashlib.sha256(archive.read(key+"/"+f)).hexdigest()==digest
        for n,s in frozen.JOBS:
            key=f"{n}-{s}"; provider=frozen.panel if n in frozen.panel.SOURCES else frozen.wide
            panel="original8" if provider==frozen.panel else "wide5"
            x,y,task,idx=provider._partition(n,s)
            test=idx["test"]; truth=y[test]
            scale=max(float(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]])),1e-12)
            rec=json.loads(archive.read(key+"/receipt.json"))
            assert rec["test"]==test.tolist() and rec["task"]==task and rec["p"]==x.shape[1]
            weights={"rows":None}
            if n in {"mice","anuran","ctloc"}:
                groups=provider._data(n)[3][test]
                _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
                weights["groups"]=1/counts[inverse]
            def load(f):
                with np.load(BytesIO(archive.read(key+"/"+f))) as a: return {k:a[k] for k in a.files}
            refit,normal,terms=(load(f) for f in frozen.FILES)
            with TemporaryDirectory() as tmp:
                official={}
                for file in ("refit.npz","outage.npz"):
                    copy=Path(tmp)/file
                    copy.write_bytes(archive.read(key+"/"+file))
                    official[file]=provider.score(n,s,copy,meta["files"][key][file])
            for weight,w in weights.items():
                common=dict(panel=panel,source=n,seed=s,task=task,weight=weight,p=rec["p"])
                for arm in frozen.REFIT:
                    for policy in ("adaptive","fixed"):
                        k=len(rec["refit"][arm]["support" if policy=="adaptive" else "fixed"])
                        times={f:float(np.median([r[f] for r in rec["clocks"][arm]]))
                               for f in ("selection_s","deployment_s")} if policy=="adaptive" else {}
                        raw=refit[arm+"_"+policy]
                        metrics=score_arrays(truth,raw,task,scale,w)
                        if weight=="rows":
                            for field,v in metrics.items():
                                np.testing.assert_allclose(v,official["refit.npz"][arm+"_"+policy][field],atol=1e-12)
                        rows.append(dict(**common,workflow="refit",policy=policy,method=arm,k=k,retained=k/rec["p"],
                            **metrics,**times))
                for arm in frozen.OUTAGE:
                    cols=rec["outage"][arm]; k=len(cols)
                    raw=normal[arm]
                    risks=max_loss(truth,raw,terms["centered_shapes"],terms["offset"],cols,task,w)
                    metrics=score_arrays(truth,raw,task,scale,w)
                    if weight=="rows":
                        for field,v in metrics.items():
                            np.testing.assert_allclose(v,official["outage.npz"][arm][field],atol=1e-12)
                    rows.append(dict(**common,workflow="outage",policy="fixed_half",method=arm,k=k,retained=k/rec["p"],
                        **metrics,**risks))
    data=pd.DataFrame(rows)
    data.to_csv(dest/"heldout_scores.csv",index=False)
    intervals(data).to_csv(dest/"heldout_contrasts.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sealed-sha256",required=True)
    score(p.parse_args().sealed_sha256)
