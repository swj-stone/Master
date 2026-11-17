"""Release only the previously sealed forecasts and reconstruct held all-mask risks."""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import predictive_mc_confirmation as conf
from experiments.minpi_rework.predictive_union_control import Terms,MatchedSearch
from experiments.minpi_rework.predictive_mask_study import curve
from minppi_mofs.predictive_quota import PredictiveGame


def main():
    lock=json.loads((conf.OUT/"lock.json").read_bytes())
    assert lock["code"]==conf.build_hashes()
    assert len(lock["receipts"])==35
    rows=[]
    for key,expected in lock["receipts"].items():
        folder=conf.STORE/key
        selection=folder/"selection.json"
        assert b.digest(selection)==expected
        rec=json.loads(selection.read_bytes())
        assert rec["code"]==lock["code"]
        assert all(b.digest(folder/file)==digest for file,digest in rec["hashes"].items())
        name,seed,task=rec["dataset"],rec["seed"],rec["task"]
        _,y,observed_task,idx=conf.source._partition(name,seed)
        assert task==observed_task
        for part,indices in rec["splits"].items():
            np.testing.assert_array_equal(indices,idx[part])
        truth=y[idx["test"]]
        scale=max(float(y[np.r_[idx["fit"],idx["vote"],idx["width"]]].std()),1e-12)
        empty=np.asarray(rec["empty"])
        target=(np.eye(len(empty))[truth.astype(int)] if task=="classification" else truth[:,None])
        centered=(target-empty).ravel()
        a=sparse.load_npz(folder/"increments.npz")
        inc=sparse.load_npz(folder/"incidence.npz")
        exact=MatchedSearch(Terms.build(a,inc,centered))
        other=PredictiveGame.from_paths(a,inc,centered)
        p=inc.shape[1]
        with np.load(folder/"forecasts.npz",allow_pickle=False) as predictions:
            for method,selected in rec["supports"].items():
                cols=np.asarray(selected,int)
                val=exact.curve(cols)
                checked=curve(other,cols)
                np.testing.assert_allclose(val,checked,rtol=3e-8,atol=1e-10)
                dropped=np.ones(p,int)
                dropped[cols]=0
                active=np.ravel(inc@dropped)==0
                rebuilt=np.asarray(a[:,active].sum(axis=1)).reshape(len(truth),-1)+empty
                forecast=predictions[method]
                verified=rebuilt if task=="classification" else rebuilt.ravel()
                np.testing.assert_allclose(forecast,verified,rtol=3e-8,atol=1e-8)
                vals=b.metrics(truth,forecast,task,scale)
                rows.append(dict(dataset=name,seed=seed,task=task,method=method,k=len(selected),
                    worst=val.max(),worst_positive=max(0.,val.max()),mean=val.mean(),intact=val[-1],
                    violation=val.max()>.02+1e-9,**vals))
    frame=pd.DataFrame(rows)
    frame.to_csv(conf.OUT/"released-scores.csv",index=False)
    x=frame[frame.method=="exact"].set_index(["dataset","seed"])
    contrasts=[]
    for m in ["mc8","mc32","mc128","direct8","direct32","direct128"]:
        y=frame[frame.method==m].set_index(["dataset","seed"])
        for metric in ("k","worst","worst_positive","mean","intact","violation","auc","accuracy","scaled_mae","r2"):
            a=x.loc[y.index,metric].astype(float)
            b_=y[metric].astype(float)
            dif=(b_-a).dropna().groupby(level="dataset").mean()
            if dif.empty:
                continue
            draws=np.random.default_rng(171126).choice(dif.to_numpy(),size=(20000,len(dif)),replace=True).mean(axis=1)
            contrasts.append(dict(method=m,metric=metric,sources=len(dif),delta_sample_minus_exact=dif.mean(),
                lower=np.quantile(draws,.025),upper=np.quantile(draws,.975),per_source=json.dumps(dif.to_dict())))
    out=pd.DataFrame(contrasts)
    out.to_csv(conf.OUT/"released-contrasts.csv",index=False)
    print(frame[frame.method.isin(["exact","mc8","mc32","mc128"])].groupby("method")
          [["k","worst","worst_positive","violation","auc","accuracy","r2","scaled_mae"]].mean().to_string())
    print(out[out.metric=="worst_positive"].drop(columns="per_source").to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(1):
        main()
