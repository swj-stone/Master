"""Compare released outcomes and both canonical and nominal width for all frozen selectors."""
import json
import numpy as np
import pandas as pd
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import predictive_reachable as reach,wide_veto_data,validation_predictive as seven


def main():
    lock=json.loads((reach.OUT/"lock.json").read_bytes())["receipts"]
    assert len(lock)==55
    rows=[]
    for key,dig in lock.items():
        folder=reach.RUN/key
        assert b.digest(folder/"selection.json")==dig
        info=json.loads((folder/"selection.json").read_bytes())
        assert b.digest(folder/"forecasts.npz")==info["forecast"]
        src=wide_veto_data if info["panel"]=="wide" else seven
        name,seed,task=info["name"],info["seed"],info["task"]
        x,y,_,idx=src.partition(name,seed) if src is wide_veto_data else src._partition(name,seed)
        np.testing.assert_array_equal(idx["test"],info["outer"])
        scale=max(float(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]])),1e-12)
        with np.load(folder/"forecasts.npz") as f:
            for arm,r in info["records"].items():
                old=arm.replace("_canonical","_original")
                v=b.metrics(y[idx["test"]],f[arm],task,scale)
                rows.append(dict(panel=info["panel"],dataset=name,seed=seed,task=task,p=x.shape[1],
                    arm=arm,nominal=len(r["nominal"]),reachable=len(r["canonical"]),path=r["path"],
                    changed_predictions=not np.array_equal(f[arm],f[old]),changed_winner=r["path"]!=info["records"][old]["path"],
                    original_risk=info["records"][old]["risk"],worst_acceptance=r["risk"],
                    max_prediction_change=float(np.abs(f[arm]-f[old]).max()),**v))
    df=pd.DataFrame(rows)
    df.to_csv(reach.OUT/"scores.csv",index=False)
    df.groupby(["panel","dataset","arm"]).mean(numeric_only=True).to_csv(reach.OUT/"source-means.csv")
    results=[]
    for panel,data in df.groupby("panel"):
        arms=set(data.arm)
        for other in ("min","pooled","static","intact"):
            arm="portfolio_"+other
            if arm+"_canonical" not in arms:
                continue
            for reference in ("mdi_canonical",arm+"_original","full"):
                x=data[data.arm==arm+"_canonical"].set_index(["dataset","seed"])
                y=data[data.arm==reference].set_index(["dataset","seed"]).loc[x.index]
                for field in ("reachable","reachable_fraction","auc","accuracy","r2","scaled_mae"):
                    v=((x.reachable-y.reachable)/x.p if field=="reachable_fraction" else x[field]-y[field]).dropna()
                    src=v.groupby(level=0).mean()
                    rng=np.random.default_rng(6032)
                    draws=rng.choice(src.to_numpy(),size=(20000,len(src)),replace=True).mean(axis=1)
                    results.append(dict(panel=panel,arm=arm,reference=reference,metric=field,
                        paired_delta=src.mean(),lo=np.quantile(draws,.025),hi=np.quantile(draws,.975),
                        sources=len(src),jobs=len(v)))
    pd.DataFrame(results).to_csv(reach.OUT/"contrasts.csv",index=False)
    print(df[df.arm.str.startswith("portfolio_")&df.arm.str.endswith("_canonical")].groupby(["panel","arm"])
          [["changed_predictions","changed_winner"]].sum().to_string())


if __name__=="__main__":
    main()
