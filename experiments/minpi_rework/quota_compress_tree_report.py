"""Source-cluster summaries from released, sealed boosted-tree comparisons."""
import numpy as np
import pandas as pd
from experiments.minpi_rework.quota_compress_tree_confirmation import DEST


def main():
    f=pd.read_csv(DEST/"scores.csv")
    assert len(f)==65*9 and f.groupby(["dataset","seed","method"]).size().eq(1).all()
    metrics=["k","p","search_s","pipeline_s","postfit_s","relative_error","auc","accuracy","r2","mae","scaled_mae"]
    source=f.groupby(["panel","dataset","method"],as_index=False)[metrics].mean()
    source.to_csv(DEST/"source-means.csv",index=False)
    rows=[]
    ratios=("k","search_s","postfit_s","pipeline_s")
    for method,baseline in [
        ("directions","gain_sweep"),("directions","gain_prefix"),("directions","full"),
        ("directions","dense"),("directions","cache_only"),("guard_directions","guard_gain_sweep"),
        ("guard_directions","guard_gain_prefix"),("guard_directions","full"),
        ("guard_directions","directions")]:
        x=f[f.method==method].set_index(["panel","dataset","seed"])
        y=f[f.method==baseline].set_index(["panel","dataset","seed"]).loc[x.index]
        for panel in ("all","core","wide"):
            for metric in (*ratios,"auc","accuracy","r2","scaled_mae","mae"):
                val=(np.log(x[metric]/y[metric]) if metric in ratios else x[metric]-y[metric]).dropna()
                if panel!="all":
                    val=val[val.index.get_level_values("panel")==panel]
                v=val.groupby("dataset").mean()
                rng=np.random.default_rng(5002)
                draws=rng.choice(v.to_numpy(),size=(20000,len(v)),replace=True).mean(axis=1)
                est=np.array([v.mean(),*np.quantile(draws,[.025,.975])])
                if metric in ratios:
                    est=np.exp(est)
                rows.append(dict(panel=panel,method=method,baseline=baseline,metric=metric,
                                 unit="ratio" if metric in ratios else "difference",
                                 sources=len(v),value=est[0],low=est[1],high=est[2]))
    out=pd.DataFrame(rows)
    out.to_csv(DEST/"source-contrasts.csv",index=False)
    print(out[(out.panel=="all")&(out.method=="directions")&
        ((out.baseline=="gain_sweep")|((out.metric=="search_s")&out.baseline.isin(["dense","cache_only"])))].to_string(index=False))


if __name__=="__main__":
    main()
