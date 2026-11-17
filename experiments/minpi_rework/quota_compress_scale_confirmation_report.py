"""Source-level predictive and computational contrasts on the independent five-source study."""
import numpy as np
import pandas as pd
from experiments.minpi_rework.quota_compress_scale_confirmation import DEST


def main():
    f=pd.read_csv(DEST/"scores.csv")
    ms=("k","p","lam","relative_error","pipeline_s","auc","accuracy","r2","mae","scaled_mae")
    means=f.groupby(["dataset","task","method"],as_index=False)[list(ms)].mean()
    means.to_csv(DEST/"source-means.csv",index=False)
    rows=[]
    comparisons=[("exact_policy",base) for base in ("exact_raw","gain_policy","ablation_policy","treeshap_policy","native_full")]
    comparisons.extend((m+"_scaled",m+"_raw") for m in ("exact","gain","ablation","treeshap"))
    comparisons.extend([("exact_rescue","ablation_rescue"),("exact_rescue","exact_raw"),("ablation_rescue","ablation_raw"),
        ("exact_policy_votes","exact_policy"),("exact_rescue_votes","exact_rescue")])
    for m,b in comparisons:
        x=f[f.method==m].set_index(["dataset","seed"])
        y=f[f.method==b].set_index(["dataset","seed"]).loc[x.index]
        for domain in ("all","classification","regression"):
            for metric in ("k","pipeline_s","search_s","checks","auc","accuracy","r2","mae","scaled_mae"):
                ratio=metric in ("k","pipeline_s","search_s","checks")
                if ratio and (y[metric]<=0).any():
                    continue
                val=(np.log(x[metric]/y[metric]) if ratio else x[metric]-y[metric]).dropna()
                if domain!="all":
                    val=val.loc[val.index.intersection(x[x.task==domain].index)]
                if not len(val):
                    continue
                v=val.groupby("dataset").mean().to_numpy()
                rng=np.random.default_rng(7736)
                dist=rng.choice(v,(20000,len(v)),replace=True).mean(axis=1)
                est=np.array([v.mean(),*np.quantile(dist,[.025,.975])])
                if ratio:
                    est=np.exp(est)
                rows.append(dict(domain=domain,method=m,baseline=b,metric=metric,sources=len(v),point=est[0],low=est[1],high=est[2]))
    report=pd.DataFrame(rows)
    report.to_csv(DEST/"source-contrasts.csv",index=False)
    print(report[(report.method=="exact_policy")&(report.domain=="all")&
            report.metric.isin(("k","auc","accuracy","scaled_mae","r2"))].to_string(index=False))


if __name__=="__main__":
    main()
