"""Paired original-unit and source-bootstrap contrasts for scaled and unscaled compression."""
import argparse
import numpy as np
import pandas as pd
from experiments.minpi_rework.quota_compress_scale_run import DEST


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--all",action="store_true")
    tag="all" if cli.parse_args().all else "pilot"
    f=pd.read_csv(DEST/f"scores-{tag}.csv")
    pairs=[(m+"_scaled",ref) for m in ("exact","gain","ablation","treeshap") for ref in (m+"_raw","full")]
    pairs.extend(("exact_scaled",m+"_scaled") for m in ("gain","ablation","treeshap"))
    pairs.append(("exact_scaled_votes","exact_scaled"))
    out=[]
    for a,b in pairs:
        x=f[f.method==a].set_index(["dataset","seed"])
        y=f[f.method==b].set_index(["dataset","seed"]).loc[x.index]
        for metric in ("k","search_s","pipeline_s","auc","accuracy","mae","r2","scaled_mae"):
            ratio=metric in ("k","search_s","pipeline_s")
            values=(np.log(x[metric]/y[metric]) if ratio else x[metric]-y[metric]).dropna()
            if len(values)==0:
                continue
            v=values.groupby("dataset").mean().to_numpy()
            rng=np.random.default_rng(6228)
            dist=rng.choice(v,(10000,len(v)),replace=True).mean(axis=1)
            est=np.array([v.mean(),*np.quantile(dist,[.025,.975])])
            if ratio:
                est=np.exp(est)
            out.append(dict(method=a,reference=b,metric=metric,point=est[0],low=est[1],high=est[2]))
    report=pd.DataFrame(out)
    report.to_csv(DEST/f"contrasts-{tag}.csv",index=False)
    print(report[(report.method=="exact_scaled")&report.metric.isin(("k","auc","accuracy","r2","scaled_mae"))].to_string(index=False))


if __name__=="__main__":
    main()
