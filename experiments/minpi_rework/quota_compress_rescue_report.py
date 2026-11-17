"""Source-level summaries of matched finite-budget exchange controls."""
import argparse
import numpy as np
import pandas as pd
from experiments.minpi_rework.quota_compress_rescue_run import DEST


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true")
    tag="all" if parser.parse_args().all else "pilot"
    f=pd.read_csv(DEST/f"scores-{tag}.csv")
    pairs=[(start+"_"+a,start+"_"+b) for start in ("exact","ablation","gain")
           for a,b in (("votes","start"),("votes","pair"),("votes","one"),("one","start"))]
    pairs.extend([("exact_votes","ablation_votes"),("exact_votes","gain_votes"),("ablation_votes","gain_votes")])
    rows=[]
    for m,c in pairs:
        x=f[f.method==m].set_index(["panel","dataset","seed"])
        y=f[f.method==c].set_index(["panel","dataset","seed"])
        for panel in ("all","historical","core","wide"):
            for metric in ("k","pipeline_s","search_s","checks","rho","auc","accuracy","mae","r2","scaled_mae"):
                ratio=metric in ("k","pipeline_s","search_s","checks")
                if ratio and (y[metric]<=0).any():
                    continue
                v=(np.log(x[metric]/y[metric]) if ratio else x[metric]-y[metric]).dropna()
                if panel!="all":
                    v=v[v.index.get_level_values("panel")==panel]
                v=v.groupby("dataset").mean()
                rng=np.random.default_rng(8847)
                draws=rng.choice(v.to_numpy(),size=(10000,len(v)),replace=True).mean(axis=1)
                estimates=np.array([v.mean(),*np.quantile(draws,[.025,.975])])
                if ratio:
                    estimates=np.exp(estimates)
                rows.append(dict(panel=panel,method=m,control=c,metric=metric,
                    point=estimates[0],low=estimates[1],high=estimates[2]))
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/f"contrasts-{tag}.csv",index=False)
    q=frame[(frame.panel=="all")&(frame.method=="exact_votes")&
        (frame.metric.isin(("k","pipeline_s","auc","accuracy","scaled_mae")))]
    print(q.to_string(index=False))


if __name__=="__main__":
    main()
