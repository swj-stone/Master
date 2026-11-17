"""Inspect matched-label votes by source and fixed versus guarded widths."""
import numpy as np
import pandas as pd
from experiments.minpi_rework.quota_coalition_proper import DEST


def main():
    f=pd.read_csv(DEST/"proper-all.csv")
    rows=[]
    for bgt in ("guard","half","quarter"):
        for m in ("min","harm","pool","last"):
            for b in ("energy","pool","last"):
                x=f[f.arm==m+"_"+bgt].set_index(["dataset","seed"])
                y=f[f.arm==b+"_"+bgt].set_index(["dataset","seed"]).loc[x.index]
                for metric in ("k","auc","accuracy","r2","mae","scaled_mae"):
                    v=(np.log(x.k/y.k) if metric=="k" else x[metric]-y[metric]).dropna()
                    values=v.groupby("dataset").mean()
                    for name,point in values.items():
                        rows.append(dict(budget=bgt,method=m,reference=b,metric=metric,source=name,
                            value=np.exp(point) if metric=="k" else point))
                    point=values.mean()
                    rows.append(dict(budget=bgt,method=m,reference=b,metric=metric,source="all",
                        value=np.exp(point) if metric=="k" else point))
    out=pd.DataFrame(rows)
    out.to_csv(DEST/"proper-contrasts.csv",index=False)
    print(out[(out.source=="all")&(out.method=="min")].pivot(index=["budget","reference"],columns="metric",values="value").round(4))


if __name__=="__main__":
    main()
