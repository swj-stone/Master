"""Source-level tables for the signed seven-source study and the six prepared datasets."""
import numpy as np
import pandas as pd
from experiments.minpi_compression.benchmark import ROOT


PAPER=ROOT/"thesis-paper"
EVIDENCE=PAPER/"evidence/minpi_rework"
OUT=EVIDENCE/"predictive-confirmation"
FAMILY="mdi_plus_min_min"
LABELS={"votes":"House voting","churn":"Iranian churn","liver":"Liver patients","german":"German credit",
        "seoul":"Seoul bicycles","beijing":"Beijing PM2.5","sgemm":"SGEMM runtime"}
BASES=("mdi_min","min_min","static_min","mdi_plus_static_min","mdi_plus_intact_min","mdi_plus_pooled_min",
       "mdi_plus_min_pooled","mdi_plus_min_intact","mdi_plus_min_clean","full")


def summarize(frame):
    scores=frame.set_index(["dataset","seed","method"]).sort_index()
    a=scores.xs(FAMILY,level="method")
    if not a.groupby(level="dataset").size().eq(5).all():
        raise ValueError("Incomplete source")
    results=[]
    for name in BASES:
        other=scores.xs(name,level="method")
        if not a.index.equals(other.index):
            raise ValueError(f"Unpaired {name}")
        effects=pd.DataFrame(index=a.index)
        for field in ("accuracy","auc","r2","scaled_mae","outer_worst","outer_positive","k"):
            if field in a:
                effects[field]=a[field]-other[field]
        effects["width_fraction"]=(a.k-other.k)/a.p
        if "outer_worst" in a:
            effects["violation"]=(a.outer_worst>.02).astype(float)-(other.outer_worst>.02).astype(float)
        if (a.k>0).all() and (other.k>0).all():
            effects["width_ratio"]=np.log(a.k/other.k)
        effects["pipeline_ratio"]=np.log(a.pipeline_s/other.pipeline_s)
        aggregate=effects.groupby(level="dataset").mean()
        for field in aggregate:
            vals=aggregate[field].dropna().to_numpy()
            if not len(vals):
                continue
            rng=np.random.default_rng(421711)
            sims=vals[rng.integers(0,len(vals),(20000,len(vals)))].mean(axis=1)
            estimate,lo,hi=vals.mean(),*np.quantile(sims,[.025,.975])
            if field.endswith("_ratio"):
                estimate,lo,hi=np.exp([estimate,lo,hi])
            results.append(dict(candidate=FAMILY,baseline=name,metric=field,sources=len(vals),
                                estimate=estimate,lower=lo,upper=hi))
    return pd.DataFrame(results)


def main():
    raw=pd.read_csv(OUT/"scores.csv")
    summarize(raw).to_csv(OUT/"paper-contrasts.csv",index=False)
    means=raw.groupby(["dataset","task","method"],sort=False).mean(numeric_only=True).reset_index()
    fields=["dataset","task","method","p","k","auc","accuracy","r2","mae","scaled_mae",
            "outer_worst","outer_positive","pipeline_s"]
    means[fields].to_csv(OUT/"source-means.csv",index=False)
    rows=[]
    for source,group in means.groupby("dataset",sort=False):
        g=group.set_index("method")
        cls=g.loc[FAMILY,"task"]=="classification"
        cols=("auc","accuracy") if cls else ("r2","mae")
        pair=lambda m:" / ".join(f"{g.loc[m,c]:.3f}" for c in cols)
        rows.append(" & ".join([LABELS[source],str(int(g.loc["full","p"])),f'{g.loc["mdi_min","k"]:.1f}',
                    f'{g.loc[FAMILY,"k"]:.1f}',pair("full"),pair(FAMILY)])+r"\\")
    (PAPER/"minpi-predictive-rows.tex").write_text("\n".join(rows)+"\n",encoding="utf8")
    fixture=EVIDENCE/"predictive-prepared/scores.csv"
    if fixture.exists():
        six=pd.read_csv(fixture)
        if six.groupby(["dataset","method"]).size().ne(5).any():
            raise ValueError("Incomplete original six")
        six.groupby(["dataset","task","method"],as_index=False)[
            ["p","k","auc","accuracy","r2","mae","scaled_mae","pipeline_s"]].mean().to_csv(fixture.with_name("source-means.csv"),index=False)
        summarize(six).to_csv(fixture.with_name("paper-contrasts.csv"),index=False)


if __name__=="__main__":
    main()
