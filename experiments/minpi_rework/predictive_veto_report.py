"""Paired source-cluster intervals, retaining the smallest and largest sources."""
import argparse
import numpy as np
import pandas as pd
from experiments.minpi_rework.predictive_veto_timing import DEST


def analyze(partial=False,native=False):
    prefix="native-" if native else ""
    f=pd.read_csv(DEST/(prefix+"repeats.csv"))
    assert not f.duplicated(["dataset","seed","repeat","method"]).any()
    if not partial:
        assert len(f)==(1875 if native else 1125) and f.dataset.nunique()==25
    med=f.groupby(["panel","dataset","seed","p","method"],as_index=False).median(numeric_only=True)
    metrics=["checks","rank_s","fit_to_rank_s"] if native else ["checks","core_s","compiled_core_s","rank_s","fit_to_rank_s"]
    results=[]
    bysource=[]
    for control in (("native_all","straight","alloc_early","blas") if native else ("native_all","blas_all")):
        x=med[med.method=="early"].set_index(["panel","dataset","seed","p"])
        y=med[med.method==control].set_index(["panel","dataset","seed","p"]).loc[x.index]
        for measure in metrics:
            if control=="blas_all" and measure=="compiled_core_s":
                continue
            val=np.log(x[measure]/y[measure]).rename("log_ratio").reset_index()
            raw=val.groupby(["panel","dataset","p"]).log_ratio.mean().reset_index()
            for row in raw.itertuples():
                bysource.append(dict(panel=row.panel,dataset=row.dataset,p=row.p,metric=measure,baseline=control,
                    ratio=np.exp(row.log_ratio)))
            groups={"all":raw,"development":raw[raw.panel=="development"],"external":raw[raw.panel=="external"],
                "p<=20":raw[raw.p<=20],"21<=p<=50":raw[raw.p.between(21,50)],"p>50":raw[raw.p>50]}
            for key,sub in groups.items():
                if not len(sub):
                    continue
                d=sub.log_ratio.to_numpy()
                rng=np.random.default_rng(91673)
                samples=rng.choice(d,(20000,len(d)),replace=True).mean(axis=1)
                point,lo,hi=np.exp([d.mean(),*np.quantile(samples,[.025,.975])])
                results.append(dict(panel=key,metric=measure,baseline=control,sources=len(sub),
                    point=point,low=lo,high=hi,faster_sources=int((d<0).sum())))
    summary=pd.DataFrame(results)
    if not partial:
        summary.to_csv(DEST/(prefix+"contrasts.csv"),index=False)
        pd.DataFrame(bysource).to_csv(DEST/(prefix+"source-contrasts.csv"),index=False)
        med.to_csv(DEST/(prefix+"split-medians.csv"),index=False)
    print(summary[summary.panel.isin(["all","external","development"])].round(4).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--partial",action="store_true")
    parser.add_argument("--native",action="store_true")
    args=parser.parse_args()
    analyze(args.partial,args.native)
