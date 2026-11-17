"""Report all ranking arms with native metrics, outage effects and fidelity failures."""
import numpy as np
import pandas as pd
from experiments.minpi_rework.quota_coalition_mean_run import DEST


def main():
    combined=pd.concat([pd.read_csv(DEST/"scores-pilot.csv"),pd.read_csv(DEST/"scores-variant-pilot.csv")])
    combined=combined.drop_duplicates(["dataset","seed","mode","method"])
    results=[]
    for (arm,mode),grp in combined.groupby(["method","mode"]):
        trial=grp.set_index(["dataset","seed"])
        baseline=combined[(combined.method=="last_full")&(combined["mode"]==mode)].set_index(["dataset","seed"]).loc[trial.index]
        outcomes={}
        for metric in ("k","pipeline_s","auc","accuracy","r2","scaled_mae"):
            vals=(np.log(trial[metric]/baseline[metric]) if metric in ("k","pipeline_s") else trial[metric]-baseline[metric]).dropna()
            val=vals.groupby("dataset").mean().mean()
            outcomes[metric]=np.exp(val) if metric in ("k","pipeline_s") else val
        results.append(dict(arm=arm,mode=mode,**outcomes))
    out=pd.DataFrame(results)
    out.to_csv(DEST/"ranking-contrasts.csv",index=False)
    print(out[(out["mode"]=="normal") & out.arm.str.endswith("_full")].round(4).to_string(index=False))


if __name__=="__main__":
    main()
