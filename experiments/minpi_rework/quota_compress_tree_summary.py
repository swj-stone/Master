"""Source means and paired ratios for sealed historical tree forecasts."""
import json
import numpy as np
import pandas as pd
from experiments.minpi_rework.quota_compress_tree_run import DEST,OUT


def main():
    suffix="42-123-456-789-1024"
    frames=[pd.read_csv(DEST/f"historical-{pre}{suffix}.csv") for pre in ("","controls-")]
    frame=pd.concat(frames,ignore_index=True)
    means=frame.groupby(["dataset","part","method"],as_index=False)[
        ["k","p","auc","accuracy","r2","mae","scaled_mae","pipeline_s","search_s","checks","ballots"]
        ].mean()
    means.to_csv(DEST/"source-means.csv",index=False)
    gm=lambda vals: np.exp(np.mean(np.log(vals)))
    pri=frame[frame.part=="primary"]
    refs=["full","gain_sweep","exact_greedy","refit_gain","gain_prefix","one_vote"]
    for focal in ["votes","exact_greedy"]:
        a=pri[pri.method==focal].set_index(["dataset","seed"])
        for method in refs:
            base=pri[pri.method==method].set_index(["dataset","seed"]).loc[a.index]
            delta=(a[["auc","accuracy","r2","mae","scaled_mae"]]-base[["auc","accuracy","r2","mae","scaled_mae"]]).groupby("dataset").mean()
            print(focal,"vs",method,"width",gm(a.k/base.k),"pipeline",gm(a.pipeline_s/base.pipeline_s),
                "worst source Δ",delta.min().round(4).to_dict(),"MAE max",delta.mae.max())
    changed=0
    for directory in (OUT/"controls").iterdir():
        cur=json.loads((directory/"selection.json").read_bytes())["records"]["one_vote"]
        new=json.loads((OUT/directory.name/"selection.json").read_bytes())["records"]["votes"]
        changed+=cur["support"]!=new["support"]
    print("one-voter vs multi-voter differing supports",changed)


if __name__=="__main__":
    main()
