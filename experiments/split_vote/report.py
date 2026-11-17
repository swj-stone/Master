"""Verify sealed forecasts first, then compute paired source-level endpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.minpi_compression import legacy_gain as legacy
from experiments.minpi_replication.data import SOURCES, SEEDS
from experiments.split_vote.run import ORIGINAL, ROOT, RUN, digest


DEST = ROOT / "thesis-paper/evidence/split_vote"
EXTERNAL_ARMS = ("min_mass","pooled_mass","gain_mass","gain_grid","exhaustive")
ORIGINAL_ARMS = ("min_mass","gain_grid","exhaustive")


def score_run(tag,names,seeds,arms):
    entries = []
    # Check the complete expected panel before accessing the first test response.
    for name in names:
        for seed in seeds:
            for arm in arms:
                folder = RUN/tag/f"{name}-{seed}"/arm
                receipt = folder/"receipt.json"
                path = folder/"forecast.npz"
                rec = json.loads(receipt.read_bytes())
                if (name,seed,arm) != (rec["dataset"],rec["seed"],rec["method"]):
                    raise ValueError(f"Mismatched receipt {receipt}")
                if digest(path) != rec["forecast_sha256"]:
                    raise ValueError(f"Forecast hash mismatch {path}")
                entries.append((rec,path,digest(receipt)))
    if len({json.dumps(r["version"],sort_keys=True) for r,_,_ in entries}) != 1:
        raise ValueError("Mixed implementations across an experiment")
    prepared = legacy.jobs() if set(names)&set(ORIGINAL) else {}
    truths = {}
    rows = []
    for entry,path,file_hash in entries:
        n,s,m = (entry[key] for key in ("dataset","seed","method"))
        if (n,s) not in truths:
            if n in ORIGINAL:
                folder,manifest = prepared[n,s]
                target = manifest["target"]
                truth = pd.read_csv(folder/manifest["test_relpath"],usecols=[target])[target].to_numpy()
            else:
                from experiments.minpi_replication.data import _records, split
                _,_,_,idx = split(n,s)
                truth = _records(n).target[idx["test"]]
            truths[n,s] = truth
        truth = truths[n,s]
        with np.load(path) as arr:
            pred = arr["primary"]
        if len(pred)!=len(truth) or not np.isfinite(pred).all():
            raise ValueError(f"Invalid forecast {path}")
        if entry["task"]=="classification":
            np.testing.assert_allclose(pred.sum(axis=1),1.,atol=2e-6)
        score=legacy.independent_metrics(truth,pred,entry["task"],1.)
        score.pop("scaled_mae",None)
        r=entry["record"]
        width=int(r["retained"] if m=="exhaustive" else r["k"])
        support=r["support"] if m=="exhaustive" else r["selected"]
        if len(set(support))!=width or not all(0<=j<entry["p"] for j in support):
            raise ValueError(f"Invalid support {path}")
        rows.append(dict(dataset=n,seed=s,method=m,p=entry["p"],k=width,
                         seconds=r.get("seconds",r.get("worker_s")),fits=r["fits"],**score,
                         primary=score.get("auc",score.get("r2")),receipt_sha256=file_hash,
                         forecasts_sha256=entry["forecast_sha256"],near_width=r.get("near_width")))
    return pd.DataFrame(rows)


def compare(frame, candidate, baseline):
    keys=["dataset","seed"]
    left=frame[frame.method==candidate]
    right=frame[frame.method==baseline]
    paired=left.merge(right,on=keys,suffixes=("_pi","_base"),validate="one_to_one")
    if len(paired)!=len(left) or len(paired)!=len(right) or not len(paired):
        raise ValueError(f"Incomplete {candidate} / {baseline} pairing")
    paired["time_delta"]=paired.seconds_pi-paired.seconds_base
    paired["width_over_allowance"]=paired.k_pi-paired.k_base-np.maximum(1.,.05*paired.k_base)
    paired["primary_delta"]=paired.primary_pi-paired.primary_base
    paired["paired_pass"]=(paired.time_delta<0)&(paired.width_over_allowance<=0)
    paired["paired_quality_pass"]=paired.paired_pass&(paired.primary_delta>=-.01)
    paired["time_ratio"]=paired.seconds_pi/paired.seconds_base
    summary=paired.groupby("dataset",sort=False).median(numeric_only=True)
    summary["paired_pass_seeds"]=paired.groupby("dataset").paired_pass.sum()
    summary["quality_pass_seeds"]=paired.groupby("dataset").paired_quality_pass.sum()
    summary["seeds"]=paired.groupby("dataset").seed.count()
    summary["pass_joint"]=(summary.time_delta<0)&(summary.width_over_allowance<=0)
    summary["pass_with_quality"]=summary.pass_joint&(summary.primary_delta>=-.01)
    summary["majority_joint_seeds"]=summary.paired_pass_seeds>summary.seeds/2
    means=paired.groupby("dataset").mean(numeric_only=True)
    summary["pass_mean_joint"]=(means.time_delta<0)&((means.k_pi-means.k_base)<=np.maximum(1.,.05*means.k_base))
    summary["pass_mean_with_quality"]=summary.pass_mean_joint&(means.primary_delta>=-.01)
    if "near_width_base" in paired and paired.near_width_base.notna().all():
        near=paired.k_pi-paired.near_width_base-np.maximum(1.,.05*paired.near_width_base)
        summary["width_pass_at_common_slack"]=near.groupby(paired.dataset).median()<=0
    summary=summary.drop(columns=["seed","paired_pass","paired_quality_pass"],errors="ignore")
    summary["candidate"],summary["baseline"]=candidate,baseline
    return summary.reset_index()


def archived():
    f=pd.read_csv(ROOT/"docs/thesis/archives/piv1/pfi-paired.csv")
    arms=[]
    for prefix in ("piv1","gain"):
        df=f[["dataset","seed"]].copy()
        df["method"]=prefix
        for old,new in (("retained","k"),("wall_s","seconds"),("test_primary","primary")):
            df[new]=f[f"{prefix}_{old}"]
        arms.append(df)
    return compare(pd.concat(arms,ignore_index=True),"piv1","gain")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",choices=["archive","development","external"])
    args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if args.panel=="archive":
        archived().to_csv(DEST/"archived-piv1.csv",index=False)
        return
    if args.panel=="external":
        frame=score_run("four-frozen",SOURCES,SEEDS,EXTERNAL_ARMS)
        controls=EXTERNAL_ARMS[1:]
    else:
        frame=score_run("six-paired",ORIGINAL,legacy.CONFIG["seeds"],ORIGINAL_ARMS)
        controls=ORIGINAL_ARMS[1:]
    frame.to_csv(DEST/f"{args.panel}-seeds.csv",index=False)
    parts=[compare(frame,"min_mass",b) for b in controls]
    sources=pd.concat(parts,ignore_index=True)
    sources.to_csv(DEST/f"{args.panel}-sources.csv",index=False)
    print(sources.groupby("baseline")[["pass_joint","pass_with_quality"]].sum().to_string())


if __name__=="__main__":
    main()
