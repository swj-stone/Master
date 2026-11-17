"""Score the sealed wide-input forecasts and report complete-path comparator timings."""
import json
import numpy as np
import pandas as pd
from experiments.minpi_compression import benchmark as b
from experiments.minpi_rework import wide_veto_run as run,wide_veto_data as data


def report(timing_only=False):
    lock=json.loads((run.OUT/"lock.json").read_bytes())
    assert len(lock["receipts"])==20 and lock["code"]==run.digest_paths()
    times=[]
    predictions=[]
    for key,hash_ in lock["receipts"].items():
        folder=run.ROOT/key
        assert b.digest(folder/"selection.json")==hash_
        row=json.loads((folder/"selection.json").read_bytes())
        assert b.digest(folder/"forecasts.npz")==row["forecast"]
        name,seed=row["dataset"],row["seed"]
        if not timing_only:
            x,y,task,idx=data.partition(name,seed)
            assert row["task"]==task and row["splits"]=={k:v.tolist() for k,v in idx.items()}
            sd=max(float(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]])),1e-12)
            with np.load(folder/"forecasts.npz",allow_pickle=False) as f:
                for method,keep in row["sets"].items():
                    vals=b.metrics(y[idx["test"]],f[method],task,sd)
                    predictions.append(dict(dataset=name,seed=seed,task=task,method=method,
                        n_test=len(idx["test"]),p=row["p"],k=len(keep),**vals))
        for raw in row["rows"]:
            common=row["fit_s"]+row["prepare_s"]+row["acceptance_s"]+row["pred_s"]
            times.append(dict(dataset=name,seed=seed,p=row["p"],active=row["voting_active"],
                **raw,fit_s=row["fit_s"],prepare_s=row["prepare_s"],acceptance_s=row["acceptance_s"],
                prediction_s=row["pred_s"],total_s=common+raw["rank_s"]))
    df=pd.DataFrame(times)
    df.to_csv(run.OUT/"timings.csv",index=False)
    if predictions:
        pred=pd.DataFrame(predictions)
        pred.to_csv(run.OUT/"scores.csv",index=False)
        pred.groupby(["dataset","method"]).mean(numeric_only=True).to_csv(run.OUT/"source-scores.csv")
    med=df.groupby(["dataset","seed","method"])[["rank_s","total_s","checks"]].median().unstack()
    wide=df[["dataset","seed","p","active"]].drop_duplicates().set_index(["dataset","seed"])
    ratio_rows=[]
    for scope in ("rank_s","total_s","checks"):
        h=med[scope]
        for kind,left,right in (("best_exact_time",["early","coalesced_early"],
                                    ["straight","blas","coalesced_straight","coalesced_blas"]),
                                ("full_straight",["early"],["straight"]),
                                ("coalesced_straight",["coalesced_early"],["coalesced_straight"]),
                                ("coalesced_blas",["coalesced_early"],["coalesced_blas"])):
            if scope=="checks" and kind=="best_exact_time":
                continue
            v=h[left].min(axis=1)
            e=h[right].min(axis=1)
            src=np.log(v/e).groupby("dataset").mean()
            rng=np.random.default_rng(19504)
            draws=np.exp(rng.choice(src.values,size=(20000,len(src)),replace=True).mean(axis=1))
            ratio_rows.append(dict(scope=scope,contrast=kind,ratio=np.exp(src.mean()),
                lo=np.quantile(draws,.025),hi=np.quantile(draws,.975),**np.exp(src).to_dict()))
            if kind=="best_exact_time":
                wide[scope+"_veto"]=v
                wide[scope+"_exhaustive"]=e
                wide[scope+"_veto_solver"]=h[left].idxmin(axis=1)
                wide[scope+"_exhaustive_solver"]=h[right].idxmin(axis=1)
    wide.to_csv(run.OUT/"paired-jobs.csv")
    ratios=pd.DataFrame(ratio_rows)
    ratios.to_csv(run.OUT/"contrasts.csv",index=False)
    print(ratios.to_string(index=False))


if __name__=="__main__":
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument("--timing-only",action="store_true")
    report(cli.parse_args().timing_only)
