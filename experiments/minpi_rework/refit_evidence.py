"""Export the paired development scores, with a guard and forecast audit."""
import argparse
import gzip
import json

import numpy as np
import pandas as pd

from experiments.minpi_compression import benchmark as b,legacy_gain as legacy
from experiments.minpi_rework.contrasts import summarize
from experiments.minpi_rework.exchange_benchmark import load_dev


def export(runs):
    frames=[]
    eps=8*np.finfo(float).eps
    for study in runs:
        path=b.ROOT/"results/minpi_rework"/study
        frame=pd.read_csv(path/"scores.csv")
        expected=30 if study.endswith(("six-five","direct-six")) else 90
        assert len(frame[["dataset","seed"]].drop_duplicates())==expected,study
        manifest=legacy.jobs() if set(frame.dataset)&set(legacy.CONFIG["datasets"]) else {}
        for (name,s),group in frame.groupby(["dataset","seed"]):
            records=json.loads((path/f"{name}-{s}.json").read_bytes())
            x,y,task,idx,outer,mask=load_dev(name,s,manifest)
            scale=max(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]]),1e-12)
            with np.load(path/f"{name}-{s}.npz") as outputs:
                by_method={r["method"]:r for r in records}
                if "direct" in by_method:
                    for field,key in (("support","adaptive"),("fixed","fixed")):
                        assert by_method["direct"][field]==by_method["width"][field]
                        np.testing.assert_array_equal(outputs["direct_"+key],outputs["width_"+key])
                for r in records:
                    for t in r["trace"]:
                        if t["k"]==r["p"]: continue
                        if task=="classification":
                            actual=t["acc_loss"]+t["acc_se"]<=.01+eps and t["auc_loss"]<=.01+eps
                        else:
                            actual=t["scaled_mae_loss"]+t["scaled_mae_se"]<=.02+eps and t["scaled_mse_loss"]+t["scaled_mse_se"]<=.02+eps
                        assert actual==t["pass_guard"],(name,s,r["method"],t)
                    for key,cols in (("adaptive","support"),("fixed","fixed")):
                        row=group[(group.method==r["method"])&(group.policy==key)].iloc[0]
                        assert row.k==len(set(r[cols])) and all(0<=j<x.shape[1] for j in r[cols])
                        independent=legacy.independent_metrics(outer[mask],outputs[r["method"]+"_"+key][mask],task,scale)
                        for metric,v in independent.items():
                            np.testing.assert_allclose(row[metric],v,rtol=1e-10,atol=1e-10)
                        frame.loc[row.name,"support"]=" ".join(map(str,r[cols]))
        # Both raw pilot clock fields describe the adaptive pipeline. The extra
        # fixed-budget forecast has no independent standalone pipeline clock.
        frame.loc[frame.policy=="fixed",["selection_s","deployment_s"]]=np.nan
        frames.append(frame.assign(study=study))
    data=pd.concat(frames,ignore_index=True)
    location=b.ROOT/"experiments/minpi_rework/refit_evidence.csv.gz"
    with location.open("wb") as raw,gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as f:
        f.write(data.to_csv(index=False,lineterminator="\n").encode("utf8"))
    print(f"Exported {len(data)} independently rescored predictions.")
    for (study,policy),group in data.groupby(["study","policy"]):
        refs=[m for m in ("full","energy","greedy","gain","direct") if m in set(group.method)]
        effects=summarize(group,"width",refs)
        print(study,policy)
        print(effects.pivot(index=["task","reference"],columns="metric",values="point").round(4).to_string())


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("runs",nargs="+")
    export(cli.parse_args().runs)
