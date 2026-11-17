"""Equal-source contrasts for the outage experiment on exposed data."""
import argparse
import numpy as np
import pandas as pd
from experiments.minpi_compression import benchmark as b


def summary(data,arm,controls):
    rng=np.random.default_rng(107)
    rows=[]
    for (study,task,choice), group in data.groupby(["study","task","choice"]):
        for control in controls:
            source_level={}
            for metric in ("k","worst","intact","teacher_worst","teacher_intact","auc","accuracy","r2","scaled_mae"):
                if metric not in group: continue
                w=group.pivot(index=["dataset","seed"],columns="method",values=metric)
                if control not in w or arm not in w:
                    continue
                w=w[[arm,control]].dropna()
                if not len(w): continue
                difference=np.log(w[arm]/w[control]) if metric=="k" else w[arm]-w[control]
                source_level[metric]=difference.groupby(level=0).mean()
            table=pd.DataFrame(source_level)
            sample=rng.integers(len(table),size=(5000,len(table)))
            for metric in table:
                vals=table[metric].values
                take=np.r_[vals.mean(),np.quantile(vals[sample].mean(axis=1),[.025,.975])]
                if metric=="k": take=np.exp(take)
                rows.append(dict(study=study,task=task,choice=choice,arm=arm,control=control,metric=metric,
                    estimate=take[0],low=take[1] if len(vals)>1 else np.nan,high=take[2] if len(vals)>1 else np.nan,
                    sources=len(vals)))
    return pd.DataFrame(rows)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs",nargs="+",required=True)
    cli.add_argument("--arm",default="pair_min")
    cli.add_argument("--controls",nargs="+",default=["pair_mean","pair_point","energy","full"])
    cli.add_argument("--out")
    args=cli.parse_args()
    source=pd.concat([pd.read_csv(b.ROOT/"results/minpi_rework"/r/"scores.csv").assign(study=r) for r in args.runs])
    result=summary(source,args.arm,args.controls)
    if args.out: result.to_csv(args.out,index=False)
    print(result[result.metric.isin(["k","intact","worst"])].pivot(
        index=["study","task","choice","control"],columns="metric",values="estimate").round(4).to_string())
