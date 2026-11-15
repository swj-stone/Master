"""Summarize paired switches and probe exposure by task without pooling repeated units for inference."""
from pathlib import Path

import numpy as np
import pandas as pd


OUT=Path(__file__).resolve().parents[2]/"results/context_review/factorial"
KEY=["dataset","seed","proposal"]


def compare(eligible):
    rows=[]
    for label, group in [("all",eligible),("rank_ends_pass",eligible[eligible.rank_ends.eq(0)])]:
        for ds,part in [("ALL",group),*group.groupby("dataset")]:
            for a,b in [("rank_ends","guard_ends"),("rank_ends","rank_interior"),
                        ("guard_ends","guard_interior"),("rank_interior","guard_interior"),
                        ("rank_ends","guard_interior")]:
                left,right=part[a].astype(bool),part[b].astype(bool)
                rows.append(dict(dataset=ds,condition=label,first=a,second=b,n=len(part),
                                 both_fail=int((left & right).sum()),both_pass=int((~left & ~right).sum()),
                                 pass_to_fail=int((~left & right).sum()),fail_to_pass=int((left & ~right).sum())))
    return pd.DataFrame(rows)


def pair_exposure(eligible):
    parts=[pd.read_csv(f) for f in sorted(OUT.glob("seed-*/*/factorial-directions.csv"))]
    directions=pd.concat([p for p in parts if len(p)],ignore_index=True)
    result=[]
    for ds,part in [("ALL",directions),*directions.groupby("dataset")]:
        candidates=eligible if ds=="ALL" else eligible[eligible.dataset.eq(ds)]
        count=candidates.set_index(KEY)[[]].copy()
        for key,subset in (
            ("pairs",part),
            ("rank_flip",part[part.rank_direction.ne(0)]),
            ("guard_flip",part[part.guard_direction.ne(0)]),
            ("both_same",part[part.guard_direction.eq(part.rank_direction)&part.rank_direction.ne(0)]),
            ("reversed",part[part.guard_direction.eq(-part.rank_direction)&part.rank_direction.ne(0)]),
            ("rank_only",part[part.rank_direction.ne(0)&part.guard_direction.eq(0)]),
            ("guard_only",part[part.guard_direction.ne(0)&part.rank_direction.eq(0)])):
            count[key]=subset.groupby(KEY).pairs.sum().reindex(count.index,fill_value=0)
        metrics={"dataset":ds,"eligible_candidates":len(count),"candidates_with_pairs":int(count.pairs.gt(0).sum())}
        for col in count:
            metrics[col+"_n_pairs"]=count[col].sum()
            metrics[col+"_n_candidates"]=count[col].gt(0).sum()
            metrics[col+"_candidate_mean_fraction"]=count[col].div(count.pairs.replace(0,np.nan)).mean()
        result.append(metrics)
    return pd.DataFrame(result)


def main():
    data=pd.read_csv(OUT/"factorial-candidates.csv")
    eligible=data[data.novel.gt(0)]
    compare(eligible).to_csv(OUT/"factorial-switches.csv",index=False)
    pair_exposure(eligible).to_csv(OUT/"factorial-pair-exposure.csv",index=False)
    split=data.groupby(["dataset","seed"]).agg(unique_trajectory=("unique_trajectory","first"),
                                                sampled=("proposal","size"),tail=("tail","first"),
                                                novel=("novel","first"),endpoints=("endpoints","first"))
    split.to_csv(OUT/"factorial-split-exposure.csv")
    print(pd.read_csv(OUT/"factorial-summary.csv").query("dataset=='ALL'").to_string(index=False))


if __name__=="__main__":
    main()
