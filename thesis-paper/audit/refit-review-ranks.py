"""Within-fold ranking of observed refit losses by elementary deletion size, rank mass or votes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/"results/context_review/refit-target"
PRIMARY=ROOT/"results/thesis_benchmark/prospective-pmlb-contexts-r1"


def main():
    verified=pd.read_csv(RUN/"decisions.csv")
    rows=[]; matched=[]
    for (name,fold,cohort),part in verified.groupby(["dataset","fold","cohort"]):
        folder=RUN/name/f"fold-{fold}"
        design=json.loads((folder/"selection.json").read_bytes())
        donor=json.loads((PRIMARY/name/f"fold-{fold}/selection.json").read_bytes())
        anchor=donor["order"][:donor["anchor_k"]]
        rank={c:len(anchor)-i for i,c in enumerate(anchor)}
        clipped=[]
        for rec in part.itertuples():
            s=set(design["supports"][rec.subset])
            removed=set(anchor)-s
            ballots=np.asarray(design["ballots"][rec.subset]["separating"])
            minimum=4-np.min(ballots.sum(axis=1))
            endpoints=4-np.min(ballots[[0,-1]].sum(axis=1))
            assert (minimum>=2)==rec.separating and (endpoints>=2)==rec.endpoints
            clipped.append(dict(fail=rec.fail,size=len(removed)/len(anchor),
                                rank_mass=sum(rank[c] for c in removed)/sum(rank.values()),
                                min_votes=minimum,endpoint_votes=endpoints))
        frame=pd.DataFrame(clipped)
        for score in ("size","rank_mass","min_votes","endpoint_votes"):
            eligible=frame.fail.nunique()==2
            rows.append(dict(dataset=name,fold=fold,cohort=cohort,score=score,n=len(frame),fail=int(frame.fail.sum()),
                             concordance=roc_auc_score(frame.fail, frame[score]) if eligible else float("nan")))
            # Count within-size positive-negative pairs; tied scores contribute one half.
            credit, pairs = 0, 0
            for _,same in frame.groupby("size"):
                pos = same.loc[same.fail,score].to_numpy()
                neg = same.loc[~same.fail,score].to_numpy()
                differences = pos[:,None]-neg[None,:]
                pairs += differences.size
                credit += (differences>0).sum()+.5*(differences==0).sum()
            matched.append(dict(dataset=name,fold=fold,cohort=cohort,score=score,pairs=pairs,
                                concordance=credit/pairs if pairs else float("nan")))
    results=pd.DataFrame(rows)
    results.to_csv(HERE/"refit-review-ranks-by-fold.csv",index=False)
    by_task=results.groupby(["cohort","dataset","score"]).concordance.mean().dropna().reset_index()
    agg=by_task.groupby(["cohort","score"]).concordance.agg(["mean","count"]).reset_index()
    agg["eligible_folds"]=agg.apply(lambda r:results[(results.cohort==r.cohort)&(results.score==r.score)].concordance.count(),axis=1)
    agg.to_csv(HERE/"refit-review-ranks-summary.csv",index=False)
    print(agg.to_string(index=False))
    controls=pd.DataFrame(matched)
    controls.to_csv(HERE/"refit-review-ranks-same-width.csv",index=False)
    for (cohort,score),sub in controls.groupby(["cohort","score"]):
        eligible=sub.dropna()
        print(cohort, score, len(eligible), "folds,", int(eligible.pairs.sum()), "pairs,",
              eligible.groupby("dataset").concordance.mean().mean())


if __name__=="__main__":
    main()
