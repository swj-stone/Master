"""Inspect the locked decisions, conditional fold differences, and costs."""
import numpy as np
import pandas as pd

from experiments.context_review.refit_vote_grid import OUT, ARMS, read


def main():
    rows = pd.read_csv(OUT/"per-fold.csv").set_index(["dataset","fold","arm"])
    costs = pd.read_csv(OUT/"costs.csv")
    counts,matched,ballots = [],[],[]
    for name,fold in rows.index.droplevel("arm").unique():
        folder = OUT/name/f"fold-{fold}"
        selection = read(folder/"selection.json")
        k = selection["widths"]["anchor"]
        with np.load(folder/"guard.npz") as arrays:
            min_count = int(np.bincount(arrays["y"]).min())
            n = len(arrays["y"])
        for w,table in selection["votes"].items():
            w = int(w)
            if w==k:
                continue
            data = np.asarray(table,dtype=bool)
            ballots.append(dict(dataset=name,fold=fold,guard_n=n,smallest_class_n=min_count,anchor_k=k,k=w,
                                global_point=bool(data[0,0]),class_point=bool(data[:,0].all()),
                                class_vote=bool((data.sum(axis=1)>=3).all())))
    b = pd.DataFrame(ballots)
    b.to_csv(OUT/"candidate-decisions.csv",index=False)
    for arm in ("anchor",*ARMS):
        df = rows.xs(arm,level="arm")
        counts.append(dict(arm=arm,folds=len(df),reduced=int((df.k<df.anchor_k).sum()),
                           exceeds=int(df.exceeds.sum()),class_exceeds=int(df.class_exceeds.sum()),
                           checks=int(df.checks.sum()),possible_checks=int(df.possible_checks.sum())))
    summary = pd.DataFrame(counts)
    summary.to_csv(OUT/"counts.csv",index=False)
    for a,baseline in (("class_point","global_point"),("class_vote","global_point"),("class_vote","class_point")):
        first,second = [rows.xs(arm,level="arm") for arm in (a,baseline)]
        same = first.k==second.k
        cols = ["k","retained","auc","accuracy","recall","exceeds","class_exceeds"]
        for dataset,sub in (first[cols].astype(float)-second[cols].astype(float)).groupby(level="dataset"):
            unchanged = int(same.loc[dataset].sum())
            d = sub.mean().to_dict()
            matched.append(dict(arm=a,reference=baseline,dataset=dataset,folds=len(sub),same_width=unchanged,**d))
    pairs = pd.DataFrame(matched)
    pairs.to_csv(OUT/"paired-differences.csv",index=False)
    print(summary.to_string(index=False))
    print(pairs.groupby(["arm","reference"])[["same_width","folds"]].sum().to_string())
    print(pairs.groupby(["arm","reference"])[["auc","accuracy","recall","retained"]].mean().to_string())
    print(costs.sum(numeric_only=True).to_string())
    switched = pd.crosstab(b.class_point,b.class_vote)
    print("Candidate-level point vs. four-panel decisions")
    print(switched.to_string())


if __name__=="__main__":
    main()
