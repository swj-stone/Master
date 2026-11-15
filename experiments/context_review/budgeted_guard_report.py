"""Aggregate paired prospective-budget decisions with the task as the resampling unit."""
import numpy as np
import pandas as pd

from experiments.context_review.budgeted_guard import RUN


def main():
    rows = pd.read_csv(RUN/"per-fold.csv")
    assert not rows.duplicated(["dataset", "fold", "budget", "method"]).any()
    rows["retained"] = rows.k / rows["columns"]
    rows["fallback"] = rows.k == rows.anchor_k
    fields = ["safe_saved", "retained", "failure", "fallback", "auc", "accuracy", "masks", "guard_candidates",
              "total_s", "vote_s", "guard_s", "final_s"]
    human = rows[rows.dataset == "optdigits"]
    human.to_csv(RUN/"optdigits.csv", index=False)
    data = rows[rows.dataset != "optdigits"]
    assert len(data) == 41*9 and len(human) == 9
    means = data.groupby(["dataset", "budget", "method"])[fields].mean()
    means.to_csv(RUN/"by-task.csv")
    avg = means.groupby(["budget", "method"]).mean()
    counts = data.groupby(["budget", "method"]).agg(outer_failures=("failure", "sum"), fallbacks=("fallback", "sum"))
    avg.join(counts).to_csv(RUN/"summary.csv")
    records, contrasts = [], []
    for budget in ("one", "three"):
        subset = data[data.budget == budget]
        for tie in ("gain", "lottery"):
            plain, voting = (subset[subset.method == m+"_"+tie].set_index(["dataset", "fold"]).sort_index()
                             for m in ("fraction", "vote"))
            assert len(plain) == 41 and plain.index.equals(voting.index)
            assert np.array_equal(plain.guard_candidates, voting.guard_candidates)
            for idx in plain.index:
                a, b = plain.loc[idx], voting.loc[idx]
                records.append(dict(dataset=idx[0], fold=idx[1], budget=budget, tie=tie,
                                    same_nominees=a.shortlist_indices == b.shortlist_indices,
                                    same_final=a.selected_index == b.selected_index, prefix_k=a.k, vote_k=b.k,
                                    prefix_fail=a.failure, vote_fail=b.failure, prefix_auc=a.auc, vote_auc=b.auc,
                                    prefix_acc=a.accuracy, vote_acc=b.accuracy, gain_saved=b.safe_saved-a.safe_saved,
                                    extra_time=b.total_s-a.total_s))
            delta = voting[fields].astype(float) - plain[fields].astype(float)
            task = delta.groupby("dataset").mean()
            draw = np.random.default_rng(33119).integers(len(task), size=(30000,len(task)))
            for field in fields:
                values = task[field].to_numpy()
                lo, hi = np.quantile(values[draw].mean(axis=1), [.025, .975])
                contrast = delta[field]
                contrasts.append(dict(budget=budget, tie=tie, metric=field, delta=values.mean(),
                                      ci95lo=lo, ci95hi=hi, pos_tasks=int((values>0).sum()),
                                      neg_tasks=int((values<0).sum()), pos_folds=int((contrast>0).sum()),
                                      neg_folds=int((contrast<0).sum())))
    pd.DataFrame(records).to_csv(RUN/"paired-decisions.csv", index=False)
    out = pd.DataFrame(contrasts)
    out.to_csv(RUN/"paired-effects.csv", index=False)
    print(out[out.metric.isin(["safe_saved", "auc", "total_s"])].to_string(index=False))


if __name__ == "__main__":
    main()
