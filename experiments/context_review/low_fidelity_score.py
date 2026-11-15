"""Score the sealed cheap-refit control and feasible one-refit opportunities."""
import numpy as np
import pandas as pd

from experiments.context_review import low_fidelity_control as control
from experiments.context_review.budgeted_guard import digest, metrics, read
from experiments.context_review.refit_opportunity import outcome
from experiments.context_review.transport_score import verify_lock


def main():
    keys = verify_lock()
    recorded = read(control.transport.EVIDENCE/"low-fidelity-lock.json")
    assert recorded == read(control.OUT/"cohort-sealed.json") and set(recorded) == set(keys)
    for key in keys:
        path = control.OUT/key
        assert digest(path/"sealed.json") == recorded[key]
        assert read(path/"sealed.json")["predictions"] == digest(path/"predictions.npz")
    candidate_rows, rows = [], []
    for key in keys:
        path = control.OUT/key
        src = control.transport.RUN/key
        r = read(path/"sealed.json")
        primary = read(src/"selection.json")
        cols = r["supports"]
        assert r["layout"] == digest(src/"layout.json") == primary["layout"]
        assert set(map(tuple,cols[1:])) == {tuple(t["columns"]) for t in primary["pool"]}
        guard = np.array([outcome(s,r["guard_metrics"][0],.03) for s in r["guard_metrics"]])
        nominees = {"small":r["low_fidelity"]}
        with np.load(path/"predictions.npz") as p, np.load(src/"predictions.npz") as previous:
            np.testing.assert_array_equal(previous["anchor"],p["o0"])
            for method, decision in primary["choices"].items():
                assert cols[0] == decision["anchor"]
                i = cols.index(decision["nominee"])
                final = i if guard[i] else 0
                assert cols[final] == decision["selected"]
                np.testing.assert_array_equal(previous[method],p[f"o{final}"])
                nominees[method] = i
            with np.load(src/"targets.npz") as truth:
                np.testing.assert_array_equal(truth["test_row"],previous["test_row"])
                scores = [metrics(truth["y"],p[f"o{i}"],r["rare"]) for i in range(len(cols))]
        passing = np.array([outcome(s,scores[0],.03) for s in scores])
        reachable = (passing & guard)[1:].any()
        possible = passing[1:].any()
        for i, support in enumerate(cols):
            candidate_rows.append(dict(dataset=r["dataset"],fold=r["fold"],idx=i,columns=" ".join(map(str,support)),
                                       guard_pass=guard[i],outer_pass=passing[i],**scores[i]))
        d = 1 - len(cols[-1])/len(cols[0])
        for method,i in nominees.items():
            final = i if guard[i] else 0
            disposition = ("singleton" if not i else "no_outer_pass" if not possible else "guard_excludes_all"
                           if not reachable else "missed_nomination" if not (guard[i] and passing[i]) else "realized")
            rows.append(dict(dataset=r["dataset"],fold=r["fold"],method=method,nominee=i,selected=final,
                             nominal=d,outer_ceiling=d*possible,joint_ceiling=d*reachable,disposition=disposition,
                             u=d*bool(final and passing[final]),breach=not passing[final],accepted=bool(final),
                             small_fits=r["fits_8"] if method=="small" else 0,
                             small_fit_seconds=r["seconds_8"] if method=="small" else 0.,**scores[final]))
    dest = control.EVIDENCE
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(dest/"supports.csv",index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(dest/"folds.csv",index=False)
    fields=["nominal","outer_ceiling","joint_ceiling","u","breach","accepted","auc","accuracy","small_fits","small_fit_seconds"]
    means=frame.groupby(["method","dataset"])[fields].mean()
    means.groupby("method").mean().to_csv(dest/"summary.csv")
    diffs=[]
    for method in ("gain","vote","slack","point"):
        paired=means.loc["small"]-means.loc[method]
        draws=np.random.default_rng(33119).integers(len(paired),size=(30000,len(paired)))
        for field in ("u","breach","auc","accuracy"):
            x=paired[field].to_numpy()
            lo,hi=np.percentile(x[draws].mean(axis=1),[2.5,97.5])
            diffs.append(dict(method="small",reference=method,field=field,delta=x.mean(),lo95=lo,hi95=hi))
    pd.DataFrame(diffs).to_csv(dest/"contrasts.csv",index=False)
    print(means.groupby("method").mean().round(4).to_string())


if __name__=="__main__":
    main()
