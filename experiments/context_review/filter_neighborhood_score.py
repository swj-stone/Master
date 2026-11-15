"""Replay both return rules for the frozen ANOVA one-exchange forest experiment."""
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.filter_neighborhood import DESIGN, DEST, ROOT
from experiments.context_review.forest_replay import (
    comparisons, diagnostics, read_member, read_metrics, tensor_scores,
)
from experiments.context_review.refit_opportunity import outcome
from experiments.context_review.receiver_transport import frozen_folds


ARMS = ("prefix", "uniform", "vote", "slack", "point")
PAIRS = (("slack", "vote"), ("slack", "point"), ("slack", "uniform"),
         ("point", "vote"), ("vote", "prefix"), ("point", "prefix"))
METRICS = ("u", "breach", "saved", "auc", "accuracy", "ap")


def support_law(selected, accepted, guarded, size):
    result = np.zeros(size)
    for i in selected:
        final = i if not guarded or accepted[i] else 0
        result[final] += 1 / len(selected)
    return result


def fold_rows(forecast, truth, key, rec, standalone):
    ident = {k: rec[k] for k in ("panel", "dataset", "fold")}
    supports = rec["supports"]
    n = len(supports)
    with read_member(forecast, key, rec["member_sha"]) as arrays, read_member(truth,key,rec["targets_sha"]) as target:
        np.testing.assert_array_equal(arrays["test_row"], target["test_row"])
        rare = "ap" in rec["metrics"]
        check = dict(rec, winners=dict(rec["winners"]))
        if n>1:
            # Single-panel scoring and the four-panel matrix can incur
            # different last-bit rounding on the first weight vector.
            tensor = arrays["tensor"][:,0,0]
            frac = np.array([.05 if m=="ap" else .03 for m in rec["metrics"]])
            allowance = frac*abs(tensor[0])
            first_panel = ((tensor[1:]-tensor[0]+allowance+1e-12)/np.maximum(allowance,1e-12)).min(axis=1)
            check["winners"]["point"] = int(1+np.argmax(first_panel))
            np.testing.assert_allclose(standalone,first_panel,rtol=0,atol=1e-10)
            assert int(1+np.argmax(standalone))==rec["winners"]["point"]
        scores = tensor_scores(arrays, check)
        for row,point in zip(scores,standalone,strict=True):
            row["point"] = point
        measures = read_metrics(arrays,arrays["guard_y"],target["y"],n,rare)
    accepted = np.array([outcome(g,measures[0][0],.03) for g,o in measures])
    passed = np.array([outcome(o,measures[0][1],.03) for g,o in measures])
    savings = np.array([1-len(s)/len(supports[0]) for s in supports])
    vals = dict(u=savings*passed,breach=~passed,saved=savings)
    for m in ("auc","accuracy","ap"):
        vals[m] = np.array([o.get(m,np.nan) for _,o in measures])
    options = list(range(1,n)) or [0]
    nominations = dict(prefix=[options[0]],uniform=options,**{arm:[v] for arm,v in rec["winners"].items()})
    assert all(i in options for subset in nominations.values() for i in subset)
    assert sorted(supports[0]) == sorted(rec["order"][:len(supports[0])])
    if n>1:
        kept = len(supports[1])
        assert np.all(savings[1:]==savings[1]) and supports[1]==sorted(rec["order"][:kept])
        assert all(len(set(c)^set(supports[1]))==2 for c in supports[2:])
    candidate_records = [dict(**ident,candidate=i,columns=" ".join(map(str,supports[i])), **row,
                             guard=accepted[i],outer=passed[i],guard_auc=measures[i][0]["auc"],
                             guard_accuracy=measures[i][0]["accuracy"],outer_auc=vals["auc"][i],
                             outer_accuracy=vals["accuracy"][i],outer_ap=vals["ap"][i])
                         for i,row in enumerate(scores,1)]
    tie_row = dict(**ident,**diagnostics(scores),changed=rec["winners"]["vote"]!=rec["winners"]["slack"],
                   joint_possible=bool(np.any(accepted[1:] & passed[1:])),any_outer=bool(np.any(passed[1:])),
                   anchor=len(supports[0]),retained=len(supports[options[0]]),
                   parents=rec["timings"]["voter_fit_s"],ranking_s=rec["timings"]["ranking_s"])
    records, bounds, costs = [], [], []
    timings = rec["timings"]
    for delivery in ("guarded","unchecked"):
        checked = delivery=="guarded"
        laws = {arm:support_law(nominees,accepted,checked,n) for arm,nominees in nominations.items()}
        for arm, law in laws.items():
            records.append(dict(**ident,delivery=delivery,method=arm,
                **{metric:float(p @ law) for metric,p in vals.items()},deleted_return=1-law[0]))
            check = timings["nomination"].get(arm,dict(masks=0,rows=0,seconds=0.))
            nominee = nominations[arm]
            # The outer anchor is an evaluation reference on all folds. The
            # deployment cost below counts only the actually returned model.
            candidate_t = np.array(timings["guard_support_s"])
            guarded_time = 0.
            if checked and n>1:
                guarded_time = candidate_t[0] + np.mean(candidate_t[nominee])
            costs.append(dict(**ident,delivery=delivery,method=arm,**check,ranking_s=timings["ranking_s"],
                diagnostic_receiver_fits=2*n,all_guard_s=timings["all_guard_s"],all_outer_s=timings["all_outer_s"],
                guard_fits=2*int(checked and n>1),guard_fit_predict_s=guarded_time,
                voter_fits=int(arm in rec["winners"]),voter_fit_s=timings["voter_fit_s"] if arm in rec["winners"] else 0.,
                returned_fit_predict_s=np.dot(law,timings["outer_support_s"])))
        for a,b in PAIRS:
            coefficients = (laws[a]-laws[b])*savings
            lower = coefficients[coefficients<0].sum()
            upper = coefficients[coefficients>0].sum()
            observed = vals["u"] @ (laws[a]-laws[b])
            assert lower-1e-12 <= observed <= upper+1e-12
            bounds.append(dict(**ident,delivery=delivery,method=a,reference=b,
                               lower=lower,upper=upper,observed=observed,
                               return_changed=int(not np.array_equal(laws[a],laws[b]))))
    return records, candidate_records, tie_row, bounds, costs


def replay():
    EVIDENCE = ROOT / "thesis-paper/evidence"
    lock = bg.read(DEST/"prediction-lock.json")
    standalone = bg.read(DEST/"point-scores.json")
    assert bg.digest(DEST/"prediction-lock.json")==standalone["prediction_seal"]
    assert lock["design_sha"] == bg.digest(DESIGN)
    assert lock["archive_sha"] == bg.digest(DEST/"predictions.zip")
    expected = {f"{panel}/{key}": h for panel,key,h in frozen_folds()}
    assert expected.keys()==lock["folds"].keys()
    fold, candidates, ties, envelope, costs = [], [], [], [], []
    with ZipFile(DEST/"predictions.zip") as forecast, ZipFile(EVIDENCE/"receiver/outcomes.zip") as truth:
        for key,rec in lock["folds"].items():
            assert rec["source_sha"] == expected[key]
            r,c,t,e,co = fold_rows(forecast,truth,key,rec,standalone["scores"][key])
            fold.extend(r)
            candidates.extend(c)
            ties.append(t)
            envelope.extend(e)
            costs.extend(co)
    folds = pd.DataFrame(fold)
    groups = ["panel","delivery"]
    tasks = folds.groupby([*groups,"dataset","method"],as_index=False)[list(METRICS)].mean()
    means = tasks.groupby([*groups,"method"],as_index=False)[list(METRICS)].mean()
    contrasts = comparisons(tasks,groups,PAIRS,("u","breach","auc","accuracy"))
    env = pd.DataFrame(envelope)
    by_task = env.groupby([*groups,"dataset","method","reference"],as_index=False)[["lower","upper","observed","return_changed"]].mean()
    summary = by_task.groupby([*groups,"method","reference"],as_index=False)[["lower","upper","observed","return_changed"]].mean()
    for name, frame in (("policies",folds),("candidates",pd.DataFrame(candidates)),("ties",pd.DataFrame(ties)),
                        ("tasks",tasks),("means",means),("contrasts",contrasts),("costs",pd.DataFrame(costs)),
                        ("fold-bounds",env),("task-bounds",by_task),("bounds",summary)):
        frame.to_csv(DEST/(name+".csv"),index=False)
    print(means[["panel","delivery","method","u","breach"]].to_string(index=False))


if __name__ == "__main__":
    replay()
