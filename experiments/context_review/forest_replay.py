"""Score the crossed forest experiment from frozen nominees, transcripts and probabilities."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.decision_opportunities import EVIDENCE, PHASES, RUNS
from experiments.context_review.forest_predict import DESIGN, DEST, SCORES
from experiments.context_review.refit_opportunity import outcome


POLICY_METRICS = ["u", "breach", "accepted", "saved", "auc", "accuracy", "ap"]
CONTRASTS = (("slack", "vote"), ("point", "vote"), ("slack", "point"),
             ("slack", "gain"), ("vote", "gain"), ("slack", "uniform"))


def read_member(archive, name, expected):
    data = archive.read(name+".npz")
    assert sha256(data).hexdigest() == expected
    return np.load(BytesIO(data))


def tensor_scores(arrays, rec):
    """Independent recovery of all ballot and continuous scores from the measured metrics."""
    tensor = arrays["tensor"]
    n = len(rec["supports"]) - 1
    assert tensor.shape[0] == n + 1
    if not n:
        assert not tensor.shape[1]
        return []
    assert np.isfinite(tensor).all() and tensor.shape[1] >= 1 and tensor.shape[2] == 4
    fractions = np.array([.05 if name == "ap" else .03 for name in rec["metrics"]])
    allowance = fractions * abs(tensor[0])
    change = tensor[1:] - tensor[0]
    approvals = np.all(-change <= allowance+1e-12, axis=3)
    votes = approvals.sum(axis=2).min(axis=1)
    margins = ((change + allowance + 1e-12) / np.maximum(allowance, 1e-12)).min(axis=3)
    slacks = np.sort(margins, axis=2)[:,:,1].min(axis=1)
    points = margins[:,0,0]
    np.testing.assert_array_equal(votes >= 3, slacks >= 0)
    scores = [dict(vote=int(v), slack=m, point=p) for v,m,p in zip(votes,slacks,points,strict=True)]
    for arm in SCORES:
        assert rec["winners"][arm] == 1+int(np.argmax([r[arm] for r in scores]))
    return scores


def read_metrics(arrays, yguard, youter, size, rare):
    metrics = []
    for i in range(size):
        pair = []
        for pre, truth in (("g", yguard), ("o", youter)):
            probabilities = arrays[f"{pre}{i}"]
            assert np.isfinite(probabilities).all() and probabilities.shape[0] == len(truth)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1., rtol=1e-6, atol=1e-6)
            pair.append(bg.metrics(truth, probabilities, rare))
        metrics.append(pair)
    return metrics


def nomination_rows(rows):
    if not rows:
        return {name:[0] for name in (*SCORES,"gain","uniform","vote_slack","top_uniform")}
    ballot = np.array([r["vote"] for r in rows])
    best = np.flatnonzero(ballot==ballot.max())
    by_slack = best[np.argmax([rows[i]["slack"] for i in best])]
    choices = {arm:[1+int(np.argmax([r[arm] for r in rows]))] for arm in SCORES}
    return {**choices, "gain":[1], "uniform":list(range(1,len(rows)+1)),
            "vote_slack":[1+by_slack], "top_uniform":list(best+1)}


def policy_values(supports, metrics):
    ref_guard, ref_outer = metrics[0]
    values = []
    for cols, (guard, outer) in zip(supports, metrics, strict=True):
        g, o = outcome(guard,ref_guard,.03), outcome(outer,ref_outer,.03)
        final = outer if g else ref_outer
        saved = (1-len(cols)/len(supports[0])) if g else 0.
        values.append(dict(guard=g, outer=o, joint=g and o, u=saved*o, breach=float(g and not o),
                           saved=saved, accepted=float(saved>0), auc=final["auc"],
                           accuracy=final["accuracy"], ap=final.get("ap",np.nan)))
    return pd.DataFrame(values)


def diagnostics(scores):
    n = len(scores)
    ballot = np.array([r["vote"] for r in scores])
    margin = np.array([r["slack"] for r in scores])
    i,j = np.triu_indices(n, k=1)
    dv, dm = ballot[i]-ballot[j], margin[i]-margin[j]
    return dict(candidates=n, pairs=len(i), tied_vote_pairs=int(np.sum(dv==0)),
                split_ties=int(np.sum((dv==0)&(dm!=0))),
                strictly_reversed=int(np.sum(dv*dm<0)),
                best_vote_tied=int(n>0 and (ballot==max(ballot)).sum()>1),
                best_margin_tied=int(n>0 and (margin==max(margin)).sum()>1))


def rank_records(scores, values):
    """Only mixed folds enter a concordance. Tied-ballot margins probe within-count order."""
    n = len(scores)
    scores = pd.DataFrame(scores, columns=SCORES)
    scores["gain"] = -np.arange(n)  # same within-anchor rank sum and column-index ordering as in the candidate pool
    rows = []
    for target in ("outer", "joint"):
        y = values[target].to_numpy(dtype=bool)[1:]
        positives, negatives = np.nonzero(y)[0], np.nonzero(~y)[0]
        for method in (*SCORES, "gain", "margin_among_equal_votes"):
            z = scores["slack" if method == "margin_among_equal_votes" else method].to_numpy()
            diff = z[positives,None]-z[negatives][None,:]
            if method == "margin_among_equal_votes":
                v = scores["vote"].to_numpy()
                diff = diff[v[positives,None] == v[negatives][None,:]]
            concordance = np.mean((diff>0)+.5*(diff==0)) if diff.size else np.nan
            if diff.size and method != "margin_among_equal_votes":
                np.testing.assert_allclose(concordance, roc_auc_score(y,z), atol=1e-12)
            rows.append(dict(target=target, method=method, concordance=concordance,
                             pairs=diff.size, passing=len(positives), candidates=n))
    return rows


def tables():
    lock = bg.read(DEST/"prediction-lock.json")
    hist = bg.read(EVIDENCE/"receiver/prediction-lock.json")
    assert lock["design_sha"] == bg.digest(DESIGN)
    assert lock["source_sha"] == bg.digest(Path(__file__).with_name("forest_predict.py"))
    assert lock["archive_sha"] == bg.digest(DEST/"predictions.zip")
    assert hist["forecast_archive_sha"] == bg.digest(EVIDENCE/"receiver/predictions.zip")
    assert len(lock["folds"]) == 46 and set(lock["folds"]) == set(hist["folds"])
    seals = {p:bg.read(EVIDENCE/val[2]) for p,val in PHASES.items()}
    folds, ranked, pairs, candidates, costs = [], [], [], [], []
    with ZipFile(DEST/"predictions.zip") as forest, ZipFile(EVIDENCE/"receiver/predictions.zip") as logit, \
         ZipFile(EVIDENCE/"receiver/outcomes.zip") as truth:
        for name, rec in lock["folds"].items():
            panel,key = name.split("/",1)
            ident = {k:rec[k] for k in ("panel","dataset","fold")}
            source, slates, _ = PHASES[panel]
            sel_path = RUNS/source/key/"selection.json"
            slate_path = RUNS/slates/key/"sealed.json"
            assert bg.digest(sel_path) == rec["selection_sha"] and bg.digest(slate_path) == seals[panel][key]
            select, slate, logrec = bg.read(sel_path), bg.read(slate_path), hist["folds"][name]
            assert rec["supports"] == logrec["supports"] == slate["supports"]
            supports = rec["supports"]
            rare = "ap" in rec["metrics"]
            assert rare == slate["rare"] == logrec["rare"]
            assert rec["targets_sha"] == logrec["targets_sha"]
            assert bg.digest(slate_path.with_name("predictions.npz")) == slate["predictions"]
            with read_member(forest,name,rec["member_sha"]) as fa, read_member(logit,name,logrec["predictions_sha"]) as la, \
                 read_member(truth,name,rec["targets_sha"]) as ta, np.load(slate_path.with_name("predictions.npz")) as xa:
                np.testing.assert_array_equal(fa["test_row"],ta["test_row"])
                np.testing.assert_array_equal(fa["test_row"],la["test_row"])
                np.testing.assert_array_equal(fa["guard_y"],la["guard_y"])
                np.testing.assert_array_equal(fa["guard"],la["guard"])
                forest_scores = tensor_scores(fa,rec)
                receivers = {n:read_metrics(a,fa["guard_y"],ta["y"],len(supports),rare)
                             for n,a in (("forest",fa),("xgb",xa),("logistic",la))}
            for i, ref in enumerate(slate["guard_metrics"]):
                for metric in ref:
                    np.testing.assert_allclose(receivers["xgb"][i][0][metric],ref[metric],rtol=0,atol=1e-12)
            for method, t in rec["timings"]["nomination"].items():
                costs.append(dict(**ident, method=method, **t, voter_fit_s=rec["timings"]["voter_fit_s"],
                                  all_guard_s=rec["timings"]["all_guard_s"], all_outer_s=rec["timings"]["all_outer_s"],
                                  receiver_fits=2*len(supports)))
            for voter, rows in (("forest",forest_scores),("xgb",select["pool"])):
                winners = nomination_rows(rows)
                pairs.append(dict(**ident,voter=voter,**diagnostics(rows),
                                  changed_nominee=int(winners["vote"] != winners["slack"])))
                for receiver, metrics in receivers.items():
                    info = dict(**ident, voter=voter,receiver=receiver)
                    v = policy_values(supports,metrics)
                    for method, selected in winners.items():
                        result = v.iloc[selected][POLICY_METRICS].mean().to_dict()
                        folds.append(dict(**info,method=method,**result))
                    ranked.extend(dict(**info,**entry) for entry in rank_records(rows,v))
                    for i, score in enumerate(rows,1):
                        candidates.append(dict(**info,index=i,columns=" ".join(map(str,supports[i])),
                                               **{k:score[k] for k in SCORES},guard=v.at[i,"guard"],
                                               outer=v.at[i,"outer"],joint=v.at[i,"joint"]))
    return tuple(map(pd.DataFrame,(folds,ranked,pairs,candidates,costs)))


def bootstrap(x):
    if not len(x):
        return np.nan, np.nan, np.nan
    draws = np.random.default_rng(33119).integers(len(x),size=(30000,len(x)))
    low,high = np.percentile(x[draws].mean(axis=1),[2.5,97.5])
    return x.mean(),low,high


def comparisons(tasks, group_keys, pairs, metrics):
    output = []
    for group,data in tasks.groupby(group_keys):
        for metric in metrics:
            values = data.pivot(index="dataset",columns="method",values=metric)
            for a,b in pairs:
                delta = (values[a]-values[b]).dropna().to_numpy()
                est,low,high = bootstrap(delta)
                output.append(dict(**dict(zip(group_keys,group,strict=True)),method=a,reference=b,metric=metric,
                                   difference=est,low95=low,high95=high,contributing_tasks=len(delta),
                                   higher=int(np.sum(delta>1e-12)),lower=int(np.sum(delta < -1e-12))))
    return pd.DataFrame(output)


def check_archived_controls(folds):
    """The XGBoost voter also reproduces the previously published decision policies."""
    arms = ("gain","vote","point","slack","uniform")
    identity = ["panel","dataset","fold","method"]
    logistic = pd.read_csv(EVIDENCE/"receiver/folds.csv")
    xgb = []
    for panel in PHASES:
        frame = pd.read_csv(EVIDENCE/"chance"/(panel+"-folds.csv"))
        xgb.append(frame.loc[frame.tolerance.eq(.03)].assign(panel=panel))
    for receiver, archive in (("logistic",logistic),("xgb",pd.concat(xgb))):
        old = archive.loc[archive.method.isin(arms)].set_index(identity).sort_index()
        new = folds.loc[folds.voter.eq("xgb") & folds.receiver.eq(receiver)
                        & folds.method.isin(arms)].set_index(identity).sort_index()
        assert old.index.is_unique and len(old) == 230 and old.index.equals(new.index)
        for key in ("u","breach","auc","accuracy","saved"):
            np.testing.assert_allclose(old[key],new[key],atol=1e-12,rtol=0)


def replay():
    folds, ranks, collisions, candidates, costs = tables()
    check_archived_controls(folds)
    keys = ["panel","voter","receiver"]
    tasks = folds.groupby([*keys,"dataset","method"],as_index=False)[POLICY_METRICS].mean()
    means = tasks.groupby([*keys,"method"],as_index=False)[POLICY_METRICS].mean()
    contrasts = comparisons(tasks,keys,CONTRASTS,("u","breach","auc","accuracy"))
    rkeys = [*keys,"target","method"]
    rtask = ranks.groupby([*rkeys,"dataset"],as_index=False).agg(
        concordance=("concordance","mean"),mixed=("concordance","count"),folds=("fold","size"),pairs=("pairs","sum"))
    rmean = rtask.groupby(rkeys,as_index=False).agg(concordance=("concordance","mean"),
        contributing_tasks=("concordance","count"),mixed=("mixed","sum"),folds=("folds","sum"),pairs=("pairs","sum"))
    rcomp = comparisons(rtask,[*keys,"target"],(("slack","vote"),("point","vote"),("slack","gain")),
                        ("concordance",))
    for name, frame in (("policies",folds),("tasks",tasks),("means",means),("contrasts",contrasts),
                        ("rank-folds",ranks),("rank-tasks",rtask),("rank-means",rmean),("rank-contrasts",rcomp),
                        ("score-collisions",collisions),("candidates",candidates),("costs",costs)):
        frame.to_csv(DEST/(name+".csv"),index=False)
    print(f"{len(folds)} policy/folds; {len(candidates)} crossed candidate records")
    print(means.query("voter=='forest' and receiver=='forest'")[["panel","method","u","breach"]].to_string(index=False))


if __name__ == "__main__":
    replay()
