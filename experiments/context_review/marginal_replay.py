"""Join sealed empirical-background nominations to previously measured guard/outer outcomes."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.context_review import budgeted_guard as bg
from experiments.context_review.marginal_background import DESIGN, DEST, EVIDENCE, REPLICATES, ROOT, SOURCE
from experiments.context_review.receiver_transport import frozen_folds
from experiments.context_review.refit_opportunity import outcome


METHODS = ("point", "slack")
METRICS = ("u", "breach", "auc", "accuracy")


def xgb_observations(panel, key, record, seal):
    slate = "low-fidelity" if panel == "first" else "wide-slate"
    folder = ROOT / "results/context_review" / slate / key
    src = ROOT / "results/context_review" / SOURCE[panel] / key
    state = bg.read(folder / "sealed.json")
    assert bg.digest(folder / "sealed.json") == seal and state["predictions"] == bg.digest(folder / "predictions.npz")
    layout = bg.read(src / "layout.json")
    assert state["layout"] == bg.digest(src / "layout.json") and layout["targets_sha256"] == bg.digest(src / "targets.npz")
    assert state["supports"] == record["supports"]
    with np.load(folder / "predictions.npz") as predictions, np.load(src / "targets.npz") as truth:
        results = [bg.metrics(truth["y"], predictions[f"o{i}"], state["rare"]) for i in range(len(record["supports"]))]
    guard = state["guard_metrics"]
    return [(outcome(g, guard[0], .03), outcome(m, results[0], .03), m["auc"], m["accuracy"])
            for m, g in zip(results, guard, strict=True)]


def logistic_observations(record, candidates):
    key = record["panel"], record["dataset"], record["fold"]
    group = candidates.loc[key].sort_index()
    supports = [[int(i) for i in str(s).split()] for s in group.columns_text]
    assert record["supports"] == supports
    return list(group[["guard_ok", "outer_ok", "outer_auc", "outer_accuracy"]].itertuples(index=False, name=None))


def policy_results(supports, observations, nominees):
    finals = []
    for i, (guard, _, _, _) in enumerate(observations):
        j = i if guard else 0
        _, passed, auc, accuracy = observations[j]
        saved = 1 - len(supports[j]) / len(supports[0])
        finals.append(dict(u=saved*passed, breach=int(not passed), auc=auc, accuracy=accuracy))
    rows = pd.DataFrame(finals)
    result = {arm: rows.iloc[j].to_dict() for arm, j in nominees.items()}
    result["uniform"] = (rows.iloc[1:] if len(rows)>1 else rows).mean().to_dict()
    return result


def assert_controls(folds):
    keys = ["panel", "dataset", "fold", "method"]
    data = pd.concat([pd.read_csv(EVIDENCE / f"chance/{panel}-folds.csv").query("tolerance == .03").assign(panel=panel)
                      for panel in SOURCE])
    logistic = pd.read_csv(EVIDENCE / "receiver/folds.csv")
    for receiver, reference in (("xgb", data), ("logistic", logistic)):
        subset = folds.loc[(folds.receiver == receiver) & folds.method.isin(["uniform", "median_point", "median_slack"])].copy()
        subset.method = subset.method.str.removeprefix("median_")
        expected = reference.loc[reference.method.isin(["uniform", "point", "slack"])].set_index(keys).sort_index()
        actual = subset.set_index(keys).sort_index()
        pd.testing.assert_index_equal(actual.index, expected.index)
        np.testing.assert_allclose(actual[list(METRICS)], expected[list(METRICS)], atol=1e-12, rtol=0)


def replay():
    lock = bg.read(DEST / "nomination-lock.json")
    assert lock["design"] == bg.digest(DESIGN)
    source = Path(__file__).with_name("marginal_background.py")
    assert lock["implementation"] == bg.digest(source)
    targets = {f"{p}/{k}":sha for p,k,sha in frozen_folds()}
    assert set(targets) == set(lock["folds"])
    logistic = (pd.read_csv(EVIDENCE / "receiver/candidates.csv", dtype={"columns": str})
                .rename(columns={"columns": "columns_text"}).set_index(["panel", "dataset", "fold", "index"]).sort_index())
    seals = {"first": bg.read(EVIDENCE / "transport/low-fidelity-lock.json"),
             "wide": bg.read(EVIDENCE / "wide_slate/prediction-lock.json")}
    fold_rows, agreement, costs = [], [], []
    for full_key, record in lock["folds"].items():
        panel, key = full_key.split("/", 1)
        assert record["selection"] == targets[full_key]
        ident = {field: record[field] for field in ("panel", "dataset", "fold")}
        supports = record["supports"]
        nominees = {"median_"+method: i for method, i in record["median_nominees"].items()}
        for method in METHODS:
            rank_scores = []
            for rep in REPLICATES:
                info = record["replicates"][rep].get(method)
                i = 0 if info is None else 1 + int(np.argmax(info["scores"]))
                assert i == (0 if info is None else info["nominee"])
                nominees[rep+"_"+method] = i
                if info is not None:
                    rank_scores.append(rankdata(info["scores"]))
                costs.append(dict(**ident, replicate=rep, method=method,
                                  **{field: 0 if info is None else info[field] for field in ("masks", "model_rows", "seconds")}))
            if len(supports)>1:
                r, s = rank_scores
                rho = np.corrcoef(r, s)[0,1] if np.var(r)>0 and np.var(s)>0 else np.nan
                agreement.append(dict(**ident, method=method, candidates=len(supports)-1, spearman=rho,
                                      same_winner=nominees["A_"+method] == nominees["B_"+method]))
        observations = {"xgb": xgb_observations(panel, key, record, seals[panel][key]),
                        "logistic": logistic_observations(record, logistic)}
        for receiver, values in observations.items():
            results = policy_results(supports, values, nominees)
            fold_rows.extend(dict(**ident, receiver=receiver, method=arm, **metrics) for arm, metrics in results.items())
    costs = pd.DataFrame(costs)
    costs.to_csv(DEST / "costs.csv", index=False)
    pd.DataFrame(agreement).to_csv(DEST / "agreement.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    assert_controls(folds)
    folds.to_csv(DEST / "folds.csv", index=False)
    tasks = folds.groupby(["receiver", "panel", "dataset", "method"])[list(METRICS)].mean()
    tasks.to_csv(DEST / "tasks.csv")
    means = tasks.groupby(["receiver", "panel", "method"]).mean()
    means.to_csv(DEST / "means.csv")
    contrasts = []
    for (receiver, panel), group in tasks.groupby(level=["receiver", "panel"]):
        table = group.droplevel(["receiver", "panel"])
        for rep in REPLICATES:
            for method in METHODS:
                arm = rep+"_"+method
                refs = ["uniform", "median_"+method]
                if method=="slack":
                    refs.append(rep+"_point")
                for ref in refs:
                    deltas = table.xs(arm, level="method") - table.xs(ref, level="method")
                    n = len(deltas)
                    draws = np.random.default_rng(33119).integers(n, size=(30_000,n))
                    for metric in ("u", "breach"):
                        values = deltas[metric].to_numpy()
                        lo, hi = np.percentile(values[draws].mean(axis=1), [2.5,97.5])
                        contrasts.append(dict(receiver=receiver, panel=panel, method=arm, reference=ref, metric=metric,
                                              mean=values.mean(), lo95=lo, hi95=hi, n_tasks=n))
    pd.DataFrame(contrasts).to_csv(DEST / "contrasts.csv", index=False)
    print(means.round(4))


if __name__ == "__main__":
    replay()
