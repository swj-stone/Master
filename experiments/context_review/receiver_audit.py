"""Refit the logistic receiver per support and check scored candidate and policy records."""
import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.context_review.receiver_transport import DEST, ROOT, SOURCE
from experiments.context_study.benchmark import store


def measure(y, p, rare):
    def area(binary, scores):
        positives = np.count_nonzero(binary)
        return (rankdata(scores)[binary].sum() - positives*(positives + 1)/2) / (positives*(len(y)-positives))
    assert np.array_equal(np.unique(y), np.arange(p.shape[1]))
    auc = area(y == 1, p[:, 1]) if p.shape[1] == 2 else np.mean([area(y == c, p[:, c]) for c in range(p.shape[1])])
    scores = dict(auc=auc,
                  accuracy=(p.argmax(axis=1) == y).mean())
    if rare:
        ids = np.argsort(-p[:, 1], kind="stable")
        positives = np.cumsum(y[ids])
        last = np.r_[np.diff(p[ids, 1]) != 0, True]
        count = positives[last]
        precision = count / np.arange(1, len(y) + 1)[last]
        scores["ap"] = np.dot(np.diff(np.r_[0, count]), precision) / positives[-1]
    return scores


def passes(actual, anchor):
    return all(actual[k] + 1e-12 >= val - abs(val)*(.05 if k == "ap" else .03) for k, val in anchor.items())


def audit():
    lock = json.loads((DEST / "prediction-lock.json").read_bytes())["folds"]
    candidates = pd.read_csv(DEST / "candidates.csv").set_index(["panel", "dataset", "fold", "index"])
    policies = pd.read_csv(DEST / "folds.csv").set_index(["panel", "dataset", "fold", "method"])
    n_refits = n_folds = n_records = 0
    max_diff = max_metric_diff = 0.
    differing_decisions = []
    with ZipFile(DEST / "predictions.zip") as predictions, ZipFile(DEST / "outcomes.zip") as outcomes, threadpool_limits(limits=1):
        for path, info in lock.items():
            panel, key = path.split("/", 1)
            src = ROOT / "results/context_review" / SOURCE[panel] / key
            with np.load(src / "inputs.npz") as fold, np.load(BytesIO(predictions.read(path+".npz"))) as cached, np.load(BytesIO(outcomes.read(path+".npz"))) as targets:
                partition = np.sort(np.r_[fold["train"], fold["proposal"]])
                x, y = fold["x"], fold["y"]
                dev = np.arange(len(y))
                test_cases = (("g", partition, x[fold["guard"]], y[fold["guard"]]),
                              ("o", dev, fold["x_outer"], targets["y"]))
                guard, outer, fresh_guard, fresh_outer = [], [], [], []
                for prefix, fit_rows, eval_x, eval_y in test_cases:
                    for i, cols in enumerate(info["supports"]):
                        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                              StandardScaler(), LogisticRegression(C=1., solver="lbfgs", tol=1e-6, max_iter=2000))
                        model.fit(np.asarray(x[fit_rows][:, cols], dtype="float64"), y[fit_rows])
                        fresh = model.predict_proba(np.asarray(eval_x[:, cols], dtype="float64"))
                        stored = cached[prefix + str(i)]
                        diff = abs(stored - fresh).max()
                        max_diff = max(max_diff, float(diff))
                        n_refits += 1
                        values = measure(eval_y, stored, info["rare"])
                        independent = measure(eval_y, fresh, info["rare"])
                        max_metric_diff = max(max_metric_diff, max(abs(independent[k] - v) for k, v in values.items()))
                        row = candidates.loc[(panel, info["dataset"], info["fold"], i)]
                        field = "guard" if prefix == "g" else "outer"
                        for k, v in values.items():
                            np.testing.assert_allclose(v, row[f"{field}_{k}"], atol=1e-12, rtol=0)
                        (guard if prefix == "g" else outer).append(values)
                        (fresh_guard if prefix == "g" else fresh_outer).append(independent)
                width = len(info["supports"][0])
                decision = []
                for i, cols in enumerate(info["supports"]):
                    admitted = i != 0 and passes(guard[i], guard[0])
                    chosen = i if admitted else 0
                    breach = not passes(outer[chosen], outer[0])
                    if passes(guard[i], guard[0]) != passes(fresh_guard[i], fresh_guard[0]) or passes(outer[i], outer[0]) != passes(fresh_outer[i], fresh_outer[0]):
                        differing_decisions.append(dict(fold=path, support=i))
                    saved = 1 - len(info["supports"][chosen])/width
                    decision.append(dict(u=saved*(not breach), breach=breach, saved=saved, accepted=admitted,
                                         **outer[chosen]))
                available = decision[1:] or decision
                for name in (*info["nominees"], "uniform"):
                    choices = available if name == "uniform" else [decision[info["nominees"][name]]]
                    row = policies.loc[(panel, info["dataset"], info["fold"], name)]
                    for k in choices[0]:
                        np.testing.assert_allclose(row[k], np.mean([c[k] for c in choices]), atol=1e-12, rtol=0)
                    np.testing.assert_allclose(row.joint_ceiling, max(c["u"] for c in available))
                    n_records += 1
            n_folds += 1
    result = dict(folds=n_folds, independent_support_refits=n_refits, policies=n_records,
                  largest_absolute_probability_difference=max_diff,
                  largest_absolute_metric_difference=max_metric_diff,
                  independent_pipeline_changed_candidate_decisions=differing_decisions,
                  candidate_metrics="rank-sum AUC; tie-aware AP; direct accuracy",
                  passing_rule="Independent joint guard and outer comparisons")
    store(DEST / "audit.json", result)
    print(result)


if __name__ == "__main__":
    audit()
