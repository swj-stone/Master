"""Independently recalculate metrics and repeat scoring on unseen predictor rows only."""
from io import BytesIO
from itertools import product
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from experiments.context_review.wide_data import EVIDENCE, RUN
from experiments.context_review.wide_score import ARMS, CONTRASTS, paired, validate_package
from experiments.context_study.data import digest


def metrics(y, p, rare):
    assert p.ndim == 2 and len(y) == len(p)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    scores = {"accuracy": accuracy_score(y, np.argmax(p, 1))}
    if p.shape[1] == 2:
        scores["auc"] = roc_auc_score(y, p[:, 1])
    else:
        scores["auc"] = np.mean([roc_auc_score(y == c, p[:, c]) for c in range(p.shape[1])])
    if rare:
        scores["ap"] = average_precision_score(y, p[:, 1])
    return scores


def audit():
    seal = validate_package()
    original = pd.read_csv(EVIDENCE / "fold-results.csv")
    sensitivity = original.copy()
    removals = []
    with ZipFile(EVIDENCE / "predictions.zip") as model, ZipFile(EVIDENCE / "outcomes.zip") as targets:
        for key in seal["folds"]:
            info = json.loads(model.read(f"{key}/selection.json"))
            layout = json.loads(model.read(f"{key}/layout.json"))
            source = RUN / key / "inputs.npz"
            assert digest(source) == layout["inputs_sha256"]
            with np.load(source) as inputs:
                # Normalize signed zeros and missing entries before exact byte comparisons.
                def keys(x):
                    x = x.copy()
                    x[x == 0] = 0
                    x[np.isnan(x)] = np.nan
                    return [row.tobytes() for row in x]
                seen = set(keys(inputs["x"]))
                novel = np.array([row not in seen for row in keys(inputs["x_outer"])])
            pred = np.load(BytesIO(model.read(f"{key}/predictions.npz")))
            label = np.load(BytesIO(targets.read(f"{key}/targets.npz")))
            np.testing.assert_array_equal(pred["test_row"], label["test_row"])
            y = label["y"]
            rare = info["choices"]["gain"]["rare"]
            unique = np.arange(len(layout["classes"]))
            assert np.array_equal(np.unique(y[novel]), unique)
            for mask in (np.ones(len(y), dtype=bool), novel):
                ref = metrics(y[mask], pred["anchor"][mask], rare)
                for method in ARMS:
                    sub = (original.dataset == layout["dataset"]) & (original.fold == layout["fold"]) & (original.arm == method)
                    row = original[sub].iloc[0]
                    actual = metrics(y[mask], pred[method][mask], rare)
                    breach = any(actual[m] + 1e-12 < r - abs(r) * (.05 if m == "ap" else .03)
                                 for m, r in ref.items())
                    u = (1 - row.k / row.anchor_k) * (not breach)
                    if mask.all():
                        assert all(np.isclose(row[m], v, rtol=0, atol=1e-12) for m, v in actual.items())
                        assert int(breach) == row.breach and np.isclose(row.u, u)
                    else:
                        for field, value in dict(breach=int(breach), u=u, **actual).items():
                            sensitivity.loc[sub, field] = value
            removals.append(dict(dataset=layout["dataset"], fold=layout["fold"], outer=len(y),
                                 removed=int((~novel).sum())))
    pd.DataFrame(removals).to_csv(EVIDENCE / "duplicate-removals.csv", index=False)
    n = original.dataset.nunique()
    draws = np.random.default_rng(33119).integers(n, size=(30000, n))
    records = [paired(sensitivity, left, right, draws)[0] for left, right in CONTRASTS]
    pd.DataFrame(records).to_csv(EVIDENCE / "duplicate-contrasts.csv", index=False)
    # With six sources all 6^6 ordered bootstrap samples fit in memory.
    exact = np.array(list(product(range(n), repeat=n)))
    by_task = original.pivot_table(index="dataset", columns="arm", values="u")
    bootstrap = []
    for left, right in CONTRASTS:
        effects = (by_task[left] - by_task[right]).to_numpy()
        lo, hi = np.percentile(effects[exact].mean(axis=1), [2.5, 97.5])
        bootstrap.append(dict(arm=left, reference=right, tasks=n, draws=len(exact),
                              u=effects.mean(), u_low=lo, u_high=hi))
    pd.DataFrame(bootstrap).to_csv(EVIDENCE / "exact-bootstrap.csv", index=False)
    print(f"Verified the independent scores on {len(original)} policy/folds; "
          f"overlap sensitivity omitted {sum(r['removed'] for r in removals)} outer records.")
    print(pd.DataFrame(records)[["arm", "reference", "u", "u_low", "u_high", "breach"]])


if __name__ == "__main__":
    audit()
