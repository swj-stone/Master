"""Held-out-task prediction of observed excess loss on frozen deletion proposals."""
import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from experiments.context_study.data import ROOT, digest


DESIGN = ROOT / "configs/context_review/incremental_prediction.json"
DONOR = ROOT / "results/context_review/refit-target"
OUT = ROOT / "results/context_review/incremental-prediction"
MODELS = ("size", "size_vote")
METRICS = ("log_loss", "brier", "auc", "same_width_auc")
ID = ["dataset", "fold", "subset"]


def read(path):
    return json.loads(path.read_bytes())


def freeze(folders):
    inputs = [DESIGN, Path(__file__).resolve(), DONOR / "protocol.json"]
    for folder in folders:
        inputs.extend([folder / "selection.json", folder / "completed.json"])
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "protocol.json"
    if lock.exists():
        if read(lock)["inputs"] != hashes:
            raise ValueError("Frozen design, code or archive changed")
    else:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"frozen_utc": datetime.now(timezone.utc).isoformat(), "inputs": hashes},
                      stream, indent=2)


def records(folders, cohorts):
    rows = []
    for folder in folders:
        complete = read(folder / "completed.json")
        assert complete["selection"] == digest(folder / "selection.json")
        assert complete["protocol"] == digest(DONOR / "protocol.json")
        selection = read(folder / "selection.json")
        assert set(selection["cohorts"]) == set(cohorts)
        anchor = set(selection["anchor"])
        reference = complete["reference"]
        for source in cohorts:
            expected = {tuple(s) for s in selection["cohorts"][source]}
            actual = set()
            for r in complete["rows"]:
                if r["cohort"] != source:
                    continue
                assert r["dataset"] == folder.parent.name and folder.name == f"fold-{r['fold']}"
                ix = r["subset"]
                support = tuple(selection["supports"][ix])
                assert set(support) < anchor and 0 < r["k"] == len(support)
                assert r["anchor_k"] == len(anchor)
                assert support not in actual
                actual.add(support)
                yes = np.asarray(selection["ballots"][ix]["separating"], dtype=bool)
                assert yes.ndim == 2 and yes.shape[1] == 4
                maximum_dissent = 4 - yes.sum(axis=1).min()
                assert (maximum_dissent >= 2) == r["separating"]
                fail = any(r[f"loss_{m}"] > (0.05 if m == "ap" else 0.03) * abs(value) + 1e-12
                           for m, value in reference.items())
                assert fail == r["fail"]
                rows.append(dict(dataset=r["dataset"], fold=r["fold"], subset=ix, cohort=source,
                                 k=len(support), anchor_k=len(anchor), fail=int(fail),
                                 x=(len(anchor) - len(support)) / len(anchor), v=maximum_dissent / 4))
            assert expected == actual
    table = pd.DataFrame(rows)
    assert not table.duplicated(ID + ["cohort"]).any()
    shared = table.loc[table.duplicated(ID, keep=False)]
    assert shared.groupby(ID)[["fail", "x", "v"]].nunique().eq(1).all().all()
    return table


def features(frame, settings):
    x = frame.x.to_numpy()
    return np.column_stack([x, *(np.maximum(x - knot, 0) for knot in settings["knots"]), frame.v])


def weights(train):
    by_fold = train.groupby(["dataset", "fold"]).x.transform("size")
    folds_per_task = train.groupby("dataset").fold.transform("nunique")
    return (1 / (by_fold * folds_per_task * train.dataset.nunique())).to_numpy()


def fit_logistic(train_x, train_y, weight, penalty):
    def objective(beta):
        z = train_x @ beta[1:] + beta[0]
        residual = weight * (expit(z) - train_y)
        loss = weight @ (np.logaddexp(0, z) - train_y * z) + 0.5 * penalty * (beta[1:] @ beta[1:])
        gradient = np.r_[residual.sum(), train_x.T @ residual + penalty * beta[1:]]
        return loss, gradient
    result = minimize(objective, np.zeros(train_x.shape[1] + 1), jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000})
    # L-BFGS can report a failed line search at floating-point convergence.
    # The gradient directly checks stationarity of this convex objective.
    if not np.isfinite(result.fun) or not np.all(np.abs(result.jac) <= 1e-7):
        raise RuntimeError(f"{result.message}, max gradient {np.max(np.abs(result.jac))}")
    return result.x


def predict(frame, held_out, training_tasks, settings):
    test = frame.loc[frame.dataset.eq(held_out)].copy()
    train = frame.loc[frame.dataset.isin(training_tasks)]
    assert held_out not in training_tasks and train.dataset.nunique() == len(training_tasks)
    assert train.fail.nunique() == 2
    weight = weights(train)
    np.testing.assert_allclose(weight.sum(), 1)
    scale = StandardScaler().fit(features(train, settings), sample_weight=weight)
    train_x = scale.transform(features(train, settings))
    test_x = scale.transform(features(test, settings))
    coefficients = []
    for name, width in zip(MODELS, (train_x.shape[1] - 1, train_x.shape[1]), strict=True):
        beta = fit_logistic(train_x[:, :width], train.fail.to_numpy(), weight, settings["lambda"])
        test[f"logit_{name}"] = test_x[:, :width] @ beta[1:] + beta[0]
        test[f"p_{name}"] = expit(test[f"logit_{name}"])
        coefficients.append(dict(dataset=held_out, model=name, training_tasks=sorted(training_tasks),
                                 intercept=float(beta[0]), coef=beta[1:].tolist(),
                                 means=scale.mean_[:width].tolist(), scales=scale.scale_[:width].tolist()))
    return test, coefficients


def concordance(logit, comparable):
    pairs = int(comparable.sum())
    if not pairs:
        return np.nan
    delta = (logit[:, None] - logit[None, :])[comparable]
    return ((delta > 0).sum() + 0.5 * (delta == 0).sum()) / pairs


def scores(predictions):
    rows = []
    for (dataset, fold), part in predictions.groupby(["dataset", "fold"]):
        y, width = part.fail.to_numpy(), part.k.to_numpy()
        opposite = (y[:, None] == 1) & (y[None, :] == 0)
        matched = opposite & (width[:, None] == width[None, :])
        row = dict(dataset=dataset, fold=fold, proposals=len(part), failures=y.sum(),
                   pairs=opposite.sum(), same_width_pairs=matched.sum())
        for name in MODELS:
            logit, p = part[f"logit_{name}"].to_numpy(), part[f"p_{name}"].to_numpy()
            row[f"log_loss_{name}"] = (np.logaddexp(0, logit) - y * logit).mean()
            row[f"brier_{name}"] = np.square(p - y).mean()
            row[f"auc_{name}"] = concordance(logit, opposite)
            row[f"same_width_auc_{name}"] = concordance(logit, matched)
        assert not matched.any() or row["same_width_auc_size"] == 0.5
        rows.append(row)
    return pd.DataFrame(rows)


def task_scores(fold_scores):
    rows = []
    for dataset, folds in fold_scores.groupby("dataset"):
        for metric in METRICS:
            cols = [f"{metric}_{name}" for name in MODELS]
            active = folds.dropna(subset=cols)
            if active.empty:
                continue
            size, augmented = active[cols].mean()
            pairs = active.same_width_pairs.sum() if metric == "same_width_auc" else active.pairs.sum()
            rows.append(dict(dataset=dataset, metric=metric, folds=len(active), proposals=active.proposals.sum(),
                             pairs=pairs, size=size, size_vote=augmented, difference=augmented - size))
    return pd.DataFrame(rows)


def analyze(frame, settings):
    names = set(frame.dataset)
    assert len(names) == 14
    predictions, models = [], []
    for name in sorted(names):
        out, coefficients = predict(frame, name, names - {name}, settings)
        predictions.append(out)
        models.extend(coefficients)
    predictions = pd.concat(predictions, ignore_index=True)
    folds = scores(predictions)
    tasks = task_scores(folds)
    # A fitted model trained without a pair of tasks can predict either task.
    # Each ordered evaluation below holds out its target and omits the other task entirely.
    omitted_parts = {name: [] for name in names}
    for first, second in combinations(sorted(names), 2):
        train = names - {first, second}
        for hold, omit in ((first, second), (second, first)):
            out, _ = predict(frame, hold, train, settings)
            omitted_parts[omit].append(scores(out))
    sensitivities = []
    for omitted in sorted(names):
        reduced = task_scores(pd.concat(omitted_parts[omitted]))
        for metric, group in reduced.groupby("metric"):
            sensitivities.append(dict(omitted=omitted, metric=metric, eligible_tasks=len(group),
                                      difference=group.difference.mean()))
    sensitivity = pd.DataFrame(sensitivities)
    rows = []
    for metric in METRICS:
        group = tasks[tasks.metric.eq(metric)].sort_values("dataset")
        delta = group.difference.to_numpy()
        rng = np.random.default_rng(settings["bootstrap_seed"])
        boot = rng.choice(delta, size=(settings["bootstrap_replicates"], len(delta))).mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        reduced = sensitivity.loc[sensitivity.metric.eq(metric), "difference"]
        rows.append(dict(metric=metric, eligible_tasks=len(delta), eligible_folds=int(group.folds.sum()),
                         proposals=int(group.proposals.sum()), pairs=int(group.pairs.sum()),
                         size=group["size"].mean(), size_vote=group.size_vote.mean(), difference=delta.mean(),
                         lower=lo, upper=hi, refit_omit_one_min=reduced.min(), refit_omit_one_max=reduced.max()))
    return (dict(predictions=predictions, folds=folds, tasks=tasks, sensitivity=sensitivity,
                 summary=pd.DataFrame(rows)), models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    settings = read(DESIGN)
    folders = sorted(p.parent for p in DONOR.glob("*/fold-*/completed.json"))
    assert len(folders) == 42
    freeze(folders)
    if args.freeze_only:
        return
    frame = records(folders, settings["cohorts"])
    outputs = {}
    models = {}
    for cohort in settings["cohorts"]:
        data = frame.loc[frame.cohort.eq(cohort)]
        tables, models[cohort] = analyze(data, settings)
        for key, values in tables.items():
            outputs.setdefault(key, []).append(values.assign(cohort=cohort))
    for key, chunks in outputs.items():
        pd.concat(chunks, ignore_index=True).to_csv(OUT / f"{key}.csv", index=False)
    (OUT / "models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
    print(pd.concat(outputs["summary"]).to_string(index=False))
    print("Unique fold-support pairs:", len(frame.drop_duplicates(ID)))
    print("Pairs shared between sources:", int(frame.duplicated(ID).sum()))


if __name__ == "__main__":
    main()
