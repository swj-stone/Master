"""Rebuild archive features, refit with scikit-learn and verify the reported comparisons."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/context_review/incremental-prediction"
DONOR = ROOT / "results/context_review/refit-target"


def verify_input():
    ids = ["cohort", "dataset", "fold", "subset"]
    predicted = pd.read_csv(OUT / "predictions.csv", float_precision="round_trip").set_index(ids).sort_index()
    archived = pd.read_csv(DONOR / "decisions.csv").set_index(ids).sort_index()
    pd.testing.assert_index_equal(predicted.index, archived.index)
    np.testing.assert_array_equal(predicted.fail, archived.fail)
    np.testing.assert_allclose(predicted.x, (archived.anchor_k - archived.k) / archived.anchor_k)
    for (dataset, fold), part in predicted.reset_index().groupby(["dataset", "fold"]):
        source = json.loads((DONOR / dataset / f"fold-{fold}/selection.json").read_bytes())
        for row in part.itertuples():
            ballots = source["ballots"][row.subset]["separating"]
            assert row.v == max(sum(not yes for yes in context) / 4 for context in ballots)
    return predicted.reset_index()


def verify_fit(source, settings, trained):
    errors = []
    for target, test in source.groupby("dataset"):
        train = source[source.dataset != target]
        w = pd.Series(0.0, index=train.index)
        for _, task in train.groupby("dataset"):
            for _, fold in task.groupby("fold"):
                w.loc[fold.index] = 1 / (13 * task.fold.nunique() * len(fold))
        assert np.isclose(w.sum(), 1)
        for model in ("size", "size_vote"):
            archive = [r for r in trained if (r["model"], r["dataset"]) == (model, target)]
            assert len(archive) == 1 and set(archive[0]["training_tasks"]) == set(train.dataset)
            def matrix(part):
                cols = [part.x] + [(part.x - knot).clip(lower=0) for knot in settings["knots"]]
                if model == "size_vote":
                    cols.append(part.v)
                return np.column_stack(cols)
            x = matrix(train)
            mean = np.average(x, weights=w, axis=0)
            sd = np.sqrt(np.average((x - mean) ** 2, weights=w, axis=0))
            sd[sd < 1e-12] = 1
            estimator = LogisticRegression(C=1 / settings["lambda"], solver="newton-cholesky", tol=1e-12)
            estimator.fit((x - mean) / sd, train.fail, sample_weight=w)
            probs = estimator.predict_proba((matrix(test) - mean) / sd)[:, 1]
            errors.extend(abs(probs - test[f"p_{model}"]))
    assert max(errors) < 1e-6
    return float(max(errors))


def verify_scores(source, summary, settings):
    aggregates = []
    for (dataset, fold), part in source.groupby(["dataset", "fold"]):
        for model in ("size", "size_vote"):
            p = part[f"p_{model}"]
            metric = {"log_loss": log_loss(part.fail, p, labels=[0, 1]),
                      "brier": brier_score_loss(part.fail, p)}
            if part.fail.nunique() == 2:
                metric["auc"] = roc_auc_score(part.fail, p)
            aucs, counts = [], []
            for _, same in part.groupby("k"):
                npos = int(same.fail.sum())
                nneg = len(same) - npos
                if npos and nneg:
                    aucs.append(roc_auc_score(same.fail, same[f"p_{model}"]))
                    counts.append(npos * nneg)
            if counts:
                metric["same_width_auc"] = np.average(aucs, weights=counts)
            for name, value in metric.items():
                aggregates.append((dataset, fold, model, name, value))
    check = pd.DataFrame(aggregates, columns=["dataset", "fold", "model", "metric", "value"])
    tasks = check.groupby(["metric", "dataset", "model"]).value.mean().unstack()
    for name, group in tasks.groupby(level="metric"):
        row = summary.loc[name]
        effect = group["size_vote"] - group["size"]
        generator = np.random.default_rng(settings["bootstrap_seed"])
        idx = generator.integers(len(effect), size=(settings["bootstrap_replicates"], len(effect)))
        bounds = np.quantile(effect.to_numpy()[idx].mean(axis=1), [0.025, 0.975])
        np.testing.assert_allclose(list(group.mean()) + [effect.mean(), *bounds],
                                   [row["size"], row["size_vote"], row.difference, row.lower, row.upper],
                                   rtol=1e-10, atol=1e-12)


def main():
    frame = verify_input()
    summary = pd.read_csv(OUT / "summary.csv").set_index(["cohort", "metric"])
    models = json.loads((OUT / "models.json").read_bytes())
    design = json.loads((ROOT / "configs/context_review/incremental_prediction.json").read_bytes())
    errors = {}
    for cohort, source in frame.groupby("cohort"):
        errors[cohort] = verify_fit(source, design, models[cohort])
        verify_scores(source, summary.loc[cohort], design)
    result = dict(source_records=len(frame), unique_fold_supports=len(frame.drop_duplicates(["dataset", "fold", "subset"])),
                  independent_held_out_fits=sum(map(len, models.values())), maximum_prediction_errors=errors,
                  comparisons_with_verified_paired_intervals=len(summary))
    print(json.dumps(result, indent=2))
    (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
