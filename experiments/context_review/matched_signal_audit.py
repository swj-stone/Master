"""Independently check fold separation, risk fits, coverage and headline estimates."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.context_review.matched_signal import DESIGN, EVIDENCE, KEY, OUT, SOURCES
from experiments.context_study.benchmark import store
from experiments.context_study.data import ROOT, digest


def read_csv(p):
    return pd.read_csv(p, float_precision="round_trip")


def count_masks(frame):
    rows = []
    for (name, fold), group in frame.groupby(["dataset", "fold"]):
        path = ROOT / "results/context_review/refit-target" / name / f"fold-{fold}" / "selection.json"
        design = json.loads(path.read_bytes())
        contexts = [frozenset(s) for s in design["contexts"]["separating"]]
        for source in ["union", *SOURCES]:
            proposals = group.loc[group[source], "subset"]
            supports = [frozenset(design["anchor"]), *(frozenset(design["supports"][i]) for i in proposals)]
            masks = [{s | e for s in supports for e in subset}
                     for subset in ([contexts[0]], [contexts[0], contexts[-1]], contexts)]
            assert masks[0] <= masks[1] <= masks[2]
            rows.append(dict(dataset=name, fold=fold, source=source, proposals=len(proposals),
                point_masks=len(masks[0]), endpoint_masks=len(masks[1]), separating_masks=len(masks[2]),
                additional_masks=len(masks[2]-masks[1])))
    pd.DataFrame(rows).to_csv(EVIDENCE / "mask-counts.csv", index=False)


def main():
    read = lambda p: json.loads(p.read_bytes())
    manifest = read(EVIDENCE / "manifest.json")
    assert all(digest(OUT / name) == sha for name, sha in manifest["outputs"].items())
    assert all(digest(ROOT / name) == sha for name, sha in read(OUT / "inputs.json").items())
    frame, pred, summary = (read_csv(OUT / f"{key}.csv") for key in ("proposals", "predictions", "summary"))
    assert len(frame) == len(frame.drop_duplicates(KEY)) == 704
    raw = read_csv(ROOT / "results/context_review/refit-target/decisions.csv")
    for source in SOURCES:
        expected = raw[raw.cohort.eq(source)].set_index(KEY).sort_index()
        source_rows = frame[frame[source]].set_index(KEY).sort_index()
        pd.testing.assert_index_equal(source_rows.index, expected.index)
        np.testing.assert_array_equal(source_rows.fail, expected.fail)
        for _, row in frame.iterrows():
            same = frame.loc[frame[source] & frame.dataset.eq(row.dataset) & frame.fold.eq(row.fold)
                             & frame.k.eq(row.k), "gain_mass"]
            assert (same.min()-1e-12 <= row.gain_mass <= same.max()+1e-12) == row[f"in_{source}"]
    spec = read(DESIGN)
    fitted = read(OUT / "models.json")
    largest_error = 0
    for model in fitted:
        held, source, name, cols = (model[k] for k in ("held", "train_source", "model", "columns"))
        train = frame[frame[source] & frame.dataset.ne(held)]
        test = pred[pred.dataset.eq(held) & pred.train_source.eq(source)]
        assert set(model["training_tasks"]) == set(train.dataset) and len(model["training_tasks"]) == 13
        w = 1 / (13 * train.groupby("dataset").fold.transform("nunique")
                 * train.groupby(["dataset", "fold"]).fail.transform("size"))
        mean = np.average(train[cols], weights=w, axis=0)
        sd = np.sqrt(np.average((train[cols] - mean)**2, axis=0, weights=w))
        sd[sd < 1e-12] = 1
        np.testing.assert_allclose(model["means"], mean, atol=1e-12)
        np.testing.assert_allclose(model["scales"], sd, atol=1e-12)
        estimator = LogisticRegression(C=1 / spec["lambda"], solver="newton-cholesky", tol=1e-12)
        estimator.fit((train[cols] - mean)/sd, train.fail, sample_weight=w)
        p = estimator.predict_proba((test[cols] - mean)/sd)[:, 1]
        largest_error = max(largest_error, np.max(np.abs(p - expit(test[name]))))
    assert largest_error < 1e-6
    checked = 0
    for row in summary.loc[summary.augmented.eq("endpoint_vote")].itertuples():
        subset = pred[pred.train_source.eq(row.train_source) & pred[row.test_source]]
        if row.sample != "all":
            subset = subset[subset[f"in_{row.train_source}"]]
        if row.sample == "range_exclusive":
            subset = subset[~subset[row.train_source]]
        values = []
        for (dataset, _), fold in subset.groupby(["dataset", "fold"]):
            pair = {}
            for model in ("endpoint", "endpoint_vote"):
                z = fold[model].to_numpy()
                if row.metric == "log_loss":
                    value = np.where(fold.fail.eq(1), np.logaddexp(0, -z), np.logaddexp(0, z)).mean()
                elif row.metric == "brier":
                    value = (expit(z) - fold.fail).pow(2).mean()
                elif row.metric == "auc":
                    value = roc_auc_score(fold.fail, z) if fold.fail.nunique() == 2 else np.nan
                else:
                    strata = [(roc_auc_score(s.fail, s[model]), s.fail.sum() * (len(s) - s.fail.sum()))
                              for _, s in fold.groupby("k") if s.fail.nunique() == 2]
                    value = np.nan
                    if strata:
                        aucs, counts = zip(*strata)
                        value = np.average(aucs, weights=counts)
                pair[model] = value
            values.append((dataset, pair["endpoint_vote"] - pair["endpoint"]))
        task = pd.DataFrame(values, columns=["dataset", "delta"]).dropna().groupby("dataset").delta.mean()
        rng = np.random.default_rng(spec["bootstrap_seed"])
        boot = rng.choice(task, size=(spec["bootstrap_replicates"], len(task))).mean(axis=1)
        np.testing.assert_allclose([task.mean(), *np.quantile(boot, [.025, .975])],
                                   [row.difference, row.lower, row.upper], atol=1e-12, rtol=0)
        assert len(task) == row.tasks
        checked += 1
    report = dict(unique_proposals=len(frame), duplicate_memberships=int((frame.path & frame.uniform_deletions).sum()),
                  verified_model_fits=len(fitted), max_independent_probability_difference=float(largest_error),
                  checked_primary_contrast_metric_rows=checked, inputs=len(read(OUT/"inputs.json")))
    count_masks(frame)
    store(EVIDENCE / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
