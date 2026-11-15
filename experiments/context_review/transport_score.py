"""Open outer targets only after the sealed 30-fold prediction manifest verifies."""
import argparse
import json

import numpy as np
import pandas as pd

from experiments.context_review.budgeted_guard import metrics
from experiments.context_review.transport_data import EVIDENCE, ROOT, RUN, SPEC, read
from experiments.context_study.benchmark import store
from experiments.context_study.data import digest
from minppi_mofs.piv3_compact_witness import _within

ARMS = ("gain", "vote", "slack", "point")
COMPARISONS = (("vote", "gain"), ("slack", "gain"), ("slack", "vote"),
               ("slack", "point"), ("point", "gain"))


def verify_lock():
    seal = read(EVIDENCE / "prediction-lock.json")
    assert seal["protocol"] == digest(RUN / "protocol.json")
    assert all(digest(ROOT / key) == value for key, value in seal["sources"].items())
    expected = {f"{name}/fold-{fold}" for name, *_ in read(SPEC)["sources"] for fold in range(3)}
    assert set(seal["selections"]) == expected
    for key, value in seal["selections"].items():
        root = RUN / key
        assert digest(root / "selection.json") == value
        info = read(root / "selection.json")
        layout = root / "layout.json"
        assert info["layout"] == digest(layout) and info["protocol"] == seal["protocol"]
        assert info["predictions"] == digest(root / "predictions.npz")
        assert read(layout)["targets_sha256"] == digest(root / "targets.npz")
    return sorted(expected)


def fold_rows(root):
    info, layout = read(root / "selection.json"), read(root / "layout.json")
    name, fold = layout["dataset"], layout["fold"]
    rows, candidates = [], []
    with np.load(root / "predictions.npz") as pred, np.load(root / "targets.npz") as target:
        np.testing.assert_array_equal(pred["test_row"], target["test_row"])
        yy = target["y"]
        rare = info["choices"]["gain"]["rare"]
        reference = metrics(yy, pred["anchor"], rare)
        for arm in ARMS:
            choice = info["choices"][arm]
            anchor, nominee, kept = (choice[k] for k in ("anchor", "nominee", "selected"))
            assert len(kept) == len(anchor) or len(kept) == (len(anchor)+1)//2
            assert (kept == nominee if choice["passes"] else kept == anchor)
            for other in ARMS:
                if info["choices"][other]["selected"] == kept:
                    np.testing.assert_array_equal(pred[arm], pred[other])
            if kept == anchor:
                np.testing.assert_array_equal(pred[arm], pred["anchor"])
            actual = metrics(yy, pred[arm], rare)
            breach = not _within(actual, reference)
            costs = info["costs"][arm]
            total_fits = sum(costs[key] for key in
                             ("initial_fits", "guard_anchor_fits", "guard_candidate_fits", "final_fits"))
            rows.append(dict(dataset=name, fold=fold, arm=arm, width=layout["p"], n_outer=len(yy),
                        anchor=json.dumps(anchor), nominee=json.dumps(nominee), selected=json.dumps(kept),
                        anchor_k=len(anchor), k=len(kept), eligible=int(len(anchor)>1),
                        nominees=len(info["pool"]), passed=int(choice["passes"]), breach=int(breach),
                        saved=1-len(kept)/len(anchor), u=(1-len(kept)/len(anchor))*(not breach),
                        retained=len(kept)/layout["p"], auc=actual["auc"], accuracy=actual["accuracy"],
                        anchor_auc=reference["auc"], anchor_accuracy=reference["accuracy"],
                        total_fits=total_fits, **costs))
    for i, scores in enumerate(info["pool"]):
        candidates.append(dict(dataset=name, fold=fold, tie_order=i, **scores))
    split = dict(dataset=name, fold=fold, uci_id=layout["id"], n=layout["n"], width=layout["p"],
                 classes=len(layout["classes"]), source=layout["source"], inputs=layout["inputs_sha256"],
                 layout=info["layout"], targets=layout["targets_sha256"], predictions=info["predictions"],
                 selection=digest(root / "selection.json"), n_train=len(layout["train"]),
                 n_proposal=len(layout["proposal"]), n_guard=len(layout["guard"]),
                 n_test=len(layout["test"]), missing_cells=layout["missing"],
                 duplicate_feature_rows=layout["duplicated_inputs"], overlap_test_rows=layout["overlap_test_rows"],
                 order=json.dumps(info["choices"]["gain"]["order"]), schedule=json.dumps(info["schedule"]))
    return rows, candidates, split


def contrast(df, comp, draws):
    arm, reference = comp
    a = df[df.arm==arm].set_index(["dataset", "fold"]).sort_index()
    b = df[df.arm==reference].set_index(["dataset", "fold"]).sort_index()
    assert a.index.equals(b.index)
    deltas = (a[["u","breach","auc","accuracy","saved","wall_s","masks","passed"]]
              - b[["u","breach","auc","accuracy","saved","wall_s","masks","passed"]])
    tasks = deltas.groupby(level="dataset").mean()
    values = tasks.u.to_numpy()
    lo, hi = np.percentile(values[draws].mean(axis=1), [2.5,97.5])
    omitted = [(tasks.u.sum()-v)/(len(tasks)-1) for v in values]
    details = tasks.reset_index()
    details.insert(1, "contrast", f"{arm}-{reference}")
    summary = dict(arm=arm, reference=reference, tasks=len(values),
                   u_low=lo, u_high=hi, loo_min=min(omitted), loo_max=max(omitted),
                   omit_rice_and_bean=tasks.drop(["rice","dry_bean"]).u.mean(),
                   changed_nominee=int((a.nominee!=b.nominee).sum()), changed_final=int((a.selected!=b.selected).sum()),
                   task_wins=int((values>0).sum()), task_ties=int((values==0).sum()),
                   task_losses=int((values<0).sum()), fold_wins=int((deltas.u>0).sum()),
                   fold_ties=int((deltas.u==0).sum()), fold_losses=int((deltas.u<0).sum()),
                   **tasks.mean().to_dict())
    return summary, details


def score():
    jobs = verify_lock()
    rows, pool, splits = [], [], []
    for key in jobs:
        records, candidates, split = fold_rows(RUN / key)
        rows.extend(records)
        pool.extend(candidates)
        splits.append(split)
    df = pd.DataFrame(rows)
    assert len(df) == 120 and not df.duplicated(["dataset", "fold", "arm"]).any()
    assert (df.groupby("dataset").size() == 12).all()
    df.to_csv(EVIDENCE / "fold-results.csv", index=False)
    pd.DataFrame(splits).to_csv(EVIDENCE / "fold-manifest.csv", index=False)
    candidate_df = pd.DataFrame(pool)
    candidate_df.to_csv(EVIDENCE / "candidate-scores.csv", index=False)
    counts = []
    for (name, fold), row in candidate_df.groupby(["dataset", "fold"]):
        def top_ties(c):
            return int((row[c] == row[c].max()).sum())
        counts.append(dict(dataset=name, fold=fold, candidates=len(row),
                           all_vote_zero=int((row.vote == 0).all()),
                           vote_levels=row.vote.nunique(), slack_levels=row.slack.nunique(),
                           point_levels=row.point.nunique(), vote_top_ties=top_ties("vote"),
                           slack_top_ties=top_ties("slack"), point_top_ties=top_ties("point")))
    resolution = pd.DataFrame(counts)
    resolution.to_csv(EVIDENCE / "resolution.csv", index=False)
    fields = ["u","breach","auc","accuracy","anchor_auc","anchor_accuracy","saved","retained","k",
              "anchor_k","eligible","passed","total_fits","guard_candidate_fits","masks","wall_s","nomination_s"]
    tasks = df.groupby(["dataset","arm"])[fields].mean()
    tasks.to_csv(EVIDENCE / "task-results.csv")
    tasks.groupby("arm").mean().to_csv(EVIDENCE / "summary.csv")
    draws = np.random.default_rng(33119).integers(10, size=(30000,10))
    summaries, paired_tasks = zip(*(contrast(df, comp, draws) for comp in COMPARISONS))
    pd.DataFrame(summaries).to_csv(EVIDENCE / "contrasts.csv", index=False)
    pd.concat(paired_tasks, ignore_index=True).to_csv(EVIDENCE / "task-contrasts.csv", index=False)
    print(tasks.groupby("arm").mean().to_string(float_format=lambda x:f"{x:.5f}"))
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x:f"{x:.5f}"))
    print("Eligible folds", len(resolution), "candidates", len(pool),
          "unanimously zero-scored pools", resolution.all_vote_zero.sum(),
          "folds with tied best", (resolution[["vote_top_ties","slack_top_ties","point_top_ties"]]>1).sum().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    score()
