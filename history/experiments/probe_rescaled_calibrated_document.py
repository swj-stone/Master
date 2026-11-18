"""Fixed four-fold development probe for analytic-scale Brier and BA budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import ROOT, panels, file_hash
from experiments.minpi_constant_transfer import fit_predict
from minppi_mofs.rescaled_calibrated_document_minpi import rescaled_calibrated_document_top_from_training
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training


def probe(output):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "src/minppi_mofs/rescaled_calibrated_document_minpi.py"
    protocol = dict(
        folds=["sms:1", "amazon_cells:1", "imdb:1", "yelp:1"], seed=1129,
        rule_a2="Shortest sqrt(p)..ceil(p/2) prefix with balanced calibration squared loss <= full teacher. "
                "Each prefix uses its exact training-row scalar ridge optimum on the fixed direction.",
        rule_b2="Shortest among 16 logarithmically spaced and rounded unique widths in sqrt(p)..ceil(p/2) "
                "with calibration BA >= full teacher BA - .02. Full p is a separate reference.",
        failure="No feasible candidate returns all p and is a failure; no outer scores choose a budget.",
        source_sha256=file_hash(source), probe_sha256=file_hash(Path(__file__)),
        development_only=True, formal_timing=False,
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf8")
    records = []
    for X, y, Xt, truth, meta in panels():
        if meta["fold"] != 1 or meta["dataset"] not in ("sms", "amazon_cells", "imdb", "yelp"):
            continue
        old, _ = fast_document_vote_top_from_training(X, y, random_state=meta["seed"])
        selected, info = rescaled_calibrated_document_top_from_training(X, y, random_state=meta["seed"])
        rank, losses = info["rank"], info["prefix_losses"]
        pd.DataFrame(dict(k=np.arange(X.shape[1] + 1), calibration_loss=losses,
                          eta=info["eta_path"])).to_csv(output / (meta["dataset"] + "-curve.csv"), index=False)
        pd.DataFrame(dict(k=info["ba_grid"], calibration_ba=info["ba_grid_scores"])).to_csv(
            output / (meta["dataset"] + "-ba-grid.csv"), index=False)
        cal = info["calibration_rows"]
        classes, labels = np.unique(y, return_inverse=True)
        weights = 1. / (len(classes) * np.bincount(labels[cal], minlength=len(classes))[labels[cal]])
        row = dict(dataset=meta["dataset"], p=X.shape[1], fit_n=len(info["fit_rows"]), calibration_n=len(cal),
                   old_width=len(old), a2_width=len(selected), b2_width=info["ba_budget"],
                   a2_feasible=info["brier_feasible"], b2_feasible=info["ba_feasible"],
                   full_teacher_loss=info["full_teacher_loss"], full_teacher_ba=info["full_teacher_ba"],
                   empirical_best_count=info["empirical_best_count"], empirical_best_loss=losses[info["empirical_best_count"]])
        for name, cols in (("full", np.arange(X.shape[1])), ("old", old),
                           ("a2", selected), ("b2", np.sort(rank[:info["ba_budget"]]))):
            pred = fit_predict(X, y, Xt, cols, "classification", meta["seed"], False)
            row[name + "_score"] = balanced_accuracy_score(truth, pred)
            if name == "old":
                continue
            k = len(cols)
            beta = info["eta_path"][k] * info["direction"][cols]
            intercept = info["prior"] - info["feature_mean"][cols] @ beta
            cal_pred = intercept + X[cal][:, cols] @ beta
            outer_pred = intercept + Xt[:, cols] @ beta
            row[name + "_eta"] = info["eta_path"][k]
            row[name + "_teacher_outer_score"] = balanced_accuracy_score(truth, classes[np.argmax(outer_pred, axis=1)])
            row[name + "_teacher_calibration_ba"] = balanced_accuracy_score(y[cal], classes[np.argmax(cal_pred, axis=1)])
            row[name + "_teacher_calibration_loss"] = float(np.sum(
                weights[:, None] * np.square(np.eye(len(classes))[labels[cal]] - cal_pred)))
            np.testing.assert_allclose(row[name + "_teacher_calibration_loss"], losses[k], atol=2e-12)
        records.append(row)
        print(json.dumps(row), flush=True)
    assert len(records) == 4
    pd.DataFrame(records).to_csv(output / "diagnostics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        probe(args.output)
