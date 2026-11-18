"""Four development folds for a fixed, independently calibrated minPI budget."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import ROOT, panels, file_hash
from experiments.minpi_constant_transfer import fit_predict
from minppi_mofs.cheap_calibrated_document_minpi import cheap_calibrated_document_top_from_training
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training


def probe(output):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "src/minppi_mofs/cheap_calibrated_document_minpi.py"
    protocol = dict(
        folds=["sms:1", "amazon_cells:1", "imdb:1", "yelp:1"], seed=1129,
        rule="Shortest prefix within sqrt(p)..ceil(p/2) whose class-balanced calibration squared loss "
             "does not exceed the full frozen teacher; numerical tolerance 1e-12 only. "
             "No feasible prefix returns all columns and is a failure. "
             "One fitted scalar on diagonal unit-ridge direction, original minPI votes, "
             "voting-row-only marginal tie break, all remaining training rows calibrate. "
             "Empirical loss minimizer is diagnostic only. Outer scores do not choose the rule.",
        source_sha256=file_hash(source), probe_sha256=file_hash(Path(__file__)),
        development_only=True, formal_timing=False,
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf8")
    records, predictions = [], []
    for X, y, Xt, truth, meta in panels():
        if meta["fold"] != 1 or meta["dataset"] not in ("sms", "amazon_cells", "imdb", "yelp"):
            continue
        old, old_info = fast_document_vote_top_from_training(X, y, random_state=meta["seed"])
        selected, info = cheap_calibrated_document_top_from_training(X, y, random_state=meta["seed"])
        losses, rank = info["prefix_losses"], info["rank"]
        positive = np.maximum(info["scores"][rank], 0.)
        mass = positive.cumsum()
        mass = mass / mass[-1] if mass[-1] > 0 else mass
        pd.DataFrame(dict(k=np.arange(X.shape[1] + 1), calibration_loss=losses,
                          positive_vote_fraction=np.r_[0., mass])).to_csv(
            output / (meta["dataset"] + "-curve.csv"), index=False)
        cal = info["calibration_rows"]
        classes, labels = np.unique(y, return_inverse=True)
        weights = 1. / (len(classes) * np.bincount(labels[cal], minlength=len(classes))[labels[cal]])

        def teacher(matrix, cols):
            return info["prior"] + matrix[:, cols] @ info["coefficient"][cols] - (
                info["feature_mean"][cols] @ info["coefficient"][cols])

        row = dict(dataset=meta["dataset"], p=X.shape[1], fit_n=len(info["fit_rows"]), calibration_n=len(cal),
                   old_width=len(old), selected_width=len(selected), feasible=info["feasible"], eta=info["eta"],
                   full_teacher_loss=info["full_teacher_loss"], selected_teacher_loss=info["selected_teacher_loss"],
                   empirical_best_count=info["empirical_best_count"], empirical_best_loss=losses[info["empirical_best_count"]],
                   loss_at_old_width=losses[len(old)], selected_positive_mass=float(mass[len(selected)-1]))
        for name, cols in (("full", np.arange(X.shape[1])), ("old", old), ("calibrated", selected)):
            pred = fit_predict(X, y, Xt, cols, "classification", meta["seed"], False)
            outer_teacher = classes[np.argmax(teacher(Xt, cols), axis=1)]
            cal_prediction = teacher(X[cal], cols)
            row[name + "_score"] = balanced_accuracy_score(truth, pred)
            row[name + "_teacher_outer_score"] = balanced_accuracy_score(truth, outer_teacher)
            row[name + "_teacher_calibration_score"] = balanced_accuracy_score(
                y[cal], classes[np.argmax(cal_prediction, axis=1)])
            row[name + "_teacher_calibration_loss"] = float(np.sum(
                weights[:, None] * np.square(np.eye(len(classes))[labels[cal]] - cal_prediction)))
            predictions.append(dict(dataset=meta["dataset"], method=name, width=len(cols),
                                    score=row[name + "_score"]))
        records.append(row)
        print(json.dumps(row), flush=True)
    assert len(records) == 4
    pd.DataFrame(records).to_csv(output / "diagnostics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "scores.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        probe(args.output)
