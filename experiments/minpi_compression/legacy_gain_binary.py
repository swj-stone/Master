"""Describe threshold and precision-recall performance on every original binary dataset."""
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score

from experiments.minpi_compression import legacy_gain as legacy
from experiments.minpi_compression.legacy_gain_receiver import as_score_matrix


def main():
    rows = []
    with ZipFile(legacy.DEST/"forecasts.zip") as ridge, ZipFile(
        legacy.DEST/"xgb-receiver/forecasts.zip") as xgb, ZipFile(
        legacy.bench.ROOT/"packages/thesis-piv1-evidence.zip") as historical:
        for key in json.loads(ridge.read("lock.json")):
            info = json.loads(ridge.read(key+"/selection.json"))
            if info["task"]!="classification" or info["dataset"]=="isolet":
                continue
            y = np.load(BytesIO(ridge.read(key+"/targets.npy")))
            def add(name,pred):
                scores = as_score_matrix(pred,"classification")
                hard = scores.argmax(axis=1)
                rows.append(dict(dataset=info["dataset"],seed=info["seed"],method=name,prevalence=y.mean(),
                    average_precision=average_precision_score(y,scores[:,1]),recall=recall_score(y,hard),
                    predicted_positive_rate=hard.mean(),balanced_accuracy=balanced_accuracy_score(y,hard)))
            for archive,prefix in ((ridge,"ridge_"),(xgb,"")):
                with np.load(BytesIO(archive.read(key+"/predictions.npz"))) as preds:
                    for k,pred in preds.items():
                        add(prefix+k,pred)
            for arm in ("PIv1","direct_gain"):
                suffix = f"/seed-{info['seed']}/{info['dataset']}/{arm}/predictions.npz"
                member, = [p for p in historical.namelist() if "PIv1-all-seeds-" in p and p.endswith(suffix)]
                with np.load(BytesIO(historical.read(member))) as pred:
                    np.testing.assert_array_equal(y,pred["y"])
                    add("old_"+arm,pred["selected"])
    frame = pd.DataFrame(rows)
    frame.to_csv(legacy.DEST/"xgb-receiver/binary-scores.csv",index=False)
    summary = frame.groupby(["dataset","method"]).mean(numeric_only=True).drop(columns="seed")
    summary.to_csv(legacy.DEST/"xgb-receiver/binary-sources.csv")
    print(summary.loc["credit_card_fraud"].round(4).to_string())


if __name__=="__main__":
    main()
