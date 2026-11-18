"""Score frozen ridge-selected supports with a second learner, without recalibration."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import CONFIG, file_hash, forest
from experiments.stable_minpi_stress import DATASETS, PRIMARY_SEED, digest, inputs, metric


def run(parent, out):
    manifest = json.loads((parent / "protocol.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {}
    methods = ("stable", "environment_half", "local_half", "full", "F_policy")
    for name in DATASETS:
        X, y, task, _, _source, _duplicates = inputs(name)
        assert digest(X) == manifest["input_hashes"][name]["X"]
        assert digest(y) == manifest["input_hashes"][name]["y"]
        for fold in (1, 2, 3):
            path = parent / f"{name}-{fold}-outputs.npz"
            split_path = parent / f"{name}-{fold}-split.npz"
            hashes[path.name], hashes[split_path.name] = file_hash(path), file_hash(split_path)
            with np.load(split_path) as split:
                tr, te = split["train"], split["test"]
                np.testing.assert_array_equal(split["gold"], y[te])
            with np.load(path) as saved:
                preds = {}
                for method in methods:
                    cols = saved[method + "_support"]
                    assert len(cols) and np.array_equal(np.unique(cols), cols)
                    key = tuple(cols)
                    if key not in preds:
                        estimator = forest(task, PRIMARY_SEED)
                        estimator.fit(X[tr][:, cols], y[tr])
                        preds[key] = estimator.predict(X[te][:, cols])
                    pred = preds[key]
                    rows.append(dict(dataset=name, fold=fold, method=method, width=len(cols),
                                     p=X.shape[1], score=metric(y[te], pred, task),
                                     support_sha256=digest(cols), prediction_sha256=digest(pred)))
    frame = pd.DataFrame(rows)
    baseline = frame[frame.method == "full"][["dataset", "fold", "score"]].rename(columns={"score":"full_score"})
    frame = frame.merge(baseline, on=["dataset", "fold"], validate="many_to_one")
    frame["delta_full"] = frame.score - frame.full_score
    frame.to_csv(out / "folds.csv", index=False)
    frame.groupby(["dataset", "method"], as_index=False)[["width","score","delta_full"]].mean().to_csv(
        out / "summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(dict(
        methods=methods, training_seed=PRIMARY_SEED, parent_protocol_sha256=file_hash(parent/"protocol.json"),
        files_sha256=hashes, config_sha256=file_hash(CONFIG), runner_sha256=file_hash(Path(__file__)),
        learner="Extremely randomized trees, project preset, 48 trees; selection and seed frozen before transfer",
        metric="balanced accuracy for classification, R2 for regression", timing="quality only"
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    with threadpool_limits(limits=1):
        run(options.parent, options.output)
