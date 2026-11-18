"""Archive the joint-coefficient candidate on the four designated development folds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import panels, select
from experiments.fast_document_vote_benchmark import pooled_binary_gain
from experiments.minpi_constant_transfer import fit_predict
from experiments.stable_minpi_stress import file_hash, write_json
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training
from minppi_mofs.joint_document_minpi import joint_document_top_from_training


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "src/minppi_mofs/joint_document_minpi.py"]
    write_json(output / "protocol.json", {
        "role": "development, fold 1 of SMS, Amazon, IMDb and Yelp",
        "max_iter": 8, "tol": .001,
        "source_sha256": {str(f.relative_to(root)): file_hash(f) for f in sources},
    })
    results = []
    for X, y, Xt, truth, meta in panels():
        if meta["fold"] != 1 or meta["dataset"] not in ("sms", "amazon_cells", "imdb", "yelp"):
            continue
        seed = meta["seed"]
        old, _ = fast_document_vote_top_from_training(X, y, random_state=seed)
        supports = {"old": old, "full": np.arange(X.shape[1])}
        for label, count in (("auto", None), ("old_width", len(old))):
            columns, _ = joint_document_top_from_training(X, y, count, random_state=seed)
            supports[label] = columns
            supports[label + "_F"] = select("F_match", X, y, seed, len(columns))
            gain = pooled_binary_gain(X, y)
            supports[label + "_pooled"] = np.sort(np.argsort(-gain, kind="stable")[:len(columns)])
        archive = {}
        for method, columns in supports.items():
            pred = fit_predict(X, y, Xt, columns, "classification", seed, False)
            results.append(dict(dataset=meta["dataset"], fold=meta["fold"], seed=seed, method=method,
                                width=len(columns), p=X.shape[1], score=balanced_accuracy_score(truth, pred)))
            archive[method + "_cols"] = columns
            archive[method + "_pred"] = pred
        np.savez_compressed(output / f"{meta['dataset']}-1-{seed}.npz", **archive)
    pd.DataFrame(results).to_csv(output / "cases.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output)
