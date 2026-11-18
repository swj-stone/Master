"""Development-only checks of full-row and standardized document voting."""
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
from minppi_mofs.consensus_document_minpi import consensus_document_top_from_training
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training
from minppi_mofs.standard_document_minpi import standard_document_top_from_training


DEVELOPMENT = ("sms", "amazon_cells", "imdb", "yelp")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "src/minppi_mofs/consensus_document_minpi.py",
               root / "src/minppi_mofs/standard_document_minpi.py"]
    write_json(output / "protocol.json", {
        "role": "development, fold 1 only, not independent confirmation",
        "datasets": DEVELOPMENT, "source_sha256": {str(f.relative_to(root)): file_hash(f) for f in sources},
        "note": "Fixed width is chosen by the old selector on the same training rows.",
    })
    results = []
    for X, y, Xt, truth, meta in panels():
        if meta["fold"] != 1 or meta["dataset"] not in DEVELOPMENT:
            continue
        seed = meta["seed"]
        old, _ = fast_document_vote_top_from_training(X, y, random_state=seed)
        supports = {"old": old, "full": np.arange(X.shape[1])}
        for name, method in (("consensus", consensus_document_top_from_training),
                             ("standard", standard_document_top_from_training)):
            for peers in (1, 4):
                for count in (None, len(old)):
                    budget = "auto" if count is None else "old_width"
                    label = f"{name}_{peers}_{budget}"
                    columns, _ = method(X, y, count, random_state=seed, max_peers=peers)
                    supports[label] = columns
                    supports[label + "_F"] = select("F_match", X, y, seed, len(columns))
                    scores = pooled_binary_gain(X, y)
                    supports[label + "_pooled"] = np.sort(np.argsort(-scores, kind="stable")[:len(columns)])
        archive = {}
        for name, columns in supports.items():
            pred = fit_predict(X, y, Xt, columns, "classification", seed, False)
            results.append(dict(dataset=meta["dataset"], fold=meta["fold"], seed=seed, method=name,
                                width=len(columns), p=X.shape[1], score=balanced_accuracy_score(truth, pred)))
            archive[name + "_cols"] = columns
            archive[name + "_pred"] = pred
        np.savez_compressed(output / f"{meta['dataset']}-1-{seed}.npz", **archive)
    pd.DataFrame(results).to_csv(output / "cases.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output)
