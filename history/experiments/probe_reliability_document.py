"""Four fixed development folds for one-sketch directional reliability votes."""
from time import perf_counter

import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import panels
from experiments.minpi_constant_transfer import fit_predict
from experiments.probe_loo_document import select_f, select_pooled
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training
from minppi_mofs.reliability_document_minpi import reliability_document_top_from_training


if __name__ == "__main__":
    rows = []
    with threadpool_limits(1):
        for X, y, Xt, truth, meta in panels():
            if meta["dataset"] not in ("sms", "amazon_cells", "imdb", "yelp") or meta["fold"] != 1:
                continue
            old_width = None
            for name, function in (
                ("previous", fast_document_vote_top_from_training),
                ("reliability", reliability_document_top_from_training),
                ("reliability_same_width", reliability_document_top_from_training),
                ("F_same_width", select_f), ("pooled_same_width", select_pooled),
            ):
                start = perf_counter()
                cols, info = function(X, y, old_width if name.endswith("same_width") else None,
                                      random_state=meta["seed"])
                middle = perf_counter()
                if name == "previous":
                    old_width = len(cols)
                pred = fit_predict(X, y, Xt, cols, "classification", meta["seed"], False)
                end = perf_counter()
                rows.append(dict(dataset=meta["dataset"], method=name, width=len(cols),
                                 score=balanced_accuracy_score(truth, pred),
                                 select_ms=1000*(middle-start), total_ms=1000*(end-start)))
    print(pd.DataFrame(rows).to_csv(index=False))
