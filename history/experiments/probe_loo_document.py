"""Inspect four predeclared development folds for all-row LOO presence voting."""
from time import perf_counter

import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import panels
from experiments.fast_document_vote_benchmark import pooled_binary_gain
from experiments.compare_predictive_minpi import choose
from experiments.minpi_constant_transfer import fit_predict
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training
from minppi_mofs.loo_document_minpi import loo_document_top_from_training
from sklearn.metrics import balanced_accuracy_score
import numpy as np


def select_f(X, y, count, *, random_state):
    return choose("f", X, y, "classification", random_state, count), {}


def select_pooled(X, y, count, *, random_state):
    rank = np.argsort(-pooled_binary_gain(X, y), kind="stable")
    return np.sort(rank[:count]), {}


def select_relative_loo(X, y, count, *, random_state):
    """Development probe using log loss ratios in both fixed LOO contexts."""
    _, info = loo_document_top_from_training(X, y, random_state=random_state)
    assert info["estimator"] == "loo_fixed_peers"
    labels = np.unique(y, return_inverse=True)[1]
    prior = labels.mean()
    baseline = 2. * prior * (1. - prior) / (1. - 1. / len(labels))**2
    single_loss = baseline - info["zero_peer_scores"]
    first, alternate = info["peers"]
    peer = np.full(X.shape[1], first)
    peer[first] = alternate
    zero = -np.log1p(-info["zero_peer_scores"] / baseline)
    one = -np.log1p(-info["one_peer_scores"] / single_loss[peer])
    scores = np.minimum(zero, one)
    scale = np.max(np.abs(scores))
    relative = np.round(scores / scale, 12) if scale > 0 else scores
    rank = np.lexsort((np.arange(X.shape[1]), -info["marginal_scores"], -relative))
    if count is None:
        positive = np.maximum(relative[rank], 0.).cumsum()
        needed = 1 + np.searchsorted(positive, .99 * positive[-1]) if positive[-1] > 0 else info["cap"]
        count = min(info["cap"], max(int(np.ceil(np.sqrt(X.shape[1]))), needed))
    return np.sort(rank[:count]), dict(info, scores=scores, zero_peer_scores=zero, one_peer_scores=one,
                                       budget=int(count), estimator="loo_relative_probe")


if __name__ == "__main__":
    records = []
    with threadpool_limits(1):
        for X, y, Xt, truth, info in panels():
            if info["dataset"] not in ("sms", "amazon_cells", "imdb", "yelp") or info["fold"] != 1:
                continue
            previous_width = None
            for name, selector in (("previous", fast_document_vote_top_from_training),
                                   ("loo", loo_document_top_from_training),
                                   ("loo_same_width", loo_document_top_from_training),
                                   ("loo_relative", select_relative_loo),
                                   ("loo_relative_same_width", select_relative_loo),
                                   ("F_same_width", select_f), ("pooled_same_width", select_pooled)):
                start = perf_counter()
                support, _ = selector(X, y, previous_width if name.endswith("same_width") else None,
                                      random_state=info["seed"])
                selected = perf_counter()
                if name == "previous":
                    previous_width = len(support)
                pred = fit_predict(X, y, Xt, support, "classification", info["seed"], False)
                end = perf_counter()
                records.append(dict(dataset=info["dataset"], method=name, width=len(support),
                                    score=balanced_accuracy_score(truth, pred),
                                    select_ms=1000*(selected-start), total_ms=1000*(end-start)))
    print(pd.DataFrame(records).to_csv(index=False))
