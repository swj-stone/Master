import json

import numpy as np
import pandas as pd
import pytest

from minppi_mofs import piv3


def test_replicated_min_and_reproducibility():
    rng = np.random.default_rng(21)
    x = pd.DataFrame(rng.normal(size=(120, 10)), columns=list("abcdefghij"))
    labels = (x.a > 0).astype(int)
    def evaluate(cols):
        return [.9, .85] if "a" in cols else [.9, .72]
    settings = dict(search_seed=11, tolerance=.05, reference_scores=[.9,.85], evaluate=evaluate)
    kept, meta = piv3.search(x, labels, **settings)
    assert (kept, meta) == piv3.search(x, labels, **settings)
    assert kept == ["a"]
    assert len(set(meta["vote_seeds"])) == 3
    np.testing.assert_allclose(meta["min_vote"], np.min(meta["tier_rates"], axis=0))
    json.dumps(meta, allow_nan=False)


def test_dual_thresholds_repairs_and_budget(monkeypatch):
    x = pd.DataFrame(np.ones((30,10)), columns=list("abcdefghij"))
    def training_order(train, y, seed, task="binary"):
        return list(range(10)), frozenset([0]), 4, {}
    monkeypatch.setattr(piv3, "_training_order", training_order)
    calls = []
    def evaluate(columns):
        calls.append(tuple(columns))
        # Small failed anchor can be repaired. AUC alone would accept every mask.
        return [.95, .92 if len(columns) >= 3 else .83]
    selected, meta = piv3.search(x, [0,1]*15, search_seed=11, tolerance=.01, reference_scores=[.95,.92],
                                 evaluate=evaluate, budget=12)
    trace = meta["steps"]
    index = {0: x.columns.tolist(), **{r["id"]:r["proposed"] for r in trace}}
    assert len(trace) == len(set(calls)) <= 12
    assert all(list(row) == [c for c in x if c in row] for row in calls)
    assert not any(r["accepted"] for r in trace if r["k"] < 3)
    assert len(selected) == min(r["k"] for r in trace if r["accepted"])
    assert selected == index[meta["winner_id"]]
    for row in trace:
        parent = set(index[row["parent"]])
        current = set(row["proposed"])
        assert current-parent == set(row["added"]) and parent-current == set(row["removed"])
        assert row["accepted"] == (row["auc"] >= .95*.99 and row["accuracy"] >= .92*.99)
    assert any(r["operation"] == "refill" and r["accepted"] and r["added"] for r in trace)
    assert any(r["operation"] == "exchange" and len(r["removed"]) == len(r["added"]) == 1 for r in trace)


def test_fallback_if_only_one_metric_passes():
    x = pd.DataFrame(np.arange(90).reshape(30,3), columns=["z","b","a"])
    selected, meta = piv3.search(x, [0,1]*15, search_seed=3, tolerance=.01, reference_scores=[.9,.9],
                                 evaluate=lambda c:[.899,.8])
    assert selected == x.columns.tolist() and meta["winner_id"] == 0
    assert 0 < len(meta["steps"]) <= 6
    with pytest.raises(ValueError):
        piv3.search(x, [0,1]*15, search_seed=3, tolerance=2, reference_scores=[.9,.9],
                    evaluate=lambda c:[.899,.8])


def test_multiclass_proposal_uses_all_classes_and_dual_floor():
    rng = np.random.default_rng(31)
    labels = np.tile([0, 1, 2], 50)
    x = pd.DataFrame(dict(signal=labels, noise=rng.normal(size=len(labels)), constant=0))
    kept, meta = piv3.search(x, labels, task="multiclass", search_seed=11, tolerance=.05,
                             reference_scores=[.9,.8], evaluate=lambda cols: [.87,.79] if "signal" in cols else [.89,.72])
    assert kept == ["signal"]
    np.testing.assert_allclose(meta["association"][0], 2*len(labels))
    assert meta["association"][-1] == 0
    assert all(r["accepted"] == (r["auc"] >= .855 and r["accuracy"] >= .76) for r in meta["steps"])


@pytest.mark.parametrize("reference,scores,expected", [
    ((.8, 10), (.77, 10.5), True), ((.8, 10), (.84, 10.51), False),
    ((.8, 10), (.75, 8), False), ((0., 0.), (.1, 0.), True), ((0., 0.), (.1, .00001), False),
])
def test_regression_relative_floors_and_mae_ceiling(reference, scores, expected):
    rng = np.random.default_rng(37)
    y = np.arange(65, dtype=float)
    x = pd.DataFrame(dict(signal=y, noise=rng.normal(size=len(y)), constant=0))
    kept, meta = piv3.search(x, y, task="regression", search_seed=11, tolerance=.05,
                             reference_scores=reference, evaluate=lambda cols: scores)
    assert meta["training_core"] == ["signal"]
    assert meta["steps"] and all(row["accepted"] == expected for row in meta["steps"])
    assert all((row["r2"], row["mae"]) == scores for row in meta["steps"])
    assert (len(kept) < len(x.columns)) == expected
    json.dumps(meta, allow_nan=False)
