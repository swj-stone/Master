import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from minppi_mofs.piv3_compact_tree import _encoded_paths, _marginals, _rank_paths, choose_prefix, select


def test_vectorized_marginals_equal_two_actual_game_evaluations():
    rng = np.random.default_rng(8)
    paths = {frozenset(map(int, rng.choice(12, rng.integers(1, 6), replace=False))): rng.uniform()
             for _ in range(100)}
    used = list(range(12))
    groups = _encoded_paths(paths, used)
    def game(s):
        return sum(weight for path, weight in paths.items() if path <= s)
    for _ in range(5):
        positions = rng.permutation(len(used))
        order = positions.argsort()
        for k in range(len(used)):
            observed = _marginals(groups, positions, k)
            for j in used:
                coalition = set(order[order != j][:k])
                assert observed[j] == pytest.approx(game(coalition | {j})-game(coalition), abs=1e-12)


def test_min_and_mean_change_the_same_draw_ranking():
    rng = np.random.default_rng(9)
    paths = {frozenset(map(int, rng.choice(20, rng.integers(1, 5), replace=False))): rng.uniform(.1, 2)
             for _ in range(90)}
    a, da = _rank_paths(paths, list(range(20)), 42, "min")
    b, db = _rank_paths(paths, list(range(20)), 42, "mean")
    np.testing.assert_array_equal(da["tier_rates"], db["tier_rates"])
    np.testing.assert_array_equal(da["vote_score"], np.min(da["tier_rates"], axis=0))
    assert not np.array_equal(a, b)


def test_frozen_mask_sizing_keeps_important_column_and_control_uses_same_gates(monkeypatch):
    rng = np.random.default_rng(9)
    table = pd.DataFrame(rng.normal(size=(600, 30)), columns=[f"sensor_{i}" for i in range(30)])
    y = (table.iloc[:,7].to_numpy() + .15*rng.normal(size=len(table)) > 0).astype(int)
    train, rank = table.iloc[:400], table.iloc[400:]
    model = XGBClassifier(n_estimators=15,max_depth=3,n_jobs=1,random_state=42).fit(train, y[:400])
    def forbidden(*args, **kwargs):
        raise AssertionError("No predictor fits during ranking or sizing")
    monkeypatch.setattr(model, "fit", forbidden)
    for agg in ("min","mean","gain"):
        keep,info = select(model,train,y[:400],rank,y[400:],task="binary",utility="roc_auc",aggregate=agg)
        assert keep == [table.columns[7]]
        assert info["front_fits"] == 0
        assert len(info["masked_proposals"]) <= 12
        again, sizing = choose_prefix(model,train,rank,y[400:],order=info["order"],task="binary",utility="roc_auc")
        assert keep == again and info["masked_proposals"] == sizing["masked_proposals"]
    with pytest.raises(ValueError,match="every column"):
        choose_prefix(model,train,rank,y[400:],task="binary",utility="roc_auc",order=[table.columns[1]])

