"""The optimized point arm must use the original point weights and compute one panel."""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from xgboost import XGBClassifier

from experiments.context_study.selection import single_panel, select_vote
from experiments.context_vote_selector import select
from minppi_mofs.piv3_compact_tree import select as gain_anchor
from minppi_mofs.piv3_compact_witness import _sample


@pytest.mark.parametrize("sizes", [[28, 30], [901, 20, 3], [12, 75, 40, 500]])
def test_only_point_reproduces_original_panel(sizes):
    labels = np.repeat(np.arange(len(sizes)), sizes)
    seed = np.random.SeedSequence([20421, 659])
    rows, weights = single_panel(labels, seed)
    original_rows, original_weights = _sample(labels, "multiclass", seed)
    np.testing.assert_array_equal(rows, original_rows)
    np.testing.assert_allclose(weights, original_weights[:1], atol=1e-16, rtol=1e-14)


@pytest.mark.parametrize("arm,old", [("point","point"),("min","min"),("pooled","mean")])
def test_fixed_nested_adapter_matches_original(arm, old):
    rng = np.random.default_rng(311)
    x = rng.normal(size=(300,10)).astype("float32")
    y = ((x[:,0]*x[:,1]+x[:,2] + .6*rng.normal(size=len(x)))>0).astype(int)
    train, rank = x[:175], x[175:]
    yt, yr = y[:175], y[175:]
    model = XGBClassifier(n_estimators=40,max_depth=3,n_jobs=1,random_state=20421).fit(train,yt)
    cols = [f"c{i}" for i in range(x.shape[1])]
    ctx = SimpleNamespace(full_model=model,train=pd.DataFrame(train,columns=cols),
                          ranking=pd.DataFrame(rank,columns=cols),yt=yt,yr=yr,
                          task="binary",seed=20421,utility="ovr_auc")
    _, gain = gain_anchor(model,ctx.train,yt,ctx.ranking,yr,task="binary",utility="ovr_auc",
                          seed=20421,aggregate="gain")
    support, trace = select_vote(arm,model,train,rank,yr,[int(c[1:]) for c in gain["order"]],gain["k"],0)
    original = select(ctx, arm=old)
    assert [cols[i] for i in support] == original.selected
    assert [(p["before"],p["accepted"]) for p in trace["trace"]] == [
        (p["before"],p["accepted"]) for p in original.diagnostics["trace"]]
    assert trace["n_panels"] == (1 if arm=="point" else 4)

