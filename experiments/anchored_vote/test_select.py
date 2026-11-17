import numpy as np
import pytest

from experiments.anchored_vote.select import select


@pytest.mark.parametrize("width",[1,4,11])
def test_guard_respects_its_observed_anchor(width):
    rng=np.random.default_rng(15)
    x=rng.normal(size=(110,width)).astype("float32")
    if width>1:
        x[:,-1]=x[:,0]
    y=(x[:,0]+rng.normal(size=len(x))>.1).astype(int)
    idx=dict(fit=np.arange(60),vote=np.arange(60,75),width=np.arange(75,90),test=np.arange(90,110))
    y[idx["test"]]=-1
    for method in ("min","mean","gain"):
        row,pred=select(x,y,"classification",idx,42,method,"cpu")
        assert set(row["selected"])<=set(row["anchor"])
        assert row["utility"]>=max(row["anchor_curve"].values())-.005-1e-14
        assert row["fits"]<=7
        assert pred.shape==(20,2)
        np.testing.assert_allclose(pred.sum(axis=1),1.,atol=1e-6)
