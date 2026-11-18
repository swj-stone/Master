import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from minppi_mofs.piv3_compact_filter import _size, _votes, select


def test_leave_one_coordinate_out_votes_equal_direct_coalitions():
    delta = np.random.default_rng(4).normal(size=(32, 9)).astype(np.float32)
    rates, strengths, tiers = _votes(delta,np.random.default_rng(14),"min")
    rng = np.random.default_rng(14)
    expected=np.zeros_like(rates)
    for h,k in enumerate(tiers):
        for _ in range(4):
            order=rng.permutation(delta.shape[1])
            for j in range(delta.shape[1]):
                others=order[order!=j][:k]
                total=delta[:,others].sum(axis=1)
                expected[h,j]+=np.sign(delta[total<=np.median(total),j]).mean()/4
    np.testing.assert_allclose(rates,expected,atol=1e-6)
    np.testing.assert_allclose(strengths,rates.min(axis=0))
    for name in ("mean","relief"):
        v, s, _ = _votes(delta,np.random.default_rng(14),name)
        np.testing.assert_array_equal(v,rates)
        want = v.mean(axis=0) if name=="mean" else np.sign(delta).mean(axis=0)
        np.testing.assert_allclose(s,want)
        assert not np.allclose(s,strengths)


def test_quality_preserving_prefix_changes_with_group_minority():
    rng=np.random.default_rng(72)
    labels=np.repeat([0,1],[80,20])
    group=np.repeat(labels,2)
    core=np.where(group==0,1.,-1.)
    extra=np.where(group==0,0.,2.)
    noise=rng.normal(0,.02,(len(group),5))
    k,info=_size(np.column_stack([core,extra,noise]),np.arange(7),labels)
    assert k==2 and info["best_coverage"]==1.


@pytest.mark.parametrize("task",["binary","multiclass","regression"])
def test_model_free_ranking_and_sparse_signal(task):
    rng=np.random.default_rng(55)
    x=rng.normal(size=(1200,48)); y=x[:,1]+rng.normal(0,.02,len(x))
    if task!="regression":
        y=np.digitize(y,[-.5,.5] if task=="multiclass" else [0.])
    frame=pd.DataFrame(x,columns=[f"f{j}" for j in range(48)])
    frame.loc[::11,"f4"]=np.nan
    frame["f6"]=1.
    with threadpool_limits(limits=1):
        s,d=select(frame[:900],y[:900],frame[900:],y[900:],task=task,utility="test")
        again=select(frame[:900],y[:900],frame[900:],y[900:],task=task,utility="test")
    assert s and "f1" in s and len(s)<48
    assert (s,d)==again
    assert set(s)==set(d["order"][:len(s)]) and set(d["order"])==set(frame)
    assert d["front_fits"]==0
    json.dumps(d,allow_nan=False)


def test_unique_names_and_empty_classes():
    x=pd.DataFrame({"a":np.arange(10),"b":np.arange(10)})
    cols,diag=select(x,np.ones(10),x,np.ones(10),task="binary",utility="auc")
    assert cols==list(x) and diag["bypass"]
    with pytest.raises(ValueError):
        select(x,np.ones(10),x.iloc[:,::-1],np.ones(10),task="binary",utility="auc")

