import numpy as np
import pandas as pd
import pytest

from minppi_mofs import piv3_compact_witness as witness


def _s(x):
    return dict(accuracy=x,auc=x)


def test_weights_and_draws_restore_natural_prevalence():
    y=np.r_[np.zeros(20000,dtype=int),np.ones(20,dtype=int)]
    indices,w=witness._sample(y,"binary",42)
    assert len(indices)==256 and y[indices].sum()==20
    np.testing.assert_allclose(w.sum(axis=1),1.)
    np.testing.assert_allclose(w[:,y[indices]==1].sum(axis=1),20/20020)
    assert np.array_equal(indices,witness._sample(y,"binary",42)[0])


def test_regular_veto_ap_safety_and_anchor_check():
    before=[[_s(.9)]*4 for _ in range(3)]
    after=[[_s(.9)]*4,[_s(.9)]*4,[_s(.8)]*4]
    for agg, good in [("min",False),("mean",True),("small",True)]:
        accepted,evidence=witness._decision(before,after,_s(.9),agg)
        assert bool(accepted) == good
        assert evidence["net"]==[1,1,-1]
    assert not witness._decision(before,before,_s(1.),"min")[0]
    assert not witness._within({"ap":.47},{"ap":.50})
    assert witness._within({"mae":1.02},{"mae":1.0})


def test_mask_caching_and_return_order_with_constant_frozen_model(monkeypatch):
    p=14
    train=pd.DataFrame(np.arange(100*p).reshape(100,p),columns=[f"j{i}" for i in range(p)])
    y=np.arange(100)%2
    gain_order=train.columns[::-1].tolist()
    k=8
    anchor=list(reversed(gain_order[:k]))
    def fixed_anchor(*args,**kwargs):
        return anchor,dict(order=gain_order,k=k)
    monkeypatch.setattr(witness,"gain_anchor",fixed_anchor)
    class Teacher:
        def __init__(self):
            self.calls=0
        def predict_proba(self,X):
            self.calls+=1
            return np.tile([.4,.6],(len(X),1))
        def fit(self,*args,**kwargs):
            pytest.fail("The witness game must not train the teacher")
    m=Teacher()
    names,d=witness.select(m,train,y,train,y,task="binary",utility="roc_auc")
    assert 1<=len(names)<len(anchor)
    assert set(names)==set(d["order"][:len(names)])
    assert set(anchor)==set(d["order"][:k])
    assert d["anchor_order"]==gain_order
    assert m.calls==d["witness_predictions"]<=18
    assert len(d["witnesses"])<=5
    assert all(x["accepted"] for x in d["witnesses"])

