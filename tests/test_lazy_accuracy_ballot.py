from itertools import combinations

import numpy as np
import pytest

from experiments.test.minpi_convergence_metrics import make_scorer
from minppi_mofs.lazy_ballot import LazyAccuracyBallot
from minppi_mofs.piv3_compact_witness import _sample, _within


@pytest.mark.parametrize("classes,rare", [(2,False), (2,True), (4,False)])
def test_every_panel_matches_masked_reference(classes, rare):
    rng = np.random.default_rng(3181)
    y = np.r_[np.zeros(96,dtype=int),np.ones(4,dtype=int)] if rare else np.tile(np.arange(classes),24)
    x = rng.integers(-2,3,size=(len(y),5)).astype(np.float32)
    coef = rng.normal(size=(5,classes))
    task = "binary" if classes==2 else "multiclass"
    ids,weights = _sample(y,task,18372)
    x, y = x[ids], y[ids]
    def predict(xx):
        z = np.exp(xx @ coef)
        return z / z.sum(axis=1,keepdims=True)
    scorer = make_scorer(y,weights,task,"ovr_auc",rare)
    bg = np.zeros(5,dtype=np.float32)
    anchor = frozenset(range(3))
    lazy = LazyAccuracyBallot(predict,x,bg,y,weights,anchor,scorer)
    def full_score(mask):
        data = np.broadcast_to(bg,x.shape).copy()
        columns = sorted(mask)
        data[:,columns] = x[:,columns]
        return scorer(predict(data))
    for size in range(1,4):
        for subset in map(frozenset,combinations(sorted(anchor),size)):
            for ext in (frozenset(), frozenset([3]), frozenset([3,4])):
                given = full_score(anchor|ext)
                after = full_score(subset|ext)
                expected = tuple(_within(a,b) for a,b in zip(after,given,strict=True))
                assert lazy(subset,ext) == expected
                assert lazy(subset,ext) == expected


def test_all_accuracy_votes_fail_without_sorting_and_both_masks_batched_once():
    y = np.array([0,1,0,1])
    x = np.column_stack((2*y-1, np.zeros_like(y)))
    def predict(rows):
        p = .5 + .4*rows[:,0]
        return np.column_stack((1-p,p))
    def omitted(_):
        raise AssertionError("Unneeded AUC/AP score")
    lazy = LazyAccuracyBallot(predict,x,np.zeros(2),y,np.ones((4,4)),{0,1},omitted)
    assert lazy(frozenset([1]),frozenset()) == (False,)*4
    assert lazy.dispatches==1 and len(lazy.accuracy)==2 and len(lazy.scores)==0
