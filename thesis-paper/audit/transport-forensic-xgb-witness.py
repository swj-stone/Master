"""A supplied synthetic split on which the actual Gain arm nominates and admits R."""
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xgboost
from threadpoolctl import threadpool_limits

from experiments.context_review.budgeted_guard import candidate_pool, metrics
from experiments.context_review.transport_predict import nominate, policy
from experiments.context_study.benchmark import fit
from minppi_mofs.piv3_compact_tree import select
from minppi_mofs.piv3_compact_witness import _within


GRID = np.array(list(product((0,1),repeat=2)),dtype=np.float32)
COUNTS = [(19,19),(9,9),(20,1),(1,5)]
GUARD = [(0,0),(1,1),(46,0),(2,33)]


def main():
    base = np.repeat(GRID,[a+b for a,b in COUNTS],axis=0)
    labels = np.concatenate([np.r_[np.zeros(a),np.ones(b)] for a,b in COUNTS]).astype(int)
    train,yt = np.tile(base,(6,1)), np.tile(labels,6)
    seed=20421
    initial=fit(train,yt,seed)
    columns=["D","R"]
    kept,info=select(initial,pd.DataFrame(train,columns=columns),yt,pd.DataFrame(base,columns=columns),
                     labels,task="binary",utility="ovr_auc",seed=seed,aggregate="gain")
    assert kept==["D","R"] and info["order"]==["R","D"] and info["k"]==2
    assert len(initial.get_booster().get_score(importance_type="weight"))==2
    assert info["masked_proposals"][0]["passes"]==False
    order=[1,0]
    nominee,_,_=nominate("gain",initial,train,base,labels,seed,123,0,order,2,False)
    assert nominee==(1,) and nominee in candidate_pool(order,1,123,0)[1]
    bg=np.median(train,axis=0)
    np.testing.assert_array_equal(bg,[0.,0.])
    inner=np.vstack([train,base])
    yi=np.r_[yt,labels]
    guard = np.repeat(GRID,[a+b for a,b in GUARD],axis=0)
    guard_y = np.concatenate([np.r_[np.zeros(a),np.ones(b)] for a,b in GUARD]).astype(int)
    models=[fit(inner[:,c],yi,seed) for c in [[0,1],[1]]]
    guard_metrics=[metrics(guard_y,m.predict_proba(guard[:,c]),False)
                   for m,c in zip(models,[[0,1],[1]])]
    assert _within(guard_metrics[1], guard_metrics[0])
    x=np.vstack([inner,guard])
    y=np.r_[yi,guard_y]
    outer_models=[fit(x[:,c],y,seed) for c in [[0,1],[1]]]
    outer_y=np.r_[np.ones(136),np.zeros(196)].astype(int)
    low=np.zeros(len(outer_y),dtype=np.float32)
    bad_r=np.roll(outer_y,136)
    assert (bad_r[outer_y==1]==0).all() and sum(bad_r)==sum(outer_y)
    worlds=[np.column_stack([low,r]).astype(np.float32) for r in (outer_y,bad_r)]
    for mask in product((False,True),repeat=2):
        for frozen in (initial,models[0],outer_models[0]):
            prob=[frozen.predict_proba(np.where(mask,w,bg)) for w in worlds]
            np.testing.assert_array_equal(prob[0],prob[1])
    reference=[]
    reduced=[]
    for w in worlds:
        reference.append(metrics(outer_y,outer_models[0].predict_proba(w),False))
        reduced.append(metrics(outer_y,outer_models[1].predict_proba(w[:,[1]]),False))
    print("training, proposal, guard, outer",len(yt),len(labels),len(guard_y),len(outer_y))
    print("Gain order, anchor, nominee",info["order"],kept,nominee,"prefix",info["masked_proposals"])
    print("initial full scores 00,01,10,11",initial.predict_proba(GRID)[:,1])
    print("XGBoost",xgboost.__version__)
    for name,m in (("initial",initial),("guard full",models[0]),("final full",outer_models[0])):
        ids=m.apply(GRID)
        if name!="final full":
            np.testing.assert_array_equal(ids[0],ids[1])
        print(name,"trees reaching distinct leaves at D=0",int(np.sum(ids[0]!=ids[1])),
              "positive probabilities at D=0",m.predict_proba(GRID[:2])[:,1])
    print("guard metrics anchor, nominee",guard_metrics)
    print("final full scores 00,01,10,11",outer_models[0].predict_proba(GRID)[:,1])
    print("final R-only probability",outer_models[1].predict_proba(GRID[:2,[1]])[:,1])
    print("final anchor and reduced metrics by world",reference,reduced)
    assert reference[0]==reference[1]=={"auc":.5,"accuracy":196/332}
    np.testing.assert_allclose([reduced[1]["auc"],reduced[1]["accuracy"]],[15/98,60/332],rtol=0,atol=1e-15)
    assert _within(reduced[0],reference[0]) and not _within(reduced[1],reference[1])
    for w in worlds:
        inputs=dict(x=x,y=y,x_outer=w,train=np.arange(498),proposal=np.arange(498,581),guard=np.arange(581,664))
        with TemporaryDirectory(prefix="transport-forensic-ephemeral-",dir=Path(__file__).parent) as scratch:
            selected, probabilities, costs, _ = policy(Path(scratch),inputs,"gain",0,123)
        assert selected["passes"] and selected["selected"]==selected["nominee"]==(1,)
        np.testing.assert_array_equal(probabilities,outer_models[1].predict_proba(w[:,[1]]))
        assert all(costs[k]==1 for k in ("initial_fits","guard_anchor_fits","guard_candidate_fits","final_fits"))
    print("Actual Gain policy retained R in both worlds; the final AUC/accuracy pair passes only in the first.")


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
