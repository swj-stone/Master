"""Audit shared real CPU fits, timing scope and held-out access in the joint experiment."""
import json
from threading import get_native_id

import numpy as np
import pandas as pd
import pytest

from experiments import literature_worker as worker


@pytest.mark.parametrize("mode,workers",[("native",1),("native",2),("frozen",1)])
@pytest.mark.parametrize("task,utility",[("binary","roc_auc"),("multiclass","ovr_auc"),("regression","r2")])
def test_pair_matches_independent_candidates_and_predictions(tmp_path,monkeypatch,mode,workers,task,utility):
    source=tmp_path/"inputs"/"synthetic"
    source.mkdir(parents=True)
    cols=list("abcd")
    rng=np.random.default_rng(73)
    data={}
    for part,rows in (("train",180),("val",120),("test",80)):
        x=rng.normal(size=(rows,len(cols))).astype("float32")
        signal=x[:,0]+.3*x[:,1]
        y=signal if task=="regression" else np.digitize(signal,[-.35,.35] if task=="multiclass" else [0.])
        data[part]=pd.DataFrame(x,columns=cols),y
        if part!="test":
            np.save(source/f"{part}.npy",x)
            np.save(source/f"{part}_y.npy",y)
    policy=dict(task=task,backend="cpu",seed=42,model_params=dict(n_estimators=5,max_depth=2,n_jobs=1))
    (source/"metadata.json").write_text(json.dumps(dict(features=cols,utility=utility,policy=policy,
          sha256={"test.npy":"synthetic","test_y.npy":"synthetic"})))
    monkeypatch.setattr(worker,"DEFAULT_ROOT",source.parent)
    pair=tmp_path/"pair"
    methods=("minshap","shapley")
    pair.mkdir()
    def load(_root,_dataset,part):
        if part=="test":
            assert all((pair/m/"frozen.json").is_file() for m in methods)
        return data[part]
    monkeypatch.setattr(worker,"load_split",load)
    if workers==2:
        fit=worker._Ledger.fit
        predict=worker._predict
        def owned_fit(*args,**kwargs):
            model=fit(*args,**kwargs)
            model.thread_owner_for_test=get_native_id()
            return model
        def owned_predict(model,x,regression):
            assert model.thread_owner_for_test==get_native_id()
            return predict(model,x,regression)
        monkeypatch.setattr(worker._Ledger,"fit",owned_fit)
        monkeypatch.setattr(worker,"_predict",owned_predict)
    joint=worker.trial("synthetic",mode,"permutation_pair",pair,enforce_guard=False,native_workers=workers)
    events=[json.loads(line) for line in (pair/"fits.jsonl").read_text().splitlines()]
    completed=[event for event in events if event["event"]=="completed"]
    assert joint["fits"]==len(completed)
    assert joint["method_fit_counts"]=={m:sum(m in event["methods"] for event in completed) for m in methods}
    assert joint["shared_reference_and_estimation_fits"]==sum(event["phase"] in ("reference","coalition") for event in completed)
    assert joint["native_workers"]==workers
    if workers==2:
        assert joint["native_executor"]=="spawn_processes"
        pids={e["pid"] for e in completed if e["phase"]=="coalition"}
        assert len(pids)==2 and pids==set(joint["native_fit_pids"])
        stages=[json.loads(line)["phase"] for line in (pair/"stages.jsonl").read_text().splitlines()]
        expected=["permutations_reduced_pool_closing","pool_joined","sizing_minshap","sizing_shapley","estimators_saved"]
        assert [phase for phase in stages if phase in expected]==expected
        pending={}
        peak=0
        hashes=set()
        for event in events:
            i=event["fit_id"]
            if event["event"]=="started":
                assert i not in pending
                pending[i]=event
                peak=max(peak,len(pending))
                if event["phase"]=="coalition":
                    assert event["prefix_sha256"] not in hashes
                    hashes.add(event["prefix_sha256"])
            else:
                start=pending.pop(i)
                assert event["thread_id"]==start["thread_id"] and start["k"]==event["k"]
                assert event.get("prefix_sha256")==start.get("prefix_sha256")
                assert start["elapsed_s"]<=event["elapsed_s"]
        assert not pending and peak==joint["peak_inflight_fits"]==2
    for m in methods:
        separate=tmp_path/("separate_"+m)
        independent=worker.trial("synthetic",mode,m,separate,enforce_guard=False)
        paired=json.loads((pair/m/"result.json").read_bytes())
        for key in ("candidate","selected","validation_full","validation_candidate","test_scores","fits","frozen_hash"):
            assert paired[key]==independent[key]
        if workers==2:
            parallel=worker.trial("synthetic",mode,m,tmp_path/("threaded_"+m),enforce_guard=False,native_workers=2)
            assert parallel["peak_inflight_fits"]==2
            for key in ("candidate","selected","fits","frozen_hash","test_scores"):
                assert parallel[key]==independent[key]
        details=[json.loads((folder/"estimator.json").read_bytes()) for folder in (separate,pair/m)]
        for key in ("scores","sampled_marginals","permutation_sha256"):
            assert details[0][key]==details[1][key]
        with (np.load(separate/"predictions.npz") as expected,np.load(pair/m/"predictions.npz") as actual):
            for key in expected.files:
                np.testing.assert_array_equal(actual[key],expected[key])
