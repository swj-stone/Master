import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments import replicate_fixed_rounds as worker


def test_inner_sets_repeat_and_are_disjoint_for_both_task_types():
    y = np.tile(np.arange(4), 25)
    for task in ("multiclass", "regression"):
        first, info = worker.training_partitions(y, task)
        second, again = worker.training_partitions(y, task)
        assert info == again and info["sizes"] == [60,20,20]
        assert set(np.concatenate(first)) == set(range(len(y)))
        for a,b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a,b)
            if task == "multiclass":
                assert set(y[a]) == set(y)


def test_freezes_validation_choice_before_refit_and_evaluates_only_training_rows(tmp_path, monkeypatch):
    from minppi_mofs.experiment_protocol import canonical_hash
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(worker, "SOURCE", source)
    labels = np.tile([0,1], 50)
    train = source / "only_train.csv"
    pd.DataFrame({"recover": labels, "unused": labels*0, "y": labels}).to_csv(train,index=False)
    row = dict(dataset_id="madelon",seed=42,train_relpath=train.name,train_file_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
               train_row_count=len(labels),target="y",selected_features_json=json.dumps(["recover","unused"]),
               task="binary",utility="roc_auc")
    with (source/"data_manifest.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (source/"resolved_protocol.yaml").write_text("backend: cpu\nmodel_params_by_dataset:\n  madelon: {n_estimators: 3}\n")
    manifest = tmp_path/"masks.json"
    manifest.write_text(json.dumps(dict(seed=42,datasets={"madelon":{"selected_mask":"0x1",
                                "feature_order_sha256":canonical_hash(["recover","unused"])}})))
    output = tmp_path/"result"
    calls = []
    splits, _ = worker.training_partitions(labels, "binary")
    original_fit, validation, held = splits

    def mock_fit(X, y, Z, **kw):
        assert kw == dict(task="binary",model_params={"n_estimators":6},backend="cpu",seed=42,checkpoints=[3,5,6])
        assert list(X.columns) == ["recover"] and list(Z.columns) == ["recover"]
        calls.append(len(X))
        expected_fit, expected_pred = ((original_fit,validation) if len(calls)==1 else
                                        (np.sort(np.r_[original_fit,validation]),held))
        np.testing.assert_array_equal(X.index,expected_fit)
        np.testing.assert_array_equal(Z.index,expected_pred)
        np.testing.assert_array_equal(y,labels[expected_fit])
        if len(calls)==2:
            state = json.loads((output/"progress.json").read_bytes())
            assert state["decisions"]["guarded"]["adopt_blend"]
            assert state["phase"] == "internal_evaluation"
        p = Z.iloc[:,0].to_numpy()
        return {3:np.full((len(Z),2),.5), 5:np.column_stack((1-p,p)), 6:np.column_stack((1-p,p))}

    monkeypatch.setattr(worker, "fit_predict_checkpoints", mock_fit)
    worker.run("madelon",output,manifest)
    result=json.loads((output/"result.json").read_bytes())
    assert calls == [60,80] and result["source_partition"] == "train"
    assert result["test_fits"] == result["validation_fits"] == 1
    assert result["method_results"]["guarded_chosen"]["test_delta_vs_original"] == pytest.approx(.5)
    with np.load(output/result["paired_test_predictions"]) as p:
        np.testing.assert_array_equal(p["y"], labels[held])
        np.testing.assert_array_equal(p["guarded_chosen"],p["guarded_blend"])
