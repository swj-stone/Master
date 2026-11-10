import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments import thesis_fixed_predictor as worker


def test_blend_gate_needs_all_three_scores_and_a_strict_mean_half_gain():
    assert worker.adopt_blend((.7,.8,.6), (.7,.81,.6))
    assert not worker.adopt_blend((.7,.8,.6), (.72,.79,.65))
    assert not worker.adopt_blend((.7,.8,.6), (.71,.8,.6))


@pytest.mark.parametrize("mask", [1,3])
@pytest.mark.parametrize("recipe", ["column_blend", "rounds"])
def test_deduplicated_fits_frozen_before_test_access(tmp_path, monkeypatch, mask, recipe):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out"
    monkeypatch.setattr(worker, "SOURCE", source)
    features = ["a", "b"]
    row = dict(dataset_id="madelon", seed=42, selected_features_json=json.dumps(features), target="y",
               task="binary", utility="roc_auc", protocol_hash="a"*64)
    for split, size in (("train",16), ("val",12), ("test",8)):
        p = source / (split + ".csv")
        y = np.arange(size) % 2
        pd.DataFrame({"a": y, "b": 1-y, "y": y}).to_csv(p, index=False)
        row.update({f"{split}_file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    f"{split}_relpath": p.name, f"{split}_row_count": size})
    with (source / "data_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (source / "resolved_protocol.yaml").write_text("backend: cpu\nmodel_params_by_dataset:\n  madelon: {n_estimators: 4}\n")
    frozen = tmp_path / "manifest.json"
    frozen.write_text(json.dumps({"seed":42,"datasets":{"madelon":{"selected_mask":hex(mask),
        "feature_order_sha256":worker.canonical_hash(features), "selected_source":"structured",
        "official_validation":{"structured":[.5]*3}, "full_validation":[.5]*3}}}))
    fitted = []

    class Fake:
        alternative = False
        def set_params(self, **params):
            assert params == {"colsample_bynode":.5}
            self.alternative = True
            return self

        def fit(self, frame, y):
            fitted.append((len(frame), frame.shape[1], self.alternative))
            assert list(frame.columns) == [f"f{i}" for i in range(frame.shape[1])]
            if len(frame) == 28:
                progress = json.loads((output / "progress.json").read_bytes())
                assert progress["decisions"]["guarded"]["adopt_blend"]
                assert (output / "test_access.json").is_file()

        def predict_proba(self, frame):
            p = frame.iloc[:,0].to_numpy() if self.alternative else np.full(len(frame), .5)
            return np.column_stack((1-p, p))

    def make(**kwargs):
        assert kwargs == dict(task="binary", model_params={"n_estimators":4}, backend="cpu", seed=42)
        return Fake()

    def checkpoints(X, labels, Z, *, task, model_params, backend, seed, checkpoints):
        assert (task, model_params, backend, seed) == ("binary", {"n_estimators":8}, "cpu", 42)
        assert tuple(checkpoints) == (4,6,8)
        fitted.append((len(X), X.shape[1], "rounds"))
        if len(X) == 28:
            assert json.loads((output/"progress.json").read_bytes())["decisions"]["guarded"]["adopt_blend"]
            assert (output/"test_access.json").is_file()
        original = np.full((len(Z),2), .5)
        p = Z.iloc[:,0].to_numpy()
        last = np.column_stack((1-p,p))
        near_last = .5 * last + .5 * original
        return {4:original, 6:near_last, 8:last}

    monkeypatch.setattr(worker, "make_model", make)
    monkeypatch.setattr(worker, "fit_predict_checkpoints", checkpoints)
    worker.run("madelon", output, frozen, recipe=recipe)
    data = json.loads((output / "result.json").read_bytes())
    count = (1 if mask == 3 else 2) * (1 if recipe == "rounds" else 2)
    assert data["phase"] == "complete" and data["validation_fits"] == data["test_fits"] == count
    assert len(fitted) == count * 2
    with np.load(output / data["paired_test_predictions"]) as predictions:
        assert set(predictions.files) == {"y", *data["method_results"]}
        assert np.array_equal(predictions["guarded_chosen"], predictions["guarded_blend"])
        if recipe == "rounds":
            np.testing.assert_allclose(predictions["guarded_blend"][:,1], .125 + .75 * predictions["y"])
            assert data["parameters"]["blend_checkpoints"] == [6,8]
            assert data["parameters"]["original_rounds"] == 4
            assert "colsample_bynode" not in data["parameters"]
    assert data["method_results"]["guarded_chosen"]["test_utility"] == 1
    assert data["decisions"]["full"]["adopt_blend"]
