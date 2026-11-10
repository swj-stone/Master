import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from minppi_mofs import residual_ranking


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_training_only_gain_and_recovery_match_task_residuals(monkeypatch, task):
    rows = np.arange(72)
    y = np.sin(rows) if task == "regression" else rows % (2 if task == "binary" else 3)
    train = pd.DataFrame({"retained[0]": rows / 71, "unused": rows % 7,
                          "restore": np.cos(rows / 2), "constant": 0., "witness": rows})
    fit, hold = (np.sort(x) for x in train_test_split(rows, test_size=.25, random_state=42,
                                                      stratify=None if task == "regression" else y))
    p = train.iloc[hold, 0].to_numpy() / 2
    if task == "regression":
        pred = p
        residual = (y[hold] - pred)[:, None]
    elif task == "binary":
        pred = np.column_stack([1-p, p])
        residual = (y[hold] - p)[:, None]
    else:
        pred = np.column_stack([p, (1-p) / 3, (1-p) * 2/3])
        residual = np.eye(3)[y[hold]] - pred

    class Model:
        def fit(self, frame, labels):
            assert list(frame.columns) == ["f0", "f1"]
            np.testing.assert_array_equal(frame.index, fit)
            np.testing.assert_array_equal(labels, y[fit])

        def predict(self, frame):
            np.testing.assert_array_equal(frame.index, hold)
            assert list(frame.columns) == ["f0", "f1"]
            return pred

        predict_proba = predict

        def get_booster(self):
            return self

        def get_score(self, *, importance_type):
            assert importance_type == "gain"
            return {"f0": 12.5}

    policy = {"task": task, "model_params": {"n_estimators": 7}, "backend": "cpu", "seed": 42}
    def model(**kwargs):
        assert kwargs == policy
        return Model()

    monkeypatch.setattr(residual_ranking, "make_model", model)
    signs = dict(zip(train.columns, [1, 0, 0, 0, -1], strict=True))
    recovered, gains, info = residual_ranking.rank_residual_recovery(
        train, y, tuple(train.columns), 3, signs, utility="unused", **policy)
    correlations = [np.corrcoef(train.iloc[hold]["restore"], col)[0, 1] for col in residual.T]
    assert set(recovered) == {"restore", "constant"}
    assert recovered["restore"] == pytest.approx(np.sqrt(np.mean(np.square(correlations))))
    assert recovered["constant"] == 0
    assert gains == {"retained[0]": 12.5, "unused": 0.}
    assert info["fit_count"] == 1
    assert info["split"]["sizes"] == [len(fit), len(hold)]
    assert info["split"]["index_sha256"] == [hashlib.sha256(x.astype("<i8").tobytes()).hexdigest() for x in (fit, hold)]


def test_no_eligible_retained_feature_skips_ranking_fit(monkeypatch):
    train = pd.DataFrame({"keep": [0., 1.], "neutral": [8., 3.]})
    monkeypatch.setattr(residual_ranking, "make_model", lambda **kw: pytest.fail("unexpected training"))
    rec, gain, meta = residual_ranking.rank_residual_recovery(
        train, [0., 1.], tuple(train), 1, {"keep": 1, "neutral": 0},
        task="regression", model_params={}, backend="cpu", seed=42)
    assert rec == {"neutral": 0} and gain == {"keep": 0}
    assert meta["fit_count"] == 0 and meta["skip_reason"]


def test_constant_residual_returns_zero_and_absent_multiclass_output_is_ignored():
    frame = np.array([[1.], [2.], [3.], [4.]])
    assert residual_ranking._recovery_correlations(frame, np.zeros((4, 1)))[0] == 0
    y = np.array([0, 1, 0, 1])
    probabilities = np.array([[.7,.2,.1], [.35,.25,.4], [.3,.2,.5], [.1,.1,.8]])
    matrix = residual_ranking._residuals(y, probabilities, "multiclass", 3)
    assert matrix.shape == (4, 3)
    np.testing.assert_array_equal(matrix[:, 2], 0.)
    np.testing.assert_allclose(matrix[:, :2], np.eye(3)[y][:, :2] - probabilities[:, :2])


def test_rejects_unordered_columns_out_of_range_mask_and_nonfinite_input():
    frame = pd.DataFrame({"a": [0., 1., 2.], "b": [1., 2., 3.]})
    policy = dict(task="regression", model_params={}, backend="cpu", seed=42)
    for features, mask in [(("b", "a"), 1), (("a", "b"), 0), (("a", "b"), 4)]:
        with pytest.raises(ValueError):
            residual_ranking.rank_residual_recovery(frame, [0,1,2], features, mask, {"a":0, "b":0}, **policy)
    with pytest.raises(ValueError, match="finite"):
        residual_ranking.rank_residual_recovery(frame.assign(a=np.nan), [0,1,2], tuple(frame), 3,
                                                {"a":0, "b":0}, **policy)
