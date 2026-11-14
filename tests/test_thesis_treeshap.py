import numpy as np
import pandas as pd
import pytest

from minppi_mofs.baselines import shap_top_k
from minppi_mofs.baselines.shap_top_k import fitted_tree_shap_importance


def test_fitted_multiclass_tree_shap_uses_absolute_contributions_excluding_bias():
    class Booster:
        def predict(self, matrix, *, pred_contribs, approx_contribs, strict_shape):
            assert pred_contribs and strict_shape and not approx_contribs
            assert matrix.num_row() == 4 and list(matrix.feature_names) == ["f0", "f1"]
            values = np.array([[-2, 1, 1000], [2, -5, 2000]], dtype=float)
            return np.broadcast_to(values, (4, 2, 3))

    class FittedModel:
        def get_booster(self):
            return Booster()

    frame = pd.DataFrame(np.arange(18).reshape(9, 2), columns=["f0", "f1"])
    np.testing.assert_array_equal(fitted_tree_shap_importance(FittedModel(), frame, seed=42, max_rows=4), [2, 3])


def test_public_entry_limits_only_explanation_rows(monkeypatch):
    frame = pd.DataFrame({"x[0]": np.arange(1100, dtype=float)})
    labels = np.arange(len(frame)) % 2
    seen = []

    class Model:
        def fit(self, x, y):
            assert list(x.columns) == ["f0"]
            np.testing.assert_array_equal(x.index, frame.index)
            np.testing.assert_array_equal(y, labels)

        def get_booster(self):
            return self

        def predict(self, matrix, *, pred_contribs, approx_contribs, strict_shape):
            assert pred_contribs and strict_shape and not approx_contribs
            rows = matrix.get_data().toarray().ravel()
            seen.append(rows)
            return np.column_stack((rows, np.zeros(len(rows))))[:, None, :]

    monkeypatch.setattr(shap_top_k, "make_model", lambda **kwargs: Model())
    for options in ({}, {"max_rows": 7}, {"max_rows": None}):
        scores = shap_top_k.compute_shap_topk_importance(frame, labels, seed=42, **options)
        assert scores == {"x[0]": seen[-1].mean()}
    for positions, count in zip(seen[:2], (1024, 7), strict=True):
        expected = np.sort(np.random.default_rng(42).choice(len(frame), count, replace=False))
        np.testing.assert_array_equal(positions, expected)
    np.testing.assert_array_equal(seen[2], frame.index)


@pytest.mark.parametrize("limit", [0, -1, False, 1.5])
def test_invalid_limit_rejected_before_expensive_fit(monkeypatch, limit):
    monkeypatch.setattr(shap_top_k, "make_model", lambda **kwargs: pytest.fail("unexpected fit"))
    with pytest.raises(ValueError, match="max_rows"):
        shap_top_k.compute_shap_topk_importance(
            pd.DataFrame({"x": [0., 1.]}), np.array([0, 1]), max_rows=limit)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_native_scores_match_official_treeshap_including_additivity(monkeypatch, task):
    import shap

    rng = np.random.default_rng(124)
    frame = pd.DataFrame(rng.normal(size=(65, 4)), columns=list("abcd"))
    signal = frame.a.to_numpy() + frame.b.to_numpy() / 3
    labels = signal if task == "regression" else np.digitize(signal, [-.25, .25] if task == "multiclass" else [0.])
    new_model = shap_top_k.make_model
    fitted = []
    def capture(**kwargs):
        model = new_model(**kwargs)
        fitted.append(model)
        return model

    monkeypatch.setattr(shap_top_k, "make_model", capture)
    monkeypatch.setattr(shap_top_k, "_MAX_CONTRIBUTIONS_PER_BATCH", 78)
    scores = shap_top_k.compute_shap_topk_importance(
        frame, labels, task=task, seed=11, max_rows=29,
        model_params={"max_depth": 2, "n_estimators": 4, "n_jobs": 1})
    assert len(fitted) == 1
    positions = np.sort(np.random.default_rng(11).choice(len(frame), 29, replace=False))
    sample = frame.iloc[positions].set_axis([f"f{i}" for i in range(frame.shape[1])], axis=1)
    official = shap.TreeExplainer(fitted[0]).shap_values(sample, check_additivity=True)
    official = official.reshape(len(sample), frame.shape[1], -1)
    expected = np.abs(official).mean(axis=(0, 2), dtype=np.float64)
    np.testing.assert_allclose(list(scores.values()), expected, rtol=1e-6, atol=1e-8)
