import itertools

import numpy as np
import pytest
from sklearn.metrics import r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from minppi_mofs.baselines._frozen_prediction import FrozenMaskPredictor, weighted_auc


@pytest.mark.parametrize("n_classes", [2, 4, 26])
def test_auc_matches_reference_bit_for_bit_with_ties_and_unequal_weights(n_classes):
    rng = np.random.default_rng(n_classes)
    labels = rng.permutation(np.arange(256) % n_classes)
    weights = rng.uniform(low=.001, high=7.1, size=len(labels))
    weights[::17] = 0.
    pred = rng.integers(1, 6, size=(len(labels), n_classes)).astype(float)
    pred /= pred.sum(axis=1, keepdims=True)
    expected = roc_auc_score(labels, pred[:, 1], sample_weight=weights) if n_classes == 2 else (
        roc_auc_score(labels, pred, multi_class="ovr", average="macro", sample_weight=weights))
    assert weighted_auc(labels, pred, weights) == expected


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_projected_masks_reproduce_full_predictor_and_utilities(task):
    rng = np.random.default_rng(42)
    x = np.zeros((180, 66), dtype=np.float32)
    x[:, [2, 15, 59]] = rng.normal(size=(len(x), 3))
    latent = (2*x[:, 15] + x[:, 59] - x[:, 2]).copy()
    x[::12, 15] = np.nan
    y = (latent > 0).astype(int) if task == "binary" else (
        np.digitize(latent, [-1,1]) if task == "multiclass" else latent)
    reg = task == "regression"
    estimator = XGBRegressor if reg else XGBClassifier
    model = estimator(n_estimators=14, max_depth=3, device="cpu", n_jobs=1).fit(x[:135],y[:135])
    active = tuple(sorted(int(f[1:]) for f in model.get_booster().get_score(importance_type="weight")))
    fill = np.nanmedian(x[:135],axis=0)
    x = x[135:]
    y = y[135:]
    weights = rng.uniform(.1,3,len(y))
    pred = FrozenMaskPredictor(model, x, fill, active, regression=reg)
    assert pred.physical_width == len(active) < x.shape[1]
    masks = [s for size in range(len(active)+1) for s in itertools.combinations(active,size)]
    expected = []
    for mask in masks:
        full = np.broadcast_to(fill,x.shape).copy()
        full[:, list(mask)] = x[:, list(mask)]
        p = (model.predict if reg else model.predict_proba)(full).astype(float)
        expected.append(p)
    idx = (0,-1)
    sample = [masks[i] for i in idx]
    np.testing.assert_array_equal(pred.predictions(sample), np.stack([expected[i] for i in idx]))
    scores = []
    for p in expected:
        if reg:
            scores.append(max(0., r2_score(y,p,sample_weight=weights)))
        elif task == "binary":
            scores.append(roc_auc_score(y,p[:,1],sample_weight=weights))
        else:
            scores.append(roc_auc_score(y,p,multi_class="ovr",average="macro",sample_weight=weights))
    np.testing.assert_array_equal(pred.utilities(masks,y,weights),scores)
    assert model.get_booster().num_features() == 66


def test_constant_forest_has_no_used_features():
    x = np.ones((48, 5), dtype=np.float32)
    y = np.tile([0,1], len(x)//2)
    model = XGBClassifier(n_estimators=2, device="cpu", n_jobs=1).fit(x,y)
    predictor = FrozenMaskPredictor(model, x, x[0], (), regression=False)
    assert predictor.utilities([()], y, np.ones(len(y))) == [.5]
