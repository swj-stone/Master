"""Synthetic verification of the new-archive adaptor. No study endpoint is read."""
import numpy as np
import pytest

from .direct_gain import run_direct_gain


@pytest.mark.parametrize("classes", [0, 2, 3])
def test_native_complete_width_curve_and_blind_predict(classes):
    generator = np.random.default_rng(42)
    x = generator.normal(size=(96, 4)).astype("float32")
    x[::4, 1] = np.nan
    y = np.arange(len(x)) % classes if classes else .9*x[:, 2] + .1*x[:, 0]
    task = "classification" if classes else "regression"
    fit, vote, width, test = np.split(np.arange(len(x)), [48, 64, 80])
    y[test] = -1 if classes else np.nan
    idx = dict(fit=fit, vote=vote, width=width, test=test)
    result, predictions = run_direct_gain(x, y, task, idx, 42, device="cpu")
    curve = np.array(result["width_curve"])
    order = result["gain_order"]
    assert len(curve) == x.shape[1] and result["fits"] == x.shape[1]+1
    assert result["near_width"] == np.flatnonzero(curve >= curve.max()-.005-1e-14)[0]+1
    assert result["near_width"] <= result["retained"]
    assert result["support"] == sorted(order[:result["retained"]])
    assert result["validation_utility"] == pytest.approx(curve.max(), abs=1e-14)
    assert predictions.shape == ((len(test), classes) if classes else (len(test),))
    assert np.isfinite(predictions).all()
