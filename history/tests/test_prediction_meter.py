import numpy as np
import pytest

from minppi_mofs.prediction_meter import PredictionMeter, minpi_sample_counts


class FixedModel:
    def get_booster(self):
        return self

    def predict(self, x, *, offset=0):
        return x.sum(axis=1) + offset

    def predict_proba(self, x):
        return np.broadcast_to([0.25, 0.25, 0.5], (x.shape[0], 3))


def test_prediction_meter_counts_input_rows_by_phase_and_preserves_model_interface():
    model = FixedModel()
    meter = PredictionMeter(model, phase="estimate")
    x = np.arange(12).reshape(4, 3)
    assert meter.get_booster() is model
    np.testing.assert_array_equal(meter.predict_proba(x), model.predict_proba(x))
    np.testing.assert_array_equal(meter.predict(x[:1], offset=3), model.predict(x[:1], offset=3))
    meter.set_phase("sizing")
    meter.predict_proba(x[:2])
    assert meter.counts == {"estimate": {"calls": 2, "rows": 5}, "sizing": {"calls": 1, "rows": 2}}
    ledger = meter.counts
    ledger["estimate"]["rows"] = 42
    assert meter.counts["estimate"]["rows"] == 5


def test_minpi_counts_separate_uncached_forwards_and_match_measured_total():
    evidence = dict(masked_proposals=[{}, {}], ranking_rows=24, witness_predictions=5,
                    witness_rows=list(range(12)), bootstraps=3)
    assert minpi_sample_counts(evidence, {"calls": 8, "rows": 132}) == {
        "prefix": {"calls": 3, "rows": 72}, "voting": {"calls": 5, "rows": 60}
    }
    with pytest.raises(ValueError, match="measured"):
        minpi_sample_counts(evidence, {"calls": 8, "rows": 130})
    # A one-column Gain anchor returns before a voting sample is drawn.
    assert minpi_sample_counts(dict(masked_proposals=[], ranking_rows=24,
                                        witness_predictions=0), {"calls": 1, "rows": 24})["voting"] == {
        "calls": 0, "rows": 0
    }
