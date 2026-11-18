import numpy as np

from minppi_mofs.bestcal_grouped_minpi import best_calibrated_support


def test_quality_priority_can_keep_more_than_first_acceptable_prefix():
    info = dict(ranking=np.array([5, 4, 3, 2, 1, 0]), candidate_widths=np.array([2, 4, 5]),
                calibration_scores=np.array([.79, .84, .84, .80]), tolerance=.02)
    cols, diagnostic = best_calibrated_support(6, info)
    np.testing.assert_array_equal(cols, [2, 3, 4, 5])
    assert diagnostic["selected_calibration_score"] == .84
    assert diagnostic["reduced"]


def test_quality_priority_still_rejects_every_prefix_below_full_tolerance():
    info = dict(ranking=np.arange(6), candidate_widths=np.array([2, 4]),
                calibration_scores=np.array([.6, .7, .9]), tolerance=.02)
    cols, diagnostic = best_calibrated_support(6, info)
    np.testing.assert_array_equal(cols, np.arange(6))
    assert not diagnostic["reduced"]
