import numpy as np
import pytest
from sklearn.feature_selection import f_classif, f_regression

from minppi_mofs.allocation_vote_minpi import _f_support, allocation_vote_top_from_training


@pytest.mark.parametrize("classification", (True, False))
def test_training_f_support_equals_standard_statistic(classification):
    rng = np.random.default_rng(211)
    x = rng.normal(size=(42, 51))
    x -= x.mean(axis=0)
    y = rng.normal(size=len(x))
    if classification:
        labels = np.digitize(y, [-.7, .2])
        target = (labels[:, None] == np.arange(4)).astype(float)
        expected = f_classif(x, labels)[0]
    else:
        labels, target = y, y[:, None]
        expected = f_regression(x, labels)[0]
    actual = _f_support(x, target-target.mean(axis=0), labels, 17, classification)
    np.testing.assert_array_equal(actual, np.sort(np.argsort(-expected)[:17]))


@pytest.mark.parametrize("ordered", (False, True))
def test_enforces_joint_loss_criterion_and_retains_spatial_coverage(ordered):
    rng = np.random.default_rng(201)
    blocks = rng.normal(size=(1200, 8))
    x = np.repeat(blocks, 8, axis=1) + rng.normal(size=(1200, 64)) * .18
    y = .7 * blocks[:, 2] + .2 * blocks[:, 5] + .3 * rng.normal(size=len(x))
    groups = np.repeat(np.arange(8), 150)
    cols, info = allocation_vote_top_from_training(x, y, task="regression", groups=groups,
                                                   ordered_groups=ordered)
    assert len(cols) <= 32 and len(info["coverage_representatives"]) == 4
    assert set(info["coverage_representatives"]).issubset(cols)
    passed = [item for item in info["checked"] if item["worst_excess"] <= .01]
    if passed:
        assert min(item["width"] for item in passed) == len(cols)
    else:
        assert min(item["worst_excess"] for item in info["checked"]) == info["worst_excess"]
    used = np.r_[info["folds"][0]["fit_rows"], info["folds"][0]["held_rows"]]
    for fold in info["calibration_folds"]:
        tr, te = fold["fit_rows"], fold["held_rows"]
        assert not np.intersect1d(np.r_[tr, te], used).size
        assert not np.intersect1d(groups[tr], groups[te]).size
        if ordered:
            assert groups[tr].max() < groups[te].min()
    assert len(info["f_reference_loss"]) == len(info["full_reference_loss"]) == len(info["calibration_folds"])
    np.testing.assert_array_equal(cols, allocation_vote_top_from_training(
        x, y, task="regression", groups=groups, ordered_groups=ordered)[0])


def test_fixed_count_and_no_independent_calibration_return_documented_support():
    x = np.arange(45.).reshape(3, 15)
    y = [0, 1, 1]
    cols, info = allocation_vote_top_from_training(x, y)
    assert len(cols) == 8 and info["allocation"] == "regional_no_calibration"
    fixed, other = allocation_vote_top_from_training(x, y, 9)
    assert len(fixed) == 9 and other["allocation"] == "regional_fixed"
