import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from minppi_mofs.band_vote_minpi import band_vote_top_from_training
from minppi_mofs.calibrated_band_minpi import _refine, calibrated_band_top_from_training


def _spectra(n):
    rng = np.random.default_rng(2026)
    latent = rng.normal(size=(n, 6))
    X = np.repeat(latent, 8, axis=1) + .035 * rng.normal(size=(n, 48))
    return X, latent[:, 1] + .2 * latent[:, 3]


@pytest.mark.parametrize("ordered,task", [(True, "regression"), (False, "classification")])
def test_calibration_uses_disjoint_rows_and_refits_both_models(ordered, task):
    x, y = _spectra(768)
    if task == "classification":
        y = np.digitize(y, [-.3, .3])
    groups = np.repeat(np.arange(6), 128)
    options = dict(task=task, groups=groups, ordered_groups=ordered, backend="numpy")
    cols, info = calibrated_band_top_from_training(x, y, **options)
    coarse, original = band_vote_top_from_training(x, y, **options)
    starts, stops, support = original["band_starts"], original["band_stops"], coarse
    scale = np.abs(original["scores"]).max()
    ranking = np.round(original["scores"] / scale, decimals=12)
    vote_rows = np.r_[original["folds"][0]["fit_rows"], original["folds"][0]["held_rows"]]
    expected_width = x.shape[1]
    seen_widths = []
    for size in (8, 4, 2):
        starts, stops, refined = _refine(starts, stops, support, ranking, size)
        assert set(support).issubset(refined)
        support = refined
        if len(support) in seen_widths:
            continue
        seen_widths.append(len(support))
        diffs = []
        for di, fold in enumerate(info["calibration_folds"]):
            tr, te = fold["fit_rows"], fold["held_rows"]
            assert not np.intersect1d(np.r_[tr, te], vote_rows).size
            assert not np.intersect1d(groups[tr], groups[te]).size
            if ordered:
                assert groups[tr].max() < groups[te].min() == groups.max()
            target = np.eye(3)[y] if task == "classification" else y[:, None]
            scaler = StandardScaler().fit(x[tr])
            a, b = scaler.transform(x[tr]), scaler.transform(x[te])
            losses = []
            for selected in (np.arange(x.shape[1]), support):
                reg = Ridge(alpha=len(tr)/len(y), solver="svd").fit(a[:, selected], target[tr])
                pred = reg.predict(b[:, selected]).reshape(len(te), -1)
                losses.append(np.square(pred - target[te]).sum())
            delta = losses[1] - losses[0]
            assert info["excess_loss"][len(seen_widths)-1][di] == pytest.approx(delta, abs=1e-9)
            assert info["allowed_excess"][di] == pytest.approx(.01 * np.sum(np.var(target[te], axis=0)) * len(te))
            diffs.append(delta)
        if np.all(np.asarray(diffs) <= info["allowed_excess"]):
            expected_width = len(support)
            break
    assert len(cols) == expected_width


def test_no_disjoint_calibration_returns_full_and_fixed_override_matches_original():
    x, y = _spectra(80)
    cols, info = calibrated_band_top_from_training(x, y, task="regression", backend="numpy")
    np.testing.assert_array_equal(cols, np.arange(x.shape[1]))
    assert info["checked_widths"] == [] and info["calibration_folds"] == []
    fixed, _ = calibrated_band_top_from_training(x, y, 5, task="regression", backend="numpy")
    expected, _ = band_vote_top_from_training(x, y, 5, task="regression", backend="numpy")
    np.testing.assert_array_equal(fixed, expected)
