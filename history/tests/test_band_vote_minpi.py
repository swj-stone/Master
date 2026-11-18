import numpy as np
import pytest

from minppi_mofs.band_vote_minpi import _partition, band_center_from_training, band_vote_top_from_training
from minppi_mofs.native_pivotal_minpi import library_path


def test_partition_splits_at_discontinuity_and_missing_channel():
    rng = np.random.default_rng(591)
    signals = rng.normal(size=(128, 4))
    x = np.column_stack((np.repeat(signals[:, :1], 8, axis=1),
                         np.repeat(signals[:, 1:2], 4, axis=1),
                         np.repeat(signals[:, 2:3], 2, axis=1), signals[:, 2:]))
    x[:, 14] = np.nan
    starts, stops = _partition(x, np.arange(128))
    np.testing.assert_array_equal(starts, [0, 8, 12, 14, 15])
    np.testing.assert_array_equal(stops, [8, 12, 14, 15, 16])


@pytest.mark.parametrize("task,ordered", [("classification", True), ("regression", False)])
def test_voted_and_center_share_bins_and_fit_rows(task, ordered):
    rng = np.random.default_rng(793)
    latent = rng.normal(size=(380, 6))
    X = np.ascontiguousarray(np.repeat(latent, 4, axis=1) + .05 * rng.normal(size=(380, 24)))
    y = latent[:, 3] + .4 * latent[:, 4]
    if task == "classification":
        y = np.digitize(y, [-.6, .6])
    g = np.repeat(np.arange(10), 38)
    settings = dict(task=task, groups=g, ordered_groups=ordered, random_state=71)
    voted, info = band_vote_top_from_training(X, y, backend="numpy", **settings)
    centers, geom = band_center_from_training(X, y, **settings)
    starts, stops = info["band_starts"], info["band_stops"]
    np.testing.assert_array_equal(starts, geom["band_starts"])
    np.testing.assert_array_equal(stops, geom["band_stops"])
    assert len(voted) == len(centers) == 6
    for fold in info["folds"]:
        assert not np.intersect1d(g[fold["fit_rows"]], g[fold["held_rows"]]).size
        assert max(len(fold["fit_rows"]), len(fold["held_rows"])) <= 48
        if ordered:
            assert g[fold["fit_rows"]].max() < g[fold["held_rows"]].min()
    for j, a, b in zip(voted, starts, stops):
        assert a <= j < b
        assert info["scores"][j] == pytest.approx(max(info["scores"][a:b]))
    np.testing.assert_allclose(info["scores"], np.minimum(info["zero_peer_scores"], info["one_peer_scores"]))
    if library_path().is_file():
        actual, other = band_vote_top_from_training(X, y, backend="native", **settings)
        np.testing.assert_array_equal(actual, voted)
        np.testing.assert_allclose(info["scores"], other["scores"], rtol=1e-10, atol=1e-12)


def test_fixed_width_ties_and_preserve_full_on_unrelated_channels():
    rng = np.random.default_rng(85)
    signal = rng.normal(size=(150, 2))
    X = np.repeat(signal, 8, axis=1)
    y = np.ones(len(X))
    auto, _ = band_vote_top_from_training(X, y)
    np.testing.assert_array_equal(auto, [3, 11])
    smaller, _ = band_vote_top_from_training(X, y, 1)
    larger, _ = band_vote_top_from_training(X, y, 3)
    assert set(smaller).issubset(auto) and set(auto).issubset(larger)
    other, _ = band_vote_top_from_training(rng.normal(size=(150, 17)), y)
    np.testing.assert_array_equal(other, np.arange(17))
