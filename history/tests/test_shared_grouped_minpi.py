import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from minppi_mofs import shared_grouped_minpi as shared
from minppi_mofs.certified_minpi import top_from_votes


@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize("shape", [(45, 13), (21, 47)])
def test_incremental_dual_predictions_match_direct_unit_ridge(classes, shape):
    rng = np.random.default_rng(17)
    X = rng.normal(size=shape) + rng.normal(size=shape[1]) * 3
    X[:, -1] = 4.25
    Xt = rng.normal(size=(19, shape[1]))
    y = np.arange(shape[0]) % classes
    y = np.asarray([f"class-{label}" for label in y])
    order = rng.permutation(shape[1])
    widths = sorted({1, shape[1] // 2, shape[1]})
    decisions, predictions = shared._prefix_ridge(X, y, Xt, order, widths)
    for width, decision, prediction in zip(widths, decisions, predictions):
        cols = order[:width]
        scaler = StandardScaler().fit(X[:, cols])
        model = RidgeClassifier(alpha=1., solver="cholesky").fit(scaler.transform(X[:, cols]), y)
        expected = model.decision_function(scaler.transform(Xt[:, cols]))
        np.testing.assert_allclose(decision, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(prediction, model.predict(scaler.transform(Xt[:, cols])))


def test_one_ranking_matches_every_float_pivotal_prefix():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(180, 43))
    X[:, -2:] = X[:, :2]
    y = np.arange(len(X)) % 3
    order, info = shared._grouped_ranking(X, y, 1129, voting=True)
    reps, votes = info["representatives"], info["votes"]
    assert len(votes) <= 256
    assert np.array_equal(reps, np.sort(reps))
    for width in (1, len(reps) // 2, len(reps)):
        expected = reps[top_from_votes(votes, width, prune=False)[0]]
        np.testing.assert_array_equal(np.sort(order[:width]), expected)


def test_calibration_rows_never_enter_representatives_or_votes(monkeypatch):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(120, 25))
    y = np.arange(len(X)) % 3
    fit, held = np.arange(84), np.arange(84, 120)
    monkeypatch.setattr(shared, "_pilot_rows", lambda *args, **kwargs: (fit, held))
    _, a = shared.shared_grouped_top_from_training(X, y, random_state=19)
    changed_x, changed_y = X.copy(), y.copy()
    changed_x[held] = rng.normal(size=changed_x[held].shape) * 100
    changed_y[held] = rng.permutation(changed_y[held])
    _, b = shared.shared_grouped_top_from_training(changed_x, changed_y, random_state=19)
    for field in ("ranking", "representatives", "votes", "pivotal_scores"):
        np.testing.assert_array_equal(a[field], b[field])


def test_pilot_caps_class_coverage_and_row_isolation():
    y = np.arange(2200) % 11
    a, b = shared._pilot_rows(y, 1129)
    assert len(a) == 512 and len(b) == 256
    assert not np.intersect1d(a, b).size
    assert np.array_equal(np.unique(y[a]), np.unique(y))
    assert np.array_equal(np.unique(y[b]), np.unique(y))
    a2, b2 = shared._pilot_rows(y, 1129)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)


def test_candidate_grid_uses_original_width_and_never_counts_full():
    assert shared._candidate_widths(17, 11) == [3, 5, 7, 9, 11]
    assert shared._candidate_widths(8, 8) == list(range(1, 8))


def test_no_vote_has_identical_pool_and_full_returns_failure_when_unavailable():
    rng = np.random.default_rng(43)
    X = rng.normal(size=(120, 21))
    y = np.arange(len(X)) % 3
    _, a = shared._grouped_ranking(X, y, 1129, voting=True)
    _, b = shared._grouped_ranking(X, y, 1129, voting=False)
    np.testing.assert_array_equal(a["representatives"], b["representatives"])
    cols, info = shared.shared_grouped_top_from_training(X[:3], y[:3])
    np.testing.assert_array_equal(cols, np.arange(X.shape[1]))
    assert not info["reduced"] and info["reason"] == "insufficient_class_rows"


def test_grouped_inputs_are_rejected_until_group_split_is_implemented():
    with pytest.raises(NotImplementedError, match="group"):
        shared.shared_grouped_top_from_training(np.ones((12, 4)), np.arange(12) % 2,
                                               groups=np.arange(12) // 2)


def test_narrow_pilots_factor_only_prefix_feature_matrices(monkeypatch):
    rng = np.random.default_rng(71)
    X, Xt = rng.normal(size=(90, 14)), rng.normal(size=(20, 14))
    y = np.arange(90) % 3
    factor_shapes = []
    factor = shared.cho_factor

    def observed(matrix, **kwargs):
        factor_shapes.append(matrix.shape)
        return factor(matrix, **kwargs)

    monkeypatch.setattr(shared, "cho_factor", observed)
    shared._prefix_ridge(X, y, Xt, np.arange(14), [2, 7, 14])
    assert factor_shapes == [(2, 2), (7, 7), (14, 14)]
