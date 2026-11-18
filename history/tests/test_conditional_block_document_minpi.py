from itertools import combinations

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from minppi_mofs import conditional_block_document_minpi as module


def test_block_endpoints_and_local_window_equal_enumerated_games():
    rng = np.random.default_rng(4097)
    for peers in (0, 1, 3, 5):
        target = rng.normal(size=(7, 3))
        common = rng.normal(size=(7, peers, 3))
        old, new = rng.normal(size=(2, 7, 3))
        actual = module._block_endpoints(target, common.sum(axis=1), old, new)
        expected = []
        for k in range(peers+1):
            gains = []
            for support in combinations(range(peers), k):
                pred = common[:, support].sum(axis=1)
                gains.append(((target-pred-old)**2-(target-pred-new)**2).sum()/len(target))
            expected.append(np.mean(gains))
        np.testing.assert_allclose(actual, [expected[0], expected[-1]], atol=2e-12)
        assert module._acceptance_gain(actual, peers, "global") == pytest.approx(min(expected))
        assert module._acceptance_gain(actual, peers, "local") == pytest.approx(min(expected[max(0, peers-1):]))
    assert module._acceptance_gain(np.array([-.01, 1.99]), 3, "global") < 0
    assert module._acceptance_gain(np.array([-.01, 1.99]), 3, "local") > 0


def test_small_block_refit_matches_direct_standardized_unit_ridge():
    rng = np.random.default_rng(6081)
    raw = (rng.random((73, 8)) < .25).astype(float)
    raw[:, 2] = raw[:, 1]
    raw[:, 7] = 0
    X = sparse.csr_matrix(raw)
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    scales[scales == 0] = 1
    chosen = np.array([1, 2, 4, 7])
    residual = rng.normal(size=(len(raw), 3))
    residual -= residual.mean(axis=0)
    coefficient = module._local_refit(X, means, scales, chosen, residual)
    Z = (raw[:, chosen]-means[chosen])/scales[chosen]
    model = Ridge(alpha=1, fit_intercept=False, solver="cholesky").fit(Z, residual)
    np.testing.assert_allclose(coefficient, model.coef_.T/scales[chosen, None], atol=1e-12)
    np.testing.assert_allclose((raw[:, chosen]-means[chosen])@coefficient, model.predict(Z), atol=1e-12)


def test_calibration_rows_cannot_change_support_proposal_or_fitted_coefficients(monkeypatch):
    rng = np.random.default_rng(7119)
    X = (rng.random((120, 31)) < .25).astype(float)
    y = (X[:, 1]+X[:, 4]+rng.random(120) > 1.1).astype(int)
    proposal, cal = np.arange(90), np.arange(90, 120)
    monkeypatch.setattr(module, "_partition_rows", lambda labels, seed: (proposal, cal))
    _, first = module.conditional_block_document_top_from_training(sparse.csr_matrix(X), y, 12)
    changed = X.copy()
    changed[cal] = 1-changed[cal]
    labels = y.copy()
    labels[cal] = 1-labels[cal]
    _, second = module.conditional_block_document_top_from_training(sparse.csr_matrix(changed), labels, 12)
    for name in ("initial_support", "proposed_support", "removed", "incoming", "partners", "common",
                 "means", "scales", "teacher_coefficients", "replacement_coefficients", "target_mean"):
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)
    assert not np.intersect1d(first["proposal_rows"], first["calibration_rows"]).size
    assert set(first["pilot_rows"]) <= set(first["proposal_rows"])


def test_same_proposal_for_two_acceptance_rules_and_presence_replay():
    rng = np.random.default_rng(9123)
    X = sparse.csr_matrix((rng.random((200, 37)) < .2).astype(float))
    y = (X[:, 2].toarray().ravel()+rng.random(200) > .8).astype(int)
    left, first = module.conditional_block_document_top_from_training(X, y, 13, acceptance="global")
    right, second = module.conditional_block_document_top_from_training(X*4, y, 13, acceptance="local")
    np.testing.assert_array_equal(first["initial_support"], second["initial_support"])
    np.testing.assert_array_equal(first["proposed_support"], second["proposed_support"])
    np.testing.assert_array_equal(first["calibration_endpoints"], second["calibration_endpoints"])
    for support, info in ((left, first), (right, second)):
        assert len(support) == 13 and np.all(support[1:] > support[:-1])
        expected = info["proposed_support"] if info["accepted"] else info["initial_support"]
        np.testing.assert_array_equal(support, expected)
        if info["accepted"]:
            assert info["calibration_endpoints"][-1] > 0


def test_stratified_partition_and_pilot_cap_are_disjoint():
    y = np.repeat(np.arange(3), [1000, 700, 300])
    proposal, calibration = module._partition_rows(y, 1129)
    pilot = module._pilot_rows(y, proposal, 1129)
    assert len(calibration) == 256 and len(pilot) == 512
    assert len(np.union1d(proposal, calibration)) == len(y)
    assert not np.intersect1d(proposal, calibration).size
    assert set(pilot) <= set(proposal)
    assert set(y[proposal]) == set(y[calibration]) == set(y[pilot]) == {0, 1, 2}


def test_positive_squared_loss_gain_cannot_override_calibration_ba_gate():
    old_prediction = np.array([[.51, .49], [.49, .51]])
    new_prediction = np.array([[1., 0.], [.51, .49]])
    truth = np.eye(2)
    gain = (np.square(truth-old_prediction)-np.square(truth-new_prediction)).sum()
    assert gain > 0
    assert not module._acceptance_gate(gain, 1., .5)
    assert module._acceptance_gate(gain, .8, .78)
    assert not module._acceptance_gate(gain, .8, .779)


@pytest.mark.parametrize("options", [dict(groups=np.arange(40)//4), dict(ordered_groups=True),
                                     dict(task="regression"), dict(acceptance="unknown"),
                                     dict(random_state=-1)])
def test_unsupported_group_and_task_inputs_are_rejected(options):
    X = sparse.csr_matrix(np.tile(np.eye(5), (8, 1)))
    with pytest.raises(ValueError):
        module.conditional_block_document_top_from_training(X, np.arange(40)%2, 3, **options)


def test_full_width_is_explicit_no_exchange_and_small_sample_rejected():
    X = sparse.csr_matrix(np.tile(np.eye(5), (8, 1)))
    cols, info = module.conditional_block_document_top_from_training(X, np.arange(40)%2, 5)
    np.testing.assert_array_equal(cols, np.arange(5))
    assert not info["accepted"] and info["reason"] == "full_width"
    with pytest.raises(ValueError, match="fixed"):
        module.conditional_block_document_top_from_training(X, np.arange(40)%2)
    with pytest.raises(ValueError):
        module.conditional_block_document_top_from_training(X[:5], np.arange(5)%2, 3)
