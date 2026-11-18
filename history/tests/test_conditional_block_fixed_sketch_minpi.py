import numpy as np
import pytest
from scipy import sparse

from minppi_mofs import conditional_block_fixed_sketch_minpi as module
from minppi_mofs.document_vote_minpi import _counts
from minppi_mofs.fast_document_vote_minpi import fast_document_vote_top_from_training
from minppi_mofs.signal_vote_minpi import _row_dot


def data():
    rng = np.random.default_rng(310977)
    X = (rng.random((840, 43)) < .2).astype(float)
    y = (X[:, 1]+.5*X[:, 5]+rng.random(len(X)) > .85).astype(int)
    return sparse.csr_matrix(X), y


def test_fixed_sketch_retains_original_endpoints_and_uses_only_voting_ties():
    X, y = data()
    _, old = fast_document_vote_top_from_training(X, y, 1, random_state=1129)
    _, info = module.fixed_sketch_block_document_top_from_training(X, y, 15, random_state=1129)
    voting = np.sort(np.concatenate((old["folds"][0]["fit_rows"], old["folds"][0]["held_rows"])))
    np.testing.assert_array_equal(voting, info["proposal_rows"])
    np.testing.assert_array_equal(voting, info["pilot_rows"])
    for field in ("scores", "zero_peer_scores", "one_peer_scores"):
        np.testing.assert_array_equal(old[field], info[field])
    stats = _counts(X[voting], y[voting], 2)
    marginal = _row_dot(stats.rhs, stats.rhs)/(stats.square+1)
    scores = old["scores"]
    scale = np.max(np.abs(scores))
    normalized = np.round(scores/scale, 12) if scale > 0 else scores
    rank = np.lexsort((np.arange(X.shape[1]), -marginal, -normalized))
    np.testing.assert_array_equal(rank, info["rank"])
    np.testing.assert_array_equal(np.sort(rank[:15]), info["initial_support"])
    assert len(voting) == 512 and len(info["calibration_rows"]) == 256
    assert not np.intersect1d(voting, info["calibration_rows"]).size


def test_all_nonvoting_data_and_labels_cannot_change_proposal():
    X, y = data()
    _, original = module.fixed_sketch_block_document_top_from_training(X, y, 15)
    other = np.setdiff1d(np.arange(len(y)), original["proposal_rows"])
    altered = X.toarray()
    altered[other] = 1-altered[other]
    changed_y = y.copy()
    changed_y[other] = 1-changed_y[other]
    _, changed = module.fixed_sketch_block_document_top_from_training(sparse.csr_matrix(altered), changed_y, 15)
    for field in ("rank", "scores", "initial_support", "proposed_support", "removed", "incoming", "partners",
                  "means", "scales", "teacher_coefficients", "replacement_coefficients", "target_mean",
                  "proposal_rows", "pilot_rows", "calibration_rows"):
        np.testing.assert_array_equal(original[field], changed[field], err_msg=field)


def test_two_rules_share_exact_proposal_and_calibration_endpoints():
    X, y = data()
    _, first = module.fixed_sketch_block_document_top_from_training(X, y, 15, acceptance="global")
    _, second = module.fixed_sketch_block_document_top_from_training(X, y, 15, acceptance="local")
    for field in ("proposed_support", "replacement_coefficients", "calibration_endpoints",
                  "calibration_old_prediction", "calibration_new_prediction"):
        np.testing.assert_array_equal(first[field], second[field])
    np.testing.assert_allclose(first["old_teacher_loss"]-first["new_teacher_loss"],
                               first["calibration_endpoints"][-1], atol=1e-12)


def test_no_independent_rows_and_group_inputs_are_rejected():
    X, y = data()
    with pytest.raises(ValueError, match="512"):
        module.fixed_sketch_block_document_top_from_training(X[:512], y[:512], 15)
    with pytest.raises(ValueError, match="group"):
        module.fixed_sketch_block_document_top_from_training(X, y, 15, groups=np.arange(len(y)))
