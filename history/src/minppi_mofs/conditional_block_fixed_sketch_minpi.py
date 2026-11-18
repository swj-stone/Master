"""Fixed-original-sketch control for isolated one-block document minPI exchange."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from minppi_mofs.conditional_block_document_minpi import (
    _acceptance_gain, _acceptance_gate, _block_endpoints, _centered_prediction, _propose_block,
)
from minppi_mofs.document_vote_minpi import _counts, _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr, fast_document_vote_top_from_training
from minppi_mofs.signal_vote_minpi import _row_dot
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _fixed_vote_ranking(X, y, seed):
    """Keep old vote endpoints; exclude all remaining rows from tie strength."""
    classes, labels = np.unique(y, return_inverse=True)
    if len(y) <= 512 or len(classes) < 2:
        raise ValueError("Independent calibration requires more than 512 rows and two classes")
    _, original = fast_document_vote_top_from_training(X, y, 1, random_state=seed)
    fold = original["folds"][0]
    voting = np.sort(np.concatenate((fold["fit_rows"], fold["held_rows"])))
    assert len(voting) == 512 and len(np.unique(voting)) == 512
    remaining = np.setdiff1d(np.arange(len(y)), voting, assume_unique=True)
    rng = np.random.default_rng(np.random.SeedSequence([93019, seed]))
    calibration = np.sort(rng.choice(remaining, min(256, len(remaining)), replace=False))
    stats = _counts(X[voting], labels[voting], len(classes))
    if (stats.classes == 0).any() or len(np.unique(y[calibration])) != len(classes):
        raise ValueError("Every class must occur in voting and calibration rows")
    scores = original["scores"]
    scale = np.max(np.abs(scores))
    normalized = np.round(scores/scale, 12) if scale > 0 else scores
    marginal = _row_dot(stats.rhs, stats.rhs)/(stats.square+1.)
    rank = np.lexsort((np.arange(X.shape[1]), -marginal, -normalized))
    return rank, voting, calibration, classes, dict(original, voting_marginal=marginal)


def fixed_sketch_block_document_top_from_training(X, y, count=None, *, task="classification",
                                                random_state=1129, groups=None, ordered_groups=False,
                                                acceptance="global"):
    """Use original 512 vote rows for all proposal fitting and separate remaining rows.

    The supplied width is fixed. Only tied ranks change relative to old minPI:
    marginal strength is computed on voting rows, not calibration rows. The
    block construction, <=8 exchanges, <=16-column refit and global/local plus
    BA acceptance rules are imported unchanged from the first prototype.
    """
    if task != "classification":
        raise ValueError("This prototype requires classification labels")
    if groups is not None or ordered_groups:
        raise ValueError("This prototype supports only ungrouped unordered rows")
    if acceptance not in ("global", "local"):
        raise ValueError("acceptance must be global or local")
    if count is None:
        raise ValueError("This attribution prototype requires a fixed count")
    X = _presence(_checked_csr(X))
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, count, task)
    _validate_sampling(y, random_state, None, False)
    if count == X.shape[1]:
        support = np.arange(X.shape[1])
        return support, dict(accepted=False, reason="full_width", initial_support=support,
                             proposed_support=support.copy(), acceptance=acceptance)
    rank, voting, calibration, classes, original = _fixed_vote_ranking(X, y, random_state)
    support = np.sort(rank[:count])
    target = (y[voting, None] == classes).astype(float)
    info = _propose_block(X[voting], target, support)
    info.update(proposal_rows=voting, pilot_rows=voting.copy(), calibration_rows=calibration,
                acceptance=acceptance, classes=classes, rank=rank,
                scores=original["scores"], zero_peer_scores=original["zero_peer_scores"],
                one_peer_scores=original["one_peer_scores"], voting_marginal=original["voting_marginal"],
                original_fold_fit=original["folds"][0]["fit_rows"],
                original_fold_held=original["folds"][0]["held_rows"])
    if not len(info["incoming"]):
        return support, dict(info, accepted=False, reason="no_varying_challenger",
                             calibration_endpoints=np.zeros(2), global_gain=0., local_gain=0.,
                             loss_accepted=False, ba_gate=True, old_teacher_ba=np.nan, new_teacher_ba=np.nan)
    held = X[calibration]
    target = (y[calibration, None] == classes).astype(float)-info["target_mean"]
    old_positions = np.searchsorted(support, info["old_block"])
    common_positions = np.searchsorted(support, info["common"])
    common = _centered_prediction(held, info["common"], info["teacher_coefficients"][common_positions], info["means"])
    old_block = _centered_prediction(held, info["old_block"], info["teacher_coefficients"][old_positions], info["means"])
    new_block = _centered_prediction(held, info["new_block"], info["replacement_coefficients"], info["means"])
    endpoints = _block_endpoints(target, common, old_block, new_block)
    old_pred = classes[np.argmax(info["target_mean"]+common+old_block, axis=1)]
    new_pred = classes[np.argmax(info["target_mean"]+common+new_block, axis=1)]
    old_ba = balanced_accuracy_score(y[calibration], old_pred)
    new_ba = balanced_accuracy_score(y[calibration], new_pred)
    global_gain = _acceptance_gain(endpoints, len(info["common"]), "global")
    local_gain = _acceptance_gain(endpoints, len(info["common"]), "local")
    gain = global_gain if acceptance == "global" else local_gain
    accepted = _acceptance_gate(gain, old_ba, new_ba)
    info.update(calibration_endpoints=endpoints, global_gain=global_gain, local_gain=local_gain,
                accepted=accepted, loss_accepted=bool(gain > 0), ba_gate=bool(new_ba >= old_ba-.02-1e-12),
                old_teacher_ba=old_ba, new_teacher_ba=new_ba,
                old_teacher_loss=float(np.square(target-common-old_block).sum()/len(target)),
                new_teacher_loss=float(np.square(target-common-new_block).sum()/len(target)),
                calibration_old_prediction=old_pred, calibration_new_prediction=new_pred,
                reason="accepted" if accepted else "rejected")
    return info["proposed_support"] if accepted else support, info
