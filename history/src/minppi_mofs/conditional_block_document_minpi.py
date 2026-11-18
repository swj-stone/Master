"""Experimental one-block document minPI exchange with isolated acceptance.

The caller must supply a fixed width. This prototype deliberately rejects
grouped or ordered rows and is not registered in the public selector API.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from minppi_mofs.document_vote_minpi import _presence
from minppi_mofs.fast_document_vote_minpi import _checked_csr, fast_document_vote_top_from_training
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _partition_rows(labels, seed):
    """Use a quarter, capped at 256 rows, only to accept the single proposal."""
    classes, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    calibration = min(256, int(np.ceil(n/4)))
    if (n < 8 or len(classes) < 2 or counts.min() < 2 or
            calibration < len(classes) or n-calibration < len(classes)):
        raise ValueError("Insufficient class observations for isolated calibration")
    proposal, held = train_test_split(np.arange(n), test_size=calibration,
                                      stratify=labels, random_state=seed)
    return np.sort(proposal), np.sort(held)


def _pilot_rows(labels, proposal, seed):
    if len(proposal) <= 512:
        return proposal.copy()
    pilot, _ = train_test_split(proposal, train_size=512, stratify=labels[proposal],
                               random_state=int(np.random.SeedSequence([93017, seed]).generate_state(1)[0]))
    return np.sort(pilot)


def _centered_prediction(X, columns, coefficient, means):
    return np.asarray(X[:, columns]@coefficient)-means[columns]@coefficient


def _local_refit(X, means, scales, columns, residual):
    """One small standardized unit-ridge solve; the rest of the predictor is frozen."""
    block = (X[:, columns].toarray()-means[columns])/scales[columns]
    gram = block.T@block
    gram.flat[::len(columns)+1] += 1.
    return np.linalg.solve(gram, block.T@residual)/scales[columns, None]


def _block_endpoints(target, common_prediction, old_prediction, new_prediction):
    difference = new_prediction-old_prediction
    zero = 2*np.sum(target*difference)+np.sum(old_prediction**2)-np.sum(new_prediction**2)
    full = zero-2*np.sum(common_prediction*difference)
    return np.array([zero, full])/len(target)


def _acceptance_gain(endpoints, peers, rule):
    if rule == "global":
        return float(np.min(endpoints))
    if rule != "local":
        raise ValueError("acceptance must be global or local")
    if peers == 0:
        return float(endpoints[0])
    adjacent = endpoints[0]+(peers-1)/peers*(endpoints[1]-endpoints[0])
    return float(min(adjacent, endpoints[1]))


def _acceptance_gate(gain, old_ba, new_ba):
    return bool(gain > 0 and new_ba >= old_ba-.02-1e-12)


def _propose_block(X, targets, support):
    """All rankings and fitted coefficients depend only on pilot/proposal rows."""
    n, p = X.shape
    term = np.bincount(X.indices, minlength=p).astype(float)
    means = term/n
    variance = means*(1-means)
    scales = np.sqrt(np.where(variance > 0, variance, 1.))
    prior = targets.mean(axis=0)
    centered = targets-prior
    model = Ridge(alpha=1., solver="lsqr", max_iter=8, tol=.001)
    model.fit(X[:, support].multiply(1/scales[support]).tocsr(), targets)
    coefficients = np.asarray(model.coef_).T/scales[support, None]
    prediction = _centered_prediction(X, support, coefficients, means)
    residual = centered-prediction
    square = n*variance
    # Conditional singleton fits against the current full support residual.
    rhs = X.T@residual-means[:, None]*residual.sum(axis=0)
    candidate_coefficients = rhs/(square+scales**2)[:, None]
    target_rhs = X.T@centered-means[:, None]*centered.sum(axis=0)
    norm = square*np.einsum("ij,ij->i", candidate_coefficients, candidate_coefficients)
    zero = 2*np.einsum("ij,ij->i", target_rhs, candidate_coefficients)-norm
    full = 2*np.einsum("ij,ij->i", rhs, candidate_coefficients)-norm
    incoming_scores = np.minimum(zero, full)
    available = np.ones(p, dtype=bool)
    available[support] = False
    available &= variance > 0
    pool = np.flatnonzero(available)
    block_size = min(8, len(support), len(pool))
    base = dict(means=means, scales=scales, target_mean=prior, teacher_coefficients=coefficients,
                teacher_iterations=np.asarray(model.n_iter_), initial_support=support.copy())
    if block_size == 0:
        return dict(base, proposed_support=support.copy(), removed=np.empty(0, dtype=int),
                    incoming=np.empty(0, dtype=int), partners=np.empty(0, dtype=int),
                    common=support.copy(), old_block=np.empty(0, dtype=int),
                    new_block=np.empty(0, dtype=int), replacement_coefficients=np.empty((0, targets.shape[1])))
    incoming = np.sort(pool[np.argsort(-incoming_scores[pool], kind="stable")[:block_size]])
    support_norm = square[support]*np.einsum("ij,ij->i", coefficients, coefficients)
    support_zero = 2*np.einsum("ij,ij->i", target_rhs[support], coefficients)-support_norm
    support_full = 2*np.einsum("ij,ij->i", rhs[support], coefficients)+support_norm
    weak = np.minimum(support_zero, support_full)
    removed_positions = np.argsort(weak, kind="stable")[:block_size]
    removed = np.sort(support[removed_positions])
    remaining = np.setdiff1d(support, removed, assume_unique=True)
    partners = np.empty(0, dtype=int)
    if len(remaining):
        cross = (X[:, remaining].T@X[:, removed]).toarray()-n*np.outer(means[remaining], means[removed])
        denominator = np.sqrt(np.outer(square[remaining], square[removed]))
        correlation = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
        partners = np.unique(remaining[np.argmax(np.abs(correlation), axis=0)])
    old_block = np.union1d(removed, partners)
    new_block = np.union1d(incoming, partners)
    common = np.setdiff1d(support, old_block, assume_unique=True)
    common_positions = np.searchsorted(support, common)
    common_prediction = _centered_prediction(X, common, coefficients[common_positions], means)
    replacement = _local_refit(X, means, scales, new_block, centered-common_prediction)
    proposed = np.union1d(remaining, incoming)
    assert len(proposed) == len(support)
    return dict(base, proposed_support=proposed, removed=removed, incoming=incoming, partners=partners,
                common=common, old_block=old_block, new_block=new_block,
                replacement_coefficients=replacement,
                proposal_incoming_scores=incoming_scores[incoming], proposal_removed_scores=weak[removed_positions])


def conditional_block_document_top_from_training(X, y, count=None, *, task="classification",
                                                random_state=1129, groups=None, ordered_groups=False,
                                                acceptance="global"):
    """Propose one <=8-column swap without calibration access, then accept once.

    At most 512 proposal rows fit the reference teacher and all swap coefficients.
    The disjoint calibration rows accept the frozen block only when its worst
    squared-loss gain over the declared peer-count window is strictly positive
    and its balanced accuracy loses at most .02 against the old teacher.
    Global protects k=0..m; local protects k=max(0,m-1)..m. These are internal
    surrogate checks, not guarantees after the eventual full-training refit.
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
    proposal, calibration = _partition_rows(y, random_state)
    pilot = _pilot_rows(y, proposal, random_state)
    support, _ = fast_document_vote_top_from_training(X[proposal], y[proposal], count, random_state=random_state)
    classes = np.unique(y[proposal])
    target = (y[pilot, None] == classes).astype(float)
    info = _propose_block(X[pilot], target, support)
    info.update(proposal_rows=proposal, pilot_rows=pilot, calibration_rows=calibration,
                acceptance=acceptance, classes=classes)
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
