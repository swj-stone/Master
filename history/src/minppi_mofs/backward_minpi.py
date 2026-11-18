"""Backward minPI for preserving a full ridge model across training environments."""
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from minppi_mofs.environment_minpi import InsufficientEnvironmentsError, _environment_splits
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.restoration_minpi import _sample_rows
from minppi_mofs.utility_minpi import _utility_inputs


class _Deletion:
    """All exact single-column ridge downdates for one support and environment."""

    def __init__(self, fit, check, target):
        self.fit = fit
        p = fit.shape[1]
        if p <= len(fit):
            gram = fit.T @ fit
            gram.flat[::p + 1] += 1.
            factor = cho_factor(gram, check_finite=False)
            # This inverse is bounded by the training sketch size squared.
            self.inverse = cho_solve(factor, np.eye(p), check_finite=False)
            self.dual_fit = None
            self.diag = np.diag(self.inverse)
            self.beta = self.inverse @ (fit.T @ target)
            self.innovation = check @ self.inverse
        else:
            gram = fit @ fit.T
            gram.flat[::len(gram) + 1] += 1.
            factor = cho_factor(gram, check_finite=False)
            self.inverse = None
            self.dual_fit = cho_solve(factor, fit, check_finite=False)
            self.diag = 1. - np.einsum("ij,ij->j", fit, self.dual_fit)
            self.beta = self.dual_fit.T @ target
            self.innovation = check - (check @ fit.T) @ self.dual_fit
        self.prediction = check @ self.beta

    def after_one(self, anchor):
        """Adjacent smaller support via one Schur update; ignore the anchor's slot."""
        if self.inverse is None:
            row = -(self.fit[:, anchor] @ self.dual_fit)
            row[anchor] += 1
        else:
            row = self.inverse[anchor]
        ratio = row / self.diag[anchor]
        innovation = self.innovation - self.innovation[:, anchor, None] * ratio
        diag = self.diag - row * ratio
        diag[anchor] = 1.
        beta = self.beta - ratio[:, None] * self.beta[anchor]
        pred = self.prediction - (self.innovation[:, anchor, None] / self.diag[anchor]) * self.beta[anchor]
        return pred, beta, diag, innovation


def _deletion_cost(teacher, pred, beta, diag, innovation):
    """Increase in squared deviation from the teacher when restoring j is disallowed."""
    # Full-minus-reduced prediction for j is innovation[:,j] * beta[j] / diag[j].
    update = beta / diag[:, None]
    linear = np.einsum("ij,ij->i", innovation.T @ (teacher - pred), update)
    quadratic = (np.einsum("ij,ij->j", innovation, innovation)
                 * np.einsum("ij,ij->i", update, update))
    return 2 * linear + quadratic


def _training_folds(X, y, task, groups, ordered_groups, seed):
    folds, audit = [], []
    classes = np.unique(y) if task == "classification" else None
    p = X.shape[1]
    for i, (train_pool, held_pool) in enumerate(_environment_splits(y, task, groups, ordered_groups, seed)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        labels = y if classes is not None else None
        ref = _sample_rows(train_pool, min(512, max(128, p)), labels, rng)
        held = _sample_rows(held_pool, min(256, max(64, p)), labels, rng)
        if len(ref) < 2 or len(held) < 2:
            continue
        fit, check = _standardize(np.asarray(X[ref], dtype=float), np.asarray(X[held], dtype=float))
        if classes is None:
            target = y[ref, None].astype(float)
            weight = np.full(len(held), 1 / len(held))
        else:
            target = (y[ref, None] == classes).astype(float)
            _, codes, counts = np.unique(y[held], return_counts=True, return_inverse=True)
            weight = 1 / (len(counts) * counts[codes])
        target -= target.mean(axis=0)
        # An additive intercept is identical for every support on standardized fitting rows.
        # Applying held-row weights to inputs also weights both sets of predictions.
        folds.append((fit, check * np.sqrt(weight)[:, None], target))
        audit.append({"fit_rows": ref, "held_rows": held})
    if len(folds) < 2:
        raise InsufficientEnvironmentsError("Need two usable environments for backward minPI")
    return folds, audit


def backward_path_from_training(X, y, counts, *, task="classification", random_state=1129,
                                groups=None, ordered_groups=False):
    """Return nested supports from a deterministic backward minPI elimination path.

    In every environment a full ridge teacher is fitted on its reference rows.
    A held row votes by the increase in its squared discrepancy from the teacher
    when removing a candidate and refitting ridge. Relative loss uses the held
    teacher's energy around the training intercept, weighting classes equally.
    When this energy is negligible at floating-point precision relative to the
    target energy, normalize by the latter (or one for a constant target).

    At each stage remove the least costly anchor by worst-environment loss;
    rank other removals by their worst cost across all environments both with
    and without that anchor. Remove at most half the live features per stage.
    This is the smallest worst deletion cost, equivalently maximizing the
    smallest signed fidelity vote for removal. The elimination order and ties
    are independent of the requested output budgets.
    """
    counts = tuple(counts)
    if not counts:
        raise ValueError("Provide at least one feature budget")
    X, y = _utility_inputs(X, y, counts[0], task)
    p = X.shape[1]
    if any(not isinstance(h, Integral) or isinstance(h, (bool, np.bool_)) or not 1 <= h <= p for h in counts):
        raise ValueError("Budgets must be integers in [1,p]")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral) or not 0 <= random_state < 2**32:
        raise ValueError("Use an integer seed in [0, 2**32)")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape or pd.isna(groups).any():
            raise ValueError("Groups must be complete and aligned with training rows")
    if ordered_groups and groups is None:
        raise ValueError("ordered_groups requires group identifiers")
    info = {"strategy": ("forward" if ordered_groups else "group") if groups is not None else "row",
            "folds": [], "checkpoints": [], "normalization": []}
    if min(counts) == p:
        return {p: np.arange(p)}, info
    folds, info["folds"] = _training_folds(X, y, task, groups, ordered_groups, int(random_state))
    live = np.arange(p)
    removed = []
    teachers, energies = [], []
    while len(live) > min(counts):
        states = [_Deletion(fit[:, live], check[:, live], target) for fit, check, target in folds]
        if not teachers:
            for state, (_, _, target) in zip(states, folds):
                teachers.append(state.prediction.copy())
                energy = float(np.square(state.prediction).sum())
                target_energy = float(np.square(target).sum() / len(target))
                if energy <= np.finfo(float).eps * target_energy:
                    energies.append(target_energy if target_energy > 0 else 1.)
                    info["normalization"].append("target_energy" if target_energy > 0 else "constant")
                else:
                    energies.append(energy)
                    info["normalization"].append("teacher_energy")
        costs = [_deletion_cost(t, s.prediction, s.beta, s.diag, s.innovation) / e
                 for t, s, e in zip(teachers, states, energies)]
        worst = np.maximum.reduce(costs)
        # On exact ties remove the larger original index first.
        anchor = int(np.lexsort((-live, worst))[0])
        step = min(max(1, len(live) // 2), len(live) - min(counts))
        positions = [anchor]
        if step > 1:
            for teacher, state, energy in zip(teachers, states, energies):
                adjacent = _deletion_cost(teacher, *state.after_one(anchor)) / energy
                worst = np.maximum(worst, adjacent)
            pool = np.flatnonzero(np.arange(len(live)) != anchor)
            order = np.lexsort((-live[pool], worst[pool]))
            positions.extend(pool[order[:step - 1]])
        info["checkpoints"].append(int(len(live)))
        removed.extend(live[positions])
        live = np.delete(live, positions)
    deletion = np.concatenate((np.asarray(removed, dtype=int), live))
    return {int(h): np.sort(deletion[p - h:]) for h in counts}, info


def backward_top_from_training(X, y, count, **kwargs):
    """Select a single width from the same backward path used for several widths."""
    supports, info = backward_path_from_training(X, y, (count,), **kwargs)
    return supports[count], info
