"""Incremental ridge updates for minPI's two adjacent realized coalition sizes."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.piv2 import _folds
from minppi_mofs.utility_minpi import _utility_inputs


class _FoldState:
    """Residualized training/held-out quantities for one nested sequence of supports."""

    def __init__(self, fit, check, fit_target, check_target, max_size):
        self.fit = fit
        self.innovations = check.copy()
        self.residual = check_target.copy()
        self.rhs = fit.T @ fit_target
        self.diagonal = np.einsum("ij,ij->j", fit, fit) + 1.
        self.size = 0
        self.projections = None
        self.dual = None
        if max_size <= len(fit):
            self.projections = np.empty((max_size, fit.shape[1]))
        else:
            # P X with P=(I+X_S X_S.T)^{-1}. Storage stays O(n*p) for wide budgets.
            self.dual = fit.copy()

    def scores(self):
        update = self.rhs / self.diagonal[:, None]
        linear = np.einsum("ij,ij->i", self.innovations.T @ self.residual, update)
        quad = (np.einsum("ij,ij->j", self.innovations, self.innovations)
                * np.einsum("ij,ij->i", update, update))
        return 2 * linear - quad

    def append(self, columns):
        # Limit the size of every block-by-p temporary to the size of the input sketch.
        for start in range(0, len(columns), len(self.fit)):
            self._append_block(columns[start:start + len(self.fit)])

    def _append_block(self, block):
        if self.dual is None:
            cross = self.fit[:, block].T @ self.fit
            if self.size:
                past = self.projections[:self.size]
                cross -= past[:, block].T @ past
        else:
            cross = self.fit[:, block].T @ self.dual

        # Whitening the new block residualizes every remaining column with one update.
        if len(block) == 1:
            norm = np.sqrt(self.diagonal[block[0]])
            weights = cross / norm
            target = self.rhs[block] / norm
            held = self.innovations[:, block] / norm
            if self.dual is not None:
                fit_basis = self.dual[:, block] / norm
        else:
            gram = cross[:, block].copy()
            gram.flat[::len(block) + 1] += 1.
            lower = cholesky(gram, lower=True, check_finite=False)

            def whiten(matrix):
                return solve_triangular(lower, matrix, lower=True, check_finite=False)

            weights = whiten(cross)
            target = whiten(self.rhs[block])
            held = whiten(self.innovations[:, block].T).T
            if self.dual is not None:
                fit_basis = whiten(self.dual[:, block].T).T
        if self.dual is None:
            self.projections[self.size:self.size + len(block)] = weights
        else:
            self.dual -= fit_basis @ weights
        self.size += len(block)
        self.diagonal -= np.einsum("ij,ij->j", weights, weights)
        self.rhs -= weights.T @ target
        self.residual -= held @ target
        self.innovations -= held @ weights


def local_refit_fast_top_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """Use one incremental Schur state per fold to implement the local refit rule.

    The vote, sketch, ridge penalty, checkpoint schedule and preceding coalition
    match local_refit_top_from_training. Both implementations use double precision.
    """
    X, y = _utility_inputs(X, y, count, task)
    if tiers is None:
        sizes = [0]
        checkpoint = 1
        while checkpoint < count:
            sizes.append(checkpoint)
            checkpoint *= 2
    else:
        sizes = sorted(_selection_tiers(tiers, X.shape[1]))
        if sizes[0] != 0 or sizes[-1] >= count:
            raise ValueError("Local refit tiers must include zero and be smaller than the budget")
    if count == X.shape[1]:
        return np.arange(count), {"sizes": [], "ballot_rows": 0}
    folds = _folds(y, task, np.random.default_rng(random_state),
                   row_limit=min(1024, max(256, 2 * X.shape[1])))
    if folds is None or min(map(len, folds)) < 2:
        raise ValueError("Insufficient training observations for cross-fitting")
    classes = np.unique(y) if task == "classification" else None
    states = []
    for fit, held in (folds, folds[::-1]):
        training, check = _standardize(X[fit], X[held])
        if classes is None:
            y_fit, y_check = y[fit, None].astype(float), y[held, None].astype(float)
        else:
            y_fit, y_check = y[fit, None] == classes, y[held, None] == classes
        center = y_fit.mean(axis=0)
        states.append(_FoldState(training, check, y_fit - center, y_check - center, max(sizes)))

    available = np.ones(X.shape[1], dtype=bool)
    block = np.empty(0, dtype=int)
    current = sum(state.scores() for state in states)
    for size, next_size in zip(sizes, sizes[1:] + [count]):
        if size:
            if len(block) > 1:
                for state in states:
                    state.append(block[:-1])
                preceding = sum(state.scores() for state in states)
            else:
                preceding = current
            for state in states:
                state.append(block[-1:])
            current = sum(state.scores() for state in states)
            score = np.minimum(current, preceding)
        else:
            score = current
        pool = np.flatnonzero(available)
        block = pool[np.argsort(-score[pool], kind="stable")[:next_size - size]]
        available[block] = False
    return np.flatnonzero(~available), {"sizes": sizes, "ballot_rows": sum(map(len, folds))}
