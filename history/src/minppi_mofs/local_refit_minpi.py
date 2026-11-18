"""MinPI using held-out ridge refit gains at two adjacent realized coalition sizes."""
import numpy as np

from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.piv2 import _folds
from minppi_mofs.refit_minpi import _refit_votes
from minppi_mofs.utility_minpi import _utility_inputs


def local_refit_top_from_training(X, y, count, *, task="classification", random_state=42, tiers=None):
    """At each block take the smaller gain for S and S with its last column removed.

    Column order is its order of admission, including the ranking within each
    block. Gain sums the held-out sample squared-loss reductions from refitting
    an augmented unit-penalty ridge model in each direction of a training split.
    The empty support uses one layer. The default block checkpoints are
    0, 1, 2, 4, ... below count. Custom checkpoints must include zero.
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
    data = []
    for ref, held in (folds, folds[::-1]):
        fit, check = _standardize(X[ref], X[held])
        if task == "classification":
            y_fit, y_held = y[ref, None] == classes, y[held, None] == classes
        else:
            y_fit, y_held = y[ref, None].astype(float), y[held, None].astype(float)
        center = y_fit.mean(axis=0)
        data.append((fit, check, y_fit - center, y_held - center))

    def gains(prefix):
        return sum(_refit_votes(*fold, prefix) for fold in data)

    selected = np.empty(0, dtype=int)
    available = np.ones(X.shape[1], dtype=bool)
    previous_size, previous_gain = -1, None
    for size, next_size in zip(sizes, sizes[1:] + [count]):
        current = gains(selected)
        score = current
        if size:
            preceding = previous_gain if previous_size == size - 1 else gains(selected[:-1])
            score = np.minimum(current, preceding)
        pool = np.flatnonzero(available)
        block = pool[np.argsort(-score[pool], kind="stable")[:next_size - size]]
        selected = np.concatenate((selected, block))
        available[block] = False
        previous_size, previous_gain = size, current
    return np.sort(selected), {"sizes": sizes, "ballot_rows": sum(map(len, folds))}
