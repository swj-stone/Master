"""Select features by their weakest normalized ridge gain across training environments."""
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.local_refit_fast import _FoldState
from minppi_mofs.masked_loss_minpi import _standardize
from minppi_mofs.restoration_minpi import _sample_rows
from minppi_mofs.utility_minpi import _utility_inputs


class InsufficientEnvironmentsError(ValueError):
    """Training rows do not support a comparison across two held environments."""


def _environment_splits(y, task, groups, ordered_groups, seed):
    rows = np.arange(len(y))
    if groups is not None:
        unique, codes = np.unique(groups, return_inverse=True)
        if ordered_groups:
            # Each of the latest three discovery groups is scored strictly forward.
            return [(rows[codes < i], rows[codes == i]) for i in range(max(1, len(unique) - 3), len(unique))]
        n = min(3, len(unique))
        return list(GroupKFold(n).split(rows, y, codes)) if n >= 2 else []
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        n = min(3, min(counts)) if len(counts) > 1 else 0
        return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(rows, y)) if n >= 2 else []
    n = min(3, len(y) // 2)
    return list(KFold(n, shuffle=True, random_state=seed).split(rows)) if n >= 2 else []


def _environment_states(X, y, task, groups, ordered_groups, seed, max_size):
    classes = np.unique(y) if task == "classification" else None
    p = X.shape[1]
    fit_cap, held_cap = min(512, max(128, p)), min(256, max(64, p))
    states, audit = [], []
    for i, (train_pool, held_pool) in enumerate(_environment_splits(y, task, groups, ordered_groups, seed)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        labels = y if classes is not None else None
        fit = _sample_rows(train_pool, fit_cap, labels, rng)
        held = _sample_rows(held_pool, held_cap, labels, rng)
        if len(fit) < 2 or len(held) < 2 or np.unique(y[fit]).size < 2:
            continue
        train, check = _standardize(np.asarray(X[fit], dtype=float), np.asarray(X[held], dtype=float))
        if classes is None:
            target = y[fit, None].astype(float)
            observed = y[held, None].astype(float)
            weights = np.ones(len(held))
        else:
            target = (y[fit, None] == classes).astype(float)
            observed = (y[held, None] == classes).astype(float)
            _, codes, counts = np.unique(y[held], return_inverse=True, return_counts=True)
            weights = 1. / counts[codes]
        center = target.mean(axis=0)
        held_target = observed - center
        null_loss = np.dot(weights, np.square(held_target).sum(axis=1))
        if null_loss <= 0:
            continue
        # Only held losses are weighted. Dividing by intercept-only held loss
        # puts a unit change on the same scale in every environment.
        scale = np.sqrt(weights / null_loss)[:, None]
        states.append(_FoldState(train, check * scale, target - center, held_target * scale, max_size))
        audit.append({"fit_rows": fit, "held_rows": held})
    if len(states) < 2:
        raise InsufficientEnvironmentsError("Need at least two evaluable training environments")
    return states, audit


def environment_path_from_training(X, y, counts, *, task="classification", random_state=42,
                                   groups=None, ordered_groups=False, tiers=None):
    """Return the nested supports for several budgets using one deterministic path.

    At each checkpoint rank unused columns by their smallest held-out loss
    reduction across environments and the current/previous-by-one selected
    sets. Each environment fits unit ridge on its own reference rows. Ties use
    input order. Optional tiers are block checkpoints below the largest
    requested non-full width and must include zero.
    """
    counts = tuple(counts)
    if not counts:
        raise ValueError("Provide at least one feature budget")
    X, y = _utility_inputs(X, y, counts[0], task)
    p = X.shape[1]
    if any(not isinstance(v, Integral) or isinstance(v, (bool, np.bool_)) or not 1 <= v <= p for v in counts):
        raise ValueError("Budgets must be integers in [1,p]")
    if not isinstance(random_state, Integral) or isinstance(random_state, (bool, np.bool_)) or not 0 <= random_state < 2**32:
        raise ValueError("Use an integer seed in [0, 2**32)")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape or pd.isna(groups).any():
            raise ValueError("Groups must be complete and aligned with training rows")
    if ordered_groups and groups is None:
        raise ValueError("ordered_groups requires group identifiers")
    strategy = ("forward" if ordered_groups else "group") if groups is not None else "row"
    supports = {p: np.arange(p)} if p in counts else {}
    widths = sorted(set(counts) - {p})
    if not widths:
        return supports, {"strategy": strategy, "folds": [], "checkpoints": []}
    width = widths[-1]
    if tiers is None:
        sizes = [0] + [1 << k for k in range((int(width) - 1).bit_length())]
    else:
        sizes = sorted(_selection_tiers(tiers, p))
        if sizes[0] != 0 or sizes[-1] >= width:
            raise ValueError("Checkpoints must include zero and be below the largest reduced width")
    states, folds = _environment_states(X, y, task, groups, ordered_groups, int(random_state), sizes[-1])
    def worst_environment_gain():
        return np.minimum.reduce([state.scores() for state in states])
    current = worst_environment_gain()
    available = np.ones(p, dtype=bool)
    order = []
    for size, end in zip(sizes, sizes[1:] + [width]):
        if size:
            if len(block) > 1:
                for state in states:
                    state.append(block[:-1])
                previous = worst_environment_gain()
            else:
                previous = current
            for state in states:
                state.append(block[-1:])
            current = worst_environment_gain()
            scores = np.minimum(current, previous)
        else:
            scores = current
        pool = np.flatnonzero(available)
        block = pool[np.argsort(-scores[pool], kind="stable")[:end-size]]
        order.extend(block)
        available[block] = False
    for count in widths:
        supports[count] = np.sort(order[:count])
    return supports, {"strategy": strategy, "folds": folds, "checkpoints": sizes}


def environment_top_from_training(X, y, count, **fit_params):
    """Return one fixed-width support and the in-training environment metadata."""
    supports, info = environment_path_from_training(X, y, (count,), **fit_params)
    return supports[count], info
