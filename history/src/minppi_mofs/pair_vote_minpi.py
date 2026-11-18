"""Size-conditioned minPI votes for a game with disjoint two-column ridge models."""
import numpy as np

from minppi_mofs.sketch_minpi import _prepared_folds, _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _pair_columns(train, rng):
    # Close projective angles favor parallel or antiparallel columns. The map
    # is computed entirely on the fit side and has only two coordinates.
    projection = rng.standard_normal((2, len(train))) @ train
    angles = np.mod(np.arctan2(projection[1], projection[0]), np.pi)
    order = np.argsort(angles, kind="stable")
    paired = order[:2 * (len(order) // 2)].reshape(-1, 2)
    mate = np.arange(len(order))
    mate[paired[:, 0]], mate[paired[:, 1]] = paired[:, 1], paired[:, 0]
    return mate


def _pair_gains(train, check, target, query, mate):
    """Loss gains on held rows when the mate is absent and when it is present."""
    def inner_rows(a, b):
        return np.einsum("ij,ij->i", a, b)

    diag = 1. + np.einsum("ij,ij->j", train, train)
    cross = np.einsum("ij,ij->j", train, train[:, mate])
    rhs = train.T @ target
    single = rhs / diag[:, None]
    pair = ((diag[mate, None] * rhs - cross[:, None] * rhs[mate]) /
            (diag * diag[mate] - cross**2)[:, None])

    rhs_held = check.T @ query
    norm = np.einsum("ij,ij->j", check, check)
    product = np.einsum("ij,ij->j", check, check[:, mate])
    alone = 2 * inner_rows(rhs_held, single) - norm * inner_rows(single, single)
    part = 2 * inner_rows(rhs_held, pair) - norm * inner_rows(pair, pair)
    together = part + part[mate] - 2 * product * inner_rows(pair, pair[mate])
    with_mate = together - alone[mate]
    singleton = mate == np.arange(len(mate))
    with_mate[singleton] = alone[singleton]
    return alone / len(check), with_mate / len(check)


def pair_vote_top_from_training(X, y, count, *, task="classification", random_state=1129,
                                groups=None, ordered_groups=False):
    """Rank columns by the weakest expected vote at zero and near-budget peers.

    Each block contributes its held squared-loss improvement from the unit-ridge
    refit on the active columns of that block. Other blocks cancel in a marginal
    comparison. For k uniform peers the probability of the mate being present is
    k/(p-1). An odd unpaired column contributes the same gain at both endpoints.
    """
    X, y = _utility_inputs(X, y, count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    high = min(p - 1, max(1, count - 1))
    info = {"ballot": "paired_refit_loss", "budget": int(count),
            "peer_window": (0, high), "folds": []}
    if count == p:
        return np.arange(p), info
    rng = np.random.default_rng(random_state)
    gains = np.zeros((2, p))
    for fit, held, train, check, target, query in _prepared_folds(
            X, y, task, random_state, groups, ordered_groups):
        mate = _pair_columns(train, rng)
        gains += _pair_gains(train, check, target, query, mate)
        info["folds"].append({"fit_rows": fit, "held_rows": held, "mate": mate})
    alone, with_mate = gains / len(info["folds"])
    scores = np.minimum(alone, alone + (with_mate - alone) * (high / (p - 1)))
    info["scores"] = scores
    return np.sort(np.argsort(-scores, kind="stable")[:count]), info
