"""Audit a small number of exchanges in a uniform random feature support."""
import numpy as np
from scipy import sparse

from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.tournament_minpi import _halves
from minppi_mofs.utility_minpi import _utility_inputs


def _constant_block(X):
    """Certify entirely finite, exactly constant columns when a quarter may be redundant."""
    n, p = X.shape
    if p < 16:
        return np.empty(0, dtype=int)
    minimum = (p + 3) // 4
    if sparse.issparse(X):
        # Duplicate coordinates can sum to zero. Work on a copy only when
        # consolidation is needed; preserve explicit NaNs as nonzero entries.
        canonical = X
        if not X.has_canonical_format:
            canonical = X.copy()
            canonical.sum_duplicates()
        present = np.zeros(p, dtype=bool)
        nonzero = canonical.data != 0
        present[canonical.indices[nonzero]] = True
        ids = np.flatnonzero(~present)
        return ids if len(ids) >= minimum else np.empty(0, dtype=int)
    probe = X[np.linspace(0, n - 1, min(n, 8), dtype=int)]
    ref = probe[0]
    ids = np.flatnonzero(np.isfinite(ref) & (probe == ref).all(axis=0))
    if len(ids) < minimum:
        return np.empty(0, dtype=int)
    step = max(1, min(2048, 524288 // len(ids)))
    for start in range(0, n, step):
        ids = ids[(X[start:start + step, ids] == ref[ids]).all(axis=0)]
        if len(ids) < minimum:
            return np.empty(0, dtype=int)
    return ids


def _loss_votes(train, check, target, query, anchor):
    """Per-feature zero-peer gain and interaction with its assigned retained anchor."""
    finite = np.isfinite(train)
    if finite.all():
        base = train[0]
        data = train - base
        mean = data.mean(axis=0)
        data -= mean
    else:
        base = train[np.argmax(finite, axis=0), np.arange(train.shape[1])]
        base = np.where(np.isfinite(base), base, 0.)
        data = np.subtract(train, base, where=finite, out=np.zeros_like(train))
        mean = data.sum(axis=0) / np.maximum(1, finite.sum(axis=0))
        np.subtract(data, mean, where=finite, out=data)
    observed = np.isfinite(check)
    held = np.subtract(check, base, where=observed, out=np.zeros_like(check))
    np.subtract(held, mean, where=observed, out=held)
    center = target.mean(axis=0)
    norm = np.einsum("ij,ij->j", data, data)
    # Unit ridge on columns of unit population variance, evaluated in raw units.
    coef = (data.T @ (target - center)) / np.where(norm > 0, norm * (1. + 1. / len(data)), 1.)[:, None]
    n = len(anchor)
    z = held[:, :n]
    b = coef[:n]
    gain = 2 * np.einsum("ij,ij->i", z.T @ (query - center), b)
    gain -= np.einsum("ij,ij->j", z, z) * np.einsum("ij,ij->i", b, b)
    interaction = (-2 * np.einsum("ij,ij->j", z, held[:, anchor])
                   * np.einsum("ij,ij->i", b, coef[anchor]))
    return gain / len(held), interaction / len(held)


def _audit_support(X, y, order, count, task, groups, ordered, rng):
    p = len(order)
    initial = p // 2 if count is None else count
    m = min(12, max(1, initial // 2), initial, p - initial)
    incumbents, challengers = order[:m], order[-m:]
    audited = np.r_[incumbents, challengers]
    if initial == 1:
        columns, anchors = audited, np.array([1, 0])
    else:
        columns = np.r_[audited, order[m:2 * m]]
        anchors = np.tile(np.arange(2 * m, 3 * m), 2)
    left, right = _halves(y, task, groups, ordered, 48, rng)
    sample = np.r_[left, right]
    selected = X[np.ix_(sample, columns)]
    if sparse.issparse(selected):
        selected = selected.toarray()
    data = np.asarray(selected, dtype=np.float64)
    if task == "classification":
        labels = y[sample]
        target = (labels[:, None] == np.unique(labels)).astype(float)
    else:
        target = np.asarray(y[sample, None], dtype=float)
    a, b = np.arange(len(left)), np.arange(len(left), len(sample))
    if ordered:
        parts = [(a, held) for held in (b[::2], b[1::2]) if len(held)]
    else:
        parts = [(a, b), (b, a)]
    stats = [_loss_votes(data[fit], data[held], target[fit], target[held], anchors)
             for fit, held in parts]
    empty, delta = np.stack(stats).transpose(1, 0, 2)
    if count is None:
        weak = np.all(np.maximum(empty, empty + delta) <= 0, axis=0).sum()
        discard = min(.5, (1. + weak) / (2. + len(audited)))
        count = p - max(1, int(np.floor(discard * p)))
    high = min(p - 1, max(1, count - 1))
    scores = np.minimum(empty, empty + delta * (high / (p - 1)))
    # Challenger positions may be inside the adaptive width. Never swap two
    # existing members or disturb untouched columns.
    winners = np.all(scores[:, m:] > scores[:, :m], axis=0) & (np.arange(p - m, p) >= count)
    keep = order[:count].copy()
    keep[np.flatnonzero(winners)] = challengers[winners]
    return np.sort(keep), dict(
        budget=int(count), active_columns=audited, anchors=columns[anchors],
        scores=scores, peer_window=(0, high), exchanged=int(winners.sum()),
        folds=[{"fit_rows": sample[fit], "held_rows": sample[held]} for fit, held in parts])


def exchange_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                    groups=None, ordered_groups=False):
    """Apply verified zero votes and small local minPI exchanges using only training data.

    The local squared-loss game includes one audited feature j and a context
    column a. Their unit-ridge singleton predictions q are additive. A held row
    votes by its loss reduction when j is added. When a is present the change
    to j's empty-peer vote is -2 <q_j,q_a>. Other features are dummies in this
    local game. With k uniformly sampled peers, a appears with probability
    k/(p-1). Rank each exchange by min(k=0, k=max(1,h-1)), and require a
    strict improvement in all available held batches. At h=1 the other
    contender acts as the single anchor. This is a restricted collection
    of local games, not an evaluation of the full fitted ridge loss.

    Without a supplied width, count the audited columns with nonpositive votes
    at both endpoints on every held batch. Add one pseudocount to each outcome,
    cap the estimated deletion fraction at one half, and delete at least one
    of p>1 columns. When at least a quarter of p>=16 features are verified
    constant across all training rows, the automatic rule retains every
    other column (one arbitrary column if all are constant).
    """
    width = 1 if count is None else count
    if sparse.issparse(X):
        if X.ndim != 2:
            raise ValueError("Expected a two-dimensional sparse matrix")
        X = X.tocsr(copy=False)
        # The common input validator only inspects X.shape, and np.asarray
        # does not preserve the shape of SciPy sparse inputs. A broadcast
        # scalar supplies the same shape without allocating an n-by-p matrix.
        _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, width, task)
    else:
        X, y = _utility_inputs(X, y, width, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    info = {"strategy": "forward" if ordered_groups else "group" if groups is not None else "row",
            "folds": [], "exchanged": 0}
    if p == 1 or count == p:
        return np.arange(p), dict(info, budget=p, branch="full")
    rng = np.random.default_rng(random_state)
    order = rng.permutation(p)
    constants = _constant_block(X)
    if len(constants):
        mask = np.zeros(p, dtype=bool)
        mask[constants] = True
        variables = order[~mask[order]]
        info["constant_columns"] = constants
        if count is None or count >= len(variables):
            width = max(1, len(variables)) if count is None else count
            keep = np.r_[variables, order[mask[order]][:width - len(variables)]]
            return np.sort(keep), dict(info, budget=int(width), branch="constant")
        order = variables
    cols, audit = _audit_support(X, y, order, count, task, groups, ordered_groups, rng)
    return cols, {**info, **audit, "branch": "constant_then_exchange" if len(constants) else "exchange"}
