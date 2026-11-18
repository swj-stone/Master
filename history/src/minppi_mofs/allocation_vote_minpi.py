"""Allocate varying density across a spectral axis with supervised minPI votes."""
import numpy as np

from minppi_mofs.budget_vote_minpi import _endpoints
from minppi_mofs.calibrated_band_minpi import _calibration_rows, _refit_loss, _validation_data
from minppi_mofs.moment_minpi import _draw, _moment_halves
from minppi_mofs.signal_vote_minpi import _half, _row_dot
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _orders(scores, marginal):
    p = len(scores)
    index = np.arange(p)
    block = index // 16
    starts = index[::16]
    mids = starts + np.minimum(starts + 16, p) - 1
    distance = np.abs(2 * index - mids[block])
    scale = np.max(np.abs(scores))
    stable = np.round(scores / scale, decimals=12) * scale if scale else scores
    representatives = np.lexsort((index, distance, -stable, block))[starts]
    secondary = np.ones(p, dtype=bool)
    secondary[representatives] = False
    local = np.convolve(np.pad(marginal, 2, mode="edge"), np.full(5, .2), mode="valid")
    # Held and pooled gradients are squared-loss reductions on the same scale.
    region_rank = np.lexsort((index, -(local + stable), secondary))
    vote_rank = np.lexsort((index, -stable, secondary))
    return (("regional", region_rank), ("vote", vote_rank)), representatives


def _f_support(x, target, labels, count, classification):
    """ANOVA or linear-regression F order on standardized, training-imputed X."""
    square = np.einsum("ij,ij->j", x, x)
    cross = x.T @ target
    if classification:
        _, sizes = np.unique(labels, return_counts=True)
        # One-hot target columns can include a class found only on held rows.
        varying = np.any(target != 0., axis=0)
        if sizes.size > 1:
            between = np.sum(cross[:, varying]**2 / sizes, axis=1)
        else:
            between = np.zeros(len(square))
    else:
        between = cross[:, 0]**2
    strength = np.divide(between, square, out=np.zeros_like(square), where=square > 0)
    return np.sort(np.argsort(-strength, kind="stable")[:count])


def _compare(checks, support):
    excess = []
    for x, z, target, query, alpha, baseline, variance in checks:
        loss = _refit_loss(x[:, support], z[:, support], target, query, alpha)
        delta = loss - baseline
        excess.append(delta / variance if variance > 0 else (0. if delta <= 0 else np.inf))
    return max(excess)


def allocation_vote_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                      groups=None, ordered_groups=False, backend="auto"):
    """Keep voted coverage of every 16-channel band; allocate extra slots by signal.

    Rows for fitting votes are disjoint from the accuracy calibration. Local
    marginal evidence pools squared-loss gains over five adjacent channels.
    Adjacent-pair held votes use the minimum expected gain with zero peers
    and half the other columns present, under the additive paired-block game.
    One candidate order allocates extra slots entirely by these votes, the
    other gives equal weight to local marginal evidence. Both first retain
    one voted representative from every 16-channel interval.

    After refitting standardized ridge on the training sketches, require the
    loss to be within 1% of held target variation of the better of full ridge
    and training-ranked F at half width, in every permitted direction. Take
    the narrowest passing budget, else the candidate with the smallest worst
    normalized excess; all default budgets are at most ceil(p/2). Without
    enough disjoint calibration rows, use the regional order at the cap.
    Input columns must be in acquisition order. A numeric count overrides
    calibration and uses the regional order.
    """
    if backend not in ("auto", "native", "numpy"):
        raise ValueError("backend must be auto, numpy or native")
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    cap = (p + 1) // 2
    info = dict(cap=cap, folds=[], calibration_folds=[], checked=[], full_reference_loss=[],
                f_reference_loss=[], peer_window=(0, (p-1)//2))
    if p == 1:
        return np.array([0]), dict(info, budget=1, scores=np.zeros(1), zero_peer_scores=np.zeros(1),
                                   coverage_representatives=np.array([0]), allocation="only_column")
    rng = np.random.default_rng(random_state)
    left, right = _moment_halves(y, task, groups, ordered_groups, rng)
    # Leaving part of each sampled side unused allows calibration even on
    # training sets smaller than the 192-row cap of the paired vote kernel.
    left = _draw(left, max(1, min(72, 3 * len(left) // 4)), rng)
    right = _draw(right, max(1, min(72, 3 * len(right) // 4)), rng)
    info["folds"].append(dict(fit_rows=left, held_rows=right))
    if not ordered_groups:
        info["folds"].append(dict(fit_rows=right, held_rows=left))
    rows = np.ascontiguousarray(np.r_[left, right], dtype=np.int64)
    mate = np.arange(p, dtype=np.int64) ^ 1
    if p % 2:
        mate[-1] = p - 1
    classification = task == "classification"
    votes, used = _endpoints(X, rows, len(left), mate, y, classification, len(info["folds"]), backend)
    zero, conditional = votes.mean(axis=0)
    fraction = info["peer_window"][1] / (p-1)
    scores = np.minimum(zero, zero + fraction * (conditional - zero))
    labels = y[rows]
    encoded = ((labels[:, None] == np.unique(labels)).astype(float) if classification else
               labels[:, None].astype(float))
    pooled = _half(np.asarray(X[rows], dtype=float), encoded)
    marginal = np.divide(_row_dot(pooled.rhs, pooled.rhs), len(rows) * pooled.square,
                         out=np.zeros(p), where=pooled.square > 0)
    orders, representatives = _orders(scores, marginal)
    info.update(scores=scores, zero_peer_scores=zero, neighbor_peer_scores=conditional,
                coverage_representatives=representatives, backend=used)
    if count is not None:
        return np.sort(orders[0][1][:count]), dict(info, budget=int(count), allocation="regional_fixed")
    folds = _calibration_rows(y, task, groups, ordered_groups, random_state, info["folds"])
    info["calibration_folds"] = folds
    if not folds:
        return np.sort(orders[0][1][:cap]), dict(info, budget=cap, allocation="regional_no_calibration")
    checks = []
    for fold, check in zip(folds, _validation_data(X, y, task, folds)):
        x, z, target, query, alpha, full, tolerance = check
        f = _f_support(x, target, y[fold["fit_rows"]], cap, classification)
        f_loss = _refit_loss(x[:, f], z[:, f], target, query, alpha)
        checks.append((x, z, target, query, alpha, min(full, f_loss), 100 * tolerance))
        info["full_reference_loss"].append(full)
        info["f_reference_loss"].append(f_loss)
    best, best_key, best_name = None, None, None
    for width in sorted({(p * i + 7) // 8 for i in range(1, 5)}):
        previous = None
        for name, ranking in orders:
            support = np.sort(ranking[:width])
            if previous is not None and np.array_equal(support, previous):
                continue
            previous = support
            risk = _compare(checks, support)
            info["checked"].append(dict(width=width, allocation=name, worst_excess=risk))
            key = (risk, -width)
            if best_key is None or key < best_key:
                best_key, best_name, best = key, name, support
        if best_key[0] <= .01:
            break
    return best, dict(info, budget=len(best), allocation=best_name, worst_excess=best_key[0])
