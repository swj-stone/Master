"""Fast relative-negative proposals from cross-calibrated coalition-size votes."""

import numpy as np
from xgboost import DMatrix

from minppi_mofs.piv0 import _PrefixUtility


ROW_LIMIT = 512
N_BASE_ORDERS = 4
N_OFFSETS = 4
NEUTRAL_PSEUDOCOUNT = 2
TIE_SCALE = .25


def _sample_rows(labels, metric, rng):
    if len(labels) <= ROW_LIMIT:
        return np.arange(len(labels))
    if metric == "r2":
        return np.sort(rng.choice(len(labels), ROW_LIMIT, replace=False))
    classes, counts = np.unique(labels, return_counts=True)
    if metric == "roc_auc" and len(classes) == 2:
        # ROC averages positive/negative pairs. Retain rare examples on either side.
        n = int(np.clip(ROW_LIMIT // 2, max(1, ROW_LIMIT-counts[1]), min(counts[0], ROW_LIMIT-1)))
        quotas = [n, ROW_LIMIT-n]
    elif metric == "ovr_auc" and 2 < len(classes) <= ROW_LIMIT:
        # Prioritize representing every class; apportion spare slots by frequency.
        shares = (ROW_LIMIT-len(classes)) * (counts-1) / (len(labels)-len(classes))
        floors = np.floor(shares).astype(int)
        quotas = 1 + floors
        extra = ROW_LIMIT - int(quotas.sum())
        quotas[np.argsort(-(shares-floors), kind="stable")[:extra]] += 1
    else:
        raise ValueError("Expected binary ROC AUC or a multiclass macro OVR AUC with all classes represented")
    selected = [rng.choice(np.flatnonzero(labels == label), int(n), replace=False)
                for label, n in zip(classes, quotas, strict=True)]
    return np.sort(np.concatenate(selected))


def _order_blocks(width, rng):
    orders = []
    for _ in range(N_BASE_ORDERS):
        base = rng.permutation(width)
        for offset in np.arange(N_OFFSETS) * width // N_OFFSETS:
            order = np.roll(base, -offset)
            orders.extend((order, order[::-1]))
    return np.array(orders)


def _vote_summary(orders, changes, tiers):
    width = orders.shape[1]
    count = np.zeros((3, width), dtype=int)
    net = np.zeros_like(count)
    centers, tolerances, fold_rates = [], [], []
    # Each fold contains whole rotation/reversal blocks. Its position-wise
    # reference is estimated exclusively from the other independent base orders.
    order_folds = np.split(orders, 2)
    change_folds = np.split(changes, 2)
    for fold, (schedule, observations) in enumerate(zip(order_folds, change_folds, strict=True)):
        reference_sample = change_folds[1-fold]
        center = reference_sample.mean(axis=0)
        spread = np.median(np.abs(reference_sample-center), axis=0)
        tie = np.maximum(1e-12, TIE_SCALE * spread)
        residual = observations - center
        ballots = (residual > tie).astype(int) - (residual < -tie).astype(int)
        fold_count = np.zeros_like(count)
        fold_net = np.zeros_like(net)
        for order, ballot in zip(schedule, ballots, strict=True):
            fold_count[tiers, order] += 1
            fold_net[tiers, order] += ballot
        count += fold_count
        net += fold_net
        fold_rates.append(fold_net / (fold_count + NEUTRAL_PSEUDOCOUNT))
        centers.append(center.tolist())
        tolerances.append(tie.tolist())
    rates = net / (count + NEUTRAL_PSEUDOCOUNT)
    worst = rates.min(axis=0)
    consistent = ((fold_rates[0] < 0) & (fold_rates[1] < 0)).any(axis=0)
    return dict(tier_votes=rates.tolist(), tier_counts=count.tolist(), min_vote=worst.tolist(),
                mean_vote=rates.mean(axis=0).tolist(), negative=(worst < 0.).tolist(),
                fold_tier_votes=[r.tolist() for r in fold_rates], negative_consistent=consistent.tolist(),
                position_reference=centers, position_tolerance=tolerances)


def relative_votes(model, valid, labels, *, metric, seed=42, progress=None):
    """Measure signed feature contributions relative to others added at the same size.

    One row sample is shared by both calibration folds and every candidate coalition.
    A negative minimum is a reversible proposal for the subsequent subset search.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(valid) or not len(valid) or valid.shape[1] < 3:
        raise ValueError("Voting needs aligned validation features and labels and at least three features")
    row_seed, order_seed = np.random.SeedSequence(seed).spawn(2)
    rows = _sample_rows(labels, metric, np.random.default_rng(row_seed))
    sampled = valid.iloc[rows] if hasattr(valid, "iloc") else valid[rows]
    matrix = DMatrix(sampled, nthread=1)
    booster = model.get_booster()
    contrib = booster.predict(matrix, pred_contribs=True, approx_contribs=False, strict_shape=True)
    bias = contrib[:, :, -1].T.astype(float)
    phi = np.ascontiguousarray(contrib[:, :, :-1].transpose(2, 1, 0), dtype=float)
    width, outputs, size = phi.shape
    if not np.all(bias == bias[:, :1]):
        raise ValueError("PIv1 needs constant class or regression intercepts across validation rows")
    full = phi.sum(axis=0) + bias
    actual = booster.predict(matrix, output_margin=True, strict_shape=True).T
    if not np.allclose(full, actual, rtol=1e-4, atol=1e-4):
        raise ValueError("PIv1 TreeSHAP contributions do not reconstruct the frozen model")
    score = _PrefixUtility(metric, labels[rows], outputs)
    empty = float(score(bias[None])[0])
    full_score = float(score(full[None])[0])
    positions = np.arange(width)
    tiers = np.searchsorted([width//3, 2*width//3], positions, side="right")
    schedule = _order_blocks(width, np.random.default_rng(order_seed))
    changes = np.empty(schedule.shape, dtype=float)
    batch = max(1, 350_000 // (outputs * size))
    for step, order in enumerate(schedule):
        margins = np.cumsum(phi[order], axis=0) + bias
        values = np.empty(width + 1)
        values[0], values[-1] = empty, full_score
        for start in range(0, width-1, batch):
            stop = min(start+batch, width-1)
            values[start+1:stop+1] = score(margins[start:stop])
        changes[step] = np.diff(values)
        if progress is not None:
            progress("relative_votes", step+1, len(schedule))
    summary = _vote_summary(schedule, changes, tiers)
    return dict(**summary, metric=metric, seed=seed, samples=len(rows), paths=len(schedule),
                row_indices=rows.tolist(), full_proxy_utility=full_score,
                empty_utility=empty, full_margin_max_abs_error=float(np.max(np.abs(full-actual))),
                policy=dict(row_limit=ROW_LIMIT, base_orders=N_BASE_ORDERS, offsets=N_OFFSETS,
                            position_tolerance_scale=TIE_SCALE, neutral_pseudocount=NEUTRAL_PSEUDOCOUNT))
