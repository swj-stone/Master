"""minPI for groups of interchangeable measurements discovered on training rows."""
import numpy as np

from minppi_mofs.moment_minpi import _moment_halves
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def _unit_columns(values):
    centered = np.array(values, dtype=float, copy=True)
    finite = np.isfinite(centered)
    if finite.all():
        centered -= centered[0]
        centered -= centered.mean(axis=0)
    else:
        centered[~finite] = 0.
        mean = centered.sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
        centered -= mean
        centered[~finite] = 0.
    # Scaling first keeps the norm calculation finite on large measurement units.
    scale = np.max(np.abs(centered), axis=0)
    centered /= np.where(scale > 0, scale, 1.)
    norm = np.sqrt(np.einsum("ij,ij->j", centered, centered))
    centered /= np.where(norm > 0, norm, 1.)
    return centered


def _target(raw, classes):
    data = (raw[:, None] == classes).astype(float) if classes is not None else raw[:, None].astype(float)
    data -= data.mean(axis=0)
    norm = np.linalg.norm(data)
    return data / (norm if norm else 1.)


def _proxy_groups(a, b, evidence, rng):
    # A hash collision proposes a comparison; acceptance uses observed correlations.
    # The participation-ratio guard excludes correlations driven by a lone outlier.
    p = a.shape[1]
    ids = np.arange(p)
    parent = ids.copy()
    eligible = (np.einsum("ij,ij->j", a*a, a*a) <= .25) & (np.einsum("ij,ij->j", b*b, b*b) <= .25)
    for _ in range(2):
        pool = np.flatnonzero(eligible)
        if len(pool) < 2:
            break
        sample = rng.choice(len(a), min(12, len(a)), replace=False)
        signs = a[np.ix_(sample, pool)] > 0
        # Canonicalize sign so inverse measurements can share a fingerprint.
        signs ^= signs[:1].copy()
        bits = np.packbits(signs, axis=0)
        key = bits[0].astype(np.int32)
        if len(bits) == 2:
            key = (key << 8) + bits[1]
        order = np.lexsort((pool, -evidence[pool], key))
        ordered = pool[order]
        first = np.r_[True, np.diff(key[order]) != 0]
        leader = ordered[np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))]
        members, anchors = ordered[~first], leader[~first]
        if not len(members):
            continue
        fit_r = np.einsum("ij,ij->j", a[:, members], a[:, anchors])
        held_r = np.einsum("ij,ij->j", b[:, members], b[:, anchors])
        match = (np.abs(fit_r) >= .90) & (np.abs(held_r) >= .90) & (fit_r * held_r > 0.)
        members, anchors = members[match], anchors[match]
        parent[members] = anchors
        eligible[members] = eligible[anchors] = False
    return parent


def _coverage_scores(evidence, parent):
    """Worst gain at zero and all peers in the sum-of-within-group-maxima game."""
    p = len(parent)
    ids = np.arange(p)
    mass = np.maximum(evidence, 0.)
    runner_up = np.zeros(p)
    followers = parent != ids
    np.maximum.at(runner_up, parent[followers], mass[followers])
    # Parents have maximal mass in their group; followers add zero at all peers.
    scores = np.where(followers, 0., mass - runner_up)
    return np.minimum(mass, scores)


def pilot_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                            groups=None, ordered_groups=False):
    """Select group representatives, or rank to a supplied hard width.

    Use at most 96 training rows per side and reduce this to a floor of 12
    on very wide matrices. Feature strength is the agreement of the two
    feature-target correlation vectors. Each group's utility is the maximum
    nonnegative strength amongst its active members. Rank by the smaller
    marginal at zero and all peers, protecting one representative per group
    when the budget permits. Strength and input index resolve vote ties.

    Automatic width retains one representative per detected group. If
    there is no confirmed proxy and p>1, drop one lowest-ranked column.
    For p=1 report the unchanged support. Ordered groups use strictly
    earlier rows for the first side of the correlation check.
    """
    X, y = _utility_inputs(X, y, 1 if count is None else count, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    ids = np.arange(p)
    if p == 1 or count == p:
        return ids, {"budget": p, "folds": [], "scores": np.zeros(p), "groups_found": p}
    rng = np.random.default_rng(random_state)
    left, right = _moment_halves(y, task, groups, ordered_groups, rng)
    cap = min(96, max(12, 32768 // p))
    if len(left) > cap:
        left = rng.choice(left, cap, replace=False)
    if len(right) > cap:
        right = rng.choice(right, cap, replace=False)
    a, b = _unit_columns(X[left]), _unit_columns(X[right])
    classes = np.unique(np.concatenate((y[left], y[right]))) if task == "classification" else None
    fit = a.T @ _target(y[left], classes)
    held = b.T @ _target(y[right], classes)
    evidence = np.einsum("ij,ij->i", fit, held)
    parent = _proxy_groups(a, b, evidence, rng)
    leaders = ids == parent
    group_count = int(leaders.sum())
    width = min(group_count, p - 1) if count is None else count
    scores = _coverage_scores(evidence, parent)
    order = np.lexsort((ids, -evidence, -scores, ~leaders))
    info = {"budget": int(width), "groups_found": group_count, "peer_window": (0, p - 1),
            "folds": [{"fit_rows": left, "held_rows": right}], "parent": parent, "scores": scores}
    return np.sort(order[:width]), info
