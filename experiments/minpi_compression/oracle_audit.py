"""Audit the sampler on the small confirmation tasks using only fitting and voting rows."""
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from experiments.minpi_compression.sources import ROOT
from minppi_mofs.sampled_linear_minpi import rank_features


def complete_votes(x, y, z, truth, task):
    """Fit each support independently using SVD. Return all quota votes by bitmask."""
    if task == "classification":
        labels = np.unique(y)
        y = (y[:, None] == labels).astype(float)
        truth = (truth[:, None] == labels).astype(float)
    else:
        y, truth = y.reshape(-1, 1), truth.reshape(-1, 1)
    p = x.shape[1]
    losses = np.empty(1 << p)
    losses[0] = np.square(truth - y.mean(axis=0)).sum() / len(z)
    for mask in range(1, 1 << p):
        cols = [j for j in range(p) if mask & (1 << j)]
        model = Ridge(alpha=1, solver="svd").fit(x[:, cols], y)
        prediction = model.predict(z[:, cols]).reshape(truth.shape)
        losses[mask] = np.square(truth - prediction).sum() / len(z)
    u = 1. / (1. + losses / (losses[0] or 1.))
    return (u >= .95 * u[-1]).astype(int)


def chance_of_witness(vote, p, j):
    """Probability that any inspected size in one shared round witnesses a -1.

    Conditional on the target's block, the next blocks concatenate into a
    uniform prefix permutation of the other columns. A subset DP propagates
    only probability mass that has not encountered a negative at a checked size.
    """
    sizes = [len(group) for group in np.array_split(np.arange(p), min(4, p))]
    b = len(sizes)
    weights = 0.
    for block, width in enumerate(sizes):
        subsequent = [sizes[(block + k) % b] for k in range(1, b)]
        cuts = set(np.cumsum(subsequent))
        if b < 4:
            cuts.add(0)
        alive = {0: 1.}
        for size in range(sum(subsequent) + 1):
            if size in cuts:
                alive = {s: prob for s, prob in alive.items() if vote[s | (1 << j)] >= vote[s]}
            if size == sum(subsequent):
                break
            more = {}
            for s, prob in alive.items():
                for k in range(p):
                    if k != j and not s & (1 << k):
                        child = s | (1 << k)
                        more[child] = more.get(child, 0.) + prob / (p-1-size)
            alive = more
        weights += (width / p) * sum(alive.values())
    return np.clip(1 - weights, 0., 1.)


def check_dynamic_program():
    # Independent finite enumeration of the actual shuffled blocks for an
    # unequal-block case. A strict equality in this small fixture would give
    # a false discrepancy due to floating point summation of probabilities.
    p = 5
    v = np.random.default_rng(123).integers(0, 2, size=1 << p)
    hit = np.zeros(p)
    for perm in permutations(range(p)):
        blocks = [perm[:2], perm[2:3], perm[3:4], perm[4:]]
        for b, block in enumerate(blocks):
            seen, mask = set(), 0
            for k in range(1, 4):
                mask |= sum(1 << j for j in blocks[(b+k) % 4])
                seen.update(j for j in block if v[mask | (1 << j)] < v[mask])
            for j in seen:
                hit[j] += 1
    np.testing.assert_allclose([chance_of_witness(v, p, j) for j in range(p)], hit/120., atol=1e-12)


def audit_one(name, x, y, z, truth, task, seed):
    votes = complete_votes(x, y, z, truth, task)
    fitted = rank_features(x, y, z, truth, task=task, seed=seed)
    rows = []
    for j, (observed, witness) in enumerate(zip(fitted.minimum, fitted.witness, strict=True)):
        base = [s for s in range(len(votes)) if not s & (1 << j)]
        deltas = votes[np.array(base) | (1 << j)] - votes[base]
        exact = deltas.min()
        assert observed >= exact
        assert (witness is not None) == (observed == -1)
        if witness is not None:
            mask = sum(1 << int(k) for k in witness)
            assert j not in witness and votes[mask | (1 << j)] - votes[mask] == -1
        q = chance_of_witness(votes, x.shape[1], j)
        if exact != -1:
            assert q < 1e-12
        rows.append(dict(source=name, seed=seed, feature=j, exact=exact, sampled=observed,
                         harmful_edges=int(sum(deltas == -1)), edges=len(base), q=q,
                         miss_given_negative=(1.-q)**4 if exact == -1 else np.nan))
    return rows


def main():
    from experiments.minpi_compression.benchmark import DESIGN, prepare
    check_dynamic_program()
    output = ROOT / "thesis-paper/evidence/minpi_compression/audit"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with threadpool_limits(limits=1):
        for name in DESIGN["confirmation"]:
            for seed in DESIGN["seeds"]:
                arrays = prepare(name, seed)
                if arrays["X_fit"].shape[1] > 10:
                    break
                rows.extend(audit_one(name, arrays["X_fit"], arrays["y_fit"],
                                      arrays["X_vote"], arrays["y_vote"], arrays["task"], seed))
    data = pd.DataFrame(rows)
    data.to_csv(output / "exact-minima.csv", index=False)
    for tag, subset in (("seed42", data[data.seed == 42]), ("all", data)):
        negative = subset[subset.exact == -1]
        print(tag, len(subset), "exact negative", len(negative),
              "seen", int((negative.sampled == -1).sum()), "unsampleable", int((negative.q < 1e-12).sum()),
              "expected discoveries", (1 - negative.miss_given_negative).sum())
        print(pd.crosstab(subset.exact, subset.sampled))


if __name__ == "__main__":
    main()
