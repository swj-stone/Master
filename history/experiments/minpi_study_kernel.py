"""Fixed-matrix exact versus Monte Carlo integration, including shared-sample corrections."""
import argparse
import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from minppi_mofs.exact_minpi import pivotal_rates


def sample_correction(votes, members, replacement):
    inside = votes[:, members]
    base = inside.sum(axis=1, dtype=np.int32)
    peers = base[:, None] - inside + votes[:, replacement, None]
    direct = np.sign(peers + inside) - np.sign(peers)
    provisional = np.sign(base[:, None] + inside) - np.sign(base[:, None])
    return ((base == -1) | (base == 0), (base == 1) | (base == 0),
            (direct - provisional).sum(axis=0))


def sampled_rates(votes, tiers, draws, seed):
    n, p = votes.shape
    positive, negative = votes > 0, votes < 0
    rates = []
    for k in tiers:
        if k == 0:
            rates.append(votes.mean(axis=0))
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, k]))
        plus = np.zeros(n)
        minus = np.zeros(n)
        correction = np.zeros(p)
        for _ in range(draws):
            order = rng.choice(p, size=k+1, replace=False)
            members = order[:-1]
            plus_inc, minus_inc, correction_inc = sample_correction(votes, members, order[-1])
            plus += plus_inc
            minus += minus_inc
            correction[members] += correction_inc
        rates.append((plus @ positive - minus @ negative + correction) / (n*draws))
    return np.stack(rates)


def replacement_rates(votes, tiers):
    """Independent draws from the entire row, including the candidate feature."""
    positive, negative = votes > 0, votes < 0
    plus = positive.mean(axis=1, keepdims=True)
    minus = negative.mean(axis=1, keepdims=True)
    zero = (votes == 0).mean(axis=1, keepdims=True)
    mass = np.ones((len(votes), 1))
    result = []
    for k in range(max(tiers) + 1):
        if k in tiers:
            at_zero = mass[:, k]
            left = at_zero + mass[:, k-1] if k else at_zero
            right = at_zero + mass[:, k+1] if k else at_zero
            result.append((left @ positive - right @ negative) / len(votes))
        if k != max(tiers):
            next_mass = np.zeros((len(votes), 2*k+3))
            next_mass[:, :-2] += minus * mass
            next_mass[:, 1:-1] += zero * mass
            next_mass[:, 2:] += plus * mass
            mass = next_mass
    return np.stack(result)


def validate():
    rng = np.random.default_rng(43)
    for p in range(2, 8):
        votes = rng.integers(-1, 2, size=(9, p), dtype=np.int8)
        for k in range(1, min(p, 4)):
            all_draws = list(itertools.permutations(range(p), k+1))
            pooled = np.zeros(p)
            for order in all_draws:
                members = np.asarray(order[:-1])
                plus, minus, correction = sample_correction(votes, members, order[-1])
                efficient = plus.astype(int) @ (votes > 0) - minus.astype(int) @ (votes < 0)
                efficient[members] += correction
                direct = []
                for j in range(p):
                    peers = [i if i != j else order[-1] for i in members]
                    margin = votes[:, peers].sum(axis=1)
                    direct.append((np.sign(margin + votes[:, j]) - np.sign(margin)).sum())
                np.testing.assert_array_equal(efficient, direct)
                pooled += efficient
            np.testing.assert_allclose(pooled / (len(votes)*len(all_draws)), pivotal_rates(votes, (k,))[0],
                                       rtol=1e-12, atol=1e-12)
            total = np.zeros(p)
            for order in itertools.product(range(p), repeat=k):
                margin = votes[:, order].sum(axis=1, keepdims=True)
                total += (np.sign(margin+votes) - np.sign(margin)).mean(axis=0)
            np.testing.assert_allclose(total / p**k, replacement_rates(votes, (k,))[0],
                                       rtol=1e-12, atol=1e-12)


def make_matrix(seed, p, abstention):
    rng = np.random.default_rng(np.random.SeedSequence([seed, p, round(abstention*100)]))
    feature_shift = rng.normal(0, .45, size=p)
    witness_shift = rng.normal(0, .8, size=(192, 1))
    positive = rng.random((192, p)) < expit(witness_shift + feature_shift)
    votes = np.where(positive, 1, -1).astype(np.int8)
    votes[rng.random(votes.shape) < abstention] = 0
    return votes


def benchmark(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = None
        for p, abstention, seed in itertools.product(
                (64, 512, 4096, 16384), (.1, .8), (127, 503, 887)):
            votes = make_matrix(seed, p, abstention)
            tiers = (0, 4, 16)
            budgets = (-1, 0, 4, 16, 64, 256)
            rng = np.random.default_rng(np.random.SeedSequence([seed, p, 602]))
            durations = {b: [] for b in budgets}
            scores = {}
            for _ in range(3):
                for b in rng.permutation(budgets):
                    start = time.perf_counter()
                    if b == -1:
                        rates = replacement_rates(votes, tiers)
                    elif b == 0:
                        rates = pivotal_rates(votes, tiers)
                    else:
                        rates = sampled_rates(votes, tiers, b, seed)
                    scores[b] = rates.min(axis=0)
                    durations[b].append(time.perf_counter() - start)
            reference = scores[0]
            count = math.ceil(.1*p)
            top = set(np.argsort(-reference, kind="stable")[:count])
            for b in budgets:
                err = np.abs(scores[b] - reference)
                approx = set(np.argsort(-scores[b], kind="stable")[:count])
                method = {-1: "replacement", 0: "exact"}.get(b, "mc")
                if b == -1:
                    bound = 1 - np.prod(1 - np.arange(1, max(tiers)+1)/p)
                    if err.max() > bound + 1e-12:
                        raise ArithmeticError("Coupling bound violated")
                row = dict(width=p, witnesses=len(votes), abstain_prob=abstention, seed=seed,
                           method=method, draws=max(0,b),
                           median_s=np.median(durations[b]), min_s=min(durations[b]),
                           max_s=max(durations[b]), mean_abs_error=err.mean(), max_abs_error=err.max(),
                           support_overlap=len(top & approx)/count, k=count)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
            stream.flush()
            print(f"p={p} abstention={abstention} seed={seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate()
        print("All shared-draw enumeration checks passed.", flush=True)
        if args.output is not None:
            small = make_matrix(13, 64, .1)
            pivotal_rates(small, (0, 4, 16))
            sampled_rates(small, (0, 4, 16), 4, 13)
            benchmark(args.output)
