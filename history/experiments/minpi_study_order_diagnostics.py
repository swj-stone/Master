"""Test the common-order sufficient condition on the training ballots used in the study."""
import argparse
from bisect import bisect_right
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
from math import ceil, lcm
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, P, load_dataset
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.integer_minpi import _layers


def optimized_range(terms, denom, rows):
    """Return the optimal normalized row-range bound and a realizing positive scale.

    Each term (d, g) with multiplicity c contributes c*|u*d-g|/(2w);
    u=1/(denom*a). The factor two accommodates three-point ranges.
    """
    weights = Counter()
    for (d, g), count in terms.items():
        if d:
            weights[Fraction(g, d)] += count*d
    half = sum(weights.values())
    cumulative = 0
    u = Fraction(1, denom)
    for knot, weight in sorted(weights.items()):
        cumulative += weight
        if 2*cumulative >= half:
            u = knot
            break
    bound = sum(count*abs(u*d-g) for (d, g), count in terms.items()) / (2*rows)
    return Fraction(bound), 1/(denom*u)


def exact_layer(votes, layer, appearances, abstention_free, scale):
    """Evaluate the selected layer in shared integer units and both optimized bounds."""
    denom, signed, _, _ = layer
    totals = np.zeros(votes.shape[1], dtype=object)
    literal_terms, ternary_terms = Counter(), Counter()
    for row, (positive, negative), present in zip(votes, signed, appearances):
        np.add(totals, positive, where=row > 0, out=totals)
        np.add(totals, negative, where=row < 0, out=totals)
        if abstention_free:
            # For a full binary row the published a_r is (A+B)/2, including
            # the paper's zero-weight convention when one sign is absent.
            literal_terms[positive-negative, 2] += 2
        weights = {-1: negative, 0: 0, 1: positive}
        coefficient = 1 if len(present) == 3 else 2
        for lo, hi in combinations(present, 2):
            ternary_terms[weights[hi]-weights[lo], hi-lo] += coefficient
    generalized = optimized_range(ternary_terms, denom, len(votes))
    original = optimized_range(literal_terms, denom, len(votes)) if abstention_free else None
    return totals * (scale//denom), original, generalized


def support(order, h):
    return tuple(sorted(order[:h]))


def signed_pair_counts(sums, bound, w, h):
    """Count strict G0 separations exceeding the exact bound without a p-by-p array."""
    rankings = sorted(sums, reverse=True)
    outside = sorted(rankings[h:])
    # Integer gaps g satisfy g/w > b iff g >= floor(w*b)+1.
    cutoff = (w*bound).__floor__() + 1
    cross = sum(bisect_right(outside, x-cutoff) for x in rankings[:h])
    all_sorted = sorted(sums)
    all_pairs = sum(bisect_right(all_sorted, x-cutoff) for x in sums)
    return cross, all_pairs


def verify_separations(sums, values, bound, w):
    """Check every claimed strict comparison by the largest challenger below its sign cutoff."""
    ascending = np.argsort(sums, kind="stable")
    sorted_sums = np.asarray(sums)[ascending]
    cutoff = (w*bound).__floor__() + 1
    end = np.searchsorted(sorted_sums, np.asarray(sums)-cutoff, side="right")
    eligible = np.flatnonzero(end > 0)
    for scores in values:
        prefix_max = np.maximum.accumulate(scores[ascending])
        assert np.all(scores[eligible] > prefix_max[end[eligible]-1])


def analyze(votes, h, key):
    w, p = votes.shape
    k_values = sorted({min(x, p-1) for x in (0, 4, 16)})
    layers = _layers(votes, k_values, None)
    unit = lcm(*(layer[0] for layer in layers))
    nz = np.count_nonzero(votes == 0, axis=1)
    free = not nz.any()
    appearances = [tuple(int(v) for v in np.unique(row)) for row in votes]
    sums = votes.sum(0).tolist()
    order0 = sorted(range(p), key=sums.__getitem__, reverse=True)
    gap = Fraction(sums[order0[h-1]] - sums[order0[h]], w)
    layer_records, scores, bounds_p, bounds_t = [], [], [], []
    for k, layer in zip(k_values, layers):
        layer_scores, literal, generalized = exact_layer(votes, layer, appearances, free, unit)
        scores.append(layer_scores)
        bounds_t.append(generalized[0])
        if free:
            bounds_p.append(literal[0])
        layer_record = dict(key, k=k, ternary_bound=str(generalized[0]),
                            ternary_scale=str(generalized[1]))
        if free:
            layer_record.update(prop3_bound=str(literal[0]), prop3_scale=str(literal[1]))
        layer_records.append(layer_record)
    ranks = [np.argsort(-values, kind="stable").tolist() for values in scores]
    score_min = np.min(np.asarray(scores, dtype=object), axis=0)
    order_min = np.argsort(-score_min, kind="stable").tolist()
    assert ranks[0] == order0
    all_same = all(support(r, h) == support(order0, h) for r in ranks + [order_min])
    full_same = all(r == order0 for r in ranks + [order_min])
    tern_bound = max(bounds_t)
    tern_cross, tern_pairs = signed_pair_counts(sums, tern_bound, w, h)
    verify_separations(sums, scores + [score_min], tern_bound, w)
    floor = (w*tern_bound).__floor__()
    certified_inside = sum(sums[j]-sums[order0[h]] > floor for j in order0[:h])
    certified_outside = sum(sums[order0[h-1]]-sums[j] > floor for j in order0[h:])
    out = dict(key, w=w, p=p, h=h, no_abstention=free, no_abstention_rows=int(np.sum(nz==0)),
               abstention=float(nz.sum()/(w*p)), g0_gap=float(gap), g0_gap_exact=str(gap),
               min_gap=float(Fraction(score_min[order_min[h-1]]-score_min[order_min[h]], unit*w)),
               ternary_bound=float(tern_bound), ternary_bound_exact=str(tern_bound),
               ternary_pass=gap>tern_bound, ternary_cross_pairs=tern_cross, ternary_all_pairs=tern_pairs,
               ternary_included=certified_inside, ternary_excluded=certified_outside,
               positive_count_profiles=len(set(np.count_nonzero(votes>0, axis=1))),
               cross_pairs=h*(p-h), all_pairs=p*(p-1)//2, all_supports_same=all_same,
               all_rankings_same=full_same, min_g0_same=support(order_min,h)==support(order0,h),
               min_largest_same=support(order_min,h)==support(ranks[-1],h),
               min_g0_order=order_min==order0, min_largest_order=order_min==ranks[-1],
               min_boundary_tie=score_min[order_min[h-1]]==score_min[order_min[h]],
               votes_sha256=sha256(votes.tobytes()).hexdigest())
    if free:
        literal_bound = max(bounds_p)
        literal_cross, literal_pairs = signed_pair_counts(sums, literal_bound, w, h)
        out.update(prop3_bound=float(literal_bound), prop3_bound_exact=str(literal_bound),
                   prop3_pass=gap>literal_bound, prop3_cross_pairs=literal_cross, prop3_all_pairs=literal_pairs)
        assert tern_bound <= literal_bound
    if gap > tern_bound:
        assert all_same
    return out, layer_records, support(order_min, h)


def observed():
    for name in P["primary_sources"]:
        x, y, task, *_ = load_dataset(name)
        for seed in P["formal_seeds"]:
            with np.load(OUTPUT/"matched"/f"formal-{name}-s{seed}.npz") as data:
                tr = data["train_indices"]
                votes = np.sign(witness_margins(x[tr], y[tr], task, seed)).astype(np.int8)
                np.testing.assert_allclose(votes.mean(0), data["vote_scores"], atol=1e-13)
                yield name, seed, votes, tuple(data["exact_support"])
    for name in ("gas", "supercon"):
        for seed in (1129, 2027, 3469):
            with np.load(OUTPUT/"new-cohorts"/f"{name}-{seed}.npz") as data:
                yield name, seed, data["train_votes"], tuple(data["columns_min"])
    with np.load(OUTPUT/"device/records.npz") as data:
        archived_order = np.argsort(-data["rates"].min(0), kind="stable")
        yield "mhealth", 0, data["train_votes"], tuple(sorted(archived_order[:4]))


def verify_small():
    """Compare the claimed pair certificates directly to literal coalition enumeration."""
    rng = np.random.default_rng(971)
    checks = 0
    for p in (3, 5, 7):
        for allowed in ((-1, 1), (-1, 0, 1)):
            for _ in range(10):
                v = rng.choice(allowed, (7, p)).astype(np.int8)
                out, layers, _ = analyze(v, 2, {})
                exact = []
                for layer in layers:
                    k = layer["k"]
                    vals = []
                    for j in range(p):
                        options = list(combinations([i for i in range(p) if i!=j], k))
                        total = 0
                        for peers in options:
                            base = v[:, peers].sum(1)
                            total += int((np.sign(base+v[:,j])-np.sign(base)).sum())
                        vals.append(Fraction(total, len(v)*len(options)))
                    exact.append(vals)
                bound = Fraction(out["ternary_bound_exact"])
                sums = v.sum(0)
                for j in range(p):
                    for i in range(p):
                        if Fraction(int(sums[j]-sums[i]),len(v))>bound:
                            assert all(vals[j]>vals[i] for vals in exact)
                            checks += 1
    return checks


def run(output):
    rows, details = [], []
    for name, seed, votes, archived in observed():
        h = ceil(.1*votes.shape[1])
        key = dict(dataset=name, seed=seed, panel="full")
        case, layers, exact = analyze(votes, h, key)
        case["archived_float_min_swaps"] = len(set(exact)-set(archived))
        rows.append(case)
        details.extend(layers)
        whole = votes[(votes != 0).all(axis=1)]
        if 0 < len(whole) < len(votes):
            case, layers, _ = analyze(whole, h, dict(dataset=name, seed=seed, panel="complete_rows"))
            rows.append(case)
            details.extend(layers)
    control = np.tile(np.r_[np.ones(10), -np.ones(10)].astype(np.int8), (12, 1))
    case, layers, _ = analyze(control, 10, dict(dataset="balanced_control", seed=0, panel="control"))
    assert case["prop3_pass"] and case["all_rankings_same"]
    rows.append(case)
    details.extend(layers)
    checks = verify_small()
    assert checks > 0
    output.mkdir(exist_ok=False, parents=True)
    pd.DataFrame(rows).to_csv(output/"cases.csv", index=False)
    pd.DataFrame(details).to_csv(output/"layers.csv", index=False)
    print(f"{len(rows)} records written; {checks} certified strict pairs checked by explicit subset enumeration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(1):
        run(parser.parse_args().output)
