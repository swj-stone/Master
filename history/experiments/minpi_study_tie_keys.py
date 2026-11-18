"""Compare score-only and index-aware certification of the same binary64 proposal."""
import argparse
import hashlib
import json
import platform
from fractions import Fraction
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, ROOT, file_hash
from experiments.minpi_study_boundary_cost import correction_inputs, oracle
from experiments.minpi_study_budget_comparison import BUDGETS, REPEATS, SCHEDULE_SEED
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from experiments.minpi_study_proposal import verified_float_top
from experiments.minpi_study_raw_cost import panels, sizes
from minppi_mofs.certified_minpi import _validate, top_from_votes
from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.integer_minpi import _bounds, _choose_boundary, _dyadic_anchor, _layers
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale


def endpoint_keys_pass(proposal, outside, lower, upper, threshold):
    """Compare the weakest proposed lower key with the strongest outside upper key."""
    if not len(outside):
        return True
    outside_bounds = upper[outside]
    best = outside_bounds.max()
    if best != threshold:
        return bool(threshold > best)
    last_inside = proposal[lower[proposal] == threshold].max()
    first_outside = outside[outside_bounds == best].min()
    return bool(last_inside < first_outside)


def verified_float_top_keys(votes, count, *, tiers=None):
    """Apply input-order tie breaking at the final acceptance of a proposed set."""
    votes = _validate(votes, count)
    count = int(count)
    p = votes.shape[1]
    tiers = _selection_tiers(tiers, p)
    proposal = np.arange(p) if count == p else top_from_votes(votes, count, tiers=tiers)[0]
    return _verify_indexed_support(votes, count, tiers, proposal)


def verify_supplied_top_keys(votes, count, proposal, *, tiers=None):
    """Certify or correct a supplied set of column indices; return indices in input order."""
    votes = _validate(votes, count)
    count = int(count)
    p = votes.shape[1]
    tiers = _selection_tiers(tiers, p)
    proposal = np.asarray(proposal)
    if proposal.shape != (count,) or proposal.dtype.kind not in "iu":
        raise ValueError("The proposal must contain exactly count integer column indices")
    if proposal.min() < 0 or proposal.max() >= p:
        raise ValueError("Proposed indices must lie in [0,p)")
    proposal = np.sort(proposal.astype(np.intp, copy=False))
    if np.any(proposal[1:] == proposal[:-1]):
        raise ValueError("Proposed column indices must be distinct")
    return _verify_indexed_support(votes, count, tiers, proposal)


def _verify_indexed_support(votes, count, tiers, proposal):
    w, p = votes.shape
    info = dict(arithmetic="all", refined=0, exact_columns=0,
                proposal_accepted=True, proposal_swaps=0)
    if count == p:
        return proposal, info

    common = native_scale(p, tiers, w)
    if common:
        info["arithmetic"] = "int64"
        layers = [(den, signed, np.asarray(signed, dtype=np.int64) * (common // den), 0)
                  for den, signed, _, _ in _layers(votes, tiers, None)]
        lower, upper = _bounds(votes, layers[0])
        silent = None
    else:
        info["arithmetic"] = "dyadic"
        scale = 1 << min(40, 62 - w.bit_length())
        layers = _layers(votes, tiers, scale)
        lower, upper, silent = _dyadic_anchor(votes, layers[0])

    def refine(ids):
        if len(layers) == 1 or not len(ids):
            return
        matrix = np.ascontiguousarray(votes[:, ids])
        for layer in layers[1:]:
            left, right = _bounds(matrix, layer)
            lower[ids] = np.minimum(lower[ids], left)
            upper[ids] = np.minimum(upper[ids], right)

    refine(proposal)
    threshold = lower[proposal].min()
    challengers = upper >= threshold
    challengers[proposal] = False
    outside = np.flatnonzero(challengers)
    refine(outside)
    info["refined"] = count + len(outside)
    if endpoint_keys_pass(proposal, outside, lower, upper, threshold):
        return proposal, info

    info["proposal_accepted"] = False
    eligible = np.sort(np.r_[proposal, outside[upper[outside] >= threshold]])
    low_cut = np.partition(lower[eligible], -count)[-count]
    high_cut = np.partition(upper[eligible], -count)[-count]
    inside = eligible[lower[eligible] > high_cut]
    boundary = eligible[(lower[eligible] <= high_cut) & (upper[eligible] >= low_cut)]
    chosen, info["exact_columns"] = _choose_boundary(
        votes, layers, lower, upper, boundary, count - len(inside), silent)
    result = np.sort(np.r_[inside, chosen])
    info["proposal_swaps"] = len(np.setdiff1d(result, proposal, assume_unique=True))
    return result, info


TARGETS = OUTPUT / "budget-comparison/targets.csv"
METHODS = {"direct": exact_top_from_votes, "strict": verified_float_top, "keys": verified_float_top_keys}


def observed_inputs():
    archived = pd.read_csv(TARGETS, dtype={"gap": str}).set_index(
        ["source", "seed", "budget_pct"], verify_integrity=True)
    for source, seed, build, _, _, _ in panels():
        if source == "madelon":
            continue
        votes = build()
        w, p = votes.shape
        tiers = sizes(p)
        layers = {k: direct_layer(votes, k) for k in tiers}
        order, denom, scores = direct_minimum(layers, tiers)
        digest = hashlib.sha256(votes.tobytes()).hexdigest()
        budgets = []
        for pct in BUDGETS:
            target = archived.loc[(source, seed, pct)]
            h = (pct * p + 99) // 100
            truth = np.sort(order[:h])
            gap = Fraction(scores[order[h - 1]] - scores[order[h]], w * denom)
            np.testing.assert_array_equal(truth, np.fromstring(target.support, dtype=int, sep=" "))
            assert (gap == Fraction(target.gap) and digest == target.votes_sha256 and h == target.h
                    and (w, p) == (target.w, target.p) and target.tiers == " ".join(map(str, tiers)))
            budgets.append((pct, h, truth, gap))
        yield source, str(seed), "observed", votes, tiers, budgets


def measure(inputs, rng, trials, decisions):
    source, seed, scope, votes, tiers, budgets = inputs
    calls = [(method, budget) for method in METHODS for budget in range(len(budgets))]
    diagnostics = {}
    for repeat in range(-1, REPEATS):
        for position, task in enumerate(rng.permutation(len(calls))):
            name, index = calls[task]
            pct, h, truth, gap = budgets[index]
            begin = perf_counter_ns()
            got, info = METHODS[name](votes, h, tiers=tiers)
            elapsed = perf_counter_ns() - begin
            np.testing.assert_array_equal(got, truth)
            obs = dict(source=source, seed=seed, scope=scope, budget=pct, h=h, method=name,
                       arithmetic=info["arithmetic"], refined=int(info["refined"]),
                       exact_columns=int(info["exact_columns"]), accepted=info.get("proposal_accepted"),
                       swaps=info.get("proposal_swaps"))
            if repeat < 0:
                diagnostics[(name, index)] = obs
            else:
                assert obs == diagnostics[(name, index)]
                trials.append(dict(**obs, repeat=repeat, position=position, ns=elapsed))
    for index, (pct, h, truth, gap) in enumerate(budgets):
        strict, keys = [diagnostics[(name, index)] for name in ("strict", "keys")]
        assert not strict["accepted"] or keys["accepted"]
        assert strict["refined"] == keys["refined"]
        floating, _ = top_from_votes(votes, h, tiers=tiers)
        restored = np.setdiff1d(truth, floating)
        removed = np.setdiff1d(floating, truth)
        assert len(restored) == strict["swaps"] == keys["swaps"]
        if len(restored):
            assert not keys["accepted"]
        decisions.append(dict(source=source, seed=seed, scope=scope, budget=pct, h=h,
                              w=len(votes), p=votes.shape[1], gap=str(gap), tiers=" ".join(map(str, tiers)),
                              vote_sha256=hashlib.sha256(votes.tobytes()).hexdigest(),
                              path=keys["arithmetic"], strict_accepted=strict["accepted"],
                              keys_accepted=keys["accepted"], refined=keys["refined"],
                              strict_exact_columns=strict["exact_columns"], keys_exact_columns=keys["exact_columns"],
                              removed=" ".join(map(str, removed)), restored=" ".join(map(str, restored))))


def run(output):
    if output.exists():
        raise FileExistsError(output)
    trials, decisions = [], []
    rng = np.random.default_rng(SCHEDULE_SEED)
    for inputs in observed_inputs():
        measure(inputs, rng, trials, decisions)
    assert len(decisions) == 126
    for name, votes, tiers, h in correction_inputs():
        truth, gap = oracle(votes, tiers, h)
        measure((name, "case", "correction", votes, tiers, [(100*h/votes.shape[1], h, truth, gap)]),
                rng, trials, decisions)
    assert len(trials) == 128 * 3 * REPEATS
    output.mkdir(parents=True)
    calls = pd.DataFrame(trials)
    calls.to_csv(output/"trials.csv", index=False)
    targets = pd.DataFrame(decisions)
    group = ["scope", "source", "seed", "budget", "h"]
    medians = calls.groupby(group + ["method"]).ns.median().unstack()
    targets = targets.set_index(group).join(medians.add_suffix("_ns"))
    targets["keys_over_strict"] = targets.keys_ns / targets.strict_ns
    targets["direct_over_keys"] = targets.direct_ns / targets.keys_ns
    targets.to_csv(output/"targets.csv")
    summary = targets.xs("observed", level="scope").groupby(level="budget").agg(
        panels=("p", "size"), strict_accepts=("strict_accepted", "sum"), key_accepts=("keys_accepted", "sum"),
        strict_exact_columns=("strict_exact_columns", "sum"), keys_exact_columns=("keys_exact_columns", "sum"),
        median_keys_over_strict=("keys_over_strict", "median"),
        median_direct_over_keys=("direct_over_keys", "median"))
    summary.to_csv(output/"summary.csv")
    names = ("docs/minpi-study/tie-keys-protocol.md", "experiments/minpi_study_tie_keys.py",
             "experiments/minpi_study_proposal.py", "experiments/minpi_study_boundary_cost.py",
             "experiments/minpi_study_precision_design.py", "experiments/minpi_study_raw_cost.py",
             "src/minppi_mofs/word_minpi.py", "src/minppi_mofs/certified_minpi.py",
             "src/minppi_mofs/integer_minpi.py", "src/minppi_mofs/exact_minpi.py", TARGETS.relative_to(ROOT).as_posix())
    manifest = dict(threads=1, seed=SCHEDULE_SEED, warmups=1, repeats=REPEATS, python=platform.python_version(),
                    numpy=np.__version__, platform=platform.platform(), source_sha256={n:file_hash(ROOT/n) for n in names})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    with threadpool_limits(1):
        run(cli.parse_args().output)
