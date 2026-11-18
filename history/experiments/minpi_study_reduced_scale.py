"""Measure observed-denominator reduction against the direct exact selector."""
import argparse
import hashlib
import json
import os
import platform
from fractions import Fraction
from math import lcm
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.minpi_study import OUTPUT, ROOT, file_hash
from experiments.minpi_study_boundary_cost import correction_inputs
from experiments.minpi_study_budget_comparison import BUDGETS, REPEATS, SCHEDULE_SEED
from experiments.minpi_study_frontier import stress_case
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from experiments.minpi_study_raw_cost import panels, sizes
from experiments.minpi_study_reduced_word import WORD_MAX, reduced_top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes


METHODS = {"direct": exact_top_from_votes, "reduced": reduced_top_from_votes}
TARGETS = OUTPUT / "budget-comparison/targets.csv"
WITNESS_REPEATS = 32


def digest(votes):
    return hashlib.sha256(votes.tobytes(order="C")).hexdigest()


def arithmetic_audit(votes, tiers, include_fractions=False):
    """Find minimal scales independently, enumerating one source row per composition."""
    positive = (votes == 1).sum(axis=1)
    negative = (votes == -1).sum(axis=1)
    _, representatives = np.unique(np.column_stack((positive, negative)), axis=0, return_index=True)
    layers = []
    fractions = []
    for k in tiers:
        reduced = 1
        for index in representatives:
            row = votes[index]
            denominator, contributions = direct_layer(row[None, :], k)
            for sign in (-1, 1):
                active = np.flatnonzero(row == sign)
                if active.size:
                    numerator = contributions[active[0]]
                    weight = Fraction(numerator, denominator)
                    reduced = lcm(reduced, weight.denominator)
                    if include_fractions:
                        assert (contributions[active] == numerator).all()
                        fractions.append(dict(k=k, plus=int(positive[index]), minus=int(negative[index]),
                                              sign=sign, pivot=str(weight)))
        layers.append(dict(k=k, raw=denominator, reduced=reduced, gcd=denominator // reduced))
    w = len(votes)
    common = lcm(*(layer["raw"] for layer in layers))
    reduced_common = lcm(*(layer["reduced"] for layer in layers))
    scales = dict(raw_common=common, reduced_common=reduced_common,
                  raw_layer=max(layer["raw"] for layer in layers),
                  reduced_layer=max(layer["reduced"] for layer in layers))
    info = dict(row_compositions=len(representatives))
    for name, value in scales.items():
        info[name] = str(value)
        info[name + "_bits"] = (w * value).bit_length()
        info[name + "_fits"] = w * value <= WORD_MAX
    return info, layers, fractions


def oracle(votes, tiers, scales):
    tallies = {k: direct_layer(votes, k) for k in tiers}
    order, denominator, scores = direct_minimum(tallies, tiers)
    rational_layers = []
    for layer in scales:
        raw, nums = tallies[layer["k"]]
        divisor, reduced = layer["gcd"], layer["reduced"]
        assert raw == divisor * reduced and all(n % divisor == 0 for n in nums)
        rational_layers.append([Fraction(n // divisor, reduced) for n in nums])
    reduced_scores = [min(column) for column in zip(*rational_layers)]
    assert all(score == Fraction(n, denominator) for score, n in zip(reduced_scores, scores))
    reduced_order = sorted(range(len(scores)), key=lambda j: (-reduced_scores[j], j))
    np.testing.assert_array_equal(reduced_order, order)
    return order, denominator, scores


def target(votes, order, denominator, scores, h):
    support = np.sort(order[:h])
    gap = Fraction(scores[order[h-1]] - scores[order[h]], len(votes) * denominator)
    return support, dict(h=h, gap=str(gap), support=" ".join(map(str, support)))


def route(info, method):
    if info["raw_common_fits"]:
        return "int64"
    if method == "reduced" and info["reduced_common_fits"]:
        return "reduced_int64"
    return "dyadic"


def measure_pair(votes, build, h, tiers, truth, info, first, key):
    records = []
    for position, method in enumerate((first, "reduced" if first == "direct" else "direct")):
        selector = METHODS[method]
        start = perf_counter_ns()
        if build is None:
            matrix = votes
            made = start
        else:
            matrix = build()
            made = perf_counter_ns()
        selected, diagnostics = selector(matrix, h, tiers=tiers)
        stopped = perf_counter_ns()
        np.testing.assert_array_equal(selected, truth)
        if build is not None:
            np.testing.assert_array_equal(matrix, votes)
        arithmetic = diagnostics["arithmetic"]
        if arithmetic != route(info, method):
            raise AssertionError((key, method, arithmetic, info))
        records.append(dict(**key, method=method, position=position, route=arithmetic,
                            build_ns=made-start, score_ns=stopped-made, total_ns=stopped-start,
                            refined=diagnostics["refined"],
                            bigint_columns=diagnostics["exact_columns"] if arithmetic == "dyadic" else 0))
    return records


def first_methods(rng, repeats):
    methods = list(METHODS) * (repeats // 2)
    if repeats % 2:
        methods.append(rng.choice(list(METHODS)))
    return rng.permutation(methods)


def panel_study(rng):
    archived = pd.read_csv(TARGETS).set_index(["source", "seed", "budget_pct"], verify_integrity=True)
    records, targets, layer_rows, inventories = [], [], [], []
    for source, seed, build, _, saved, _ in panels():
        if source == "madelon":
            continue
        key = dict(source=source, seed=seed)
        votes = build()
        w, p = votes.shape
        tiers = sizes(p)
        identity = dict(w=w, p=p, votes_sha256=digest(votes), tiers=" ".join(map(str, tiers)))
        if saved is not None:
            np.testing.assert_array_equal(votes, saved)
        info, layers, _ = arithmetic_audit(votes, tiers)
        inventories.append(dict(**key, **identity, **info))
        layer_rows.extend(dict(**key, **layer) for layer in layers)
        order, denominator, scores = oracle(votes, tiers, layers)
        inventories[-1]["reduced_layer_order_matches"] = True
        truths = {}
        first = {}
        for pct in BUDGETS:
            truth, result = target(votes, order, denominator, scores, (pct*p+99)//100)
            ref = archived.loc[source, seed, pct]
            assert (w, p, identity["votes_sha256"], result["h"], result["support"]) == (
                ref.w, ref.p, ref.votes_sha256, ref.h, ref.support)
            assert info["raw_common_fits"] == ref.native
            targets.append(dict(**key, budget_pct=pct, **result))
            truths[pct] = truth
            first[pct] = first_methods(rng, REPEATS)
        for repeat in range(-1, REPEATS):
            for order_position, pct in enumerate(rng.permutation(BUDGETS)):
                truth = truths[pct]
                pair_key = dict(**key, budget_pct=pct, h=len(truth), repeat=repeat, budget_position=order_position)
                first_method = first[pct][repeat] if repeat >= 0 else rng.choice(list(METHODS))
                rows = measure_pair(votes, build, len(truth), tiers, truth, info, first_method, pair_key)
                if repeat >= 0:
                    records.extend(rows)
        print(f"{source}/{seed} {route(info, 'reduced')} ({info['reduced_common_bits']} bits)", flush=True)
    targets = pd.DataFrame(targets)
    assert set(targets.set_index(archived.index.names).index) == set(archived.index)
    return {"panels": pd.DataFrame(inventories), "layers": pd.DataFrame(layer_rows),
            "targets": targets, "trials": pd.DataFrame(records)}


def witness_study(rng):
    half = np.r_[np.ones(64, dtype=np.int8), -np.ones(64, dtype=np.int8)]
    cases = [("complement-128", np.stack((half, -half)), (0, 4, 16), 1),
             ("tied-4096", stress_case(4096, "near"), (0, 4, 16), 1), *correction_inputs()]
    records, witnesses = [], {}
    for name, votes, tiers, h in cases:
        info, layers, fractions = arithmetic_audit(votes, tiers, include_fractions=True)
        order, denominator, scores = oracle(votes, tiers, layers)
        truth, result = target(votes, order, denominator, scores, h)
        if name == "complement-128":
            assert np.all(scores == 0) and truth.tolist() == [0]
        elif name == "tied-4096":
            np.testing.assert_array_equal(np.flatnonzero(scores == scores.max()), np.arange(1, 2048))
        elif name == "unique-4096":
            assert (scores == scores.max()).sum() == 1 and truth.tolist() == [1]
        else:
            assert name == "gas-1129" and h == 64 and result["gap"] == "0"
        witnesses[name] = dict(shape=list(votes.shape), tiers=list(tiers), votes_sha256=digest(votes),
                               **result, **info, reduced_layer_order_matches=True,
                               layers=layers, row_fractions=fractions)
        first = first_methods(rng, WITNESS_REPEATS)
        for repeat in range(-1, WITNESS_REPEATS):
            rows = measure_pair(votes, None, h, tiers, truth, info,
                                first[repeat] if repeat >= 0 else rng.choice(list(METHODS)),
                                dict(case=name, h=h, repeat=repeat))
            if repeat >= 0:
                records.extend(rows)
    return witnesses, pd.DataFrame(records)


def summarize(frames, witnesses):
    keys = ["source", "seed", "budget_pct", "h"]
    trials = frames["trials"]
    groups = trials.groupby(keys + ["method"])
    assert len(groups) == 42 * len(BUDGETS) * len(METHODS)
    assert groups.repeat.nunique().eq(REPEATS).all() and groups.size().eq(REPEATS).all()
    assert groups.position.sum().isin([2, 3]).all()
    medians = groups[["score_ns", "total_ns"]].median().unstack("method").div(1e6)
    medians.columns = [f"{method}_{'score' if scope == 'score_ns' else 'training'}_ms"
                       for scope, method in medians.columns]
    for scope in ("score", "training"):
        medians[scope + "_ratio"] = medians["reduced_" + scope + "_ms"] / medians["direct_" + scope + "_ms"]
    paired = medians.reset_index().merge(frames["panels"][["source", "seed", "raw_common_fits", "reduced_common_fits"]],
                                          on=["source", "seed"], validate="many_to_one")
    paired["stratum"] = np.select([paired.raw_common_fits, paired.reduced_common_fits],
                                  ["raw_word", "rerouted"], default="still_dyadic")
    pooled = pd.concat([paired, paired.assign(stratum="all"), paired.loc[~paired.raw_common_fits].assign(stratum="raw_overflow")])
    summary = []
    for (stratum, pct), panel in pooled.groupby(["stratum", "budget_pct"]):
        for scope in ("score", "training"):
            ratio = panel[scope + "_ratio"]
            summary.append(dict(stratum=stratum, budget_pct=pct, scope=scope, panels=len(panel),
                                direct_ms=panel["direct_" + scope + "_ms"].median(),
                                reduced_ms=panel["reduced_" + scope + "_ms"].median(),
                                median=ratio.median(), q1=ratio.quantile(.25), q3=ratio.quantile(.75),
                                minimum=ratio.min(), maximum=ratio.max()))
    stats = []
    for name, frame in witnesses.groupby("case"):
        elapsed = frame.pivot(index="repeat", columns="method", values="score_ns") / 1e6
        assert elapsed.shape == (WITNESS_REPEATS, 2) and not elapsed.isna().any().any()
        assert frame.groupby("method").position.sum().eq(WITNESS_REPEATS//2).all()
        ratio = elapsed.reduced / elapsed.direct
        row = dict(case=name, pairs=len(elapsed), direct_ms=elapsed.direct.median(),
                   reduced_ms=elapsed.reduced.median(), paired_median=ratio.median(),
                   paired_q1=ratio.quantile(.25), paired_q3=ratio.quantile(.75))
        for method, calls in frame.groupby("method"):
            for metric in ("route", "refined", "bigint_columns"):
                assert calls[metric].nunique() == 1
                row[f"{method}_{metric}"] = calls[metric].iloc[0]
        stats.append(row)
    return dict(paired=paired, summary=pd.DataFrame(summary), witness_summary=pd.DataFrame(stats))


def manifest(output):
    sources = ["configs/minpi-study.json", "docs/minpi-study/reduced-scale-protocol.md",
               "experiments/minpi_study.py", "experiments/minpi_study_boundary_cost.py",
               "experiments/minpi_study_budget_comparison.py", "experiments/minpi_study_device.py",
               "experiments/minpi_study_frontier.py", "experiments/minpi_study_new_cohorts.py",
               "experiments/minpi_study_precision_design.py", "experiments/minpi_study_raw_cost.py",
               "experiments/minpi_study_reduced_scale.py", "experiments/minpi_study_reduced_word.py",
               "experiments/audit_minpi_study_exact_targets.py", "tests/test_minpi_reduced_scale.py",
               "src/minppi_mofs/certified_minpi.py", "src/minppi_mofs/exact_minpi.py",
               "src/minppi_mofs/integer_minpi.py", "src/minppi_mofs/word_minpi.py",
               "src/minppi_mofs/piv2.py", "src/minppi_mofs/piv3_compact_filter.py",
               TARGETS.relative_to(ROOT).as_posix()]
    return dict(source_sha256={path: file_hash(ROOT / path) for path in sources},
                evidence_sha256={path.name: file_hash(path) for path in sorted(output.iterdir())},
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, platform=platform.platform(),
                cpu=os.environ.get("PROCESSOR_IDENTIFIER"), threads=1, panel_repeats=REPEATS,
                witness_pairs=WITNESS_REPEATS, warmups=1, schedule_seed=SCHEDULE_SEED,
                libraries=[{k: v for k, v in lib.items() if k != "filepath"} for lib in threadpool_info()])


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SCHEDULE_SEED)
    frames = panel_study(rng)
    witnesses, trials = witness_study(rng)
    frames.update(summarize(frames, trials))
    frames["witness_trials"] = trials
    for name, data in frames.items():
        data.to_csv(output / (name + ".csv"), index=False)
    (output / "witnesses.json").write_text(json.dumps(witnesses, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest(output), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "reduced-scale")
    with threadpool_limits(1):
        run(parser.parse_args().output)
