"""Time separate word accumulators with exact comparisons on the measured panels."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.minpi_study import OUTPUT, ROOT, file_hash
from experiments.minpi_study_budget_comparison import BUDGETS, REPEATS, SCHEDULE_SEED
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from experiments.minpi_study_raw_cost import panels, sizes
from minppi_mofs.layer_word_minpi import layer_word_top_from_votes, layers_fit_word
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale


POLICIES = {"existing": exact_top_from_votes, "separate": layer_word_top_from_votes}
SCOPES = ("training", "fixed")
TARGETS = OUTPUT / "budget-comparison/targets.csv"
PROTOCOL = ROOT / "docs/minpi-study/layer-word-protocol.md"
INDEX = ["source", "seed", "range", "budget_pct", "h", "scope"]


def benchmark(output):
    if output.exists():
        raise FileExistsError(output)
    archived = pd.read_csv(TARGETS).set_index(["source", "seed", "budget_pct"], verify_integrity=True)
    schedule = np.random.default_rng(SCHEDULE_SEED)
    tasks = [(pct, scope) for pct in BUDGETS for scope in SCOPES]
    trials, targets = [], []
    for source, seed, build, _, saved, _ in panels():
        if source == "madelon":
            continue
        votes = build()
        w, p = votes.shape
        tiers = sizes(p)
        if saved is not None:
            np.testing.assert_array_equal(saved, votes)
        word = layers_fit_word(p, tiers, w)
        common = bool(native_scale(p, tiers, w))
        stratum = "common_fits" if common else ("layer_only" if word else "neither")
        digest = hashlib.sha256(votes.tobytes()).hexdigest()
        order, _, _ = direct_minimum({k: direct_layer(votes, k) for k in tiers}, tiers)
        truth = {}
        for pct in BUDGETS:
            h = (pct*p+99)//100
            ref = archived.loc[(source, seed, pct)]
            assert (w, p, h, digest, common) == (ref.w, ref.p, ref.h, ref.votes_sha256, ref.native)
            truth[pct] = np.sort(order[:h])
            np.testing.assert_array_equal(truth[pct], np.fromstring(ref.support, dtype=int, sep=" "))
            targets.append(dict(source=source, seed=seed, range=stratum, budget_pct=pct,
                                w=w, p=p, h=h, votes_sha256=digest, support=ref.support,
                                tiers=" ".join(map(str, tiers))))
        for repeat in range(-1, REPEATS):
            for pair_position, task in enumerate(schedule.permutation(len(tasks))):
                pct, scope = tasks[task]
                h = len(truth[pct])
                for position, method in enumerate(schedule.permutation(list(POLICIES))):
                    select = POLICIES[method]
                    start = perf_counter_ns()
                    matrix = build() if scope == "training" else votes
                    support, info = select(matrix, h)
                    elapsed = perf_counter_ns() - start
                    if scope == "training":
                        np.testing.assert_array_equal(matrix, votes)
                    np.testing.assert_array_equal(support, truth[pct])
                    expected_route = ("layer_int64" if word else "dyadic") if method == "separate" else (
                        "int64" if common else "dyadic")
                    if info["arithmetic"] != expected_route:
                        raise ValueError(f"Unexpected {method} route on {source}/{seed}: {info}")
                    if repeat >= 0:
                        trials.append(dict(source=source, seed=seed, range=stratum,
                                           budget_pct=pct, h=h, scope=scope, method=method,
                                           repeat=repeat, pair_position=pair_position, position=position,
                                           ns=elapsed, arithmetic=info["arithmetic"], refined=info["refined"],
                                           exact_columns=info["exact_columns"]))
        print(f"{source}/{seed} {stratum}", flush=True)
    target_frame = pd.DataFrame(targets)
    assert set(target_frame.set_index(["source", "seed", "budget_pct"]).index) == set(archived.index)
    output.mkdir(parents=True)
    target_frame.to_csv(output/"targets.csv", index=False)
    pd.DataFrame(trials).to_csv(output/"trials.csv", index=False)
    previous = json.loads((TARGETS.parent/"manifest.json").read_text(encoding="utf-8"))
    sources = set(previous["source_sha256"]) | {
        "src/minppi_mofs/layer_word_minpi.py", "experiments/minpi_study_layer_word.py",
        "docs/minpi-study/layer-word-protocol.md", TARGETS.relative_to(ROOT).as_posix()}
    # These two historical sources do not participate in this comparison.
    sources -= {"experiments/minpi_study_proposal.py", "docs/minpi-study/budget-comparison-protocol.md"}
    metadata = dict(source_sha256={path: file_hash(ROOT/path) for path in sorted(sources)},
                    python=platform.python_version(), system=platform.platform(),
                    processor=os.environ.get("PROCESSOR_IDENTIFIER"), numpy=np.__version__,
                    pandas=pd.__version__, threads=1, warmups=1, repeats=REPEATS,
                    schedule_seed=SCHEDULE_SEED,
                    libraries=[{k: v for k, v in lib.items() if k != "filepath"} for lib in threadpool_info()])
    (output/"manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def report(output):
    trials = pd.read_csv(output/"trials.csv")
    groups = trials.groupby(INDEX + ["method"])
    assert len(groups) == 42 * len(BUDGETS) * len(SCOPES) * len(POLICIES)
    assert groups.size().eq(REPEATS).all() and groups.repeat.nunique().eq(REPEATS).all()
    panel = groups.ns.median().unstack("method").div(1e6).rename(
        columns={"existing": "existing_ms", "separate": "separate_ms"})
    panel["separate_over_existing"] = panel.separate_ms / panel.existing_ms
    panel = panel.reset_index()
    panel.to_csv(output/"paired.csv", index=False)
    eligible = panel.loc[panel["range"] != "neither"].assign(range="all_layer_eligible")
    combined = pd.concat([panel, eligible], ignore_index=True)
    group = combined.groupby(["range", "scope", "budget_pct"])
    summary = group.separate_over_existing.agg(["count", "median", "min", "max"])
    summary["q1"] = group.separate_over_existing.quantile(.25)
    summary["q3"] = group.separate_over_existing.quantile(.75)
    summary.to_csv(output/"summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT/"layer-word")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        if not args.report_only:
            benchmark(args.output)
        report(args.output)
