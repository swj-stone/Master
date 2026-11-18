"""Measure minPI against exact and unverified selectors on matched training panels."""
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
from experiments.minpi_study_budget_comparison import BUDGETS, KEY, REPEATS, SCHEDULE_SEED, native_aware_anchor
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from experiments.minpi_study_proposal import verified_float_top
from experiments.minpi_study_raw_cost import panels, sizes
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale


TARGETS = OUTPUT / "budget-comparison/targets.csv"
POLICIES = {"minpi": exact_top_from_votes, "float": top_from_votes,
            "anchor": native_aware_anchor, "proposal": verified_float_top}
CONTROLS = ("float", "anchor", "proposal")


def benchmark(output):
    if output.exists():
        raise FileExistsError(output)
    previous = pd.read_csv(TARGETS).set_index(["source", "seed", "budget_pct"], verify_integrity=True)
    schedule = np.random.default_rng(SCHEDULE_SEED)
    trials = []
    for source, seed, build, _, saved_votes, _ in panels():
        if source == "madelon":
            continue
        votes = build()
        w, p = votes.shape
        tiers = sizes(p)
        native = bool(native_scale(p, tiers, w))
        digest = hashlib.sha256(votes.tobytes()).hexdigest()
        if saved_votes is not None:
            np.testing.assert_array_equal(votes, saved_votes)
        layers = {k: direct_layer(votes, k) for k in tiers}
        order, _, _ = direct_minimum(layers, tiers)
        truth, proposals = {}, {}
        for pct in BUDGETS:
            target = previous.loc[(source, seed, pct)]
            h = (pct*p + 99)//100
            if (target.votes_sha256 != digest or (target.w, target.p, target.h) != (w, p, h)
                    or native != target.native or target.tiers != " ".join(map(str, tiers))):
                raise ValueError(f"Target changed for {source}/{seed} at budget {pct}")
            truth[pct] = np.sort(order[:h])
            np.testing.assert_array_equal(truth[pct], np.fromstring(target.support, dtype=int, sep=" "))
            proposals[pct], _ = top_from_votes(votes, h)
        policies = [method for method in POLICIES if method != "anchor" or not native]
        tasks = [(method, pct) for method in policies for pct in BUDGETS]
        for repeat in range(-1, REPEATS):
            for position, task in enumerate(schedule.permutation(len(tasks))):
                method, pct = tasks[task]
                h = len(truth[pct])
                start = perf_counter_ns()
                fresh = build()
                middle = perf_counter_ns()
                support, info = POLICIES[method](fresh, h)
                end = perf_counter_ns()
                np.testing.assert_array_equal(fresh, votes)
                np.testing.assert_array_equal(support, proposals[pct] if method == "float" else truth[pct])
                swaps = len(np.setdiff1d(proposals[pct], truth[pct], assume_unique=True))
                if method == "proposal" and info["proposal_swaps"] != swaps:
                    raise ValueError("Inconsistent floating-point correction count")
                if repeat < 0:
                    continue
                row = dict(source=source, seed=seed, budget_pct=pct, native=native, method=method,
                           kernel=method, repeat=repeat, position=position, h=h,
                           build_ns=middle-start, score_ns=end-middle, total_ns=end-start,
                           refined=info["refined"], float_swaps=swaps,
                           proposal_accepted=info.get("proposal_accepted"),
                           fallback_cols=info.get("exact_columns", 0) if not native else 0)
                trials.append(row)
                if method == "minpi" and native:
                    trials.append(dict(row, method="anchor"))
        print(f"{source}/{seed}, {w} x {p}, three budgets", flush=True)
    frame = pd.DataFrame(trials)
    validate(frame)
    keys = frame[["source", "seed", "budget_pct"]].drop_duplicates()
    if set(keys.itertuples(index=False, name=None)) != set(previous.index):
        raise ValueError("Panel inventory differs from archive")
    output.mkdir(parents=True)
    frame.to_csv(output/"trials.csv", index=False)

    previous_manifest = json.loads((TARGETS.parent/"manifest.json").read_text(encoding="utf-8"))
    sources = set(previous_manifest["source_sha256"]) | {
        "docs/minpi-study/arithmetic-price-protocol.md", "experiments/minpi_study_arithmetic_price.py",
        TARGETS.relative_to(ROOT).as_posix()}
    manifest = dict(source_sha256={name: file_hash(ROOT/name) for name in sorted(sources)},
                    system=platform.platform(), processor=os.environ.get("PROCESSOR_IDENTIFIER"),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    libraries=[{k: v for k, v in lib.items() if k != "filepath"} for lib in threadpool_info()],
                    threads=1, warmups=1, repeats=REPEATS, schedule_seed=SCHEDULE_SEED)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate(trials):
    groups = trials.groupby(KEY + ["method"])
    if (len(groups) != 42*len(BUDGETS)*len(POLICIES) or set(trials.method) != set(POLICIES)
            or not groups.size().eq(REPEATS).all() or trials.duplicated(KEY + ["method", "repeat"]).any()):
        raise ValueError("Expected five calls for each of four policies and all 126 targets")
    strata = trials[KEY].drop_duplicates().groupby(["budget_pct", "native"])
    if len(strata) != 6 or not strata.size().eq(21).all():
        raise ValueError("Expected two groups of 21 panels at each budget")
    if not (trials.build_ns + trials.score_ns == trials.total_ns).all():
        raise ValueError("Timing components do not add to total")


def report(output, table):
    trials = pd.read_csv(output/"trials.csv")
    validate(trials)
    medians = trials.groupby(KEY + ["method"])[["build_ns", "score_ns", "total_ns"]].median().unstack()
    panel = medians / 1e6
    panel.columns = [f"{policy}_{phase[:-2]}ms" for phase, policy in panel.columns]
    ratios = []
    for phase in ("total", "score"):
        for control in CONTROLS:
            name = f"{phase}_vs_{control}"
            panel[name] = panel[f"minpi_{phase}_ms"] / panel[f"{control}_{phase}_ms"]
            ratios.append(name)
    panel.to_csv(output/"paired.csv")
    group = ["budget_pct", "native"]
    source = panel.groupby(group + ["source"]).median()
    source.to_csv(output/"by-source.csv")
    strata = panel.groupby(group)[ratios]
    summary = pd.concat([strata.median().add_suffix("_median"), strata.quantile(.25).add_suffix("_q1"),
                         strata.quantile(.75).add_suffix("_q3"), strata.min().add_suffix("_min"),
                         strata.max().add_suffix("_max")], axis=1)
    summary.to_csv(output/"summary.csv")
    if table is not None:
        write_table(summary, table)


def write_table(summary, table):
    body = []
    for pct in BUDGETS:
        cells = [rf"{pct}\%"]
        for native in (True, False):
            for method in CONTROLS:
                stats = summary.loc[(pct, native)]
                key = f"total_vs_{method}"
                median, q1, q3 = (stats[f"{key}_{col}"] for col in ("median", "q1", "q3"))
                if native and method == "anchor":
                    cells.append("1.00")
                else:
                    cells.append(rf"\shortstack[r]{{{median:.2f}\\[-1pt]{{\scriptsize {q1:.2f}--{q3:.2f}}}}}")
        body.append(" & ".join(cells) + r" \\")
    table.write_text("\n".join([
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Time from training arrays to selected columns, minPI divided by comparator. "
        r"Each cell gives the median and interquartile range across 21 panels, using five-call "
        r"medians within panels. minPI and the exact anchor share one call when the scale fits.}",
        r"\label{tab:crossbudget}", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}\toprule",
        r"& \multicolumn{3}{c}{Common scale fits} & \multicolumn{3}{c}{Above common-scale bound}\\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Budget & Pure float & Exact anchor & Verified float & Pure float & Exact anchor & Verified float\\\midrule",
        *body, r"\bottomrule\end{tabular*}", r"\end{table}", ""]), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT/"arithmetic-price")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        if not args.report_only:
            benchmark(args.output)
        report(args.output, args.table)
