"""Rebuild and time every Table 2 policy in one balanced series."""
import argparse
import hashlib
import json
import os
import platform
from fractions import Fraction
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.minpi_study import OUTPUT, ROOT, file_hash
from experiments.minpi_study_budget_comparison import BUDGETS, KEY, REPEATS, SCHEDULE_SEED, native_aware_anchor
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from experiments.minpi_study_raw_cost import panels, sizes
from experiments.minpi_study_training_verifier import POLICIES, TARGETS
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale


METHODS = {**POLICIES, "anchor": native_aware_anchor}
LABELS = {"float": "Pure float", "anchor": "Exact anchor", "strict": "Float + strict check",
          "indexed": "Float + index check"}


def call_schedule(rng):
    assert len(METHODS) == REPEATS
    orders = {}
    for pct in BUDGETS:
        cycle = rng.permutation(list(METHODS))
        rotations = [np.roll(cycle, offset) for offset in rng.permutation(REPEATS)]
        orders[pct] = [rng.permutation(list(METHODS)), *rotations]
    for iteration in range(REPEATS + 1):
        for block, pct in enumerate(rng.permutation(BUDGETS)):
            for slot, method in enumerate(orders[pct][iteration]):
                yield iteration-1, block, slot, int(pct), str(method)


def benchmark(output):
    if output.exists():
        raise FileExistsError(output)
    previous = pd.read_csv(TARGETS, dtype={"gap": str}).set_index(["source", "seed", "budget_pct"])
    assert previous.index.is_unique
    rng = np.random.default_rng(SCHEDULE_SEED)
    trials, targets = [], []
    for source, seed, build, _, archived_votes, _ in panels():
        if source == "madelon":
            continue
        votes = build()
        w, p = votes.shape
        tiers = sizes(p)
        native = bool(native_scale(p, tiers, w))
        digest = hashlib.sha256(votes.tobytes()).hexdigest()
        if archived_votes is not None:
            np.testing.assert_array_equal(votes, archived_votes)
        order, common, scores = direct_minimum({k: direct_layer(votes,k) for k in tiers}, tiers)
        targets_for_budget = {}
        for pct in BUDGETS:
            h = (pct*p + 99)//100
            truth = np.sort(order[:h])
            gap = Fraction(scores[order[h-1]]-scores[order[h]], w*common)
            old = previous.loc[source, seed, pct]
            assert (old.votes_sha256, old.w, old.p, old.h, old.native) == (digest, w, p, h, native)
            assert old.tiers == " ".join(map(str, tiers)) and Fraction(old.gap) == gap
            np.testing.assert_array_equal(truth, np.fromstring(old.support, sep=" ", dtype=int))
            proposal = POLICIES["float"](votes, h)[0]
            swaps = len(np.setdiff1d(truth, proposal, assume_unique=True))
            assert swaps == old.float_swaps
            targets_for_budget[pct] = (h, truth, proposal, swaps)
            targets.append(dict(source=source, seed=seed, budget_pct=pct, native=native, w=w, p=p, h=h,
                                tiers=old.tiers, votes_sha256=digest, gap=str(gap), support=old.support,
                                float_swaps=swaps))
        calls = dict(METHODS)
        if native:
            # These are two separately timed training-to-return calls of the same implementation.
            calls["anchor"] = exact_top_from_votes
        for repeat, block, slot, pct, method in call_schedule(rng):
            h, truth, proposal, swaps = targets_for_budget[pct]
            select = calls[method]
            start = perf_counter_ns()
            fresh = build()
            middle = perf_counter_ns()
            chosen, info = select(fresh, h)
            stop = perf_counter_ns()
            np.testing.assert_array_equal(fresh, votes)
            np.testing.assert_array_equal(chosen, proposal if method == "float" else truth)
            if method in ("strict","indexed"):
                assert info["proposal_swaps"] == swaps
            if repeat >= 0:
                trials.append(dict(source=source, seed=seed, budget_pct=pct, native=native, method=method,
                                   h=h, repeat=repeat, budget_position=block, method_position=slot,
                                   build_ns=middle-start, score_ns=stop-middle, total_ns=stop-start,
                                   refined=info["refined"], proposal_accepted=info.get("proposal_accepted"),
                                   proposal_swaps=info.get("proposal_swaps")))
        print(f"{source}/{seed}", flush=True)
    target_frame = pd.DataFrame(targets)
    assert set(target_frame.set_index(previous.index.names).index) == set(previous.index)
    output.mkdir(parents=True)
    target_frame.to_csv(output/"targets.csv", index=False)
    pd.DataFrame(trials).to_csv(output/"trials.csv", index=False)
    earlier = json.loads((OUTPUT/"training-verifier/manifest.json").read_text(encoding="utf-8"))
    files = set(earlier["source_sha256"]) | {"experiments/minpi_study_training_table.py",
                                             "docs/minpi-study/training-table-protocol.md"}
    manifest = dict(source_sha256={f:file_hash(ROOT/f) for f in sorted(files)}, threads=1, repeats=REPEATS,
                    warmups=1, schedule_seed=SCHEDULE_SEED, platform=platform.platform(),
                    processor=os.environ.get("PROCESSOR_IDENTIFIER"), python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__,
                    libraries=[{k:v for k,v in l.items() if k != "filepath"} for l in threadpool_info()])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


def aggregate(output):
    calls = pd.read_csv(output/"trials.csv")
    target = pd.read_csv(output/"targets.csv").set_index(KEY)
    assert target.index.is_unique and len(target)==126
    assert target.groupby(["budget_pct","native"]).size().eq(21).all()
    assert set(calls.method)==set(METHODS) and set(calls.repeat)==set(range(REPEATS))
    assert set(calls.method_position)==set(range(REPEATS))
    per_method = calls.groupby(KEY+["method"])
    assert per_method.size().eq(REPEATS).all() and len(per_method)==126*len(METHODS)
    assert not calls.duplicated(KEY+["repeat", "method_position"]).any()
    assert not calls.duplicated(KEY+["method", "repeat"]).any()
    assert not calls.duplicated(KEY+["method","method_position"]).any()
    assert calls.build_ns.add(calls.score_ns).eq(calls.total_ns).all() and calls.total_ns.gt(0).all()
    medians = per_method[["build_ns","score_ns","total_ns"]].median().unstack()/1e6
    medians.columns = [f"{method}_{stage[:-2]}ms" for stage,method in medians.columns]
    paired = target[["h","w","p"]].join(medians, validate="1:1")
    assert paired.notna().all().all()
    for comparator in LABELS:
        paired[f"direct_over_{comparator}"] = paired.direct_total_ms / paired[f"{comparator}_total_ms"]
    paired["indexed_over_strict"] = paired.indexed_total_ms / paired.strict_total_ms
    groups = []
    for pct in BUDGETS:
        subset = paired.xs(pct, level="budget_pct")
        for label, group in [("All",subset), ("Fits",subset.xs(True,level="native")),
                             ("Exceeds",subset.xs(False,level="native"))]:
            stats = group.quantile([.25,.5,.75]).T
            stats = stats.loc[[c for c in group if "_over_" in c or c.endswith("_total_ms")]]
            stats.columns = ["q1","median","q3"]
            groups.append(stats.rename_axis("metric").reset_index().assign(
                budget_pct=pct, group=label, panels=len(group)))
    return paired, pd.concat(groups, ignore_index=True)


def table(summary):
    values = summary.set_index(["group", "budget_pct","metric"])
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Training-to-selection time ratios. Entries give the median [25th, 75th percentiles] "
             r"of within-panel ratios of five-call medians (minPI/comparator). Fits denotes $wD\le2^{63}-1$, "
             r"where exact anchor and minPI use the same kernel in separately timed calls. "
             r"Both checks include generating the floating-point proposal.}",
             r"\label{tab:crossbudget}\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule",
             r"Panels & Comparator & 2\% budget & 10\% budget & 50\% budget\\\midrule"]
    for label in ("All", "Fits", "Exceeds"):
        if label != "All":
            lines.append(r"\addlinespace[2pt]")
        for index, method in enumerate(["float"] if label=="All" else LABELS):
            stats = [values.loc[label, pct, f"direct_over_{method}"] for pct in BUDGETS]
            panel = f"{label} ({int(stats[0].panels)})" if index==0 else ""
            cells = [f"{s['median']:.2f} [{s.q1:.2f}, {s.q3:.2f}]" for s in stats]
            lines.append(" & ".join([panel,LABELS[method],*cells])+r" \\")
    return "\n".join([*lines,r"\bottomrule\end{tabular*}",r"\end{table}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT/"training-verifier/five-arm")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    if not args.report_only:
        with threadpool_limits(1):
            benchmark(args.output)
    panel, summary = aggregate(args.output)
    if not args.report_only:
        panel.to_csv(args.output/"paired.csv")
        summary.to_csv(args.output/"summary.csv", index=False)
    if args.table:
        args.table.write_text(table(summary), encoding="utf-8")
    print(summary.loc[summary.metric.str.startswith("direct_over")].to_string(index=False))
