"""Pair exact implementations at three feature budgets on observed training panels."""
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
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from experiments.minpi_study_proposal import verified_float_top
from experiments.minpi_study_raw_cost import integer_anchor_pruning, panels, sizes
from minppi_mofs.certified_minpi import _validate, top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale, word_support


BUDGETS = (2, 10, 50)
REPEATS = 5
SCHEDULE_SEED = 73139
METHODS = ("dispatch", "anchor", "proposal")
KEY = ["source", "seed", "budget_pct", "native"]


def native_aware_anchor(votes, h):
    votes = _validate(votes, h)
    w, p = votes.shape
    tiers = sizes(p)
    scale = native_scale(p, tiers, w)
    if scale:
        support, refined = word_support(votes, h, tiers, scale, prune=True)
        return support, {"refined": refined}
    return integer_anchor_pruning(votes, h)


def indices(columns):
    return " ".join(map(str, columns))


def benchmark(output):
    if output.exists():
        raise FileExistsError(output)
    original = pd.read_csv(OUTPUT/"word-design/supports.csv").set_index(["source", "seed"])
    original = original.drop("madelon", level="source")
    rng = np.random.default_rng(SCHEDULE_SEED)
    targets, trials = [], []
    for source, seed, build, _, saved_votes, _ in panels():
        if source == "madelon":
            continue
        votes = build()
        w, p = votes.shape
        tiers = sizes(p)
        native = bool(native_scale(p, tiers, w))
        if saved_votes is not None:
            np.testing.assert_array_equal(votes, saved_votes)
        layers = {k: direct_layer(votes, k) for k in tiers}
        order, denominator, scores = direct_minimum(layers, tiers)
        digest = hashlib.sha256(votes.tobytes()).hexdigest()
        calls = {"dispatch": exact_top_from_votes, "proposal": verified_float_top}
        if not native:
            calls["anchor"] = native_aware_anchor
        cases = {}
        for pct in BUDGETS:
            h = (pct*p + 99)//100
            truth = np.sort(order[:h])
            if pct == 10:
                np.testing.assert_array_equal(truth, np.fromstring(
                    original.loc[(source, seed), "support"], dtype=int, sep=" "))
            float_cols, _ = top_from_votes(votes, h)
            cutoff = scores[order[h - 1]]
            omitted = np.setdiff1d(truth, float_cols)
            added = np.setdiff1d(float_cols, truth)
            gap = Fraction(cutoff - scores[order[h]], w*denominator)
            case = dict(source=source, seed=seed, budget_pct=pct, native=native,
                        w=w, p=p, h=h, tiers=indices(tiers), votes_sha256=digest,
                        gap=str(gap), ties_at_cut=int(np.sum(scores == cutoff)),
                        float_swaps=len(omitted), float_omitted=indices(omitted),
                        float_added=indices(added), support=indices(truth))
            cases[pct] = (h, truth, case)
            targets.append(case)
        tasks = [(pct, name) for pct in BUDGETS for name in calls]
        for repeat in range(-1, REPEATS):
            for position, task_id in enumerate(rng.permutation(len(tasks))):
                pct, method = tasks[task_id]
                h, truth, case = cases[pct]
                start = perf_counter_ns()
                fresh = build()
                middle = perf_counter_ns()
                selected, info = calls[method](fresh, h)
                stop = perf_counter_ns()
                np.testing.assert_array_equal(votes, fresh)
                np.testing.assert_array_equal(truth, selected)
                if method == "proposal":
                    if info["proposal_swaps"] != case["float_swaps"]:
                        raise ValueError("Proposal correction count disagrees with independent float replay")
                    if info["proposal_accepted"] and case["float_swaps"]:
                        raise ValueError("Accepted incorrect floating proposal")
                if repeat < 0:
                    continue
                record = dict(source=source, seed=seed, budget_pct=pct, native=native,
                              repeat=repeat, position=position, kernel=method,
                              build_ms=(middle-start)/1e6, score_ms=(stop-middle)/1e6,
                              select_ms=(stop-start)/1e6, refined=info["refined"],
                              fallback_cols=(info["exact_columns"] if method == "proposal" or
                                             (method == "dispatch" and not native) else 0),
                              proposal_accepted=info.get("proposal_accepted"),
                              proposal_swaps=info.get("proposal_swaps"))
                trials.append(dict(record, method=method))
                if native and method == "dispatch":
                    trials.append(dict(record, method="anchor"))
        print(f"{source} {seed}: {p} columns, three budgets verified", flush=True)
    targets, trials = pd.DataFrame(targets), pd.DataFrame(trials)
    expected = {(src, seed, pct) for src, seed in original.index for pct in BUDGETS}
    if set(targets[["source", "seed", "budget_pct"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete observed panel/budget inventory")
    require_complete(targets, trials)
    output.mkdir(parents=True)
    targets.to_csv(output/"targets.csv", index=False)
    trials.to_csv(output/"trials.csv", index=False)
    paths = ["docs/minpi-study/budget-comparison-protocol.md",
             "configs/minpi-study.json", "experiments/minpi_study.py",
             "experiments/minpi_study_device.py", "experiments/minpi_study_new_cohorts.py",
             "experiments/minpi_study_precision_design.py", "experiments/minpi_study_raw_cost.py",
             "experiments/minpi_study_proposal.py", "experiments/minpi_study_budget_comparison.py",
             "src/minppi_mofs/exact_minpi.py", "src/minppi_mofs/word_minpi.py",
             "src/minppi_mofs/certified_minpi.py", "src/minppi_mofs/integer_minpi.py",
             "results/minpi-study/word-design/supports.csv"]
    manifest = dict(source_sha256={path: file_hash(ROOT/path) for path in paths},
                    system=platform.platform(), processor=os.environ.get("PROCESSOR_IDENTIFIER"),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    libraries=[{key: value for key, value in library.items() if key != "filepath"}
                               for library in threadpool_info()],
                    threads=1, schedule_seed=SCHEDULE_SEED,
                    warmups=1, repeats=REPEATS)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def require_complete(targets, trials):
    if len(targets) != 42*len(BUDGETS) or targets.duplicated(KEY).any():
        raise ValueError("Expected 42 observed targets in each nominal budget stratum")
    if not (targets.groupby(["budget_pct", "native"]).size() == 21).all():
        raise ValueError("Expected 21 observed panels per budget and arithmetic range")
    groups = trials.groupby(KEY + ["method"])
    if (len(groups) != len(targets)*len(METHODS) or trials.duplicated(KEY + ["method", "repeat"]).any()
            or not groups.size().eq(REPEATS).all() or set(trials.method) != set(METHODS)):
        raise ValueError("Expected five calls per target and policy")
    if groups.fallback_cols.nunique().max() != 1 or groups.refined.nunique().max() != 1:
        raise ValueError("Nondeterministic verification work")


def report(output, table):
    targets = pd.read_csv(output/"targets.csv")
    trials = pd.read_csv(output/"trials.csv")
    require_complete(targets, trials)
    agg = trials.groupby(KEY + ["method"]).agg(
        select_ms=("select_ms", "median"), score_ms=("score_ms", "median"),
        build_ms=("build_ms", "median"), refined=("refined", "first"),
        fallback_cols=("fallback_cols", "first"), proposal_accepted=("proposal_accepted", "first"))
    panel = agg.unstack("method")
    panel.columns = [f"{method}_{metric}" for metric, method in panel.columns]
    for stage in ("select_ms", "score_ms"):
        for comparator in ("anchor", "proposal"):
            panel[f"{stage}_vs_{comparator}"] = panel[f"dispatch_{stage}"]/panel[f"{comparator}_{stage}"]
    panel = panel.join(targets.set_index(KEY)[["p", "h", "gap", "float_swaps"]])
    panel.to_csv(output/"paired.csv")
    by = ["budget_pct", "native"]
    ratios = [f"{stage}_vs_{control}" for stage in ("select_ms", "score_ms")
              for control in ("anchor", "proposal")]
    absolute = [f"{method}_select_ms" for method in METHODS]
    by_source = panel.groupby(by + ["source"])[ratios + absolute].median()
    by_source.to_csv(output/"by-source.csv")
    strata = panel.groupby(by)
    summary = strata[ratios + absolute].median()
    summary = summary.join(by_source.groupby(level=by).median().add_prefix("source_median_"))
    summary["panels"] = strata.size()
    summary["zero_gap_panels"] = strata.gap.apply(lambda col: col.eq("0").sum())
    summary["float_changed_panels"] = strata.float_swaps.apply(lambda col: col.gt(0).sum())
    summary["proposal_accepted"] = strata.proposal_proposal_accepted.sum().astype(int)
    for method in ("dispatch", "proposal"):
        fallback = strata[f"{method}_fallback_cols"]
        summary[f"{method}_fallback_cols"] = fallback.sum().astype(int)
        summary[f"{method}_fallback_panels"] = fallback.apply(lambda col: col.gt(0).sum())
    summary.to_csv(output/"summary.csv")
    if table is None:
        return
    rows = []
    for pct in BUDGETS:
        ratios_at_budget = [summary.loc[(pct, native), f"select_ms_vs_{method}"]
                            for native in (True, False) for method in ("anchor", "proposal")]
        rows.append(rf"{pct}\%" + " & " + " & ".join(f"{r:.2f}" for r in ratios_at_budget) + r" \\")
    table.write_text("\n".join([
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Training-to-support time for minPI relative to each exact comparator. "
        r"Within a panel, we divide minPI's median time over five calls by the comparator's "
        r"five-call median. Cells give the median ratio across the 21 panels in each arithmetic "
        r"range. The proposal time includes floating-point selection and exact verification.}",
        r"\label{tab:crossbudget}", r"\begin{tabular}{@{}lrrrr@{}}\toprule",
        r"& \multicolumn{2}{c}{Signed-word range} & \multicolumn{2}{c}{Above signed-word range}\\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Budget & Anchor & Proposal & Anchor & Proposal\\\midrule",
        *rows, r"\bottomrule\end{tabular}", r"\end{table}", ""]), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT/"budget-comparison")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        if not args.report_only:
            benchmark(args.output)
        report(args.output, args.table)

