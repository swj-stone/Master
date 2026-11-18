"""Time direct selection and floating-proposal verification from the same training arrays."""
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
from experiments.minpi_study_arithmetic_price import TARGETS
from experiments.minpi_study_budget_comparison import BUDGETS, KEY, REPEATS, SCHEDULE_SEED
from experiments.minpi_study_precision_design import direct_layer, direct_minimum
from experiments.minpi_study_proposal import verified_float_top
from experiments.minpi_study_raw_cost import panels, sizes
from experiments.minpi_study_table_audit import CONTROLS, GROUPS, summarize
from experiments.minpi_study_tie_keys import verified_float_top_keys
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes, native_scale


POLICIES = {"direct": exact_top_from_votes, "float": top_from_votes,
            "strict": verified_float_top, "indexed": verified_float_top_keys}


def schedule(rng):
    policies = list(POLICIES)
    orders = {}
    for budget in BUDGETS:
        cycle = rng.permutation(policies)
        orders[budget] = [rng.permutation(policies)]
        orders[budget].extend(np.roll(cycle, offset) for offset in rng.permutation(len(cycle)))
        orders[budget].append(rng.permutation(policies))
    for iteration in range(REPEATS + 1):
        for block, budget in enumerate(rng.permutation(BUDGETS)):
            for slot, method in enumerate(orders[budget][iteration]):
                yield iteration - 1, block, slot, int(budget), str(method)


def benchmark(output):
    if output.exists():
        raise FileExistsError(output)
    previous = pd.read_csv(TARGETS, dtype={"gap": str}).set_index(
        ["source", "seed", "budget_pct"], verify_integrity=True)
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
        order, common, scores = direct_minimum({k: direct_layer(votes, k) for k in tiers}, tiers)
        exact, floating = {}, {}
        for pct in BUDGETS:
            old = previous.loc[(source, seed, pct)]
            h = (p*pct + 99)//100
            gap = Fraction(scores[order[h-1]] - scores[order[h]], w*common)
            assert (digest, w, p, h, native) == (old.votes_sha256, old.w, old.p, old.h, old.native)
            assert gap == Fraction(old.gap) and old.tiers == " ".join(map(str, tiers))
            exact[pct] = np.sort(order[:h])
            np.testing.assert_array_equal(exact[pct], np.fromstring(old.support, sep=" ", dtype=int))
            floating[pct] = top_from_votes(votes, h, tiers=tiers)[0]
            swaps = len(np.setdiff1d(exact[pct], floating[pct], assume_unique=True))
            assert swaps == old.float_swaps
            targets.append(dict(source=source, seed=seed, budget_pct=pct, native=native,
                                w=w, p=p, h=h, tiers=old.tiers, votes_sha256=digest, gap=str(gap),
                                support=old.support, float_swaps=swaps))
        for repeat, block, slot, pct, method in schedule(rng):
            select, h = POLICIES[method], len(exact[pct])
            start = perf_counter_ns()
            fresh = build()
            middle = perf_counter_ns()
            selected, info = select(fresh, h, tiers=tiers)
            stop = perf_counter_ns()
            np.testing.assert_array_equal(fresh, votes)
            np.testing.assert_array_equal(selected, floating[pct] if method == "float" else exact[pct])
            if method in ("strict", "indexed"):
                assert info["proposal_swaps"] == len(np.setdiff1d(floating[pct], exact[pct]))
            if repeat >= 0:
                trials.append(dict(source=source, seed=seed, budget_pct=pct, native=native,
                                   method=method, h=h, repeat=repeat, budget_position=block,
                                   method_position=slot, build_ns=middle-start, score_ns=stop-middle,
                                   total_ns=stop-start, proposal_accepted=info.get("proposal_accepted"),
                                   refined=info["refined"], proposal_swaps=info.get("proposal_swaps"),
                                   fallback_cols=info.get("exact_columns", 0) if not native else 0))
        print(f"{source}/{seed}", flush=True)
    target_frame = pd.DataFrame(targets)
    assert set(target_frame.set_index(previous.index.names).index) == set(previous.index)
    output.mkdir(parents=True)
    target_frame.to_csv(output/"targets.csv", index=False)
    pd.DataFrame(trials).to_csv(output/"trials.csv", index=False)
    old_manifest = json.loads((OUTPUT/"arithmetic-price/manifest.json").read_text(encoding="utf-8"))
    files = set(old_manifest["source_sha256"]) | {
        "experiments/minpi_study_tie_keys.py", "experiments/minpi_study_table_audit.py",
        "experiments/minpi_study_training_verifier.py", "docs/minpi-study/training-verifier-protocol.md"}
    manifest = dict(source_sha256={f: file_hash(ROOT/f) for f in sorted(files)},
                    schedule_seed=SCHEDULE_SEED, repeats=REPEATS, warmups=1, threads=1,
                    platform=platform.platform(), processor=os.environ.get("PROCESSOR_IDENTIFIER"),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    libraries=[{k:v for k,v in lib.items() if k != "filepath"} for lib in threadpool_info()])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


def summarize_calls(output):
    calls = pd.read_csv(output/"trials.csv")
    targets = pd.read_csv(output/"targets.csv").set_index(KEY, verify_integrity=True)
    assert len(targets) == 126 and targets.groupby(level=["budget_pct", "native"]).size().eq(21).all()
    groups = calls.groupby(KEY + ["method"])
    assert set(calls.method) == set(POLICIES) and len(groups) == 126*len(POLICIES)
    assert groups.size().eq(REPEATS).all() and not calls.duplicated(KEY+["method", "repeat"]).any()
    assert set(calls.repeat) == set(range(REPEATS)) and set(calls.method_position) == set(range(len(POLICIES)))
    assert calls.groupby(KEY+["method", "method_position"]).size().between(1, 2).all()
    assert not calls.duplicated(KEY+["repeat", "method_position"]).any()
    assert (calls.build_ns+calls.score_ns).eq(calls.total_ns).all() and calls.total_ns.gt(0).all()
    medians = groups[["total_ns", "build_ns", "score_ns"]].median().unstack()/1e6
    medians.columns = [f"{method}_{phase[:-2]}ms" for phase, method in medians.columns]
    panel = targets[["h", "p", "w"]].join(medians, validate="1:1")
    assert panel.notna().all().all()
    for method in ("float", "strict", "indexed"):
        panel[f"direct_over_{method}"] = panel.direct_total_ms / panel[f"{method}_total_ms"]
    panel["indexed_over_strict"] = panel.indexed_total_ms / panel.strict_total_ms
    summary = []
    for pct in BUDGETS:
        at_budget = panel.xs(pct, level="budget_pct")
        for group, subset in [("All", at_budget), ("Fits", at_budget.xs(True, level="native")),
                              ("Exceeds", at_budget.xs(False, level="native"))]:
            for column in ["direct_over_float", "direct_over_strict", "direct_over_indexed",
                           "indexed_over_strict", *(f"{m}_total_ms" for m in POLICIES)]:
                q1, middle, q3 = subset[column].quantile([.25,.5,.75])
                summary.append(dict(budget_pct=pct, group=group, metric=column, panels=len(subset),
                                    median=middle, q1=q1, q3=q3))
    return panel, pd.DataFrame(summary)


def write_table(summary, path):
    prior, _ = summarize()
    indexed = summary.loc[summary.metric.eq("direct_over_indexed")].set_index(["group", "budget_pct"])
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Training-to-selection time ratios. Entries are medians [25th, 75th percentiles] "
        r"of minPI/comparator ratios of five-call panel medians, pooled or split at $wD=2^{63}-1$. "
        r"Below the bound minPI and the exact anchor share each call. The index checks use a "
        r"separate paired run including ballot construction and proposal generation.}",
        r"\label{tab:crossbudget}\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule",
        r"Panels & Comparator & 2\% budget & 10\% budget & 50\% budget\\\midrule"]
    for label, group in GROUPS:
        if group != "all":
            lines.append(r"\addlinespace[2pt]")
        methods = ["float"] if group == "all" else [*CONTROLS, "indexed"]
        for i, method in enumerate(methods):
            values = ([indexed.loc[label, pct] for pct in BUDGETS] if method == "indexed" else
                      [prior[group, method, pct] for pct in BUDGETS])
            cells = [f"{v['median']:.2f} [{v['q1']:.2f}, {v['q3']:.2f}]" for v in values]
            panels = f"{label} ({int(values[0]['panels'])})" if i == 0 else ""
            comparator = "Float + index check" if method == "indexed" else CONTROLS[method]
            lines.append(" & ".join([panels, comparator, *cells]) + r" \\")
    lines.extend([r"\bottomrule\end{tabular*}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT/"training-verifier")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    if not args.report_only:
        with threadpool_limits(1):
            benchmark(args.output)
    panel, summary = summarize_calls(args.output)
    if not args.report_only:
        panel.to_csv(args.output/"paired.csv")
        summary.to_csv(args.output/"summary.csv", index=False)
    if args.table:
        write_table(summary, args.table)
    print(summary.loc[summary.metric.eq("direct_over_indexed")].to_string(index=False))
