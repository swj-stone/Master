"""Time certification of a supplied feature list against exact selection on the same ballots."""
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
from experiments.minpi_study_boundary_cost import PAIRS, correction_inputs, oracle
from experiments.minpi_study_budget_comparison import REPEATS, SCHEDULE_SEED
from experiments.minpi_study_tie_keys import observed_inputs, verify_supplied_top_keys
from minppi_mofs.certified_minpi import top_from_votes
from minppi_mofs.word_minpi import exact_top_from_votes


def measure(source, seed, scope, votes, tiers, pct, h, truth, gap, schedule, trials):
    # A supplied list is the only intermediate result shared between calls.
    # Keep its computation and the independent reference outside both timers.
    proposal = top_from_votes(votes, h, tiers=tiers)[0]
    proposal.setflags(write=False)
    removed = np.setdiff1d(proposal, truth)
    restored = np.setdiff1d(truth, proposal)
    repeats = REPEATS if scope == "observed" else PAIRS
    first = schedule.permutation(np.arange(repeats))
    first = first % 2
    if repeats % 2 and schedule.integers(2):
        first = 1 - first
    calls = {
        "audit": lambda: verify_supplied_top_keys(votes, h, proposal, tiers=tiers),
        "direct": lambda: exact_top_from_votes(votes, h, tiers=tiers),
    }
    identity = dict(scope=scope, source=source, seed=seed, budget_pct=pct, h=h)
    for repeat, first_method in enumerate([0, *first], start=-1):
        methods = ("audit", "direct") if first_method == 0 else ("direct", "audit")
        for position, method in enumerate(methods):
            select = calls[method]
            start = perf_counter_ns()
            selected, info = select()
            elapsed = perf_counter_ns() - start
            np.testing.assert_array_equal(selected, truth)
            if method == "audit":
                assert info["proposal_accepted"] == (not len(removed))
                assert info["proposal_swaps"] == len(removed)
                audit = info
            if repeat >= 0:
                trials.append(dict(**identity, method=method, repeat=repeat, position=position,
                                   ns=elapsed, arithmetic=info["arithmetic"], refined=info["refined"],
                                   exact_columns=info["exact_columns"]))
    return dict(**identity, w=votes.shape[0], p=votes.shape[1], tiers=" ".join(map(str, tiers)),
                vote_sha256=hashlib.sha256(votes.tobytes()).hexdigest(), gap=str(gap),
                proposal=" ".join(map(str, proposal)), reference=" ".join(map(str, truth)),
                accepted=audit["proposal_accepted"], refined=audit["refined"],
                rescored=audit["exact_columns"], arithmetic=audit["arithmetic"],
                removed=" ".join(map(str, removed)), restored=" ".join(map(str, restored)))


def run(output):
    if output.exists():
        raise FileExistsError(output)
    trials, targets = [], []
    rng = np.random.default_rng(SCHEDULE_SEED)
    for source, seed, scope, votes, tiers, budgets in observed_inputs():
        for pct, h, truth, gap in budgets:
            targets.append(measure(source, seed, scope, votes, tiers, pct, h, truth, gap, rng, trials))
        print(f"{source}/{seed}", flush=True)
    for source, votes, tiers, h in correction_inputs():
        truth, gap = oracle(votes, tiers, h)
        targets.append(measure(source, "case", "correction", votes, tiers, 100*h/votes.shape[1],
                               h, truth, gap, rng, trials))
    target_frame = pd.DataFrame(targets)
    calls = pd.DataFrame(trials)
    keys = ["scope", "source", "seed", "budget_pct", "h"]
    groups = calls.groupby(keys + ["method"])
    counts = groups.size()
    assert len(counts) == 128 * 2
    assert counts.xs("observed", level="scope").eq(REPEATS).all()
    assert counts.xs("correction", level="scope").eq(PAIRS).all()
    medians = (groups.ns.median().unstack("method") / 1e6).add_suffix("_ms")
    target_frame = target_frame.set_index(keys, verify_integrity=True).join(medians, how="left", validate="1:1")
    assert target_frame.audit_ms.notna().all()
    target_frame["audit_over_direct"] = target_frame.audit_ms / target_frame.direct_ms
    observed = target_frame.xs("observed", level="scope")
    assert len(observed) == 126 and observed.accepted.all()
    corrected = target_frame.xs("correction", level="scope")
    assert len(corrected) == 2 and not corrected.accepted.any()
    assert dict(zip(corrected.index.get_level_values("source"), corrected.rescored)) == {
        "gas-1129": 3, "unique-4096": 2}
    grouped = observed.groupby(level="budget_pct").audit_over_direct
    summary = grouped.agg(["count", "median", "min", "max"])
    summary["q25"] = grouped.quantile(0.25)
    summary["q75"] = grouped.quantile(0.75)
    summary["audit_faster"] = grouped.agg(lambda ratios: (ratios < 1).sum())
    output.mkdir(parents=True)
    calls.to_csv(output / "trials.csv", index=False)
    target_frame.to_csv(output / "targets.csv")
    summary.to_csv(output / "summary.csv")
    inherited = json.loads((OUTPUT / "tie-keys/manifest.json").read_text(encoding="utf-8"))
    files = set(inherited["source_sha256"]) | {"experiments/minpi_study_supplied_support.py",
        "experiments/minpi_study_budget_comparison.py", "docs/minpi-study/supplied-support-protocol.md"}
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    processor=os.environ.get("PROCESSOR_IDENTIFIER"), numpy=np.__version__,
                    pandas=pd.__version__, threads=1, schedule_seed=SCHEDULE_SEED, warmups=1,
                    observed_pairs=REPEATS, correction_pairs=PAIRS, clock="perf_counter_ns",
                    libraries=[{k:v for k,v in lib.items() if k != "filepath"} for lib in threadpool_info()],
                    source_sha256={path:file_hash(ROOT/path) for path in sorted(files)})
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "supplied-support")
    with threadpool_limits(1):
        run(parser.parse_args().output)
