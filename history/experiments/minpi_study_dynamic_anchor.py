"""Compare frozen and updated cutoffs under the same exact anchor ordering."""
import argparse
import heapq
import hashlib
import json
import platform
from math import lcm
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.minpi_study import OUTPUT, ROOT, file_hash
from experiments.minpi_study_budget_comparison import REPEATS, SCHEDULE_SEED
from experiments.minpi_study_tie_keys import observed_inputs
from minppi_mofs.certified_minpi import _validate
from minppi_mofs.exact_minpi import _selection_tiers
from minppi_mofs.integer_minpi import _layers

METHODS = ("static_batch", "static_stream", "dynamic_stream")
INDEX = ["source", "seed", "budget_pct", "arithmetic", "h"]


def select(votes, h, *, tiers=None, method):
    """Return exact input-order support and the number of completed columns."""
    if method not in METHODS:
        raise ValueError(f"Unknown policy: {method}")
    votes = np.asarray(_validate(votes, h).real, dtype=np.int8)
    w, p = votes.shape
    tiers = _selection_tiers(tiers, p)
    if h == p:
        return np.arange(p), 0, "all"
    layers = _layers(votes, tiers, None)
    common = lcm(*(denominator for denominator, *_ in layers))
    native = w * common <= np.iinfo(np.int64).max
    dtype = np.int64 if native else object
    arithmetic = "int64" if native else "python_int"
    weights = []
    for denominator, signed, *_ in layers:
        multiplier = common // denominator
        # Signed ballot indices select 0, +1 or -1 (the last table column).
        weights.append(np.asarray([(0, pos * multiplier, neg * multiplier)
                                   for pos, neg in signed], dtype=dtype))
    row_ids = np.arange(w)[:, None]

    def totals(matrix, layer):
        return layer[row_ids, matrix].sum(axis=0, dtype=dtype)

    anchor = totals(votes, weights[0])
    order = np.argsort(-anchor, kind="stable")
    head = order[:h]
    if len(weights) == 1:
        return np.sort(head), p, arithmetic

    def complete(columns):
        values = anchor[columns].copy()
        selected_votes = votes[:, columns]
        for layer in weights[1:]:
            np.minimum(values, totals(selected_votes, layer), out=values)
        return values

    head_values = complete(head)
    cutoff = min((int(score), -int(j)) for score, j in zip(head_values, head))
    remaining = order[h:]
    first = int(remaining[0])
    if (int(anchor[first]), -first) < cutoff:
        return np.sort(head), h, arithmetic

    if method == "static_batch":
        qualifies = (anchor[remaining] > cutoff[0]) | (
            (anchor[remaining] == cutoff[0]) & (remaining < -cutoff[1]))
        outside = remaining[qualifies]
        ids = np.concatenate((head, outside))
        values = np.concatenate((head_values, complete(outside)))
        best = np.lexsort((ids, -values))[:h]
        return np.sort(ids[best]), len(ids), arithmetic

    heap = [(int(score), -int(j)) for score, j in zip(head_values, head)]
    heapq.heapify(heap)
    completed = h
    for column in remaining:
        j = int(column)
        threshold = heap[0] if method == "dynamic_stream" else cutoff
        if (int(anchor[j]), -j) < threshold:
            break
        score = int(complete([j])[0])
        completed += 1
        if (score, -j) > heap[0]:
            heapq.heapreplace(heap, (score, -j))
    return np.sort([-index for _, index in heap]), completed, arithmetic


def measure(output):
    if output.exists():
        raise FileExistsError(output)
    rows = []
    targets = []
    schedule = np.random.default_rng(SCHEDULE_SEED)
    for source, seed, _, votes, tiers, budgets in observed_inputs():
        digest = hashlib.sha256(votes.tobytes()).hexdigest()
        for pct, h, reference, gap in budgets:
            identity = dict(source=source, seed=seed, budget_pct=pct, h=h)
            counts = {}
            cycle = schedule.permutation(METHODS)
            rotations = schedule.permutation(np.arange(REPEATS) % len(METHODS))
            for repeat, rotation in enumerate([0, *rotations], start=-1):
                for position, method in enumerate(np.roll(cycle, int(rotation))):
                    start = perf_counter_ns()
                    support, n_completed, arithmetic = select(votes, h, tiers=tiers, method=method)
                    elapsed = perf_counter_ns() - start
                    np.testing.assert_array_equal(support, reference)
                    if repeat < 0:
                        counts[method] = n_completed
                    else:
                        assert counts[method] == n_completed
                        rows.append(dict(**identity, arithmetic=arithmetic, method=method, repeat=repeat,
                                         position=position, ns=elapsed, complete=n_completed))
            assert counts["dynamic_stream"] <= counts["static_stream"] == counts["static_batch"]
            targets.append(dict(**identity, arithmetic=arithmetic, w=len(votes), p=votes.shape[1],
                                tiers=" ".join(map(str, tiers)), vote_sha256=digest, gap=str(gap),
                                support=" ".join(map(str, reference)), **counts))
        print(f"{source}/{seed}", flush=True)
    output.mkdir(parents=True)
    pd.DataFrame(targets).to_csv(output/"targets.csv", index=False)
    pd.DataFrame(rows).to_csv(output/"trials.csv", index=False)
    paths = ("experiments/minpi_study_dynamic_anchor.py", "experiments/minpi_study_tie_keys.py",
             "experiments/minpi_study_budget_comparison.py", "experiments/minpi_study_precision_design.py",
             "experiments/minpi_study_raw_cost.py", "src/minppi_mofs/integer_minpi.py",
             "src/minppi_mofs/certified_minpi.py", "src/minppi_mofs/exact_minpi.py",
             "docs/minpi-study/dynamic-anchor-protocol.md", "results/minpi-study/budget-comparison/targets.csv")
    manifest = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    platform=platform.platform(), threads=1, schedule_seed=SCHEDULE_SEED, warmups=1,
                    repeats=REPEATS, source_sha256={path: file_hash(ROOT/path) for path in paths},
                    libraries=[{k:v for k,v in lib.items() if k != "filepath"} for lib in threadpool_info()])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def summarize(output):
    calls = pd.read_csv(output/"trials.csv")
    group = calls.groupby(INDEX + ["method"])
    assert len(group) == 126 * len(METHODS) and group.size().eq(REPEATS).all()
    assert group.repeat.nunique().eq(REPEATS).all() and group.position.nunique().eq(len(METHODS)).all()
    medians = (group.ns.median().unstack() / 1e6).add_suffix("_ms")
    targets = pd.read_csv(output/"targets.csv").set_index(INDEX)
    assert targets.index.is_unique
    paired = targets.join(medians, validate="one_to_one")
    assert len(paired) == len(medians) == 126 and paired.static_batch_ms.notna().all()
    paired["saved_columns"] = paired.static_batch - paired.dynamic_stream
    paired["dynamic_over_batch"] = paired.dynamic_stream_ms / paired.static_batch_ms
    paired["dynamic_over_static_stream"] = paired.dynamic_stream_ms / paired.static_stream_ms
    paired.reset_index().to_csv(output/"paired.csv", index=False)
    summary = []
    for pct, sample in paired.groupby(level="budget_pct"):
        summary.append(dict(budget_pct=pct, panels=len(sample),
                            fewer_columns=int((sample.saved_columns > 0).sum()),
                            total_dynamic=int(sample.dynamic_stream.sum()),
                            total_static=int(sample.static_batch.sum()),
                            largest_saving=int(sample.saved_columns.max()),
                            dynamic_over_batch=sample.dynamic_over_batch.median(),
                            batch_q1=sample.dynamic_over_batch.quantile(.25),
                            batch_q3=sample.dynamic_over_batch.quantile(.75),
                            dynamic_over_static_stream=sample.dynamic_over_static_stream.median(),
                            stream_q1=sample.dynamic_over_static_stream.quantile(.25),
                            stream_q3=sample.dynamic_over_static_stream.quantile(.75)))
    pd.DataFrame(summary).to_csv(output/"summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT/"dynamic-anchor")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        if not args.report_only:
            measure(args.output)
        summarize(args.output)
