"""Check exact pivotal ranking and time it on frozen development anchor supports."""

import argparse
from csv import DictWriter
from fractions import Fraction
from itertools import combinations
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.minpi_replication import direct_gain
from experiments.split_vote.run import ROOT, load
from minppi_mofs import split_vote as frozen, split_vote_fast as exact


OUT = ROOT / "thesis-paper/evidence/anchored_vote/fast-ranking"


def oracle(covers, weights, gain, rule):
    """Exhaustively enumerate panels of peer columns and normalize their marginal credits."""
    mass = [Fraction.from_float(float(w)) for w in weights]
    p = covers.shape[1]
    live = list(range(p))
    out = []
    while live:
        sizes = np.unique(
            np.rint(np.array([1 / 6, 0.5, 5 / 6]) * (len(live) - 1)).astype(int)
        )
        normalized = []
        for size in sizes:
            scores = []
            for j in live:
                panels = list(combinations([i for i in live if i != j], size))
                count = sum(
                    m
                    for row, m in zip(covers, mass, strict=True)
                    if row[j]
                    for panel in panels
                    if not any(row[i] for i in panel)
                )
                scores.append(Fraction(count) / len(panels))
            total = sum(scores)
            if total:
                normalized.append([v / total for v in scores])
        scores = [0] * len(live)
        if normalized:
            scores = [
                (min(s) if rule == "min" else sum(s))
                for s in zip(*normalized, strict=True)
            ]
        leader = max(
            range(len(live)), key=lambda i: (scores[i], gain[live[i]], live[i])
        )
        j = live.pop(leader)
        out.append(j)
        mass = [0 if row[j] else m for row, m in zip(covers, mass, strict=True)]
    return out


def validate():
    rng = np.random.default_rng(20261117)
    sketches = 0
    for dtype in (np.float32, np.float64):
        for n in (2, 7, 8, 9, 31, 32, 33, 512, 515):
            x = rng.choice([-2.0, -0.0, 1.0, 3.0, np.nan], size=(n, 8)).astype(dtype)
            x[:, 0] = np.nan
            x[:, 1] = 1.0
            x[:, 2] = np.linspace(0.999999, 1.000001, n).astype(dtype)
            x[:, 3] = x[:, 2]
            for cls in (True, False):
                y = np.arange(n) % 3 if cls else rng.uniform(size=n)
                a = (x, y, rng.uniform(size=8), cls, 42)
                expected = frozen.split_voters(*a)
                actual = exact.split_voters(*a)
                for left, right in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(left, right)
                sketches += 1
    cases = []
    for p in range(1, 8):
        for _ in range(18):
            c = rng.random((11, p)) < rng.uniform(0.1, 0.9)
            c[np.arange(len(c)), rng.integers(p, size=len(c))] = True
            cases.append((c, rng.integers(0, 9, len(c)), rng.integers(0, 3, p)))
    cases.extend(
        [
            (np.eye(5, dtype=bool), np.zeros(5), np.ones(5)),
            (np.ones((4, 7), dtype=bool), np.arange(4), np.ones(7)),
        ]
    )
    # The first feature owns a real dyadic amount below an ulp of the second one's mass.
    tiny = (
        np.array([[1, 0], [1, 0], [0, 1]], bool),
        np.array([1.0, 2.0**-60, 1.0]),
        np.array([0.0, 1.0]),
    )
    cases.append(tiny)
    extreme = (
        np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]], bool),
        np.array([float.fromhex("0x1p-1074"), float.fromhex("0x1p+1023"), 1.0]),
        np.arange(3),
    )
    cases.append(extreme)
    mismatches = {}
    for i, (c, w, g) in enumerate(cases):
        for rule in ("min", "pooled"):
            expected = oracle(c, w, g, rule)
            for prune in (False, True):
                np.testing.assert_array_equal(
                    exact.pivotal_order(c, w, g, rule, prune=prune), expected
                )
            with np.errstate(over="ignore", invalid="ignore"):
                old = frozen.pivotal_order(c, w, g, rule)
            if not np.array_equal(old, expected):
                mismatches[f"{i}-{rule}"] = dict(oracle=expected, frozen=old.tolist())
    return dict(
        cases=len(cases),
        rules=["min", "pooled"],
        exact_oracle_mismatches=0,
        sketches_compared_bit_for_bit=sketches,
        old_disagreements=mismatches,
        sub_ulp_case=len(cases) - 2,
        extreme_case=len(cases) - 1,
    )


def benchmark(name, repeats):
    seed = 42
    src = (
        ROOT
        / "results/anchored_vote/revisit-cpu"
        / f"{name}-{seed}"
        / "min/receipt.json"
    )
    receipt = json.loads(src.read_bytes())
    if receipt["device"] != "cpu":
        raise ValueError("Comparator requires a CPU anchor")
    x, y, task, idx = load(name, seed)
    fit = np.sort(idx["fit"])
    full = direct_gain.fit_native(x[fit], y[fit], task, seed, "cpu")
    booster = full.get_booster()
    rank = direct_gain.gain_order(full, x.shape[1])
    a = receipt["record"]["anchor"]
    anchor = rank[: len(a)]
    assert sorted(anchor) == a
    train = x[np.ix_(fit, anchor)]
    y = y[fit]
    gain = np.array(
        [booster.get_score(importance_type="gain").get(f"f{j}", 0.0) for j in anchor]
    )
    totals = booster.get_score(importance_type="total_gain")
    weights = np.array([totals.get(f"f{j}", 0.0) for j in anchor])

    def sketch(impl):
        build = exact.split_voters if impl == "packed" else frozen.split_voters
        return build(train, y, weights, task == "classification", seed)

    c, w = sketch("frozen")
    fc, fw = sketch("packed")
    np.testing.assert_array_equal(fc, c)
    np.testing.assert_array_equal(fw, w)

    def order(impl, cc, ww, rule):
        if impl == "frozen":
            return frozen.pivotal_order(cc, ww, gain, rule)
        return exact.pivotal_order(
            cc, ww, gain, rule, prune=impl in ("pruned", "packed")
        )

    impls = ["frozen", "integer", "pruned", "packed"]
    rows = []
    for rule in ("min", "pooled"):
        reference = order("frozen", c, w, rule)
        work = {}
        answer = exact.pivotal_order(c, w, gain, rule, work=work)
        np.testing.assert_array_equal(answer, order("integer", c, w, rule))
        for i in range(repeats):
            for j in range(len(impls)):
                impl = impls[(j + i) % len(impls)]
                started = perf_counter()
                actual = order(impl, c, w, rule)
                rank_s = perf_counter() - started
                started = perf_counter()
                cc, ww = sketch(impl)
                recomputed = order(impl, cc, ww, rule)
                total_s = perf_counter() - started
                np.testing.assert_array_equal(actual, recomputed)
                same = bool(np.array_equal(actual, reference))
                rows.append(
                    dict(
                        dataset=name,
                        seed=seed,
                        rule=rule,
                        implementation=impl,
                        repeat=i,
                        columns=len(gain),
                        voters=len(w),
                        degrees=len(np.unique(c.sum(axis=1))),
                        same_frozen=same,
                        exact_min_full=work["possible"],
                        exact_min_after_bound=work["evaluated"],
                        order_s=rank_s,
                        sketch_and_order_s=total_s,
                    )
                )
    return rows


def pipeline_timings(repeats):
    from contextlib import ExitStack
    from unittest.mock import patch
    from experiments.anchored_vote import select as policy
    import pandas as pd

    rows = []
    methods = ["frozen", "packed", "gain"]
    warm_x = np.arange(160, dtype=np.float32).reshape(40, 4)
    warm = direct_gain.fit_native(
        warm_x, np.arange(40) % 2, "classification", 42, "cpu"
    )
    direct_gain.predict_native(warm, warm_x, "classification")
    for i, name in enumerate(("bankruptcy", "forecast", "posture", "turbine")):
        x, y, task, idx = load(name, 42)
        # Archive contains published forecasts, independent of label access here.
        src = ROOT / "results/anchored_vote/revisit-cpu" / f"{name}-42"
        for r in range(repeats):
            for arm in np.roll(methods, (i + r) % len(methods)):
                method = "gain" if arm == "gain" else "min"
                with ExitStack() as stack:
                    if arm == "packed":
                        stack.enter_context(
                            patch.object(policy, "split_voters", exact.split_voters)
                        )
                        stack.enter_context(
                            patch.object(policy, "pivotal_order", exact.pivotal_order)
                        )
                    receipt, pred = policy.select(
                        x, y, task, idx, 42, method, device="cpu"
                    )
                with np.load(src / method / "forecast.npz") as prior:
                    np.testing.assert_array_equal(pred, prior["primary"])
                prior = json.loads((src / method / "receipt.json").read_bytes())[
                    "record"
                ]
                assert all(
                    receipt[k] == prior[k]
                    for k in ("selected", "fits", "anchor", "floor", "utility")
                )
                rows.append(
                    dict(
                        dataset=name,
                        seed=42,
                        repeat=r,
                        arm=arm,
                        retained=receipt["k"],
                        fits=receipt["fits"],
                        seconds=receipt["seconds"],
                    )
                )
        print(name, "pipeline complete", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pipeline-repeats.csv", index=False)
    print(frame.groupby(["dataset", "arm"]).seconds.median().unstack(), flush=True)


def main():
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=19)
    p.add_argument("--cpu", type=int, default=12)
    p.add_argument("--pipeline-only", action="store_true")
    args = p.parse_args()
    import os

    if os.name == "nt":
        import ctypes

        if not ctypes.windll.kernel32.SetProcessAffinityMask(
            ctypes.c_void_p(-1), ctypes.c_size_t(1 << args.cpu)
        ):
            raise RuntimeError("Failed to pin timing process")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.pipeline_only:
        with threadpool_limits(1):
            pipeline_timings(args.repeats)
        return
    checks = validate()
    (OUT / "oracle.json").write_text(json.dumps(checks, indent=2), encoding="utf8")
    print(
        "Exact oracle games:",
        checks["cases"],
        "frozen float disagreements:",
        len(checks["old_disagreements"]),
        flush=True,
    )
    with threadpool_limits(1):
        rows = []
        for name in ("bankruptcy", "forecast", "posture", "turbine"):
            rows.extend(benchmark(name, args.repeats))
            print(name, rows[-1]["columns"], rows[-1]["voters"], flush=True)
    with (OUT / "repeats.csv").open("w", newline="", encoding="utf8") as f:
        writer = DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    q = pd.DataFrame(rows)
    med = q.groupby(["dataset", "rule", "implementation"], as_index=False).agg(
        columns=("columns", "first"),
        voters=("voters", "first"),
        degrees=("degrees", "first"),
        same_frozen=("same_frozen", "all"),
        order_ms=("order_s", "median"),
        rank_ms=("sketch_and_order_s", "median"),
        exact_min_full=("exact_min_full", "first"),
        exact_min_after_bound=("exact_min_after_bound", "first"),
    )
    med[["order_ms", "rank_ms"]] *= 1000
    med.to_csv(OUT / "timing.csv", index=False)
    print(med[med.rule == "min"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
