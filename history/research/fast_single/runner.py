"""Replay the paper and candidate single-peer selectors with their own matched controls."""
import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from research.peer_crossover import runner as replay
from research.peer_crossover.single import select_single
from research.wide_blocks.benchmark import compare_arrays, validate_control
from research.wide_blocks.panels import KEY, bench, cases, registry


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "peer_crossover/evidence/paired-cases.csv"
ARMS = ("fast", "single", "full", "F_fast", "random_fast", "F_single", "random_single")


def deploy(case, arm, width, select_fast):
    # The original timed region covers imputation, selection, fit and prediction.
    rule = select_fast if arm == "fast" else select_single
    return replay.deploy(case, "single" if arm == "fast" else arm, width, rule)


def paper_check(stats, arm, old):
    for field in ("k", "cols_sha", "pred_sha"):
        if stats[field] != old[f"{field}_{arm}"]:
            raise ValueError(f"Archived {arm} {field} mismatch on {old.name}")
    np.testing.assert_allclose(stats["score"], old[f"score_{arm}"], rtol=0, atol=2e-12)


def warmup(case, original, paper, select_fast):
    reference, quality, widths, aliases = {}, {}, {}, {}
    with np.load(case.archive) as archive:
        for arm in ARMS:
            kind, _, owner = arm.partition("_")
            k = widths.get(owner)
            physical = kind + "_fast" if owner == "single" and k == widths["fast"] else arm
            aliases[arm] = physical
            if arm == physical:
                arrays, _, _ = deploy(case, arm, k, select_fast)
                cols, pred, decision = arrays
                np.testing.assert_array_equal(cols, np.unique(cols))
                if not (0 < len(cols) <= original.p and 0 <= cols[0] <= cols[-1] < original.p):
                    raise ValueError(f"Invalid support for {arm} on {case.key}")
                if arm in ("fast", "single"):
                    widths[arm] = len(cols)
                else:
                    if owner and len(cols) != k:
                        raise ValueError(f"Width mismatch for {arm} on {case.key}")
                    validate_control(case, kind, k, arrays, archive, original)
                score = float(case.score(pred, decision))
                if not np.isfinite(score):
                    raise ValueError(f"Non-finite {arm} score on {case.key}")
                quality[arm] = dict(k=len(cols), score=score, **dict(zip(
                    replay.HASHES, (bench.digest(a) if a is not None else "" for a in arrays))))
                reference[arm] = arrays
            if arm in ("single", "full", "F_single", "random_single"):
                paper_check(quality[physical], arm, paper)
    return reference, quality, widths, aliases


def run_case(case, original, paper, output, process, repeats, select_fast):
    reference, quality, widths, aliases = warmup(case, original, paper, select_fast)
    if process == 2:
        expected = replay.read_quality(output.parent.parent / "pass1" / output.name)
        for arm, physical in aliases.items():
            s = quality[physical]
            if expected[arm] != (s["k"], s["score"], *(s[key] for key in replay.HASHES)):
                raise ValueError(f"Independent processes disagree on {case.key}, {arm}")

    order = np.random.default_rng([216712, *case.key[2:]]).permutation(list(reference))
    pending = output.with_suffix(".tmp")
    with pending.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=replay.FIELDS)
        writer.writeheader()
        for rep in range(repeats):
            schedule = np.roll(order, rep * len(order) // repeats)
            if process == 2:
                schedule = schedule[::-1]
            for position, arm in enumerate(schedule):
                width = widths.get(arm.partition("_")[2])
                arrays, select_ms, total_ms = deploy(case, arm, width, select_fast)
                compare_arrays(arrays, reference[arm])
                if not (0 <= select_ms <= total_ms and total_ms > 0):
                    raise ValueError(f"Invalid timing on {case.key}, {arm}")
                row = dict(zip(KEY, case.key), case_id=paper.case_id, p=original.p,
                           task=original.task, metric=original.metric, process=process,
                           timed_arm=arm, repeat=rep, position=position,
                           select_ms=select_ms, total_ms=total_ms, **quality[arm])
                for label, physical in aliases.items():
                    if arm == physical:
                        writer.writerow(dict(arm=label, **row))
    os.replace(pending, output)


def protocol(repeats, selector):
    state = replay.protocol(repeats)
    state["arms"] = list(ARMS)
    sources = [Path(__file__), Path(selector.__file__), PAPER, *HERE.glob("*.c"), *HERE.glob("*.h")]
    if hasattr(selector, "prepare"):
        sources.append(Path(selector.prepare()._name))
    for source in sources:
        key = source.relative_to(HERE.parents[1]).as_posix()
        state["hashes"][key] = sha256(source.read_bytes()).hexdigest()
    return state


def run(base, process, repeats, ids, cohorts):
    from research.fast_single import selector

    if repeats < 3:
        raise ValueError("At least three timed observations are required")
    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise RuntimeError("Numerical runtime is not single-threaded")
    original = registry()
    paper = pd.read_csv(PAPER, keep_default_na=False, float_precision="round_trip").set_index(KEY)
    if (not paper.index.equals(original.index)
            or not np.array_equal(paper.case_id, np.arange(len(original)))):
        raise ValueError("Paper record IDs differ from the frozen input registry")
    wanted = {key: int(row.case_id) for key, row in paper.iterrows()
              if (not ids or row.case_id in ids) and (not cohorts or key[0] in cohorts)}
    if not wanted or (ids and set(wanted.values()) != set(ids)):
        raise ValueError("Unknown or conflicting case filters")

    folder = base / f"pass{process}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = base / "protocol.json"
    state = protocol(repeats, selector)
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf8")) != state:
            raise ValueError("Source or protocol changed; use a fresh output directory")
    elif process == 1:
        with manifest.open("x", encoding="utf8") as stream:
            json.dump(state, stream, indent=2)
    else:
        raise ValueError("Pass 1 results are required for independent replay")
    pending = {key: folder / f"case-{i:03}.csv" for key, i in wanted.items()}
    pending = {key: path for key, path in pending.items() if not path.exists()}
    for number, case in enumerate(cases(pending), start=1):
        run_case(case, original.loc[case.key], paper.loc[case.key], pending[case.key],
                 process, repeats, selector.select_fast)
        print(f"pass {process}, {number}/{len(pending)}, {case.key}", flush=True)
    if all((folder / f"case-{i:03}.csv").is_file() for i in paper.case_id):
        (folder / "complete.json").write_text(
            json.dumps(dict(process=process, cases=len(paper), repeats=repeats)), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--process", type=int, required=True, choices=(1, 2))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--cohorts", nargs="+")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.process, args.repeats, args.ids, args.cohorts)
