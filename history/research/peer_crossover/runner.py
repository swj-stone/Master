"""Time both peer rules and their width-matched controls on registered inputs."""
import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import scipy
from scipy import sparse
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

from research.conditional_dense.original_tasks.regression import regression_support
from research.conditional_dense.vote_width.method import vote_support
from research.lowdim import wide_csr
from research.lowdim.build_native import library_path
from research.wide_blocks.benchmark import compare_arrays, validate_control
from research.wide_blocks.panels import INDEX, KEY, bench, cases, registry


RULES = ("single", "nested")
ARMS = (*RULES, "full", "F_single", "random_single", "F_nested", "random_nested")
HASHES = ("cols_sha", "pred_sha", "decision_sha")
FIELDS = [*KEY, "case_id", "p", "task", "metric", "process", "arm", "timed_arm",
          "repeat", "position", "k", "score", "select_ms", "total_ms", *HASHES]


def deploy(case, arm, width, select_single):
    start = perf_counter_ns()
    if arm == "single":
        cols = select_single(case.selection_X(), case.y, task=case.task, seed=case.seed)[0]
    elif arm == "nested":
        if sparse.issparse(case.X):
            cols = wide_csr.csr_vote_support(case.X, case.y, seed=case.seed)[0]
        else:
            select = regression_support if case.task == "regression" else vote_support
            cols = select(case.selection_X(), case.y, seed=case.seed)[0]
    elif arm == "full":
        cols = np.arange(case.X.shape[1])
    elif arm.startswith("F_"):
        cols = case.select_F(width)
    elif arm.startswith("random_"):
        cols = case.select_random(width)
    else:
        raise ValueError(arm)
    selected = perf_counter_ns()
    pred, decision = case.predict(cols)
    stop = perf_counter_ns()
    return (cols, pred, decision), (selected - start) / 1e6, (stop - start) / 1e6


def warmup(case, old, select_single):
    reference, widths, aliases, quality = {}, {}, {}, {}
    with np.load(case.archive) as archive:
        for arm in ARMS:
            kind, _, rule = arm.partition("_")
            width = widths.get(rule)
            physical = arm
            if rule == "nested" and width == widths["single"]:
                physical = kind + "_single"
            aliases[arm] = physical
            if physical != arm:
                continue
            arrays, _, _ = deploy(case, arm, width, select_single)
            cols, pred, decision = arrays
            np.testing.assert_array_equal(cols, np.unique(cols))
            assert 0 < len(cols) <= old.p and 0 <= cols[0] <= cols[-1] < old.p
            if arm in RULES:
                widths[arm] = len(cols)
            elif rule:
                assert len(cols) == width
            if arm == "nested" or (arm == "single" and old.p <= 32):
                label = "previous" if arm == "nested" else "new"
                np.testing.assert_array_equal(cols, archive[label+"_cols"])
                np.testing.assert_array_equal(pred, archive[label+"_pred"])
                if label+"_decision" in archive:
                    np.testing.assert_array_equal(decision, archive[label+"_decision"])
                np.testing.assert_allclose(case.score(pred, decision), old["score_"+label], rtol=0, atol=2e-12)
            elif arm == "full" or rule:
                validate_control(case, kind, width, arrays, archive, old)
            reference[arm] = arrays
            quality[arm] = dict(k=len(cols), score=float(case.score(pred, decision)),
                                **dict(zip(HASHES, (bench.digest(a) if a is not None else "" for a in arrays))))
            assert np.isfinite(quality[arm]["score"])
    return reference, widths, aliases, quality


def read_quality(path):
    with path.open(newline="", encoding="utf8") as stream:
        return {r["arm"]: (int(r["k"]), float(r["score"]), *(r[f] for f in HASHES))
                for r in csv.DictReader(stream)}


def run_case(case, case_id, old, output, process, repeats, select_single):
    reference, widths, aliases, quality = warmup(case, old, select_single)
    if process == 2:
        previous = output.parent.parent / "pass1" / output.name
        expected = read_quality(previous)
        for arm, physical in aliases.items():
            r = quality[physical]
            if expected[arm] != (r["k"], r["score"], *(r[f] for f in HASHES)):
                raise ValueError(f"Replay mismatch on {case.key}, {arm}")

    order = np.random.default_rng([216712, *case.key[2:]]).permutation(list(reference))
    tmp = output.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for rep in range(repeats):
            shift = rep * len(order) // repeats
            schedule = np.roll(order, shift)
            if process == 2:
                schedule = schedule[::-1]
            for position, arm in enumerate(schedule):
                width = widths.get(arm.partition("_")[2])
                arrays, selection, total = deploy(case, arm, width, select_single)
                compare_arrays(arrays, reference[arm])
                assert 0 <= selection <= total and total > 0
                for label, physical in aliases.items():
                    if arm == physical:
                        writer.writerow(dict(zip(KEY, case.key), case_id=case_id, p=old.p, task=old.task,
                                             metric=old.metric, process=process, arm=label, timed_arm=arm,
                                             repeat=rep, position=position, select_ms=selection,
                                             total_ms=total, **quality[arm]))
    os.replace(tmp, output)


def protocol(repeats):
    from research.peer_crossover.single import prepare

    here = Path(__file__).resolve().parent
    project = here.parent.parent
    paths = [Path(__file__), INDEX, library_path(), Path(prepare()._name),
             *here.glob("*.c"), *here.glob("single*.py")]
    for name in ("research/wide_blocks/panels.py", "research/wide_blocks/benchmark.py",
                 "research/pair_selection/benchmark.py", "research/lowdim/method.py",
                 "research/lowdim/native_votes.c", "research/lowdim/wide_csr.py",
                 "research/conditional_dense/original_tasks/regression.py",
                 "research/conditional_dense/vote_width/method.py", "research/conditional_dense/shared_method.py",
                 "research/conditional_dense/original_tasks/paper_splits.py", "research/feature_game_eval/panels.py",
                 "research/four_stage/benchmark.py", "experiments/minpi_study.py",
                 "research/lowdim/extra_study.py", "research/lowdim/survey_inputs.py", "research/lowdim/text_study.py"):
        paths.append(project/name)
    return dict(repeats=repeats, threads=1, arms=list(ARMS), numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, python=platform.python_version(),
                hashes={p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def run(base, process, repeats, ids, cohorts):
    from research.peer_crossover.single import select_single

    if repeats < 3:
        raise ValueError("Expected at least three timed observations")
    index = registry()
    case_ids = {key: i for i, key in enumerate(index.index)}
    selected = {key for key, i in case_ids.items() if
                (not ids or i in ids) and (not cohorts or key[0] in cohorts)}
    if not selected or (ids and set(ids) != {case_ids[key] for key in selected}):
        raise ValueError("Unregistered or conflicting filter")
    folder = base / f"pass{process}"
    folder.mkdir(parents=True, exist_ok=True)
    state = protocol(repeats)
    manifest = base / "protocol.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf8")) != state:
            raise ValueError("Protocol changed; use a new output directory")
    elif process == 1:
        with manifest.open("x", encoding="utf8") as info:
            json.dump(state, info, indent=2)
    else:
        raise ValueError("A first timing process is required")
    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise RuntimeError("Numerical runtime is not single-threaded")
    pending = {key: folder / f"case-{case_ids[key]:03d}.csv" for key in selected}
    pending = {key: path for key, path in pending.items() if not path.exists()}
    for i, case in enumerate(cases(pending), 1):
        run_case(case, case_ids[case.key], index.loc[case.key], pending[case.key], process, repeats, select_single)
        print(f"pass {process}, {i}/{len(pending)}, {case.key}", flush=True)
    if all((folder / f"case-{i:03d}.csv").is_file() for i in case_ids.values()):
        (folder / "complete.json").write_text(json.dumps(dict(process=process, cases=len(index),
                                                              repeats=repeats)), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--process", type=int, choices=(1, 2), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--cohorts", nargs="+")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.process, args.repeats, args.ids, args.cohorts)
