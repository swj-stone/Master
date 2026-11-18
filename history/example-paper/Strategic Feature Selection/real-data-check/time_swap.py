"""Time six original folds in two sequential fresh Python workers."""
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from swap_check import HERE, KEY, bench, cross_products, f_support, fast_support, one_swap


ARMS = ("full", "F", "F+swap", "peer", "peer+swap")
FOLDS = {
    ("abalone", 1, 1129), ("abalone", 2, 1129), ("abalone", 3, 1129),
    ("diabetes", 1, 1129), ("diabetes", 2, 2027), ("diabetes", 3, 3469),
}


def timed_fit(panel, width, arm):
    start = perf_counter_ns()
    if arm == "full":
        cols = np.arange(panel.X.shape[1])
    elif arm in ("F", "F+swap"):
        cols = f_support(panel.X, panel.y, panel.task, width)
    else:
        cols, _ = fast_support(panel.X, panel.y, task=panel.task, seed=panel.seed)
    if arm.endswith("+swap"):
        stats = cross_products(panel.X, panel.y, panel.task, panel.seed)
        cols, _ = one_swap(cols, *stats)
    selected = perf_counter_ns()
    pred, _ = bench.fit(panel, cols)
    end = perf_counter_ns()
    return cols, pred, (selected-start)/1e6, (end-start)/1e6


def worker(process):
    reference = pd.read_csv(HERE / "cases.csv").set_index(KEY + ["start"])
    trial_rows = []
    seen = set()
    for cohort, panel, _, _, _ in bench.panels():
        case = (panel.name, panel.fold, panel.seed)
        if cohort != "original" or case not in FOLDS:
            continue
        seen.add(case)
        reference_key = (cohort, *case)
        width = int(reference.loc[reference_key + ("peer",), "k"])
        warm = {}
        for arm in ARMS:
            cols, pred, *_ = timed_fit(panel, width, arm)
            if arm != "full":
                row = reference.loc[reference_key + (arm.split("+")[0],)]
                field = "refined_cols" if arm.endswith("+swap") else "initial_cols"
                np.testing.assert_array_equal(cols, [int(x) for x in row[field].split()])
            warm[arm] = cols, pred
        rng = np.random.default_rng([87613, process, panel.fold, panel.seed])
        order = rng.permutation(ARMS)
        for repeat in range(len(ARMS)):
            for position, arm in enumerate(np.roll(order, repeat)):
                cols, pred, select_ms, total_ms = timed_fit(panel, width, arm)
                np.testing.assert_array_equal(cols, warm[arm][0])
                np.testing.assert_array_equal(pred, warm[arm][1])
                trial_rows.append(dict(dataset=panel.name, fold=panel.fold, seed=panel.seed,
                                       k=len(cols), process=process, repeat=repeat+1, position=position+1,
                                       method=str(arm), select_ms=select_ms, total_ms=total_ms))
        if seen == FOLDS:
            break
    assert seen == FOLDS and len(trial_rows) == len(FOLDS)*len(ARMS)**2
    return trial_rows


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        with threadpool_limits(1):
            print(json.dumps(worker(int(sys.argv[2]))))
    else:
        rows = []
        for process in (1, 2):
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(process)]
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            rows.extend(json.loads(result.stdout))
        pd.DataFrame(rows).to_csv(HERE / "timing-trials.csv", index=False)
