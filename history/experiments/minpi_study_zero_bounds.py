"""Replay the forced-dyadic precision study with point bounds for silent columns."""
import argparse
import hashlib
import importlib.util
import json
import platform
import tempfile
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, ROOT, file_hash
from experiments.minpi_study_precision_design import BITS, direct_layer, direct_minimum, inputs
from minppi_mofs.integer_minpi import rational_top_from_votes


SCHEDULE_SEED = 44317
REPEATS = 3
PRECISION = OUTPUT / "precision-design"
SNAPSHOT = ROOT / "docs/minpi-study/rowwide-source-snapshot.zip"
KERNEL = "src/minppi_mofs/integer_minpi.py"
KEY = ["panel", "seed", "game", "h"]


def old_selector(folder):
    expected = json.loads((PRECISION / "manifest.json").read_text(encoding="utf-8"))[KERNEL]
    with ZipFile(SNAPSHOT) as source:
        kernel = source.read(KERNEL)
    if hashlib.sha256(kernel).hexdigest() != expected:
        raise ValueError("Archived row-wide implementation differs from the precision-study source")
    path = folder / "frozen_integer_minpi.py"
    path.write_bytes(kernel)
    spec = importlib.util.spec_from_file_location("frozen_integer_minpi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rational_top_from_votes


def run(output, baseline):
    targets = pd.read_csv(PRECISION / "targets.csv")
    recorded = pd.read_csv(PRECISION / "counts.csv").set_index(KEY + ["bits"])
    selectors = {"rowwide": baseline, "zero_point": rational_top_from_votes}
    tasks = [(policy, b) for policy in selectors for b in BITS]
    rng = np.random.default_rng(SCHEDULE_SEED)
    counts, trials, panels = [], [], []
    for name, seed, votes in inputs():
        w, p = votes.shape
        subset = targets[(targets.panel == name) & (targets.seed == seed)]
        silent = ~np.any(votes, axis=0)
        panels.append(dict(panel=name, seed=seed, w=w, p=p, silent_columns=int(silent.sum()),
                           votes_sha256=hashlib.sha256(votes.tobytes()).hexdigest()))
        exact_layers = {}
        for game, group in subset.groupby("game", sort=False):
            tiers = tuple(map(int, group.tiers.iloc[0].split()))
            for k in tiers:
                if k not in exact_layers:
                    exact_layers[k] = direct_layer(votes, k)
            order, _, _ = direct_minimum(exact_layers, tiers)
            for target in group.itertuples():
                h = target.h
                truth = np.sort(order[:h])
                np.testing.assert_array_equal(truth, np.fromstring(target.support, dtype=int, sep=" "))
                key = dict(panel=name, seed=seed, game=game, h=h)
                for repeat in range(-1, REPEATS):
                    for position, task in enumerate(rng.permutation(len(tasks))):
                        policy, bits = tasks[task]
                        start = perf_counter_ns()
                        selected, stats = selectors[policy](votes, h, bits=bits, tiers=tiers)
                        elapsed = (perf_counter_ns() - start) / 1e6
                        np.testing.assert_array_equal(selected, truth)
                        if repeat < 0:
                            if policy == "rowwide":
                                old = recorded.loc[(name, seed, game, h, bits)]
                                assert (stats["refined"], stats["exact_columns"]) == (old.refined, old.exact_columns)
                            counts.append(dict(**key, w=w, p=p, policy=policy, **stats, oracle_match=True))
                        else:
                            trials.append(dict(**key, bits=bits, policy=policy, repeat=repeat,
                                               position=position, elapsed_ms=elapsed))
    counts = pd.DataFrame(counts)
    trials = pd.DataFrame(trials)
    assert len(counts) == 2 * len(recorded) and len(trials) == len(counts) * REPEATS
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(panels).to_csv(output / "panels.csv", index=False)
    counts.to_csv(output / "counts.csv", index=False)
    trials.to_csv(output / "trials.csv", index=False)
    medians = trials.groupby(KEY + ["bits", "policy"]).elapsed_ms.median().unstack()
    medians["zero_point_over_rowwide"] = medians.zero_point / medians.rowwide
    medians.to_csv(output / "paired-times.csv")
    files = [KERNEL, "experiments/minpi_study_proposal.py", "experiments/minpi_study_zero_bounds.py",
             "experiments/minpi_study_precision_design.py", "experiments/minpi_study_frontier.py",
             "experiments/minpi_study_raw_cost.py", "docs/minpi-study/rowwide-source-snapshot.zip",
             "docs/minpi-study/zero-bounds-protocol.md", "results/minpi-study/precision-design/targets.csv",
             "results/minpi-study/precision-design/counts.csv"]
    manifest = dict(files={f: file_hash(ROOT / f) for f in files}, schedule_seed=SCHEDULE_SEED,
                    repeats=REPEATS, blas_threads=1, numpy=np.__version__, python=platform.python_version())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    destination = parser.parse_args().output
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory, threadpool_limits(1):
        run(destination, old_selector(Path(directory)))
