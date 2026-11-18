"""Same-support timing after removing unnecessary randomization on certified CSR inputs."""
import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from experiments import minpi_constant_transfer as study
from experiments.fast_minpi_five_source import summarize_timings
from experiments.minpi_budget_transfer import SEEDS
from experiments.minpi_study_external import FILES as CHALLENGE
from experiments.stable_minpi_stress import ROOT, digest, file_hash, metric
from minppi_mofs.sparse_exchange_minpi import sparse_exchange_top_from_training


METHODS = ("sparse_exchange", "exchange", "constant", "zero_scan", "F", "random", "full")


def panels():
    for panel in itertools.islice(study.panels(), 6):
        assert panel[0].startswith("farm-ads-")
        yield panel
    p, checksum = CHALLENGE["dexter"]
    path = ROOT/"data/raw/nips2003/dexter.zip"
    assert file_hash(path).lower() == checksum
    halves = []
    with ZipFile(path) as archive:
        for part in ("train", "valid"):
            prefix = f"DEXTER/dexter_{part}"
            label_file = prefix + ".labels" if part == "train" else "dexter_valid.labels"
            y = np.loadtxt(archive.open(label_file), dtype=int)
            columns, values, offsets = [], [], [0]
            with archive.open(prefix + ".data") as stream:
                for line in stream:
                    items = np.fromstring(line.replace(b":", b" "), sep=" ", dtype=int)
                    columns.extend(items[::2] - 1)
                    values.extend(items[1::2])
                    offsets.append(len(values))
            X = csr_matrix((values, columns, offsets), shape=(len(y), p), dtype=np.float32)
            order = np.random.default_rng(391).permutation(p)
            X = X[:, order]
            X.sum_duplicates()
            X.eliminate_zeros()
            X.sort_indices()
            halves.append((X, y))
    (X, y), (Xt, truth) = halves
    for seed in SEEDS:
        yield "dexter", 1, seed, X, y, Xt, truth, "classification", {}


def run(out, verify):
    sources = [Path(__file__), ROOT/"experiments/minpi_constant_transfer.py",
               ROOT/"src/minppi_mofs/sparse_exchange_minpi.py", ROOT/"src/minppi_mofs/exchange_vote_minpi.py",
               ROOT/"experiments/exchange_vote_benchmark.py", ROOT/"configs/minpi-study.json"]
    spec = {"source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources},
            "data": {"farm": study.FILES["farm-ads"], "dexter": CHALLENGE["dexter"][1]},
            "methods": list(METHODS), "repeats": 7, "threads": 1, "layout": "canonical CSR"}
    if verify:
        assert json.loads((out/"protocol.json").read_text()) == spec
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out/"protocol.json").write_text(json.dumps(spec, indent=2), encoding="utf8")
    trials, audit = [], []
    for name, fold, seed, X, y, Xt, truth, task, split in panels():
        assert X.has_canonical_format and not (X.data == 0).any()
        path = out/f"{name}-{fold}-{seed}.npz"
        traces, scores, width = {**split, "truth": truth}, {}, X.shape[1]
        order = np.random.default_rng([4043, seed, fold]).permutation(METHODS)
        for repeat in range(-1, 0 if verify else 7):
            for pos, m in enumerate(METHODS if repeat < 0 else np.roll(order, repeat)):
                start = perf_counter_ns()
                if m == "sparse_exchange":
                    cols, info = sparse_exchange_top_from_training(X, y, random_state=seed)
                else:
                    cols, _ = study.select(m, X, y, task, seed, width, False)
                selected = perf_counter_ns()
                pred = study.fit_predict(X, y, Xt, cols, task, seed, False)
                end = perf_counter_ns()
                if repeat < 0:
                    if m == "sparse_exchange":
                        width = len(cols)
                    traces[m+"_cols"], traces[m+"_pred"] = cols, pred
                    scores[m] = metric(truth, pred, task)
                else:
                    np.testing.assert_array_equal(cols, traces[m+"_cols"])
                    np.testing.assert_array_equal(pred, traces[m+"_pred"])
                    trials.append(dict(dataset=name, fold=fold, seed=seed, method=m, width=len(cols),
                                       p=X.shape[1], repeat=repeat, position=pos, score=scores[m],
                                       select_ms=(selected-start)/1e6, refit_ms=(end-selected)/1e6,
                                       total_ms=(end-start)/1e6))
        for suffix in ("_cols", "_pred"):
            np.testing.assert_array_equal(traces["sparse_exchange"+suffix], traces["exchange"+suffix])
        if verify:
            with np.load(path) as old:
                for key in traces:
                    np.testing.assert_array_equal(traces[key], old[key])
            audit.extend(dict(dataset=name, fold=fold, seed=seed, method=m,
                              support_sha256=digest(traces[m+"_cols"]), prediction_sha256=digest(traces[m+"_pred"]))
                         for m in METHODS)
        else:
            np.savez_compressed(path, **traces)
        print(name, fold, seed, flush=True)
    if verify:
        pd.DataFrame(audit).to_csv(out/"fresh-process-audit.csv", index=False)
    else:
        summarize_timings(out, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
