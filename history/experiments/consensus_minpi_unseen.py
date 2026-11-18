"""Frozen chronological semiconductor and stratified text classification checks."""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.minpi_study import ROOT, file_hash, train
from experiments.minpi_study_chain import dataset, split
from minppi_mofs.consensus_minpi import ConsensusMinPISelector, consensus_top_from_training
from minppi_mofs.local_refit_fast import local_refit_fast_top_from_training
from minppi_mofs.stable_minpi import StableMinPISelector, _impute_validation


def run(out, verify):
    if not verify:
        out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/"src/minppi_mofs/consensus_minpi.py",
               ROOT/"src/minppi_mofs/stable_minpi.py", ROOT/"src/minppi_mofs/environment_minpi.py",
               ROOT/"experiments/minpi_study_chain.py", ROOT/"experiments/minpi_study.py"]
    protocol = dict(seeds=[1129, 2027, 3469], hashes={s.name: file_hash(s) for s in sources},
                    split="SECOM last chronological quarter, weekly training environments; CNAE-9 stratified sample splits",
                    primary="balanced accuracy with 0.02 tolerance, SECOM also average precision; unit ridge",
                    floor="0.02 is a training criterion; fixed-half methods always compress",
                    widths=[.1, .5, .75], threads=1, timing="one run per configuration, descriptive only")
    path = out/"protocol.json"
    if verify:
        assert protocol == json.loads(path.read_text(encoding="utf-8"))
    else:
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = []
    for name in ("secom", "cnae"):
        X, y, task, _, archive_hash, times = dataset(name)
        for seed in protocol["seeds"]:
            tr, te = split(name, y, task, times, seed)
            groups = None if times is None else (times[tr] - times[tr].min()) // np.timedelta64(7, "D")
            kwargs = dict(groups=groups, ordered_groups=times is not None)
            if groups is not None:
                assert times[tr].max() < times[te].min()
            A, B = X[tr], X[te]
            path = out/f"{name}-{seed}.npz"
            saved = {"tr": tr, "te": te, "truth": y[te]}
            methods = ("auto", "fixed_half") if verify else ("auto", "old_auto", "fixed_half", "local_half", "F_half", "full")
            for method in methods:
                start = perf_counter()
                half = ceil(A.shape[1] / 2)
                if method in ("auto", "old_auto"):
                    cls = ConsensusMinPISelector if method == "auto" else StableMinPISelector
                    estimator = cls(task=task, random_state=seed)
                    columns = estimator.fit(A, y[tr], **kwargs).get_support(indices=True)
                elif method == "fixed_half":
                    columns, _ = consensus_top_from_training(A, y[tr], half, task=task, random_state=seed, **kwargs)
                elif method == "full":
                    columns = np.arange(A.shape[1])
                elif method == "F_half":
                    fit_imputed, _ = _impute_validation(A, A[:0])
                    columns = choose("f", fit_imputed, y[tr], task, seed, half)
                else:
                    columns, _ = local_refit_fast_top_from_training(A, y[tr], half, random_state=seed)
                selected = perf_counter()
                fit, test = (A, B) if len(columns)==A.shape[1] else (A[:, columns], B[:, columns])
                fit, test = _impute_validation(fit, test)
                pred, score = train("ridge", fit, y[tr], test, task, seed)
                end = perf_counter()
                saved[method+"_cols"], saved[method+"_pred"] = columns, pred
                quality = balanced_accuracy_score(y[te], pred)
                ap = average_precision_score(y[te], score) if name == "secom" else np.nan
                rows.append(dict(source=name, seed=seed, method=method, width=len(columns), p=A.shape[1],
                                 score=quality, ap=ap, select_ms=1000*(selected-start), total_ms=1000*(end-start),
                                 archive_sha256=archive_hash))
            if verify:
                with np.load(path) as old:
                    for key, value in saved.items():
                        np.testing.assert_array_equal(value, old[key])
            else:
                np.savez_compressed(path, **saved)
            print(name, seed, flush=True)
    pd.DataFrame(rows).to_csv(out / ("audit.csv" if verify else "cases.csv"), index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--verify", action="store_true")
    args = cli.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.verify)
