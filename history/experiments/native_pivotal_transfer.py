"""Evaluate native minPI on the common transfer panels, with a timed Python reference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_transfer as transfer
from experiments import exchange_vote_benchmark as benchmark
from experiments.stable_minpi_stress import ROOT, file_hash
from minppi_mofs.moment_minpi import moment_top_from_training
from minppi_mofs.native_pivotal_minpi import native_top_from_training
from minppi_mofs.sketch_minpi import reduced_width


BASE = ("native_auto", "random_native", "F_native", "constant", "full")
FIXED = ("native_75", "moment_75", "random_75", "F_75")


def methods(panel):
    return BASE + FIXED if panel.family.startswith("previous") else BASE


def select(method, panel, widths):
    if method in ("native_auto", "native_75", "moment_75"):
        count = None if method == "native_auto" else reduced_width(panel.X.shape[1], .75)
        kwargs = dict(task=panel.task, random_state=panel.seed, groups=panel.groups, ordered_groups=panel.ordered)
        if method == "moment_75":
            return moment_top_from_training(panel.X, panel.y, count, **kwargs)[0]
        return native_top_from_training(panel.X, panel.y, count, backend="native", **kwargs)[0]
    if method == "constant":
        return benchmark.constant_columns(panel.X)
    return transfer.select(method, panel, widths)


def protocol():
    state = transfer.protocol()
    files = [Path(__file__), ROOT/"experiments/exchange_vote_benchmark.py",
             ROOT/"src/minppi_mofs/native_pivotal_minpi.py", ROOT/"src/minppi_mofs/native_pivotal_kernel.c",
             ROOT/"experiments/build_native_pivotal.py"]
    manifest = ROOT / "tmp/pivotal_native/build.json"
    build = json.loads(manifest.read_text(encoding="utf-8"))
    # Compilation wall time changes on a clean rebuild without changing the ABI
    # or the deterministic binary. Pin compiler, flags, source and artifact.
    files.append(manifest.parent / build["library"])
    state["hashes"].update({p.relative_to(ROOT).as_posix(): file_hash(p) for p in files})
    return dict(state, methods=list(BASE), previous_panels_extra_methods=list(FIXED),
                native_build={key: build[key] for key in ("library", "compiler", "flags",
                                                          "library_sha256", "source_sha256")},
                native_build_command=r".\thesis-python.ps1 -m experiments.build_native_pivotal",
                implementation_reference="Time moment_75 and compare support, predictions, selection "
                                         "and total time against native_75 in the 12 old cases.")


def compare_reference(out):
    cases = pd.read_csv(out/"cases.csv")
    ref = cases[cases.method == "moment_75"].set_index(["dataset", "fold", "seed"])
    rows = []
    for case in cases[cases.method == "native_75"].itertuples():
        key = case.dataset, case.fold, case.seed
        previous = ref.loc[key]
        with np.load(out/f"{case.dataset}-{case.fold}-{case.seed}.npz") as new:
            with np.load(transfer.OLD/f"{case.dataset}.npz") as old:
                for suffix in ("cols", "pred"):
                    np.testing.assert_array_equal(new["moment_75_"+suffix], old[f"moment_75_{case.seed}_{suffix}"])
            a, b = new["native_75_cols"], new["moment_75_cols"]
            rows.append(dict(dataset=key[0], fold=key[1], seed=key[2],
                             identical_support=np.array_equal(a, b),
                             support_jaccard=len(np.intersect1d(a, b))/len(np.union1d(a, b)),
                             identical_prediction=np.array_equal(new["native_75_pred"], new["moment_75_pred"]),
                             select_ratio=case.select_ms/previous.select_ms,
                             total_ratio=case.total_ms/previous.total_ms))
    pd.DataFrame(rows).to_csv(out/"moment-reference.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        transfer.run(args.output, args.verify, make_protocol=protocol, get_methods=methods,
                     chooser=select, adaptive_names={"native_auto": "native"})
        if not args.verify:
            compare_reference(args.output)
