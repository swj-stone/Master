"""Verify guarded native minPI, including reference supports and fallback reasons per panel."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import adaptive_minpi_transfer as transfer
from experiments import native_pivotal_transfer as native
from experiments.stable_minpi_stress import ROOT, file_hash, prediction
from minppi_mofs.sketch_minpi import reduced_width
from minppi_mofs.stable_native_minpi import stable_native_top_from_training


BASE = ("stable_auto", "random_stable", "F_stable", "constant", "full")
FIXED = ("stable_75", "moment_75", "random_75", "F_75")


def methods(panel):
    return BASE + FIXED if panel.family.startswith("previous") else BASE


def candidate(panel, method, backend):
    count = None if method == "stable_auto" else reduced_width(panel.X.shape[1], .75)
    return stable_native_top_from_training(panel.X, panel.y, count, task=panel.task, random_state=panel.seed,
                                           groups=panel.groups, ordered_groups=panel.ordered, backend=backend)


def select(method, panel, widths):
    if method in ("stable_auto", "stable_75"):
        return candidate(panel, method, "native")[0]
    return native.select(method, panel, widths)


def protocol():
    state = native.protocol()
    for path in (Path(__file__), ROOT/"src/minppi_mofs/stable_native_minpi.py"):
        state["hashes"][path.relative_to(ROOT).as_posix()] = file_hash(path)
    return dict(state, methods=list(BASE), previous_panels_extra_methods=list(FIXED),
                implementation_reference="Cross-check every tested adaptive/fixed support AND the "
                                         "downstream predictions against backend=numpy. Save the actual "
                                         "selection backend and fallback reason per frozen case.")


def audit_guard(out, verify):
    cases = pd.read_csv(out/"cases.csv").set_index(["dataset", "fold", "seed", "method"])
    rows = []
    for panel in transfer.panels():
        key = panel.name, panel.fold, panel.seed
        with np.load(out/f"{panel.name}-{panel.fold}-{panel.seed}.npz") as trace:
            for arm in ("stable_auto", "stable_75") if panel.family.startswith("previous") else ("stable_auto",):
                cols, info = candidate(panel, arm, "native")
                reference, _ = candidate(panel, arm, "numpy")
                np.testing.assert_array_equal(cols, trace[arm+"_cols"])
                np.testing.assert_array_equal(cols, reference, err_msg=f"{key} {arm}")
                pred = prediction(panel.X, panel.y, panel.Xt, reference, panel.task, panel.seed)
                np.testing.assert_array_equal(pred, trace[arm+"_pred"])
                row = dict(dataset=panel.name, fold=panel.fold, seed=panel.seed, method=arm,
                           width=len(cols), selection_backend=info["selection_backend"],
                           guard_reason=info["guard_reason"], fallback=info["guard_reason"] != "none",
                           numpy_support_equal=True, numpy_prediction_equal=True)
                if arm == "stable_75":
                    np.testing.assert_array_equal(trace["moment_75_cols"], cols)
                    np.testing.assert_array_equal(trace["moment_75_pred"], pred)
                    with np.load(transfer.OLD/f"{panel.name}.npz") as old:
                        for what in ("cols", "pred"):
                            np.testing.assert_array_equal(trace["moment_75_" + what],
                                                          old[f"moment_75_{panel.seed}_{what}"])
                    ref, model = cases.loc[(*key, "moment_75")], cases.loc[(*key, arm)]
                    row.update(moment_support_equal=True, select_ratio=model.select_ms/ref.select_ms,
                               total_ratio=model.total_ms/ref.total_ms)
                rows.append(row)
    result = pd.DataFrame(rows)
    assert len(result) == 69
    audit = out/"guard-cases.csv"
    if verify:
        prior = pd.read_csv(audit, keep_default_na=False)
        cols = ["dataset", "fold", "seed", "method", "width", "selection_backend", "guard_reason",
                "fallback", "numpy_support_equal", "numpy_prediction_equal"]
        pd.testing.assert_frame_equal(result[cols], prior[cols], check_dtype=False)
    else:
        result.to_csv(audit, index=False)
        result.groupby(["dataset", "method"]).agg(configs=("fold", "size"),
                                                  fallbacks=("fallback", "sum")).to_csv(out/"guard-by-panel.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        transfer.run(args.output, args.verify, make_protocol=protocol, get_methods=methods,
                     chooser=select, adaptive_names={"stable_auto": "stable"})
        audit_guard(args.output, args.verify)
