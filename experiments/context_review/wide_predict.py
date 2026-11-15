"""Run the archived four policies without any outer targets; package full predictions."""
import platform
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import sklearn
import xgboost
from threadpoolctl import threadpool_limits

from experiments.context_review import transport_predict
from experiments.context_review.wide_data import EVIDENCE, ROOT, RUN, SPEC, read
from experiments.context_study.benchmark import store
from experiments.context_study.data import digest


def dependencies():
    original = [
        "configs/context_review/transport_external.json", "experiments/context_review/transport_predict.py",
        "experiments/context_review/transport_data.py", "experiments/context_review/budgeted_guard.py",
        "experiments/context_review/refit_target.py", "experiments/context_review/timing_process.py",
        "experiments/context_study/benchmark.py", "experiments/context_study/data.py",
        "experiments/test/minpi_convergence_metrics.py", "src/minppi_mofs/context_design.py",
        "src/minppi_mofs/piv3_compact_tree.py", "src/minppi_mofs/piv3_compact_witness.py",
        "src/minppi_mofs/utility.py", "experiments/context_review/wide_data.py",
    ]
    files = [SPEC, EVIDENCE / "fold-status.json", EVIDENCE / "sources.json", Path(__file__)]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files + [ROOT / name for name in original]}


def predict():
    files = dependencies()
    saved = RUN / "protocol.json"
    lock = dict(files=files, versions={"python": platform.python_version(), "numpy": np.__version__,
                                       "pandas": pd.__version__, "sklearn": sklearn.__version__,
                                       "xgboost": xgboost.__version__})
    if saved.exists():
        assert read(saved) == lock, "Frozen files changed"
    store(saved, lock)
    sha = digest(saved)
    status = read(EVIDENCE / "fold-status.json")
    sources = read(SPEC)["sources"]
    expected = [f"{name}/fold-{i}" for name, meta in sources.items() for i in range(meta["folds"])]
    assert list(status) == expected
    cpu = transport_predict.bind_logical_cpu()
    receipts = {}
    with patch.object(transport_predict, "RUN", RUN), threadpool_limits(limits=1):
        for ordinal, dest in enumerate(expected):
            if not status[dest]["feasible"]:
                continue
            base = RUN / dest
            layout = read(base / "layout.json")
            assert digest(base / "layout.json") == status[dest]["layout"]
            transport_predict.predict_fold(layout["dataset"], layout["fold"], 1000 + layout["id"],
                                           ordinal % len(transport_predict.ARMS), cpu, sha)
            receipts[dest] = {file: digest(base / file) for file in
                              ("layout.json", "selection.json", "predictions.npz")}
            print(f"Sealed {dest} ({len(receipts)}/{len(expected)} planned)", flush=True)
    package = EVIDENCE / "predictions.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(saved, "protocol.json")
        for dest, inventory in receipts.items():
            for file in inventory:
                archive.write(RUN / dest / file, f"{dest}/{file}")
    store(EVIDENCE / "prediction-lock.json", dict(protocol=sha, dependencies=lock,
          package_sha256=digest(package), folds=receipts, planned=len(expected), feasible=len(receipts)))


if __name__ == "__main__":
    predict()
