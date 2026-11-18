"""Archive exact sources and compare every stored array in a fresh development replay."""
import argparse
from importlib import import_module
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.stable_minpi_stress import ROOT, file_hash, write_json


def run(source, replay, module):
    state = json.loads((source / "protocol.json").read_text(encoding="utf8"))
    for name, expected in state["source_sha256"].items():
        assert file_hash(ROOT/name) == expected, name
    snapshot = source / "source-snapshot.zip"
    with ZipFile(snapshot, "x", compression=ZIP_DEFLATED) as archive:
        for name in state["source_sha256"]:
            archive.write(ROOT/name, name)
        archive.write(source / "protocol.json", "run-protocol.json")
    import_module(module).run(replay)
    assert state == json.loads((replay / "protocol.json").read_text(encoding="utf8"))
    arrays, records = 0, []
    for path in sorted(source.glob("*.npz")):
        with np.load(path) as first, np.load(replay/path.name) as second:
            assert set(first.files) == set(second.files)
            for field in first.files:
                np.testing.assert_array_equal(first[field], second[field], err_msg=path.name+" "+field)
                arrays += 1
        records.append(dict(filename=path.name, fields=len(first.files), every_array_exact=True))
    first = pd.read_csv(source/"cases.csv", float_precision="round_trip")
    second = pd.read_csv(replay/"cases.csv", float_precision="round_trip")
    exact = [name for name in first.columns if "_ms" not in name]
    pd.testing.assert_frame_equal(first[exact], second[exact], check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(source/"diagnostics.csv", float_precision="round_trip"),
                                  pd.read_csv(replay/"diagnostics.csv", float_precision="round_trip"),
                                  check_exact=True)
    pd.DataFrame(records).to_csv(source/"fresh-process-replay.csv", index=False)
    write_json(source/"fresh-process-replay.json", dict(
        runner_module=module, replay=str(replay), source_snapshot_sha256=file_hash(snapshot),
        audit_source_sha256=file_hash(Path(__file__)), arrays_exact=arrays,
        prediction_support_score_teacher_calibration_all_exact=True,
        diagnostic_statistics_exact=True, classification_rows_exact=len(first),
        timing="Fresh call timings are not compared and are not formal speed evidence",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--runner", default="experiments.probe_conditional_block_document")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.source, args.replay, args.runner)
