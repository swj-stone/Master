"""Replay frozen refined-vote outputs with normalized optional CSV metadata fields."""
import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import refined_document_benchmark as benchmark


def run(args):
    args.mode = "verify"
    candidates = benchmark.load_candidates(args.candidate)
    state = benchmark.snapshot(candidates, args)
    assert state == json.loads((args.output/"protocol.json").read_text(encoding="utf8"))
    metadata, audits = [], []
    for X, y, Xt, truth, info in benchmark.panels(args.stage):
        _, audit = benchmark.run_case(X, y, Xt, truth, info, candidates, args)
        metadata.append(info)
        audits.extend(audit)
        print("Verified", info["dataset"], info["fold"], info["seed"], flush=True)
    actual = pd.DataFrame(metadata)
    expected = pd.read_csv(args.output/"panels.csv")
    fields = actual.columns.drop("common_prepare_ms")
    # The UCI and mail providers expose different optional metadata fields.
    # CSV empty cells represent absent values, including the empty video name.
    pd.testing.assert_frame_equal(actual[fields].fillna(""), expected[fields].fillna(""), check_dtype=False)
    assert len(audits) == len(expected)*len(benchmark.methods(candidates))
    assert benchmark.snapshot(candidates, args) == state
    pd.DataFrame(audits).to_csv(args.output/"fresh-process-audit.csv", index=False)
    benchmark.original.write_json(
        args.output/"fresh-process-verification.json",
        dict(configs=len(expected), method_outputs=len(audits),
             metadata_normalization="CSV missing optional fields and empty video names compare as empty strings",
             driver_sha256=benchmark.original.file_hash(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--stage", choices=("dev", "validation", "all", "external"), required=True)
    parser.add_argument("--pass-id", type=int, choices=(1, 2), required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args)
