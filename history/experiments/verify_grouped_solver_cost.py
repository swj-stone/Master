"""Replay the frozen cost runner with JSON tuple/list normalization only."""
import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from experiments import grouped_solver_cost_audit as frozen


def run(output, pass_id):
    original = frozen.protocol
    expected = json.loads((output/"protocol.json").read_text(encoding="utf8"))
    normalized = json.loads(json.dumps(original(pass_id)))
    assert normalized == expected  # All source/input hashes remain mandatory.
    try:
        frozen.protocol = lambda value: json.loads(json.dumps(original(value)))
        frozen.run(output, pass_id, verify=True)
    finally:
        frozen.protocol = original
    (output/"fresh-process-verification.json").write_text(json.dumps(dict(
        cases=2, method_solver_outputs=32,
        metadata_normalization="JSON changes selector and solver tuples to lists; normalize by JSON roundtrip only",
        verification_driver_sha256=frozen.file_hash(Path(__file__)),
    ), indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass-id", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.pass_id)
