"""First frozen-rule validation on independently sourced Turkish mail.

Reuse the archived benchmark implementation without altering its source files;
the only runtime substitutions are the panel provider and manifest builder.
"""
import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from experiments import refined_document_benchmark as benchmark
from experiments import reliability_mail_external as external


PROTOCOL = benchmark.ROOT / "docs/minpi-study/reliability-mail-external-protocol.md"
CANDIDATE = ("reliability=minppi_mofs.reliability_document_minpi:"
             "reliability_document_top_from_training")


def panels(stage):
    assert stage == "external"
    yield from external.panels()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "verify", "random"), default="measure")
    parser.add_argument("--pass-id", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.candidate = [CANDIDATE]
    args.stage = "external"
    args.source = [str(Path(__file__).resolve()), str(Path(external.__file__).resolve()),
                   str(external.ASSET), str(PROTOCOL),
                   str(benchmark.ROOT / "docs/minpi-study/reliability-document-protocol.md")]
    archived_snapshot = benchmark.snapshot

    def snapshot(candidates, config):
        state = archived_snapshot(candidates, config)
        state["external_asset"] = external.SHA256
        state["external_source"] = external.SOURCE_URL
        state["external_provider"] = "experiments.reliability_mail_external:panels"
        state["source_record_note"] = "825 labeled records; publisher says 826; non-sample ,,496 tail excluded"
        return state

    benchmark.PROTOCOL = PROTOCOL
    benchmark.panels = panels
    benchmark.snapshot = snapshot
    with threadpool_limits(1):
        benchmark.run(args)


if __name__ == "__main__":
    main()
