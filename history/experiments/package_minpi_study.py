"""Create and verify a local, self-contained research bundle for the pivotal minPI study."""
import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

if __package__:
    from .review_minpi_study import STUDY_STAGES, study_inputs, verify
else:
    from review_minpi_study import STUDY_STAGES, study_inputs, verify


ROOT = Path(__file__).resolve().parents[1]


def files_for_release():
    files = study_inputs(ROOT)
    files.update(ROOT / name for name in ("AGENTS.md", "docs/minpi-study/README.md",
                                          "docs/thesis/archives/piv1/README.md",
                                          "paper/minpi/README.md", "paper/minpi/formal/README.md"))
    for pattern in ("src/minppi_mofs/**/*.py", "experiments/*minpi_study*.py", "experiments/audit_spambase_duplicates.py",
                    "tests/test_exact_minpi.py", "tests/test_certified_minpi.py", "tests/test_integer_minpi.py",
                    "tests/test_minpi_review.py", "tests/test_word_minpi.py", "tests/test_minpi_proposal.py",
                    "tests/test_layer_word_minpi.py", "tests/test_minpi_tie_keys.py", "tests/test_minpi_supplied_support.py",
                    "tests/test_minpi_dynamic_anchor.py", "tests/test_minpi_reduced*.py",
                    "docs/minpi-study/fixed-tiers-source-snapshot.zip", "docs/minpi-study/rowwide-source-snapshot.zip",
                    "tests/test_piv3_compact_filter.py", "tests/test_piv2.py",
                    "docs/minpi-study/*.md", "docs/thesis/archives/piv1/*.csv",
                    "paper/minpi/**/*.tex", "paper/minpi/**/*.bib", "paper/minpi/**/*.lean",
                    "paper/minpi/*.sty", "paper/minpi/*.bst",
                    "paper/minpi/**/*.wl", "paper/minpi/**/*.py", "paper/minpi/**/*.pdf"):
        files.update(ROOT.glob(pattern))
    results = ROOT/"results/minpi-study"
    for pattern in ("*tests.xml", "*tests.log", "validation-retest.xml", "validation-retest.log"):
        files.update(results.glob(pattern))
    for stage in STUDY_STAGES:
        path = results/stage
        if not path.is_dir():
            raise FileNotFoundError(path)
        files.update(p for p in path.rglob("*") if p.is_file())
    for name in ("main.pdf", "main.tex", "formal/Pivotal.lean"):
        if ROOT/"paper/minpi"/name not in files:
            raise FileNotFoundError(name)
    return sorted(files)


def create(output):
    if output.exists():
        raise FileExistsError(output)
    verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp)/"minpi.zip"
        digests = {}
        with ZipFile(staged, "x", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for path in files_for_release():
                name = path.relative_to(ROOT).as_posix()
                with path.open("rb") as source:
                    digests[name] = hashlib.file_digest(source, "sha256").hexdigest()
                archive.write(path, arcname=name)
            manifest = "".join(f"{h}  {name}\n" for name, h in sorted(digests.items()))
            archive.writestr("SHA256SUMS", manifest)
            archive.writestr("README.txt",
                "Pivotal minPI research bundle\n\n"
                "Manuscript: paper/minpi/main.pdf. Study guide: docs/minpi-study/README.md.\n"
                "The data and outputs in this local archive reproduce the stated experiments.\n"
                "For a fresh rerun, unpack an independent copy. Keep the recorded results for comparison.\n"
                "In a separate environment use the pyproject.toml dependencies, add src to PYTHONPATH,\n"
                "and invoke the study modules from this directory. On the thesis workstation use\n"
                "thesis-python.ps1 as described in the guide. SHA256SUMS lists the archived sources,\n"
                "data, predictions and manuscript. No submission or public upload is part of this archive.\n")
        with ZipFile(staged, "r") as archive:
            for name, expected in digests.items():
                with archive.open(name) as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                        raise ValueError(f"Archive verification failed: {name}")
        os.rename(staged, output)
    with output.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    output.with_suffix(output.suffix+".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"Verified {len(digests)} files, {output.stat().st_size} archived bytes, SHA-256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    create(parser.parse_args().output)
