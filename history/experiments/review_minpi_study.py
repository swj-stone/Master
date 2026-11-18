"""Prepare a manuscript for independent review or verify the decision for a release."""
import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/minpi"
REPORT = ROOT / "docs/minpi-study/current-review.md"
STUDY_STAGES = ("pilot", "formal", "summary", "external-official", "subject-transfer",
                "chain", "certificate", "kernel", "matched", "mechanism", "device",
                "new-cohorts", "frontier", "integer-final", "order-diagnostics", "transfer-gap", "raw-cost",
                "anchor-design", "precision-design", "witness-design", "word-design",
                "budget-comparison", "gas-replay", "zero-bounds", "arithmetic-price",
                "boundary-cost", "arithmetic-consequence", "layer-word", "table-audit",
                "tie-keys", "one-peer", "supplied-support", "rounding-witnesses", "figure-witnesses", "training-verifier",
                "dynamic-anchor", "neighbour-ties", "reduced-scale", "predictive-controls")

# These are the local TeX dependency commands used by this manuscript.
INPUT = re.compile(r"\\(input|include|bibliography|bibliographystyle|usepackage|includegraphics)"
                   r"(?:\[[^\]]*\])?\{([^{}]+)\}")
SUFFIX = {"input": ".tex", "include": ".tex", "bibliography": ".bib",
          "bibliographystyle": ".bst", "usepackage": ".sty", "includegraphics": ".pdf"}


def study_inputs(root):
    files = {root / s for s in (
        "configs/minpi-study.json", "pyproject.toml", "requirements.txt", "thesis-python.ps1",
        "paper/minpi/formal/lean-toolchain", "experiments/audit_spambase_duplicates.py",
        "data/raw/nips2003/arcene.zip", "data/raw/nips2003/dexter.zip",
        "data/subject-transfer/uci-har-240.zip", "data/chain/cnae.zip", "data/chain/secom.zip",
        "data/revision-cohorts/gas.zip", "data/revision-cohorts/supercon.zip",
        "data/revision-cohorts/mhealth-windows.npz")}
    for name in ("gisette", "isolet", "madelon", "spambase", "crimedata"):
        files.update((root / f"configs/{name}.yaml", root / f"data/raw/{name}.csv"))
    return files


def review_inputs(paper):
    paths = {paper / "README.md"}
    pending = [paper / "main.tex"]
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        paths.add(path)
        if path.suffix != ".tex":
            continue
        source = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8"))
        for command, names in INPUT.findall(source):
            for name in names.split(","):
                item = paper / name.strip()
                if not item.suffix:
                    item = item.with_suffix(SUFFIX[command])
                if item.is_file():
                    pending.append(item)
                elif command not in {"usepackage", "bibliographystyle"}:
                    raise FileNotFoundError(item)
    return sorted(paths)


def fingerprint(base, paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(base).as_posix()):
        digest.update(path.relative_to(base).as_posix().encode("utf-8") + b"\0")
        with path.open("rb") as source:
            digest.update(hashlib.file_digest(source, "sha256").digest())
    return digest.hexdigest()


def evidence_inputs(root):
    files = {p for p in study_inputs(root) if p.is_file()}
    for folder in STUDY_STAGES:
        files.update(p for p in (root / "results/minpi-study" / folder).rglob("*") if p.is_file())
    for pattern in ("AGENTS.md", "docs/minpi-study/README.md", "docs/minpi-study/independent-audit.md",
                    "paper/minpi/formal/README.md", "tests/test_word_minpi.py", "tests/test_minpi_review.py",
                    "tests/test_minpi_proposal.py", "tests/test_layer_word_minpi.py", "tests/test_minpi_tie_keys.py",
                    "tests/test_minpi_supplied_support.py", "tests/test_minpi_dynamic_anchor.py", "tests/test_minpi_reduced*.py",
                    "tests/test_integer_minpi.py", "tests/test_certified_minpi.py", "tests/test_exact_minpi.py",
                    "results/minpi-study/round3-final-tests.xml", "results/minpi-study/round3-final-tests.log",
                    "results/minpi-study/round5-full-tests.xml", "results/minpi-study/round6-full-tests.xml", "results/minpi-study/round7-full-tests.xml", "results/minpi-study/round8-full-tests.xml",
                    "docs/minpi-study/*protocol.md", "docs/minpi-study/fixed-tiers-source-snapshot.zip",
                    "docs/minpi-study/rowwide-source-snapshot.zip",
                    "experiments/*minpi_study*.py",
                    "src/minppi_mofs/**/*.py", "paper/minpi/formal/*.lean", "paper/minpi/formal/*.wl"):
        files.update(root.glob(pattern))
    return files


def snapshot(paper=PAPER):
    root = paper.parent.parent
    return {"inputs_sha256": fingerprint(paper, review_inputs(paper)),
            "evidence_sha256": fingerprint(root, evidence_inputs(root)),
            "pdf_sha256": hashlib.sha256((paper / "main.pdf").read_bytes()).hexdigest()}


def verify(paper=PAPER, report=REPORT):
    current = snapshot(paper)
    if not report.is_file():
        raise ValueError("Independent review is pending.")
    text = report.read_text(encoding="utf-8-sig")
    header = re.match(r"\A---\n(.+?)\n---\n", text, re.DOTALL)
    if not header:
        raise ValueError("The independent report needs its review decision and manuscript hashes.")
    record = yaml.safe_load(header.group(1))
    if not isinstance(record, dict) or record.get("decision") != "ready_for_external_review":
        raise ValueError("Manuscript returned for revision. Read the independent report.")
    if any(record.get(key) != value for key, value in current.items()):
        raise ValueError("The reviewed snapshot differs from the current manuscript, evidence or reviewer comments.")
    return current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Build the candidate and print its review identity.")
    args = parser.parse_args()
    if args.prepare:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                       cwd=PAPER, check=True)
        print(yaml.safe_dump(snapshot(), sort_keys=False), end="")
    else:
        try:
            verify()
        except (OSError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        print("The independent decision matches the current manuscript and reviewer comments.")
