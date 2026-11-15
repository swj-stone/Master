"""Build the submission source and optionally the local replication supplement."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
PACKAGES = ROOT / "packages"
ARTIFACTS = PAPER / "evidence/revision/artifacts.json"
REASSESSMENT = PACKAGES/"thesis-refit-reassessment-evidence.zip"
OPERATING = PACKAGES / "thesis-operating-evidence.zip"
TRANSPORT = PACKAGES / "thesis-transport-evidence.zip"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def paper_source():
    inputs = {PAPER / "main.tex"}
    pending = list(inputs)
    while pending:
        tex = pending.pop().read_text(encoding="utf8")
        for command, value in re.findall(
                r"\\(input|include|bibliography|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", tex):
            for token in value.split(","):
                path = PAPER / token.strip()
                if command == "bibliography":
                    path = path.with_suffix(".bib")
                elif command in ("include", "input") and not path.suffix:
                    path = path.with_suffix(".tex")
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path not in inputs:
                    inputs.add(path)
                    if path.suffix == ".tex":
                        pending.append(path)
    result = {p.relative_to(PAPER).as_posix():p for p in inputs}
    for tex in inputs:
        if tex.suffix == ".tex":
            for packages in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf8")):
                for name in packages.split(","):
                    local = PAPER / (name.strip() + ".sty")
                    if local.is_file():
                        result[local.name] = local
    result["main.bbl"] = PAPER / "build/main.bbl"
    return result


def write_archive(destination, members):
    tmp = destination.with_suffix(".partial.zip")
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=3) as pack:
        for member, path in sorted(members.items()):
            if not path.is_file():
                raise FileNotFoundError(path)
            pack.write(path, member)
    with ZipFile(tmp) as pack:
        invalid = pack.testzip()
        if invalid or set(pack.namelist()) != set(members):
            raise ValueError(f"Archive verification failed at {invalid}")
    tmp.replace(destination)


def verify_source(archive, pdf):
    scratch = ROOT/"tmp"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="minpi-source-", dir=scratch) as work:
        with ZipFile(archive) as source:
            source.extractall(work)
        subprocess.run(["latexmk","-pdf","-silent","-interaction=nonstopmode","-halt-on-error",
                        "main.tex"],cwd=work,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        documents = [pdf, Path(work)/"main.pdf"]
        texts = [subprocess.check_output(["pdftotext","-layout",str(doc),"-"]) for doc in documents]
        if texts[0] != texts[1]:
            raise ValueError("Rebuilt submission and delivered PDF text differ")
    print("Standalone source rebuilt with identical PDF text and page breaks.")


def local_module(name):
    for directory in (ROOT, ROOT/"src"):
        prefix = directory.joinpath(*name.split("."))
        for path in (prefix.with_suffix(".py"), prefix / "__init__.py"):
            if path.is_file():
                return path
    return None


def include_imports(files):
    pending = [p for p in files if p.suffix == ".py"]
    while pending:
        path = pending.pop()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        base = ROOT/"src" if path.is_relative_to(ROOT/"src") else ROOT
        package = path.relative_to(base).parts[:-1]
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else ()
                module = ".".join((*prefix, node.module) if node.module else prefix)
                if module:
                    names.append(module)
                    names.extend(f"{module}.{a.name}" for a in node.names if a.name != "*")
        for name in names:
            if name.split(".")[0] not in {"minppi_mofs", "experiments"}:
                continue
            new = local_module(name)
            if new and new not in files:
                files.add(new)
                pending.append(new)


def operating():
    """The earlier three archives supply the frozen selectors, folds and donor proposals."""
    files = {ROOT/"thesis-python.ps1", ROOT/"pyproject.toml", ROOT/"README.md", ROOT/"AGENTS.md",
             PAPER/"README.md", PAPER/"operating_results.py", PAPER/"package_revision.py",
             PAPER/"audit/rounds/pmlb_baselines.py", ROOT/"configs/context_study/prospective.json",
             ROOT/"experiments/test/run_piv0_seed42.py"}
    files.update((PAPER/"audit").glob("operating-*"))
    files.update((PAPER/"tables").glob("operating-*.tex"))
    files.add(PACKAGES/"refit-review-before-operating/minpi.pdf")
    for reading in (1, 2, 3):
        files.add(PACKAGES/f"operating-review-drafts/reading-{reading}.pdf")
    folders = ["experiments/context_review", "configs/context_review", "thesis-paper/formal",
               "thesis-paper/evidence/operating"]
    campaigns = ["rule-sensitivity", "cluster-cost", "incremental-prediction", "budgeted-guard",
                 "learned-guard", "guard-clock"]
    folders.extend(f"results/context_review/{name}" for name in campaigns)
    for path in folders:
        directory = ROOT/path
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files.update(p for p in directory.rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix not in {".partial",".tmp"})
    include_imports(files)
    return {p.relative_to(ROOT).as_posix():p for p in files}


def verify_operating(archive):
    """Rebuild the operating tables and decision summaries using only the portable supplement."""
    csv_dir = "thesis-paper/evidence/operating/"
    script = "thesis-paper/operating_results.py"
    outputs = ["thesis-paper/tables/operating-" + stem + ".tex" for stem in ("rates", "rules", "guards")]
    outputs.extend(csv_dir + name for name in ("posterior-costs.csv", "guard-net-benefit.csv"))
    scratch = ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="minpi-operating-", dir=scratch) as folder, ZipFile(archive) as source:
        for name in source.namelist():
            if name == script or (name.startswith(csv_dir) and name not in outputs):
                source.extract(name, folder)
        subprocess.run([sys.executable, script], cwd=folder, check=True)
        for name in outputs:
            if (Path(folder) / name).read_bytes() != source.read(name):
                raise ValueError(f"Portable reproduction differs: {name}")
    print("Three operating tables and two decision-cost summaries reproduced from the evidence ZIP.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="store_true")
    parser.add_argument("--verify-source", action="store_true", help="Compile the ZIP alone and compare every page's text")
    args = parser.parse_args()
    pdf = PAPER/"minpi.pdf"
    archive = PAPER/"minpi-source.zip"
    copyfile(PAPER/"build/main.pdf", pdf)
    write_archive(archive, paper_source())
    if args.verify_source:
        verify_source(archive, pdf)
    if args.evidence:
        PACKAGES.mkdir(exist_ok=True)
        write_archive(OPERATING, operating())
        verify_operating(OPERATING)
    materials = [pdf, archive, PACKAGES/"thesis-pmlb-evidence-r1.zip",
                 PACKAGES/"thesis-context-revision-evidence.zip", REASSESSMENT, OPERATING]
    entries = []
    for path in materials:
        if path.exists():
            item = dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                        sha256=digest(path))
            if path.suffix == ".zip":
                with ZipFile(path) as pack:
                    item["members"] = len(pack.namelist())
            entries.append(item)
    ARTIFACTS.write_text(json.dumps(entries, indent=2)+"\n", encoding="utf8")
    for item in entries:
        print(item["path"], item["sha256"])


if __name__ == "__main__":
    main()
