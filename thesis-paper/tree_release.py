"""Build the MinPI tree manuscript and its offline replication archives."""
import argparse
import ast
import csv
import hashlib
from importlib.util import resolve_name
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
PACKAGES = ROOT / "packages"
EXPERIMENT = ROOT / "experiments/minpi_rework"
EVIDENCE = PAPER / "evidence/minpi_rework"
COHORTS = (
    ("minpi-quota-tree-forecasts.zip","quota-tree-confirmation","selection.json","forecasts.npz",65),
    ("minpi-scale-five-forecasts.zip","quota-scale-confirmation","complete.json","all-forecasts.npz",25),
)
TABLES = {
    "quota-tree-confirmation": ("scores.csv","rank-control-scores.csv","source-means.csv","source-contrasts.csv"),
    "quota-scale-confirmation": ("scores.csv","source-means.csv","source-contrasts.csv"),
    "stratified-tree": ("contrasts.csv","source-results.csv","subject-scores.csv"),
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def checked(command, cwd, capture=False, **kwargs):
    try:
        return subprocess.run(command,cwd=cwd,check=True,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.STDOUT if capture else None,**kwargs)
    except subprocess.CalledProcessError as exc:
        if capture:
            print(exc.stdout.decode(errors="replace")[-3500:],file=sys.stderr)
        raise


def archive(path, members, manifest_name="SHA256SUMS.json"):
    """Sorted member names, fixed ZIP metadata and a manifest of the exact contents."""
    temporary = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True,exist_ok=True)
    hashes = {}
    try:
        with ZipFile(temporary,"w",compression=ZIP_DEFLATED,compresslevel=4) as z:
            for name, source in sorted(members.items()):
                info = ZipInfo(name,date_time=(1980,1,1,0,0,0))
                info.compress_type = ZIP_DEFLATED
                h = hashlib.sha256()
                with z.open(info,"w") as dest, source.open("rb") as src:
                    while chunk := src.read(1024**2):
                        h.update(chunk)
                        dest.write(chunk)
                hashes[name] = h.hexdigest()
            z.writestr(ZipInfo(manifest_name),json.dumps(hashes,sort_keys=True,indent=2)+"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{path.name}: {len(hashes)} files, {path.stat().st_size/1024**2:.1f} MiB, SHA256 {digest(path)}",flush=True)


def tex_files():
    """Follow braced inputs, the paper's plain table-input alias and local styles."""
    pending = [PAPER/"main.tex"]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix!=".tex":
            continue
        body = re.sub(r"(?<!\\)%[^\n]*","",path.read_text(encoding="utf8"))
        inputs = re.findall(r"\\(?:input|include)\s*\{([^}]+)\}",body)
        inputs += re.findall(r"\\tableinput\s+([\w./-]+\.tex)",body)
        for value in inputs:
            child = path.parent/value
            pending.append(child if child.suffix else child.with_suffix(".tex"))
        for value in re.findall(r"\\bibliography\{([^}]+)\}",body):
            pending.extend(PAPER/(name.strip()+".bib") for name in value.split(","))
        for value in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}",body):
            for name in value.split(","):
                style = PAPER/(name.strip()+".sty")
                if style.exists():
                    seen.add(style)
        for value in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",body):
            image = PAPER/value
            if not image.suffix:
                image = next(p for p in (image.with_suffix(s) for s in (".pdf",".png",".jpg")) if p.is_file())
            seen.add(image)
    return seen


def compile_paper(folder):
    env = {**os.environ,"SOURCE_DATE_EPOCH":"1794873600","FORCE_SOURCE_DATE":"1"}
    tex = ["pdflatex","-interaction=nonstopmode","-halt-on-error","-no-shell-escape","main.tex"]
    for command in (tex,["bibtex","main"],tex,tex):
        checked(command,folder,capture=True,env=env)
    log = (folder/"main.log").read_text(errors="replace")
    if re.search(r"undefined (references|citations)|Citation .* undefined|Reference .* undefined",log,re.I):
        raise RuntimeError("Unresolved manuscript references or citations")


def raw_sources():
    base = ROOT/"data/minpi_rework"
    read = lambda name: json.loads((EXPERIMENT/name).read_bytes())
    pins = {v[0]:v[2] for v in read("validation_sources.json").values()}
    pins.update({f"wide-{k}.zip":v[1] for k,v in read("validation_wide_sources.json").items()})
    pins.update({v["filename"]:v["sha256"] for v in read("validation_exchange_sources.json").values()})
    if len(pins)!=18:
        raise ValueError("Expected 18 independent provider inputs")
    for name, pin in pins.items():
        path = base/name
        if digest(path)!=pin:
            raise ValueError(f"Provider digest mismatch: {name}")
        yield path


def local_modules(paths):
    """Include local transitive imports without copying unrelated active studies."""
    more = list(paths)
    seen = set()
    while more:
        path = more.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_bytes(),filename=str(path))):
            if isinstance(node,ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    base = ROOT/"src" if path.is_relative_to(ROOT/"src") else ROOT
                    package = ".".join(path.relative_to(base).parts[:-1])
                    mod = resolve_name("."*node.level+mod,package)
                names = [mod,*(mod+"."+a.name for a in node.names)]
            else:
                continue
            for name in names:
                if name.startswith("minppi_mofs"):
                    base = ROOT/"src"
                elif name.startswith("experiments."):
                    base = ROOT
                else:
                    continue
                stem = base/Path(*name.split("."))
                file = stem/"__init__.py" if stem.is_dir() else stem.with_suffix(".py")
                if file.is_file():
                    more.append(file)
    return seen


def release_files():
    files = set(raw_sources())
    for name, group, _, _, _ in COHORTS:
        path = PACKAGES/name
        expected = (EVIDENCE/group/"forecast-sha256.txt").read_text().split()[0]
        if digest(path)!=expected:
            raise ValueError(f"Sealed forecast archive mismatch: {name}")
        files.add(path)
    roots = []
    for pattern in ("quota_compress_tree*.py","quota_compress_scale*.py","tree_*py","stratified_tree*.py",
                    "span_*.py","scalar_tree*.py","compress_model*.py","validation*.py"):
        roots.extend(EXPERIMENT.glob(pattern))
    roots.append(EXPERIMENT/"quota_compress_rescue_check.py")
    roots.append(ROOT/"src/minppi_mofs/__init__.py")
    modules = local_modules(roots)
    files.update(modules)
    files.update(p.with_suffix(".md") for p in modules if p.with_suffix(".md").exists())
    files.update(EXPERIMENT.glob("validation*.json"))
    files.update((EXPERIMENT/"README.md",EXPERIMENT/"span_findings.md",EXPERIMENT/"stratified_tree_protocol.md",
                  EXPERIMENT/"validation_protocol.md",ROOT/"experiments/minpi_compression/protocol.json",
                  ROOT/"experiments/minpi_compression/legacy_gain_protocol.md",
                  ROOT/"docs/thesis/archives/piv1/protocol.json",ROOT/"pyproject.toml",Path(__file__).resolve(),
                  PAPER/"tree-replication.md",PAPER/"tree-requirements.txt"))
    files.add(PAPER/"evidence/minpi_compression/legacy-gain/novel-input-rows.npz")
    for folder in ("quota-tree","quota-tree-confirmation","quota-scale-confirmation","stratified-tree","exchange-validation"):
        files.update(p for p in (EVIDENCE/folder).rglob("*") if p.is_file())
    for folder in ("tree-ranking-controls","quota-tree"):
        files.update(p for p in (ROOT/"results/minpi_rework"/folder).rglob("*")
                     if p.suffix in {".npz",".json"})
    for pattern in ("Tree*.lean","Tree*.wl","tree-*-receipt.json","lean-toolchain"):
        files.update((PAPER/"formal").glob(pattern))
    return files


def prepared_files():
    files = set()
    tasks = set()
    for manifest in (ROOT/"results/icomp2026_formal").glob("*/data_manifest.csv"):
        files.add(manifest)
        with manifest.open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                if row["dataset_id"] not in {"abalone","credit_card_fraud","gisette","isolet","madelon","spambase"}:
                    continue
                tasks.add((row["dataset_id"],row["seed"]))
                files.update(manifest.parent/row[key] for key in ("train_relpath","val_relpath","test_relpath"))
    if len(tasks)!=30:
        raise ValueError(f"Missing prepared splits, found {len(tasks)}/30")
    # The historical tree/guard experiments use these previously frozen receivers.
    files.update(p for p in (ROOT/"results/minpi_compression/legacy-gain/xgb-receiver").rglob("*")
                 if p.suffix in {".json",".npz"})
    return files


def build(with_prepared):
    (ROOT/"tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT/"tmp",prefix="tree-manuscript-") as tmp:
        folder = Path(tmp)
        manuscript = {p.relative_to(PAPER).as_posix():p for p in tex_files()}
        for name, p in manuscript.items():
            target = folder/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
        compile_paper(folder)
        pdf = PAPER/"minpi-tree.pdf"
        shutil.copyfile(folder/"main.pdf",pdf)
        # The live manuscript can be edited while the clean copy is compiling.
        staged = {name:folder/name for name in manuscript}
        archive(PACKAGES/"minpi-tree-source.zip",{**staged,"main.pdf":folder/"main.pdf"})
        members = {p.relative_to(ROOT).as_posix():p for p in release_files()}
        members.update({"thesis-paper/"+name:source for name,source in staged.items()})
        guide = PAPER/"tree-replication.md"
        entry = folder/"README.md"
        entry.write_text(guide.read_text(encoding="utf8").replace("](../experiments/","](experiments/"),encoding="utf8")
        members.update({"README.md":entry,"thesis-paper/minpi-tree.pdf":folder/"main.pdf"})
        archive(PACKAGES/"minpi-tree-replication.zip",members)
    if with_prepared:
        archive(PACKAGES/"minpi-tree-prepared-six.zip",{p.relative_to(ROOT).as_posix():p for p in prepared_files()},
                "PREPARED-SHA256SUMS.json")


def verify_manifest(name="SHA256SUMS.json"):
    hashes = json.loads((ROOT/name).read_bytes())
    for name, expected in hashes.items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f"Release digest mismatch: {name}")
    print(f"Verified {len(hashes)} released file hashes",flush=True)


def restore():
    for filename, group, receipt_name, forecast_name, size in COHORTS:
        sealed = PACKAGES/filename
        lock_file = EVIDENCE/group/"lock.json"
        locks = json.loads(lock_file.read_bytes())["receipts"]
        if len(locks)!=size or digest(sealed)!=(EVIDENCE/group/"forecast-sha256.txt").read_text().split()[0]:
            raise ValueError(f"Forecast archive mismatch: {group}")
        with ZipFile(sealed) as z:
            if z.read("lock.json")!=lock_file.read_bytes():
                raise ValueError(f"Sealed receipt lock differs: {group}")
            for key,expected in locks.items():
                d = ROOT/"results/minpi_rework"/group/key
                d.mkdir(parents=True,exist_ok=True)
                rec = z.read(f"{key}/{receipt_name}")
                preds = z.read(f"{key}/{forecast_name}")
                if hashlib.sha256(rec).hexdigest()!=expected or hashlib.sha256(preds).hexdigest()!=json.loads(rec)["forecast"]:
                    raise ValueError(f"Corrupt receipt/predictions for {key}")
                for name,raw in ((receipt_name,rec),(forecast_name,preds)):
                    dest = d/name
                    if dest.exists() and dest.read_bytes()!=raw:
                        raise ValueError(f"Existing run differs from archive: {dest}")
                    if not dest.exists():
                        dest.write_bytes(raw)
        print(f"Verified and restored {size} locked {group} runs",flush=True)


def verify(check_tex, prepared):
    import pandas as pd
    from pandas.testing import assert_frame_equal

    verify_manifest()
    if prepared:
        verify_manifest("PREPARED-SHA256SUMS.json")
    list(raw_sources())
    restore()
    for family in ("", "controls", "guard"):
        suffix = (family+"-" if family else "") + "42-123-456-789-1024"
        lock = json.loads((EVIDENCE/"quota-tree"/f"lock-{suffix}.json").read_bytes())
        if len(lock)!=30:
            raise ValueError("Incomplete historical six-source study")
        for key, expected_hash in lock.items():
            folder = ROOT/"results/minpi_rework/quota-tree"/family/key
            receipt = folder/"selection.json"
            if digest(receipt)!=expected_hash or digest(folder/"predictions.npz")!=json.loads(receipt.read_bytes())["forecast_hash"]:
                raise ValueError(f"Historical original-six receipt mismatch: {family}/{key}")
    print("Verified 90 historical original-six arm-family receipt/prediction pairs",flush=True)
    expected = {EVIDENCE/group/name:pd.read_csv(EVIDENCE/group/name) for group,names in TABLES.items() for name in names}
    table = PAPER/"minpi-tree-rows.tex"
    table_before = table.read_text(encoding="utf8")
    env = {**os.environ,"PYTHONPATH":os.pathsep.join(map(str,(ROOT/"src",ROOT))),
           "PYTHONDONTWRITEBYTECODE":"1","NUMBA_CACHE_DIR":str(ROOT/"tmp/numba")}
    checked([sys.executable,"-B","-c",
             "import minppi_mofs as m; from pathlib import Path; "
             "assert Path(m.__file__).resolve().is_relative_to(Path.cwd()/'src')"],ROOT,env=env)
    jobs = [("quota_compress_tree_confirmation","score"),("tree_rank_controls","score"),
            ("quota_compress_scale_confirmation","score"),("quota_compress_tree_report",),
            ("quota_compress_scale_confirmation_report",),("tree_paper_results",)]
    snapshots = {p:p.read_bytes() for p in [*expected,table]}
    if prepared:
        seeds = ("--seeds","42","123","456","789","1024")
        jobs.extend([("quota_compress_tree_run","score",*seeds),
                     ("quota_compress_tree_run","score","--controls",*seeds),
                     ("quota_compress_tree_guard","score",*seeds)])
        for short in ("","controls-","guard-"):
            key = short+"42-123-456-789-1024"
            for pattern in (f"lock-{key}.json",f"receipts-{key}.zip",f"historical-{key}.csv"):
                p = EVIDENCE/"quota-tree"/pattern
                snapshots[p] = p.read_bytes()
            expected[p] = pd.read_csv(p)
    try:
        for name,*arguments in jobs:
            print("Replaying",name,*arguments,flush=True)
            checked([sys.executable,"-B","-m","experiments.minpi_rework."+name,*arguments],ROOT,env=env,capture=True)
        for path, frame in expected.items():
            assert_frame_equal(pd.read_csv(path),frame,check_dtype=False,rtol=1e-12,atol=1e-12)
        if table.read_text(encoding="utf8")!=table_before:
            raise ValueError("Regenerated LaTeX source table changed")
    finally:
        # Restore exact archive bytes (including line endings and ZIP metadata).
        for path,raw in snapshots.items():
            if path.read_bytes()!=raw:
                path.write_bytes(raw)
    if check_tex:
        compile_paper(PAPER)
    print(f"Recomputed {len(expected)} archived CSV tables and verified the typeset numerical table.",flush=True)


def verify_archives(prepared):
    """Reproduce the published scores using only files inside fresh extracted copies."""
    (ROOT/"tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT/"tmp",prefix="tree-release-test-") as tmp:
        dest = Path(tmp)/"replication"
        with ZipFile(PACKAGES/"minpi-tree-replication.zip") as z:
            z.extractall(dest)
        if prepared:
            with ZipFile(PACKAGES/"minpi-tree-prepared-six.zip") as z:
                z.extractall(dest)
        env = {**os.environ,"PYTHONPATH":os.pathsep.join(map(str,(dest/"src",dest))),
               "PYTHONDONTWRITEBYTECODE":"1","NUMBA_CACHE_DIR":str(dest/"tmp/numba")}
        args = [sys.executable,"-B","thesis-paper/tree_release.py","verify","--tex"]
        checked(args+(["--prepared"] if prepared else []),dest,env=env)
        manuscript = Path(tmp)/"source"
        with ZipFile(PACKAGES/"minpi-tree-source.zip") as z:
            z.extractall(manuscript)
        for name,h in json.loads((manuscript/"SHA256SUMS.json").read_bytes()).items():
            if digest(manuscript/name)!=h:
                raise ValueError(f"Standalone source archive differs at {name}")
        compile_paper(manuscript)
        included = dest/"thesis-paper/minpi-tree.pdf"
        # Different directories can affect PDF trailer IDs. Compare complete page text.
        contents = [checked(["pdftotext","-layout",str(pdf),"-"],dest,capture=True).stdout
                    for pdf in (included,manuscript/"main.pdf",dest/"thesis-paper/main.pdf")]
        if not contents[0].strip() or len(set(contents))!=1:
            raise ValueError("Independently compiled documents differ in page text")
        print("Standalone and replication manuscripts reproduce the complete included PDF text.",flush=True)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action",required=True)
    make = sub.add_parser("build")
    make.add_argument("--with-prepared",action="store_true")
    replay = sub.add_parser("verify")
    replay.add_argument("--tex",action="store_true")
    replay.add_argument("--prepared",action="store_true")
    audit = sub.add_parser("verify-archives")
    audit.add_argument("--prepared",action="store_true")
    args = cli.parse_args()
    if args.action=="build":
        build(args.with_prepared)
    elif args.action=="verify-archives":
        verify_archives(args.prepared)
    else:
        verify(args.tex,args.prepared)
