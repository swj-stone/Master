"""Inventory, package and verify the archived Single-peer experiment."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sysconfig
from tempfile import NamedTemporaryFile
import zipfile


HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1]
CHECKOUT = HISTORY.parent
MANIFEST = HERE/"keep.csv"
BUNDLE = HISTORY/"packages/single-peer-paper-56f47fa.zip"
RUNS = ("fast-single-20261121b","fast-single-extension-20261121b","fast-single-transfer-20261121b")
PAPER_FIGURES = ("lowdim-runtime","subset-quality","matched-curves")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def local(relative):
    path = (HISTORY/relative).resolve(strict=True)
    if not path.is_relative_to(CHECKOUT) or not path.is_file():
        raise ValueError(f"Not a thesis file: {relative}")
    return path


def environment_file(relative):
    package = PurePosixPath(relative).relative_to("Lib/site-packages")
    root = Path(sysconfig.get_path("purelib")).resolve()
    path = (root/package).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Invalid package resource: {relative}")
    return path


def inventory():
    selected = {}

    def add(path, role):
        path = local(path)
        selected.setdefault(path,set()).add(role)

    for name,role in (("inputs.csv","runtime"),("inputs-origins.csv","provenance")):
        for row in read_rows(HERE/name):
            path = local(row["path"])
            if digest(path) != row["sha256"]:
                raise ValueError(f"Source changed after tracing: {path}")
            add(path,role)
    for name in ("pyproject.toml","requirements.txt","thesis-python.ps1"):
        add(name,"environment")
    rp = HISTORY/"paper/minpi-lowdim"
    for name in ("main.tex","main.pdf","README.md","plot_runtime.py","plot_subsets.py","plot_matched_curves.py"):
        add(rp/name,"paper")
    for name in PAPER_FIGURES:
        for ext in ("pdf","png"):
            add(rp/"figures"/f"{name}.{ext}","paper")
    fs = HISTORY/"research/fast_single"
    for name in ("README.md","runner.py","summarize.py","extension.py","transfer.py","ablation.py",
                 "selector.py","panel_votes.c","plot_figure3.py","test_selector.py"):
        add(fs/name,"paper-method")
    for name in ("evidence","evidence-extension"):
        for path in (fs/name).iterdir():
            if path.suffix in (".csv",".md"):
                add(path,"paper-evidence")
    for name in ("peer_crossover/plot_curves.py","peer_crossover/plot_widths.py","peer_crossover/summarize.py","lowdim/ablate_peers.py",
                 "lowdim/build_native.py","lowdim/native_votes.c","lowdim/peer-protocol.md",
                 "lowdim/comparison-protocol.md","lowdim/peer_evidence/cases.csv",
                 "lowdim/peer_evidence/abalone-subset-scores.csv","lowdim/peer_evidence/verification.json"):
        add(HISTORY/"research"/name,"paper-evidence")
    for run in RUNS:
        root = HISTORY/"results/minpi-study"/run
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in (".csv",".json"):
                add(path,"recorded-timings")
        protocol = json.loads((root/"protocol.json").read_text(encoding="utf8"))
        for key in ("hashes","extension_hashes"):
            for name,checksum in protocol.get(key,{}).items():
                path = local(name)
                if digest(path)!=checksum:
                    raise ValueError(f"Original protocol disagrees: {run}, {name}")
                add(path,"protocol-hash")
    for name in ("README.md","replay.py","paper_evidence.py","bundle.py","inputs_trace.py","inputs_origins.py",
                 "inputs.csv","inputs-verified.json","inputs-environment.csv","inputs-origins.csv",
                 "inputs-origins-verified.json","environment.json"):
        add(HERE/name,"reproduction")
    with MANIFEST.open("w",encoding="utf8",newline="") as stream:
        writer = csv.writer(stream,lineterminator="\n")
        writer.writerow(("path","bytes","sha256","role"))
        for path,roles in sorted(selected.items()):
            writer.writerow((Path(os.path.relpath(path,HISTORY)).as_posix(),
                             path.stat().st_size,digest(path),"|".join(sorted(roles))))


def entries():
    for row in read_rows(MANIFEST):
        path = local(row["path"])
        yield path,path.relative_to(CHECKOUT).as_posix(),int(row["bytes"]),row["sha256"]
    yield MANIFEST,MANIFEST.relative_to(CHECKOUT).as_posix(),MANIFEST.stat().st_size,digest(MANIFEST)
    for row in read_rows(HERE/"inputs-environment.csv"):
        path = environment_file(row["path"])
        yield path,"environment-data/"+row["path"],int(row["bytes"]),row["sha256"]


def verify_files(items):
    for path,_,size,sha in items:
        if path.stat().st_size!=size or digest(path)!=sha:
            raise ValueError(f"File does not match ledger: {path}")


def verify_zip(path,items):
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist())!=sorted(name for _,name,_,_ in items):
            raise ValueError("ZIP membership differs from the ledger")
        for _,name,size,sha in items:
            info = archive.getinfo(name)
            with archive.open(info) as stream:
                if info.file_size!=size or hashlib.file_digest(stream,"sha256").hexdigest()!=sha:
                    raise ValueError(f"ZIP entry differs: {name}")


def pack(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=path.parent,prefix="single-paper-",suffix=".partial",delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary,"w",allowZip64=True) as archive:
            for source,name,_,_ in items:
                info = zipfile.ZipInfo(name,(1980,1,1,0,0,0))
                info.compress_type = (zipfile.ZIP_STORED if source.suffix.lower() in
                    {".npz",".zip",".gz",".bz2",".png",".pdf",".xz"} else zipfile.ZIP_DEFLATED)
                info.compress_level = 3
                with source.open("rb") as original, archive.open(info,"w",force_zip64=True) as dest:
                    shutil.copyfileobj(original,dest,1024*1024)
        verify_zip(temporary,items)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    sha=digest(path)
    path.with_suffix(path.suffix+".sha256").write_text(f"{sha}  {path.name}\n",encoding="ascii")
    print(f"{path}: {path.stat().st_size:,} bytes; SHA-256 {sha}")


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=("inventory","verify","pack","verify-zip"))
    cli.add_argument("--zip",type=Path,default=BUNDLE)
    args=cli.parse_args()
    if args.action=="inventory":
        inventory()
    items=list(entries())
    if args.action=="pack":
        pack(args.zip,items)
    elif args.action=="verify-zip":
        verify_zip(args.zip,items)
        checksum = args.zip.with_suffix(args.zip.suffix+".sha256").read_text(encoding="ascii").split()[0]
        if digest(args.zip)!=checksum:
            raise ValueError("Archive differs from its detached SHA-256")
    else:
        verify_files(items)
    print(f"Verified {len(items):,} files including the ledger and bundled environment data.")
