"""Extract the sealed ZIP in isolation and exercise the commands shipped inside it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import zipfile


HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1]


def run(archive, receipt, fresh):
    started = datetime.now(timezone.utc).isoformat()
    with archive.open("rb") as f:
        checksum = hashlib.file_digest(f,"sha256").hexdigest()
    if checksum != archive.with_suffix(".zip.sha256").read_text(encoding="ascii").split()[0]:
        raise ValueError("Detached archive checksum mismatch")
    stages=[]
    scratch = HISTORY/"tmp"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="sr-",dir=scratch) as folder:
        root = Path(folder).resolve()
        with zipfile.ZipFile(archive) as z:
            if any(not (root/n).resolve().is_relative_to(root) for n in z.namelist()):
                raise ValueError("Unsafe archive path")
            z.extractall(root)
        h = root/"history"
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join(map(str,(h/"src",h))),PYTHONDONTWRITEBYTECODE="1",
                   MPLBACKEND="Agg",MPLCONFIGDIR=str(root/"mpl"),NUMBA_CACHE_DIR=str(root/"numba"),
                   PYTHONNOUSERSITE="1",PYTHONUTF8="1")
        def check(name, *cmd):
            print("Starting",name,flush=True)
            before = perf_counter()
            result = subprocess.run([sys.executable,"-B",*cmd],cwd=h,env=env,
                                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            output = result.stdout.decode("utf8",errors="replace")
            if result.returncode:
                raise RuntimeError(f"{name} failed with status {result.returncode}\n{output[-5000:]}")
            print(output[-425:],flush=True)
            stages.append(dict(stage=name,seconds=round(perf_counter()-before,2)))
        check("isolated-imports","-c",
              "import pathlib, research.fast_single.selector as s, minppi_mofs as m; "
              "assert all(pathlib.Path(x.__file__).resolve().is_relative_to(pathlib.Path.cwd()) for x in (s,m))")
        check("manifest","-m","research.single_reproduction.bundle","verify")
        check("algorithm-tests","-m","pytest","research/fast_single/test_selector.py","-q")
        check("original-timing-tables","-m","research.single_reproduction.replay","archived","--out",str(root/"archive-tables"))
        check("paper-and-three-figures","-m","research.single_reproduction.paper_evidence","--render")
        check("frozen-inputs-and-refits","-m","research.single_reproduction.inputs_trace","--out",str(root/"quality"))
        counts=json.loads((root/"quality/inputs-verified.json").read_text(encoding="utf8"))
        if fresh:
            check("fresh-two-process-experiment","-m","research.single_reproduction.replay","fresh","--out",str(root/"new-run"))
    state=dict(archive=archive.name,sha256=checksum,started_utc=started,
               finished_utc=datetime.now(timezone.utc).isoformat(),stages=stages,quality_counts=counts)
    receipt.write_text(json.dumps(state,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--zip",type=Path,default=HISTORY/"packages/single-peer-paper-56f47fa.zip")
    cli.add_argument("--receipt",type=Path,default=HERE/"restore-verification.json")
    cli.add_argument("--fresh",action="store_true",help="Also repeat the paired runtimes.")
    args=cli.parse_args()
    run(args.zip,args.receipt,args.fresh)
