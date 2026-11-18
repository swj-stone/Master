"""Build the optional binary CSR document-vote library once, outside inference."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

from minppi_mofs.fast_document_vote_minpi import BUILD, SOURCE, library_path


def build(compiler):
    cc = shutil.which(compiler)
    if not cc:
        raise FileNotFoundError(f"{compiler} unavailable; the reference backend remains available")
    flags = ["-std=c11", "-O3", "-fno-fast-math", "-ffp-contract=off", "-shared", "-Wall", "-Wextra"]
    flags.append("-Wl,--no-insert-timestamp" if sys.platform == "win32" else "-fPIC")
    BUILD.mkdir(parents=True, exist_ok=True)
    target = library_path()
    with tempfile.TemporaryDirectory(dir=BUILD) as tmp:
        obj = Path(tmp) / target.name
        start = perf_counter()
        subprocess.run([cc, *flags, str(SOURCE), "-o", str(obj), "-lm"], check=True)
        duration = perf_counter() - start
        binary = sha256(obj.read_bytes()).hexdigest()
        if not target.exists() or sha256(target.read_bytes()).hexdigest() != binary:
            os.replace(obj, target)
        meta = dict(library=target.name, source_sha256=sha256(SOURCE.read_bytes()).hexdigest(),
                    library_sha256=binary, compiler=subprocess.check_output([cc, "--version"], text=True).splitlines()[0],
                    flags=flags, compile_seconds=duration)
        record = Path(tmp) / "build.json"
        record.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(record, BUILD / "build.json")
    print(json.dumps(meta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="gcc")
    build(parser.parse_args().compiler)
