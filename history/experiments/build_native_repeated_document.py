"""Build the shared-statistics repeated document minPI kernel outside model fitting."""
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

from minppi_mofs.robust_document_minpi import BUILD, SOURCE, BASE_SOURCE, library_path, source_digest


def build(compiler):
    executable = shutil.which(compiler)
    if not executable:
        raise FileNotFoundError(f"{compiler} unavailable; the reference backend remains available")
    flags = ["-std=c11", "-O3", "-fno-fast-math", "-ffp-contract=off", "-shared", "-Wall", "-Wextra"]
    flags.append("-Wl,--no-insert-timestamp" if sys.platform == "win32" else "-fPIC")
    BUILD.mkdir(parents=True, exist_ok=True)
    target = library_path()
    with tempfile.TemporaryDirectory(dir=BUILD) as temporary:
        library = Path(temporary) / target.name
        start = perf_counter()
        subprocess.run([executable, *flags, str(SOURCE), "-o", str(library), "-lm"], check=True)
        duration = perf_counter() - start
        binary = sha256(library.read_bytes()).hexdigest()
        if not target.exists() or sha256(target.read_bytes()).hexdigest() != binary:
            os.replace(library, target)
        metadata = dict(
            library=target.name, combined_source_sha256=source_digest(),
            source_sha256=sha256(SOURCE.read_bytes()).hexdigest(),
            included_source_sha256=sha256(BASE_SOURCE.read_bytes()).hexdigest(),
            library_sha256=binary, flags=flags, compile_seconds=duration,
            compiler=subprocess.check_output([executable, "--version"], text=True).splitlines()[0],
        )
        record = Path(temporary) / "build.json"
        record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(record, BUILD / "build.json")
    print(json.dumps(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="gcc")
    build(parser.parse_args().compiler)
