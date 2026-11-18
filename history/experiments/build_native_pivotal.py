"""Explicitly build the optional minPI batch voting kernel under history/tmp."""
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

from minppi_mofs.native_pivotal_minpi import BUILD, SOURCE, library_path


def build(compiler="gcc"):
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(f"Compiler {compiler!r} was not found; the NumPy backend remains available")
    BUILD.mkdir(parents=True, exist_ok=True)
    final = library_path()
    flags = ["-std=c11", "-O3", "-shared", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra"]
    if sys.platform == "win32":
        flags.append("-Wl,--no-insert-timestamp")
    else:
        flags.append("-fPIC")
    with tempfile.TemporaryDirectory(dir=BUILD) as tmp:
        candidate = Path(tmp) / final.name
        start = perf_counter()
        subprocess.run([executable, *flags, str(SOURCE), "-o", str(candidate), "-lm"], check=True)
        elapsed = perf_counter() - start
        binary = sha256(candidate.read_bytes()).hexdigest()
        if not final.exists() or sha256(final.read_bytes()).hexdigest() != binary:
            os.replace(candidate, final)
        version = subprocess.check_output([executable, "--version"], text=True).splitlines()[0]
        report = {"library": final.name, "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
                  "library_sha256": binary, "compiler": version, "flags": flags,
                  "compile_seconds": elapsed}
        record = Path(tmp) / "build.json"
        record.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(record, BUILD/"build.json")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="gcc")
    build(parser.parse_args().compiler)
