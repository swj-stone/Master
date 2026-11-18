"""Build the low-dimensional minPI kernel with a system C compiler."""
from hashlib import sha256
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

SOURCE = Path(__file__).with_name("native_votes.c")
FLAGS = ("-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off")


def library_path():
    platform_id = f"{sys.platform}-{platform.machine()}".lower()
    suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    fingerprint = sha256(SOURCE.read_bytes() + " ".join(FLAGS).encode()).hexdigest()[:16]
    return (SOURCE.parents[2] / "results/minpi-study/lowdim-native"
            / f"votes-{platform_id}-{fingerprint}{suffix}")


def build():
    output = library_path()
    if not output.exists():
        compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler is None:
            raise RuntimeError("Building native_votes.c requires a C11 compiler")
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_name(output.name + f".{os.getpid()}.tmp")
        shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
        if sys.platform != "win32":
            shared.append("-fPIC")
        start = perf_counter()
        try:
            subprocess.run([compiler, *FLAGS, *shared, str(SOURCE), "-o", str(staged), "-lm"], check=True)
            os.replace(staged, output)
        finally:
            staged.unlink(missing_ok=True)
        print(f"Compiled in {perf_counter()-start:.3f} s with {compiler}")
    print(output)
    print("SHA256", sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
