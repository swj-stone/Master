"""Compile the separate C11 wide-block kernel in the existing thesis environment."""
from hashlib import sha256
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from research.lowdim.build_native import FLAGS


SOURCE = Path(__file__).with_name("native.c")


def library_path():
    digest = sha256(SOURCE.read_bytes() + " ".join(FLAGS).encode()).hexdigest()[:16]
    suffix = { "win32": ".dll", "darwin": ".dylib" }.get(sys.platform, ".so")
    name = f"wide-{sys.platform}-{platform.machine()}-{digest}{suffix}".lower()
    return SOURCE.parents[2] / "results/minpi-study/wide-block-native" / name


def build():
    dest = library_path()
    if not dest.exists():
        compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler is None:
            raise RuntimeError("Building the wide-block kernel requires a C11 compiler")
        dest.parent.mkdir(parents=True, exist_ok=True)
        staged = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
        if sys.platform != "win32":
            flags.append("-fPIC")
        try:
            subprocess.run([compiler, *FLAGS, *flags, str(SOURCE), "-o", str(staged), "-lm"],
                           check=True)
            os.replace(staged, dest)
        finally:
            staged.unlink(missing_ok=True)
    print(dest.name, sha256(dest.read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
