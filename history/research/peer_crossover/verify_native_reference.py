"""Compare the wide single-peer port with the original C code at modest widths.

Compile the original C source after raising only its compile-time array bound.
Run separately from the timed benchmark.
"""
import ctypes
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from research.lowdim import build_native, method
from research.peer_crossover.single import select_single
from research.wide_blocks.panels import cases


def reference_kernel():
    source = build_native.SOURCE.read_bytes()
    marker = b"#define MAX_P 32"
    assert source.count(marker) == 1
    revised = source.replace(marker, b"#define MAX_P 128")
    directory = build_native.SOURCE.parents[2] / "tmp/peer-crossover-reference"
    directory.mkdir(parents=True, exist_ok=True)
    stem = "reference-" + sha256(revised).hexdigest()[:12]
    src = directory / (stem + ".c")
    lib = directory / (stem + (".dll" if sys.platform == "win32" else ".so"))
    src.write_bytes(revised)
    compiler = shutil.which("gcc") or shutil.which("cc")
    if not compiler:
        raise RuntimeError("The reference check requires a C11 compiler.")
    shared = ["-shared"] if sys.platform == "win32" else ["-shared", "-fPIC"]
    subprocess.run([compiler, *build_native.FLAGS, *shared, str(src), "-o", str(lib), "-lm"], check=True)
    function = ctypes.CDLL(str(lib)).lowdim_votes
    function.argtypes = method._kernel().argtypes
    function.restype = ctypes.c_int
    return function


def compare(kernel, name, X, y, task, seed):
    dense = np.ascontiguousarray(X.toarray() if sparse.issparse(X) else X, dtype=np.float64)
    n, p = dense.shape
    if task == "classification":
        labels, ids = np.unique(y, return_inverse=True)
        y = np.asarray(y)
        response = ids.astype(np.float64)
        classes = len(labels)
    else:
        response = np.ascontiguousarray(y, dtype=np.float64)
        classes = 0
    scores = np.zeros((3, p))
    expected = np.zeros(p, dtype=np.int64)
    width = kernel(dense.ctypes.data, *dense.strides, 8, response.ctypes.data, response.strides[0],
                   n, p, classes, seed, *(s.ctypes.data for s in scores), expected.ctypes.data)
    assert 0 < width <= p
    columns, info = select_single(X, y, task=task, seed=seed)
    np.testing.assert_array_equal(columns, expected[:width], err_msg=name)
    np.testing.assert_allclose(info["rank"], scores[0], rtol=1e-8, atol=1e-9, err_msg=name)
    print(f"{name}: p={p}, k={width}, exact selected column indices")


def main():
    kernel = reference_kernel()
    rng = np.random.default_rng(74213)
    for p, n in ((33, 101), (64, 1053), (101, 1211)):
        x = rng.normal(size=(n, p))
        x[:, :2] = 0
        b = x[:, 2] - x[:, 4] + rng.normal(size=n)
        compare(kernel, f"dense-regression-{p}", x, b, "regression", 1129)
        x[np.abs(x) < 1.45] = 0
        y = np.digitize(b, [-1, 1])
        compare(kernel, f"csr-multiclass-{p}", sparse.csr_matrix(x), y, "classification", 4003)
    chosen = {("mechanism", "sim_proxy_regression_p64", 1, 64064),
              ("spectra", "22-Mango_S1_dm", 1, 1129)}
    for case in cases(chosen):
        compare(kernel, case.key[1], case.selection_X(), case.y, case.task, case.seed)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
