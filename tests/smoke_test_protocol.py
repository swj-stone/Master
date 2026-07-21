"""
Smoke test: verify prepare_split() correctness on real datasets (Madelon, ISOLET).

Checks:
  1. train/val/test indices are disjoint and cover all original rows.
  2. Column names and order are identical across all three splits.
  3. VT.fit / mutual_info / StandardScaler.fit_transform never see val or test.
  4. Same seed produces byte-identical output.
  5. Train is standardised (mean≈0, std≈1).
  6. Different seeds produce different splits.

Run:
    python tests/smoke_test_protocol.py
"""

from __future__ import annotations

from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from minppi_mofs import data_protocol
from minppi_mofs.data_protocol import prepare_split

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_PARAMS = {"variance_threshold": 0.01, "max_features": 100, "train_frac": 0.64,
           "val_frac": 0.16, "test_frac": 0.20}


def _load(ds: str) -> tuple[pd.DataFrame, np.ndarray, str]:
    df = pd.read_csv(RAW / f"{ds}.csv")
    t = "label" if ds == "madelon" else "letter"
    task = "multiclass" if ds == "isolet" else "binary"
    return df.drop(columns=[t]), df[t].values.astype(int), task


# ---------------------------------------------------------------------------
# spies
# ---------------------------------------------------------------------------

vt_rows: list[int] = []
mi_rows: list[tuple[int, int]] = []
sc_rows: list[int] = []

_orig_vt = VarianceThreshold.fit
_orig_mi = data_protocol.mutual_info_classif
_orig_sc = StandardScaler.fit_transform


def _spy_vt(self: VarianceThreshold, Xa: np.ndarray, y=None) -> VarianceThreshold:
    vt_rows.append(Xa.shape[0])
    return _orig_vt(self, Xa, y)


def _spy_mi(Xa: np.ndarray, ya: np.ndarray, **kw) -> np.ndarray:
    mi_rows.append((Xa.shape[0], len(ya)))
    return _orig_mi(Xa, ya, **kw)


def _spy_sc(self: StandardScaler, Xa: np.ndarray, *a, **kw) -> np.ndarray:
    sc_rows.append(Xa.shape[0])
    return _orig_sc(self, Xa, *a, **kw)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def smoke(name: str, X: pd.DataFrame, y: np.ndarray, task: str, seed: int = 42) -> None:
    n = len(X)
    vt_rows.clear()
    mi_rows.clear()
    sc_rows.clear()

    print(f"{'='*60}")
    print(f"{name}  seed={seed}  n={n}  task={task}")
    print(f"{'='*60}")

    with mock.patch.object(VarianceThreshold, "fit", _spy_vt), \
         mock.patch.object(data_protocol, "mutual_info_classif", _spy_mi), \
         mock.patch.object(StandardScaler, "fit_transform", _spy_sc):
        s1 = prepare_split(X, y, task=task, seed=seed, **_PARAMS)
        s2 = prepare_split(X, y, task=task, seed=seed, **_PARAMS)

    tn = len(s1.y_train)
    vn, en = len(s1.y_val), len(s1.y_test)
    print(f"  splits: train={tn}  val={vn}  test={en}  (total={tn+vn+en})")

    # 1. indices disjoint + complete
    ti = set(s1.X_train.index)
    vi = set(s1.X_val.index)
    ei = set(s1.X_test.index)
    assert len(ti & vi) == 0, "train ∩ val not empty"
    assert len(ti & ei) == 0, "train ∩ test not empty"
    assert len(vi & ei) == 0, "val ∩ test not empty"
    assert ti | vi | ei == set(X.index), f"union {len(ti|vi|ei)} != {n}"
    print("  [PASS] indices disjoint & complete")

    # 2. column names and order identical
    cols = list(s1.X_train.columns)
    for tag in ("val", "test"):
        sc = getattr(s1, f"X_{tag}")
        assert list(sc.columns) == cols, f"{tag} columns differ"
    print(f"  [PASS] {len(cols)} columns identical across train/val/test")

    # 3. VT / MI / Scaler saw only train (every call, not just first)
    assert all(r == tn for r in vt_rows), f"VT rows seen: {vt_rows}"
    assert all(r[0] == tn for r in mi_rows), f"MI rows seen: {mi_rows}"
    assert all(r == tn for r in sc_rows), f"Scaler rows seen: {sc_rows}"
    print(f"  [PASS] VT{vt_rows} MI{[r[0] for r in mi_rows]} Scaler{sc_rows} all = {tn} (train)")

    # 4. determinism
    assert np.array_equal(s1.X_train.values, s2.X_train.values)
    assert np.array_equal(s1.X_val.values, s2.X_val.values)
    assert np.array_equal(s1.X_test.values, s2.X_test.values)
    assert np.array_equal(s1.y_train, s2.y_train)
    print("  [PASS] deterministic — same seed → identical byte-level output")

    # 5. train standardised
    tr = s1.X_train.values
    nz = tr.std(axis=0, ddof=0) > 1e-8
    assert np.allclose(tr[:, nz].mean(axis=0), 0.0, atol=1e-6), (
        f"train mean max abs = {np.abs(tr[:,nz].mean(axis=0)).max():.2e}"
    )
    assert np.allclose(tr[:, nz].std(axis=0, ddof=0), 1.0, atol=1e-6), (
        f"train std max diff = {np.abs(tr[:,nz].std(axis=0,ddof=0)-1).max():.2e}"
    )
    print(f"  [PASS] train standardised — {nz.sum()}/{tr.shape[1]} non-const cols have mean≈0 std≈1")

    # 6. different seed → different split
    s3 = prepare_split(X, y, task=task, seed=123, **_PARAMS)
    diff = (
        not np.array_equal(s1.X_train.values, s3.X_train.values)
        or not np.array_equal(s1.X_val.values, s3.X_val.values)
    )
    assert diff, "seeds 42 and 123 produced identical splits"
    print("  [PASS] different seeds → different splits")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for ds in ("madelon", "isolet"):
        X, y, task = _load(ds)
        smoke(ds.title(), X, y, task)

    print("ALL CHECKS PASSED")
