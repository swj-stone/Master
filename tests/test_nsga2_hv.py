"""
Unit tests for the corrected 2-D hypervolume computation and post-hoc
convergence metric in ``nsga2_fs``.

Run::

    python -m pytest tests/test_nsga2_hv.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from minppi_mofs.baselines.nsga2_fs import _compute_hv, _converged_gen


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


def test_hv_two_point_front() -> None:
    """Known two-point front → HV = 0.28 with ref (1.0, 1.0)."""
    front = np.array([[0.2, 0.8], [0.8, 0.2]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.28)


def test_hv_single_point() -> None:
    """Single point (0.25, 0.40) → HV = (1-0.25)*(1-0.40) = 0.45."""
    front = np.array([[0.25, 0.40]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.45)


def test_hv_ignores_duplicate_and_dominated_points() -> None:
    """Duplicates and a dominated point (0.9, 0.9) are filtered out."""
    front = np.array([[0.2, 0.8], [0.2, 0.8], [0.9, 0.9], [0.8, 0.2]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.28)


def test_hv_empty_front() -> None:
    """Empty front → 0."""
    assert _compute_hv(np.empty((0, 2)), np.array([1.0, 1.0])) == 0.0


def test_hv_all_outside_ref() -> None:
    """Points strictly worse than reference → 0 (all filtered out)."""
    front = np.array([[1.5, 0.5], [0.5, 1.5]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.0)


def test_hv_with_dominated_single_point() -> None:
    """A single dominated point is removed; HV from non-dominated only."""
    front = np.array([[0.6, 0.7], [0.3, 0.4]])
    ref = np.array([1.0, 1.0])
    # (0.6, 0.7) is dominated by (0.3, 0.4) → only the latter counts
    # HV = (1-0.3)*(1-0.4) = 0.7 * 0.6 = 0.42
    assert _compute_hv(front, ref) == pytest.approx(0.42)


def test_hv_many_points_correctness() -> None:
    """Front of 5 well-spaced non-dominated points → manual verification."""
    # All are mutually non-dominated (x↓ y↑)
    front = np.array(
        [
            [0.1, 0.9],
            [0.3, 0.6],
            [0.5, 0.4],
            [0.7, 0.3],
            [0.9, 0.1],
        ]
    )
    ref = np.array([1.0, 1.0])
    hv = _compute_hv(front, ref)
    # Manual (right→left):
    #   (1-0.9)*(1-0.1)=0.1*0.9=0.09
    #   (1-0.7)*(0.9-0.3)=0.3*0.6=0.18
    #   (1-0.5)*(0.6-0.4)=0.5*0.2=0.10
    #   (1-0.3)*(0.4-0.6)... wait y increases so no.
    # Left→right sweep:
    #   best_y=1.0
    #   (0.1,0.9): y<1→(1-0.1)*(1-0.9)=0.9*0.1=0.09, best_y=0.9
    #   (0.3,0.6): y<0.9→(1-0.3)*(0.9-0.6)=0.7*0.3=0.21, best_y=0.6
    #   (0.5,0.4): y<0.6→(1-0.5)*(0.6-0.4)=0.5*0.2=0.10, best_y=0.4
    #   (0.7,0.3): y<0.4→(1-0.7)*(0.4-0.3)=0.3*0.1=0.03, best_y=0.3
    #   (0.9,0.1): y<0.3→(1-0.9)*(0.3-0.1)=0.1*0.2=0.02, best_y=0.1
    #   total = 0.09+0.21+0.10+0.03+0.02 = 0.45
    assert hv == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Convergence metric
# ---------------------------------------------------------------------------


def test_converged_gen_empty() -> None:
    """Empty history → 0."""
    assert _converged_gen([]) == 0


def test_converged_gen_strictly_increasing() -> None:
    """Monotonically increasing HV: gen where cum-best hits 95%."""
    h = [0.10, 0.20, 0.30, 0.40, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
    # cum-best: 0.10, 0.20, 0.30, 0.40, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55
    # 0.95*0.55 = 0.5225
    # gen 7 cum-best=0.52 → not yet
    # gen 8 cum-best=0.53 → yes
    assert _converged_gen(h) == 8


def test_converged_gen_plateau() -> None:
    """HV plateaus after gen 4; cumulative best reaches 95% early."""
    h = [0.20, 0.50, 0.60, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    # cum-best: 0.20, 0.50, 0.60, 0.75, 0.75, ...
    # 0.95*0.75 = 0.7125
    # gen 4 cum-best=0.75 → yes
    assert _converged_gen(h) == 4


def test_converged_gen_zero_hv() -> None:
    """All zero HV → returns len(history)."""
    assert _converged_gen([0.0, 0.0, 0.0]) == 3


def test_converged_gen_fluctuating() -> None:
    """HV fluctuates due to estimation noise; cumulative best is monotonic."""
    h = [0.30, 0.25, 0.35, 0.28, 0.33, 0.40, 0.39, 0.42, 0.41, 0.43]
    # cum-best: 0.30, 0.30, 0.35, 0.35, 0.35, 0.40, 0.40, 0.42, 0.42, 0.43
    # 0.95*0.43 = 0.4085
    # gen 8 cum-best=0.42 → yes
    assert _converged_gen(h) == 8
