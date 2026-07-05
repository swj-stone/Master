"""
MinPPI-MOFS — Min-Marginal Power Indices for Multi-Objective Feature Selection.

A Voting-Game Approach that synthesises:

- **WVG framework** (Weighted Voting Game): binary  v_alpha(S) ∈ {0, 1}
- **Min operator** (inspired by MinShap): min over coalitions instead of mean
- **MinPPI NSGA-II**: multi-objective evolutionary search with MinPPI-guided
  mutation and crossover operators
- **ShapG integration**: correlation-graph importance signal used for
  adversarial search guidance (Algorithm B)

Public API
----------
``from minppi_mofs import compute_v_alpha, estimate_minppi``
"""

from __future__ import annotations

import logging
import os

# ── pre-import XGBoost patches ──────────────────────────────────────────
# XGBoost ≥3.x `fit()` has a `verbose` parameter whose default may print
# per-iteration metrics even when the constructor's `verbosity=0` and
# `xgb.set_config(verbosity=0)` are set.  Monkey-patch so every `fit()`
# call in the whole project defaults to silent.
import xgboost as xgb

_fit_orig = xgb.XGBModel.fit


def _fit_silent(self, *args, **kwargs):
    kwargs.setdefault("verbose", False)
    return _fit_orig(self, *args, **kwargs)


xgb.XGBModel.fit = _fit_silent  # type: ignore[method-assign]

xgb.set_config(verbosity=0)
os.environ.setdefault("OMP_NUM_THREADS", "1")  # thread-pool hygiene
logging.getLogger("xgboost").setLevel(logging.ERROR)
# ─────────────────────────────────────────────────────────────────────────

from minppi_mofs.voting_game import AccFullCache, compute_v_alpha, make_acc_full_cache
from minppi_mofs.min_ppi import (
    estimate_minppi,
    estimate_minppi_exact,
    estimate_minppi_hybrid,
    estimate_minppi_random_mc,
    estimate_minppi_shapg_guided,
)
from minppi_mofs.shapg_integration import (
    compute_shapg_guidance_signal,
    get_correlated_neighbors,
)

__all__ = [
    # voting_game
    "compute_v_alpha",
    "AccFullCache",
    "make_acc_full_cache",
    # min_ppi
    "estimate_minppi",
    "estimate_minppi_exact",
    "estimate_minppi_random_mc",
    "estimate_minppi_shapg_guided",
    "estimate_minppi_hybrid",
    # shapg_integration
    "compute_shapg_guidance_signal",
    "get_correlated_neighbors",
]
