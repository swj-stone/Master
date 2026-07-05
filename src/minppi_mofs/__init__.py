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

import xgboost as xgb

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

# Suppress XGBoost training progress output globally (XGBoost ≥3.x
# may emit [0]...validation_0-logloss:... even with per-model verbosity=0).
xgb.set_config(verbosity=0)
logging.getLogger("xgboost").setLevel(logging.ERROR)

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
