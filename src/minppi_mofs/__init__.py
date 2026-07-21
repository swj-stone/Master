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

import xgboost as xgb

# ── XGBoost C++ stderr suppression ──────────────────────────────────────
# XGBoost ≥3.0 writes per-iteration evaluation metrics
# (``[98]    validation_0-logloss:0.17605``) to **stderr via the C++ core**,
# using ``fprintf(stderr, ...)`` / ``std::cerr``.  This output writes to
# OS file-descriptor 2 directly — it bypasses *every* Python-level control:
# ``verbosity=0``, ``verbose_eval=False``, ``callbacks=[]``, ``logging``,
# and ``contextlib.redirect_stderr`` are all ignored.
#
# The ONLY reliable way to stop it is to close fd 2 and replace it with
# the null device at the OS level.  We patch ``xgb.train()`` (the single
# choke-point ALL XGBoost training funnels through) to:
#   1. dup fd 2 onto ``os.devnull`` (catches C++ ``fprintf(stderr)``)
#   2. redirect ``sys.stderr`` to ``os.devnull`` (belt + suspenders)
# Both are restored after training completes.  Genuine training errors
# still raise Python exceptions.

_train_orig = xgb.train
_null_fd: int | None = None  # opened lazily, reused across all calls


def _train_silent(*args, **kwargs):
    global _null_fd
    if _null_fd is None:
        _null_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(_null_fd, 2)
    try:
        return _train_orig(*args, **kwargs)
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


xgb.train = _train_silent  # type: ignore[assignment]

xgb.set_config(verbosity=0)
os.environ.setdefault("OMP_NUM_THREADS", "1")
logging.getLogger("xgboost").setLevel(logging.ERROR)
# ────────────────────────────────────────────────────────────────────────

from minppi_mofs.voting_game import AccFullCache, UtilityView, compute_v_alpha, make_acc_full_cache  # noqa: E402
from minppi_mofs.min_ppi import (  # noqa: E402
    estimate_minppi,
    estimate_minppi_exact,
    estimate_minppi_hybrid,
    estimate_minppi_random_mc,
    estimate_minppi_mi_guided,
)
from minppi_mofs.shapg_integration import (  # noqa: E402
    compute_mi_guidance_signal,
    get_correlated_neighbors,
)
from minppi_mofs.data_protocol import DataSplit, prepare_split  # noqa: E402
from minppi_mofs.utility import compute_utility  # noqa: E402

__all__ = [
    # voting_game
    "compute_v_alpha",
    "AccFullCache",
    "UtilityView",
    "make_acc_full_cache",
    # min_ppi
    "estimate_minppi",
    "estimate_minppi_exact",
    "estimate_minppi_random_mc",
    "estimate_minppi_mi_guided",
    "estimate_minppi_hybrid",
    # shapg_integration
    "compute_mi_guidance_signal",
    "get_correlated_neighbors",
    # data_protocol
    "DataSplit",
    "prepare_split",
    # utility
    "compute_utility",
]
