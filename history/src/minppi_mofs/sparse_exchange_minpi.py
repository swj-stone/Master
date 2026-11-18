"""Single-pass CSR zero-vote route for budget-adaptive minPI exchanges."""
import numpy as np
from scipy import sparse

from minppi_mofs.exchange_vote_minpi import _audit_support, exchange_vote_top_from_training
from minppi_mofs.sketch_minpi import _validate_sampling
from minppi_mofs.utility_minpi import _utility_inputs


def sparse_exchange_top_from_training(X, y, count=None, *, task="classification", random_state=1129,
                                      groups=None, ordered_groups=False):
    """Keep the support of exchange_vote, skipping randomization when CSR already certifies it."""
    if count is not None or not sparse.issparse(X) or X.ndim != 2 or X.shape[1] < 16:
        return exchange_vote_top_from_training(X, y, count, task=task, random_state=random_state,
                                               groups=groups, ordered_groups=ordered_groups)
    X = X.tocsr(copy=False)
    _, y = _utility_inputs(np.broadcast_to(0., X.shape), y, 1, task)
    groups = _validate_sampling(y, random_state, groups, ordered_groups)
    p = X.shape[1]
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()
    # NaNs count as present. An explicitly stored zero is equivalent to an unstored entry.
    stored_zero = X.data == 0
    indices = X.indices[~stored_zero] if stored_zero.any() else X.indices
    seen = np.bincount(indices, minlength=p) != 0
    info = {"strategy": "forward" if ordered_groups else "group" if groups is not None else "row",
            "folds": [], "exchanged": 0}
    if p - np.count_nonzero(seen) >= (p + 3) // 4:
        cols = np.flatnonzero(seen)
        if not len(cols):
            # Preserve the reference seed's choice when every column is interchangeable.
            cols = np.random.default_rng(random_state).permutation(p)[:1]
        return cols, {**info, "branch": "constant", "budget": len(cols),
                      "constant_columns": np.flatnonzero(~seen)}
    rng = np.random.default_rng(random_state)
    cols, audit = _audit_support(X, y, rng.permutation(p), None, task, groups, ordered_groups, rng)
    return cols, {**info, **audit, "branch": "exchange"}
