"""Convert a stopped forest's feature requirements to indivisible acquisition units."""
import numpy as np
from scipy import sparse


def collapse_path_groups(a, incidence, unit_ids):
    """Return row increments and path requirements indexed by whole input units.

    The integer IDs 0..m-1 identify the unit supplying each original feature.
    An edge is traversable once every distinct unit on its prefix is available.
    Increments from identical sets of units must be added before forming products.
    """
    mapping = np.asarray(unit_ids)
    incidence = incidence.tocsr()
    labels = np.unique(mapping)
    if (mapping.shape != (incidence.shape[1],) or mapping.dtype.kind not in "iu"
            or mapping.size == 0 or not np.array_equal(labels, np.arange(len(labels)))):
        raise ValueError("Provide one contiguous, zero-based acquisition-unit ID per feature")
    if a.shape[1] != incidence.shape[0]:
        raise ValueError("Each path increment needs one requirement set")
    units = len(labels)
    patterns = {}
    recoded = []
    for start, stop in zip(incidence.indptr[:-1], incidence.indptr[1:]):
        required = tuple(np.unique(mapping[incidence.indices[start:stop]]))
        if not required:
            raise ValueError("An increment must require at least one unit")
        recoded.append(patterns.setdefault(required, len(patterns)))
    aggregation = sparse.csr_matrix((np.ones(len(recoded)), (np.arange(len(recoded)), recoded)),
                                    shape=(len(recoded), len(patterns)))
    row_indices = [i for base, i in patterns.items() for _ in base]
    col_indices = [j for base in patterns for j in base]
    required = sparse.csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)),
                                 shape=(len(patterns), units))
    return (a @ aggregation).tocsc(), required
