"""Exact native MinShap support for a fixed permutation sample, with negative certificates."""
from bisect import insort
from itertools import islice

import numpy as np

from ._permutation import ALPHA, PATHS, _schedule
from ._selection_context import Selection


def select_support(context):
    """Skip only endpoint evaluations no longer needed by an undecided feature.

    A negative sampled loss reduction implies the final minimum is negative and
    the associated standard-error threshold is nonnegative. The original test
    can never accept that feature. A positive marginal below its own threshold
    alone does not imply rejection: a later minimum can have a smaller variance.
    """
    if context.mode != "native" or context.method != "minshap":
        raise ValueError("Negative certificates require the native MinShap test")
    columns = context.columns
    width = len(columns)
    if not width or len(set(columns)) != width:
        raise ValueError("Expected unique input columns")
    paths, repeated, digest = _schedule(width, list(range(width)), PATHS, context.seed)
    minima = np.full(width, np.inf)
    variance = np.zeros(width)
    counts = np.zeros(width, dtype=int)
    rejected_at = np.full(width, -1)
    shared = {}
    evaluations = 0
    nrows = len(context.yr)
    if nrows < 2:
        raise ValueError("At least two paired ranking rows are required")

    def endpoints(path, pending):
        """Only endpoints adjacent to an undecided column; include every predecessor."""
        included = []
        bits = 0
        for position in range(width+1):
            if position:
                j = path[position-1]
                insort(included, j)
                bits |= 1 << j
            ends = position > 0 and pending[path[position-1]]
            begins = position < width and pending[path[position]]
            if ends or begins:
                yield position, bits, tuple(included), ends

    batch = getattr(context,"native_losses_batch",None) if getattr(context,"native_workers",1)>1 else None
    for path_id, path in enumerate(paths):
        # A feature occurs once per path, so all endpoints needed on this path
        # are already determined by certificates from strictly earlier paths.
        stream = endpoints(path,rejected_at<0)
        previous = None
        while chunk := list(islice(stream,8 if batch else 1)):
            fresh = [(key,mask) for _,key,mask,_ in chunk if key not in shared]
            masks = [mask for _,mask in fresh]
            values = batch(masks) if batch and masks else [context.losses(mask) for mask in masks]
            observations = dict(zip((key for key,_ in fresh),values,strict=True))
            evaluations += len(observations)
            for position, key, _, ends in chunk:
                current = np.asarray(shared[key] if key in shared else observations[key],dtype=float)
                if current.shape != (nrows,) or not np.isfinite(current).all():
                    raise ValueError("Expected finite ranking-row losses")
                if key==0 or key in repeated:
                    shared[key] = current
                if ends:
                    j = path[position-1]
                    paired = previous-current
                    delta = float(paired.mean())
                    counts[j] += 1
                    if delta < minima[j]:
                        minima[j] = delta
                        variance[j] = paired.var(ddof=1)/nrows
                    if delta < 0:
                        rejected_at[j] = path_id
                previous = current
            if position and (position % 16 == 0 or position == width):
                context.progress("minshap_support", path=path_id+1, position=position,
                                 path_width=width, evaluated_coalitions=evaluations,
                                 certified_negative=int(np.count_nonzero(rejected_at >= 0)))
    thresholds = np.sqrt(-2*np.log(ALPHA)*variance)
    accepted = (rejected_at < 0) & (minima >= thresholds)
    if not np.all((counts == PATHS) | (rejected_at >= 0)):
        raise RuntimeError("Unresolved feature without all sampled marginals")
    return Selection(selected=[c for c, keep in zip(columns,accepted) if keep], diagnostics=dict(
        support_only=True, paths=PATHS, alpha=ALPHA, permutation_sha256=digest,
        oracle_prefixes=evaluations, sampled_marginal_count=int(counts.sum()),
        per_feature_marginals=counts.tolist(), negative_certificate_path=rejected_at.tolist(),
        observed_minimum=minima.tolist(), threshold_of_observed_minimum=thresholds.tolist(),
        loss_rows=nrows))
