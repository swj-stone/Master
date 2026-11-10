"""Finite-schedule sampling helpers for binary threshold games."""

from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np


def sample_random_round(n: int, rng: np.random.Generator) -> Iterator[tuple[int, int, int]]:
    """Propose one size-uniform context per feature in a shuffled round."""
    for raw_feature in rng.permutation(n):
        feature = int(raw_feature)
        size = int(rng.integers(n))
        positions = rng.choice(n - 1, size=size, replace=False)
        # Map positions in the n-1 other features to the original feature order.
        before = sum(1 << (int(j) + int(j >= feature)) for j in positions)
        yield feature, before, before | (1 << feature)


def update_sample_minimum(current: int, before: bool, after: Callable[[], bool]) -> int:
    """Return the new ternary minimum. An omitted endpoint yields no observed marginal."""
    if current == -1 or (current == 0 and not before):
        return current
    return min(current, int(after()) - int(before))


class ThresholdMinimumView:
    """Compute a sampled minimum from this view's threshold decisions."""

    def is_winning(self, coalition: set[str]) -> bool:
        raise NotImplementedError

    def update_minimum(self, current: int, coalition: set[str], feature: str) -> int:
        if current == -1:
            return current
        return update_sample_minimum(
            current, self.is_winning(coalition), lambda: self.is_winning(coalition | {feature}),
        )


class _MarginalView(Protocol):
    def eval_marginal(self, coalition: set[str], feature: str, /) -> int: ...


def advance_minimum(view: _MarginalView, current: int, coalition: set[str], feature: str) -> int:
    """Use the view's traced or lazy update, or a legacy view's full marginal."""
    if current == -1:
        return current
    update = getattr(view, "update_minimum", None)
    if update is not None:
        return update(current, coalition, feature)
    return min(current, view.eval_marginal(coalition, feature))
