"""Named selector inputs and outputs shared by the sampled-literature benchmark."""
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class Selection:
    """A ranking in original input order or an intrinsically selected feature subset."""
    scores: np.ndarray | None = None
    selected: list[str] | None = None
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SelectionContext:
    """Training and ranking data only; downstream guard/test partitions never enter a selector."""
    method: str
    mode: str
    columns: list[str]
    train: pd.DataFrame
    yt: np.ndarray
    ranking: pd.DataFrame
    yr: np.ndarray
    task: str
    utility: str
    seed: int
    full_model: Any
    active: tuple[int, ...] | None
    value: Callable
    values: Callable
    losses: Callable
    progress: Callable
