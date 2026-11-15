"""Fixed restoration families and common, unseen masks for paired proposal comparisons."""
from itertools import combinations
from math import comb
import hashlib
import json
from pathlib import Path

import numpy as np

from minppi_mofs.context_design import matched_random_contexts, separating_contexts


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/context_diagnostics/design.json"
FROZEN = "fd701fa4c447ae39d826f03dbe74933d3abd6fc8f67523b3e9197e4b7c75d8ab"
OUTPUT = ROOT / "results/thesis_benchmark/context-mechanism-r1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config():
    if digest(CONFIG) != FROZEN:
        raise ValueError("The frozen diagnostic protocol differs from its recorded hash")
    return json.loads(CONFIG.read_bytes())


def sample_masks(tail, size, count, forbidden, rng):
    """Uniform without replacement from the available masks of precisely this size."""
    forbidden = {mask for mask in forbidden if len(mask) == size}
    possible = comb(len(tail), size)
    take = min(count, possible - len(forbidden))
    if take <= 0:
        return []
    if possible < 4096:
        choices = [m for c in combinations(tail, size) if (m := frozenset(c)) not in forbidden]
        return [choices[i] for i in rng.choice(len(choices), take, replace=False)]
    chosen = []
    while len(chosen) < take:
        selected = frozenset(tail[i] for i in rng.choice(len(tail), size, replace=False))
        if selected not in forbidden:
            chosen.append(selected)
            forbidden.add(selected)
    return chosen


def families(tail, k, rng):
    empty, full = frozenset(), frozenset(tail)
    ends = tuple(dict.fromkeys((empty, full)))
    middle = frozenset(tail[:k])
    ranked = tuple(dict.fromkeys((empty, middle, full)))
    code = tuple(dict.fromkeys(separating_contexts(tail)))

    def randomize(family):
        return matched_random_contexts(family,tail,seed=int(rng.integers(2**62)))

    return {"empty": (empty,), "ends": ends, "nested": ranked, "pooled": ranked,
            "random_middle": randomize(ranked), "separating": code, "random_matched": randomize(code)}


def novel_masks(tail, k, seen, rng, per_size):
    if len(tail) < 2:
        return []
    sizes = {max(1, min(len(tail)-1, s)) for s in
             [round(len(tail)*p) for p in (.25, .5, .75)] + [min(k, len(tail))]}
    return [mask for size in sorted(sizes)
            for mask in sample_masks(tail, size, per_size, set(seen), rng)]


def decide(rule, vote_counts):
    """Positive counts out of four on each context, shared by all decision rules."""
    if not len(vote_counts):
        raise ValueError("A decision requires at least one context")
    if rule == "pooled":
        return (sum(vote_counts) > 2 * len(vote_counts), len(vote_counts))
    for position, yes in enumerate(vote_counts, 1):
        if yes < 3:
            return False, position
    return True, len(vote_counts)
