"""Vary one training-witness design choice while keeping the pivotal game fixed."""
import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study import OUTPUT, ROOT, file_hash, load_dataset
from experiments.minpi_study_new_cohorts import load_cohort
from minppi_mofs.exact_minpi import witness_margins
from minppi_mofs.integer_minpi import rational_top_from_votes
from minppi_mofs.piv2 import _folds
from minppi_mofs.piv3_compact_filter import _scaled

SEED = 1129


@dataclass(frozen=True)
class Design:
    cap: int = 256
    supervised: float = .8
    second_rank: int = 4
    bands: int = 5


def variants(task):
    base = Design()
    yield "default", base
    for name, values in (("cap", (128, 512)), ("supervised", (0., 1.)), ("second_rank", (1, 8))):
        for value in values:
            yield name, replace(base, **{name: value})
    if task == "regression":
        for bands in (3, 7):
            yield "bands", replace(base, bands=bands)


def weights(ref, groups, supervised):
    overall, var = ref.mean(axis=0), ref.var(axis=0)
    labels = np.unique(groups)
    signal = np.zeros(ref.shape[1], dtype=np.float32)
    for label in labels:
        mask = groups == label
        signal += mask.mean() * (ref[mask].mean(axis=0) - overall)**2
    ratio = np.divide(signal, var, out=np.zeros_like(signal), where=var > 0)
    corrected = np.maximum(0., ratio - (len(labels)-1) / max(1, len(ref)-1))
    if not corrected.sum():
        return np.full(ref.shape[1], 1/ref.shape[1], dtype=np.float32)
    return (1-supervised)/ref.shape[1] + supervised*corrected/corrected.sum()


def ballots(X, y, task, design):
    ref, query = _folds(y, task, np.random.default_rng(SEED), row_limit=design.cap)
    ref_y, query_y = y[ref], y[query]
    if task == "regression":
        cuts = np.unique(np.quantile(ref_y, np.arange(1, design.bands)/design.bands))
        ref_y, query_y = np.searchsorted(cuts, ref_y), np.searchsorted(cuts, query_y)

    raw = np.asarray(X[ref], dtype=np.float32)
    finite = np.isfinite(raw)
    observed = np.maximum(finite.sum(axis=0), 1)
    mean = np.where(finite, raw, 0.).sum(axis=0)/observed
    residual = np.where(finite, raw-mean, 0.)
    scale = np.sqrt((residual*residual).sum(axis=0)/observed)
    scale = np.where(scale > 1e-7, scale, 1.).astype(np.float32)
    ref_x, query_x = _scaled(raw, mean, scale), _scaled(X[query], mean, scale)
    metric = weights(ref_x, ref_y, design.supervised)
    distance = ((ref_x**2) @ metric)[None, :] - 2.*((query_x*metric) @ ref_x.T)

    hits, misses = [], []
    for i, label in enumerate(query_y):
        for mask, out in ((ref_y == label, hits), (ref_y != label, misses)):
            pool = np.flatnonzero(mask)
            if not pool.size:
                raise ValueError("A training query lacks a hit or miss")
            depth = min(len(pool), design.second_rank)
            near = pool[np.argpartition(distance[i, pool], depth-1)[:depth]]
            near = near[np.lexsort((near, distance[i, near]))]
            out.extend((near[0], near[-1]))
    q = np.repeat(np.arange(len(query_x)), 2)
    margins = np.abs(query_x[q]-ref_x[misses]) - np.abs(query_x[q]-ref_x[hits])
    return np.sign(margins).astype(np.int8)


def training_data(name):
    if name == "gisette":
        X, y, task, *_ = load_dataset(name)
        with np.load(OUTPUT/"matched"/f"formal-{name}-s{SEED}.npz") as saved:
            train = saved["train_indices"]
    else:
        X, y, _, train, _, task = load_cohort(name)
    return X[train], y[train], task


def run(output):
    if output.exists():
        raise FileExistsError(output)
    reference = pd.read_csv(OUTPUT/"order-diagnostics/cases.csv").query("seed == @SEED and panel == 'full'").set_index("dataset")
    rows, supports = [], []
    for name in ("gisette", "gas", "supercon"):
        X, y, task = training_data(name)
        published = np.sign(witness_margins(X, y, task, SEED)).astype(np.int8)
        h = ceil(.1*X.shape[1])
        for factor, design in variants(task):
            votes = ballots(X, y, task, design)
            digest = sha256(votes.tobytes()).hexdigest()
            chosen, stats = rational_top_from_votes(votes, h)
            oracle, _ = rational_top_from_votes(votes, h, prune=False)
            np.testing.assert_array_equal(chosen, oracle)
            if factor == "default":
                np.testing.assert_array_equal(votes, published)
                if digest != reference.loc[name, "votes_sha256"]:
                    raise ValueError(f"Published {name} ballot differs from archived diagnostic")
                baseline = chosen
            intersection = len(np.intersect1d(chosen, baseline))
            union = 2*h-intersection
            inputs = dict(source=name, seed=SEED, factor=factor, **asdict(design))
            rows.append(dict(**inputs, rows=len(votes), p=X.shape[1], h=h, intersection=intersection,
                             union=union, jaccard=intersection/union, ballot_sha256=digest,
                             exact_columns=stats["exact_columns"], equals_exhaustive=True))
            supports.append(dict(**inputs, selected=chosen.tolist()))
    files = [Path(__file__), ROOT/"docs/minpi-study/witness-design-protocol.md",
             OUTPUT/"order-diagnostics/cases.csv", OUTPUT/"matched"/f"formal-gisette-s{SEED}.npz",
             ROOT/"data/raw/gisette.csv", ROOT/"configs/gisette.yaml",
             ROOT/"experiments/minpi_study.py", ROOT/"experiments/minpi_study_new_cohorts.py"]
    files += [ROOT/"data/revision-cohorts"/f"{name}.zip" for name in ("gas", "supercon")]
    files += [ROOT/"src/minppi_mofs"/f"{name}.py" for name in
              ("piv2", "piv3_compact_filter", "exact_minpi", "integer_minpi", "certified_minpi")]
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/"cases.csv", index=False)
    (output/"supports.json").write_text(json.dumps(supports, separators=(",", ":"))+"\n", encoding="utf-8")
    (output/"inputs.sha256").write_text("".join(f"{file_hash(path)}  {path.relative_to(ROOT).as_posix()}\n"
                                                 for path in files), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)

